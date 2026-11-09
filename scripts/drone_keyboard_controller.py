"""Teaching direct-pose controller: physics steps own motion, UI owns input."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import time
from typing import Iterable
from .drone_control_math import (
    MotionConfig, AttitudeConfig, body_command_from_keys, attitude_command_from_keys,
    integrate_pose, smooth_velocity, target_velocity_from_command, step_tilt,
    orientation_wxyz_from_rpy, rpy_from_orientation_wxyz,
)

CONTROL_KEYS = frozenset('WASDRFQEIKJLX')
DEFAULT_EMERGENCY_KEY = 'X'
TASK_SUCCESS_KEY = 'T'


@dataclass
class KeyState:
    pressed: set[str] = field(default_factory=set)

    def __init__(self, pressed: Iterable[str] | None = None):
        self.pressed = {k.upper() for k in (pressed or ())}

    def press(self, key):
        if key.upper() in CONTROL_KEYS:
            self.pressed.add(key.upper())

    def release(self, key):
        self.pressed.discard(key.upper())

    def emergency_stop(self):
        self.pressed.clear()


class DroneKeyboardController:
    def __init__(self, prim_path='/Quadcopter', config=MotionConfig(), minimum_height=1.,
                 emergency_key=DEFAULT_EMERGENCY_KEY, attitude_config=None,
                 task_instruction: str | None = None):
        if not math.isfinite(minimum_height):
            raise ValueError('minimum_height must be finite')
        self.prim_path, self.config = prim_path, config
        self.attitude_config = attitude_config or AttitudeConfig()
        self.minimum_height = minimum_height
        self.task_instruction = task_instruction
        self.emergency_key = emergency_key.upper()
        self.enabled = True
        self.keys = KeyState()
        self.position = (0., 0., minimum_height)
        self.roll = self.pitch = self.yaw = 0.
        self.roll_rate = self.pitch_rate = 0.
        self.velocity = (0., 0., 0., 0.)
        self._home = None
        self._runtime = self._articulation = None
        self._timeline = self._app_window = self._window = None
        self._input = self._keyboard = self._keyboard_sub = None
        self._update_sub = self._timeline_sub = self._focus_sub = None
        self._callback_ids = []
        self._manager = None
        self._running = self._ready = self._emergency_held = False
        self._waiting_since = None
        self._error = None
        self.task_success_callback = None
        self.task_success_marked = False
        self._task_status = None
        self.inspection_callback = None
        self._inspection_held = False
        self.collision_guard = self.collision_callback = self.undo_callback = None
        self.undo_committed_callback = None
        self.collision_held = False

    def start(self):
        import carb.input
        import omni.appwindow
        import omni.kit.app
        import omni.timeline
        from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
        from .drone_physics_runtime import DronePhysicsRuntime
        self.stop()
        self._error = None
        self.task_success_marked = False
        try:
            self._runtime = DronePhysicsRuntime(self.prim_path)
            self._articulation = self._runtime.articulation
            self._timeline = omni.timeline.get_timeline_interface()
            self._app_window = omni.appwindow.get_default_app_window()
            if self._app_window is None:
                raise RuntimeError('No application window available')
            self._create_control_window()
            self._keyboard = self._app_window.get_keyboard()
            self._input = carb.input.acquire_input_interface()
            self._keyboard_sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
            self._manager = SimulationManager
            for callback, event in ((self._on_physics_pre, SimulationEvent.PHYSICS_PRE_STEP),
                                    (self._on_physics_post, SimulationEvent.PHYSICS_POST_STEP)):
                self._callback_ids.append(SimulationManager.register_callback(callback, event=event))
            self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(self._on_timeline)
            self._focus_sub = self._app_window.get_window_focus_event_stream().create_subscription_to_pop(self._on_focus)
            self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update)
            self._running = True
            print(f'[drone] waiting for physics: {self.prim_path}; Play, then click Drone Control')
            return self
        except Exception:
            self.stop()
            raise

    def _create_control_window(self):
        import omni.ui as ui
        self._window = ui.Window('Drone Control', width=470, height=265 if self.task_instruction else 200)
        self._window.set_focused_changed_fn(lambda focused: None if focused else self._suspend())
        with self._window.frame:
            with ui.VStack(spacing=6):
                ui.Label('Click this window to control the drone.\nLeaving this window clears all input.')
                ui.Label('W/S A/D: move | R/F: height | Q/E: yaw\nI/K: nose down/up | J/L: bank left/right\nHold X: freeze | Release X: level slowly')
                if self.task_instruction:
                    instruction = self.task_instruction
                    if not instruction.isascii():
                        instruction = 'loaded (full text is in run_metadata.json)'
                    ui.Label(f'Task: {instruction}', word_wrap=True)
                self._task_status = ui.Label(
                    'T: mark task success (requires trajectory recording)'
                    if self.task_instruction else 'T: no task loaded')
                with ui.HStack():
                    ui.Button('Enable', clicked_fn=lambda: self.set_enabled(True))
                    ui.Button('Hold', clicked_fn=lambda: self.set_enabled(False))
                    ui.Button('Home', clicked_fn=self.reset_motion)
                self._collision_status = ui.Label('',word_wrap=True,visible=False)
                self._undo_button=ui.Button('Undo collision: back 3 seconds', clicked_fn=self.undo_collision, visible=False)

    @staticmethod
    def _release_subscription(sub):
        if sub is not None:
            release = getattr(sub, 'reset', None) or getattr(sub, 'unsubscribe', None)
            if callable(release):
                release()

    def stop(self):
        self._running = False
        self._suspend(stopped=True)
        errors = []
        if self._input is not None and self._keyboard_sub is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            except Exception as exc:
                errors.append(str(exc))
        self._keyboard_sub = None
        for uid in self._callback_ids:
            try:
                self._manager.deregister_callback(uid)
            except Exception as exc:
                errors.append(str(exc))
        self._callback_ids.clear()
        for name in ('_update_sub', '_timeline_sub', '_focus_sub'):
            sub = getattr(self, name)
            setattr(self, name, None)
            try:
                self._release_subscription(sub)
            except Exception as exc:
                errors.append(str(exc))
        if self._window is not None:
            self._window.destroy()
            self._window = None
        self._input = self._keyboard = self._timeline = self._app_window = None
        self._runtime = self._articulation = None
        self._home = None
        if errors:
            print('[drone] cleanup warnings: ' + '; '.join(errors))

    def _clear_motion(self):
        self.keys.emergency_stop()
        self.velocity = (0., 0., 0., 0.)
        self.roll_rate = self.pitch_rate = 0.

    def _suspend(self, stopped=False):
        self._clear_motion()
        self._emergency_held = False
        if stopped:
            self._ready = False
            self._waiting_since = None

    def emergency_stop(self):
        self._clear_motion()
        self._emergency_held = True
        if self._ready and self._runtime and self._runtime.ready:
            self._write_pose()

    def set_enabled(self, enabled):
        self._suspend()
        self.enabled = enabled

    def reset_motion(self):
        if self.collision_guard is not None:
            return  # Home would create an untracked teleport in a guarded recording.
        self._suspend()
        if self._home is not None and self._ready and self._playing():
            self.position, self.roll, self.pitch, self.yaw = self._home
            self._write_pose()

    def set_task_success_callback(self, callback):
        if callback is not None and not callable(callback):
            raise TypeError('task success callback must be callable or None')
        self.task_success_callback = callback

    def _mark_task_success(self):
        if self.collision_held:
            return False
        if self.task_success_marked:
            return False
        if self.task_success_callback is None:
            if self._task_status is not None:
                self._task_status.text = 'T: success not saved; trajectory recording is off'
            return False
        try:
            marked = self.task_success_callback()
        except Exception as exc:
            if self._task_status is not None:
                self._task_status.text = f'T mark failed: {exc}'
            print(f'[drone] task success mark failed: {exc}')
            return False
        if not marked:
            if self._task_status is not None:
                self._task_status.text = 'Task result already recorded; first result kept'
            return False
        self.task_success_marked = True
        if self._task_status is not None:
            self._task_status.text = 'Task success marked with T'
        return True

    def _playing(self):
        return self._timeline is not None and self._timeline.is_playing()

    def _has_control_focus(self):
        return bool(self._app_window and self._app_window.is_focused() and
                    self._window and self._window.visible and self._window.focused)

    def _on_focus(self, event):
        self._suspend()

    def _on_timeline(self, event):
        import omni.timeline
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            self._suspend(stopped=True)
        elif event.type == int(omni.timeline.TimelineEventType.PAUSE):
            self._suspend()

    def _on_keyboard_event(self, event, *args):
        import carb.input
        key = getattr(event.input, 'name', str(event.input).split('.')[-1]).upper()
        if key == 'H' and self.inspection_callback is not None:
            if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self._inspection_held=False
                return False
            if (event.type == carb.input.KeyboardEventType.KEY_PRESS and not self._inspection_held
                    and self._running and self.enabled and self._ready and self._playing() and self._has_control_focus()):
                self._inspection_held=True
                self.inspection_callback()
            return False
        if key == TASK_SUCCESS_KEY:
            if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                return not self._has_control_focus()
            if event.type != carb.input.KeyboardEventType.KEY_PRESS:
                return not self._has_control_focus()
            if not (self._running and self.enabled and self._ready and self._playing() and self._has_control_focus()):
                self._suspend()
                return True
            self._mark_task_success()
            return False
        if key not in CONTROL_KEYS:
            return True  # Carb: True propagates, False consumes the event.
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.keys.release(key)
            if key == self.emergency_key:
                self._emergency_held = False
            return not self._has_control_focus()
        if not (self._running and self.enabled and not self.collision_held and self._ready and self._playing() and self._has_control_focus()):
            self._suspend()
            return True
        # Repeats must not re-activate a key held across a focus change.
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key == self.emergency_key:
                self.emergency_stop()
            elif not self._emergency_held:
                self.keys.press(key)
        return False

    def _fail(self, message):
        if self._error is None:
            self._error = message
            print(f'[drone] STOPPED: {message}')
        self.stop()

    def _check_runtime(self):
        if not self._running:
            return False
        if self._runtime is None or not self._runtime.valid:
            self._fail('Drone or stage changed. Run the launcher again.')
            return False
        if not self._runtime.ready:
            self._ready = False
            self._clear_motion()
            return False
        return True

    def _initialize_hover(self):
        before = self._runtime.snapshot()
        if getattr(self, 'initial_pose', None) is not None:
            position, orientation = self.initial_pose
            self._runtime.write_pose(position, orientation)
            before = self._runtime.snapshot()
        print(f'[drone] physics before hover: {before}')
        self._runtime.prepare_hover()
        self.position = tuple(float(v) for v in before['position'])
        self.roll, self.pitch, self.yaw = rpy_from_orientation_wxyz(before['orientation'])
        if self.position[2] < self.minimum_height:
            print(f'[drone] Raising root Z from {self.position[2]:.3f} to minimum {self.minimum_height:.3f} m')
            self.position = (*self.position[:2], self.minimum_height)
        self._clear_motion()
        self._home = (self.position, self.roll, self.pitch, self.yaw)
        self._ready = True
        self._waiting_since = None
        self._write_pose()
        if self.collision_guard is not None:
            hit=self.collision_guard(self.position,self.position)
            if hit:raise RuntimeError(f'Initial drone envelope overlaps obstacle: {hit}')
        print(f'[drone] hover initialized: {self._runtime.snapshot()}')

    def _on_physics_pre(self, dt, context):
        try:
            if not self._playing() or not self._check_runtime():
                return
            if not self._ready:
                self._initialize_hover()
            if self.collision_held or (self.collision_guard is not None and self.task_success_marked) or not self.enabled or not self._has_control_focus():
                self._suspend()
            elif not self._emergency_held:
                before=(self.position,self.roll,self.pitch,self.yaw)
                self._advance(dt)
                if self.collision_guard is not None:
                    hit=self.collision_guard(before[0],self.position)
                    if hit:
                        self.position,self.roll,self.pitch,self.yaw=before
                        self._clear_motion()
                        self.collision_held=True
                        if getattr(self,'_collision_status',None) is not None:
                            self._collision_status.text='Obstacle contact: held. Click Undo collision.'
                        if self.collision_callback is not None:self.collision_callback(hit)
            self._write_pose()
        except Exception as exc:
            self._fail(f'Physics pre-step failed: {exc}')

    def undo_collision(self):
        if not (self.collision_held and self.undo_callback and self._ready and self._playing()):
            return False
        self._clear_motion()
        try:
            anchor=self.undo_callback()
            self.position=tuple(anchor['position'])
            self.roll,self.pitch,self.yaw=(anchor.get(k,0.) for k in ('roll','pitch','yaw'))
            self._write_pose()
            if self.undo_committed_callback is not None:self.undo_committed_callback()
            self.collision_held=False
            self._emergency_held=False
            if getattr(self,'_collision_status',None) is not None:
                self._collision_status.text='Restored safe sample. Click Drone Control; press movement keys again.'
            return True
        except Exception as exc:
            self.collision_held=True
            self._fail(f'Undo failed: {exc}')
            return False

    def _on_physics_post(self, dt, context):
        try:
            if self._running and self._ready and self._playing() and self._check_runtime():
                self._write_pose()
        except Exception as exc:
            self._fail(f'Physics post-step failed: {exc}')

    def _advance(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            return
        target = target_velocity_from_command(body_command_from_keys(self.keys.pressed), self.config)
        self.velocity = smooth_velocity(self.velocity, target, dt, self.config)
        self.position, self.yaw = integrate_pose(self.position, self.yaw, self.velocity, dt, self.minimum_height)
        if self.position[2] <= self.minimum_height and self.velocity[2] < 0:
            self.velocity = (*self.velocity[:2], 0., self.velocity[3])
        roll_command, pitch_command = attitude_command_from_keys(self.keys.pressed)
        self.roll, self.roll_rate = step_tilt(self.roll, self.roll_rate, roll_command, dt, self.attitude_config)
        self.pitch, self.pitch_rate = step_tilt(self.pitch, self.pitch_rate, pitch_command, dt, self.attitude_config)

    def _write_pose(self):
        self._runtime.write_pose(self.position, orientation_wxyz_from_rpy(self.roll, self.pitch, self.yaw))

    def _on_update(self, event):
        # UI/health only: never integrate using render-frame delta time.
        if not self._running:
            return
        try:
            if not self._runtime.valid:
                self._fail('Drone or stage changed. Run the launcher again.')
                return
            if not self._playing():
                self._suspend(stopped=self._timeline.is_stopped())
                self._waiting_since = None
                return
            if not self._has_control_focus():
                self._suspend()
            if not self._runtime.ready:
                self._ready = False
                self._clear_motion()
            if not self._ready:
                now = time.monotonic()
                if self._waiting_since is None:
                    self._waiting_since = now
                elif now - self._waiting_since > 10:
                    self._fail('Physics did not become ready within 10 seconds. Check PhysicsScene and Articulation Root.')
        except Exception as exc:
            self._fail(f'Runtime monitor failed: {exc}')

    def diagnostics(self):
        return {'target_position': self.position,
                'rpy_degrees': [math.degrees(v) for v in (self.roll, self.pitch, self.yaw)],
                'running': self._running, 'ready': self._ready, 'error': self._error,
                'physics': self._runtime.snapshot() if self._runtime else None}

    def state_snapshot(self):
        """Return a serializable state sample for trajectory recording."""
        command = body_command_from_keys(self.keys.pressed)
        physics = self._runtime.snapshot() if self._runtime and self._runtime.ready else {}
        return {
            'position': list(self.position),
            'roll': self.roll, 'pitch': self.pitch, 'yaw': self.yaw,
            'orientation': list(orientation_wxyz_from_rpy(self.roll, self.pitch, self.yaw)),
            'linear_velocity': physics.get('linear_velocity', [0., 0., 0.]),
            'angular_velocity': physics.get('angular_velocity', [0., 0., 0.]),
            'command': {'forward': command.forward, 'right': command.right,
                        'vertical': command.vertical, 'yaw': command.yaw,
                        'keys': sorted(self.keys.pressed), 'enabled': self.enabled,
                        'emergency_held': self._emergency_held},
        }


_active_controller = None


def start(**kwargs):
    global _active_controller
    stop()
    _active_controller = DroneKeyboardController(**kwargs)
    return _active_controller.start()


def stop():
    global _active_controller
    if _active_controller is not None:
        _active_controller.stop()
        _active_controller = None
