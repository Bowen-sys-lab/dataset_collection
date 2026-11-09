"""Standalone RGB + software thermal preview for the Crazyflie stage.

Run with ``D:\\isaacsim\\python.bat scripts\\start_crazyflie_dual_camera_view.py``.
The source USD is opened read-only; cameras and UI live only for this process.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
THERMAL_ROOT = PROJECT_ROOT / 'third_party' / 'isaac-thermal-camera-simulation'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(THERMAL_ROOT / 'src'))

from scripts.collection_capture_defaults import TRAJECTORY_PERIOD_S, capture_contract
from scripts.dual_camera_runtime import CameraMount
from scripts.dual_camera_support import CleanupStack, PreviewClock, ui_rgba_array, preview_sensor_data
from scripts.dual_camera_support import scale_camera_optics_for_resolution
from scripts.dual_camera_support import ExplicitRenderIncomplete, capture_interruption_reason
from scripts.thermal_config import DEFAULT_THERMAL_CONFIG, load_thermal_config
from scripts.scene_config import (DEFAULT_SCENE_CONFIG, SceneConfig, load_scene_config,
                                  resolve_task_config)
from scripts.trajectory_recorder import (TrajectoryRecorder, trajectory_state_from_world_transform,
                                         configure_recording_timeline, validate_recording_time,
                                         prepare_frozen_capture_timeline)
from scripts.drone_control_math import MotionConfig
from scripts.thermal_imaging import display_from_outputs, imaging_metadata
from scripts.thermal_display import display_range_metadata
from scripts.preview_scheduler import PreviewScheduler, SplitPreviewScheduler, percentile_summary, verify_renderer_evidence, validate_split_options
from scripts.preview_metrics import ControlMetrics, configure_live_viewports, render_dispatcher_state
from scripts.live_frame_validation import PreviewContentTracker
from scripts.frame_metadata import FrameMetadataWriter

DEFAULT_STAGE = PROJECT_ROOT / 'lesson_crazyflie.usd'
DEFAULT_SENSOR = THERMAL_ROOT / 'configs/sensors/flir_boson_640_lwir.yaml'
DEFAULT_SCENE = THERMAL_ROOT / 'configs/scenes/quad_flight_clear_noon.yaml'
SOURCE_CAMERA = '/World/Crazyflie/body/Camera_mount/Camera'
BODY_PATH = '/World/Crazyflie/body'


@dataclass
class ThermalFrame:
    display8: np.ndarray
    radiance: np.ndarray
    apparent_t: np.ndarray
    dn16: np.ndarray | None = None


class DualCameraPreview:
    def __init__(self, stage, mount, thermal, rgb_annotator, rgb_product, *, width, height,
                 thermal_width=None, thermal_height=None, cleanup, metadata,
                 show_ui=True, display_profile=None):
        self.stage = stage
        self.mount = mount
        self.thermal = thermal
        self.rgb_annotator = rgb_annotator
        self.rgb_product = rgb_product
        self.width, self.height = width, height
        self.thermal_width = width if thermal_width is None else thermal_width
        self.thermal_height = height if thermal_height is None else thermal_height
        self.clock = PreviewClock()
        # Pilot HUD state; populated by set_pilot_target() when a task carries a target.
        self.pilot_target = None
        self.pilot_hud = None
        self.pilot_hold_required_s = 0.0
        self.pilot_distance_range_m = (0.0, 0.0)
        self.pilot_hold_s = 0.0
        self.pilot_geometry = None
        self.pilot_timer = None
        self._pilot_last_sim = None
        self._pilot_last_pos = None
        # Render subframes issued per frozen-state capture. Four drains the asynchronous USD
        # -> Hydra path reliably, but each one is a full PathTracing pass, so offline replay
        # may lower it after verifying the frames still differ.
        self.frozen_subframes = 4
        self.window = None
        self.rgb_provider = None
        self.thermal_provider = None
        self.last_capture_wall = None
        self.frames = 0
        self._fps = 0.0
        self._fps_total = 0.0
        self._fps_count = 0
        self._first_publish_wall = None
        self._last_publish_wall = None
        self._closed = False
        self.cleanup = cleanup
        self.metadata = metadata
        self.display_profile = display_profile
        self.show_ui = show_ui
        self.skip_live_preconversion = False
        # Lightweight wall-clock timing is always kept; function profiling is opt-in.
        self.profile_enabled = False
        self.last_timings = {}
        self._timing_totals = {}
        self.profiler = None
        self._metadata_writers = {}
        self.last_thermal_refresh_wall = None
        self.last_thermal_request_sim = None
        self.on_demand_ir = False
        self.inspection_pending = False
        self.inspection_error = None

    def sample_cruise_clock(self,simulation_time):
        return self.clock.sample(simulation_time)

    def inspection_initialization(self,app):
        import omni.replicator.core as rep
        from irsim_isaac.pipeline import ir_camera
        from scripts.on_demand_ir import frozen_initialization
        return frozen_initialization(rep.orchestrator,ir_camera,app)

    def publish_rgb(self):
        """A live RGB refresh must not run any thermal computation."""
        self._ensure_window()
        rgb = np.asarray(self.rgb_annotator.get_data()).copy()
        if rgb.shape[:2] != (self.height, self.width):
            raise RuntimeError(f'RGB shape {rgb.shape} differs from configured resolution')
        if self.show_ui:
            pixels, size = ui_rgba_array(rgb)
            self.rgb_provider.set_data_array(pixels, size, strict=True)
        return rgb

    def publish_thermal(self, thermal, meta):
        self._ensure_window()
        if self.show_ui:
            pixels, size = ui_rgba_array(thermal.display8)
            self.thermal_provider.set_data_array(pixels, size, strict=True)
        self.last_thermal_refresh_wall = time.perf_counter()
        self.last_thermal_request_sim = meta['simulation_time_s']
        self.frames += 1
        if self.show_ui:
            self.status.text=(f'IR last refresh 0.00s ago | request sim {self.last_thermal_request_sim:.3f}s | '
                              'RGB/IR independent; AOV source age unknown')

    def update_live_hud(self):
        if not self.show_ui:
            return
        self._ensure_window()
        import omni.timeline
        from pxr import UsdGeom
        sim = float(omni.timeline.get_timeline_interface().get_current_time())
        pose = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.mount.rgb_path)).ComputeLocalToWorldTransform(0)
        self._update_pilot_hud(pose, sim)
        if getattr(self,'anchor_guidance_note',None):
            self.pilot_hud.text=self.anchor_guidance_note
        provider = getattr(self, 'mileage_provider', None)
        if provider is not None:
            self.mileage_hud.text = provider()
        if self.on_demand_ir and self.last_thermal_refresh_wall is None:
            self.status.text='RGB-only flight | H: pause + IR snapshot | T: manual confirmation'
        if self.last_thermal_refresh_wall is not None:
            age = time.perf_counter() - self.last_thermal_refresh_wall
            self.status.text = (f'IR last refresh {age:.2f}s ago | request sim {self.last_thermal_request_sim:.3f}s | '
                                'RGB/IR independent; AOV source age unknown' +
                                (' | Paused/stopped' if not omni.timeline.get_timeline_interface().is_playing() else ''))
        if self.on_demand_ir and self.inspection_error:
            self.status.text=f'IR inspection failed: {self.inspection_error} | H to retry; do not use the old snapshot'

    def set_pilot_target(self, target_m, hold_required_s=3.0, distance_range_m=(6.0, 8.0),
                         inspection=None):
        """Attach collection guidance; only replay review can qualify the task."""
        import numpy as np
        from scripts.factory_pilot_hud import EquipmentClearance, ObservationTimer
        self.pilot_target = np.asarray(target_m, dtype=float).reshape(3)
        self.pilot_hold_required_s = float(hold_required_s)
        self.pilot_distance_range_m = (float(distance_range_m[0]), float(distance_range_m[1]))
        inspection = inspection or {}
        self.pilot_geometry = (EquipmentClearance.load(PROJECT_ROOT, inspection)
                               if inspection.get('geometry_file') else None)
        req = inspection.get('inspection_requirements', {})
        self.pilot_timer = ObservationTimer(self.pilot_distance_range_m, self.pilot_hold_required_s,
            req.get('maximum_observation_speed_m_s', 0.3), req.get('maximum_sample_gap_s', 0.25))

    def _update_pilot_hud(self, rgb_pose_matrix, sim_time):
        """Surface clearance and geometric hold guidance; no visibility certification."""
        if not self.show_ui or self.pilot_target is None or self.pilot_hud is None:
            return
        import numpy as np
        from pxr import UsdGeom
        from scripts.factory_pilot_hud import projected_inside
        body_path = self.metadata.get('drone_body_path')
        body_prim = self.stage.GetPrimAtPath(body_path) if body_path else None
        if body_prim and body_prim.IsValid():
            body_pos = np.array(UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(0))[3, :3]
        else:
            body_pos = np.array(rgb_pose_matrix)[3, :3]

        clearance = (self.pilot_geometry.distance(body_pos)
                     if self.pilot_geometry is not None and body_prim and body_prim.IsValid() else None)
        lo, hi = self.pilot_distance_range_m
        camera = UsdGeom.Camera(self.stage.GetPrimAtPath(self.mount.rgb_path))
        optics = (camera.GetFocalLengthAttr().Get(), camera.GetHorizontalApertureAttr().Get(),
                  camera.GetVerticalApertureAttr().Get()) if camera else (0.,0.,0.)
        in_view = projected_inside(self.pilot_target, rgb_pose_matrix, self.width, self.height, *optics)
        state = self.pilot_timer.update(sim_time, body_pos, clearance, in_view)
        self.pilot_hold_s = state['hold_s']
        speed = state['speed_m_s']

        mark = lambda ok: 'OK' if ok else '--'
        speed_text = '  ?  ' if speed is None else f'{speed:4.2f}'
        distance_text = (f'设备表面净距 {clearance:5.2f} m [{mark(state["in_range"])} {lo:g}-{hi:g}]'
                         if clearance is not None else
                         f'参考点距 {np.linalg.norm(body_pos-self.pilot_target):5.2f} m [无设备几何]')
        text = (f'{distance_text}   几何观察计时 {self.pilot_hold_s:4.1f}/{self.pilot_hold_required_s:g} s   '
                f'速度 {speed_text} m/s [{mark(state["slow_enough"])}]   '
                f'热点投影入画 [{mark(in_view)}]（遮挡须目视核实）')
        try:
            self.pilot_hud.text = text
        except Exception:
            pass

        self._pilot_last_sim = float(sim_time)
        self._pilot_last_pos = body_pos

    def _ensure_window(self):
        if not self.show_ui or self.window is not None:
            return
        import omni.ui as ui
        self.rgb_provider = ui.ByteImageProvider()
        self.thermal_provider = ui.ByteImageProvider()
        profile = self.display_profile
        caption = (f'Thermal apparent T ({profile.temperature_min_c:g}..{profile.temperature_max_c:g} C)'
                   if profile is not None and profile.mode == 'fixed_apparent_temperature'
                   else 'Thermal (sensor ISP)')
        # Both panes take half the width and fill the height, and the images keep their aspect
        # ratio: enlarging the window enlarges the pictures instead of pinning them at 400 px.
        self.window = ui.Window('Crazyflie RGB + Thermal', width=900, height=640,
                                position_x=30, position_y=95)
        with self.window.frame:
            with ui.VStack(spacing=4):
                ui.Label(f'RGB {self.width} x {self.height} | Thermal {self.thermal_width} x '
                         f'{self.thermal_height} | click Drone Control to fly', height=22)
                with ui.HStack(spacing=8, height=ui.Percent(72)):
                    with ui.VStack(width=ui.Percent(50), spacing=2):
                        ui.Label('RGB', height=22)
                        ui.ImageWithProvider(
                            self.rgb_provider, width=ui.Percent(100), height=ui.Percent(100),
                            fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                    with ui.VStack(width=ui.Percent(50), spacing=2):
                        ui.Label(caption, height=22)
                        ui.ImageWithProvider(
                            self.thermal_provider, width=ui.Percent(100), height=ui.Percent(100),
                            fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                self.status = ui.Label('Ready. Press Play to start.', height=22)
                self.flight_status = ui.Label('', height=20)
                self.perf_status = ui.Label('', height=20)
                # Pilot HUD: live distance / hold timer / speed / target-in-view. Without it the
                # operator cannot tell whether the 6-8 m, 3 s observation window has been met,
                # which is exactly the criterion the offline evaluator scores.
                self.pilot_hud = ui.Label('', height=22)
                self.mileage_hud = ui.Label('', height=20)

    def capture(self, *, render=True, simulation_time=None, frozen_state=False, publish=True):
        capture_started = time.perf_counter()
        source_state = render_dispatcher_state() if not render else None
        self._ensure_window()
        import omni.timeline
        import omni.replicator.core as rep
        from pxr import UsdGeom
        # Replicator's default step advances or pauses the timeline.  A preview
        # must render the current simulation frame without taking ownership of
        # the controller's clock, so use an explicit zero-duration step.
        timeline = omni.timeline.get_timeline_interface()
        if frozen_state and (not render or timeline.is_playing()):
            raise RuntimeError('Frozen-state capture requires explicit render and a paused/stopped timeline')
        timeline_before = float(timeline.get_current_time())
        sim_before = timeline_before if simulation_time is None else float(simulation_time)
        if frozen_state:
            prepare_frozen_capture_timeline(timeline, sim_before)
            import omni.kit.app
            # Render subframes alone do not process pending USD/Hydra edits.
            # Pump three app ticks at a frozen simulation timestamp BEFORE the
            # one explicit capture step. These ticks may render automatically.
            for _ in range(3):
                omni.kit.app.get_app().update()
            if abs(float(timeline.get_current_time()) - timeline_before) > 1e-8:
                raise RuntimeError('Scene update barrier advanced the frozen simulation timestamp')
        thermal_t, epoch = self.clock.sample(sim_before)
        self.thermal._t_rel_s = thermal_t
        t0 = time.perf_counter()
        self.thermal.refresh_pose()
        rgb_render_finished_wall = None
        render_status_before = render_status_after = None
        if render:
            if rep.orchestrator.get_status() == rep.orchestrator.Status.STOPPING:
                rep.orchestrator.wait_until_complete()
            render_status_before = str(rep.orchestrator.get_status())
            # USD edits reach Hydra asynchronously on this build. Multiple
            # render subframes drain that pipeline while simulation stays frozen;
            # this is one explicit step call, not one GPU render pass.
            rep.orchestrator.step(rt_subframes=self.frozen_subframes if frozen_state else 1,
                                  pause_timeline=bool(frozen_state), delta_time=0.0,
                                  wait_for_render=True)
            rgb_render_finished_wall = time.perf_counter()
            render_status_after = str(rep.orchestrator.get_status())
            if rep.orchestrator.get_status() != rep.orchestrator.Status.STEPPED:
                raise ExplicitRenderIncomplete(f'Explicit render did not complete: {render_status_before} -> {render_status_after}')
        t1 = time.perf_counter()
        if self.profiler is not None:
            self.profiler.enable()
        outputs = display_from_outputs(self.thermal.config,
                                       self.thermal.get_outputs(step=False, rt_subframes=1),
                                       self.display_profile)
        if self.profiler is not None:
            self.profiler.disable()
        t2 = time.perf_counter()
        rgb = np.asarray(self.rgb_annotator.get_data())
        t3 = time.perf_counter()
        display = np.asarray(outputs.display8)
        dn16 = None if getattr(outputs, 'dn16', None) is None else np.asarray(outputs.dn16)
        if rgb.shape[:2] != (self.height, self.width):
            raise RuntimeError(f'RGB shape {rgb.shape} != {(self.height, self.width)}')
        if display.shape[:2] != (self.thermal_height, self.thermal_width):
            raise RuntimeError(
                f'thermal shape {display.shape} != {(self.thermal_height, self.thermal_width)}')
        if self.show_ui and not (self.skip_live_preconversion and not publish):
            rgb_data, rgb_size = ui_rgba_array(rgb)
            thermal_data, thermal_size = ui_rgba_array(display)
        t4 = time.perf_counter()
        if self.show_ui and publish:
            self.rgb_provider.set_data_array(rgb_data, rgb_size, strict=True)
            self.thermal_provider.set_data_array(thermal_data, thermal_size, strict=True)
        t5 = time.perf_counter()
        rgb_pose = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.mount.rgb_path)).ComputeLocalToWorldTransform(0)
        thermal_pose = UsdGeom.Xformable(self.stage.GetPrimAtPath(self.mount.thermal_path)).ComputeLocalToWorldTransform(0)
        pose_error = max(abs(float(rgb_pose[i][j] - thermal_pose[i][j])) for i in range(4) for j in range(4))
        self._update_pilot_hud(rgb_pose, sim_before)
        meta = {
            **self.metadata,
            'frame': self.frames,
            'capture_mode': ('offline_frozen_state_render' if frozen_state else
                             'explicit_render' if render else 'live_preview'),
            'preview_sync_status': None if render else 'live_preview_not_strict',
            'capture_started_wall_s': capture_started,
            'capture_finished_wall_s': time.perf_counter(),
            'rgb_thermal_capture_time': sim_before,
            'rgb_source_simulation_time': None if render else source_state['simulation_time'],
            'thermal_geometry_source_simulation_time': None,
            'individual_aov_timestamps_verified': False,
            'capture_time_basis': 'CPU sample request; individual AOV source time is unverified',
            'frame_id': self.frames,
            'simulation_time_before': sim_before,
            'simulation_time_after': float(timeline.get_current_time()),
            'timeline_time_before': timeline_before,
            'timeline_time_after': float(timeline.get_current_time()),
            'simulation_time_s': sim_before,
            'thermal_elapsed_s': thermal_t,
            'thermal_frame_time_s': float(self.thermal.last_frame_t_s),
            'rgb_render_finished_wall_s': rgb_render_finished_wall,
            'thermal_read_finished_wall_s': time.perf_counter(),
            'explicit_render_triggered': bool(render),
            'explicit_render_completed': bool(render and render_status_after is not None),
            'replicator_status_before': render_status_before,
            'replicator_status_after': render_status_after,
            'frozen_state_capture': bool(frozen_state),
            'scene_update_barrier_app_ticks': 3 if frozen_state else 0,
            'requested_render_subframes': (4 if frozen_state else 1) if render else 0,
            'strict_sync_verified': False,
            'sync_status': 'strict_not_verified' if not render else 'explicit_pair_runtime_contract',
            'sync_note': ('RGB is the latest app render and may be stale.' if not render else
                          'Explicit zero-time render was triggered before thermal read; GUI strict pixel-buffer verification remains pending.'),
            'epoch': epoch,
            'pair_pose_max_error': pose_error,
            'simulation_capture_drift_s': float(timeline.get_current_time() - timeline_before),
            'thermal_imaging': imaging_metadata(self.thermal.config, self.thermal.state,
                                               self.display_profile),
            'resolution': [self.width, self.height],
            'rgb_resolution': [self.width, self.height],
            'thermal_resolution': [self.thermal_width, self.thermal_height],
            'display_range_diagnostics': display_range_metadata(self.display_profile, outputs.apparent_t),
            'thermal_clock_drift_s': float(
                self.thermal.last_frame_t_s - self.thermal.scene.t0_s - thermal_t
            ),
            'wall_fps': None,
            'thermal_target_temperatures_k': {
                name: float(value) for name, value in self.thermal.bridge.temperatures().items()
                if name in self.metadata['thermal_configuration']['initial_target_temperatures_k']
            },
            'camera_world_matrix': [[float(rgb_pose[i][j]) for j in range(4)] for i in range(4)],
            'thermal_camera_world_matrix': [[float(thermal_pose[i][j]) for j in range(4)] for i in range(4)],
        }
        # A magenta surface is missing infrared physics, not a temperature.
        # Keep the vendor's visible warning and expose the actual renderer paths.
        last = getattr(self.thermal, '_last', None)
        if last is not None:
            unmapped = np.asarray(last.unmapped, dtype=bool)
            ids = np.asarray(last.instance_id)
            labels = last.labels
            missing_ids = np.unique(ids[unmapped]).tolist()
            meta['thermal_coverage'] = {
                'unmapped_pixels': int(unmapped.sum()), 'gbuffer_pixels': int(unmapped.size),
                'unmapped_paths': [labels.get(int(i), f'<unlabelled id {i}>') for i in missing_ids],
                'complete_for_this_view': not bool(unmapped.any()),
                'apparent_t_nonfinite_pixels': int((~np.isfinite(outputs.apparent_t)).sum()),
            }
        now = time.perf_counter()
        if self.last_capture_wall is not None and now > self.last_capture_wall:
            meta['wall_fps'] = 1.0 / (now - self.last_capture_wall)
        self.last_capture_wall = now
        if publish:
            self.frames += 1
        self.last_timings = {
            'render_step_s': t1 - t0,
            'explicit_render_s': t1 - t0 if render else 0.0,
            'render_wait_s': t1 - t0 if render else 0.0,
            'thermal_get_outputs_s': t2 - t1,
            'rgb_read_s': t3 - t2,
            'ui_payload_s': t4 - t3,
            'ui_upload_s': t5 - t4,
            'capture_total_s': t5 - capture_started,
        }
        for name, value in self.last_timings.items():
            self._timing_totals[name] = self._timing_totals.get(name, 0.0) + value
        if self.profile_enabled:
            meta['timings_s'] = dict(self.last_timings)
        if self.show_ui and publish:
            self.status.text = f"frame {self.frames} | sim {sim_before:.3f}s | paired FPS {meta['wall_fps'] or 0:.1f}"
        return rgb, ThermalFrame(display, np.asarray(outputs.radiance), np.asarray(outputs.apparent_t), dn16), meta

    def publish_pair(self, rgb, thermal, meta):
        """Only accepted preview pairs receive IDs and reach the UI."""
        started = time.perf_counter()
        if self.show_ui:
            for provider, pixels in ((self.rgb_provider, rgb), (self.thermal_provider, thermal.display8)):
                data, size = ui_rgba_array(pixels)
                provider.set_data_array(data, size, strict=True)
            self.status.text = (f"frame {meta['frame_id']} | controller {meta['controller_sim_time']:.3f}s | "
                                f"capture request {meta['rgb_thermal_capture_time']:.3f}s | "
                                f"compute {meta['preview_processing_ms']:.0f}ms | source age unknown")
            now_wall = time.perf_counter()
            if self._first_publish_wall is None:
                self._first_publish_wall = now_wall
            if self._last_publish_wall is not None:
                interval = now_wall - self._last_publish_wall
                # Ignore duplicate publishes inside the same millisecond: they are bookkeeping,
                # not new pairs, and they would inflate the rate.
                if interval > 0.002:
                    instant = min(1.0 / interval, 240.0)
                    self._fps = instant if self._fps == 0.0 else 0.85 * self._fps + 0.15 * instant
                    self._fps_count += 1
            self._last_publish_wall = now_wall
            average_fps = (self._fps_count / (now_wall - self._first_publish_wall)
                           if self._first_publish_wall and now_wall > self._first_publish_wall else 0.0)
            self.flight_status.text = (f"live_preview_not_strict | dropped {meta['dropped_preview_frames']} | "
                                      f"capture wall {meta['capture_finished_wall_s']:.3f}s | "
                                      f"sim {meta['controller_sim_time']:.2f}s")
            self.perf_status.text = (f"FPS {self._fps:5.1f} (avg {average_fps:5.1f}) | "
                                     f"capture {meta['rgb_thermal_capture_time'] * 1000:.0f} ms | "
                                     f"compute {meta['preview_processing_ms']:.0f} ms | "
                                     f"frame {meta['frame_id']}")
        self.last_timings['ui_upload_s'] = time.perf_counter() - started
        self.last_timings['capture_total_s'] = time.perf_counter() - meta['capture_started_wall_s']
        self.frames += 1

    def timing_summary(self):
        """Return average wall-clock timings for completed captures."""
        if not self.frames:
            return {}
        return {name: value / self.frames for name, value in self._timing_totals.items()}

    def save_frame(self, output, index, rgb, thermal, meta, *, metadata_writer=None):
        write_started = time.perf_counter()
        output.mkdir(parents=True, exist_ok=True)
        output_key = pathlib.Path(output).resolve()
        frame_stem = f'frame_{index:06d}'
        existing_outputs = [output / name for name in (
            f'{frame_stem}.json', f'{frame_stem}_rgb.npy', f'{frame_stem}_display8.npy',
            f'{frame_stem}_radiance.npy', f'{frame_stem}_apparent_t.npy',
            f'{frame_stem}_dn16.npy', f'{frame_stem}_rgb.png', f'{frame_stem}_thermal.png')
            if (output / name).exists()]
        if existing_outputs:
            raise FileExistsError(existing_outputs[0])
        if metadata_writer is None:
            writers = getattr(self, '_metadata_writers', None)
            if writers is None:
                writers = self._metadata_writers = {}
            metadata_writer = writers.get(output_key)
            if metadata_writer is None:
                candidate = output_key / f'frame_{index:06d}.json'
                if candidate.exists():
                    raise FileExistsError(candidate)
                metadata_writer = FrameMetadataWriter(output_key, meta)
                writers[output_key] = metadata_writer
        metadata_writer.ensure_available(index)
        np.save(output / f'frame_{index:06d}_rgb.npy', np.asarray(rgb))
        np.save(output / f'frame_{index:06d}_display8.npy', thermal.display8)
        np.save(output / f'frame_{index:06d}_radiance.npy', thermal.radiance)
        np.save(output / f'frame_{index:06d}_apparent_t.npy', thermal.apparent_t)
        if thermal.dn16 is not None:
            np.save(output / f'frame_{index:06d}_dn16.npy', thermal.dn16)
        from PIL import Image
        Image.fromarray(np.asarray(rgb)).save(output / f'frame_{index:06d}_rgb.png')
        Image.fromarray(thermal.display8).save(output / f'frame_{index:06d}_thermal.png')
        meta['file_write_s'] = time.perf_counter() - write_started
        metadata_writer.write(index, meta)

    def frame_metadata_summary(self, output):
        output_key = pathlib.Path(output).resolve()
        writer = getattr(self, '_metadata_writers', {}).get(output_key)
        return None if writer is None else writer.summary

    def close(self):
        if self._closed:
            return
        self._closed = True
        import omni.replicator.core as rep
        _stop_timeline()
        rep.orchestrator.wait_until_complete()
        if self.window is not None:
            self.window.destroy()
        errors = self.cleanup.close()
        if errors:
            print('[dual-camera] cleanup warnings: ' + '; '.join(errors), file=sys.stderr)


def _stop_timeline():
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.stop()
    timeline.commit()


def _destroy_thermal_product(thermal):
    product = getattr(thermal, '_render_product', None)
    if product is not None:
        product.destroy()


def _build_thermal(stage, camera_path, *, width, height, focal_length, horizontal_aperture,
                   sensor_path=DEFAULT_SENSOR, scene_path=DEFAULT_SCENE,
                   thermal_config_path=DEFAULT_THERMAL_CONFIG,
                   flat_field_enabled=False, sensor_seed=0,
                   material_cache_enabled=False, material_directory=None):
    # Same user temperature/material configuration as static capture, never authored to USD.
    from scripts.capture_crazyflie_thermal_static import _build_camera
    from irsim.config.loader import load_sensor_config
    from irsim.config.sensor import SensorConfig
    original = load_sensor_config(sensor_path)
    sensor = SensorConfig.model_validate(preview_sensor_data(
        original.model_dump(), width, height, focal_length, horizontal_aperture))
    return _build_camera(stage, sensor_path, scene_path, camera_path, capture_rgb=False,
                         sensor_config=sensor, thermal_config_path=thermal_config_path,
                         flat_field_enabled=flat_field_enabled, sensor_seed=sensor_seed,
                         material_cache_enabled=material_cache_enabled, material_directory=material_directory)


def open_preview(app, *, scene_config: SceneConfig | None = None,
                 stage_path=None, width=None, height=None,
                 thermal_width=None, thermal_height=None, fast=None,
                 thermal_config_path=None, start_controller=True,
                 motion_config=None, show_ui=True, renderer='PathTracing', disable_external_viewport=False,
                 shared_live_render_product=False, material_cache_enabled=False,
                 frozen_subframes=None,
                 task_instruction=None, defer_thermal=False):
    if defer_thermal and shared_live_render_product:
        raise ValueError('Deferred IR requires a separate RGB render product')
    import omni.replicator.core as rep
    import omni.usd
    import omni.timeline
    from scripts.start_crazyflie_controller import start_crazyflie

    config = scene_config or load_scene_config(DEFAULT_SCENE_CONFIG)
    from scripts.thermal_display import load_display_profile
    display_profile = (None if config.sensors.display_config is None else
                       load_display_profile(config.sensors.display_config))
    stage_path = config.scene.stage if stage_path is None else pathlib.Path(stage_path)
    width = config.sensors.width if width is None else width
    height = config.sensors.height if height is None else height
    thermal_width = width if thermal_width is None else thermal_width
    thermal_height = height if thermal_height is None else thermal_height
    if shared_live_render_product and (thermal_width, thermal_height) != (width, height):
        raise ValueError('Shared live render product requires equal RGB and thermal resolutions')
    fast = config.sensors.preview_fast if fast is None else fast
    thermal_config_path = config.sensors.thermal_config if thermal_config_path is None else thermal_config_path
    source_camera = config.drone.camera_path
    body_path = config.drone.body_path
    context = omni.usd.get_context()
    if not context.get_stage() or not context.get_stage().GetPrimAtPath(source_camera).IsValid():
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f'Could not open stage: {stage_path}')
        for _ in range(90):
            app.update()
    _stop_timeline()
    app.update()
    stage = context.get_stage()
    from scripts.city_material_partition import verify_overlay_source
    verify_overlay_source(stage)
    mount = CameraMount(stage, source_camera, body_path,
                        align_thermal_to_rgb=config.sensors.align_thermal_to_rgb)
    cleanup = CleanupStack()
    cleanup.add(mount.close)
    try:
        # User-selected static search targets live only in the mount's anonymous
        # session layer; the UAV controller and source robot asset stay intact.
        from pxr import Usd, UsdPhysics
        for target_path in config.scene.static_targets:
            target = stage.GetPrimAtPath(target_path)
            if not target or not target.IsActive():
                raise RuntimeError(f'Static search target missing/inactive: {target_path}')
            for prim in Usd.PrimRange(target):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                if prim.IsA(UsdPhysics.Joint):
                    UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        from omni.kit.viewport.window import get_viewport_window_instances
        for window in get_viewport_window_instances(None):
            viewport = window.viewport_api
            if viewport and viewport.usd_context_name == context.get_name():
                replacement = mount.camera_redirects.get(str(viewport.camera_path))
                if replacement:
                    viewport.camera_path = replacement
        thermal, sensor, unresolved, fallback = _build_thermal(
            stage, mount.thermal_path, width=thermal_width, height=thermal_height,
            focal_length=mount.source_focal, horizontal_aperture=mount.source_aperture,
            sensor_path=config.sensors.sensor_config,
            scene_path=config.scene.thermal_scene,
            thermal_config_path=thermal_config_path,
            flat_field_enabled=config.sensors.flat_field_enabled,
            sensor_seed=config.sensors.sensor_seed,
            material_cache_enabled=material_cache_enabled,
            material_directory=config.sensors.material_directory)
        cleanup.add(lambda: _destroy_thermal_product(thermal))
        cleanup.add(thermal.close)
        if fast:
            from scripts.preview_sky import PreviewSky
            sky = PreviewSky(thermal.config.sky, thermal.scene.t0_s)
            cleanup.add(sky.close)
            thermal.config = replace(thermal.config, sky=sky)
            thermal.bridge.sky = sky
        from irsim_isaac.pipeline.ir_camera import author_camera
        rgb_optics = scale_camera_optics_for_resolution(thermal.optics, width, height)
        author_camera(stage, mount.rgb_path, rgb_optics, sensor.sensor.optics.distortion)
        from irsim_isaac.geometry_probe import configure_renderer
        configure_renderer({'/rtx/rendermode': renderer})
        rgb_annotator = rep.AnnotatorRegistry.get_annotator('LdrColor')
        if not shared_live_render_product:
            rgb_product = rep.create.render_product(mount.rgb_path, (width, height))
            cleanup.add(rgb_product.destroy)
            cleanup.add(lambda: rgb_annotator.detach(rgb_product))
            rgb_annotator.attach(rgb_product)
        if not defer_thermal:
            thermal.open(settle_frames=8, rt_subframes=1)
        else:
            # RGB needs a usable buffer before entering the control loop; no IR AOVs attached.
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=1,delta_time=0.,pause_timeline=True,wait_for_render=True)
        from scripts.authored_lighting import apply_authored_lighting
        import carb
        lighting_report, restore_lighting = apply_authored_lighting(
            stage, carb.settings.get_settings())
        if restore_lighting is not None:
            cleanup.add(restore_lighting)
        if shared_live_render_product:
            # Two USD cameras remain, but coincident live views can read colour
            # and geometry from one Hydra product. Never used by offline replay.
            if not config.sensors.align_thermal_to_rgb:
                raise ValueError('Shared live product requires aligned cameras')
            rgb_product = thermal._render_product
            rgb_annotator.attach(rgb_product)
            cleanup.add(lambda: rgb_annotator.detach(rgb_product))
            # Colour is attached after the thermal reader's initialization.
            # Initialize that new SD graph before permitting live buffer reads.
            if rep.orchestrator.get_status() == rep.orchestrator.Status.STOPPING:
                rep.orchestrator.wait_until_complete()
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.0,
                                      pause_timeline=True, wait_for_render=True)
            initial_rgb = np.asarray(rgb_annotator.get_data())
            if initial_rgb.shape[:2] != (height, width):
                raise RuntimeError('Shared preview colour AOV did not initialize: '
                                   f'{initial_rgb.shape}; use separate products')
        # Third-party settling advances physics before our controller exists.
        # Stop while the same edit layer is active, restoring the starting pose.
        _stop_timeline()
        rep.orchestrator.wait_until_complete()
        omni.timeline.get_timeline_interface().set_auto_update(True)
        app.update()
        preview = DualCameraPreview(stage, mount, thermal, rgb_annotator, rgb_product,
                                    width=width, height=height,
                                    thermal_width=thermal_width, thermal_height=thermal_height,
                                    cleanup=cleanup, show_ui=show_ui,
                                    display_profile=display_profile, metadata={
            'scene_config_path': str(config.source_path),
            'scene_config_sha256': config.sha256,
            'stage_path': str(stage_path),
            'static_target_paths': list(config.scene.static_targets),
            'drone_prim_path': config.drone.prim_path,
            'drone_body_path': config.drone.body_path,
            'rgb_camera_path': mount.rgb_path, 'thermal_camera_path': mount.thermal_path,
            'resolution': [width, height], 'rgb_resolution': [width, height],
            'thermal_resolution': [thermal_width, thermal_height],
            'rgb_intrinsics_px': asdict(rgb_optics.intrinsics),
            'thermal_intrinsics_px': asdict(thermal.optics.intrinsics),
            'sensor_config_path': str(config.sensors.sensor_config),
            'sensor_config_sha256': __import__('hashlib').sha256(
                config.sensors.sensor_config.read_bytes()).hexdigest(),
            'thermal_config_path': str(thermal_config_path),
            'sensor_config': sensor.model_dump(mode='json'),
            'display_profile': None if display_profile is None else display_profile.metadata(),
            'fallback_material': fallback, 'fallback_material_records': unresolved,
            'temperature_unit': 'K', 'thermal_target_binding': 'per-Prim YAML configuration',
            'thermal_configuration': thermal.thermal_config_report,
            'thermal_config_sha256': thermal.thermal_config_report['sha256'],
            'thermal_environment_config_path': str(config.scene.thermal_scene),
            'thermal_environment_config_sha256': __import__('hashlib').sha256(
                config.scene.thermal_scene.read_bytes()).hexdigest(),
            'display_config_path': (None if config.sensors.display_config is None else
                                    str(config.sensors.display_config)),
            'display_config_sha256': (None if config.sensors.display_config is None else
                                      __import__('hashlib').sha256(
                                          config.sensors.display_config.read_bytes()).hexdigest()),
            'sky_time_interpolation_s': 30.0 if fast else None,
            'limitation': 'Prescribed per-object temperatures; library infrared materials; not calibrated',
            'ao_aov_enabled': False,
            'ao_policy': 'unsupported optional channel not attached; vendor default occlusion=1',
            'data_usage': 'pilot/development data',
            'shared_live_render_product': bool(shared_live_render_product),
            'material_cache_enabled': bool(material_cache_enabled),
            'rgb_render_camera_path': mount.thermal_path if shared_live_render_product else mount.rgb_path,
        })
        from scripts.experiment_conditions import snapshot_conditions
        preview.metadata['experiment_conditions'] = snapshot_conditions(
            stage, config, width, height,
            thermal_width=thermal_width, thermal_height=thermal_height)
        if lighting_report is not None:
            preview.metadata['authored_lighting'] = lighting_report
            preview.metadata['experiment_conditions']['low_light'].update(
                preset=lighting_report['preset'], runtime_lighting_override=True,
                configuration_verified=True,
                dependency_sha256=lighting_report['dependency_sha256'])
        preview.metadata['renderer'] = renderer
        if disable_external_viewport:
            preview.metadata['disabled_viewports'] = configure_live_viewports(cleanup, enabled=False)
        preview._ensure_window()
        controller = None
        if start_controller:
            controller = start_crazyflie(prim_path=config.drone.prim_path,
                                     minimum_height=config.drone.minimum_height,
                                     motion_config=motion_config,
                                     task_instruction=task_instruction)
            controller._window.position_x = 890
            controller._window.position_y = 95
        if frozen_subframes is not None:
            preview.frozen_subframes = max(1, int(frozen_subframes))
        return preview, controller
    except Exception:
        _stop_timeline()
        cleanup.close()
        raise


def preview_motion_config(speed_scale=1.0, horizontal_speed=None, vertical_speed=None):
    if not math.isfinite(speed_scale) or not 0.25 <= speed_scale <= 4.0:
        raise ValueError('--speed-scale must be between 0.25 and 4.0')
    if horizontal_speed is not None and (
            not math.isfinite(horizontal_speed) or not 0.1 <= horizontal_speed <= 20.0):
        raise ValueError('--horizontal-speed must be between 0.1 and 20.0 m/s')
    if vertical_speed is not None and (
            not math.isfinite(vertical_speed) or not 0.1 <= vertical_speed <= 5.0):
        raise ValueError('--vertical-speed must be between 0.1 and 5.0 m/s')
    base = MotionConfig()
    horizontal = base.max_horizontal_speed * speed_scale if horizontal_speed is None else horizontal_speed
    # --speed-scale scales climb and yaw together, which is wrong when only the climb rate
    # needs to change (doubling yaw to 90 deg/s makes the inspection view unusable). The
    # dedicated --vertical-speed overrides just the climb axis.
    vertical = base.max_vertical_speed * speed_scale if vertical_speed is None else vertical_speed
    return MotionConfig(
        max_horizontal_speed=horizontal,
        max_vertical_speed=vertical,
        max_yaw_speed=base.max_yaw_speed * speed_scale,
        # Keep the time to reach maximum horizontal speed at its original value.
        horizontal_acceleration=base.horizontal_acceleration * horizontal / base.max_horizontal_speed,
        # Scale with the vertical speed so the climb still reaches top speed in about a second.
        vertical_acceleration=base.vertical_acceleration * (vertical / base.max_vertical_speed),
        yaw_acceleration=base.yaw_acceleration * speed_scale,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=pathlib.Path, default=DEFAULT_SCENE_CONFIG,
                        help='single YAML file containing stage, drone and sensor paths')
    parser.add_argument('--stage', type=pathlib.Path, default=None,
                        help='legacy override; prefer --config')
    parser.add_argument('--thermal-config', type=pathlib.Path, default=None,
                        help='per-object Celsius temperatures and infrared material bindings')
    parser.add_argument('--frames', type=int, default=0, help='0 means run until preview closes')
    parser.add_argument('--out', type=pathlib.Path, default=None)
    parser.add_argument('--width', type=int, default=None, help='legacy resolution override')
    parser.add_argument('--height', type=int, default=None, help='legacy resolution override')
    parser.add_argument('--profile', action='store_true',
                        help='record per-stage capture timings and print their averages')
    parser.add_argument('--reference', action='store_true', help='use original per-frame sky calculation')
    parser.add_argument('--no-save', action='store_true', help='benchmark without frame file I/O')
    parser.add_argument('--explicit-render', action='store_true', help='force an extra Replicator render per pair')
    parser.add_argument('--preview-mode', choices=('live', 'offline'), default=None,
                        help='live: scheduled latest-buffer preview; offline: frozen explicit capture')
    parser.add_argument('--preview-fps', type=float, default=15.,
                        help='target preview rate; 0 disables artificial rate limiting')
    parser.add_argument('--live-refresh', choices=('paired', 'split', 'on-demand'), default='paired')
    parser.add_argument('--probe-ir-once',action='store_true',help='synthetic inspection request; requires --probe-control')
    parser.add_argument('--thermal-preview-fps', type=float, default=2.)
    parser.add_argument('--benchmark-seconds', type=float, default=0.)
    parser.add_argument('--benchmark-warmup', type=float, default=10.)
    parser.add_argument('--start-position', type=float, nargs=3)
    parser.add_argument('--start-yaw-degrees', type=float)
    parser.add_argument('--max-preview-age-ms', type=float, default=500.)
    parser.add_argument('--renderer', choices=('PathTracing', 'RealTimePathTracing'), default='PathTracing')
    parser.add_argument('--renderer-proof', type=pathlib.Path)
    parser.add_argument('--disable-external-viewport', action='store_true')
    parser.add_argument('--save-preview', action='store_true', help='save diagnostic preview ONLY, never benchmark data')
    parser.add_argument('--skip-live-preconversion', action='store_true',
                        help='live preview only: avoid converting a frame before publish_pair repeats it')
    parser.add_argument('--shared-live-render-product', action='store_true',
                        help='experimental live-only coincident RGB/thermal view from one render product')
    parser.add_argument('--disable-material-cache', action='store_true',
                        help='diagnostic comparison: rebuild instance-to-material mapping each frame')
    parser.add_argument('--enable-material-cache', action='store_true',
                        help='live-only experimental mapping cache; default remains off')
    parser.add_argument('--compact-motion', action='store_true',
                        help='live-only invert only visible transform matrices; retain motion effects')
    parser.add_argument('--headless', action='store_true', help='automated performance run, no UI')
    parser.add_argument('--probe-control', action='store_true', help='synthetic held-key probe; NOT human input latency')
    parser.add_argument('--record-trajectory', action='store_true', help='buffer flight states and write trajectory.jsonl on exit')
    parser.add_argument('--collision-undo', action=argparse.BooleanOptionalAction, default=None,
                        help='static obstacle hold and manual 3-second undo; default on for manual recordings')
    parser.add_argument('--trajectory-root', type=pathlib.Path,
                        default=PROJECT_ROOT / 'validation' / 'trajectory_runs')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--trajectory-period', type=float, default=TRAJECTORY_PERIOD_S,
                        help='minimum simulation seconds between trajectory samples')
    parser.add_argument('--speed-scale', type=float, default=1.0,
                        help='scale vertical/yaw speed and matching acceleration (default: 1.0)')
    parser.add_argument('--horizontal-speed', type=float, default=5.0,
                        help='horizontal speed in m/s (0.1-20; default: 5); leave height/yaw settings unchanged')
    parser.add_argument('--vertical-speed', type=float, default=None,
                        help='vertical (climb) speed in m/s (0.1-5; default: 0.5 x speed-scale); '
                             'leaves horizontal and yaw settings unchanged')
    parser.add_argument('--task-file', type=pathlib.Path, default=None,
                        help='override the scene-config task YAML; scene task is used by default')
    parser.add_argument('--task-id', default=None)
    parser.add_argument('--instruction', default=None)
    parser.add_argument('--target-prim-path', default=None)
    parser.add_argument('--target-description', default=None)
    args = parser.parse_args(argv)
    from scripts.manual_collision_undo import collision_undo_enabled, RECOVERY_CONTRACT
    try:
        args.collision_undo = collision_undo_enabled(record_trajectory=args.record_trajectory,
            probe_control=args.probe_control, requested=args.collision_undo)
    except ValueError as exc:
        parser.error(str(exc))
    if args.trajectory_period <= 0:
        parser.error('--trajectory-period must be positive')
    try:
        # The scene may carry its own cruise speed; the CLI still overrides it.  The scene config
        # is loaded a few lines below, so read the value there and fall back to the CLI default.
        motion_config = preview_motion_config(args.speed_scale, args.horizontal_speed,
                                              args.vertical_speed)
    except ValueError as exc:
        parser.error(str(exc))
    config = load_scene_config(args.config)
    if args.horizontal_speed is None and config.drone.horizontal_speed is not None:
        motion_config = preview_motion_config(
            args.speed_scale, config.drone.horizontal_speed, args.vertical_speed)
    task_cli_overrides = {
        'task_id': args.task_id,
        'instruction': args.instruction,
        'target_prim_path': args.target_prim_path,
        'target_description': args.target_description,
    }
    try:
        task = resolve_task_config(config.task, task_file=args.task_file,
                                   cli_overrides=task_cli_overrides)
    except ValueError as exc:
        parser.error(str(exc))
    task_source_path = args.task_file.expanduser().resolve() if args.task_file else config.task_path
    task_source_sha256 = (config.task_sha256 if args.task_file is None else
                          __import__('hashlib').sha256(task_source_path.read_bytes()).hexdigest())
    task_cli_overrides = {key: value for key, value in task_cli_overrides.items() if value is not None}
    mode = args.preview_mode or ('legacy_explicit' if args.explicit_render else 'live')
    if args.explicit_render and args.preview_mode == 'live':
        parser.error('--explicit-render conflicts with --preview-mode live')
    if args.shared_live_render_product and mode != 'live':
        parser.error('Shared render product is live-preview-only; offline capture stays unchanged')
    if args.live_refresh=='on-demand' and (mode!='live' or args.shared_live_render_product or args.save_preview):
        parser.error('on-demand requires live mode, separate RGB product and no saved preview pairs')
    if args.probe_ir_once and (not args.probe_control or args.live_refresh!='on-demand'):
        parser.error('--probe-ir-once requires synthetic --probe-control and on-demand mode')
    if mode == 'offline' and (args.record_trajectory or args.probe_control):
        parser.error('Offline frozen captures do not drive or record a live trajectory; use replay_trajectory_highres.py')
    if mode != 'live' and args.renderer != 'PathTracing':
        parser.error('Offline and legacy explicit modes keep the validated PathTracing renderer')
    if args.renderer != 'PathTracing' and (args.stage or args.thermal_config or args.reference):
        parser.error('Renderer proof requires unmodified scene-config conditions')
    try:
        validate_split_options(args.live_refresh,args.save_preview)
        scheduler = PreviewScheduler(args.preview_fps, args.max_preview_age_ms / 1000.)
        split_scheduler = (SplitPreviewScheduler(args.preview_fps, args.thermal_preview_fps)
                           if args.live_refresh in ('split','on-demand') else None)
        content_tracker = PreviewContentTracker()
        verify_renderer_evidence(args.renderer_proof, config, args.renderer,
                                 args.width or config.sensors.width, args.height or config.sensors.height)
    except ValueError as exc:
        parser.error(str(exc))
    # Legacy flags remain useful for quick experiments, but do not create a second source of truth.
    stage = args.stage or config.scene.stage
    thermal_config = args.thermal_config or config.sensors.thermal_config
    load_thermal_config(thermal_config)  # Report YAML errors before starting Kit.
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': args.headless, 'renderer': args.renderer, 'multi_gpu': False,
                         'samples_per_pixel_per_frame': 1})
    preview = controller = None
    recorder = None
    control_metrics = None
    timeline_sub = None
    count = 0
    frame_times = []
    capture_interruptions = []
    app_update_timings = []
    capture_timings = []
    render_evidence = []
    last_meta = None
    render_calls = 0
    completion = 'running'
    benchmark_start = measurement_start = None
    measured = False
    rgb_refresh_times = []
    inspection_events=[]
    inspection_probed=False
    cruise_thermal_compute_calls=0
    output = args.out or PROJECT_ROOT / 'validation' / ('dual-camera-' + time.strftime('%Y%m%d-%H%M%S'))
    try:
        preview, controller = open_preview(app, scene_config=config, stage_path=stage,
                                           width=args.width, height=args.height,
                                           fast=(False if args.reference else None),
                                           thermal_config_path=thermal_config,
                                           motion_config=motion_config, show_ui=not args.headless,
                                           renderer=args.renderer,
                                           disable_external_viewport=args.disable_external_viewport,
                                           shared_live_render_product=args.shared_live_render_product,
                                           material_cache_enabled=args.enable_material_cache and mode == 'live'
                                           and not args.disable_material_cache,
                                           task_instruction=task.get('instruction'),defer_thermal=args.live_refresh=='on-demand')
        preview.on_demand_ir = args.live_refresh=='on-demand'
        if preview.on_demand_ir:
            controller.inspection_callback=lambda:setattr(preview,'inspection_pending',True)
            if getattr(controller,'_task_status',None) is not None:
                controller._task_status.text='H: IR snapshot | T: manual confirmation'
            last_meta={**preview.metadata,'capture_mode':'RGB_only_cruise','strict_sync_verified':False}
            ir_counts={'total':0,'cruise':0,'inspection':0}
            preview.inspection_active=False
            original_get_outputs=preview.thermal.get_outputs
            def counted_ir_outputs(*pos,**kw):
                ir_counts['total']+=1
                ir_counts['inspection' if preview.inspection_active else 'cruise']+=1
                return original_get_outputs(*pos,**kw)
            preview.thermal.get_outputs=counted_ir_outputs
        # Operator guidance uses the evaluator's surface-distance convention.
        # Projected image coordinates are not visibility/occlusion evidence.
        _task_ev = (task.get('environment_evidence') or {}).get('factory_inspection') or {}
        _target = task.get('target_position') or _task_ev.get('hotspot_position_m')
        if (task.get('environment_evidence') or {}).get('collection_episode',{}).get('scene')=='forest':
            preview.anchor_guidance_note='Forest: anchor distance is a point distance; aim about 8m. T remains manual.'
        elif _target:
            _req = _task_ev.get('inspection_requirements') or {}
            preview.set_pilot_target(_target,
                                     hold_required_s=_req.get('hold_duration_s', 3.0),
                                     distance_range_m=_req.get('observation_distance_range_m', (6.0, 8.0)),
                                     inspection=_task_ev)
        preview.thermal.compact_motion_enabled = args.compact_motion and mode == 'live'
        # This is a live-preview-only micro-optimization. Offline captures
        # still use the complete capture path and are unaffected.
        preview.skip_live_preconversion = bool(args.skip_live_preconversion and mode == 'live')
        control_metrics = ControlMetrics(controller)
        if args.start_position is not None:
            from scripts.drone_control_math import orientation_wxyz_from_rpy
            if args.start_yaw_degrees is None or not all(math.isfinite(v) for v in [*args.start_position, args.start_yaw_degrees]):
                raise ValueError('start position requires finite yaw')
            controller.initial_pose = (tuple(args.start_position),
                orientation_wxyz_from_rpy(0., 0., math.radians(args.start_yaw_degrees)))
        if args.probe_control:
            # Performance instrumentation only; controller source and motion math unchanged.
            controller._has_control_focus = lambda: True
        preview.metadata['controller_parameters'] = asdict(motion_config)
        print(f'[flight] horizontal={motion_config.max_horizontal_speed:g} m/s, '
              f'vertical={motion_config.max_vertical_speed:g} m/s, '
              f'yaw={math.degrees(motion_config.max_yaw_speed):g} deg/s '
              '(per simulation second)', flush=True)
        if args.record_trajectory:
            import omni.timeline
            parameter_snapshot = None
            if config.sensors.material_directory is not None:
                from scripts.runtime_parameter_provenance import thermal_input_snapshot
                if pathlib.Path(thermal_config).resolve()!=config.sensors.thermal_config:
                    raise ValueError('Project-material recording uses the thermal config named by the scene YAML; update that YAML instead of a CLI override')
                parameter_snapshot = thermal_input_snapshot(config)
                if (parameter_snapshot['material_library_sha256'] != preview.thermal.thermal_config_report['material_library_sha256'] or
                        parameter_snapshot['weather_content_sha256'] != preview.thermal.scene.weather.content_hash or
                        parameter_snapshot['file_sha256'][str(pathlib.Path(thermal_config).resolve())] != preview.thermal.thermal_config_report['sha256']):
                    raise ValueError('Thermal inputs changed between preview initialization and recording')
            timeline = omni.timeline.get_timeline_interface()
            recording_timeline = configure_recording_timeline(timeline)
            print('[trajectory] recording clock: '+json.dumps(recording_timeline), flush=True)
            task_start_simulation_time = float(timeline.get_current_time())
            recorder = TrajectoryRecorder.create(args.trajectory_root, {
                'run_id': args.run_id or 'timestamped', 'created_local': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'stage_path': str(config.scene.stage),
                'stage_sha256': __import__('hashlib').sha256(config.scene.stage.read_bytes()).hexdigest(),
                'scene_config_path': str(config.source_path),
                'scene_config_sha256': config.sha256, 'drone_prim_path': config.drone.prim_path,
                'body_path': config.drone.body_path, 'rgb_camera_path': preview.mount.rgb_path,
                'thermal_camera_path': preview.mount.thermal_path,
                'sensor_config': str(config.sensors.sensor_config),
                'sensor_config_sha256': __import__('hashlib').sha256(config.sensors.sensor_config.read_bytes()).hexdigest(),
                **({'authored_lighting': preview.metadata['authored_lighting']}
                   if 'authored_lighting' in preview.metadata else {}),
                'thermal_config': str(thermal_config),
                'thermal_config_sha256': __import__('hashlib').sha256(pathlib.Path(thermal_config).read_bytes()).hexdigest(),
                **({} if parameter_snapshot is None else {'thermal_input_snapshot': parameter_snapshot}),
                'resolution': [preview.width, preview.height], 'sample_period_s': args.trajectory_period,
                'capture_contract': capture_contract(args.trajectory_period),
                'collision_undo': args.collision_undo,
                'collision_contract': RECOVERY_CONTRACT if args.collision_undo else 'disabled',
                'trajectory_sync_status': 'timestamped_state_records',
                'preview_sync_status': 'live_preview_not_strict',
                'sync_validation_status': 'offline_capture_pipeline_validated',
                'sync_validation_scope': 'offline_replay_capture_pipeline; live preview images are asynchronous',
                'sync_validation_evidence': {
                    'report': 'validation/dual-camera-sync/spot-002-pixel-sync-recheck-20261013-b/summary.json',
                    'report_sha256': '9b5d87b6a9e6ce969f95a00237bfe592a98e4f105dee124333b5324ea01c6b73',
                    'result': 'PASS', 'frames_tested': 16,
                },
                'trajectory_source': 'synthetic_control_probe' if args.probe_control else 'manual_driving',
                'data_usage': 'pilot/development data', 'strict_sync_verified': False,
                'controller_version': 'direct_pose_physics_step_v1', 'task': task,
                'trajectory_pose_schema': 'actual_usd_body_world_transform_v1',
                'trajectory_pose_semantics': (
                    'position/orientation are sampled from body_path in the same USD state as camera matrices; '
                    'controller_target_pose preserves the controller internal target for diagnostics'),
                'task_config_path': str(task_source_path) if task_source_path else None,
                'task_config_sha256': task_source_sha256,
                'task_cli_overrides': task_cli_overrides,
                'controller_parameters': asdict(motion_config),
                'speed_scale': args.speed_scale,
                'task_start_simulation_time_s': task_start_simulation_time,
                'recording_timeline': recording_timeline,
                'start_position': None, 'target_position': task.get('target_position'), 'success': None,
                'success_time': None, 'notes': ''}, run_id=args.run_id)
            from scripts.on_demand_ir import bind_inspection_history
            bind_inspection_history(recorder,inspection_events)
            preview.mileage_provider = lambda: recorder.mileage.status_text(target=recorder.metadata.get('target_position'))
            def mark_task_success():
                simulation_time = float(timeline.get_current_time())
                elapsed_time = max(0.0, simulation_time - task_start_simulation_time)
                from pxr import UsdGeom
                body_pose = np.asarray(UsdGeom.Xformable(
                    preview.stage.GetPrimAtPath(config.drone.body_path)
                ).ComputeLocalToWorldTransform(0)).tolist()
                sampled_pose = trajectory_state_from_world_transform(
                    controller.state_snapshot(), body_pose)
                if not controller._ready or not recorder.records:
                    return False
                sampled_pose.update(frame_id=-1, simulation_time=simulation_time,
                    rgb_camera_world_matrix=np.asarray(UsdGeom.Xformable(
                        preview.stage.GetPrimAtPath(preview.mount.rgb_path)).ComputeLocalToWorldTransform(0)).tolist(),
                    thermal_camera_world_matrix=np.asarray(UsdGeom.Xformable(
                        preview.stage.GetPrimAtPath(preview.mount.thermal_path)).ComputeLocalToWorldTransform(0)).tolist())
                marked = recorder.mark_success(simulation_time, sampled_pose['position'],
                                               elapsed_time=elapsed_time, terminal_snapshot=sampled_pose)
                if marked:
                    print(f'[task] success marked by T at {elapsed_time:.3f}s; '
                          f'position={tuple(round(float(v), 3) for v in sampled_pose["position"])}', flush=True)
                return marked
            controller.set_task_success_callback(mark_task_success)
            if args.collision_undo:
                from scripts.manual_collision_undo import prepare_guard,RecordingRecovery
                guard,guard_metadata=prepare_guard(preview.stage,config.drone.prim_path,config.drone.body_path,
                    exclude_paths=config.scene.collision_exclude_paths)
                recorder.metadata['collision_guard']=guard_metadata
                recovery=RecordingRecovery(recorder,timeline.get_current_time,guard)
                controller.collision_guard=guard
                controller.collision_callback=recovery.collision
                controller.undo_callback=recovery.prepare_undo
                controller.undo_committed_callback=recovery.commit_undo
                controller._undo_button.visible=True
                controller._collision_status.visible=True
                controller._window.height=330 if controller.task_instruction else 260
                controller._collision_status.text='Collision protection active. Home disabled; undo only after contact.'
                print(f'[collision] session-only static meshes={guard_metadata["mesh_count"]}; sphere={guard.radius:.3f}m',flush=True)
            # Stop rewinds simulation time, so it ends this recording epoch.
            def _close_on_stop(event):
                if event.type == int(omni.timeline.TimelineEventType.STOP):
                    controller.set_task_success_callback(None)
                    try:
                        recorder.close()
                        print(f'[trajectory] recording ended on Stop: {recorder.run_dir}; '
                              'start a new run to record again', flush=True)
                    except Exception as exc:
                        print(f'[trajectory] stop save failed: {exc}', file=sys.stderr, flush=True)
            timeline_sub = omni.timeline.get_timeline_interface().get_timeline_event_stream().create_subscription_to_pop(_close_on_stop)
        preview.profile_enabled = args.profile
        if args.profile:
            import cProfile
            preview.profiler = cProfile.Profile()
        import omni.timeline
        omni.timeline.get_timeline_interface().play()
        if mode == 'offline':
            omni.timeline.get_timeline_interface().pause()
        print('[dual-camera] started; close the window or press Ctrl+C to stop', flush=True)
        next_trajectory_time = None
        def visible():
            return preview.window is None or preview.window.visible
        probe_sent = False
        probe_last_count = -1
        while app.is_running() and visible() and (args.frames <= 0 or count < args.frames):
            update_started = time.perf_counter()
            app.update()
            app_update_timings.append(time.perf_counter() - update_started)
            if not app.is_running() or not visible():
                break
            if not controller._running:
                raise RuntimeError(f'Controller stopped: {controller.diagnostics()}')
            wall = time.perf_counter()
            if benchmark_start is None and controller._ready:
                benchmark_start = wall
            if args.benchmark_seconds > 0 and benchmark_start is not None:
                if not measured and wall - benchmark_start >= args.benchmark_warmup:
                    measured = True
                    measurement_start = wall
                    frame_times.clear(); capture_timings.clear(); app_update_timings.clear()
                    rgb_refresh_times.clear()
                    control_metrics.clear_measurements()
                if measured and wall - measurement_start >= args.benchmark_seconds:
                    break
            if not controller._playing():
                preview.clock.sample(omni.timeline.get_timeline_interface().get_current_time())
                if preview.show_ui:
                    preview.status.text = 'Paused/stopped. Press Play to resume; click Drone Control to fly.'
            if args.probe_control and args.benchmark_seconds > 0 and controller._ready:
                # Same wall-clock input schedule in paired and split runs, independent of IR count.
                pressed = benchmark_start is not None and int((wall-benchmark_start)/2.) % 2 == 0
                if pressed != ('W' in controller.keys.pressed):
                    control_metrics.note_input('synthetic_key_state')
                    (controller.keys.press if pressed else controller.keys.release)('W')
            elif args.probe_control and controller._ready and not probe_sent:
                control_metrics.note_input('synthetic_key_state')
                controller.keys.press('W')
                probe_sent = True
            elif args.probe_control and controller._ready and count != probe_last_count and count > 0 and count % 8 == 0:
                # Repeat probes instead of inferring a latency distribution from one event.
                if 'W' in controller.keys.pressed:
                    controller.keys.release('W')
                else:
                    control_metrics.note_input('synthetic_key_state')
                    controller.keys.press('W')
                probe_last_count = count
            # Trajectory sampling follows app/physics updates, independent of preview schedule.
            if recorder is not None and not recorder._closed and controller._playing() and controller._ready:
                sim = float(omni.timeline.get_timeline_interface().get_current_time())
                validate_recording_time(recorder.records, sim)
                if ((not args.collision_undo or recovery.can_sample(sim)) and
                        (next_trajectory_time is None or sim >= next_trajectory_time - 1e-9)):
                    from pxr import UsdGeom
                    body_pose = np.asarray(UsdGeom.Xformable(
                        preview.stage.GetPrimAtPath(config.drone.body_path)
                    ).ComputeLocalToWorldTransform(0)).tolist()
                    poses = [np.asarray(UsdGeom.Xformable(preview.stage.GetPrimAtPath(p)).ComputeLocalToWorldTransform(0)).tolist()
                             for p in (preview.mount.rgb_path, preview.mount.thermal_path)]
                    state = trajectory_state_from_world_transform(
                        controller.state_snapshot(), body_pose)
                    if not recorder.records:
                        recorder.metadata['start_position'] = state['position']
                        recorder.metadata['start_yaw_radians'] = state['yaw']
                        recorder.metadata['first_record_simulation_time_s'] = sim
                    recorder.append({'frame_id': len(recorder.records), 'simulation_time': sim,
                        **state, 'rgb_camera_world_matrix': poses[0],
                        'thermal_camera_world_matrix': poses[1], 'scene_config_path': str(config.source_path),
                        'scene_config_sha256': config.sha256, 'resolution': [preview.width, preview.height],
                        'controller_version': 'direct_pose_physics_step_v1', 'controller_parameters': asdict(motion_config)})
                    next_trajectory_time = sim + args.trajectory_period
            ready = mode == 'offline' or (controller._playing() and controller._ready)
            if mode == 'live':
                preview.update_live_hud()
            started = time.perf_counter()
            if ready and preview.on_demand_ir:
                preview.sample_cruise_clock(float(omni.timeline.get_timeline_interface().get_current_time()))
                if getattr(preview.thermal,'_render_product',None) is not None:
                    raise RuntimeError('IR render product retained during RGB-only cruise')
                if args.probe_ir_once and not inspection_probed and count>=5:
                    preview.inspection_pending=True
                    inspection_probed=True
                if preview.inspection_pending:
                    preview.inspection_pending=False
                    preview.inspection_active=True
                    preview.inspection_error=None
                    try:
                        from scripts.on_demand_ir import capture_inspection
                        event=capture_inspection(preview,omni.timeline.get_timeline_interface(),app,controller)
                        inspection_events.append(event)
                        render_calls+=event['explicit_render_calls']
                        frame_times.append(time.perf_counter())
                        capture_timings.append(dict(preview.last_timings))
                        print('[inspection] '+json.dumps(event),flush=True)
                    except Exception as exc:
                        preview.inspection_error=str(exc)
                        inspection_events.append({'status':'FAILED','error':str(exc)})
                        if preview.show_ui:preview.status.text=f'IR inspection failed: {exc}; H to retry'
                        print(f'[inspection] FAILED: {exc}',file=sys.stderr,flush=True)
                    finally:
                        preview.inspection_active=False
                if split_scheduler.poll(time.perf_counter())['rgb']:
                    preview.publish_rgb()
                    rgb_refresh_times.append(time.perf_counter())
                    count+=1
                continue
            if ready and mode == 'live' and split_scheduler is not None:
                due = split_scheduler.poll(started)
                if due['rgb']:
                    preview.publish_rgb()
                    rgb_refresh_times.append(time.perf_counter())
                if due['thermal']:
                    rgb, thermal, meta = preview.capture(render=False, frozen_state=False, publish=False)
                    preview.publish_thermal(thermal, meta)
                    last_meta = meta
                    frame_times.append(time.perf_counter())
                    capture_timings.append(dict(preview.last_timings))
                    count += 1
                continue
            if ready and (mode != 'live' or scheduler.due(started)):
                source = render_dispatcher_state()
                if mode == 'live' and not scheduler.begin(started, source['compute_count']):
                    continue
                try:
                    render = mode != 'live'
                    render_calls += int(render)
                    rgb, thermal, meta = preview.capture(render=render, frozen_state=mode == 'offline',
                                                         publish=mode != 'live')
                except ExplicitRenderIncomplete:
                    reason = capture_interruption_reason(
                        app_running=app.is_running(), window_visible=visible(),
                        timeline_playing=controller._playing())
                    if reason is None:
                        raise  # Never treat a real render failure as a valid frame.
                    capture_interruptions.append({'reason': reason, 'next_frame_id': count})
                    print(f'[dual-camera] discarded interrupted pair: {reason}', flush=True)
                    if reason == 'window_closed':
                        break
                    if preview.show_ui:
                        preview.status.text = 'Paused/stopped; unfinished pair discarded. Press Play to resume.'
                    continue
                if mode == 'live':
                    meta.update(content_tracker.inspect(rgb, thermal.display8))
                    finished = time.perf_counter()
                    frame_id = scheduler.finish(started, finished, source['compute_count'],
                        valid=controller._playing() and app.is_running() and visible())
                    if frame_id is None:
                        capture_timings.append(dict(preview.last_timings))
                        continue
                    meta.update(frame_id=frame_id, frame=frame_id,
                        controller_sim_time=float(omni.timeline.get_timeline_interface().get_current_time()),
                        preview_age_ms=None, preview_processing_ms=(finished-started)*1000,
                        dropped_preview_frames=scheduler.dropped,
                        source_age_ms_verified=False,
                        source_dispatcher=source, actual_render_calls=0,
                        age_scope='Individual AOV age unavailable; processing duration is reported separately')
                    preview.publish_pair(rgb, thermal, meta)
                xyz = ', '.join(f'{v:.2f}' for v in controller.position)
                if preview.show_ui:
                    preview.flight_status.text += f' | XYZ {xyz} | yaw {np.degrees(controller.yaw):.1f}'
                last_meta = meta
                render_evidence.append(source)
                capture_timings.append(dict(preview.last_timings))
                frame_times.append(time.perf_counter())
                rgb_refresh_times.append(time.perf_counter())
                if not args.no_save and (mode != 'live' or args.save_preview) and (args.out is not None or args.frames > 0):
                    preview.save_frame(output, count, rgb, thermal, meta)
                count += 1
        print(f'[dual-camera] captured {count} preview frames; output={output}', flush=True)
        completion = 'completed'
    except KeyboardInterrupt:
        completion = 'user_interrupted'
        print('[dual-camera] stopped by user', flush=True)
    except Exception:
        completion = 'failed'
        # Kit shutdown can terminate Python before an unhandled traceback is printed.
        output.mkdir(parents=True, exist_ok=True)
        failure = traceback.format_exc()
        (output / 'failure.txt').write_text(failure, encoding='utf-8')
        print(f'[dual-camera] FAILED; details: {output / "failure.txt"}\n{failure}',
              file=sys.stderr, flush=True)
        raise
    finally:
        if preview is not None and (frame_times or rgb_refresh_times):
            intervals = np.diff(frame_times[2:])
            report = {'completion': completion, 'frames': count,
                      'paired_fps': float(1 / np.mean(intervals)) if len(intervals) and split_scheduler is None else None,
                      'live_refresh': args.live_refresh,
                      'inspection_events':inspection_events,
                      'thermal_compute_counts':ir_counts if preview.on_demand_ir else None,
                      'cruise_thermal_product_active':getattr(preview.thermal,'_render_product',None) is not None,
                      'rgb_refresh_fps': ((len(rgb_refresh_times)-1)/(rgb_refresh_times[-1]-rgb_refresh_times[0])
                                          if len(rgb_refresh_times)>1 else None),
                      'thermal_refresh_fps': float(1/np.mean(intervals)) if len(intervals) else None,
                      'thermal_target_fps': None if preview.on_demand_ir else args.thermal_preview_fps if split_scheduler else args.preview_fps,
                      'benchmark_measurement_s': (time.perf_counter()-measurement_start if measurement_start else None),
                      'p95_frame_ms': float(np.percentile(intervals, 95)*1000) if len(intervals) else None,
                      'p99_frame_ms': float(np.percentile(intervals, 99)*1000) if len(intervals) else None,
                      'resolution': [preview.width, preview.height],
                      'frame_files_saved': split_scheduler is None and not args.no_save and (mode != 'live' or args.save_preview),
                      'preview_mode': mode, 'renderer': args.renderer,
                      'preview_sync_status': 'live_preview_not_strict' if mode == 'live' else 'see_independent_marker_validation',
                      'strict_sync_verified': False, 'preview_target_fps': args.preview_fps,
                      'preview_rate_mode': 'uncapped' if args.preview_fps == 0 else 'capped',
                      'dropped_preview_frames': scheduler.dropped,
                      'actual_render_calls': render_calls,
                      'actual_gpu_render_passes': None,
                      'render_call_scope': 'project explicit step calls; NOT GPU passes or viewport renders',
                      'render_dispatcher_samples': render_evidence,
                      'stage_timings_s': {k: percentile_summary([r[k] for r in capture_timings[2:]])
                                          for k in capture_timings[0]} if capture_timings else {},
                      'app_update_s': percentile_summary(app_update_timings),
                      'control_metrics': control_metrics.report() if control_metrics else {},
                      **(content_tracker.report() if mode == 'live' else {
                          'old_aov_frames_verified': False,
                          'hash_repeated_pair_diagnostics': None}),
                      'verified_fresh_pair_fps': None,
                      'fps_scope': 'Published samples; hash similarity is diagnostic and individual AOV age is unverified' if mode == 'live' else 'see independent frozen marker test',
                      'vram_measurement': 'NOT_MEASURED',
                      'last_frame_metadata': last_meta,
                      'explicit_render': args.explicit_render, 'reference_sky': args.reference,
                      'timings_s': preview.timing_summary(),
                      'app_update_mean_s': float(np.mean(app_update_timings)),
                      'capture_interruptions': capture_interruptions,
                      'controller_parameters': asdict(motion_config),
                      'experiment_conditions': preview.metadata.get('experiment_conditions')}
            output.mkdir(parents=True, exist_ok=True)
            (output / 'performance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            if args.profile and preview.profiler is not None:
                preview.profiler.dump_stats(str(output / 'thermal.prof'))
            print('[dual-camera] performance: ' + json.dumps({
                key: value for key, value in report.items() if key != 'experiment_conditions'}), flush=True)
        if recorder is not None:
            try:
                recorder.metadata['preview_inspection_events']=inspection_events
                print(f'[trajectory] wrote {recorder.close()}', flush=True)
            except Exception as exc:
                print(f'[trajectory] failed to write: {exc}', file=sys.stderr, flush=True)
        if timeline_sub is not None:
            release = getattr(timeline_sub, 'reset', None) or getattr(timeline_sub, 'unsubscribe', None)
            if callable(release):
                release()
        if controller is not None:
            if control_metrics is not None:
                control_metrics.close()
            controller.stop()
        if preview is not None:
            preview.close()
        app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
