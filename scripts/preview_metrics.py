"""Read-only physics/input instrumentation; never moves the drone or owns its callbacks."""
import time
from scripts.preview_scheduler import percentile_summary


def render_dispatcher_state():
    """Global SD dispatcher evidence, NOT proof of individual AOV frame identity."""
    import omni.graph.core as og
    node = og.get_node_by_path('/Render/PostProcess/SDGPipeline/PostProcessDispatcher')
    if not node:
        return {'compute_count': None, 'simulation_time': None}
    numerator = node.get_attribute('outputs:referenceTimeNumerator').get()
    denominator = node.get_attribute('outputs:referenceTimeDenominator').get()
    return {'compute_count': int(node.get_compute_count()),
            'simulation_time': float(numerator / denominator) if denominator else None}


class ControlMetrics:
    def __init__(self, controller):
        from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
        self.controller = controller
        self.manager = SimulationManager
        self.steps = 0
        self.callback_times = []
        self.step_wall_times = []
        self.input_latencies = []
        self.pending = None
        self._started = None
        self._tokens = [
            self.manager.register_callback(self._before, event=SimulationEvent.PHYSICS_PRE_STEP, order=-100),
            self.manager.register_callback(self._after, event=SimulationEvent.PHYSICS_PRE_STEP, order=100),
            self.manager.register_callback(self._post, event=SimulationEvent.PHYSICS_POST_STEP, order=100)]
        self._last_keys = set()

    def note_input(self, source='human_keyboard'):
        if self.pending is None and self.controller.enabled:
            self.pending = (time.perf_counter(), tuple(self.controller.position), self.controller.yaw, source)

    def clear_measurements(self):
        self.steps = 0
        self.callback_times.clear()
        self.step_wall_times.clear()
        self.input_latencies.clear()
        self.pending = None
        self._started = None

    def _before(self, dt, context):
        self._started = time.perf_counter()
        keys = set(self.controller.keys.pressed)
        if keys - self._last_keys:
            self.note_input('physics_observed_key_state')
        self._last_keys = keys

    def _after(self, dt, context):
        self.steps += 1
        now = time.perf_counter()
        self.step_wall_times.append(now)
        if self._started is not None:
            self.callback_times.append(now - self._started)

    def _post(self, dt, context):
        if self.pending is None:
            return
        start, xyz, yaw, source = self.pending
        moved = max(abs(a-b) for a, b in zip(self.controller.position, xyz)) > 1e-6
        moved = moved or abs(self.controller.yaw - yaw) > 1e-6
        if moved:
            self.input_latencies.append({'seconds': time.perf_counter()-start, 'source': source})
            self.pending = None
        elif time.perf_counter() - start > 2:
            self.pending = None

    def report(self):
        return {'controller_step_count': self.steps,
                'controller_pre_callback_bracket_s': percentile_summary(self.callback_times),
                'controller_wall_step_interval_s': percentile_summary([
                    b-a for a, b in zip(self.step_wall_times, self.step_wall_times[1:])]),
                'input_to_pose_update_s': percentile_summary([x['seconds'] for x in self.input_latencies]),
                'input_latency_samples': self.input_latencies,
                'latency_scope': 'First physics observation of key state to pose update; excludes event queue, OS and display latency. Synthetic samples are separately labelled.',
                'human_input_latency_verified': False}

    def close(self):
        for token in self._tokens:
            self.manager.deregister_callback(token)


def configure_live_viewports(cleanup, enabled=False):
    from omni.kit.viewport.window import get_viewport_window_instances
    count = 0
    for window in get_viewport_window_instances(None):
        viewport = window.viewport_api
        if viewport:
            old = viewport.updates_enabled
            viewport.updates_enabled = enabled
            cleanup.add(lambda viewport=viewport, old=old: setattr(viewport, 'updates_enabled', old))
            count += 1
    return count
