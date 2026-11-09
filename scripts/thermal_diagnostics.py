"""Opt-in, restored timing wrappers. Never installed by production capture."""
from contextlib import contextmanager
from functools import wraps
import importlib
import time

import numpy as np


def statistics(values):
    values = list(values)
    return {'samples': len(values), 'mean': float(np.mean(values)) if values else None,
            'p95': float(np.percentile(values, 95)) if values else None,
            'p99': float(np.percentile(values, 99)) if values else None,
            'max': float(max(values)) if values else None}


class StageTimings:
    def __init__(self, *, clock=time.perf_counter, thread_clock=time.thread_time):
        self.clock, self.thread_clock = clock, thread_clock
        self.values = {}
        self._stack = []

    def _wrapped(self, function, name):
        @wraps(function)
        def timed(*args, **kwargs):
            wall, cpu = self.clock(), self.thread_clock()
            entry = [0., 0.]
            self._stack.append(entry)
            try:
                return function(*args, **kwargs)
            finally:
                dw, dc = self.clock()-wall, self.thread_clock()-cpu
                self._stack.pop()
                if self._stack:
                    self._stack[-1][0] += dw
                    self._stack[-1][1] += dc
                record = self.values.setdefault(name, {
                    'calls': 0, 'inclusive_wall_s': 0., 'exclusive_wall_s': 0.,
                    'inclusive_thread_cpu_s': 0., 'exclusive_thread_cpu_s': 0.})
                record['calls'] += 1
                record['inclusive_wall_s'] += dw
                record['exclusive_wall_s'] += max(0., dw-entry[0])
                record['inclusive_thread_cpu_s'] += dc
                record['exclusive_thread_cpu_s'] += max(0., dc-entry[1])
        return timed

    @contextmanager
    def wrap(self, specs):
        originals = []
        try:
            for target, attribute, group in specs:
                original = getattr(target, attribute)
                originals.append((target, attribute, original))
                setattr(target, attribute, self._wrapped(original, group))
            yield self
        finally:
            for target, attribute, original in reversed(originals):
                setattr(target, attribute, original)

    def reset(self):
        if self._stack:
            raise RuntimeError('Cannot reset active timings')
        self.values = {}


def thermal_specs(camera):
    """Actual call sites in the installed vendor code, not guessed profiler labels."""
    ir = importlib.import_module('irsim_isaac.pipeline.ir_camera')
    frame = importlib.import_module('irsim.pipeline.frame')
    motion = importlib.import_module('irsim_isaac.pipeline.motion_isaac')
    display = importlib.import_module('scripts.thermal_imaging')
    preview_module = importlib.import_module('scripts.start_crazyflie_dual_camera_view')
    specs = [(camera, 'get_outputs', 'thermal_total'),
             (camera, 'planes', 'assembly_other'),
             (camera._reader, 'read', 'aov_read'),
             (camera, '_motion_plane', 'motion_geometry'),
             (camera, '_moving_leaf_paths', 'mapping_lookup'),
             (camera.bridge, 'advance_to', 'temperature_advance'),
             (camera.bridge, '_advance_one_tick', 'temperature_target_tick'),
             (camera.bridge, '_advance_field_to', 'temperature_field_advance'),
             (camera.bridge, 'temperature_plane', 'temperature_plane'),
             (ir, 'run_frame', 'numerical_other'),
             (preview_module, 'display_from_outputs', 'project_display')]
    specs.extend(mapping_specs())
    for name in ('ray_directions', 'geometry_planes', 'elevation_from_rays',
                 'azimuth_from_rays', 'to_gbuffer'):
        specs.append((ir, name, 'motion_geometry'))
    for name in ('sample', 'transform_stacks'):
        specs.append((camera._motion, name, 'motion_geometry'))
    specs.append((motion, 'image_plane_motion', 'motion_projection'))
    for name in ('stage_illumination', 'band_radiance', 'clip_to_gain_ceiling', 'gain_ceiling_radiance'):
        specs.append((frame, name, 'radiance'))
    for name in ('apply_atmosphere_gbuffer', 'apply_layered_gbuffer'):
        specs.append((frame, name, 'atmosphere'))
    for name in ('apply_optics', '_resolve_focus', 'layered_defocus'):
        specs.append((frame, name, 'optics'))
    for name in ('_detector_signal', 'shutter_flux'):
        specs.append((frame, name, 'detector_noise_history'))
    for name in ('quantise', '_scene_radiance_from_signal', 'apparent_temperature'):
        specs.append((frame, name, 'radiometric_adc'))
    specs.append((frame, 'run_display_branch', 'vendor_display'))
    # The project display function imports this call site at invocation time.
    isp = importlib.import_module('irsim.isp.display')
    specs.append((isp, 'run_display_branch', 'project_display_isp'))
    gbuffer = importlib.import_module('irsim_isaac.pipeline.gbuffer_isaac')
    specs.append((gbuffer, '_to_numpy', 'aov_conversion'))
    for annotator in camera._reader._annotators.values():
        specs.append((annotator, 'get_data', 'aov_gpu_read'))
    if camera.config.chain is not None:
        for name in ('begin_frame', 'finish_frame', 'record_shutter'):
            specs.append((camera.config.chain, name, 'detector_noise_history'))
    return specs


def mapping_specs(modules=None):
    if modules is None:
        modules = [importlib.import_module('irsim_isaac.pipeline.'+name) for name in
                   ('ir_camera', 'aerial_bridge', 'motion_isaac', 'point_bridge', 'mesh_bridge', 'material_ids')]
    specs = []
    for module in modules:
        for name, group in (('labels_from_payload', 'label_normalization'),
                            ('labels_to_paths', 'label_normalization'),
                            ('material_id_plane', 'mapping_lookup')):
            if hasattr(module, name):
                specs.append((module, name, group))
    return specs


@contextmanager
def profile_material_mapping_scope(timer, *, runtime=None, modules=None):
    """Install mapping timers after the project runtime has replaced vendor hooks.

    The outer timers cover cache-off. This inner scope covers cache-on and is
    restored before the runtime restores the vendor hooks. Inclusive groups
    overlap, so consumers must not add them.
    """
    runtime = runtime or importlib.import_module('scripts.thermal_runtime')
    original = runtime.cached_material_mapping
    @contextmanager
    def profiled(*args, **kwargs):
        with original(*args, **kwargs):
            with timer.wrap(mapping_specs(modules)):
                yield
    runtime.cached_material_mapping = profiled
    try:
        yield
    finally:
        runtime.cached_material_mapping = original


@contextmanager
def temperature_counts(camera):
    """Exact untimed call counters, identical overhead in both throughput conditions."""
    counts, originals = {}, []
    specs = [(camera.bridge, 'advance_to', 'bridge_advance_requests'),
             (camera.bridge, '_advance_one_tick', 'target_solver_ticks'),
             (camera.bridge, '_advance_field_to', 'field_advance_requests'),
             (camera.bridge, 'temperature_plane', 'temperature_planes')]
    if camera.scene.thermal is not None:
        specs.append((camera.scene.thermal, 'advance_to', 'field_solver_advance_requests'))
    for name in ('pointwise', 'mesh_pointwise'):
        target = getattr(camera, name, None)
        if target is not None:
            specs.append((target, 'advance_to', name+'_advance_requests'))
    def counted(function, name):
        @wraps(function)
        def call(*args, **kwargs):
            counts[name] = counts.get(name, 0)+1
            return function(*args, **kwargs)
        return call
    try:
        for target, name, group in specs:
            function = getattr(target, name)
            originals.append((target, name, function))
            counts[group] = 0
            setattr(target, name, counted(function, group))
        yield counts
    finally:
        for target, name, function in reversed(originals):
            setattr(target, name, function)


def summarize_segments(samples):
    groups = sorted(set().union(*(s for s in samples))) if samples else []
    report = {}
    for group in groups:
        counts = [int(s.get(group, {}).get('calls', 0)) for s in samples]
        report[group] = {key: statistics(s.get(group, {}).get(key, 0.) for s in samples)
                         for key in ('inclusive_wall_s', 'exclusive_wall_s',
                                     'inclusive_thread_cpu_s', 'exclusive_thread_cpu_s')}
        report[group]['calls'] = {'total': sum(counts), 'per_frame': counts}
    return report
