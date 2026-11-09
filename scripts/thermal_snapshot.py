"""Owned numerical snapshots. Pickle files are local diagnostics, never dataset inputs."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import math
import pathlib
import pickle

import numpy as np


@contextmanager
def anonymous_session_layer(stage):
    from pxr import Sdf, Usd
    original_target = stage.GetEditTarget()
    session = stage.GetSessionLayer()
    original_sublayers = list(session.subLayerPaths)
    root_contents = stage.GetRootLayer().ExportToString()
    layer = Sdf.Layer.CreateAnonymous('thermal-baseline-diagnostics.usda')
    session.subLayerPaths = [layer.identifier] + original_sublayers
    stage.SetEditTarget(Usd.EditTarget(layer))
    try:
        yield layer
    finally:
        stage.SetEditTarget(original_target)
        session.subLayerPaths = original_sublayers
        if stage.GetRootLayer().ExportToString() != root_contents:
            raise RuntimeError('Diagnostic snapshots changed the root USD layer in memory')


def owned_copy(value):
    return deepcopy(value)


def compare_arrays(left, right):
    if left is None or right is None:
        return {'exact': left is None and right is None, 'max_abs_error': None}
    a, b = np.asarray(left), np.asarray(right)
    layout = a.shape == b.shape and a.dtype == b.dtype
    exact = layout and a.tobytes() == b.tobytes()
    error = None
    changed = None
    if a.shape == b.shape and np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        with np.errstate(invalid='ignore'):
            delta = np.abs(a.astype(np.float64)-b.astype(np.float64))
        finite = delta[np.isfinite(delta)]
        error = float(finite.max()) if finite.size else None
        changed = int(np.count_nonzero(~(a == b) & ~(np.isnan(a) & np.isnan(b))))
    return {'exact': bool(exact), 'dtype_left': str(a.dtype), 'dtype_right': str(b.dtype),
            'shape_left': list(a.shape), 'shape_right': list(b.shape),
            'changed_values': changed, 'max_abs_error': error,
            'tolerance': 0.0}


def state_tree(value, seen=None):
    """Stable state audit: no memory addresses, includes RNG and private lag buffers."""
    seen = {} if seen is None else seen
    if isinstance(value, np.ndarray):
        return {'array_shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return state_tree(value.item(), seen)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, np.random.Generator):
        return state_tree(value.bit_generator.state, seen)
    if isinstance(value, (pathlib.Path, bytes)):
        return str(value)
    ident = id(value)
    if ident in seen:
        return {'reference': seen[ident]}
    seen[ident] = len(seen)
    if isinstance(value, dict):
        return {str(k): state_tree(v, seen) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    if isinstance(value, (tuple, list)):
        return [state_tree(v, seen) for v in value]
    if is_dataclass(value):
        attributes = {f.name: getattr(value, f.name) for f in fields(value)}
    elif hasattr(value, '__dict__'):
        attributes = vars(value)
    else:
        return {'type': f'{type(value).__module__}.{type(value).__qualname__}', 'value': str(value)}
    return {'type': f'{type(value).__module__}.{type(value).__qualname__}',
            'attributes': {k: state_tree(v, seen) for k, v in sorted(attributes.items())}}


def state_digest(value):
    return hashlib.sha256(json.dumps(state_tree(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def sensor_history(config, state):
    return {'pipeline': state_tree(state), 'sensor_chain': state_tree(config.chain),
            'noise': state_tree(config.noise)}


def camera_history(camera):
    from scripts.preview_sky import PreviewSky
    memo = {}
    if isinstance(camera.bridge.sky, PreviewSky):
        memo.update(preview_sky_memo(camera.bridge.sky, camera.scene.t0_s + camera._t_rel_s))
    return {'sensor': sensor_history(camera.config, camera.state),
            'thermal_bridge': state_tree(deepcopy(camera.bridge, memo)),
            'motion': state_tree(camera._motion),
            'thermal_elapsed_s': camera._t_rel_s,
            'last_frame_time_s': camera._last_frame_t_s}


def assembly_checkpoint(camera, raw):
    """Copy the actual assembly inputs before it advances temperature/history.

    Renderer/Kit handles are deliberately excluded. This is diagnostic-only;
    neither this copy nor transform reads run in the normal preview.
    """
    from irsim_isaac.pipeline.material_ids import labels_from_payload
    from irsim_isaac.pipeline.motion_isaac import MotionTracker
    from scripts.preview_sky import PreviewSky
    from irsim_isaac.pipeline.gbuffer_isaac import RawAovs
    raw_copy = RawAovs(**{f.name: owned_copy(getattr(raw, f.name))
                         for f in fields(RawAovs) if f.name != 'device_handles'},
        device_handles={'instance': {'info': {'idToLabels':
            owned_copy(labels_from_payload(raw.device_handles.get('instance')))}}})
    labels = labels_from_payload(raw_copy.device_handles['instance'])
    wanted = [camera.camera_path, *camera.moving_prim_paths]
    wanted += [p for p in labels.values() if any(
        p == r or p.startswith(r.rstrip('/')+'/') for r in camera.moving_prim_paths)]
    for bridge in (camera.pointwise, camera.mesh_pointwise):
        if bridge is not None:
            wanted += list(bridge.local_frames)
    read = camera._motion_read or MotionTracker._usd_reader(camera._stage)
    transforms = {p: np.array(read(p), copy=True) for p in dict.fromkeys(wanted)}
    if camera.rotor_mounts:
        raise ValueError('Assembly checkpoint rotor mounts require USD; not covered by this tool')
    ignored = {'_reader', '_render_product', '_stage', '_motion_read',
               '_material_mapping_cache', '_material_cache_owner', '_material_cache_context'}
    values = {k: v for k, v in vars(camera).items() if k not in ignored}
    memo = {}
    for sky in (camera.config.sky, camera.bridge.sky):
        if isinstance(sky, PreviewSky) and id(sky) not in memo:
            memo.update(preview_sky_memo(sky, camera.scene.t0_s + camera._t_rel_s))
    return {'attributes': deepcopy(values, memo), 'raw_aovs': raw_copy,
            'transforms': transforms, 'history_before': camera_history(camera)}


def run_assembly_checkpoint(checkpoint, *, cache_enabled, cache=None):
    """Run the actual vendor assembly, numeric and display paths on owned inputs."""
    from types import SimpleNamespace
    from scripts.thermal_runtime import camera_class
    from scripts.thermal_imaging import display_from_outputs
    cls = camera_class(material_cache_enabled=cache_enabled)
    camera = cls.__new__(cls)
    camera.__dict__.update(owned_copy(checkpoint['attributes']))
    camera._stage = None
    transforms = owned_copy(checkpoint['transforms'])
    camera._motion_read = lambda path: transforms[path]
    raw = owned_copy(checkpoint['raw_aovs'])
    camera._reader = SimpleNamespace(read=lambda: raw)
    camera._render_product = None
    camera.mapping_cache_enabled = cache_enabled
    camera._material_mapping_cache = cache
    outputs = display_from_outputs(camera.config, camera.get_outputs(step=False))
    return {'planes': owned_copy(camera.last_frame.planes),
            'outputs': owned_copy(outputs), 'history': camera_history(camera)}


def save_snapshot(path, snapshot):
    path = pathlib.Path(path)
    payload = pickle.dumps(snapshot, protocol=5)
    path.write_bytes(payload)
    path.with_suffix('.manifest.json').write_text(json.dumps({
        'format': 'trusted-local-numerical-snapshot-v1',
        'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload),
        'scope': 'Post-assembly numerical chain; not a sensor synchronization certificate',
        'pickle_warning': 'Only deserialize self-generated trusted local diagnostics.'}, indent=2), encoding='utf-8')


def load_snapshot(path):
    path = pathlib.Path(path)
    manifest = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest['sha256']:
        raise ValueError('Snapshot hash mismatch')
    # Caller must explicitly opt into trusted local snapshots. A hash is integrity, not trust.
    return pickle.loads(payload)


class FrozenPreviewSky:
    """Same captured 30-s interpolation tables, with no executor or future objects."""
    def __init__(self, source, t_s):
        self.source = owned_copy(source.source)
        self.start_s, self.interval_s = source.start_s, source.interval_s
        self.elevation, self.tilt = owned_copy((source.elevation, source.tilt))
        self.index = math.floor((float(t_s)-self.start_s)/self.interval_s)
        self.tables = {i: owned_copy(source._table(i)) for i in (self.index, self.index+1)}

    def __getattr__(self, name):
        source = self.__dict__.get('source')
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    def _sample(self, t_s, angles, channel):
        fraction = (float(t_s)-self.start_s)/self.interval_s
        index, weight = math.floor(fraction), fraction-math.floor(fraction)
        if index not in self.tables or index+1 not in self.tables:
            raise ValueError('Snapshot sky tables do not cover requested time')
        values = self.tables[index][channel]*(1-weight)+self.tables[index+1][channel]*weight
        return np.interp(angles, self.elevation if channel == 0 else self.tilt, values)

    def radiance(self, t_s, angles):
        return self._sample(t_s, angles, 0)

    def apparent_temperature_k(self, t_s, elevation_rad):
        return self.source.lut.apparent_temperature(self.radiance(t_s, elevation_rad),
                                                     self.source.quantity)

    def effective_radiance_from_sky_view(self, t_s, sky_view):
        tilt = np.arccos(np.clip(2*np.asarray(sky_view)-1, -1., 1.))
        return self._sample(t_s, tilt, 1)


def preview_sky_memo(source, t_s):
    frozen = FrozenPreviewSky(source, t_s)
    memo = {id(source): frozen, id(source.source): frozen.source}
    # Preserve shared weather/atmosphere/LUT objects across camera, sky and scene.
    # Otherwise numeric memoization mutates a different atmosphere after copying.
    for name, value in vars(source.source).items():
        if hasattr(frozen.source, name):
            memo[id(value)] = getattr(frozen.source, name)
    return memo


def copy_config(config, t_s):
    from scripts.preview_sky import PreviewSky
    memo = {}
    if isinstance(config.sky, PreviewSky):
        memo.update(preview_sky_memo(config.sky, t_s))
    return deepcopy(config, memo)


def run_numerical(snapshot, *, config=None, state=None, frame_function=None):
    from irsim.pipeline.frame import run_frame
    from scripts.thermal_imaging import display_from_outputs
    config = owned_copy(snapshot['config']) if config is None else config
    state = owned_copy(snapshot['state_before']) if state is None else state
    state.t_s = snapshot['state_before'].t_s
    function = run_frame if frame_function is None else frame_function
    result = function(owned_copy(snapshot['planes']), config, state,
                      owned_copy(snapshot['point_targets']), owned_copy(snapshot['rotor_veils']))
    result = display_from_outputs(config, result)
    return result, config, state


def output_comparison(left, right):
    arrays = {name: compare_arrays(getattr(left, name), getattr(right, name))
              for name in ('radiance', 'apparent_t', 'dn16', 'display8', 'signal_dn', 'flux')}
    metadata_equal = state_digest((left.isp_hash, left.report)) == state_digest((right.isp_hash, right.report))
    return {'exact': all(a['exact'] for a in arrays.values()) and metadata_equal,
            'arrays': arrays, 'output_metadata_exact': metadata_equal}


@contextmanager
def capture_numerical_boundary(camera, metadata):
    """Intercept exactly one call to the existing serial numeric kernel, restoring it."""
    from irsim_isaac.pipeline import ir_camera
    original = ir_camera.run_frame
    captured = {}
    def record(planes, config, state, point_targets=(), rotor_veils=(), **kwargs):
        if captured:
            raise RuntimeError('Expected exactly one numerical frame per snapshot')
        captured.update(schema_version=1, metadata=owned_copy(metadata),
                        planes=owned_copy(planes), config=copy_config(config, state.t_s),
                        state_before=owned_copy(state), point_targets=owned_copy(point_targets),
                        rotor_veils=owned_copy(rotor_veils))
        captured['history_before'] = sensor_history(config, state)
        result = original(planes, config, state, point_targets, rotor_veils, **kwargs)
        captured['raw_outputs'] = owned_copy(result)
        captured['history_after'] = sensor_history(config, state)
        return result
    ir_camera.run_frame = record
    try:
        yield captured
    finally:
        ir_camera.run_frame = original
