"""Snapshot actual experiment conditions; this does not change scene lighting."""
import hashlib


def file_record(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def snapshot_conditions(stage, config, width, height, *, thermal_width=None, thermal_height=None):
    import carb
    from pxr import UsdLux, UsdGeom
    from scripts.thermal_config import load_thermal_config
    settings = carb.settings.get_settings()
    lights = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI):
            attrs = {}
            for name in ('inputs:intensity', 'inputs:exposure', 'inputs:color',
                         'inputs:enableColorTemperature', 'inputs:colorTemperature',
                         'inputs:texture:file', 'visibility'):
                attr = prim.GetAttribute(name)
                value = attr.Get() if attr else None
                if value is not None:
                    attrs[name] = list(value) if name == 'inputs:color' else (
                        str(value) if name == 'inputs:texture:file' else value)
            lights.append({'path': str(prim.GetPath()), 'type': prim.GetTypeName(),
                           'attributes': attrs,
                           'effective_visibility': str(UsdGeom.Imageable(prim).ComputeVisibility()),
                           'world_matrix': [list(row) for row in
                               UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)]})
    thermal = load_thermal_config(config.sensors.thermal_config)
    binding = thermal.binding_for('/World/Target_01')
    target = thermal.target(thermal.target_for('/World/Target_01'))
    display_files = ({'display_config': file_record(config.sensors.display_config)}
                     if config.sensors.display_config is not None else {})
    return {
        'files': {name: file_record(path) for name, path in {
            'scene_config': config.source_path, 'usd': config.scene.stage,
            'thermal_config': config.sensors.thermal_config,
            'sensor_config': config.sensors.sensor_config,
            'thermal_environment_config': config.scene.thermal_scene}.items()} | display_files,
        'resolution': [width, height],
        'rgb_resolution': [width, height],
        'thermal_resolution': [width if thermal_width is None else thermal_width,
                               height if thermal_height is None else thermal_height],
        'flat_field_enabled': config.sensors.flat_field_enabled,
        'sensor_seed': config.sensors.sensor_seed,
        'ao_aov_enabled': False, 'default_occlusion': 1.0,
        'low_light': {'status': 'NOT_VERIFIED', 'runtime_lighting_override': False,
                      'lights': lights,
                      'renderer_settings': {name: settings.get(name) for name in (
                          '/rtx/rendermode', '/rtx/post/tonemap/op',
                          '/rtx/post/tonemap/filmIso', '/rtx/post/tonemap/cameraShutter',
                          '/rtx/post/tonemap/fNumber', '/rtx/post/histogram/enabled')},
                      'note': 'Actual authored lighting/settings, not measured illuminance or a low-light certification.'},
        'spot': {'prim_path': '/World/Target_01', 'target': target.name,
                 'mode': target.mode,
                 'prescribed_temperature_c': None if target.temperature_k is None else target.temperature_k - 273.15,
                 'infrared_material': None if binding is None else binding.material,
                 'meaning': 'Prescribed synthetic temperature, not a measured robot temperature'},
        'data_usage': 'pilot/development data',
        'hash_scope': 'Root USD and config files; external asset contents are not all pinned.'}
