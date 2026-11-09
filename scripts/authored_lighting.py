"""Apply the small, opt-in exposure block authored in a project USD layer."""
import hashlib
import math
from pathlib import Path


EXPOSURE_KEYS = ('/rtx/post/tonemap/filmIso', '/rtx/post/tonemap/cameraShutter',
                 '/rtx/post/tonemap/fNumber')
ALLOWED_KEYS = {*EXPOSURE_KEYS, '/rtx/post/tonemap/op', '/rtx/post/histogram/enabled'}


def validate_settings(data):
    if data is None:
        return {}
    if not isinstance(data, dict) or set(data) != ALLOWED_KEYS:
        raise ValueError('Authored lighting must specify exactly the supported exposure settings')
    if data['/rtx/post/histogram/enabled'] is not False:
        raise ValueError('Authored weak-light capture requires automatic exposure disabled')
    if type(data['/rtx/post/tonemap/op']) is not int or data['/rtx/post/tonemap/op'] != 6:
        raise ValueError('This authored lighting profile uses filmic tone mapping (op 6)')
    for key in EXPOSURE_KEYS:
        value = data[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid authored exposure: {key}')
    return dict(data)


def apply_authored_lighting(stage, settings):
    root = stage.GetRootLayer()
    profile = root.customLayerData.get('uavLighting')
    if profile is None:
        return None, None
    values = validate_settings(profile['settings'])
    dependencies = profile.get('dependencies', {})
    for relative, expected in dependencies.items():
        path = (Path(root.realPath).parent / relative).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Authored lighting dependency changed: {relative}')
    previous = {key: settings.get(key) for key in values}

    def restore():
        for key, value in previous.items():
            if value is None:
                settings.destroy_item(key)
            else:
                settings.set(key, value)

    try:
        actual = {}
        for key, value in values.items():
            settings.set(key, value)
            actual[key] = settings.get(key)
            matches = (actual[key] is value if type(value) is bool else
                       type(actual[key]) in (int, float) and
                       math.isclose(actual[key], value, rel_tol=1e-6, abs_tol=1e-6))
            if not matches:
                raise RuntimeError(f'Authored exposure did not apply: {key}={actual[key]!r}')
    except Exception:
        restore()
        raise
    return {'preset': profile['preset'], 'settings': actual,
            'dependency_sha256': dict(dependencies),
            'configuration_verified': True, 'measured_illuminance': False}, restore
