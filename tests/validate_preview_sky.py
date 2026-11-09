"""Compare the interactive sky approximation to this checkout's radiometric reference."""
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party/isaac-thermal-camera-simulation'
sys.path[:0] = [str(ROOT), str(VENDOR / 'src')]


def main():
    from irsim.atmosphere.skylight import skylight_for_sensor
    from irsim.config.loader import load_sensor_config
    from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
    from irsim.scene import Scene
    from scripts.preview_sky import PreviewSky
    sensor = load_sensor_config(VENDOR / 'configs/sensors/flir_boson_640_lwir.yaml')
    band = sensor.sensor.band.band_id
    lut = load_band_lut_for_config(sensor, VENDOR / 'data/lut')
    scene = Scene.from_file(VENDOR / 'configs/scenes/quad_flight_clear_noon.yaml', {band: lut},
                            responses={band: load_band_response_for_config(sensor, VENDOR/'data')},
                            quantity=sensor.sensor.quantity,
                            skylights={band: skylight_for_sensor(sensor, sensor.sensor.quantity)})
    reference = scene.sky_models[band]
    # Include horizon, oblique angles and zenith, and both sides of a cache rollover.
    elevations = np.radians(np.linspace(0., 90., 361))
    views = np.linspace(0., 1., 181)
    rows = []
    with PreviewSky(reference, scene.t0_s) as sky:
        for offset in (0., 7.5, 15., 29.99, 30., 45., 61.):
            t = scene.t0_s + offset
            start = time.perf_counter()
            actual = sky.apparent_temperature_k(t, elevations)
            reflected = sky.effective_radiance_from_sky_view(t, views)
            elapsed = time.perf_counter()-start
            expected = reference.apparent_temperature_k(t, elevations)
            expected_reflected = reference.effective_radiance_from_sky_view(t, views)
            error = float(np.max(np.abs(actual-expected)))
            relative = float(np.max(np.abs(reflected-expected_reflected)/expected_reflected))
            assert error < .01, (offset, error)
            assert relative < 1e-4, (offset, relative)
            rows.append({'offset_s': offset, 'max_sky_temperature_error_K': error,
                         'max_reflected_relative_error': relative, 'lookup_s': elapsed})
    report = {'result': 'PASS', 'interval_s': 30., 'samples': rows,
              'scope': 'default LWIR scene, first 61 simulation seconds; not arbitrary weather'}
    target = ROOT / 'validation/preview-sky-accuracy.json'
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
