import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party' / 'isaac-thermal-camera-simulation' / 'src'))

from scripts.thermal_display import (
    DisplayProfileError,
    fixed_temperature_display,
    load_display_profile,
    profile_variants,
)
from scripts.thermal_imaging import display_from_outputs, imaging_metadata
from types import SimpleNamespace


class ThermalDisplayTests(unittest.TestCase):
    def test_fixed_temperature_display_is_monotonic_and_clips(self):
        apparent_t = np.array([[278.15, 293.15, 313.15]], dtype=np.float32)
        display = fixed_temperature_display(apparent_t, 15.0, 40.0)
        self.assertEqual(display.shape, (1, 3, 4))
        self.assertEqual(display.dtype, np.uint8)
        self.assertEqual(display[0, 0, 0], 0)
        self.assertEqual(display[0, 1, 0], 51)
        self.assertEqual(display[0, 2, 0], 255)
        self.assertTrue(np.all(display[..., 3] == 255))

    def test_profile_loader_rejects_reversed_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'display.yaml'
            path.write_text(
                'schema_version: 1\nmode: fixed_apparent_temperature\n'
                'temperature_min_c: 40\ntemperature_max_c: 15\n',
                encoding='utf-8',
            )
            with self.assertRaises(DisplayProfileError):
                load_display_profile(path)

    def test_variants_have_one_fixed_range_and_preserve_raw_inputs(self):
        profiles = profile_variants(15.0, 40.0)
        self.assertEqual([p.name for p in profiles],
                         ['current', 'no_detail', 'linear', 'fixed_range'])
        self.assertEqual(profiles[-1].mode, 'fixed_apparent_temperature')
        raw = np.array([[293.15]], dtype=np.float32)
        before = raw.tobytes()
        fixed_temperature_display(raw, 15.0, 40.0)
        self.assertEqual(raw.tobytes(), before)

    def test_profile_metadata_is_json_serializable(self):
        profile = profile_variants(15.0, 40.0)[-1]
        json.dumps(profile.metadata())

    def test_fixed_profile_changes_display_only(self):
        from irsim.pipeline.frame import Outputs
        raw = np.array([[293.15]], dtype=np.float32)
        outputs = Outputs(
            apparent_t=raw.copy(), display8=np.zeros((1, 1, 4), dtype=np.uint8),
            signal_dn=np.array([[100.]], dtype=np.float32), isp_hash='old',
            dn16=np.array([[100]], dtype=np.uint16), radiance=np.ones_like(raw),
            flux=np.ones_like(raw),
        )
        config = SimpleNamespace(flat_field=None)
        before = raw.tobytes()
        result = display_from_outputs(config, outputs, profile_variants(15.0, 40.0)[-1])
        self.assertEqual(raw.tobytes(), before)
        self.assertEqual(result.display8[0, 0, 0], 51)
        self.assertEqual(result.display8.shape, outputs.display8.shape)
        self.assertEqual(result.signal_dn[0, 0], 100.)
        for name in ('apparent_t', 'signal_dn', 'radiance', 'dn16', 'flux'):
            self.assertIs(getattr(result, name), getattr(outputs, name))

    def test_invalid_temperature_remains_visible_warning(self):
        display = fixed_temperature_display(np.array([[np.nan]], dtype=np.float32), 15., 40.)
        self.assertEqual(display[0, 0].tolist(), [255, 0, 255, 255])

    def test_same_temperature_has_same_brightness_when_scene_range_changes(self):
        first = fixed_temperature_display(np.array([[293.15, 294.15]], dtype=np.float32), 15., 40.)
        second = fixed_temperature_display(np.array([[293.15, 353.15]], dtype=np.float32), 15., 40.)
        np.testing.assert_array_equal(first[0, 0], second[0, 0])

    def test_fixed_temperature_display_rejects_non_image(self):
        with self.assertRaises(DisplayProfileError):
            fixed_temperature_display(np.ones((2, 2, 3)), 15., 40.)

    def test_profile_loader_rejects_duplicate_keys_and_bool_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'display.yaml'
            for content in (
                'schema_version: 1\nmode: vendor\nmode: linear\n',
                'schema_version: true\nmode: vendor\n',
            ):
                path.write_text(content, encoding='utf-8')
                with self.assertRaises(DisplayProfileError):
                    load_display_profile(path)

    def test_no_detail_profile_preserves_physical_planes_and_sensor_isp(self):
        from irsim.config.loader import load_sensor_config
        from irsim.pipeline.frame import Outputs
        sensor = load_sensor_config(ROOT / 'third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml')
        signal = np.arange(12, dtype=np.float32).reshape(3, 4) + 1000
        outputs = Outputs(radiance=signal.copy(), apparent_t=signal.copy(),
                          dn16=signal.astype(np.uint16), display8=np.zeros((3, 4, 4), dtype=np.uint8),
                          signal_dn=signal, flux=signal.copy())
        config = SimpleNamespace(flat_field=None, chain=None, sensor=sensor)
        before = {name: getattr(outputs, name).tobytes() for name in
                  ('radiance', 'apparent_t', 'dn16', 'signal_dn', 'flux')}
        result = display_from_outputs(config, outputs, profile_variants(15., 40.)[1])
        self.assertEqual(sensor.sensor.isp.dde_gain, .35)
        for name, payload in before.items():
            self.assertEqual(getattr(result, name).tobytes(), payload)

    def test_fixed_display_metadata_does_not_claim_detector_flat_field_input(self):
        from irsim.config.loader import load_sensor_config
        sensor = load_sensor_config(ROOT / 'third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml')
        flat = SimpleNamespace(gain=np.ones((2, 3)), offset=np.zeros((2, 3)), pedestal=0.)
        config = SimpleNamespace(flat_field=flat, chain=None, sensor=sensor,
                                 t_housing_cal_k=293.15, sensor_seed=0)
        state = SimpleNamespace(housing_temp_k=293.15)
        metadata = imaging_metadata(config, state, profile_variants(-30., 45.)[-1])
        self.assertTrue(metadata['flat_field_enabled'])
        self.assertFalse(metadata['project_display_flat_field_applied'])
        self.assertEqual(metadata['display_correction_source'], 'none_fixed_apparent_temperature')
        self.assertEqual(metadata['display_input_quantity'], 'apparent_t (K)')
        self.assertEqual(metadata['vendor_isp_parameters']['dde_gain'], .35)


if __name__ == '__main__':
    unittest.main()
