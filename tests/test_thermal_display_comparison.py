import sys
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/isaac-thermal-camera-simulation/src'))

from scripts.compare_thermal_display import (
    temperature_statistics, compare_display_frame, redisplay_replay,
    uniform_display_diagnostic,
)
from scripts.thermal_display import profile_variants


class DisplayComparisonTests(unittest.TestCase):
    def test_clipping_counts_invalid_separately(self):
        a = np.array([[273.15, 293.15, 323.15, np.nan]], dtype=np.float32)
        stats = temperature_statistics(a, 15., 40.)
        self.assertEqual(stats['finite_pixels'], 3)
        self.assertEqual(stats['nonfinite_pixels'], 1)
        self.assertAlmostEqual(stats['below_range_fraction'], 1 / 3)
        self.assertAlmostEqual(stats['above_range_fraction'], 1 / 3)

    def test_comparison_proves_raw_and_state_unchanged(self):
        from types import SimpleNamespace
        from irsim.config.loader import load_sensor_config
        from irsim.pipeline.frame import Outputs
        sensor = load_sensor_config(ROOT / 'third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml')
        apparent = np.full((4, 5), 293.15, dtype=np.float32)
        signal = np.arange(20, dtype=np.float32).reshape(4, 5) + 2000.
        outputs = Outputs(radiance=apparent.copy(), apparent_t=apparent,
                          dn16=signal.astype(np.uint16), display8=np.zeros((4, 5, 4), dtype=np.uint8),
                          signal_dn=signal, flux=apparent.copy())
        config = SimpleNamespace(flat_field=None, chain=None, sensor=sensor)
        state = {'frame': 7, 'history': np.array([1., 2.])}
        results, report = compare_display_frame(config, outputs, profile_variants(15., 40.), state=state)
        self.assertTrue(report['raw_arrays_byte_exact'])
        self.assertTrue(report['sensor_config_and_state_unchanged'])
        self.assertEqual(len(results), 4)
        self.assertEqual(report['variants']['fixed_range']['roi']['gray_std'], 0.)

    def test_redisplay_metadata_describes_new_display_and_preserves_source_provenance(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'source', Path(directory) / 'output'
            source.mkdir()
            output.mkdir()
            stem = 'frame_000000'
            arrays = {
                'apparent_t': np.full((2, 3), 293.15, dtype=np.float32),
                'radiance': np.ones((2, 3), dtype=np.float32),
                'dn16': np.full((2, 3), 1000, dtype=np.uint16),
                'rgb': np.zeros((2, 3, 4), dtype=np.uint8),
                'display8': np.full((2, 3, 4), 123, dtype=np.uint8),
            }
            for name, value in arrays.items():
                np.save(source / f'{stem}_{name}.npy', value)
            Image.fromarray(arrays['rgb']).save(source / f'{stem}_rgb.png')
            old_imaging = {
                'display_parameters': {'dde_gain': .35, 'agc': 'plateau_equalization'},
                'display_parameters_sha256': 'old', 'display_correction_source': 'factory_flat_field',
            }
            metadata = {'frame_id': 0, 'simulation_time_s': 1., 'thermal_imaging': old_imaging,
                        'strict_sync_verified': False}
            (source / f'{stem}.json').write_text(json.dumps(metadata), encoding='utf-8')
            (source / 'replay_metadata.json').write_text(json.dumps({'frames': 1}), encoding='utf-8')
            redisplay_replay(source, output, -30., 45.)
            from scripts.frame_metadata import load_frame_metadata
            actual = load_frame_metadata(output / f'{stem}.json')
            self.assertEqual(actual['thermal_imaging']['display_parameters']['mode'],
                             'fixed_apparent_temperature')
            self.assertEqual(actual['thermal_imaging']['display_parameters']['dde_gain'], 0.)
            self.assertEqual(actual['thermal_imaging']['display_input_quantity'], 'saved apparent_t (K)')
            self.assertFalse(actual['thermal_imaging']['project_display_flat_field_applied'])
            self.assertEqual(actual['source_display_provenance']['thermal_imaging'], old_imaging)
            self.assertFalse(actual['strict_sync_verified'])
            self.assertTrue(actual['redisplay_only'])
            for kind in ('dn16', 'radiance', 'apparent_t', 'rgb'):
                self.assertEqual((source / f'{stem}_{kind}.npy').read_bytes(),
                                 (output / f'{stem}_{kind}.npy').read_bytes())

    def test_uniform_diagnostic_uses_requested_display_range(self):
        from types import SimpleNamespace
        from irsim.config.loader import load_sensor_config
        sensor = load_sensor_config(ROOT / 'third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml')
        report = uniform_display_diagnostic(SimpleNamespace(sensor=sensor), frames=2,
                                            min_c=0., max_c=40.)
        self.assertEqual(report['fixed_range_c'], [0., 40.])
        self.assertGreater(report['variants']['fixed_range']['mean_spatial_gray_std'], 0.)
        self.assertLess(report['variants']['fixed_range']['mean_spatial_gray_std'], 1.)


if __name__ == '__main__':
    unittest.main()
