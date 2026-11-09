"""Exercise real irsim solvers/material bindings without starting Kit."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
THERMAL_ROOT = ROOT / 'third_party/isaac-thermal-camera-simulation'
sys.path.insert(0, str(THERMAL_ROOT / 'src'))

from scripts.thermal_config import ThermalConfigError, ThermalTarget, load_thermal_config
from scripts.thermal_scene_config import apply_thermal_config


class ThermalSceneConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from irsim.config.loader import load_sensor_config
        from irsim.radiometry.lut_files import load_band_lut_for_config
        from irsim.scene import Scene
        from irsim.materials.library import MaterialLibrary
        from irsim.materials.table import MaterialTable
        sensor = load_sensor_config(THERMAL_ROOT / 'configs/sensors/flir_boson_640_lwir.yaml')
        band = sensor.sensor.band.band_id
        lut = load_band_lut_for_config(sensor, THERMAL_ROOT / 'data/lut')
        cls.scene = Scene.from_file(THERMAL_ROOT / 'configs/scenes/quad_flight_clear_noon.yaml',
                                    {band: lut})
        cls.material_names = MaterialTable.from_library(MaterialLibrary.load(), band).names

    def setUp(self):
        self.config = load_thermal_config()
        self.paths = ['/World/Cube', '/World/GroundPlane/CollisionMesh',
                      '/World/Crazyflie/body/visual/mesh', '/World/Unknown']

    def apply(self, config=None):
        return apply_thermal_config(self.scene, config or self.config, self.paths, self.material_names)

    def test_applies_per_prim_temperatures_materials_and_fallback_without_mutating_scene(self):
        scene, mapping, resolutions, report = self.apply()
        self.assertEqual(mapping[self.paths[0]], 'cube')
        self.assertEqual(mapping[self.paths[1]], 'ground')
        self.assertEqual(mapping[self.paths[2]], 'drone_body')
        self.assertEqual(mapping[self.paths[3]], 'ambient')
        self.assertNotIn('cube', self.scene.targets)
        self.assertAlmostEqual(scene.targets['cube'].temperature(), 318.15)
        self.assertAlmostEqual(scene.targets['ground'].temperature(), 293.15)
        self.assertAlmostEqual(scene.targets['drone_body'].temperature(), 308.15)
        for _ in range(2):
            scene.targets['cube'].advance(scene.t0_s, 30.0)
            self.assertAlmostEqual(scene.targets['cube'].temperature(), 318.15)
        self.assertEqual(resolutions[0].material, 'concrete')
        self.assertEqual(self.material_names[resolutions[0].material_id], 'concrete')
        self.assertTrue(all(item.mapped for item in resolutions))
        self.assertEqual(report['fallback_paths'], ['/World/Unknown'])
        self.assertEqual(report['unmatched_bindings'], [])
        self.assertEqual(report['binding_counts'], {'drone_body': 1, 'ground': 1, 'cube': 1})

    def test_ambient_and_offset_start_at_scene_time_and_follow_weather(self):
        scene, _, _, _ = self.apply()
        for relative_time in (0.0, 30.0, 90.0):
            air = float(scene.weather.at(scene.t0_s + relative_time).t_air_k)
            for name, difference in [('ambient', 0.0), ('warm_relative', 10.0)]:
                actual = scene.targets[name].advance(scene.t0_s + relative_time, 0.0)
                self.assertAlmostEqual(actual, air + difference, places=3)

    def test_reports_paths_that_match_no_geometry(self):
        config = replace(self.config, bindings=(replace(self.config.bindings[0],
                         prim_path='/World/Misspelled'),))
        _, _, _, report = self.apply(config)
        self.assertEqual(report['unmatched_bindings'], ['drone_body'])
        self.assertEqual(len(report['fallback_paths']), 4)

    def test_unknown_material_is_an_error_even_for_an_unmatched_binding(self):
        config = replace(self.config, bindings=(replace(self.config.bindings[0],
                         prim_path='/World/Missing', material='concret_typo'),))
        with self.assertRaisesRegex(ThermalConfigError, 'concret_typo'):
            self.apply(config)

    def test_does_not_silently_replace_existing_scene_targets(self):
        config = replace(self.config, targets=(*self.config.targets, ThermalTarget('airframe', 'ambient')))
        with self.assertRaisesRegex(ThermalConfigError, 'airframe'):
            self.apply(config)

    def test_changed_configuration_changes_solver_and_leaves_original_configuration_alone(self):
        config = replace(self.config, targets=tuple(
            replace(item, temperature_k=333.15) if item.name == 'cube' else item
            for item in self.config.targets))
        original, _, _, _ = self.apply()
        changed, _, _, _ = self.apply(config)
        self.assertAlmostEqual(changed.targets['cube'].temperature(), 333.15)
        self.assertAlmostEqual(original.targets['cube'].temperature(), 318.15)


if __name__ == '__main__':
    unittest.main()
