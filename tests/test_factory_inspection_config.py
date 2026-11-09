"""Contract tests for the isolated factory experiment, without starting Kit."""
from pathlib import Path
import unittest
import math
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.scene_config import load_scene_config
from scripts.thermal_config import load_thermal_config
from scripts.thermal_display import load_display_profile


class FactoryInspectionConfigurationTests(unittest.TestCase):
    def test_scene_is_loadable_by_existing_runtime_and_uses_separate_assets(self):
        config = load_scene_config(ROOT/'configs/factory_inspection_scene.yaml')
        self.assertEqual(config.scene.stage, ROOT/'factory/factory_inspection.usda')
        self.assertEqual(config.drone.prim_path, '/World/Crazyflie')
        self.assertEqual(config.drone.body_path, '/World/Crazyflie/body')
        self.assertEqual(config.drone.camera_path, '/World/Crazyflie/body/Camera_mount/Camera')
        self.assertTrue(config.sensors.align_thermal_to_rgb)
        self.assertEqual((config.sensors.width, config.sensors.height), (160, 128))
        self.assertEqual(config.sensors.sensor_seed, 0)

    def test_route_and_task_do_not_expose_truth_in_the_pilot_instruction(self):
        config = load_scene_config(ROOT/'configs/factory_inspection_scene.yaml')
        task = config.task
        data = task['environment_evidence']['factory_inspection']
        waypoints = data['reference_waypoints']
        self.assertEqual(waypoints, data['reference_waypoints_m'])
        length = sum(math.dist(a, b) for a, b in zip(waypoints, waypoints[1:]))
        self.assertGreaterEqual(length, 180)
        self.assertLessEqual(length, 220)
        self.assertAlmostEqual(length, data['reference_path_length_m'], places=5)
        self.assertEqual(waypoints[0], task['start_position'])
        self.assertIn(data['observation_position'], waypoints)
        self.assertEqual(data['inspection_requirements']['hold_duration_s'], 6)
        self.assertEqual(data['inspection_requirements']['minimum_equipment_distance_m'], 5)
        self.assertNotIn('78', task['instruction'])
        self.assertNotIn('/World', task['instruction'])
        self.assertNotIn('material_1_29_gc270', task['target_prim_path'])

    def test_only_a_specific_real_equipment_patch_is_hot(self):
        config = load_scene_config(ROOT/'configs/factory_inspection_scene.yaml')
        data = config.task['environment_evidence']['factory_inspection']
        thermal = load_thermal_config(config.sensors.thermal_config)
        hotspot = config.task['target_prim_path']
        self.assertNotEqual(hotspot, data['equipment_prim_path'])
        self.assertTrue(hotspot.startswith(data['equipment_prim_path']+'/'))
        hot = thermal.binding_for(hotspot)
        normal = thermal.binding_for(data['normal_surface_prim_path'])
        targets = {item.name: item for item in thermal.targets}
        self.assertAlmostEqual(targets[hot.target].temperature_k, 351.15)
        self.assertAlmostEqual(targets[normal.target].temperature_k, 303.15)
        self.assertEqual(hot.material, normal.material)
        other = thermal.binding_for(data['comparison_equipment_prim_path'])
        self.assertNotEqual(other.target if other else thermal.default_target, hot.target)
        self.assertEqual(sum(targets[b.target].temperature_k == 351.15 for b in thermal.bindings), 1)

    def test_display_profile_covers_experiment_without_changing_shared_display(self):
        profile = load_display_profile(ROOT/'configs/factory_inspection_display.yaml')
        self.assertEqual(profile.temperature_min_c, -30)
        self.assertEqual(profile.temperature_max_c, 100)
        self.assertEqual(profile.dde_gain, 0)
        self.assertEqual(profile.polarity, 'white_hot')


if __name__ == '__main__':
    unittest.main()
