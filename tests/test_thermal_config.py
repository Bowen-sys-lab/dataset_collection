import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.thermal_config import ThermalConfigError, load_thermal_config


class ThermalConfigTests(unittest.TestCase):
    def write_config(self, data):
        handle = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        handle.close()
        path = Path(handle.name)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def base_config(self):
        return {
            'schema_version': 1,
            'defaults': {'target': 'ambient', 'material': 'abs_plastic_white'},
            'targets': {
                'ambient': {'mode': 'ambient'},
                'drone': {'mode': 'fixed', 'temperature_c': 35.0},
                'warm': {'mode': 'offset', 'offset_c': 10.0},
            },
            'bindings': [
                {'name': 'drone', 'prim_path': '/World/Crazyflie/body',
                 'include_children': True, 'target': 'drone', 'material': 'painted_composite'},
                {'name': 'cube', 'prim_path': '/World/Cube',
                 'include_children': False, 'target': 'warm', 'material': 'concrete'},
            ],
        }

    def test_converts_celsius_to_kelvin_and_supports_target_modes(self):
        config = load_thermal_config(self.write_config(self.base_config()))

        self.assertAlmostEqual(config.target('drone').temperature_k, 308.15, places=6)
        self.assertIsNone(config.target('ambient').temperature_k)
        self.assertAlmostEqual(config.target('warm').offset_k, 10.0, places=6)

    def test_matches_exact_paths_and_descendants(self):
        config = load_thermal_config(self.write_config(self.base_config()))

        self.assertEqual(config.target_for('/World/Crazyflie/body'), 'drone')
        self.assertEqual(config.target_for('/World/Crazyflie/body/visual/mesh'), 'drone')
        self.assertEqual(config.material_for('/World/Crazyflie/body/visual/mesh'), 'painted_composite')
        self.assertEqual(config.target_for('/World/Cube'), 'warm')
        self.assertEqual(config.target_for('/World/Cube/child'), 'ambient')
        self.assertEqual(config.target_for('/World/GroundPlane'), 'ambient')
        self.assertEqual(config.target_for('/World/Crazyflie/body_extra'), 'ambient')

    def test_more_specific_binding_wins(self):
        data = self.base_config()
        data['bindings'].append({
            'name': 'battery', 'prim_path': '/World/Crazyflie/body/battery',
            'include_children': True, 'target': 'warm', 'material': 'concrete',
        })
        config = load_thermal_config(self.write_config(data))

        self.assertEqual(config.target_for('/World/Crazyflie/body/battery/case'), 'warm')
        self.assertEqual(config.material_for('/World/Crazyflie/body/arm'), 'painted_composite')

    def test_rejects_unknown_target_empty_material_and_empty_path(self):
        for change in (
            {'bindings': [{**self.base_config()['bindings'][0], 'target': 'missing'}]},
            {'defaults': {'target': 'ambient', 'material': ''}},
            {'bindings': [{**self.base_config()['bindings'][0], 'prim_path': ''}]},
        ):
            data = self.base_config()
            data.update(change)
            with self.assertRaises(ThermalConfigError):
                load_thermal_config(self.write_config(data))

    def test_rejects_duplicate_binding_names(self):
        data = self.base_config()
        data['targets']['drone_copy'] = data['targets']['drone']
        data['bindings'].append(dict(data['bindings'][0]))
        with self.assertRaises(ThermalConfigError):
            load_thermal_config(self.write_config(data))

    def test_default_project_configuration_loads(self):
        path = Path(__file__).resolve().parents[1] / 'configs' / 'crazyflie_thermal.yaml'
        config = load_thermal_config(path)

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.target_for('/World/Unconfigured'), config.default_target)
        self.assertAlmostEqual(config.target(config.target_for('/World/GroundPlane')).temperature_k,
                               293.15)
        self.assertTrue(config.material_for('/World/Cube'))

    def test_rejects_invalid_temperatures_modes_and_typo_fields(self):
        for target in [
            {'mode': 'fixed'}, {'mode': 'unknown'},
            {'mode': 'fixed', 'temperature_c': -273.15},
            {'mode': 'fixed', 'temperature_c': float('nan')},
            {'mode': 'fixed', 'temperature_c': True},
            {'mode': 'fixed', 'temperature_c': '35'},
            {'mode': 'ambient', 'temperature_c': 20},
            {'mode': 'offset', 'offset_c': float('inf')},
            {'mode': 'fixed', 'temprature_c': 35},
        ]:
            with self.subTest(target=target):
                data = self.base_config()
                data['targets']['drone'] = target
                with self.assertRaises(ThermalConfigError):
                    load_thermal_config(self.write_config(data))

    def test_school_configuration_explicitly_maps_scene_categories(self):
        path = Path(__file__).resolve().parents[1] / 'configs' / 'school_vln_thermal.yaml'
        config = load_thermal_config(path)
        school_surfaces = {
            1: ('school_road', 'asphalt_dry'),
            2: ('school_road', 'asphalt_dry'),
            3: ('school_road', 'concrete'),
            4: ('school_vegetation', 'vegetation_leaf'),
            6: ('school_road', 'asphalt_dry'),
            **{index: ('school_road', 'concrete') for index in (5, 7, 8, 9, 10, 11, 12)},
            **{index: ('school_building', 'concrete') for index in (13, 20, 21, 22)},
            **{index: ('school_glass', 'glass_windshield') for index in (*range(14, 20), 23)},
            24: ('school_plastic', 'abs_plastic_white'),
            25: ('school_plastic', 'abs_plastic_white'),
            26: ('school_metal', 'aircraft_aluminium_painted'),
        }
        expected = {
            f'/World/E_school3_{index:03d}_{index + 2}': values
            for index, values in school_surfaces.items()
        }
        expected.update({
            '/World/GroundPlane': ('school_ground', 'soil_dry'),
            '/World/Target_01': ('school_spot', 'painted_composite'),
            '/World/Crazyflie': ('school_drone', 'painted_composite'),
            '/World/PaintTool/cube_52187': ('school_ground', 'concrete'),
            '/World/PaintTool/tree2_e0865': ('school_vegetation', 'vegetation_leaf'),
        })
        for prim_path, (target, material) in expected.items():
            with self.subTest(prim_path=prim_path):
                self.assertIsNotNone(config.binding_for(prim_path))
                self.assertEqual(config.target_for(prim_path + '/mesh'), target)
                self.assertEqual(config.material_for(prim_path + '/mesh'), material)
        self.assertEqual(config.target_for('/World/NewUnconfiguredObject'), config.default_target)
        self.assertGreater(config.target('school_spot').temperature_k,
                           config.target('school_ground').temperature_k)

    def test_rejects_bad_paths_and_nonboolean_include_children(self):
        for patch in [{'prim_path': 'World/Cube'}, {'prim_path': '/World/Cube/'},
                      {'prim_path': '/World//Cube'}, {'prim_path': '/World/Cube.attr'},
                      {'include_children': 'false'}]:
            data = self.base_config()
            data['bindings'][0].update(patch)
            with self.assertRaises(ThermalConfigError):
                load_thermal_config(self.write_config(data))

    def test_rejects_ambiguous_duplicate_paths(self):
        data = self.base_config()
        data['bindings'].append({**data['bindings'][0], 'name': 'duplicate_path'})
        with self.assertRaises(ThermalConfigError):
            load_thermal_config(self.write_config(data))

    def test_rejects_duplicate_yaml_keys_instead_of_silently_overwriting(self):
        path = self.write_config(self.base_config())
        path.write_text('schema_version: 1\nschema_version: 2\n', encoding='utf-8')
        with self.assertRaisesRegex(ThermalConfigError, 'duplicate'):
            load_thermal_config(path)

    def test_reports_content_hash_for_reproducibility(self):
        import hashlib
        path = self.write_config(self.base_config())
        config = load_thermal_config(path)
        self.assertEqual(config.sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(config.source_path, str(path.resolve()))


if __name__ == '__main__':
    unittest.main()
