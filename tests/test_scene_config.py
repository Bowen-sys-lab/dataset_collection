import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts import scene_config
from scripts.scene_config import SceneConfigError, load_scene_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'configs' / 'crazyflie_scene.yaml'


class SceneConfigTests(unittest.TestCase):
    def test_forest_explicitly_excludes_visual_fire_from_physical_obstacles(self):
        config = load_scene_config(ROOT / 'configs/forest_pilot_two_target_scene.yaml')
        self.assertIn('/World/Site/Target_BurningBarrel_001/Flames', config.scene.collision_exclude_paths)

    def test_optional_display_profile_loads_and_is_validated_before_kit(self):
        import yaml
        data = yaml.safe_load(DEFAULT.read_text(encoding='utf-8'))
        for section, key in (('scene', 'stage'), ('scene', 'thermal_scene'),
                             ('sensors', 'sensor_config'), ('sensors', 'thermal_config')):
            data[section][key] = str((DEFAULT.parent / data[section][key]).resolve())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            display = root / 'display.yaml'
            display.write_text('schema_version: 1\nmode: vendor\ndde_gain: 0\n', encoding='utf-8')
            data['sensors']['display_config'] = 'display.yaml'
            path = root / 'scene.yaml'
            path.write_text(yaml.safe_dump(data), encoding='utf-8')
            config = load_scene_config(path)
            self.assertEqual(config.sensors.display_config, display.resolve())
            display.write_text('schema_version: 1\nmode: wrong\n', encoding='utf-8')
            with self.assertRaises(SceneConfigError):
                load_scene_config(path)

    def test_school_config_keeps_paths_and_temperatures_separate(self):
        config = load_scene_config(ROOT/'configs/school_vln_scene.yaml')
        self.assertEqual(config.scene.stage.name, 'school_vln_01.usd')
        self.assertEqual(config.scene.static_targets,
                         ('/World/Target_01', '/World/office', '/World/hospital'))
        self.assertTrue(config.sensors.align_thermal_to_rgb)
        self.assertEqual(config.sensors.thermal_config.name, 'school_vln_thermal.yaml')

    def test_school_scene_loads_its_default_search_task(self):
        config = load_scene_config(ROOT/'configs/school_vln_scene.yaml')

        self.assertEqual(config.task['task_id'], 'school_find_spot_01')
        self.assertIn('灌木绿化、树木', config.task['instruction'])
        self.assertNotIn('/World/Target_01', config.task['instruction'])
        self.assertEqual(config.task['target_prim_path'], '/World/Target_01')
        self.assertEqual(config.task['environment_landmarks'],
                         ['铺装空地', '灌木或绿化带', '树木', '低矮边界墙或花坛边缘'])

    def test_legacy_scene_without_task_remains_loadable(self):
        config = load_scene_config(DEFAULT)
        self.assertIsNone(config.task)

    def test_offline_scene_loading_uses_recorded_task_if_current_task_is_unavailable(self):
        import yaml
        data = yaml.safe_load(DEFAULT.read_text(encoding='utf-8'))
        for section, key in (('scene', 'stage'), ('scene', 'thermal_scene'),
                             ('sensors', 'sensor_config'), ('sensors', 'thermal_config')):
            data[section][key] = str((DEFAULT.parent / data[section][key]).resolve())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data['task'] = {'file': 'task.yaml'}
            path = root / 'scene.yaml'
            path.write_text(yaml.safe_dump(data), encoding='utf-8')
            for content in (None, 'instruction: [broken'):
                with self.subTest(task_content=content):
                    if content is not None:
                        (root / 'task.yaml').write_text(content, encoding='utf-8')
                    with self.assertRaises(SceneConfigError):
                        load_scene_config(path)
                    try:
                        config = load_scene_config(path, include_task=False)
                    except TypeError as exc:
                        self.fail(f'Offline loading still depends on the current task file: {exc}')
                    self.assertIsNone(config.task)
                    self.assertIsNone(config.task_sha256)
                    self.assertEqual(config.scene.stage.name, 'lesson_crazyflie.usd')

    def test_scene_task_rejects_blank_instruction_before_kit_startup(self):
        import yaml
        data = yaml.safe_load(DEFAULT.read_text(encoding='utf-8'))
        for section, key in (('scene', 'stage'), ('scene', 'thermal_scene'),
                             ('sensors', 'sensor_config'), ('sensors', 'thermal_config')):
            data[section][key] = str((DEFAULT.parent / data[section][key]).resolve())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_path = root / 'task.yaml'
            task_path.write_text('task_id: bad-task\ninstruction: " "\n', encoding='utf-8')
            data['task'] = {'file': str(task_path)}
            path = root / 'scene.yaml'
            path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding='utf-8')
            with self.assertRaisesRegex(SceneConfigError, 'task.instruction'):
                load_scene_config(path)

    def test_task_resolution_merges_scene_defaults_file_and_cli_in_order(self):
        resolver = getattr(scene_config, 'resolve_task_config', None)
        self.assertTrue(callable(resolver), 'Task defaults and overrides are not resolved centrally')
        with tempfile.TemporaryDirectory() as directory:
            task_file = Path(directory) / 'override.yaml'
            task_file.write_text(
                'target_description: "黄色 Spot"\ninstruction: "从侧门开始找 Spot"\n',
                encoding='utf-8')
            resolved = resolver(
                {'task_id': 'school_find_spot_01', 'instruction': '场景默认任务',
                 'target_prim_path': '/World/Target_01', 'target_description': 'Spot'},
                task_file=task_file,
                cli_overrides={'instruction': '从起点寻找 Spot', 'task_id': None},
            )

        self.assertEqual(resolved, {
            'task_id': 'school_find_spot_01',
            'instruction': '从起点寻找 Spot',
            'target_prim_path': '/World/Target_01',
            'target_description': '黄色 Spot',
        })

    def test_legacy_instruction_only_cli_override_remains_supported(self):
        resolved = scene_config.resolve_task_config(
            None, cli_overrides={'instruction': 'find the target'})
        self.assertEqual(resolved, {'instruction': 'find the target'})

    def test_legacy_task_file_null_results_do_not_become_new_run_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.yaml'
            path.write_text('task_id: spot\ninstruction: find Spot\nsuccess: null\nsuccess_time: null\n',
                            encoding='utf-8')
            task = scene_config.resolve_task_config(None, task_file=path)
        self.assertEqual(task, {'task_id': 'spot', 'instruction': 'find Spot'})

    def test_task_yaml_rejects_values_that_cannot_be_saved_as_finite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'task.yaml'
            for extra in ('environment_evidence: {date: 2026-10-12}\n',
                          'target_position: [.nan, 0, 0]\n'):
                with self.subTest(extra=extra):
                    path.write_text('task_id: spot\ninstruction: find Spot\n' + extra, encoding='utf-8')
                    with self.assertRaisesRegex(SceneConfigError, 'JSON'):
                        scene_config.resolve_task_config(None, task_file=path)

    def test_default_config_resolves_paths_and_hash(self):
        config = load_scene_config(DEFAULT)
        self.assertEqual(config.drone.prim_path, '/World/Crazyflie')
        self.assertEqual(config.sensors.width, 160)
        self.assertTrue(config.scene.stage.is_file())
        self.assertEqual(config.sha256, hashlib.sha256(DEFAULT.read_bytes()).hexdigest())

    def test_duplicate_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.yaml'
            path.write_text('schema_version: 1\nschema_version: 1\n', encoding='utf-8')
            with self.assertRaises(SceneConfigError):
                load_scene_config(path)

    def test_missing_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.yaml'
            path.write_text('schema_version: 1\nscene: {}\ndrone: {}\nsensors: {}\n', encoding='utf-8')
            with self.assertRaises(SceneConfigError):
                load_scene_config(path)

    def test_invalid_prim_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.yaml'
            path.write_text(DEFAULT.read_text(encoding='utf-8').replace('/World/Crazyflie', 'Crazyflie'), encoding='utf-8')
            with self.assertRaises(SceneConfigError):
                load_scene_config(path)


if __name__ == '__main__':
    unittest.main()
