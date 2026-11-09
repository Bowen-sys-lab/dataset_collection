from pathlib import Path
import tempfile
import unittest
import yaml

from scripts.scene_config import load_scene_config,SceneConfigError
from scripts.model02_migration import bind_surface_rows,validate_export_snapshot,night_weather
import hashlib
from scripts.project_materials import load_project_material_library

ROOT=Path(__file__).resolve().parents[1]


class MigrationTests(unittest.TestCase):
    def test_night_weather_follows_authored_air_wind_and_elapsed_night(self):
        text,start=night_weather({'air_temperature_c':23.,'wind_m_s':2.,'hours_after_sunset':3.})
        self.assertEqual(start,'2026-10-18T13:00:00Z')
        self.assertIn(',23.0,0.60,2.0,0.10,0,0,23000,0',text)
        with self.assertRaises(ValueError):night_weather({'air_temperature_c':20,'wind_m_s':1,'hours_after_sunset':20})

    def test_temperature_edit_reuses_pinned_geometry_without_authoring_hash_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            blend=Path(folder)/'snapshot.blend';asset=Path(folder)/'asset.usdc'
            blend.write_bytes(b'geometry snapshot');asset.write_bytes(b'USD geometry')
            exported={'authoring_sha256':'old temperature snapshot',
                'configured_blend_sha256':hashlib.sha256(blend.read_bytes()).hexdigest(),
                'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest()}
            validate_export_snapshot(exported,blend,asset)
            authored={'targets':{'ta':{'mode':'fixed','temperature_c':99}},
                'surfaces':{'A':{'object':'obj','target':'ta','material':'concrete'}}}
            rows=[{'surface_id':'A','object':'obj','usd_prim_path':'/World/Actual/Mesh'}]
            self.assertEqual(bind_surface_rows(rows,authored)[0]['target'],'ta')
            asset.write_bytes(b'changed geometry')
            with self.assertRaises(ValueError):validate_export_snapshot(exported,blend,asset)

    def test_material_merge_retains_existing_and_rejects_conflicting_override(self):
        directory=ROOT/'factory/model02_thermal_v3/configuration/materials'
        base=load_project_material_library()
        merged=load_project_material_library(directory)
        self.assertTrue(set(base)<=set(merged))
        self.assertIn('industrial_painted_steel',merged)
        for name in base:self.assertEqual(base[name].content_hash(),merged[name].content_hash())
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as folder:
            data=yaml.safe_load((directory/'concrete.yaml').read_text(encoding='utf-8'))
            data['material']['thermal']['density_kg_m3']=1
            (Path(folder)/'concrete.yaml').write_text(yaml.safe_dump(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'override'):load_project_material_library(Path(folder))

    def test_surface_bindings_use_actual_paths_and_reject_missing_duplicate_or_stale(self):
        authored={'surfaces':{'A':{'object':'obj','target':'ta','material':'concrete'}}}
        rows=[{'surface_id':'A','object':'obj','usd_prim_path':'/World/Actual/Mesh'}]
        self.assertEqual(bind_surface_rows(rows,authored)[0]['prim_path'],'/World/Actual/Mesh')
        for invalid in [[],rows+rows,[dict(rows[0],object='changed')]]:
            with self.assertRaises(ValueError):bind_surface_rows(invalid,authored)

    def test_optional_material_directory_is_loaded_and_checked_before_isaac(self):
        path=ROOT/'configs/crazyflie_scene.yaml'
        data=yaml.safe_load(path.read_text(encoding='utf-8'))
        for section,key in [('scene','stage'),('scene','thermal_scene'),('sensors','sensor_config'),('sensors','thermal_config')]:
            data[section][key]=str((path.parent/data[section][key]).resolve())
        with tempfile.TemporaryDirectory(dir=ROOT/'tests') as directory:
            config=Path(directory)/'scene.yaml'
            data['sensors']['material_directory']=str(ROOT/'factory/model02_thermal_v3/configuration/materials')
            config.write_text(yaml.safe_dump(data),encoding='utf-8')
            self.assertEqual(load_scene_config(config).sensors.material_directory,Path(data['sensors']['material_directory']))
            data['sensors']['material_directory']=str(Path(directory)/'missing')
            config.write_text(yaml.safe_dump(data),encoding='utf-8')
            with self.assertRaises(SceneConfigError):load_scene_config(config)

    def test_two_bound_tasks_share_night_and_keep_distinct_actual_cores(self):
        tank=load_scene_config(ROOT/'configs/model02_v3_tank_scene.yaml')
        pipe=load_scene_config(ROOT/'configs/model02_v3_pipe_scene.yaml')
        self.assertEqual(tank.scene.stage,pipe.scene.stage)
        self.assertNotEqual(tank.task['target_prim_path'],pipe.task['target_prim_path'])
        self.assertNotEqual(tank.task['task_id'],pipe.task['task_id'])
        from scripts.thermal_config import load_thermal_config
        thermal=load_thermal_config(tank.sensors.thermal_config)
        self.assertGreaterEqual(len(thermal.bindings),8327)
        from irsim.config.scene import load_scene_config as load_environment
        from irsim.thermal.weather_io import load_weather_csv
        env=load_environment(tank.scene.thermal_scene).scene
        weather=load_weather_csv(env.weather_file)
        self.assertTrue((weather.t_air_k==293.15).all())
        self.assertTrue((weather.dni_w_m2==0).all() and (weather.dhi_w_m2==0).all())
        for config in [tank,pipe]:
            evidence=config.task['environment_evidence']['factory_inspection']
            self.assertGreater(evidence['minimum_route_body_clearance_lower_bound_m'],0)
            self.assertIn(config.task['target_prim_path'],evidence['hotspot_prim_paths'])


if __name__=='__main__':unittest.main()
