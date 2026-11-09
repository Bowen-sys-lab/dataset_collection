"""Recording identity is checked before rendering, using real project loaders."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))
from scripts.scene_config import load_scene_config
from scripts.runtime_parameter_provenance import thermal_input_snapshot
try:
    from scene_intake.forest_provenance import validate_recording_source
except ImportError:
    validate_recording_source=None


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ForestProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        base=load_scene_config(ROOT/'configs/forest_fire_scene.yaml',include_task=False)
        self.stage=self.root/'scene.usda';self.stage.write_text('#usda 1.0\n')
        self.raw=dict(schema_version=1,scene=dict(stage=str(self.stage),thermal_scene=str(base.scene.thermal_scene),static_targets=[]),
                      drone=dict(prim_path=base.drone.prim_path,body_path=base.drone.body_path,camera_path=base.drone.camera_path,minimum_height=1.5),
                      sensors=dict(width=320,height=256,sensor_config=str(base.sensors.sensor_config),
                                   thermal_config=str(base.sensors.thermal_config),preview_fast=True,align_thermal_to_rgb=True,
                                   flat_field_enabled=True,sensor_seed=741,display_config=str(base.sensors.display_config),
                                   material_directory=str(base.sensors.material_directory)))
        self.config_path=self.root/'source.yaml';self.config_path.write_text(yaml.safe_dump(self.raw),encoding='utf-8')
        self.config=load_scene_config(self.config_path,include_task=False)
        self.trajectory=self.root/'trajectory.jsonl'
        self.records=[]
        for i in range(2):
            matrix=np.eye(4);matrix[3,0]=i*.1
            self.records.append(dict(frame_id=i,simulation_time=i*.1,position=[i*.1,0,0],rgb_camera_world_matrix=matrix.tolist()))
        self.write_records()
        self.meta=dict(stage_path=str(self.stage),stage_sha256=digest(self.stage),scene_config_path=str(self.config_path),
                       scene_config_sha256=digest(self.config_path),sensor_config_sha256=digest(base.sensors.sensor_config),
                       thermal_config_sha256=digest(base.sensors.thermal_config),thermal_input_snapshot=thermal_input_snapshot(self.config),
                       task=dict(task_id='recorded-task',instruction='Recorded instruction'),
                       task_result=dict(status='success',marked_by='keyboard_T',simulation_time_s=.1))
        self.write_meta()

    def write_meta(self):
        (self.root/'run_metadata.json').write_text(json.dumps(self.meta),encoding='utf-8')

    def write_records(self):
        self.trajectory.write_text(''.join(json.dumps(r)+'\n' for r in self.records),encoding='utf-8')

    def validate(self,config=None):
        self.assertIsNotNone(validate_recording_source,'missing forest preflight')
        return validate_recording_source(config or self.config,self.trajectory)

    def test_matching_source_preserves_recorded_task_and_manual_confirmation(self):
        result=self.validate()
        self.assertEqual(result['recording']['task']['task_id'],'recorded-task')
        self.assertEqual(result['recording']['task_result']['marked_by'],'keyboard_T')
        self.assertEqual(result['validation']['status'],'PASS')

    def test_display_override_is_explicit_but_physical_identity_stays_checked(self):
        raw=copy.deepcopy(self.raw);raw['sensors']['display_config']=str(ROOT/'configs/forest_fire_display_night.yaml')
        p=self.root/'night.yaml';p.write_text(yaml.safe_dump(raw),encoding='utf-8')
        result=self.validate(load_scene_config(p,include_task=False))
        self.assertTrue(result['render_overrides']['display']['changed'])
        self.assertEqual(result['render_overrides']['display']['current']['path'],str(ROOT/'configs/forest_fire_display_night.yaml'))

    def test_changed_or_missing_stage_config_sensor_and_weather_are_rejected(self):
        original=copy.deepcopy(self.meta)
        cases=[('stage_sha256','0'*64),('stage_sha256',None),('scene_config_sha256','0'*64),
               ('sensor_config_sha256','0'*64),('thermal_config_sha256','0'*64),('thermal_input_snapshot',None)]
        for key,value in cases:
            with self.subTest(field=key,value=value):
                self.meta=copy.deepcopy(original);self.meta[key]=value;self.write_meta()
                with self.assertRaises(ValueError):self.validate()
        self.meta=copy.deepcopy(original);self.meta['thermal_input_snapshot']['weather_content_sha256']='0'*64;self.write_meta()
        with self.assertRaises(ValueError):self.validate()

    def test_absent_run_metadata_is_not_certified(self):
        (self.root/'run_metadata.json').unlink()
        with self.assertRaises(ValueError):self.validate()

    def test_missing_historical_task_remains_missing(self):
        self.meta['task']=None;self.write_meta()
        self.assertIsNone(self.validate()['recording']['task'])

    def test_invalid_camera_pose_and_duplicate_time_are_rejected(self):
        original=copy.deepcopy(self.records)
        self.records[1]['rgb_camera_world_matrix']=[[1,0],[0,1]];self.write_records()
        with self.assertRaises(ValueError):self.validate()
        self.records=copy.deepcopy(original);self.records[1]['simulation_time']=0.;self.write_records()
        with self.assertRaises(ValueError):self.validate()


if __name__=='__main__':unittest.main()
