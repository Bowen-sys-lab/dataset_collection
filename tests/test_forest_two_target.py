import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from scripts import collection_episode as episodes


class ForestSelectionTests(unittest.TestCase):
    def test_two_goals_share_each_start_and_have_three_language_variants(self):
        rows=[r for r in episodes.load_episodes() if r['scene']=='forest']
        self.assertEqual({r['target'] for r in rows},{'barrel','seated_person'})
        self.assertEqual({r['start_version'] for r in rows},{'forest_two_target_v1'})
        for target in ('barrel','seated_person'):
            selected=[r for r in rows if r['target']==target]
            self.assertEqual(len(selected),5)
            self.assertEqual({r['instruction_variant'] for r in selected},{1,2,3})
        for group in ('A','B','D','E','F'):
            pair=[r for r in rows if r['start_group']==group]
            self.assertEqual(pair[0]['candidate_position_m'],pair[1]['candidate_position_m'])
            self.assertEqual(pair[0]['candidate_yaw_degrees'],pair[1]['candidate_yaw_degrees'])

    def test_person_snapshot_selects_person_and_new_scene(self):
        row=episodes.resolve_episode('forest','R02')
        with tempfile.TemporaryDirectory() as folder:
            config_text,task_text=episodes.runtime_snapshot(row,Path(folder))
        config=yaml.safe_load(config_text);task=yaml.safe_load(task_text)
        self.assertIn('forest_fire_flight_1103.usda',config['scene']['stage'])
        self.assertEqual(task['target_prim_path'],'/World/Site/Target_Person_Injured_002')
        self.assertEqual(task['target_position'],[91.5,17.85,19.2])
        self.assertIn('bandage',task['instruction'])
        self.assertNotIn('barrel',task['target_description'].lower())

    def test_new_person_target_is_in_runtime_source_provenance(self):
        row=episodes.resolve_episode('forest','R02')
        inputs=episodes.required_input_paths(row)
        self.assertIn('configs/forest_pilot_person_target.yaml',inputs)
        self.assertIn('forest_fire_scene_20261103_local/scene.usda',inputs)
        self.assertNotIn('forest_fire_scene_local/scene.usda',inputs)

    def test_legacy_forest_rows_keep_the_old_source_and_barrel(self):
        from scripts.forest_collection_config import scene_config_path,target_config_path
        row={'scene':'forest','target':'barrel','start_version':'pilot30_20261031'}
        self.assertEqual(scene_config_path(row).name,'forest_fire_flight_scene_night.yaml')
        self.assertEqual(target_config_path(row).name,'forest_fire_target.yaml')

    def test_new_row_refuses_legacy_readiness_without_target_bound_evidence(self):
        row=episodes.resolve_episode('forest','R02')
        with tempfile.TemporaryDirectory(dir=episodes.ROOT/'validation') as folder:
            path=Path(folder)/'old_report.json'
            key=path.relative_to(episodes.ROOT).as_posix()
            path.write_text(json.dumps(dict(schema_version=1,result='PASS',scene='forest',episodes=['R02'],
                source_manifest_sha256=row['manifest_sha256'],source_inputs={},spawn_passed=True,
                route_clearance_passed=True,view_passed=True,long_replay_passed=True,
                position_m=row['candidate_position_m'],yaw_degrees=row['candidate_yaw_degrees'])))
            proof=dict(native_report=key,input_sha256={},native_evidence_sha256={key:episodes.sha256(path)})
            with patch.object(episodes,'required_input_paths',return_value=set()):
                self.assertFalse(episodes.valid_readiness(row,proof))


class TargetTruthTests(unittest.TestCase):
    def test_standing_person_cannot_supply_seated_person_evidence(self):
        from scripts.forest_target_evidence import target_surface_evidence
        ids=np.array([[1,1],[2,0]])
        labels={1:'/World/Site/Target_Person_Open_001/Figure/CC_Base_Body',
                2:'/World/Site/Target_Person_Injured_002/Figure/Field_Jacket'}
        temp=np.array([[306.15,306.15],[300.15,290.15]])
        result=target_surface_evidence(ids,labels,temp,
            '/World/Site/Target_Person_Injured_002',[27.,33.])
        self.assertEqual(result['target_pixels'],1)
        self.assertEqual(result['expected_temperature_pixels'],1)
        self.assertEqual(result['result'],'PASS')
        result=target_surface_evidence(ids[:1],labels,temp[:1],
            '/World/Site/Target_Person_Injured_002',[27.,33.])
        self.assertEqual(result['result'],'FAIL')

    def test_background_and_wrong_temperature_do_not_pass(self):
        from scripts.forest_target_evidence import target_surface_evidence
        path='/World/Site/Target_Person_Injured_002'
        for ids,labels,temp in [(np.array([[0]]),{},np.array([[256.]])),
                                (np.array([[1]]),{1:path+'/Figure'},np.array([[383.15]]))]:
            self.assertEqual(target_surface_evidence(ids,labels,temp,path,[27.,33.])['result'],'FAIL')
        with self.assertRaises(ValueError):
            target_surface_evidence(np.zeros((2,2)),{},np.zeros((1,1)),path,[27.])

    def test_raw_instance_plane_is_copied_by_existing_truth_adapter(self):
        from scene_intake.forest_collection import capture_truth
        from types import SimpleNamespace
        ids=np.array([[3]],dtype=np.int32)
        camera=SimpleNamespace(last_frame=SimpleNamespace(
            planes={'temperature_k':np.array([[300.15]])}, labels={3:'/World/Person'},instance_id=ids))
        planes,_=capture_truth(camera)
        self.assertIn('instance_id',planes)
        ids[0,0]=0
        self.assertEqual(int(planes['instance_id'][0,0]),3)


class ReadinessReplaySealTests(unittest.TestCase):
    def test_short_replay_proof_refuses_changed_capture_input_and_original_recording(self):
        import importlib.util
        base=episodes.ROOT/'validation/diagnostics/forest-two-target-20261105'
        spec=importlib.util.spec_from_file_location('verify_replay_proof',base/'verify_replay_proof.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);capture=root/'replay';capture.mkdir();record=root/'record';record.mkdir()
            source=root/'capture_input.yaml';source.write_text('recorded source')
            frame=capture/'frame.json';frame.write_text('{}')
            metadata=record/'run_metadata.json';metadata.write_text('{}')
            trajectory=record/'trajectory.jsonl';trajectory.write_text('{}\n')
            sha=episodes.sha256
            (capture/'collection_manifest.json').write_text(json.dumps(dict(schema_version=1,
                files={'frame.json':sha(frame)},inputs={str(source):sha(source)})))
            contract=dict(runtime_contract_verified=True,source_metadata_sha256=sha(metadata),source_trajectory_sha256=sha(trajectory))
            self.assertEqual(module.verify_short_replay_seal(capture,record,contract),(capture/'collection_manifest.json').resolve())
            source.write_text('changed source')
            with self.assertRaisesRegex(ValueError,'input changed'):
                module.verify_short_replay_seal(capture,record,contract)
            source.write_text('recorded source');trajectory.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError,'recording changed'):
                module.verify_short_replay_seal(capture,record,contract)


if __name__=='__main__':unittest.main()
