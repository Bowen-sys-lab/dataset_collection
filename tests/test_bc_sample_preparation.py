import copy
import unittest

from tests.test_bc_pose_targets import pose


class BCSamplePoseChecks(unittest.TestCase):
    def fixture(self):
        state = pose(1.,[0.,0.,3.],0.)
        state.update(frame_id=0,source_frame_id=2,action_type='MOTION',
                     rgb_camera_world_matrix=copy.deepcopy(state['recorded_body_world_matrix']),
                     thermal_camera_world_matrix=copy.deepcopy(state['recorded_body_world_matrix']))
        frame = {'frame_index': 0,'observation_time_s':1.,'replay_source_frame_id':0,
                 'original_source_frame_id':2,'source_event_ref':None,'action_type':'MOTION'}
        rendered = {'trajectory_frame_id':0,'replay_simulation_time':1.,
                    'frame_id':0,'trajectory_sha256':'source-hash','task_id':'tank','task_result_status':'success',
                    'restored_body_world_matrix':state['recorded_body_world_matrix'],
                    'camera_world_matrix':state['rgb_camera_world_matrix'],
                    'thermal_camera_world_matrix':state['thermal_camera_world_matrix']}
        return state,frame,rendered

    def test_frame_pose_and_ids_match_canonical_source(self):
        from scripts.prepare_bc_samples import validate_frame_pose
        state,frame,rendered = self.fixture()
        validate_frame_pose(frame,state,rendered,index=0,source_provenance=self.provenance())

    def provenance(self):
        return {'trajectory_sha256':'source-hash','task':{'task_id':'tank'},'task_result':{'status':'success'}}

    def test_changed_time_camera_pose_or_source_id_cannot_pass(self):
        from scripts.prepare_bc_samples import validate_frame_pose
        for field in ['replay_simulation_time','camera_world_matrix','restored_body_world_matrix']:
            state,frame,rendered = self.fixture()
            changed=copy.deepcopy(rendered)
            if field.endswith('matrix'):
                changed[field][3][0]+=1.
            else:
                changed[field]+=1.
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_frame_pose(frame,state,changed,index=0,source_provenance=self.provenance())
        state,frame,rendered=self.fixture()
        frame['original_source_frame_id']=99
        with self.assertRaises(ValueError):
            validate_frame_pose(frame,state,rendered,index=0,source_provenance=self.provenance())

    def test_stop_requires_exact_source_event_instead_of_penultimate_motion(self):
        from scripts.prepare_bc_samples import validate_frame_pose
        state,frame,rendered=self.fixture()
        frame['action_type']='STOP'
        with self.assertRaises(ValueError):
            validate_frame_pose(frame,state,rendered,index=0,source_provenance=self.provenance())

    def test_rendered_task_source_and_observation_identity_are_required(self):
        from scripts.prepare_bc_samples import validate_frame_pose
        for field,value in [('frame_id',99),('trajectory_sha256','wrong'),('task_id','other'),('task_result_status','failed')]:
            state,frame,rendered=self.fixture()
            rendered[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_frame_pose(frame,state,rendered,index=0,source_provenance=self.provenance())


class BCExportIntegrationTests(unittest.TestCase):
    def fixture(self, root, period=.1):
        import hashlib,json
        from scripts.trajectory_recorder import build_training_sequence,trajectory_provenance
        from scripts.episode_dataset import write_episode_index
        run=root/'raw';run.mkdir()
        replay=root/'processed/replay';replay.mkdir(parents=True)
        rows=[pose(0.,[0.,0.,3.],0.),pose(period,[0.,0.,3.],0.)]
        terminal=pose(2*period,[.1,0.,3.],0.)
        for i,row in enumerate(rows+[terminal]):
            row.update(frame_id=i,rgb_camera_world_matrix=copy.deepcopy(row['recorded_body_world_matrix']),
                       thermal_camera_world_matrix=copy.deepcopy(row['recorded_body_world_matrix']))
        task={'task_id':'tank','instruction':'Inspect 储罐','environment_evidence':{'collection_episode':
              {'scene':'factory','split':'train','episode':'F01','start_group':'A','target':'tank'}}}
        meta={'trajectory_source':'manual_driving','task':task,'terminal_snapshot':terminal,
              'task_result':{'status':'success','marked_by':'keyboard:T','simulation_time_s':2*period,
                             'drone_position_m':[.1,0.,3.]}}
        if period != .1:
            meta['sample_period_s'] = period
        (run/'run_metadata.json').write_text(json.dumps(meta),encoding='utf-8')
        (run/'trajectory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        states=build_training_sequence(rows,meta)
        derived=root/'source_sequence';derived.mkdir()
        source=derived/'trajectory.jsonl'
        source.write_text(''.join(json.dumps(r)+'\n' for r in states),encoding='utf-8')
        dmeta=dict(meta,source_derivation_schema_version=1,source_trajectory_path=str(run/'trajectory.jsonl'),
                   source_trajectory_sha256=hashlib.sha256((run/'trajectory.jsonl').read_bytes()).hexdigest(),
                   source_run_metadata_sha256=hashlib.sha256((run/'run_metadata.json').read_bytes()).hexdigest())
        (derived/'run_metadata.json').write_text(json.dumps(dmeta),encoding='utf-8')
        provenance=trajectory_provenance(source)
        (replay/'replay_metadata.json').write_text(json.dumps({'result':'PASS','trajectory':str(source),
                                               'trajectory_provenance':provenance}),encoding='utf-8')
        for i,state in enumerate(states):
            frame={'frame_id':i,'trajectory_frame_id':i,'replay_simulation_time':state['simulation_time'],
                   'restored_body_world_matrix':state['recorded_body_world_matrix'],
                   'camera_world_matrix':state['rgb_camera_world_matrix'],
                   'thermal_camera_world_matrix':state['thermal_camera_world_matrix'],
                   'task_id':'tank','task_result_status':'success','trajectory_sha256':provenance['trajectory_sha256']}
            (replay/f'frame_{i:06d}.json').write_text(json.dumps(frame),encoding='utf-8')
            for kind in ['rgb','thermal']:
                (replay/f'frame_{i:06d}_{kind}.png').write_bytes(b'reference-fixture')
        write_episode_index(run,replay,replay.parent/'dataset')
        return run,replay,source

    def test_one_second_export_retains_real_durations_and_terminal(self):
        import json, tempfile
        from pathlib import Path
        from scripts.prepare_bc_samples import prepare_candidates
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, replay, _ = self.fixture(root, period=1.)
            out = root/'candidates'
            report = prepare_candidates(run, replay, out)
            self.assertEqual(report['result'], 'PASS_POSE_TARGET_CANDIDATES')
            samples = [json.loads(line) for line in (out/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual([row['pose_target']['duration_s'] for row in samples[:-1]], [1., 1.])
            self.assertTrue(samples[-1]['pose_target']['stop'])

    def test_export_hash_matches_actual_bytes_and_keeps_hold_stop_and_derived_sources(self):
        import hashlib,json,tempfile
        from pathlib import Path
        from scripts.prepare_bc_samples import prepare_candidates
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,replay,source=self.fixture(root)
            out=root/'candidates'
            report=prepare_candidates(run,replay,out,horizon_s=.1)
            self.assertEqual(report['samples_sha256'],hashlib.sha256((out/'samples.jsonl').read_bytes()).hexdigest())
            self.assertEqual(report['counts'],{'MOVE':1,'HOLD':1,'STOP':1})
            manifest=json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
            self.assertIn(str(source.resolve()),manifest)
            self.assertIn(str((source.parent/'run_metadata.json').resolve()),manifest)
            self.assertFalse(report['training_ready'])

    def test_rewrite_between_frame_consumption_and_final_check_is_rejected(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts.frame_metadata import FrameMetadataReader
        from scripts.prepare_bc_samples import prepare_candidates
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run,replay,_=self.fixture(root)
            original=FrameMetadataReader.read
            def changing(reader,path,**kwargs):
                value=original(reader,path,**kwargs)
                if Path(path).name=='frame_000001.json':
                    path=Path(path)
                    path.write_text(path.read_text(encoding='utf-8')+' ',encoding='utf-8')
                return value
            with patch.object(FrameMetadataReader,'read',changing),self.assertRaises(ValueError):
                prepare_candidates(run,replay,root/'candidates',horizon_s=.1)
