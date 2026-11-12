import json
from pathlib import Path
import tempfile
import unittest

from scripts import episode_replay_source as replay


class ReplaySourceTests(unittest.TestCase):
    def test_pending_collision_cannot_export_raw_attempt_as_corrected_replay(self):
        rows=[{'frame_id':0,'simulation_time':0.,'position':[0,0,0]}]
        with self.assertRaisesRegex(ValueError,'collision'):
            replay.select_source(rows,{'collision_pending':True},1.)

    def test_unconfirmed_recovered_attempt_replays_only_retained_branch(self):
        rows=[{'frame_id':i,'simulation_time':float(i),'position':[i,0,0]} for i in range(5)]
        rows.append({'frame_id':5,'simulation_time':6.,'position':[2,0,0]})
        metadata={'collection_edit_events':[{'type':'undo','simulation_time_s':5.,
            'anchor_frame_id':1,'next_raw_frame_id':5}]}
        selected=replay.select_source(rows,metadata,1.)
        self.assertEqual([row['position'][0] for row in selected],[0,1,2])
        self.assertEqual([row['simulation_time'] for row in selected],[0.,1.,2.])

    def test_recorded_physical_assets_are_checked_independently_of_global_plan(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);asset=root/'scene.usda';asset.write_text('original')
            snapshot=root/'assets.json';snapshot.write_text(json.dumps({str(asset):hashlib.sha256(asset.read_bytes()).hexdigest()}))
            metadata={'task':{'environment_evidence':{'collection_episode':{'source_snapshot_version':1,
                'scene_assets_path':str(snapshot),'scene_assets_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest()}}}}
            replay.validate_recorded_scene_assets(metadata)
            asset.write_text('changed')
            with self.assertRaises(ValueError):replay.validate_recorded_scene_assets(metadata)
    def test_forest_target_comes_from_immutable_recorded_snapshot(self):
        import hashlib,yaml
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'target.yaml'
            path.write_text(yaml.safe_dump({'target':{'prim_path':'/World/Barrel','position_m':[1,2,3]}}))
            task={'target_prim_path':'/World/Barrel','target_position':[1,2,3],
                'environment_evidence':{'collection_episode':{'target_config_path':str(path),
                    'target_config_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
            self.assertEqual(replay.recorded_target_snapshot({'task':task}),path.resolve())
            path.write_text(path.read_text()+'# changed\n')
            with self.assertRaises(ValueError):replay.recorded_target_snapshot({'task':task})
    def test_backend_comes_from_recorded_episode_not_filename(self):
        for scene in ('factory','forest'):
            metadata={'task':{'environment_evidence':{'collection_episode':{'scene':scene}}}}
            self.assertEqual(replay.recorded_backend(metadata),scene)
        with self.assertRaises(ValueError):replay.recorded_backend({'scene_config_path':'forest_fake.yaml'})
    def test_legacy_derived_provenance_keeps_its_existing_shape(self):
        from scripts.trajectory_recorder import trajectory_provenance
        from scripts.replay_source import write_terminal_replay_source
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'raw';source.mkdir()
            row={'frame_id':0,'simulation_time':0,'position':[0,0,0]}
            path=source/'trajectory.jsonl';path.write_text(json.dumps(row)+'\n')
            (source/'run_metadata.json').write_text('{}')
            derived=write_terminal_replay_source(root/'legacy',path,[row])
            self.assertNotIn('source_derivation',trajectory_provenance(derived))

    def test_sampling_keeps_last_source_time_and_refuses_bad_interval(self):
        rows=[{'frame_id':i,'simulation_time':i*.1,'position':[0,0,0]} for i in range(6)]
        selected=replay.select_source(rows,{},.2)
        self.assertEqual([round(row['simulation_time'],3) for row in selected],[0,.2,.4,.5])
        for bad in (0,float('nan'),float('inf')):
            with self.assertRaises(ValueError):replay.select_source(rows,{},bad)

    def test_exact_terminal_included_and_post_t_removed(self):
        import numpy as np
        rows=[{'frame_id':i,'simulation_time':i*.1,'position':[0.,0.,0.]} for i in range(5)]
        terminal={'frame_id':None,'simulation_time':.15,'position':[0.,0.,0.]}
        for key in ('recorded_body_world_matrix','rgb_camera_world_matrix','thermal_camera_world_matrix'):
            terminal[key]=np.eye(4).tolist()
        metadata={'trajectory_source':'manual_driving','task_result':{'status':'success','marked_by':'keyboard:T',
            'simulation_time_s':.15,'drone_position_m':[0.,0.,0.]},'terminal_snapshot':terminal}
        selected=replay.select_source(rows,metadata,.1)
        self.assertEqual([row['simulation_time'] for row in selected],[0,.1,.15])
        self.assertEqual(selected[-1]['action_type'],'STOP')
        self.assertIsNone(selected[-1]['source_frame_id'])

    def test_derived_source_preserves_parent_hash_and_rejects_parent_changes(self):
        from scripts.trajectory_recorder import trajectory_provenance
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'original';run.mkdir()
            original=run/'trajectory.jsonl';original.write_text(json.dumps({'frame_id':0,'simulation_time':0,'position':[0,0,0]})+'\n')
            (run/'run_metadata.json').write_text(json.dumps({'task':{'task_id':'test'},'trajectory_source':'manual_driving'}))
            derived=replay.prepare_source(original,root/'derived',.1)
            context=trajectory_provenance(derived)
            self.assertEqual(context['source_derivation']['trajectory_path'],str(original.resolve()))
            original.write_text(original.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'source.*changed'):
                trajectory_provenance(derived)


if __name__=='__main__':unittest.main()
