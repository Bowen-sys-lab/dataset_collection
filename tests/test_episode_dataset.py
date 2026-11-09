import tempfile,json
from pathlib import Path
import unittest
from scripts.episode_dataset import write_episode_index


class EpisodeDatasetTests(unittest.TestCase):
    def test_city_v2_index_uses_city_task_identity_and_split(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root=Path(folder);run=root/'raw';run.mkdir();replay=root/'replay';replay.mkdir()
            metadata={'trajectory_source':'manual_driving','task':{'task_id':'city-v2-heatmaincover-s1',
                'instruction':'Find the manhole cover.','environment_evidence':{'city_collection':{
                    'version':'city20_long_start_trial_v1','episode':'C01','split':'train'}}},
                'task_result':{'status':'success','marked_by':'keyboard:T'}}
            (run/'run_metadata.json').write_text(json.dumps(metadata))
            trajectory=run/'trajectory.jsonl'
            trajectory.write_text(json.dumps({'frame_id':0,'simulation_time':1.,'command':{}})+'\n')
            from scripts.trajectory_recorder import trajectory_provenance
            (replay/'replay_metadata.json').write_text(json.dumps({'result':'PASS','trajectory':str(trajectory),
                'trajectory_provenance':trajectory_provenance(trajectory),'sample_interval_s':1.,
                'rgb_resolution':[256,205],'thermal_resolution':[128,102]}))
            (replay/'frame_000000.json').write_text(json.dumps({'trajectory_frame_id':0,'replay_simulation_time':1.}))
            for stream in ('rgb','thermal'):(replay/f'frame_000000_{stream}.png').write_bytes(b'image-reference-fixture')
            header=write_episode_index(run,replay,root/'index')
            self.assertEqual((header['scene'],header['episode'],header['split']),('city','C01','train'))
            self.assertEqual(header['task_version'],'city20_long_start_trial_v1')

    def test_common_index_retains_source_time_instruction_and_uncertainty(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root=Path(folder);run=root/'raw';run.mkdir();replay=root/'replay';replay.mkdir()
            metadata={'trajectory_source':'manual_driving','task':{'task_id':'task','instruction':'Inspect the barrel.',
                'environment_evidence':{'collection_episode':{'scene':'forest','episode':'R01','split':'train'}}},
                'task_result':{'status':'unjudged'}}
            (run/'run_metadata.json').write_text(json.dumps(metadata))
            trajectory=run/'trajectory.jsonl';trajectory.write_text(json.dumps({'frame_id':0,'simulation_time':8.1,
                'command':{'forward':1.}})+'\n')
            from scripts.trajectory_recorder import trajectory_provenance
            (replay/'replay_metadata.json').write_text(json.dumps({'result':'PASS','trajectory':str(trajectory),
                'trajectory_provenance':trajectory_provenance(trajectory)}))
            (replay/'frame_000000.json').write_text(json.dumps({'trajectory_frame_id':0,'replay_simulation_time':8.1}))
            for stream in ('rgb','thermal'):(replay/f'frame_000000_{stream}.png').write_bytes(b'image-reference-fixture')
            out=root/'index';write_episode_index(run,replay,out)
            result=json.loads((out/'episode.json').read_text());row=json.loads((out/'frames.jsonl').read_text())
            self.assertEqual(result['instruction'],'Inspect the barrel.')
            self.assertEqual(row['observation_time_s'],8.1)
            self.assertEqual(row['sampled_command'],{'forward':1.})
            self.assertEqual(result['action_alignment_status'],'NOT_CERTIFIED')
            self.assertFalse(result['training_ready'])
            self.assertNotIn('target_position',row)
            with self.assertRaises(FileExistsError):write_episode_index(run,replay,out)
            other=root/'other';other.mkdir()
            (other/'run_metadata.json').write_text(json.dumps(metadata))
            (other/'trajectory.jsonl').write_bytes(trajectory.read_bytes())
            with self.assertRaises(ValueError):write_episode_index(other,replay,root/'mismatched')
            self.assertFalse((root/'mismatched').exists())

    def test_index_records_capture_version_actual_sizes_and_offline_interval(self):
        from scripts.collection_capture_defaults import capture_contract
        from scripts.trajectory_recorder import trajectory_provenance
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root=Path(folder);run=root/'raw';run.mkdir();replay=root/'replay';replay.mkdir()
            metadata={'capture_contract':capture_contract(), 'sample_period_s':.1,
                      'trajectory_source':'manual_driving','task_result':{'status':'unjudged'}}
            (run/'run_metadata.json').write_text(json.dumps(metadata))
            trajectory=run/'trajectory.jsonl'
            trajectory.write_text('{"frame_id":0,"simulation_time":0}\n')
            report={'result':'PASS','trajectory':str(trajectory),
                    'trajectory_provenance':trajectory_provenance(trajectory),
                    'capture_contract':capture_contract(),'sample_interval_s':1.,
                    'rgb_resolution':[256,205],'thermal_resolution':[128,102]}
            (replay/'replay_metadata.json').write_text(json.dumps(report))
            (replay/'frame_000000.json').write_text('{"trajectory_frame_id":0,"replay_simulation_time":0}')
            for stream in ('rgb','thermal'):
                (replay/f'frame_000000_{stream}.png').write_bytes(b'image-reference-fixture')
            header=write_episode_index(run,replay,root/'index')
            self.assertEqual(header['capture_contract'],capture_contract())
            self.assertEqual(header['source_sample_period_s'],.1)
            self.assertEqual(header['sample_interval_s'],1.)
            self.assertEqual(header['rgb_resolution'],[256,205])
            self.assertEqual(header['thermal_resolution'],[128,102])


if __name__=='__main__':unittest.main()
