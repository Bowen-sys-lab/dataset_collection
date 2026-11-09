import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import numpy as np
from tests.test_forest_merge import make_pass
from scene_intake.forest_collection import write_manifest
from scene_intake import make_replay_metadata as bridge
from scene_intake.capture_integrity import verify_manifest
from scripts.frame_metadata import load_frame_metadata


class ForestReplayLinkTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.capture=self.root/'capture';make_pass(self.capture,hot=True)
        self.trajectory=self.root/'trajectory.jsonl'
        self.trajectory.write_text(json.dumps(dict(frame_id=5,simulation_time=1.,position=[0,0,0],
                                                    rgb_camera_world_matrix=np.eye(4).tolist()))+'\n',encoding='utf-8')
        (self.root/'run_metadata.json').write_text(json.dumps(dict(stage_sha256='e'*64,
              sensor_config_sha256='a'*64,thermal_config_sha256='b'*64)),encoding='utf-8')
        p=self.capture/'frame_000000.json';m=json.loads(p.read_text())
        m.update(replay_simulation_time=1.,trajectory_sha256=hashlib.sha256(self.trajectory.read_bytes()).hexdigest(),
                 task_id=None,task_result_status='unjudged');p.write_text(json.dumps(m),encoding='utf-8')
        (self.capture/'collection_report.json').write_text(json.dumps(dict(result='PASS',frames=1,
                                      rgb_resolution=[4,4],thermal_resolution=[2,2])),encoding='utf-8')

    def test_a_sealed_manifest_cannot_be_silently_rewritten(self):
        write_manifest(self.capture,[])
        original=(self.capture/'collection_manifest.json').read_bytes()
        with self.assertRaises(FileExistsError):write_manifest(self.capture,[])
        self.assertEqual((self.capture/'collection_manifest.json').read_bytes(),original)

    def test_video_bridge_has_an_explicit_separate_output(self):
        self.assertTrue(hasattr(bridge,'prepare_video_bundle'),'missing sealed-capture video bundle API')

    def test_compact_stage_identity_survives_video_and_original_still_packs(self):
        from scene_intake.forest_dataset import pack_dataset
        from scripts.generate_replay_video import load_replay
        self.assertNotIn('source_stage_sha256',json.loads((self.capture/'frame_000000.json').read_text()))
        write_manifest(self.capture,[])
        before={p.name:p.read_bytes() for p in self.capture.iterdir() if p.is_file()}
        result=bridge.prepare_video_bundle(self.capture,self.trajectory,self.root/'video-input')
        self.assertEqual(result['source_stage_sha256'],'e'*64)
        self.assertEqual(len(load_replay(self.root/'video-input').frames),1)
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.capture.iterdir() if p.is_file()})
        verify_manifest(self.capture)
        self.assertEqual(pack_dataset(self.capture,self.root/'index')['frames'],1)

    def test_missing_stage_or_wrong_trajectory_is_rejected_before_publication(self):
        p=self.capture/'frame_000000.json';m=json.loads(p.read_text());m['source_stage_sha256']=None
        p.write_text(json.dumps(m));write_manifest(self.capture,[])
        with self.assertRaises(ValueError):bridge.prepare_video_bundle(self.capture,self.trajectory,self.root/'video-input')
        self.assertFalse((self.root/'video-input').exists())

    def test_wrong_source_trajectory_and_changed_common_metadata_are_rejected(self):
        write_manifest(self.capture,[])
        original=self.trajectory.read_text();self.trajectory.write_text(original+'\n')
        with self.assertRaises(ValueError):bridge.prepare_video_bundle(self.capture,self.trajectory,self.root/'video-input')
        self.trajectory.write_text(original)
        common=self.capture/'common_metadata.json';common.write_text(common.read_text()+'\n')
        with self.assertRaises(ValueError):bridge.prepare_video_bundle(self.capture,self.trajectory,self.root/'video-input')
        self.assertFalse((self.root/'video-input').exists())

    def test_new_capture_can_seal_replay_metadata_before_indexing(self):
        result=bridge.build_replay_metadata(self.capture,self.trajectory,require_manifest=False)
        (self.capture/'replay_metadata.json').write_text(json.dumps(result),encoding='utf-8')
        manifest=write_manifest(self.capture,[])
        self.assertIn('replay_metadata.json',manifest['files'])
        verify_manifest(self.capture)

    def test_video_bridge_rejects_sensor_identity_mismatch(self):
        p=self.root/'run_metadata.json';m=json.loads(p.read_text());m['sensor_config_sha256']='0'*64
        p.write_text(json.dumps(m));write_manifest(self.capture,[])
        with self.assertRaises(ValueError):bridge.prepare_video_bundle(self.capture,self.trajectory,self.root/'video-input')
        self.assertFalse((self.root/'video-input').exists())

    def test_bridge_rejects_changed_sealed_instruction_and_confirmation(self):
        from scripts.trajectory_recorder import trajectory_provenance
        run=self.root/'run_metadata.json';original=json.loads(run.read_text())
        original.update(task={'task_id':None,'instruction':'original'},
                        task_result={'status':'unjudged','marked_by':'keyboard:T','simulation_time_s':1.})
        run.write_text(json.dumps(original))
        saved=trajectory_provenance(self.trajectory)
        report=self.capture/'collection_report.json';data=json.loads(report.read_text())
        data['recording_context']={'recording':saved};report.write_text(json.dumps(data))
        result=bridge.build_replay_metadata(self.capture,self.trajectory,require_manifest=False)
        (self.capture/'replay_metadata.json').write_text(json.dumps(result))
        write_manifest(self.capture,[])
        import copy
        for field,value in [('instruction','changed'),('marked_by','other'),('simulation_time_s',999.)]:
            with self.subTest(field=field):
                changed=copy.deepcopy(original)
                changed['task' if field=='instruction' else 'task_result'][field]=value
                run.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    bridge.prepare_video_bundle(self.capture,self.trajectory,self.root/('bad-'+field))
                self.assertFalse((self.root/('bad-'+field)).exists())

    def test_bridge_rejects_conflicting_stored_provenance(self):
        from scripts.trajectory_recorder import trajectory_provenance
        saved=trajectory_provenance(self.trajectory);saved['task_cli_overrides']={'instruction':'conflict'}
        report=self.capture/'collection_report.json';data=json.loads(report.read_text())
        data['recording_context']={'recording':saved};report.write_text(json.dumps(data))
        write_manifest(self.capture,[])
        with self.assertRaises(ValueError):
            bridge.prepare_video_bundle(self.capture,self.trajectory,self.root/'video-input')

    def test_video_consumer_enforces_bundle_and_source_seals(self):
        from scripts.generate_replay_video import load_replay
        from PIL import Image
        write_manifest(self.capture,[])
        out=self.root/'video-input';bridge.prepare_video_bundle(self.capture,self.trajectory,out)
        rgb=out/'frame_000000_rgb.png';original=rgb.read_bytes()
        Image.new('RGB',(4,4),(20,40,60)).save(rgb)
        with self.assertRaises(ValueError):load_replay(out)
        rgb.write_bytes(original)
        load_replay(out)
        manifest=self.capture/'collection_manifest.json';manifest.write_text(manifest.read_text()+'\n')
        with self.assertRaises(ValueError):load_replay(out)

    def test_video_consumer_enforces_direct_capture_seal(self):
        from scripts.generate_replay_video import load_replay
        from PIL import Image
        result=bridge.build_replay_metadata(self.capture,self.trajectory,require_manifest=False)
        (self.capture/'replay_metadata.json').write_text(json.dumps(result))
        write_manifest(self.capture,[])
        load_replay(self.capture)
        Image.new('RGB',(4,4),(20,40,60)).save(self.capture/'frame_000000_rgb.png')
        with self.assertRaises(ValueError):load_replay(self.capture)


if __name__=='__main__':unittest.main()
