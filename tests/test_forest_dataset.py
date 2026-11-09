import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scene_intake import forest_dataset as dataset
from tests.test_forest_merge import make_pass


class ForestDatasetTests(unittest.TestCase):
    def test_replay_index_preserves_recorded_manual_confirmation(self):
        from scene_intake.forest_collection import write_manifest
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);make_pass(p/'capture',hot=True)
            f=p/'capture/frame_000000.json';m=json.loads(f.read_text())
            m.update(collection_profile={'name':'native-readable'},task_id='recorded-task',
                     task_result_status='success',trajectory_sha256='f'*64)
            f.write_text(json.dumps(m),encoding='utf-8')
            recorded=dict(task={'task_id':'recorded-task'},task_result={'status':'success','marked_by':'keyboard_T'},trajectory_sha256='f'*64)
            (p/'capture/collection_report.json').write_text(json.dumps(dict(result='PASS',frames=1,
                      task={'task_id':'recorded-task'},recording_context={'recording':recorded})),encoding='utf-8')
            write_manifest(p/'capture',[])
            result=dataset.pack_dataset(p/'capture',p/'dataset',target_position=[0,0,-10])
            self.assertEqual(result['task_result'],{'status':'success','marked_by':'keyboard_T'})

    def test_manifest_cannot_omit_a_modified_consumed_plane(self):
        from scene_intake.forest_collection import write_manifest
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);make_pass(p/'capture',hot=True)
            write_manifest(p/'capture',[])
            f=p/'capture/collection_manifest.json';m=json.loads(f.read_text())
            del m['files']['frame_000000_apparent_t.npy'];f.write_text(json.dumps(m))
            np.save(p/'capture/frame_000000_apparent_t.npy',np.full((2,2),300.,np.float32))
            with self.assertRaises(ValueError):dataset.pack_dataset(p/'capture',p/'dataset')
            self.assertFalse((p/'dataset').exists())

    def test_interrupted_new_collection_cannot_be_packed_as_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);make_pass(p/'capture',hot=True)
            f=p/'capture/frame_000000.json';m=json.loads(f.read_text())
            m['collection_profile']={'name':'native-readable'};f.write_text(json.dumps(m))
            with self.assertRaises(ValueError):dataset.pack_dataset(p/'capture',p/'dataset')
            self.assertFalse((p/'dataset').exists())

    def test_projection_uses_recorded_intrinsics(self):
        k=dict(fx_px=100.,fy_px=100.,cx_px=320.,cy_px=256.,width=640,height=512)
        u,v,visible=dataset.project_target(np.eye(4),[1.,0.,-10.],k)
        self.assertEqual((u,v,visible),(330.,256.,True))
        self.assertEqual(dataset.project_target(np.eye(4),[0.,0.,10.],k),(None,None,False))

    def test_taskless_samples_pack_with_correct_temperature_and_saturation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);make_pass(p/'capture',hot=True)
            result=dataset.pack_dataset(p/'capture',p/'dataset',target_position=[0,0,-10])
            self.assertEqual(result['frames'],1)
            self.assertIsNone(result['task_result'])
            row=json.loads((p/'dataset/index.jsonl').read_text().strip())
            self.assertEqual(row['dn_saturated_pixels'],1)
            self.assertAlmostEqual(row['t_max_c'],650.,places=3)
            self.assertEqual(row['target_u_px'],2.)
            self.assertTrue((p/'capture'/row['rgb_png']).is_file())

    def test_corrupt_sample_is_rejected_before_dataset_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);make_pass(p/'capture',hot=True)
            (p/'capture/frame_000000_dn16.npy').unlink()
            with self.assertRaises(ValueError):dataset.pack_dataset(p/'capture',p/'dataset',target_position=[0,0,-10])
            self.assertFalse((p/'dataset').exists())


if __name__=='__main__':unittest.main()
