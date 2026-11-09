import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


class BCModelInputTests(unittest.TestCase):
    def fixture(self, root):
        from PIL import Image
        root=root.resolve()
        manifest={};rows=[]
        for i,color in enumerate([(255,0,0),(0,0,255)]):
            path=root/f'{i}.png';Image.new('RGB',(80,60),color).save(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest();manifest[str(path)]=digest
            yaw=0. if i==0 else .2;c,s=math.cos(yaw),math.sin(yaw)
            label={'forward_m':.1 if i==0 else 0.,'left_m':0.,'up_m':0.,
                   'yaw_delta_rad':.2 if i==0 else 0.,'duration_s':.5 if i==0 else 0.,
                   'relative_start_yaw_rad':yaw,'motion_kind':'MOVE' if i==0 else 'STOP','stop':i==1}
            rows.append({'sample_id':f'fixture:{i:06d}','run_id':'fixture','frame_index':i,
                         'instruction':'Find the tank.','split':'train','start_group':'A',
                         'images':{'rgb':{'path':str(path),'sha256':digest},'thermal':{'path':'NEVER_READ'}},
                         'body_world_matrix':[[c,s,0,0],[-s,c,0,0],[0,0,1,0],[.1*i,0,3,1]],
                         'source_event_ref':'run_metadata.json#terminal_snapshot' if i else None,
                         'pose_target':label,'future_endpoint':{'marker':'FUTURE_MUST_NOT_ENTER_PROMPT'}})
        (root/'samples.jsonl').write_bytes(''.join(json.dumps(r)+'\n' for r in rows).encode())
        (root/'source_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        self.reseal(root)
        return rows

    def reseal(self,root):
        report={'result':'PASS_POSE_TARGET_CANDIDATES','training_ready':False,'samples':2,'run_id':'fixture',
                'split':'train','start_group':'A','action_schema':'yaw_local_future_pose_v1',
                'samples_sha256':hashlib.sha256((root/'samples.jsonl').read_bytes()).hexdigest(),
                'source_manifest_sha256':hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest()}
        (root/'report.json').write_text(json.dumps(report),encoding='utf-8')

    def test_one_front_rgb_and_current_relative_heading_without_future_leakage(self):
        from scripts.bc_model_inputs import BCDiagnosticDataset
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            data=BCDiagnosticDataset(root)
            first,stop=data[0],data[1]
            self.assertEqual(first['image'].size,(224,224))
            self.assertEqual(first['image'].getpixel((0,0)),(255,0,0))
            self.assertEqual(stop['image'].getpixel((0,0)),(0,0,255))
            self.assertIn('+0.200000000',stop['prompt'])
            self.assertNotIn('FUTURE_MUST_NOT_ENTER_PROMPT',first['prompt'])
            self.assertEqual(stop['target_text'],'STOP')
            self.assertFalse(first['training_ready'])
            self.assertEqual(first['input_schema'],'factory_front_rgb_pose_v1')

    def test_image_rewrite_after_dataset_open_is_rejected(self):
        from scripts.bc_model_inputs import BCDiagnosticDataset
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);data=BCDiagnosticDataset(root)
            (root/'0.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                data[0]

    def test_sample_hash_and_independently_derived_heading_are_checked(self):
        from scripts.bc_model_inputs import BCDiagnosticDataset
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=self.fixture(root)
            (root/'samples.jsonl').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                BCDiagnosticDataset(root)
            rows[1]['pose_target']['relative_start_yaw_rad']=1.
            (root/'samples.jsonl').write_bytes(''.join(json.dumps(r)+'\n' for r in rows).encode())
            self.reseal(root)
            with self.assertRaises(ValueError):
                BCDiagnosticDataset(root)[1]

    def test_original_schema_or_premature_stop_is_rejected(self):
        from scripts.bc_model_inputs import BCDiagnosticDataset
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=self.fixture(root)
            report=json.loads((root/'report.json').read_text());report['action_schema']='aerovla_original'
            (root/'report.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                BCDiagnosticDataset(root)
            rows[0]['pose_target'].update(forward_m=0.,yaw_delta_rad=0.,duration_s=0.,stop=True,motion_kind='STOP')
            (root/'samples.jsonl').write_bytes(''.join(json.dumps(r)+'\n' for r in rows).encode())
            self.reseal(root)
            with self.assertRaises(ValueError):
                BCDiagnosticDataset(root)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Requires independent model environment')
    def test_diagnostic_packing_masks_prompt_and_padding_and_retains_stop_eos(self):
        import torch
        from scripts.bc_model_inputs import BCDiagnosticDataset,pack_diagnostic_batch
        class Tokenizer:
            eos_token_id=2
            pad_token_id=0
            def __call__(self,text,add_special_tokens=True):
                return {'input_ids':([1] if add_special_tokens else [])+[ord(c)+10 for c in text]}
        class Processor:
            def __call__(self,*,images,return_tensors):
                return {'pixel_values':torch.tensor(images.getpixel((0,0)),dtype=torch.float32)[None,:,None,None]}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root);data=BCDiagnosticDataset(root)
            packed=pack_diagnostic_batch([data[0],data[1]],Tokenizer(),Processor())
            inputs=packed['model_inputs']
            self.assertFalse(packed['training_ready'])
            for index in range(2):
                n=packed['prompt_token_counts'][index];length=int(inputs['attention_mask'][index].sum())
                self.assertTrue(torch.all(inputs['labels'][index,:n]==-100))
                self.assertEqual(inputs['labels'][index,length-1].item(),2)
                self.assertTrue(torch.all(inputs['labels'][index,length:]==-100))
            self.assertEqual(inputs['pixel_values'].shape,(2,3,1,1))
            stop_tokens=inputs['labels'][1][inputs['labels'][1]!=-100].tolist()
            self.assertEqual(stop_tokens,[ord(c)+10 for c in 'STOP']+[2])


if __name__=='__main__':
    unittest.main()
