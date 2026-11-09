import unittest

try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires independent model environment')
class BCTrainingCheckTests(unittest.TestCase):
    def packed(self):
        return {'input_schema':'factory_front_rgb_pose_v1','action_schema':'factory_pose_action_v1',
                'training_ready':False,'sample_ids':['fixture:0'],'prompt_token_counts':[2],
                'model_inputs':{'input_ids':torch.tensor([[1,8,5,2,0]]),
                    'attention_mask':torch.tensor([[1,1,1,1,0]]),
                    'labels':torch.tensor([[-100,-100,5,2,-100]]),
                    'pixel_values':torch.zeros(1,6,2,2)}}

    def test_held_out_or_unsealed_run_cannot_supply_backward_samples(self):
        from scripts.bc_training_checks import require_train_run
        report={'result':'PASS_POSE_TARGET_CANDIDATES','split':'train','start_group':'B'}
        require_train_run(report)
        for changed in [{'split':'validation'},{'split':'test'},{'start_group':'E'},
                        {'start_group':'F'},{'result':'RUNNING'}]:
            with self.assertRaises(ValueError):require_train_run({**report,**changed})

    def test_prompt_and_padding_are_not_supervised_and_exactly_one_eos_is(self):
        from scripts.bc_training_checks import validate_packed_batch
        self.assertEqual(validate_packed_batch(self.packed(),eos_id=2)['supervised_tokens'],2)
        for mutate in ['prompt','padding','eos','image','empty','attention','token']:
            packed=self.packed();inputs=packed['model_inputs']
            if mutate=='prompt':inputs['labels'][0,0]=1
            if mutate=='padding':inputs['labels'][0,4]=2
            if mutate=='eos':inputs['labels'][0,2]=2;inputs['input_ids'][0,2]=2
            if mutate=='image':inputs['pixel_values'][0,0,0,0]=float('nan')
            if mutate=='empty':inputs['labels'].fill_(-100)
            if mutate=='attention':inputs['attention_mask'][0,1]=0
            if mutate=='token':inputs['labels'][0,2]=7
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                validate_packed_batch(packed,eos_id=2)

    def test_patch_tokens_masked_after_bos_and_causal_loss_matches(self):
        from scripts.bc_training_checks import verify_causal_loss
        labels=self.packed()['model_inputs']['labels']
        torch.manual_seed(7);logits=torch.randn(1,8,11,requires_grad=True)
        expanded=torch.tensor([[-100,-100,-100,-100,-100,5,2,-100]])
        expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,11),expanded[:,1:].reshape(-1))
        checked=verify_causal_loss(logits,labels,expected,patch_tokens=3)
        self.assertAlmostEqual(checked['independent_loss'],float(expected.detach()),places=6)
        self.assertEqual(checked['supervised_tokens'],2)
        with self.assertRaises(ValueError):verify_causal_loss(logits,labels,expected+1,patch_tokens=3)
        with self.assertRaises(ValueError):verify_causal_loss(logits,labels,expected,patch_tokens=2)

    def model(self):
        model=torch.nn.Module()
        model.register_parameter('lora_A',torch.nn.Parameter(torch.ones(2)))
        model.projector=torch.nn.Linear(2,1)
        model.register_parameter('frozen',torch.nn.Parameter(torch.ones(2),requires_grad=False))
        return model

    def test_finite_nonzero_lora_and_projector_gradients_and_unchanged_weights(self):
        from scripts.bc_training_checks import trainable_digest,summarize_gradients
        model=self.model();before=trainable_digest(model)
        (model.lora_A.sum()+model.projector(torch.ones(2)).sum()).backward()
        summary=summarize_gradients(model)
        self.assertGreater(summary['lora_norm'],0)
        self.assertGreater(summary['projector_norm'],0)
        self.assertEqual(summary['frozen_with_grad'],0)
        self.assertEqual(before,trainable_digest(model))
        with torch.no_grad():model.lora_A.add_(1)
        self.assertNotEqual(before,trainable_digest(model))

    def test_nan_missing_zero_or_frozen_gradients_rejected(self):
        from scripts.bc_training_checks import summarize_gradients
        for mode in ['nan','missing','zero','frozen']:
            model=self.model()
            (model.lora_A.sum()+model.projector(torch.ones(2)).sum()).backward()
            if mode=='nan':model.lora_A.grad[0]=float('nan')
            if mode=='missing':model.lora_A.grad=None
            if mode=='zero':model.lora_A.grad.zero_()
            if mode=='frozen':model.frozen.grad=torch.ones(2)
            with self.subTest(mode=mode),self.assertRaises(ValueError):summarize_gradients(model)

    def test_success_only_published_after_source_byte_check(self):
        import hashlib
        from pathlib import Path
        import tempfile
        from scripts.bc_training_checks import finish_evidence
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source';path.write_bytes(b'original')
            seals={str(path):hashlib.sha256(b'original').hexdigest()}
            report={'status':'RUNNING'}
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):finish_evidence(report,seals,'PASS')
            self.assertEqual(report['status'],'RUNNING')
            path.write_bytes(b'original');finish_evidence(report,seals,'PASS')
            self.assertEqual(report['status'],'PASS');self.assertTrue(report['source_files_unchanged'])


if __name__=='__main__':unittest.main()
