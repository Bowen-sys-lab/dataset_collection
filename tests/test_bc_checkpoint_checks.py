import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires independent model environment')
class CheckpointCheckTests(unittest.TestCase):
    def test_state_digest_binds_names_shapes_dtypes_values(self):
        from scripts.bc_checkpoint_checks import state_digest
        source={'x':torch.tensor([1.,2.])};initial=state_digest(source)
        self.assertEqual(initial,state_digest({'x':source['x'].clone()}))
        for changed in [{'y':source['x']},{'x':source['x'].view(1,2)},
                        {'x':source['x'].double()},{'x':torch.tensor([1.,3.])}]:
            self.assertNotEqual(initial,state_digest(changed))

    def model(self):
        model=torch.nn.Module();model.lora_A=torch.nn.Parameter(torch.ones(2))
        model.projector=torch.nn.Linear(2,1);model.frozen=torch.nn.Parameter(torch.ones(2),requires_grad=False)
        return model

    def test_one_step_changes_trainable_only_and_reports_clipping(self):
        from scripts.bc_checkpoint_checks import take_one_diagnostic_step,state_digest
        model=self.model();frozen=model.frozen.clone();before=state_digest(dict(model.named_parameters()))
        (model.lora_A.sum()+model.projector(torch.ones(2)).sum()).backward()
        result=take_one_diagnostic_step(model)
        self.assertEqual(result['optimizer_steps'],1)
        self.assertGreater(result['pre_clip_global_norm'],0)
        self.assertNotEqual(before,state_digest(dict(model.named_parameters())))
        self.assertTrue(torch.equal(frozen,model.frozen))

    def test_nonfinite_gradient_cannot_update(self):
        from scripts.bc_checkpoint_checks import take_one_diagnostic_step,state_digest
        model=self.model();before=state_digest(dict(model.named_parameters()))
        (model.lora_A.sum()+model.projector(torch.ones(2)).sum()).backward();model.lora_A.grad[0]=float('nan')
        with self.assertRaises(ValueError):take_one_diagnostic_step(model)
        self.assertEqual(before,state_digest(dict(model.named_parameters())))

    def test_checkpoint_contract_is_diagnostic_only_and_schema_specific(self):
        from scripts.bc_checkpoint_checks import validate_checkpoint_contract
        meta={'input_schema':'factory_front_rgb_pose_v1','action_schema':'factory_pose_action_v1',
              'qualification':'DIAGNOSTIC_ONE_UPDATE_NOT_FOR_FLIGHT','training_ready':False,'optimizer_steps':1}
        validate_checkpoint_contract(meta)
        for changed in [{'training_ready':True},{'optimizer_steps':0},{'optimizer_steps':True},
                        {'action_schema':'aerovla_bins'},{'input_schema':'duplicate_front'},
                        {'qualification':'TRAINED'}]:
            with self.assertRaises(ValueError):validate_checkpoint_contract({**meta,**changed})


if __name__=='__main__':unittest.main()
