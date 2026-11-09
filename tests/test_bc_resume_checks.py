import copy
import random
import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires independent model environment')
class ResumeChecks(unittest.TestCase):
    def test_prompt_only_removes_answers_and_padding(self):
        from scripts.bc_resume_checks import prompt_only
        packed={'model_inputs':{'input_ids':torch.tensor([[1,12,90,2,0]]),
                'attention_mask':torch.tensor([[1,1,1,1,0]]),'labels':torch.tensor([[-100,-100,90,2,-100]]),
                'pixel_values':torch.zeros(1,6,2,2)},'prompt_token_counts':[2]}
        actual=prompt_only(packed)
        self.assertEqual(actual['input_ids'].tolist(),[[1,12]])
        self.assertEqual(actual['attention_mask'].tolist(),[[1,1]])
        self.assertNotIn('labels',actual)
        self.assertEqual(packed['model_inputs']['input_ids'].tolist(),[[1,12,90,2,0]])
        packed['model_inputs']['labels'][0,1]=12
        with self.assertRaises(ValueError):prompt_only(packed)

    def test_nested_digest_binds_optimizer_values_and_types(self):
        from scripts.bc_resume_checks import tree_digest
        value={'state':{0:{'moment':torch.tensor([1.,2.])}},'index':1,'ids':['a','b']}
        initial=tree_digest(value)
        self.assertEqual(initial,tree_digest(dict(reversed(list(value.items())))))
        for mutation in ('tensor','index','order','type'):
            changed=copy.deepcopy(value)
            if mutation=='tensor':changed['state'][0]['moment'][0]=3
            if mutation=='index':changed['index']=2
            if mutation=='order':changed['ids'].reverse()
            if mutation=='type':changed['index']=True
            self.assertNotEqual(initial,tree_digest(changed))

    def test_sampler_and_parameter_mapping_are_validated_before_restore(self):
        from scripts.bc_resume_checks import validate_resume_state
        state={'completed_steps':2,'next_example_index':0,'sample_ids':['a','b'],
               'optimizer_parameter_names':['weight'],
               'optimizer':{'param_groups':[{'params':[0]}],
                            'state':{0:{'step':torch.tensor(2.),'exp_avg':torch.ones(2),'exp_avg_sq':torch.ones(2)}}}}
        validate_resume_state(state,['weight'],['a','b'],expected_steps=2)
        for key,value in [('completed_steps',True),('next_example_index',1),
                          ('sample_ids',['b','a']),('optimizer_parameter_names',['bias'])]:
            changed=copy.deepcopy(state);changed[key]=value
            with self.assertRaises(ValueError):validate_resume_state(changed,['weight'],['a','b'],expected_steps=2)
        changed=copy.deepcopy(state);changed['optimizer']['state'][0]['exp_avg'][0]=float('nan')
        with self.assertRaises(ValueError):validate_resume_state(changed,['weight'],['a','b'],expected_steps=2)

    def test_restore_reproduces_actual_adamw_and_random_continuation(self):
        from scripts.bc_resume_checks import capture_rng,restore_rng,restore_optimizer,tree_digest
        random.seed(17);np.random.seed(17);torch.manual_seed(17)
        first=torch.nn.Linear(2,1);opt=torch.optim.AdamW(first.parameters(),lr=.01,foreach=False)
        def step(model,optimizer):
            optimizer.zero_grad(set_to_none=True)
            x=torch.rand(1,2)+random.random()+float(np.random.random())
            loss=model(x).square().sum();loss.backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
            return float(loss.detach())
        step(first,opt)
        weights=copy.deepcopy(first.state_dict())
        saved={**capture_rng(),'completed_steps':1,'next_example_index':1,'sample_ids':['a','b'],
               'optimizer_parameter_names':['weight','bias'],'optimizer':copy.deepcopy(opt.state_dict())}
        expected_loss=step(first,opt);expected_model=tree_digest(first.state_dict());expected_opt=tree_digest(opt.state_dict())
        second=torch.nn.Linear(2,1);second.load_state_dict(weights)
        resumed=torch.optim.AdamW(second.parameters(),lr=.2,foreach=False)
        restore_optimizer(second,resumed,saved,['a','b'],expected_steps=1);restore_rng(saved)
        self.assertEqual(step(second,resumed),expected_loss)
        self.assertEqual(tree_digest(second.state_dict()),expected_model)
        self.assertEqual(tree_digest(resumed.state_dict()),expected_opt)

    def test_generation_must_be_complete_and_obey_existing_action_limits(self):
        from scripts.bc_resume_checks import classify_generation
        self.assertTrue(classify_generation('STOP</s>',eos_emitted=True)['valid'])
        pose=classify_generation('POSE +0.1 -0.2 +0 +0.1 +0.5</s>',eos_emitted=True)
        self.assertTrue(pose['valid']);self.assertEqual(pose['action']['left_m'],-.2)
        for text,eos in [('STOP',False),('STOP extra</s>',True),
                         ('POSE 100 0 0 0 .5</s>',True),('62 49 16</s>',True)]:
            result=classify_generation(text,eos_emitted=eos)
            self.assertFalse(result['valid']);self.assertIsNone(result['action'])


if __name__=='__main__':unittest.main()
