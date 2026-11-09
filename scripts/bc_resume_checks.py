"""Contracts for diagnostic restarts and answer-free generation; no flight control."""
import hashlib
import json
import math
import random


def tree_digest(value):
    """Canonical typed hash of nested optimizer/RNG state, including tensor bytes."""
    import torch
    digest=hashlib.sha256()
    def visit(item):
        kind=type(item)
        if isinstance(item,torch.Tensor):
            digest.update(b'tensor\0')
            visit(str(item.dtype));visit(list(item.shape))
            raw=item.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(str(len(raw)).encode()+b':'+raw)
        elif isinstance(item,dict):
            digest.update(b'dict\0')
            for key in sorted(item,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(item[key])
            digest.update(b'end\0')
        elif kind in (list,tuple):
            digest.update(kind.__name__.encode()+b'\0')
            for child in item:visit(child)
            digest.update(b'end\0')
        elif item is None or kind in (str,int,float,bool):
            if kind is float and not math.isfinite(item):raise ValueError('Nonfinite state scalar')
            digest.update(kind.__name__.encode()+b':'+json.dumps(item,ensure_ascii=True).encode()+b'\0')
        else:raise ValueError(f'Unsupported state type: {kind}')
    visit(value)
    return digest.hexdigest()


def prompt_only(packed):
    """Copy exactly one masked prompt/image; never expose its supervised suffix."""
    import torch
    data=packed['model_inputs'];ids=data['input_ids'];labels=data['labels'];mask=data['attention_mask']
    prefixes=packed['prompt_token_counts']
    if ids.ndim!=2 or len(ids)!=1 or labels.shape!=ids.shape or mask.shape!=ids.shape or len(prefixes)!=1:
        raise ValueError('One training tensor row required')
    n=prefixes[0]
    if (type(n) is not int or not 1<n<ids.shape[1] or not torch.all(labels[0,:n]==-100)
        or labels[0,n].item()==-100 or not torch.all(mask[0,:n]==1)):
        raise ValueError('Prompt boundary overlaps labels or padding')
    return {'input_ids':ids[:,:n].clone(),'attention_mask':mask[:,:n].clone(),
            'pixel_values':data['pixel_values'].clone()}


def validate_resume_state(saved,names,sample_ids,*,expected_steps):
    import torch
    if (type(saved['completed_steps']) is not int or saved['completed_steps']!=expected_steps
        or not sample_ids or len(sample_ids)!=len(set(sample_ids)) or saved['sample_ids']!=sample_ids
        or type(saved['next_example_index']) is not int
        or saved['next_example_index']!=expected_steps%len(sample_ids)
        or not names or len(names)!=len(set(names)) or saved['optimizer_parameter_names']!=names):
        raise ValueError('Optimizer names, sample order, step or next index differs')
    optimizer=saved['optimizer'];indices=[i for g in optimizer['param_groups'] for i in g['params']]
    if len(indices)!=len(names) or len(set(indices))!=len(indices) or set(indices)!=set(optimizer['state']):
        raise ValueError('Optimizer state coverage differs from named parameters')
    for row in optimizer['state'].values():
        if float(row['step'])!=expected_steps:raise ValueError('Adam step differs')
        for key in ('exp_avg','exp_avg_sq'):
            if not isinstance(row[key],torch.Tensor) or not torch.isfinite(row[key]).all():
                raise ValueError('Missing or nonfinite Adam moments')
        if torch.any(row['exp_avg_sq']<0):raise ValueError('Negative Adam squared moment')


def restore_optimizer(model,optimizer,saved,sample_ids,*,expected_steps):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    validate_resume_state(saved,[n for n,p in named],sample_ids,expected_steps=expected_steps)
    actual=[p for group in optimizer.param_groups for p in group['params']]
    if [id(p) for p in actual]!=[id(p) for n,p in named]:raise ValueError('Live optimizer parameter order differs')
    indices=[i for g in saved['optimizer']['param_groups'] for i in g['params']]
    for (_,param),index in zip(named,indices):
        if any(saved['optimizer']['state'][index][key].shape!=param.shape for key in ('exp_avg','exp_avg_sq')):
            raise ValueError('Optimizer moment shape differs from live parameter')
    optimizer.load_state_dict(saved['optimizer'])
    if tree_digest(optimizer.state_dict())!=tree_digest(saved['optimizer']):
        raise ValueError('Optimizer changed while restoring')


def capture_rng():
    import numpy as np
    import torch
    state=np.random.get_state()
    return {'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
            'python_rng':random.getstate(),'numpy_rng':{'kind':state[0],'keys':torch.from_numpy(state[1].copy()),
            'position':state[2],'has_gauss':state[3],'cached_gauss':state[4]}}


def restore_rng(saved):
    import numpy as np
    import torch
    torch.set_rng_state(saved['cpu_rng']);random.setstate(saved['python_rng'])
    state=saved['numpy_rng']
    np.random.set_state((state['kind'],state['keys'].cpu().numpy(),state['position'],state['has_gauss'],state['cached_gauss']))
    if saved['cuda_rng']:
        if len(saved['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA RNG device count differs')
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if tree_digest(capture_rng())!=tree_digest({k:saved[k] for k in capture_rng()}):
        raise ValueError('RNG restore differs')


def classify_generation(text,*,eos_emitted):
    from scripts.bc_action_protocol import parse_action
    try:
        if not eos_emitted:raise ValueError('No final EOS; output may be truncated')
        action=parse_action(text)
        return {'valid':True,'action':action,'error':None}
    except ValueError as error:
        return {'valid':False,'action':None,'error':str(error)}
