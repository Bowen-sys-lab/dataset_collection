"""Engineering checks for bounded TRAIN-only forward/backward diagnostics."""
import hashlib
import math
from pathlib import Path

from scripts.bc_action_protocol import SCHEMA as ACTION_SCHEMA
from scripts.bc_model_inputs import INPUT_SCHEMA


def require_train_run(report):
    if (report.get('result')!='PASS_POSE_TARGET_CANDIDATES' or report.get('split')!='train'
        or report.get('start_group') not in {'A','B','D'}):
        raise ValueError('Backward diagnostic requires a sealed training-group run; held-out data forbidden')


def validate_packed_batch(packed,*,eos_id):
    import torch
    if (packed.get('input_schema')!=INPUT_SCHEMA or packed.get('action_schema')!=ACTION_SCHEMA
        or packed.get('training_ready') is not False):
        raise ValueError('Explicit diagnostic input/action schemas and unqualified status required')
    inputs=packed['model_inputs'];ids=inputs['input_ids'];mask=inputs['attention_mask'];labels=inputs['labels']
    pixels=inputs['pixel_values'];prefixes=packed['prompt_token_counts']
    if (ids.ndim!=2 or ids.shape!=mask.shape or labels.shape!=ids.shape
        or len(prefixes)!=len(ids) or len(packed['sample_ids'])!=len(ids)
        or pixels.ndim!=4 or pixels.shape[0]!=len(ids) or not torch.isfinite(pixels).all()
        or torch.any((mask!=0)&(mask!=1))):
        raise ValueError('Invalid model tensor shapes, attention mask or image values')
    supervised=0
    for i,prefix in enumerate(prefixes):
        length=int(mask[i].sum())
        if (not 1<=prefix<length or not torch.all(mask[i,:length]==1)
            or not torch.all(mask[i,length:]==0) or not torch.all(labels[i,:prefix]==-100)
            or not torch.all(labels[i,length:]==-100)
            or not torch.equal(labels[i,prefix:length],ids[i,prefix:length])):
            raise ValueError('Prompt/padding mask or target token identity differs')
        target=labels[i,prefix:length]
        if target[-1].item()!=eos_id or int((target==eos_id).sum())!=1:
            raise ValueError('Exactly one final supervised EOS required')
        supervised+=len(target)
    return {'supervised_tokens':supervised,'examples':len(ids)}


def verify_causal_loss(logits,text_labels,loss,*,patch_tokens):
    """Independently reconstruct Prismatic labels: visual patches follow BOS."""
    import torch
    if (isinstance(patch_tokens,bool) or not isinstance(patch_tokens,int) or patch_tokens<=0
        or logits.ndim!=3 or text_labels.ndim!=2 or logits.shape[0]!=text_labels.shape[0]
        or logits.shape[1]!=text_labels.shape[1]+patch_tokens):
        raise ValueError('Patch count and text/model output dimensions differ')
    patches=text_labels.new_full((text_labels.shape[0],patch_tokens),-100)
    labels=torch.cat([text_labels[:,:1],patches,text_labels[:,1:]],dim=1)
    shifted=labels[:,1:].contiguous();count=int((shifted!=-100).sum())
    if not count:raise ValueError('No action/EOS supervision remains after causal shift')
    # Detached audit must not create another backward graph or alter model gradients.
    reference=torch.nn.functional.cross_entropy(logits.detach()[:,:-1].float().contiguous().view(-1,logits.shape[-1]),
                                                 shifted.view(-1),ignore_index=-100)
    actual=float(loss.detach());expected=float(reference)
    if not all(map(math.isfinite,[actual,expected])) or not math.isclose(actual,expected,rel_tol=1e-5,abs_tol=1e-5):
        raise ValueError(f'Model causal loss disagrees with independent masked CE: {actual} vs {expected}')
    return {'model_loss':actual,'independent_loss':expected,'supervised_tokens':count,
            'visual_patch_tokens':patch_tokens,'loss_absolute_error':abs(actual-expected)}


def trainable_digest(model):
    import torch
    digest=hashlib.sha256()
    for name,param in model.named_parameters():
        if param.requires_grad:
            digest.update(name.encode());digest.update(str(tuple(param.shape)).encode())
            digest.update(str(param.dtype).encode())
            digest.update(param.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def summarize_gradients(model):
    import torch
    counts={'trainable_tensors':0,'frozen_with_grad':0,'missing_trainable_gradients':0}
    squares={'lora':0.,'projector':0.}
    for name,param in model.named_parameters():
        if not param.requires_grad:
            if param.grad is not None:counts['frozen_with_grad']+=1
            continue
        counts['trainable_tensors']+=1
        if param.grad is None:
            counts['missing_trainable_gradients']+=1;continue
        grad=param.grad.detach()
        if not torch.isfinite(grad).all():raise ValueError(f'Nonfinite trainable gradient: {name}')
        group='lora' if 'lora_' in name else 'projector' if 'projector' in name else None
        if group is None:raise ValueError(f'Unexpected trainable parameter: {name}')
        squares[group]+=float(grad.float().square().sum())
    if (counts['frozen_with_grad'] or counts['missing_trainable_gradients']
        or not all(math.isfinite(v) and v>0 for v in squares.values())):
        raise ValueError(f'Unproven gradient path: {counts}; squared norms {squares}')
    return {**counts,**{f'{key}_norm':math.sqrt(value) for key,value in squares.items()}}


def finish_evidence(report,sources,success_status):
    for name,expected in sources.items():
        with Path(name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=expected:raise ValueError(f'Source changed during diagnostic: {name}')
    report.update(status=success_status,source_files_unchanged=True)
