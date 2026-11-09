"""Single-update diagnostic helpers, not a general training loop."""
import hashlib
import math

from scripts.bc_training_checks import summarize_gradients
from scripts.bc_model_inputs import INPUT_SCHEMA
from scripts.bc_action_protocol import SCHEMA


def state_digest(state):
    import torch
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        if not isinstance(tensor,torch.Tensor):raise ValueError('Tensor-only adapter state required')
        for value in (name,str(tuple(tensor.shape)),str(tensor.dtype)):
            digest.update(value.encode());digest.update(b'\0')
        digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
        digest.update(b'\0')
    return digest.hexdigest()


def take_one_diagnostic_step(model):
    import torch
    gradients=summarize_gradients(model)
    parameters=[p for p in model.parameters() if p.requires_grad]
    before=state_digest({n:p for n,p in model.named_parameters() if p.requires_grad})
    norm=torch.nn.utils.clip_grad_norm_(parameters,max_norm=1.,error_if_nonfinite=True)
    if not math.isfinite(float(norm)):raise ValueError('Nonfinite global gradient norm')
    optimizer=torch.optim.SGD(parameters,lr=1e-3,momentum=0.,foreach=False)
    frozen_ids={id(p) for p in model.parameters() if not p.requires_grad}
    if any(id(p) in frozen_ids for group in optimizer.param_groups for p in group['params']):
        raise ValueError('Frozen parameter entered optimizer')
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in parameters):raise ValueError('Nonfinite updated parameter')
    after=state_digest({n:p for n,p in model.named_parameters() if p.requires_grad})
    if before==after:raise ValueError('Diagnostic optimizer made no numeric parameter change')
    return {'optimizer_steps':1,'optimizer':'SGD','learning_rate':1e-3,'momentum':0.,
            'clip_max_norm':1.,'pre_clip_global_norm':float(norm),'gradients':gradients,
            'trainable_sha256_before':before,'trainable_sha256_after':after,'frozen_parameters_in_optimizer':0}


def validate_checkpoint_contract(metadata):
    if (metadata.get('input_schema')!=INPUT_SCHEMA or metadata.get('action_schema')!=SCHEMA
        or metadata.get('qualification')!='DIAGNOSTIC_ONE_UPDATE_NOT_FOR_FLIGHT'
        or metadata.get('training_ready') is not False or type(metadata.get('optimizer_steps')) is not int
        or metadata['optimizer_steps']!=1):
        raise ValueError('Checkpoint is not the declared one-update diagnostic schema')
