"""Single-front candidate inputs and loss masks; diagnostic use, not a training release."""
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np

from scripts.bc_action_protocol import SCHEMA as ACTION_SCHEMA,encode_action,parse_action

INPUT_SCHEMA='factory_front_rgb_pose_v1'


def prepare_observation(image,instruction,relative_start_yaw_rad):
    """Shared input convention for future training and inference adapters."""
    from PIL import Image
    if not isinstance(instruction,str) or not instruction.strip():
        raise ValueError('Nonempty recorded task instruction required')
    if isinstance(relative_start_yaw_rad,bool) or not math.isfinite(relative_start_yaw_rad):
        raise ValueError('Finite current relative heading required')
    heading=math.atan2(math.sin(relative_start_yaw_rad),math.cos(relative_start_yaw_rad))
    front=image.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
    prompt=(f'<image>\n{instruction}\nObservation: front RGB.\n'
            f'Relative start yaw (rad, positive left): {heading:+.9f}\nAction: ')
    return {'image':front,'prompt':prompt,'input_schema':INPUT_SCHEMA,
            'action_schema':ACTION_SCHEMA,'training_ready':False}


def _heading(matrix):
    value=np.asarray(matrix,dtype=float)
    if (value.shape!=(4,4) or not np.isfinite(value).all()
        or not np.allclose(value[:3,:3] @ value[:3,:3].T,np.eye(3),rtol=0,atol=1e-5)
        or abs(np.linalg.det(value[:3,:3])-1.)>1e-5):
        raise ValueError('Candidate body transform is not finite and rigid')
    return math.atan2(value[0,1],value[0,0])


class BCDiagnosticDataset:
    """Read a sealed candidate run. This class cannot confer training qualification."""
    training_ready=False

    def __init__(self,folder):
        self.folder=Path(folder).resolve()
        if any((path/'superseded.json').exists() for path in [self.folder,self.folder.parent]):
            raise ValueError('Superseded candidate batch')
        report=json.loads((self.folder/'report.json').read_text(encoding='utf-8'))
        raw=(self.folder/'samples.jsonl').read_bytes()
        manifest=(self.folder/'source_manifest.json').read_bytes()
        if (report.get('result')!='PASS_POSE_TARGET_CANDIDATES'
            or report.get('action_schema')!='yaw_local_future_pose_v1'
            or hashlib.sha256(raw).hexdigest()!=report['samples_sha256']
            or hashlib.sha256(manifest).hexdigest()!=report['source_manifest_sha256']):
            raise ValueError('Candidate schema/status or actual file hashes are invalid')
        self._manifest=json.loads(manifest.decode('utf-8'))
        self._rows=[json.loads(line) for line in raw.decode('utf-8').splitlines()]
        if not self._rows or len(self._rows)!=report['samples']:
            raise ValueError('Candidate count differs from report')
        self._initial_yaw=_heading(self._rows[0]['body_world_matrix'])
        for index,row in enumerate(self._rows):
            terminal=index==len(self._rows)-1
            if (row['frame_index']!=index or row['run_id']!=report['run_id']
                or row['split']!=report['split'] or row['start_group']!=report['start_group']
                or row['instruction']!=self._rows[0]['instruction']
                or row['pose_target']['stop']!=terminal
                or (terminal and row.get('source_event_ref')!='run_metadata.json#terminal_snapshot')):
                raise ValueError('Candidate identity/split/STOP sequence is inconsistent')
        self.report=report

    def __len__(self):
        return len(self._rows)

    def __getitem__(self,index):
        from PIL import Image
        row=self._rows[index]
        ref=row['images']['rgb']
        path=Path(ref['path']).resolve()
        if self._manifest.get(str(path))!=ref['sha256']:
            raise ValueError('RGB reference is not bound to the source manifest')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=ref['sha256']:
            raise ValueError('RGB bytes changed after candidate preparation')
        relative=_heading(row['body_world_matrix'])-self._initial_yaw
        relative=math.atan2(math.sin(relative),math.cos(relative))
        recorded=row['pose_target']['relative_start_yaw_rad']
        if not math.isfinite(recorded) or abs(math.atan2(math.sin(relative-recorded),math.cos(relative-recorded)))>1e-6:
            raise ValueError('Relative heading differs from current observed body pose')
        with Image.open(BytesIO(raw)) as source:
            example=prepare_observation(source,row['instruction'],relative)
        example.update(target_text=encode_action(row['pose_target']),sample_id=row['sample_id'],
                       source_rgb_sha256=ref['sha256'])
        return example


def pack_diagnostic_batch(examples,tokenizer,processor,*,max_tokens=1024):
    """Build CPU-ready model tensors, masking task/state text and padded positions."""
    import torch
    if not examples or tokenizer.eos_token_id is None:
        raise ValueError('Nonempty examples and an explicit EOS token required')
    pad=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    sequences=[];labels=[];pixels=[];prefix_lengths=[]
    for example in examples:
        if example['input_schema']!=INPUT_SCHEMA or example['action_schema']!=ACTION_SCHEMA:
            raise ValueError('Model input/action schema differs from this adapter')
        parse_action(example['target_text'])
        prefix=tokenizer(example['prompt'],add_special_tokens=True)['input_ids']
        target=tokenizer(example['target_text'],add_special_tokens=False)['input_ids']+[tokenizer.eos_token_id]
        if not prefix or not target or len(prefix)+len(target)>max_tokens:
            raise ValueError('Empty or overlong sequence; no silent truncation')
        sequences.append(prefix+target)
        labels.append([-100]*len(prefix)+target)
        prefix_lengths.append(len(prefix))
        tensor=processor(images=example['image'],return_tensors='pt')['pixel_values']
        if tensor.ndim!=4 or tensor.shape[0]!=1 or not torch.isfinite(tensor).all():
            raise ValueError('Expected one finite front-image tensor')
        pixels.append(tensor.squeeze(0))
    width=max(map(len,sequences))
    ids=torch.full((len(examples),width),pad,dtype=torch.long)
    targets=torch.full_like(ids,-100)
    mask=torch.zeros_like(ids)
    for index,(sequence,label) in enumerate(zip(sequences,labels)):
        ids[index,:len(sequence)]=torch.tensor(sequence,dtype=torch.long)
        targets[index,:len(label)]=torch.tensor(label,dtype=torch.long)
        mask[index,:len(sequence)]=1
    return {'model_inputs':{'input_ids':ids,'attention_mask':mask,'labels':targets,'pixel_values':torch.stack(pixels)},
            'prompt_token_counts':prefix_lengths,'sample_ids':[e['sample_id'] for e in examples],
            'input_schema':INPUT_SCHEMA,'action_schema':ACTION_SCHEMA,'training_ready':False,
            'qualification':'DIAGNOSTIC_ONLY_NOT_TRAINING_RELEASE'}
