"""Bounded forward/backward engineering diagnostic; no optimizer or checkpoint output."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def gpu_processes():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],
                          capture_output=True,text=True,check=True,timeout=15)
    return [line for line in result.stdout.splitlines() if line.strip()]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--backward',action='store_true',help='Run native forward/backward after CPU preparation; never update weights')
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HOME=str(out/'hf-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    if not args.backward:os.environ['CUDA_VISIBLE_DEVICES']=''
    report={'status':'RUNNING','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'pid':os.getpid(),'batch':str(args.batch.resolve()),'training_ready':False,'optimizer_steps':0,
            'weights_saved':False,'model_load_count':0,'cases':[],
            'scope':'Teacher-forced single-front loss and gradient plumbing only, not learning/task/flight certification'}
    sources={};model=None
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def event(stage):
        report['stage']=stage;save();print(json.dumps({'stage':stage}),flush=True)
    def seal(path,expected=None):
        path=Path(path).resolve();value=sha(path)
        if (expected is not None and value!=expected) or (str(path) in sources and sources[str(path)]!=value):
            raise ValueError(f'Actual source bytes differ: {path}')
        sources[str(path)]=value
        return value
    try:
        if args.backward:
            busy=gpu_processes();report['gpu_processes_before']=busy
            if busy:
                report.update(status='BLOCKED_GPU_BUSY',termination_reason='other_gpu_compute_process');save();return 2
        import torch
        from transformers import AutoTokenizer,AutoImageProcessor
        from scripts.bc_model_inputs import BCDiagnosticDataset,pack_diagnostic_batch,INPUT_SCHEMA
        from scripts.bc_action_protocol import SCHEMA,parse_action,encode_action
        from scripts.bc_training_checks import (require_train_run,validate_packed_batch,verify_causal_loss,
                                                 trainable_digest,summarize_gradients,finish_evidence)
        torch.manual_seed(7)
        event('select_sealed_train_only_examples')
        batch=args.batch.resolve();seal(batch/'inventory.json')
        inventory=json.loads((batch/'inventory.json').read_text(encoding='utf-8'))
        selected=[];selected_targets=set();case_metadata=[]
        for run in inventory['runs']:
            if run.get('status')!='PASS_POSE_TARGET_CANDIDATES' or run.get('split')!='train':continue
            if run['target'] in selected_targets:continue
            folder=batch/run['run_id'];data=BCDiagnosticDataset(folder);require_train_run(data.report)
            for name in ('report.json','samples.jsonl','source_manifest.json'):seal(folder/name)
            manifest=json.loads((folder/'source_manifest.json').read_text(encoding='utf-8'))
            for path,expected in manifest.items():seal(path,expected)
            rows=[json.loads(line) for line in (folder/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
            seen=set()
            for i,row in enumerate(rows):
                kind=parse_action(encode_action(row['pose_target']))['motion_kind']
                if kind in seen:continue
                example=data[i]
                # Teacher text is canonical, EOS-free; the collator appends exactly one EOS.
                if example['target_text']!=encode_action(parse_action(example['target_text'])):
                    raise ValueError('Noncanonical teacher text')
                selected.append(example);seen.add(kind)
                case_metadata.append({'sample_id':example['sample_id'],'kind':kind,'target':run['target'],
                    'split':data.report['split'],'start_group':data.report['start_group'],
                    'candidate_folder':str(folder),'frame_index':i,'source_rgb_sha256':example['source_rgb_sha256'],
                    'prompt':example['prompt'],'target_text':example['target_text']})
                example['image'].save(out/f'input-{len(selected)-1:02d}.png')
                if seen=={'MOVE','HOLD','STOP'}:break
            if seen!={'MOVE','HOLD','STOP'}:raise ValueError('Run does not cover all three required action kinds')
            selected_targets.add(run['target'])
            if selected_targets=={'tank','pipe'}:break
        if selected_targets!={'tank','pipe'}:raise ValueError('TRAIN-only tank and pipe examples required')
        base=ROOT/'assets/models/aerovla-pilot/openvla-7b'
        adapter=ROOT/'assets/models/aerovla-pilot/aerovla/aero_vla'
        prior=ROOT/'validation/diagnostics/factory-autonomous-pilot-20261103/feedback-20261103T164230735Z/model-worker/ready.json'
        seal(prior)
        pinned=json.loads(prior.read_text(encoding='utf-8'))['model_input_files']
        for root in (base,adapter):
            for path in sorted(root.rglob('*')):
                if not path.is_file() or '.cache' in path.parts:continue
                if str(path.resolve()) not in pinned:raise ValueError(f'Unpinned model file: {path}')
                seal(path,pinned[str(path.resolve())])
        code=['scripts/verify_bc_training_interface.py','scripts/bc_training_checks.py','scripts/bc_model_inputs.py',
              'scripts/bc_action_protocol.py','scripts/aerovla_offline_probe.py']
        for name in code:
            seal(ROOT/name);dest=out/'implementation'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,dest)
        event('actual_tokenizer_and_single_front_tensors')
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=True)
        processor=AutoImageProcessor.from_pretrained(base,local_files_only=True,trust_remote_code=True)
        packed=pack_diagnostic_batch(selected,tokenizer,processor)
        packed_summary=validate_packed_batch(packed,eos_id=tokenizer.eos_token_id)
        for i,meta in enumerate(case_metadata):
            labels=packed['model_inputs']['labels'][i];target=labels[labels!=-100].tolist()
            if tokenizer.decode(target[:-1],skip_special_tokens=False).strip()!=meta['target_text']:
                raise ValueError('Actual action tokens do not decode to recorded canonical teacher')
        report.update(input_schema=INPUT_SCHEMA,action_schema=SCHEMA,examples=case_metadata,
                      batch_checks=packed_summary,tensor_shapes={k:list(v.shape) for k,v in packed['model_inputs'].items()},
                      source_file_sha256=sources,held_out_examples_consumed=0)
        if args.backward:
            from transformers import AutoModelForVision2Seq
            from peft import PeftModel,prepare_model_for_kbit_training
            from scripts.aerovla_offline_probe import quantization_config
            busy=gpu_processes()
            if busy:raise RuntimeError(f'GPU became busy before model load; do not interfere: {busy}')
            event('load_pinned_nf4_model_once')
            torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
            model=AutoModelForVision2Seq.from_pretrained(base,local_files_only=True,trust_remote_code=True,
                torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0},attn_implementation='eager',
                quantization_config=quantization_config())
            model.resize_token_embeddings(len(tokenizer))
            model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                                  gradient_checkpointing_kwargs={'use_reentrant':False})
            model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=True)
            model.config.use_cache=False;model.train()
            model.base_model.model.vision_backbone.eval()
            torch.cuda.synchronize()
            report.update(model_load_count=1,load_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name(),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                trainable_tensors=sum(1 for p in model.parameters() if p.requires_grad))
            before=trainable_digest(model);report['trainable_sha256_before']=before
            for i,(example,metadata) in enumerate(zip(selected,case_metadata)):
                event(f'case_{i}_{metadata["kind"]}_forward_backward')
                batch_one=pack_diagnostic_batch([example],tokenizer,processor)
                validate_packed_batch(batch_one,eos_id=tokenizer.eos_token_id)
                inputs={key:value.to(device='cuda:0',dtype=torch.bfloat16 if value.is_floating_point() else value.dtype)
                        for key,value in batch_one['model_inputs'].items()}
                model.zero_grad(set_to_none=True);started=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    outputs=model(**inputs,use_cache=False,output_projector_features=True,return_dict=True)
                patches=int(outputs.projector_features.shape[1])
                loss_check=verify_causal_loss(outputs.logits,inputs['labels'],outputs.loss,patch_tokens=patches)
                outputs.loss.backward();torch.cuda.synchronize()
                gradients=summarize_gradients(model)
                if trainable_digest(model)!=before:raise ValueError('Parameter values changed despite no optimizer')
                report['cases'].append({**metadata,**loss_check,'gradients':gradients,
                    'seconds':time.perf_counter()-started,'weights_unchanged':True,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
                del outputs,inputs;model.zero_grad(set_to_none=True);save()
            report['trainable_sha256_after']=trainable_digest(model)
            success_status='PASS_FORWARD_LOSS_BACKWARD_NO_UPDATE'
            report['termination_reason']='six_train_examples_verified_no_optimizer'
            report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        else:
            if torch.cuda.is_initialized():raise ValueError('CPU preparation unexpectedly initialized CUDA')
            success_status='PASS_CPU_PREPARATION'
            report['termination_reason']='prepare_only_no_large_model_loaded'
        event('verify_source_bytes_unchanged')
        finish_evidence(report,sources,success_status)
        return 0
    except Exception as error:
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}',termination_reason='diagnostic_error')
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        return 1
    finally:
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps({key:report.get(key) for key in ['status','stage','error','optimizer_steps','training_ready']}),flush=True)


if __name__=='__main__':raise SystemExit(main())
