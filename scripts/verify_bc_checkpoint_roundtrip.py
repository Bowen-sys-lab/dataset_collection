"""Exactly one isolated diagnostic update, then adapter disk save/load verification."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.verify_bc_training_interface import sha,gpu_processes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-report',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HOME=str(out/'hf-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    report={'status':'RUNNING','pid':os.getpid(),'optimizer_steps':0,'training_ready':False,
            'model_load_count':0,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'qualification':'DIAGNOSTIC_ONE_UPDATE_NOT_FOR_FLIGHT','heldout_examples_consumed':0}
    sources={}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def event(stage):
        report['stage']=stage;save();print(json.dumps({'stage':stage}),flush=True)
    try:
        busy=gpu_processes()
        if busy:
            report.update(status='BLOCKED_GPU_BUSY',gpu_processes=busy);return 2
        event('bind_passed_no_update_evidence')
        native_path=args.native_report.resolve();sources[str(native_path)]=sha(native_path)
        native=json.loads(native_path.read_text(encoding='utf-8'))
        if (native['status']!='PASS_FORWARD_LOSS_BACKWARD_NO_UPDATE' or native['optimizer_steps']!=0
            or native['training_ready'] or not native['source_files_unchanged'] or len(native['cases'])!=6
            or native['trainable_sha256_before']!=native['trainable_sha256_after']):
            raise ValueError('Completed no-update native evidence is required before one-update diagnostic')
        sources.update(native['source_file_sha256'])
        for name,expected in sources.items():
            if sha(name)!=expected:raise ValueError(f'Native input/version changed: {name}')
        import torch
        import numpy as np
        from transformers import AutoModelForVision2Seq,AutoTokenizer,AutoImageProcessor
        from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
        from scripts.aerovla_offline_probe import quantization_config
        from scripts.bc_model_inputs import BCDiagnosticDataset,pack_diagnostic_batch,INPUT_SCHEMA
        from scripts.bc_action_protocol import SCHEMA
        from scripts.bc_training_checks import require_train_run,validate_packed_batch,verify_causal_loss,finish_evidence
        from scripts.bc_checkpoint_checks import state_digest,take_one_diagnostic_step,validate_checkpoint_contract
        for name in ('scripts/verify_bc_checkpoint_roundtrip.py','scripts/bc_checkpoint_checks.py'):
            path=ROOT/name;sources[str(path.resolve())]=sha(path)
            snapshot=out/'implementation'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,snapshot)
        selected=next(row for row in native['examples'] if row['kind']=='MOVE' and row['target']=='pipe')
        dataset=BCDiagnosticDataset(selected['candidate_folder']);require_train_run(dataset.report)
        example=dataset[selected['frame_index']]
        if example['sample_id']!=selected['sample_id'] or example['target_text']!=selected['target_text']:
            raise ValueError('Selected TRAIN example differs from verified native example')
        base=ROOT/'assets/models/aerovla-pilot/openvla-7b'
        adapter=ROOT/'assets/models/aerovla-pilot/aerovla/aero_vla'
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=True)
        processor=AutoImageProcessor.from_pretrained(base,local_files_only=True,trust_remote_code=True)
        packed=pack_diagnostic_batch([example],tokenizer,processor);validate_packed_batch(packed,eos_id=tokenizer.eos_token_id)
        busy=gpu_processes()
        if busy:raise RuntimeError(f'GPU became busy before model load: {busy}')
        torch.manual_seed(7);torch.cuda.reset_peak_memory_stats();event('load_base_once_and_existing_adapter')
        model=AutoModelForVision2Seq.from_pretrained(base,local_files_only=True,trust_remote_code=True,
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0},attn_implementation='eager',
            quantization_config=quantization_config())
        report['model_load_count']=1
        model.resize_token_embeddings(len(tokenizer))
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={'use_reentrant':False})
        model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=True)
        model.config.use_cache=False
        inputs={k:v.to(device='cuda:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                for k,v in packed['model_inputs'].items()}
        def evaluate():
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(**inputs,use_cache=False,output_projector_features=True,return_dict=True)
            check=verify_causal_loss(result.logits,inputs['labels'],result.loss,
                                    patch_tokens=int(result.projector_features.shape[1]))
            logits=result.logits.detach().float().cpu().clone();del result
            return check,logits
        original_state=state_digest(get_peft_model_state_dict(model))
        before_eval,before_logits=evaluate();del before_logits
        event('one_train_example_one_sgd_step')
        model.train();model.base_model.model.vision_backbone.eval();model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(**inputs,use_cache=False,output_projector_features=True,return_dict=True)
        forward_check=verify_causal_loss(output.logits,inputs['labels'],output.loss,
                                         patch_tokens=int(output.projector_features.shape[1]))
        output.loss.backward();del output
        step=take_one_diagnostic_step(model);report.update(step);save()
        model.zero_grad(set_to_none=True)
        after_eval,expected_logits=evaluate();updated_state=state_digest(get_peft_model_state_dict(model))
        if original_state==updated_state:raise ValueError('Adapter state did not change')
        event('save_new_diagnostic_adapter_only')
        checkpoint=out/'checkpoint';model.save_pretrained(checkpoint,safe_serialization=True,save_embedding_layers=False)
        contract={'input_schema':INPUT_SCHEMA,'action_schema':SCHEMA,
            'qualification':'DIAGNOSTIC_ONE_UPDATE_NOT_FOR_FLIGHT','training_ready':False,'optimizer_steps':1,
            'sample_id':example['sample_id'],'split':dataset.report['split'],'start_group':dataset.report['start_group'],
            'adapter_tensor_sha256':updated_state,'base':str(base),'native_report_sha256':sources[str(native_path)]}
        validate_checkpoint_contract(contract)
        (checkpoint/'bc_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
        checkpoint_hashes={str(p.resolve()):sha(p) for p in checkpoint.iterdir() if p.is_file()}
        sources.update(checkpoint_hashes)
        event('reload_adapter_from_disk_on_same_base')
        loaded_contract=json.loads((checkpoint/'bc_contract.json').read_text(encoding='utf-8'))
        validate_checkpoint_contract(loaded_contract)
        model.load_adapter(checkpoint,adapter_name='roundtrip',is_trainable=False,local_files_only=True)
        model.set_adapter('roundtrip');model.eval()
        loaded_state=state_digest(get_peft_model_state_dict(model,adapter_name='roundtrip'))
        if loaded_state!=loaded_contract['adapter_tensor_sha256']:raise ValueError('Disk-reloaded adapter tensor state differs')
        loaded_eval,loaded_logits=evaluate()
        max_error=float((expected_logits-loaded_logits).abs().max())
        if not torch.allclose(expected_logits,loaded_logits,rtol=1e-5,atol=1e-5):
            raise ValueError(f'Reloaded logits differ, max error {max_error}')
        if abs(after_eval['model_loss']-loaded_eval['model_loss'])>1e-6:
            raise ValueError('Reloaded eval loss differs')
        np.savez_compressed(out/'output-comparison.npz',expected=expected_logits.numpy(),reloaded=loaded_logits.numpy())
        torch.cuda.synchronize()
        report.update(sample=selected,input_schema=INPUT_SCHEMA,action_schema=SCHEMA,forward_before_update=forward_check,
            eval_before_update=before_eval,eval_after_update=after_eval,eval_after_reload=loaded_eval,
            original_adapter_tensor_sha256=original_state,updated_adapter_tensor_sha256=updated_state,
            reloaded_adapter_tensor_sha256=loaded_state,reloaded_logits_max_abs_error=max_error,
            checkpoint=str(checkpoint.resolve()),checkpoint_file_sha256=checkpoint_hashes,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            source_file_sha256=sources,learning_or_task_success_certified=False,
            termination_reason='exactly_one_diagnostic_update_disk_roundtrip_verified')
        event('verify_sources_and_saved_files_unchanged')
        finish_evidence(report,sources,'PASS_ONE_UPDATE_SAVE_RELOAD_DIAGNOSTIC')
        return 0
    except Exception as error:
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');return 1
    finally:
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps({k:report.get(k) for k in ('status','stage','error','optimizer_steps','training_ready')}),flush=True)


if __name__=='__main__':raise SystemExit(main())
