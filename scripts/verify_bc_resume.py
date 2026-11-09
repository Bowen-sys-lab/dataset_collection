"""Bounded fresh-process restart and TRAIN-only free-generation diagnostic."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.verify_bc_training_interface import sha,gpu_processes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tinyfit-report',type=Path,required=True)
    ap.add_argument('--reference',type=Path,help='Reference process report; omit for process A')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HOME=str(out/'hf-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    report={'status':'RUNNING','pid':os.getpid(),'mode':'resume' if args.reference else 'reference',
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'training_ready':False,'flight_ready':False,'optimizer_steps_this_process':0,
            'model_load_count':0,'heldout_examples':0,'generations':[],'steps':[]}
    sources={}
    def save():
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def event(stage):
        report['stage']=stage;save();print(json.dumps({'stage':stage,'updates':report['optimizer_steps_this_process']}),flush=True)
    def seal(path,expected=None):
        p=Path(path).resolve();actual=sha(p)
        if expected is not None and actual!=expected:raise ValueError(f'Source changed: {p}')
        if str(p) in sources and sources[str(p)]!=actual:raise ValueError(f'Conflicting source: {p}')
        sources[str(p)]=actual
    try:
        if gpu_processes():report['status']='BLOCKED_GPU_BUSY';return 2
        event('bind_sources')
        tiny_path=args.tinyfit_report.resolve();tiny=json.loads(tiny_path.read_text(encoding='utf-8'))
        if tiny['status']!='PASS_BOUNDED_TRAIN_LEARNABILITY_DIAGNOSTIC' or tiny['optimizer_steps']!=18:
            raise ValueError('Passed18-step diagnostic required')
        for path,expected in tiny['source_file_sha256'].items():seal(path,expected)
        seal(tiny_path)
        reference=None
        if args.reference:
            reference_path=args.reference.resolve();reference=json.loads(reference_path.read_text(encoding='utf-8'))
            if (reference['status']!='REFERENCE_COMPLETE' or reference['mode']!='reference'
                or reference['pid']==os.getpid()):raise ValueError('Completed independent reference process required')
            for path,expected in reference['source_file_sha256'].items():seal(path,expected)
            seal(reference_path);checkpoint=Path(reference['checkpoint19']);expected_steps=19
        else:checkpoint=Path(tiny['checkpoint']);expected_steps=18
        for name in ('scripts/verify_bc_resume.py','scripts/bc_resume_checks.py','scripts/bc_checkpoint_checks.py',
                     'scripts/bc_training_checks.py','scripts/bc_model_inputs.py','scripts/bc_action_protocol.py',
                     'scripts/verify_bc_training_interface.py','scripts/aerovla_offline_probe.py'):
            p=ROOT/name;seal(p);snapshot=out/'implementation'/name
            snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,snapshot)
        import numpy as np
        import torch
        from transformers import AutoModelForVision2Seq,AutoTokenizer,AutoImageProcessor
        from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
        from scripts.bc_model_inputs import BCDiagnosticDataset,pack_diagnostic_batch,INPUT_SCHEMA
        from scripts.bc_action_protocol import SCHEMA
        from scripts.bc_training_checks import require_train_run,validate_packed_batch,verify_causal_loss,summarize_gradients,finish_evidence
        from scripts.bc_checkpoint_checks import state_digest
        from scripts.bc_resume_checks import (tree_digest,prompt_only,restore_optimizer,capture_rng,restore_rng,classify_generation)
        from scripts.aerovla_offline_probe import quantization_config
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        examples=[];datasets={}
        for meta in tiny['examples']:
            folder=meta['candidate_folder']
            if folder not in datasets:datasets[folder]=BCDiagnosticDataset(folder)
            dataset=datasets[folder];require_train_run(dataset.report);item=dataset[meta['frame_index']]
            if item['sample_id']!=meta['sample_id'] or item['target_text']!=meta['target_text']:
                raise ValueError('TRAIN sample differs from prior diagnostic')
            examples.append(item)
        ids=[e['sample_id'] for e in examples]
        contract=json.loads((checkpoint/'bc_contract.json').read_text(encoding='utf-8'))
        if (contract['sample_ids']!=ids or contract['optimizer_steps']!=expected_steps
            or contract['input_schema']!=INPUT_SCHEMA or contract['action_schema']!=SCHEMA
            or contract['training_ready'] is not False or contract['dropout_disabled'] is not True
            or contract['qualification'] not in {'DIAGNOSTIC_TINY_FIT_NOT_FOR_FLIGHT','DIAGNOSTIC_RESTART_NOT_FOR_FLIGHT'}):
            raise ValueError('Checkpoint schema or qualification differs')
        base=Path(contract['base'])
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=True)
        processor=AutoImageProcessor.from_pretrained(base,local_files_only=True,trust_remote_code=True)
        packed=[pack_diagnostic_batch([e],tokenizer,processor) for e in examples]
        for batch in packed:validate_packed_batch(batch,eos_id=tokenizer.eos_token_id)
        saved=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=True)
        if gpu_processes():raise RuntimeError('GPU became busy before model load')
        torch.cuda.reset_peak_memory_stats();event('load_model_once')
        model=AutoModelForVision2Seq.from_pretrained(base,local_files_only=True,trust_remote_code=True,
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0},attn_implementation='eager',
            quantization_config=quantization_config());report['model_load_count']=1
        model.resize_token_embeddings(len(tokenizer))
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={'use_reentrant':False})
        model=PeftModel.from_pretrained(model,checkpoint,local_files_only=True,is_trainable=True);model.config.use_cache=False
        for module in model.modules():
            if isinstance(module,torch.nn.Dropout):module.p=0.
        initial_adapter=state_digest(get_peft_model_state_dict(model))
        if initial_adapter!=contract['adapter_tensor_sha256']:raise ValueError('Loaded adapter differs from sealed tensor state')
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=0.,foreach=False)
        restore_optimizer(model,optimizer,saved,ids,expected_steps=expected_steps)
        report.update(resumed_from_step=expected_steps,input_schema=INPUT_SCHEMA,action_schema=SCHEMA,
                      sample_ids=ids,initial_adapter_sha256=initial_adapter,
                      restored_optimizer_sha256=tree_digest(optimizer.state_dict()),trainable_tensors=len(trainable))
        def cuda(data):
            return {k:v.to('cuda:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype) for k,v in data.items()}
        inputs=[cuda(p['model_inputs']) for p in packed]
        if reference is None:
            event('prompt_only_greedy_generation_at18');model.eval()
            for example,meta,p in zip(examples,tiny['examples'],packed):
                prompt=prompt_only(p);native=cuda(prompt);start=time.perf_counter()
                # No labels, teacher actions, target coordinates or guided decoding are passed.
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model.generate(**native,do_sample=False,num_beams=1,max_new_tokens=80,use_cache=True,
                        eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                prefix=native['input_ids'].shape[1]
                if not torch.equal(output[:,:prefix],native['input_ids']):raise ValueError('Generation returned a different prompt')
                tokens=output[0,prefix:].cpu().tolist();text=tokenizer.decode(tokens,skip_special_tokens=False)
                classification=classify_generation(text,eos_emitted=bool(tokens and tokens[-1]==tokenizer.eos_token_id))
                report['generations'].append({'sample_id':example['sample_id'],'kind':meta['kind'],
                    'prompt':example['prompt'],'source_rgb_sha256':example['source_rgb_sha256'],
                    'prompt_token_ids':prompt['input_ids'][0].tolist(),'generated_token_ids':tokens,
                    'generated_text':text,'expected_text_for_analysis_only':example['target_text'],
                    'answer_tokens_supplied':0,'elapsed_s':time.perf_counter()-start,**classification})
                del output;save()
            if state_digest(get_peft_model_state_dict(model))!=initial_adapter:
                raise ValueError('Generation changed weights')
            report['generation_summary']={'valid':sum(r['valid'] for r in report['generations']),
                                           'total':len(examples),'train_examples_only':True,
                                           'heldout_or_flight_performance':False}
        restore_rng(saved);report['restored_rng_sha256']=tree_digest(capture_rng());del saved
        def take_step(completed):
            index=completed%len(examples);batch=inputs[index]
            model.train();model.base_model.model.vision_backbone.eval();optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(**batch,use_cache=False,output_projector_features=True,return_dict=True)
            loss=verify_causal_loss(output.logits,batch['labels'],output.loss,patch_tokens=int(output.projector_features.shape[1]))
            output.loss.backward();del output
            gradients=summarize_gradients(model);norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            report['optimizer_step_attempted']=completed+1;save()
            optimizer.step();report['optimizer_steps_this_process']+=1;save()
            optimizer.zero_grad(set_to_none=True)
            if not all(torch.isfinite(p).all() for p in trainable):raise ValueError('Nonfinite updated weight')
            row={'completed_steps':completed+1,'next_example_index':(completed+1)%len(examples),
                 'sample_id':ids[index],**loss,'gradients':gradients,'pre_clip_norm':float(norm),
                 'adapter_sha256':state_digest(get_peft_model_state_dict(model)),
                 'optimizer_sha256':tree_digest(optimizer.state_dict()),'rng_sha256':tree_digest(capture_rng())}
            report['steps'].append(row);event('diagnostic_update_complete');return row
        if reference is None:
            take_step(18);event('save_new_step19_checkpoint')
            new=out/'checkpoint19';model.save_pretrained(new,safe_serialization=True,save_embedding_layers=False)
            state={**capture_rng(),'optimizer':optimizer.state_dict(),
                   'optimizer_parameter_names':[n for n,p in model.named_parameters() if p.requires_grad],
                   'completed_steps':19,'next_example_index':1,'sample_ids':ids}
            torch.save(state,new/'training_state.pt');del state
            meta={**contract,'qualification':'DIAGNOSTIC_RESTART_NOT_FOR_FLIGHT','optimizer_steps':19,
                  'adapter_tensor_sha256':report['steps'][0]['adapter_sha256'],'full_training_resume_verified':False}
            (new/'bc_contract.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
            report['checkpoint19']=str(new.resolve())
            for p in new.iterdir():
                if p.is_file():seal(p)
            # Saving may not consume any RNG: enforce continuity against the pre-save state.
            if tree_digest(capture_rng())!=report['steps'][0]['rng_sha256']:raise ValueError('Saving consumed RNG')
            report['reference_step20']=take_step(19);success='REFERENCE_COMPLETE'
        else:
            actual=take_step(19);expected=reference['reference_step20']
            keys=('completed_steps','next_example_index','sample_id','model_loss','independent_loss',
                  'supervised_tokens','adapter_sha256','optimizer_sha256','rng_sha256')
            comparisons={key:actual[key]==expected[key] for key in keys};report['exact_restart_comparisons']=comparisons
            if not all(comparisons.values()):raise ValueError(f'Fresh-process parity failed: {comparisons}')
            report['reference_pid']=reference['pid'];report['reference_report']=str(args.reference.resolve())
            success='PASS_FRESH_PROCESS_DIAGNOSTIC_RESTART'
        torch.cuda.synchronize()
        report.update(source_file_sha256=sources,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),formal_training=False)
        event('verify_unchanged_sources');finish_evidence(report,sources,success);return 0
    except Exception as error:
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}',source_file_sha256=sources)
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');return 1
    finally:
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps({k:report.get(k) for k in ('status','error','mode','optimizer_steps_this_process')}),flush=True)


if __name__=='__main__':raise SystemExit(main())
