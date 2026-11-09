"""One bounded six-TRAIN-example learning attempt; never a flight-ready model."""
import argparse,datetime,json,os,random,shutil,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.verify_bc_training_interface import sha,gpu_processes


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native-report',type=Path,required=True);ap.add_argument('--cohort-report',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(HF_HOME=str(out/'hf-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    report={'status':'RUNNING','pid':os.getpid(),'training_ready':False,'formal_batch_training':False,
            'optimizer_steps':0,'model_load_count':0,'heldout_backward_examples':0,'steps':[],'evaluations':[],
            'qualification':'DIAGNOSTIC_TINY_FIT_NOT_FOR_FLIGHT','seed':7,
            'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    sources={}
    def save():(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    def event(stage):
        report['stage']=stage;save();print(json.dumps({'stage':stage,'steps':report['optimizer_steps']}),flush=True)
    try:
        busy=gpu_processes()
        if busy:report.update(status='BLOCKED_GPU_BUSY',gpu_processes=busy);return 2
        event('bind_passed_native_and_cohort_inputs')
        native_path=args.native_report.resolve();cohort_path=args.cohort_report.resolve()
        native=json.loads(native_path.read_text(encoding='utf-8'));cohort=json.loads(cohort_path.read_text(encoding='utf-8'))
        if native['status']!='PASS_FORWARD_LOSS_BACKWARD_NO_UPDATE' or len(native['examples'])!=6:
            raise ValueError('Six-example no-update native evidence required')
        if cohort['result']!='PASS_EIGHT_RUN_DATA_AND_DECLARED_STATIC_PATHS':
            raise ValueError('Complete eight-run data and composed native static path check required')
        sources.update(native['source_file_sha256']);sources[str(native_path)]=sha(native_path);sources[str(cohort_path)]=sha(cohort_path)
        for name,expected in sources.items():
            if sha(name)!=expected:raise ValueError(f'Native source/version changed: {name}')
        for name in ('scripts/verify_bc_tiny_fit.py','scripts/bc_tinyfit_checks.py'):
            p=ROOT/name;sources[str(p.resolve())]=sha(p)
            snapshot=out/'implementation'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,snapshot)
        import numpy as np
        import torch
        from transformers import AutoModelForVision2Seq,AutoTokenizer,AutoImageProcessor
        from peft import PeftModel,prepare_model_for_kbit_training,get_peft_model_state_dict
        from scripts.bc_model_inputs import BCDiagnosticDataset,pack_diagnostic_batch,INPUT_SCHEMA
        from scripts.bc_action_protocol import SCHEMA
        from scripts.bc_training_checks import require_train_run,validate_packed_batch,verify_causal_loss,summarize_gradients,finish_evidence
        from scripts.bc_checkpoint_checks import state_digest
        from scripts.bc_tinyfit_checks import assess_fit,summarize_losses
        random.seed(7);np.random.seed(7);torch.manual_seed(7)
        examples=[];cache={}
        for row in native['examples']:
            folder=row['candidate_folder']
            if folder not in cache:cache[folder]=BCDiagnosticDataset(folder)
            dataset=cache[folder];require_train_run(dataset.report)
            item=dataset[row['frame_index']]
            if item['sample_id']!=row['sample_id'] or item['target_text']!=row['target_text']:
                raise ValueError('TRAIN example differs from passed native evidence')
            examples.append(item)
        base=ROOT/'assets/models/aerovla-pilot/openvla-7b';adapter=ROOT/'assets/models/aerovla-pilot/aerovla/aero_vla'
        tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=True)
        processor=AutoImageProcessor.from_pretrained(base,local_files_only=True,trust_remote_code=True)
        packed=[pack_diagnostic_batch([e],tokenizer,processor) for e in examples]
        for batch in packed:validate_packed_batch(batch,eos_id=tokenizer.eos_token_id)
        if gpu_processes():raise RuntimeError('GPU became busy before model load')
        from scripts.aerovla_offline_probe import quantization_config
        torch.cuda.reset_peak_memory_stats();event('load_pinned_model_once')
        model=AutoModelForVision2Seq.from_pretrained(base,local_files_only=True,trust_remote_code=True,
            torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':0},attn_implementation='eager',
            quantization_config=quantization_config());report['model_load_count']=1
        model.resize_token_embeddings(len(tokenizer))
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={'use_reentrant':False})
        model=PeftModel.from_pretrained(model,adapter,local_files_only=True,is_trainable=True);model.config.use_cache=False
        dropout={}
        for name,module in model.named_modules():
            if isinstance(module,torch.nn.Dropout):dropout[name]=module.p;module.p=0.
        report.update(dropout_disabled_for_diagnostic=dropout,examples=native['examples'],
                      input_schema=INPUT_SCHEMA,action_schema=SCHEMA,
                      optimizer={'name':'AdamW','lr':1e-4,'weight_decay':0.,'microbatch':1,'clip_norm':1.,'max_steps':18})
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=0.,foreach=False)
        original_state=state_digest(get_peft_model_state_dict(model))
        inputs=[{k:v.to('cuda:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
                 for k,v in p['model_inputs'].items()} for p in packed]
        def evaluate(step):
            model.eval();rows=[]
            for meta,batch in zip(native['examples'],inputs):
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(**batch,use_cache=False,output_projector_features=True,return_dict=True)
                checked=verify_causal_loss(output.logits,batch['labels'],output.loss,patch_tokens=int(output.projector_features.shape[1]))
                rows.append({**checked,'kind':meta['kind'],'sample_id':meta['sample_id']});del output
            report['evaluations'].append({'completed_steps':step,'rows':rows,'summary':summarize_losses(rows)})
            save();return rows
        event('initial_fixed_train_evaluation');initial=evaluate(0)
        for step in range(18):
            model.train();model.base_model.model.vision_backbone.eval();optimizer.zero_grad(set_to_none=True)
            index=step%6;batch=inputs[index]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(**batch,use_cache=False,output_projector_features=True,return_dict=True)
            check=verify_causal_loss(output.logits,batch['labels'],output.loss,patch_tokens=int(output.projector_features.shape[1]))
            output.loss.backward();del output
            gradients=summarize_gradients(model)
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            report['optimizer_step_attempted']=step+1;optimizer.step();report['optimizer_steps']=step+1
            if not all(torch.isfinite(p).all() for p in trainable):raise ValueError('Nonfinite updated parameter')
            report['steps'].append({'step':step+1,'sample_id':examples[index]['sample_id'],**check,
                                    'pre_clip_norm':float(norm),'gradients':gradients})
            event('bounded_train_step')
            if (step+1)%6==0:
                optimizer.zero_grad(set_to_none=True);final=evaluate(step+1)
        assessment=assess_fit(initial,final,steps=report['optimizer_steps']);report['assessment']=assessment
        optimizer.zero_grad(set_to_none=True);event('save_unqualified_diagnostic_checkpoint')
        checkpoint=out/'checkpoint';model.save_pretrained(checkpoint,safe_serialization=True,save_embedding_layers=False)
        updated_state=state_digest(get_peft_model_state_dict(model))
        if updated_state==original_state:raise ValueError('No numeric adapter change')
        state={'optimizer':optimizer.state_dict(),'optimizer_parameter_names':[n for n,p in model.named_parameters() if p.requires_grad],
               'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),
               'numpy_rng':{'kind':np.random.get_state()[0],'keys':torch.from_numpy(np.random.get_state()[1].copy()),
                            'position':np.random.get_state()[2],'has_gauss':np.random.get_state()[3],'cached_gauss':np.random.get_state()[4]},
               'completed_steps':18,'next_example_index':0,'sample_ids':[e['sample_id'] for e in examples]}
        torch.save(state,checkpoint/'training_state.pt');del state
        contract={'qualification':'DIAGNOSTIC_TINY_FIT_NOT_FOR_FLIGHT','training_ready':False,
                  'input_schema':INPUT_SCHEMA,'action_schema':SCHEMA,'optimizer_steps':18,
                  'adapter_tensor_sha256':updated_state,'dropout_disabled':True,'sample_ids':[e['sample_id'] for e in examples],
                  'full_training_resume_verified':False,'base':str(base)}
        (checkpoint/'bc_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
        files={str(p.resolve()):sha(p) for p in checkpoint.iterdir() if p.is_file()};sources.update(files)
        torch.cuda.synchronize()
        report.update(source_file_sha256=sources,checkpoint_file_sha256=files,adapter_tensor_sha256_before=original_state,
                      adapter_tensor_sha256_after=updated_state,checkpoint=str(checkpoint.resolve()),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      full_training_resume_verified=False,task_success_or_generalization_certified=False)
        event('verify_unchanged_inputs_and_saved_outputs')
        status='PASS_BOUNDED_TRAIN_LEARNABILITY_DIAGNOSTIC' if assessment['passed'] else 'FAIL_PREDECLARED_LEARNABILITY_CRITERION'
        finish_evidence(report,sources,status)
        return 0 if assessment['passed'] else 1
    except Exception as error:
        report.update(status='FAIL',error=f'{type(error).__name__}: {error}')
        (out/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');return 1
    finally:
        report['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps({k:report.get(k) for k in ('status','error','optimizer_steps','training_ready')}),flush=True)


if __name__=='__main__':raise SystemExit(main())
