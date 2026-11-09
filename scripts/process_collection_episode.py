"""Explicit recorded-Episode replay adapters; no guessing from scene filenames."""
import argparse
import base64
import datetime
import json
from pathlib import Path
import subprocess
import sys

from scripts.episode_replay_source import prepare_source,recorded_backend,recorded_target_snapshot,validate_recorded_scene_assets
from scripts.collection_shell import powershell_executable
ROOT=Path(__file__).resolve().parents[1]


def run_logged(command,log):
    with Path(log).open('wb') as stream:
        code=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT).returncode
    text=Path(log).read_text(encoding='utf-8',errors='replace').lower()
    if code or 'out of gpu memory' in text or 'error_out_of_device_memory' in text:
        raise RuntimeError(f'Stage failed; see {log}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--interval',type=float,default=1.0)
    args=parser.parse_args()
    import math
    if not math.isfinite(args.interval) or args.interval<=0:raise ValueError('Interval must be finite and positive')
    run=args.run.resolve()
    if run.is_file():run=run.parent
    metadata=json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
    backend=recorded_backend(metadata)
    validate_recorded_scene_assets(metadata)
    target_snapshot=recorded_target_snapshot(metadata) if backend=='forest' else None
    config=Path(metadata['scene_config_path']).resolve()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=args.out.resolve() if args.out else ROOT/'validation/episodes'/run.name/stamp
    if out.exists():raise ValueError(f'Output exists: {out}')
    if backend=='factory':
        code=subprocess.run([str(powershell_executable()),'-NoProfile','-ExecutionPolicy','Bypass','-File',
            str(ROOT/'scripts/process_factory_inspection.ps1'),'-Run',str(run),'-Out',str(out),
            '-RouteInterval',str(args.interval),'-ManualOnly'],cwd=ROOT).returncode
        if code:return code
        logs=out/'logs';logs.mkdir(exist_ok=True)
        run_logged([sys.executable,'-B','-m','scripts.episode_dataset','--run',str(run),
                    '--replay',str(out/'replay'),'--out',str(out/'dataset')],logs/'index.log')
        (out/'pipeline_status.json').write_text(json.dumps({'schema_version':1,'run_id':run.name,
            'scene_id':backend,'backend':'factory_manual','status':'COMPLETE',
            'video':None,'index':str(out/'dataset/episode.json'),
            'stages':[{'name':'video','status':'SKIPPED','reason':'manual_only'}]},indent=2),encoding='utf-8')
        return 0
    out.mkdir(parents=True,exist_ok=False)
    logs=out/'logs';logs.mkdir()
    status={'schema_version':1,'run_id':run.name,'scene_id':backend,'backend':'forest_collection',
            'status':'RUNNING','video':None,
            'stages':[{'name':'video','status':'SKIPPED','reason':'manual_only'}],
            'original_trajectory':str(run/'trajectory.jsonl')}
    def save():
        (out/'pipeline_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    save()
    try:
        source=prepare_source(run/'trajectory.jsonl',out/'derived_input',args.interval)
        status['replay_source']=str(source);save()
        replay=out/'replay';dataset=out/'dataset'
        literal=lambda value:"'"+str(value).replace("'","''")+"'"
        arguments=[str(ROOT/'scene_intake/forest_collection.py'),'--','--config',str(config),
            '--target-config',str(target_snapshot),'--trajectory',str(source),
            '--profile','native-readable','--sensor-range','original','--out',str(replay)]
        command=(f"& {literal(ROOT/'scripts/run_isaac_with_clean_env.ps1')} -ScriptPath "
            f"{literal(ROOT/'scene_intake/run_with_local_assets.py')} -ScriptArgs @("
            +','.join(map(literal,arguments))+')')
        encoded=base64.b64encode(command.encode('utf-16-le')).decode('ascii')
        run_logged([str(powershell_executable()),'-NoProfile','-ExecutionPolicy','Bypass','-EncodedCommand',encoded],logs/'replay.log')
        final=replay/'replay_metadata.json'
        if not final.is_file() or json.loads(final.read_text(encoding='utf-8')).get('result')!='PASS':
            raise RuntimeError('Forest replay incomplete; final metadata missing or not PASS')
        status['stages'].append({'name':'replay','status':'PASS'});save()
        steps=[('dataset','scene_intake/forest_dataset.py',['--capture',str(replay),'--out',str(dataset)]),
               ('validation','scene_intake/verify_dataset.py',['--capture',str(replay),'--dataset',str(dataset/'dataset.json'),
                   '--json',str(out/'validation.json')])]
        for name,script,options in steps:
            run_logged([sys.executable,'-B',str(ROOT/script),*options],logs/(name+'.log'))
            status['stages'].append({'name':name,'status':'PASS'});save()
        run_logged([sys.executable,'-B','-m','scripts.episode_dataset','--run',str(run),
                    '--replay',str(replay),'--out',str(dataset)],logs/'index.log')
        status['status']='COMPLETE';save()
        print(f'Done: {out}\nVideo generation skipped (manual_only). Use video.ps1 when needed.')
        return 0
    except Exception as error:
        status.update(status='FAILED',error=str(error));save();raise


if __name__=='__main__':raise SystemExit(main())
