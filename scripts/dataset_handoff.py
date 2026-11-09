"""Portable preflight, integrity checks and human-review scaffolds; no Kit startup."""
import argparse
import csv
import hashlib
import importlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_manifest(root, files):
    root = Path(root).resolve()
    return [{'path': p.relative_to(root).as_posix(), 'size': p.stat().st_size,
             'sha256': sha(p)} for p in sorted({Path(p).resolve() for p in files})]


def verify_manifest(root, manifest):
    root = Path(root).resolve(); issues = []
    for row in manifest:
        path = (root/row['path']).resolve()
        if not path.is_relative_to(root):
            issues.append({'path':row['path'],'reason':'outside_root'})
        elif not path.is_file():
            issues.append({'path':row['path'],'reason':'missing'})
        elif path.stat().st_size != row['size'] or sha(path) != row['sha256']:
            issues.append({'path':row['path'],'reason':'changed'})
    return issues


def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def create_review(run, batch, out):
    run, batch, out = (Path(p).resolve() for p in (run,batch,out))
    sources = [run/'run_metadata.json',run/'trajectory.jsonl',
               batch/'dataset/episode.json',batch/'dataset/frames.jsonl']
    header = json.loads(sources[2].read_text(encoding='utf-8-sig'))
    if header['run_id'] != run.name:
        raise ValueError('RunId does not match the processing batch')
    rows = [json.loads(s) for s in sources[3].read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    if not rows:
        raise ValueError('No indexed frames')
    out.mkdir(parents=True,exist_ok=False)
    review = dict(schema_version=1, annotation_version='human_review_v1',
                  run_id=run.name,processing_batch=str(batch),instruction=header.get('instruction'),
                  scene=header.get('scene'),task_id=header.get('task_id'),split=header.get('split'),
                  sources=[{'path':str(p),'sha256':sha(p)} for p in sources],
                  decision='PENDING',technical_check='UNKNOWN',target_correct='UNKNOWN',
                  terminal_check='UNKNOWN',task_success='UNKNOWN',path_check='UNKNOWN',
                  terminal_source=None,terminal_time_s=None,terminal_original_frame_id=None,
                  evidence=[],issues=[],reason='',annotator='',reviewer='',reviewed_at='',
                  training_ready=False,
                  note='Independent human review. Does not modify or certify the source recording.')
    dump(out/'review.json',review)
    fields=['frame_index','observation_time_s','original_source_frame_id','rgb','thermal',
            'rgb_target_visibility','thermal_target_visibility','identity_confirmed',
            'phase','issue_code','evidence','annotator_note']
    with (out/'frame_review.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row.get(k,'') for k in fields[:5]},
                             'rgb_target_visibility':'NOT_REVIEWED',
                             'thermal_target_visibility':'NOT_REVIEWED',
                             'identity_confirmed':'UNKNOWN','phase':'UNASSIGNED'})
    return review


def review_issues(review):
    issues=[]
    if review.get('decision') not in {'INCLUDE','EXCLUDE','PENDING'}:
        issues.append('decision must be INCLUDE, EXCLUDE or PENDING')
    if review.get('decision') == 'INCLUDE':
        expected={'technical_check':'PASS','target_correct':'YES','terminal_check':'PASS',
                  'task_success':'YES','path_check':'PASS'}
        issues += [f'{k} must be {v} for INCLUDE' for k,v in expected.items() if review.get(k)!=v]
        if (not str(review.get('annotator','')).strip() or not str(review.get('reviewer','')).strip()
                or review.get('annotator')==review.get('reviewer')):
            issues.append('INCLUDE requires two different named reviewers')
        if not review.get('evidence'):
            issues.append('INCLUDE requires evidence references')
        if review.get('issues'):
            issues.append('Resolve issues before INCLUDE; retain resolution history separately')
    if review.get('decision') == 'EXCLUDE' and not str(review.get('reason','')).strip():
        issues.append('EXCLUDE requires a reason')
    return issues


def check_environment(config=None):
    report={'schema_version':1,'checked_at':datetime.now(timezone.utc).isoformat(),
            'project_root':str(ROOT),'python':sys.version,'platform':platform.platform(),
            'checks':[], 'scope':'Software and direct config paths only; no GPU, USD closure or human flight certification.'}
    def check(name, ok, detail):
        report['checks'].append({'name':name,'ok':bool(ok),'detail':str(detail)})
    check('fixed_project_directory',str(ROOT).lower()==r'd:\isaacsim_demo',ROOT)
    isaac=Path(r'D:\isaacsim')
    check('isaac_python',(isaac/'python.bat').is_file(),isaac/'python.bat')
    version=(isaac/'VERSION').read_text().strip() if (isaac/'VERSION').is_file() else ''
    check('isaac_version',version=='6.1.0-rc.26+release.49347.2d230af4.gl',version)
    sys.path.insert(0,str(ROOT/'third_party/isaac-thermal-camera-simulation/src'))
    for name in ['numpy','scipy','pydantic','yaml','imageio','PIL','irsim']:
        try:
            module=importlib.import_module(name)
            check('import_'+name,True,getattr(module,'__version__','imported'))
        except Exception as exc:
            check('import_'+name,False,f'{type(exc).__name__}: {exc}')
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
                              capture_output=True,text=True,timeout=20)
        check('nvidia_smi',result.returncode==0,result.stdout.strip() or result.stderr.strip())
    except (OSError,subprocess.TimeoutExpired) as exc:
        check('nvidia_smi',False,exc)
    for name in ['record.ps1','process.ps1','run_isaac_with_clean_env.ps1',
                 'start_crazyflie_dual_camera_view.py','replay_trajectory_highres.py',
                 'validate_replay_outputs.py','episode_dataset.py']:
        check('entry_'+name,(ROOT/'scripts'/name).is_file(),ROOT/'scripts'/name)
    if config:
        try:
            from scripts.scene_config import load_scene_config
            from scripts.thermal_config import load_thermal_config
            c=load_scene_config(config)
            load_thermal_config(c.sensors.thermal_config)
            check('scene_config',True,c.source_path)
            import yaml
            env=yaml.safe_load(c.scene.thermal_scene.read_text(encoding='utf-8-sig'))
            weather=Path(env['scene']['weather_file'])
            if not weather.is_absolute():
                weather=c.scene.thermal_scene.parent/weather
            check('weather_file',weather.is_file(),weather)
            report['scene']={'stage':str(c.scene.stage),'rgb_camera':c.drone.camera_path,
                             'preview_resolution':[c.sensors.width,c.sensors.height],
                             'task_id':(c.task or {}).get('task_id')}
        except Exception as exc:
            check('scene_config',False,f'{type(exc).__name__}: {exc}')
    report['result']='PASS' if all(c['ok'] for c in report['checks']) else 'FAIL'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    env=subs.add_parser('check'); env.add_argument('--config',type=Path); env.add_argument('--out',type=Path,required=True)
    review=subs.add_parser('init-review')
    for name in ['run','batch','out']: review.add_argument('--'+name,type=Path,required=True)
    verify=subs.add_parser('verify-files'); verify.add_argument('--root',type=Path,required=True); verify.add_argument('--manifest',type=Path,required=True)
    validate=subs.add_parser('verify-review'); validate.add_argument('--review',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='check':
        if args.out.exists(): raise FileExistsError(args.out)
        report=check_environment(args.config); args.out.parent.mkdir(parents=True,exist_ok=True); dump(args.out,report)
        print(json.dumps(report,ensure_ascii=True,indent=2)); return int(report['result']!='PASS')
    if args.command=='init-review':
        create_review(args.run,args.batch,args.out); print(args.out); return 0
    if args.command=='verify-files':
        issues=verify_manifest(args.root,json.loads(args.manifest.read_text(encoding='utf-8-sig'))['files'])
    else:
        data=json.loads(args.review.read_text(encoding='utf-8-sig')); issues=review_issues(data)
        for source in data.get('sources',[]):
            p=Path(source['path'])
            if not p.is_file() or sha(p)!=source['sha256']: issues.append('Source missing or changed: '+str(p))
        if not data.get('sources'): issues.append('Missing source hashes')
        # This check only validates declared values and source integrity, never image truth.
    print(json.dumps({'result':'FAIL' if issues else 'PASS','issues':issues},ensure_ascii=True,indent=2))
    return int(bool(issues))


if __name__=='__main__':
    raise SystemExit(main())
