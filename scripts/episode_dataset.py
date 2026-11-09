"""Common episode/frame references for both backends, without training certification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from scripts.frame_metadata import FrameMetadataReader
from scripts.trajectory_recorder import read_trajectory,trajectory_provenance


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_episode_index(run,replay,out):
    run=Path(run).resolve();replay=Path(replay).resolve();out=Path(out).resolve()
    if (out/'episode.json').exists() or (out/'frames.jsonl').exists():raise FileExistsError(out)
    metadata=json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
    final=replay/'replay_metadata.json';report=json.loads(final.read_text(encoding='utf-8'))
    if report.get('result')!='PASS':raise ValueError('Only a validated replay may be indexed')
    trajectory=Path(report['trajectory']).resolve()
    provenance=trajectory_provenance(trajectory)
    if report.get('trajectory_provenance')!=provenance:
        raise ValueError('Replay source provenance changed or was not recorded')
    original=run/'trajectory.jsonl';original_provenance=trajectory_provenance(original)
    current=trajectory;seen=set()
    while current!=original:
        if current in seen or len(seen)>=8:raise ValueError('Invalid derived source chain')
        seen.add(current)
        derived=json.loads((current.parent/'run_metadata.json').read_text(encoding='utf-8'))
        value=derived.get('source_trajectory_path')
        if not value:raise ValueError('Replay belongs to another original run')
        parent=Path(value).resolve()
        if (not parent.is_file() or sha(parent)!=derived.get('source_trajectory_sha256')
            or sha(parent.parent/'run_metadata.json')!=derived.get('source_run_metadata_sha256')):
            raise ValueError('Replay original source identity changed')
        current=parent
    if (provenance['task']!=original_provenance['task']
        or provenance['task_result']!=original_provenance['task_result']):
        raise ValueError('Replay labels do not match the original recording')
    sources={row['frame_id']:row for row in read_trajectory(trajectory)}
    reader=FrameMetadataReader();rows=[]
    for frame in sorted(replay.glob('frame_*.json')):
        info=reader.read(frame, fields={'trajectory_frame_id','replay_simulation_time','simulation_time_s'})
        source=sources.get(info.get('trajectory_frame_id'))
        if source is None:raise ValueError('Replay frame has no traceable source record')
        stamp=float(info.get('replay_simulation_time',info.get('simulation_time_s',source['simulation_time'])))
        if (not math.isfinite(stamp) or abs(stamp-float(source['simulation_time']))>1e-8
            or rows and stamp<=rows[-1]['observation_time_s']):raise ValueError('Observation/source time mismatch')
        rgb=frame.with_name(frame.stem+'_rgb.png');thermal=frame.with_name(frame.stem+'_thermal.png')
        if not rgb.is_file() or not thermal.is_file():raise ValueError('Missing observation image')
        rows.append({'frame_index':len(rows),'observation_time_s':stamp,'rgb':str(rgb),'thermal':str(thermal),
            'replay_source_frame_id':source['frame_id'],'original_source_frame_id':source.get('source_frame_id',source['frame_id']),
            'source_event_ref':source.get('source_event_ref'),'sampled_command':source.get('command'),
            **({'source_simulation_time_s':source['source_simulation_time_s']} if 'source_simulation_time_s' in source else {}),
            'action_type':source.get('action_type','MOTION')})
    if not rows:raise ValueError('Replay has no frames')
    task=metadata.get('task') or {};evidence=task.get('environment_evidence') or {}
    episode=evidence.get('collection_episode') or {}
    city=evidence.get('city_collection') or {}
    header={'schema_version':1,'run_id':run.name,'scene':episode.get('scene') or ('city' if city else None),
        'episode':episode.get('episode') or city.get('episode'),
        'split':episode.get('split') or city.get('split'),'task_id':task.get('task_id'),'instruction':task.get('instruction'),
        'task_version':city.get('version'),
        'trajectory_source':metadata.get('trajectory_source'),'original_trajectory':str(run/'trajectory.jsonl'),
        'original_trajectory_sha256':sha(run/'trajectory.jsonl'),'original_metadata_sha256':sha(run/'run_metadata.json'),
        'replay_metadata_sha256':sha(final),'replay_source_provenance':provenance,'frames':len(rows),
        'capture_contract':metadata.get('capture_contract'),
        'replay_capture_contract':report.get('replay_capture_contract'),
        'source_sample_period_s':metadata.get('sample_period_s'),
        'sample_interval_s':report.get('sample_interval_s'),
        'rgb_resolution':report.get('rgb_resolution'), 'thermal_resolution':report.get('thermal_resolution'),
        'manual_task_result':metadata.get('task_result'),'automatic_task_acceptance':'NOT_EVALUATED',
        **({'collection_edit_schema':metadata.get('collection_edit_schema'),
            'collection_edit_events':metadata['collection_edit_events'],
            'qualification':('corrected_manual_flight' if metadata.get('trajectory_source')=='manual_driving'
                             else 'corrected_development_recording'),
            'independent_manual_flights':int(metadata.get('trajectory_source')=='manual_driving'),
            'time_semantics':'observation_time_s is corrected branch time; source_simulation_time_s preserves raw clock'}
           if metadata.get('collection_edit_events') else {}),
        'action_alignment_status':'NOT_CERTIFIED','strict_pixel_sync_verified':False,'training_ready':False,
        'limitations':'Sampled commands are references, not independently verified causal action labels. Review before training.'}
    out.mkdir(parents=True,exist_ok=True)
    (out/'frames.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
    (out/'episode.json').write_text(json.dumps(header,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return header


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','replay','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(write_episode_index(args.run,args.replay,args.out),ensure_ascii=False))


if __name__=='__main__':main()
