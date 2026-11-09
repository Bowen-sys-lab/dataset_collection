"""Create an explicitly derived replay input without rewriting a manual recording."""
import argparse
import copy
import json
import math
import hashlib
from pathlib import Path


def recorded_backend(metadata):
    backend=(metadata.get('task') or {}).get('environment_evidence',{}).get('collection_episode',{}).get('scene')
    if backend not in ('factory','forest'):raise ValueError('No supported recorded Episode backend')
    return backend


def recorded_target_snapshot(metadata):
    import yaml
    task=metadata.get('task') or {}
    episode=task.get('environment_evidence',{}).get('collection_episode',{})
    value=episode.get('target_config_path')
    if not value or not Path(value).is_absolute():raise ValueError('Recorded target snapshot path is required')
    path=Path(value).resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=episode.get('target_config_sha256'):
        raise ValueError('Recorded target snapshot changed or is missing')
    target=yaml.safe_load(path.read_text(encoding='utf-8'))['target']
    positions=[target.get('position_m'),task.get('target_position')]
    if (target.get('prim_path')!=task.get('target_prim_path')
        or any(not isinstance(p,list) or len(p)!=3 or not all(math.isfinite(float(v)) for v in p) for p in positions)
        or math.dist(*positions)>1e-6):raise ValueError('Target snapshot differs from recorded task')
    return path


def validate_recorded_scene_assets(metadata):
    episode=(metadata.get('task') or {}).get('environment_evidence',{}).get('collection_episode',{})
    if not episode.get('source_snapshot_version'):return
    value=episode.get('scene_assets_path')
    if not value or not Path(value).is_absolute():raise ValueError('Recorded scene asset snapshot is required')
    path=Path(value)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=episode.get('scene_assets_sha256'):
        raise ValueError('Scene asset snapshot changed or is missing')
    assets=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(assets,dict) or not assets:raise ValueError('Empty scene asset snapshot')
    for name,digest in assets.items():
        asset=Path(name)
        if not asset.is_absolute() or not asset.is_file() or hashlib.sha256(asset.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Recorded scene asset changed: {name}; restore its original version before replay')


def select_source(records, metadata, interval):
    if not math.isfinite(interval) or interval<=0:raise ValueError('Replay interval must be finite and positive')
    if metadata.get('collision_pending'):
        raise ValueError('Resolve the pending collision before exporting a replay sequence')
    from scripts.trajectory_recorder import build_training_sequence
    if metadata.get('collection_branch_applied'):
        rows=copy.deepcopy(records)
    elif metadata.get('terminal_snapshot') is not None:
        rows=build_training_sequence(records,metadata)
    elif metadata.get('collection_edit_events'):
        from scripts.recording_undo import active_branch
        rows,_=active_branch(records,metadata)
    else:
        rows=copy.deepcopy(records)
    selected=[];previous=-math.inf;next_time=-math.inf
    for index,row in enumerate(rows):
        stamp=float(row['simulation_time'])
        if not math.isfinite(stamp) or stamp<=previous:raise ValueError('Source time must strictly increase')
        previous=stamp
        if stamp>=next_time-1e-8 or index==len(rows)-1:
            row.setdefault('source_frame_id',row['frame_id'])
            row['frame_id']=len(selected)
            selected.append(row);next_time=stamp+interval
    if not selected:raise ValueError('Source recording has no frames')
    return selected


def prepare_source(original, output, interval):
    from scripts.trajectory_recorder import read_trajectory
    from scripts.replay_factory_inspection import write_terminal_replay_source
    original=Path(original).resolve()
    metadata=json.loads((original.parent/'run_metadata.json').read_text(encoding='utf-8'))
    rows=select_source(read_trajectory(original),metadata,interval)
    derived=write_terminal_replay_source(output,original,rows)
    meta_path=derived.parent/'run_metadata.json'
    derived_meta=json.loads(meta_path.read_text(encoding='utf-8'))
    derived_meta.update(derivation='sampled replay input; exact T endpoint only when present; not an independent demonstration',
                        replay_sample_interval_s=interval,source_derivation_schema_version=1)
    if metadata.get('collection_edit_events'):
        derived_meta['collection_branch_applied']=True
    meta_path.write_text(json.dumps(derived_meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return derived


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trajectory',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=1.0)
    args=parser.parse_args()
    print(prepare_source(args.trajectory,args.out,args.interval))


if __name__=='__main__':main()
