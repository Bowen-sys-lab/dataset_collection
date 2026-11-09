"""Prepare traceable future-pose BC candidates without certifying task success."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bc_pose_targets import future_pose_target, select_future_endpoint, validate_pose, wrap_yaw


def validate_frame_pose(frame, state, rendered, *, index, source_provenance):
    validate_pose(state)
    if (frame['frame_index'] != index or frame['replay_source_frame_id'] != state['frame_id']
        or frame.get('original_source_frame_id') != state.get('source_frame_id')
        or frame.get('source_event_ref') != state.get('source_event_ref')
        or frame['action_type'] != state['action_type']
        or rendered['trajectory_frame_id'] != state['frame_id'] or rendered.get('frame_id') != index
        or rendered.get('trajectory_sha256') != source_provenance['trajectory_sha256']
        or rendered.get('task_id') != source_provenance['task']['task_id']
        or rendered.get('task_result_status') != source_provenance['task_result']['status']):
        raise ValueError('Frame action/source identity differs from canonical recording')
    for stamp in (frame['observation_time_s'], rendered['replay_simulation_time']):
        if not math.isfinite(stamp) or abs(stamp-state['simulation_time']) > 1e-8:
            raise ValueError('Observation time differs from canonical recording')
    for actual, expected in [('restored_body_world_matrix','recorded_body_world_matrix'),
                             ('camera_world_matrix','rgb_camera_world_matrix'),
                             ('thermal_camera_world_matrix','thermal_camera_world_matrix')]:
        a,b=np.asarray(rendered.get(actual),dtype=float),np.asarray(state.get(expected),dtype=float)
        if (a.shape != (4,4) or b.shape != (4,4) or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not np.allclose(a,b,rtol=0,atol=1e-5)):
            raise ValueError(f'Observation {actual} differs from recorded pose')


def prepare_candidates(run, replay, out, *, horizon_s=.5):
    from scripts.frame_metadata import FrameMetadataReader
    from scripts.model_observation import validate_first_observation
    from scripts.trajectory_recorder import build_training_sequence
    run,replay,out = Path(run).resolve(),Path(replay).resolve(),Path(out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    report = {'result':'RUNNING','training_ready':False,'samples':0}
    sources = {}

    def read(path):
        raw=Path(path).read_bytes()
        seal(path,raw=raw)
        return json.loads(raw.decode('utf-8'))

    def seal(path, *, raw=None):
        path=Path(path).resolve()
        if raw is None:
            with path.open('rb') as stream:
                value=hashlib.file_digest(stream,'sha256').hexdigest()
        else:
            value=hashlib.sha256(raw).hexdigest()
        if str(path) in sources and sources[str(path)] != value:
            raise ValueError(f'Source changed while exporting: {path}')
        sources[str(path)]=value
        return value

    def read_lines(path):
        raw=Path(path).read_bytes()
        seal(path,raw=raw)
        rows=[json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        if not rows or any(not isinstance(row,dict) for row in rows):
            raise ValueError('Expected nonempty JSONL objects')
        return rows

    def preseal_frame(path):
        raw=read(path)
        ref=raw.get('common_metadata_ref')
        if ref is not None:
            if not isinstance(ref,str) or not ref or '/' in ref or '\\' in ref or ':' in ref:
                raise ValueError('Invalid common metadata reference')
            common=path.parent/ref
            if str(common.resolve()) not in sources:
                seal(common)
        return raw

    try:
        metadata=read(run/'run_metadata.json')
        raw_records=read_lines(run/'trajectory.jsonl')
        ids=[r.get('frame_id') for r in raw_records]
        if any(type(value) is not int for value in ids) or len(set(ids)) != len(ids):
            raise ValueError('Raw source frame IDs must be unique integers')
        states=build_training_sequence(raw_records,metadata)
        # Validate all canonical records, including intervals skipped in the replay.
        for state in states:
            validate_pose(state)
        select_future_endpoint(states,0,horizon_s=horizon_s,sample_period_s=metadata.get('sample_period_s',.1))
        lookup={state['frame_id']:state for state in states}
        header=read(replay.parent/'dataset/episode.json')
        frames=read_lines(replay.parent/'dataset/frames.jsonl')
        replay_metadata=read(replay/'replay_metadata.json')
        source_provenance=replay_metadata['trajectory_provenance']
        source=Path(replay_metadata['trajectory'])
        if not source.is_absolute():
            source=replay/source
        seen=set()
        while True:
            source=source.resolve()
            if source in seen or len(seen)>=8:
                raise ValueError('Invalid derived source chain')
            seen.add(source)
            seal(source)
            derived=read(source.parent/'run_metadata.json')
            if not derived.get('source_trajectory_path') or derived.get('source_derivation_schema_version') != 1:
                break
            source=Path(derived['source_trajectory_path'])
        episode=metadata['task']['environment_evidence']['collection_episode']
        if (header['trajectory_source'] != 'manual_driving' or header['run_id'] != run.name
            or header['instruction'] != metadata['task']['instruction'] or not header['instruction'].strip()
            or header['scene'] != 'factory' or header['split'] != episode['split']
            or len(frames) != header['frames'] or not frames):
            raise ValueError('Index/header differs from the manual task and split')
        preseal_frame(replay/'frame_000000.json')
        validate_first_observation(run,Path(frames[0]['rgb']))
        for path in [run/'run_metadata.json',run/'trajectory.jsonl',replay/'replay_metadata.json',
                     replay.parent/'dataset/episode.json',replay.parent/'dataset/frames.jsonl']:
            seal(path)
        reader=FrameMetadataReader()
        fields={'frame_id','trajectory_sha256','task_id','task_result_status',
                'trajectory_frame_id','replay_simulation_time','restored_body_world_matrix',
                'camera_world_matrix','thermal_camera_world_matrix'}
        samples=[]
        previous=-math.inf
        for index,frame in enumerate(frames):
            state=lookup.get(frame['replay_source_frame_id'])
            if state is None:
                raise ValueError('Indexed source frame is not in the canonical pre-T sequence')
            if frame['observation_time_s'] <= previous:
                raise ValueError('Indexed frame times must strictly increase')
            previous=frame['observation_time_s']
            frame_path=replay/f'frame_{index:06d}.json'
            preseal_frame(frame_path)
            rendered=reader.read(frame_path,fields=fields)
            validate_frame_pose(frame,state,rendered,index=index,source_provenance=source_provenance)
            seal(frame_path)
            image_refs={}
            for stream in ['rgb','thermal']:
                image=Path(frame[stream]).resolve()
                expected=replay/f'frame_{index:06d}_{stream}.png'
                if image != expected:
                    raise ValueError('Image reference does not belong to this replay frame')
                image_refs[stream]={'path':str(image),'sha256':seal(image)}
            dn=replay/f'frame_{index:06d}_dn16.npy'
            if dn.exists():
                image_refs['thermal_dn16']={'path':str(dn),'sha256':seal(dn)}
            terminal=state['action_type']=='STOP'
            if terminal:
                if index != len(frames)-1 or state.get('source_event_ref') != 'run_metadata.json#terminal_snapshot':
                    raise ValueError('Only final exact T can become STOP')
                label={'forward_m':0.,'left_m':0.,'up_m':0.,'yaw_delta_rad':0.,'duration_s':0.,
                       'relative_start_yaw_rad':wrap_yaw(state['yaw']-states[0]['yaw']),
                       'motion_kind':'STOP','stop':True}
                future=None
            else:
                end=select_future_endpoint(states,state['frame_id'],horizon_s=horizon_s,sample_period_s=metadata.get('sample_period_s',.1))
                label=future_pose_target(state,end,initial_yaw=states[0]['yaw'])
                future={'simulation_time_s':end['simulation_time'],'original_source_frame_id':end.get('source_frame_id'),
                        'source_event_ref':end.get('source_event_ref'),'position_m':end['position'],'yaw_rad':end['yaw']}
            samples.append({'sample_id':f'{run.name}:{index:06d}','run_id':run.name,'frame_index':index,
                'observation_time_s':state['simulation_time'],'instruction':header['instruction'],
                'split':header['split'],'start_group':episode['start_group'],'target':episode['target'],
                'images':image_refs,'body_world_matrix':state['recorded_body_world_matrix'],
                'original_source_frame_id':state.get('source_frame_id'),'source_event_ref':state.get('source_event_ref'),
                'requested_horizon_s':horizon_s,'future_endpoint':future,'pose_target':label,
                'sampled_command_reference_only':frame.get('sampled_command')})
        if samples[-1]['pose_target']['motion_kind'] != 'STOP':
            raise ValueError('Indexed replay is missing the exact T image')
        # Recheck every consumed source before publishing a candidate manifest.
        for name,expected in list(sources.items()):
            if seal(Path(name)) != expected:
                raise ValueError(f'Source mutated during preparation: {name}')
        payload=''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in samples)
        (out/'samples.jsonl').write_bytes(payload.encode('utf-8'))
        (out/'source_manifest.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
        counts={kind:sum(s['pose_target']['motion_kind']==kind for s in samples) for kind in ['MOVE','HOLD','STOP']}
        report.update(result='PASS_POSE_TARGET_CANDIDATES',samples=len(samples),counts=counts,run_id=run.name,
            split=header['split'],start_group=episode['start_group'],target=episode['target'],
            action_schema='yaw_local_future_pose_v1',requested_horizon_s=horizon_s,
            samples_sha256=hashlib.sha256(payload.encode()).hexdigest(),
            backward_samples=sum(s['pose_target']['forward_m'] < -1e-3 for s in samples),
            lateral_samples=sum(abs(s['pose_target']['left_m']) > 1e-3 for s in samples),
            alignment='VERIFIED_RECORDED_FUTURE_POSE_TARGETS',intended_keyboard_actions_certified=False,
            strict_pixel_sync_verified=False,task_semantic_review='PENDING',collision_review='PENDING',
            source_manifest_sha256=hashlib.sha256((out/'source_manifest.json').read_bytes()).hexdigest(),
            limitations='Candidates only. Original task confirmation retained; no automatic success, collision or trainability upgrade.')
        return report
    except Exception as error:
        report.update(result='FAIL',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['run','replay','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--horizon',type=float,default=.5)
    args=parser.parse_args()
    report=prepare_candidates(args.run,args.replay,args.out,horizon_s=args.horizon)
    print(json.dumps(report,ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
