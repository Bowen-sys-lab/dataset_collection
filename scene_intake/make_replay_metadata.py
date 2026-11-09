"""Prepare video input from verified capture facts, without changing sealed raw data.

New trajectory captures already contain replay_metadata.json before sealing.
For older captures use --out to create a separate video-input bundle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.frame_metadata import FrameMetadataReader
from scripts.trajectory_recorder import trajectory_provenance,read_trajectory
from scene_intake.capture_integrity import verify_manifest,file_sha256
from scene_intake.merge_replay_passes import _load_planes,RGB_FILES,THERMAL_FILES,validate_pair
from scene_intake.forest_provenance import _hash


def build_replay_metadata(capture,trajectory,*,require_manifest=True):
    """Resolve common metadata and validate every source frame before publishing."""
    capture=Path(capture).resolve();trajectory=Path(trajectory).resolve()
    if require_manifest:verify_manifest(capture)
    if (capture/'failure.txt').exists():raise ValueError('capture is marked failed')
    report=json.loads((capture/'collection_report.json').read_text(encoding='utf-8'))
    count=report.get('frames')
    if report.get('result')!='PASS' or type(count) is not int or count<=0:raise ValueError('capture is not complete')
    if len(list(capture.glob('frame_*.json')))!=count:raise ValueError('capture metadata frame count mismatch')
    prov=trajectory_provenance(trajectory)
    saved_context=report.get('recording_context')
    if saved_context is not None:
        saved=saved_context.get('recording') if isinstance(saved_context,dict) else None
        if not isinstance(saved,dict) or not saved.get('source_run_metadata_sha256') or saved!=prov:
            raise ValueError('recording provenance differs from the sealed collection context')
    replay_path=capture/'replay_metadata.json'
    if replay_path.is_file():
        saved=json.loads(replay_path.read_text(encoding='utf-8')).get('trajectory_provenance')
        if not isinstance(saved,dict) or not saved.get('source_run_metadata_sha256') or any(
                prov.get(key)!=value for key,value in saved.items()):
            raise ValueError('recording provenance differs from the sealed replay metadata')
    stage_hash=_hash(prov.get('recording_stage_sha256'),'stage')
    if not prov.get('source_run_metadata_sha256'):
        raise ValueError('recorded stage identity and run metadata are required')
    sensor_hash=_hash(prov.get('recording_sensor_config_sha256'),'sensor')
    thermal_hash=_hash(prov.get('recording_thermal_config_sha256'),'thermal configuration')
    records={r['frame_id']:r for r in read_trajectory(trajectory)}
    reader=FrameMetadataReader();first=None;previous=None;times=[]
    for index in range(count):
        meta=reader.read(capture/f'frame_{index:06d}.json')
        if meta.get('frame_id')!=index:raise ValueError('capture frame ID mismatch')
        if not meta.get('source_stage_sha256') or meta['source_stage_sha256']!=stage_hash:
            raise ValueError('capture stage differs from the recorded source USD')
        if meta.get('sensor_config_sha256')!=sensor_hash or meta.get('thermal_config_sha256')!=thermal_hash:
            raise ValueError('capture sensor/thermal configuration differs from the recording')
        if meta.get('trajectory_sha256')!=prov['trajectory_sha256']:raise ValueError('capture trajectory hash mismatch')
        if meta.get('task_id')!=(prov['task'] or {}).get('task_id') or meta.get('task_result_status')!=prov['task_result']['status']:
            raise ValueError('capture task/confirmation provenance mismatch')
        record=records.get(meta.get('trajectory_frame_id'))
        if record is None:raise ValueError('capture refers to an unknown trajectory frame')
        t=float(meta.get('replay_simulation_time',float('nan')))
        if not np.isfinite(t) or abs(t-float(record['simulation_time']))>1e-8 or (previous is not None and t<=previous):
            raise ValueError('capture replay time does not match the recording')
        pose=np.asarray(meta.get('camera_world_matrix'),dtype=float)
        source=np.asarray(record.get('rgb_camera_world_matrix'),dtype=float)
        if pose.shape!=(4,4) or source.shape!=(4,4) or not np.isfinite(source).all() or not np.allclose(pose,source,rtol=0,atol=1e-6):
            raise ValueError('capture camera pose differs from the recording')
        validate_pair(meta,meta)
        _load_planes(capture,index,RGB_FILES,meta);_load_planes(capture,index,THERMAL_FILES,meta)
        if first is None:first=meta
        elif any(meta[k]!=first[k] for k in ('rgb_resolution','thermal_resolution')):
            raise ValueError('stream dimensions changed inside a capture')
        previous=t;times.append(t)
    if report.get('rgb_resolution')!=first['rgb_resolution'] or report.get('thermal_resolution')!=first['thermal_resolution']:
        raise ValueError('report and frame stream dimensions differ')
    # Keep the original project's exact provenance structure for video consumers.
    result=dict(result='PASS',frames=count,resolution=first['rgb_resolution'],
                rgb_resolution=first['rgb_resolution'],thermal_resolution=first['thermal_resolution'],
                trajectory=str(trajectory),source_stage_sha256=stage_hash,trajectory_provenance=prov,
                sample_interval_s=(json.loads((trajectory.parent/'run_metadata.json').read_text(encoding='utf-8')).get(
                    'replay_sample_interval_s', times[-1]-times[-2] if len(times)>1 else .1)),
                capture_contract=prov.get('capture_contract'),
                strict_sync_verified=False,strict_per_frame_pixel_sync_verified=False,
                data_usage='paired synthetic development capture; independent pixel synchronization unverified',
                meaning='same recorded stage; RGB normal pass and temporary IR proxy; labels copied from recording')
    return result


def prepare_video_bundle(capture,trajectory,out):
    """Copy a small video view of sealed data; raw capture remains byte-identical."""
    capture=Path(capture).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if out.is_relative_to(capture):raise ValueError('derived video input must be outside the sealed capture')
    manifest=verify_manifest(capture)
    metadata=build_replay_metadata(capture,trajectory)
    names=['common_metadata.json'] if (capture/'common_metadata.json').is_file() else []
    for index in range(metadata['frames']):
        names.extend(f'frame_{index:06d}{suffix}' for suffix in ('.json','_rgb.png','_thermal.png'))
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.forest-video-',dir=out.parent) as staging:
        work=Path(staging)
        for name in names:
            if name not in manifest['files']:raise ValueError(f'video source is not sealed: {name}')
            shutil.copy2(capture/name,work/name)
            if file_sha256(work/name)!=manifest['files'][name]:raise ValueError(f'video source changed during copy: {name}')
        (work/'replay_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        from scripts.generate_replay_video import load_replay
        verify_manifest(capture)
        bundle=dict(schema_version=1,source_capture=str(capture),
                    source_manifest_sha256=file_sha256(capture/'collection_manifest.json'),
                    files={p.name:file_sha256(p) for p in work.iterdir() if p.is_file()})
        (work/'video_input_manifest.json').write_text(json.dumps(bundle,indent=2),encoding='utf-8')
        load_replay(work)
        if out.exists():raise FileExistsError(out)
        work.replace(out)
    return metadata


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--trajectory',type=Path,required=True)
    ap.add_argument('--out',type=Path,help='separate video-input directory; default <capture>-video-input')
    args=ap.parse_args(argv)
    out=args.out or args.capture.with_name(args.capture.name+'-video-input')
    result=prepare_video_bundle(args.capture,args.trajectory,out)
    print(json.dumps({k:result[k] for k in ('result','frames','rgb_resolution','thermal_resolution','source_stage_sha256')},indent=2))
    print('video input:',out.resolve());return 0


if __name__=='__main__':raise SystemExit(main())
