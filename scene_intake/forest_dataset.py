"""Forest dataset indexer using saved intrinsics and validated raw planes."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.frame_metadata import FrameMetadataReader
from scene_intake.merge_replay_passes import frames,_load_planes,RGB_FILES,THERMAL_FILES,quality_record,sha256,validate_pair
from scene_intake.forest_collection import TARGET_POSITION,TARGET_PATH
from scene_intake.capture_integrity import verify_manifest

def project_target(matrix,target,intrinsics):
    m=np.asarray(matrix,dtype=float);target=np.asarray(target,dtype=float)
    if m.shape!=(4,4) or target.shape!=(3,) or not np.isfinite(m).all() or not np.isfinite(target).all():raise ValueError('invalid projection pose/target')
    cam=np.append(target,1.)@np.linalg.inv(m)
    depth=-float(cam[2])
    if depth<=1e-8:return None,None,False
    k=intrinsics
    u=k['fx_px']*cam[0]/depth+k['cx_px'];v=k['cy_px']-k['fy_px']*cam[1]/depth
    if not np.isfinite([u,v]).all():raise ValueError('invalid projection intrinsics')
    return float(u),float(v),bool(0<=u<k['width'] and 0<=v<k['height'])

def pack_dataset(capture,out,*,target_position=None):
    capture=Path(capture).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if (capture/'failure.txt').exists():raise ValueError('capture is marked failed')
    manifest_path=capture/'collection_manifest.json'
    manifest=verify_manifest(capture,required=False)
    ids=frames(capture)
    if not ids:raise ValueError('no frame metadata')
    reader=FrameMetadataReader();rows=[];provenance=[];scene_label='forest_fire'
    report_task={};recording_context=None
    for i in ids:
        m=reader.read(capture/f'frame_{i:06d}.json')
        scene_label=m.get('forest_target',{}).get('dataset_scene',scene_label)
        if 'collection_profile' in m:
            report_path=capture/'collection_report.json'
            if not manifest or not report_path.is_file():
                raise ValueError('new collection is not finalized: report and manifest required')
            report=json.loads(report_path.read_text(encoding='utf-8'))
            if report.get('result')!='PASS' or report.get('frames')!=len(ids):
                raise ValueError('new collection report is incomplete or has a frame-count mismatch')
            report_task=report.get('task') or {}
            recording_context=report.get('recording_context')
            if recording_context:
                recorded=recording_context['recording']
                expected_id=(recorded.get('task') or {}).get('task_id')
                if (report_task.get('task_id')!=expected_id or m.get('task_id')!=expected_id or
                    m.get('task_result_status')!=recorded['task_result']['status'] or
                    m.get('trajectory_sha256')!=recorded['trajectory_sha256']):
                    raise ValueError('frame/index recording labels do not match provenance')
        if m['frame_id']!=i:raise ValueError('frame ID mismatch')
        validate_pair(m,m)
        rgb=_load_planes(capture,i,RGB_FILES,m);ir=_load_planes(capture,i,THERMAL_FILES,m)
        q,_=quality_record(rgb['_rgb.npy'],ir['_apparent_t.npy'],ir['_dn16.npy'],m.get('display_profile') or {})
        target=np.asarray(target_position if target_position is not None else m.get('forest_target',{}).get('position_m',TARGET_POSITION),dtype=float)
        u,v,visible=project_target(m['camera_world_matrix'],target,m['rgb_intrinsics_px'])
        pose=np.asarray(m['camera_world_matrix'])
        rows.append(dict(frame_id=i,sim_time_s=m['simulation_time_s'],requested_time_s=m.get('requested_time_s'),
                         target_prim_path=m.get('forest_target',{}).get('prim_path',TARGET_PATH),
                         target_u_px=u,target_v_px=v,target_in_camera_frustum=visible,
                         target_visibility_meaning='geometric projection only; occlusion is not inferred',
                         dist_to_target_m=float(np.linalg.norm(pose[3,:3]-target)),
                         rgb_png=f'frame_{i:06d}_rgb.png',thermal_png=f'frame_{i:06d}_thermal.png',
                         apparent_t_npy=f'frame_{i:06d}_apparent_t.npy',dn16_npy=f'frame_{i:06d}_dn16.npy',
                         t_min_c=q['apparent_t_min_c'],t_max_c=q['apparent_t_max_c'],
                         dn_saturated_pixels=q['dn_saturated_pixels'],pixels_above_150c=q['pixels_above_150c'],
                         rgb_mean_luma=q['rgb_mean_luma'],rgb_dark_lt10_fraction=q['rgb_dark_lt10_fraction'],
                         strict_pixel_sync_verified=m.get('strict_per_frame_pixel_sync_verified') is True,
                         dynamic_flame_time_matches_rgb=m.get('flame_thermal_proxy',{}).get('animation_time_matches_rgb') is True,
                         rgb_width=m['rgb_resolution'][0],rgb_height=m['rgb_resolution'][1],
                         thermal_width=m['thermal_resolution'][0],thermal_height=m['thermal_resolution'][1],
                         task_id=report_task.get('task_id'),task_source=report_task.get('source'),
                         task_sha256=report_task.get('sha256')))
        provenance.append(dict(frame_id=i,metadata_sha256=sha256(capture/f'frame_{i:06d}.json'),
                               rgb_intrinsics_px=m['rgb_intrinsics_px'],thermal_intrinsics_px=m['thermal_intrinsics_px'],
                               source_stage_sha256=m.get('source_stage_sha256'),merged_passes=m.get('merged_passes')))
    result=dict(dataset_version=1,scene=scene_label,frames=len(rows),frame_root=str(capture),
                task_result=(recording_context['recording']['task_result'] if recording_context else None),
                task_result_meaning=('copied recording confirmation; no automatic task qualification' if recording_context else
                                     'static/legacy observation; no task success fabricated'),
                recording_context=recording_context,
                task=dict(task_id=report_task.get('task_id'),source=report_task.get('source'),
                          sha256=report_task.get('sha256'),start_position=report_task.get('start_position'),
                          meaning='per-frame task_id is stamped from collection_report.json; the legacy frame writer does not carry it'),
                integrity_manifest_verified=bool(manifest),source_manifest_sha256=sha256(manifest_path) if manifest else None,
                strict_pixel_sync_verified=all(r['strict_pixel_sync_verified'] for r in rows),
                dynamic_flame_time_matches_rgb=all(r['dynamic_flame_time_matches_rgb'] for r in rows),
                temperature_meaning='apparent temperature in saved arrays; prescribed truth is separate',
                sensor_limitations='not real instrument calibration; saturated DN excluded from temperature regression',
                tables=dict(index='index.csv',jsonl='index.jsonl',provenance='provenance.json'))
    out.mkdir(parents=True)
    with (out/'index.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'index.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (out/'dataset.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(argv);print(json.dumps(pack_dataset(args.capture,args.out),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
