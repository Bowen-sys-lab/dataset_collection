"""Validated forest two-pass merge, preserving raw planes and legacy CLI."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.frame_metadata import FrameMetadataReader,FrameMetadataWriter
RGB_FILES=('_rgb.png','_rgb.npy')
THERMAL_FILES=('_thermal.png','_apparent_t.npy','_dn16.npy','_radiance.npy','_display8.npy')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frames(directory):
    return sorted(int(p.stem.split('_')[1]) for p in Path(directory).glob('frame_*.json')
                  if re.fullmatch(r'frame_\d+',p.stem))

def quality_record(rgb,temperature,dn16,profile):
    """Saved-pixel diagnostics, never a correction of raw values."""
    if not np.isfinite(temperature).all(): raise ValueError('nonfinite apparent temperature')
    if dn16.dtype!=np.uint16: raise ValueError('raw DN must be uint16')
    sat=dn16==65535
    luma=np.asarray(rgb)[...,:3].astype(float)@np.array([.299,.587,.114])
    tc=np.asarray(temperature,dtype=float)-273.15
    low,high=profile.get('temperature_min_c'),profile.get('temperature_max_c')
    clipped=((tc<low)|(tc>high)) if low is not None and high is not None else None
    return dict(dn_saturated_pixels=int(sat.sum()),dn_zero_pixels=int((dn16==0).sum()),
                dn_saturation_fraction=float(sat.mean()),rgb_mean_luma=float(luma.mean()),
                rgb_dark_lt10_fraction=float((luma<10).mean()),apparent_t_min_c=float(tc.min()),
                apparent_t_max_c=float(tc.max()),pixels_above_150c=int((tc>150).sum()),
                temperature_regression_excludes_saturated=True,
                display_clipped_pixels=None if clipped is None else int(clipped.sum())),sat

def _matrix(value):
    a=np.asarray(value,dtype=float)
    if a.shape!=(4,4) or not np.isfinite(a).all(): raise ValueError('invalid camera pose')
    return a

def validate_pair(a,b):
    for key in ('sensor_config_sha256','thermal_config_sha256','thermal_environment_config_sha256','display_config_sha256'):
        if not a.get(key) or a[key]!=b.get(key): raise ValueError(f'pair configuration mismatch: {key}')
    for meta in (a,b):
        t=float(meta['simulation_time_s'])
        if not np.isfinite(t): raise ValueError('invalid capture time')
        for key in ('timeline_time_before','timeline_time_after'):
            if key in meta and (not np.isfinite(float(meta[key])) or abs(float(meta[key])-t)>1e-8):
                raise ValueError('capture time not frozen')
        if meta.get('thermal_coverage',{}).get('complete_for_this_view') is not True:
            raise ValueError('thermal geometry coverage not complete')
    if abs(float(a['simulation_time_s'])-float(b['simulation_time_s']))>1e-8: raise ValueError('pair time mismatch')
    if a.get('trajectory_frame_id')!=b.get('trajectory_frame_id'): raise ValueError('trajectory frame mismatch')
    for field,key in [('thermal_configuration','material_library_sha256'),
                      ('thermal_input_snapshot','material_library_sha256'),
                      ('thermal_input_snapshot','weather_content_sha256')]:
        av=a.get(field,{}).get(key);bv=b.get(field,{}).get(key)
        if (av is not None or bv is not None) and av!=bv:
            raise ValueError(f'pair thermal dependency mismatch: {field}.{key}')
    error=float(np.max(np.abs(_matrix(a['camera_world_matrix'])-_matrix(b['camera_world_matrix']))))
    error=max(error,float(np.max(np.abs(_matrix(a['camera_world_matrix'])-_matrix(b['thermal_camera_world_matrix'])))))
    if error>1e-6: raise ValueError('pair camera pose mismatch')
    normalized=[]
    for meta,prefix in ((a,'rgb'),(b,'thermal')):
        w,h=meta[f'{prefix}_resolution'];k=meta[f'{prefix}_intrinsics_px']
        if min(w,h)<=0 or (k['width'],k['height'])!=(w,h): raise ValueError('intrinsics dimensions mismatch')
        v=np.array([k['fx_px']/w,k['fy_px']/h,k['cx_px']/w,k['cy_px']/h])
        if not np.isfinite(v).all() or min(v[:2])<=0: raise ValueError('invalid intrinsics')
        normalized.append(v)
    if not np.allclose(*normalized,atol=1e-6,rtol=1e-6): raise ValueError('pair intrinsics mismatch')
    return dict(camera_pose_max_error=error,time_error_s=abs(a['simulation_time_s']-b['simulation_time_s']),
                configuration_matches=True,normalized_intrinsics_match=True)

def _load_planes(directory,index,suffixes,meta):
    stem=f'frame_{index:06d}';planes={}
    for suffix in suffixes:
        path=directory/f'{stem}{suffix}'
        if not path.is_file(): raise ValueError(f'missing required plane: {path}')
        try:
            if suffix.endswith('.npy'):
                arr=np.load(path,allow_pickle=False)
                if not np.isfinite(arr).all(): raise ValueError(f'nonfinite plane: {path}')
            else:
                with Image.open(path) as img: arr=np.asarray(img).copy()
        except (OSError,TypeError) as exc: raise ValueError(f'invalid plane: {path}') from exc
        prefix='rgb' if suffix.startswith('_rgb') else 'thermal'
        w,h=meta[f'{prefix}_resolution']
        if arr.shape[:2]!=(h,w): raise ValueError(f'plane dimensions mismatch: {path}')
        if suffix in ('_apparent_t.npy','_radiance.npy'):
            if arr.ndim!=2 or arr.dtype not in (np.dtype('float32'),np.dtype('float64')):
                raise ValueError(f'scalar thermal plane must be 2D float32/float64: {path}')
        elif suffix=='_dn16.npy':
            if arr.ndim!=2 or arr.dtype!=np.uint16:raise ValueError(f'raw DN plane must be 2D uint16: {path}')
        elif prefix=='rgb':
            if arr.ndim!=3 or arr.shape[2] not in (3,4) or arr.dtype!=np.uint8:
                raise ValueError(f'RGB plane must be uint8 RGB/RGBA: {path}')
        elif arr.dtype!=np.uint8 or not (arr.ndim==2 or (arr.ndim==3 and arr.shape[2] in (3,4))):
            raise ValueError(f'display plane must be uint8 grayscale/RGB/RGBA: {path}')
        planes[suffix]=arr
    if '_rgb.png' in planes and not np.array_equal(planes['_rgb.png'],planes['_rgb.npy']):
        raise ValueError('RGB PNG does not match RGB array')
    if '_thermal.png' in planes and not np.array_equal(planes['_thermal.png'],planes['_display8.npy']):
        raise ValueError('thermal PNG does not match display array')
    return planes

def write_replay_metadata(rgb_dir,ir_dir,out,frame_count):
    """为合并产物合成 replay_metadata.json（供离线视频脚本消费）。

    依据：两次回放各自写出的 replay_metadata.json。缺任何一个就跳过，不猜。
    """
    paths={k:d/'replay_metadata.json' for k,d in (('rgb',rgb_dir),('thermal',ir_dir))}
    if not all(p.is_file() for p in paths.values()):return None
    src={k:json.loads(p.read_text(encoding='utf-8')) for k,p in paths.items()}
    for k,meta in src.items():
        if meta.get('result')!='PASS':raise ValueError(f'{k} replay did not pass; refusing to synthesize video metadata')
        if int(meta.get('frames',-1))!=frame_count:raise ValueError(f'{k} replay frame count mismatch')
    rgb,thermal=src['rgb'],src['thermal']
    if rgb.get('trajectory')!=thermal.get('trajectory'):
        raise ValueError('the two replays used different trajectories')
    report=dict(result='PASS',frames=frame_count,
                resolution=rgb.get('resolution'),rgb_resolution=rgb.get('rgb_resolution',rgb.get('resolution')),
                thermal_resolution=thermal.get('thermal_resolution',thermal.get('resolution')),
                trajectory=rgb.get('trajectory'),source_stage_sha256=rgb.get('source_stage_sha256'),
                trajectory_provenance=rgb.get('trajectory_provenance'),
                start_frame=rgb.get('start_frame'),sample_interval_s=rgb.get('sample_interval_s'),
                rgb_run=str(rgb_dir),thermal_run=str(ir_dir),
                merge=dict(method='two-pass merge',
                           rgb_stage_sha256=rgb.get('source_stage_sha256'),
                           thermal_stage_sha256=thermal.get('source_stage_sha256'),
                           note=('source_stage_sha256 is the stage the trajectory was recorded on (RGB pass); '
                                 'the thermal pass deliberately used a derived stage with the baked flame proxy'),
                           strict_pixel_sync_verified=False),
                meaning='synthesized by merge_replay_passes.py from the two replay_metadata.json files')
    (out/'replay_metadata.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return dict(frames=frame_count,rgb_stage_sha256=report['merge']['rgb_stage_sha256'],
                thermal_stage_sha256=report['merge']['thermal_stage_sha256'])

def merge_runs(rgb_dir,ir_dir,out):
    rgb_dir,ir_dir,out=map(lambda p:Path(p).resolve(),(rgb_dir,ir_dir,out))
    if out.exists(): raise FileExistsError(out)
    ids=frames(rgb_dir)
    if not ids or ids!=frames(ir_dir): raise ValueError('empty run or frame mismatch')
    reader=FrameMetadataReader();checked=[]
    for i in ids:
        a=reader.read(rgb_dir/f'frame_{i:06d}.json');b=reader.read(ir_dir/f'frame_{i:06d}.json')
        pair=validate_pair(a,b)
        if a['frame_id']!=i or b['frame_id']!=i: raise ValueError('frame metadata ID mismatch')
        rgb=_load_planes(rgb_dir,i,RGB_FILES,a);ir=_load_planes(ir_dir,i,THERMAL_FILES,b)
        profile=b.get('display_profile') or {}
        q,_=quality_record(rgb['_rgb.npy'],ir['_apparent_t.npy'],ir['_dn16.npy'],profile)
        diag=deepcopy(b.get('display_range_diagnostics',{}))
        low,high=profile.get('temperature_min_c'),profile.get('temperature_max_c')
        if low is not None and high is not None:
            t=ir['_apparent_t.npy'].astype(float)-273.15
            diag.update(finite_pixels=int(t.size),nonfinite_pixels=0,below_range_pixels=int((t<low).sum()),
                        above_range_pixels=int((t>high).sum()),clipped_fraction_of_finite=float(((t<low)|(t>high)).mean()),
                        display_temperature_min_c=low,display_temperature_max_c=high,
                        clipping_scope='8-bit display only; raw DN saturation is recorded separately')
        m=deepcopy(b)
        for key,value in a.items():
            if key.startswith('rgb_') or key=='camera_world_matrix':m[key]=deepcopy(value)
        m.update(frame_id=i,frame=i,display_range_diagnostics=diag,quality=q,
                 strict_sync_verified=False,strict_per_frame_pixel_sync_verified=False,
                 sync_status='two_pass_pose_time_verified_pixel_sync_unverified',sync_validation_status='not_verified',
                 sync_validation_evidence=None,
                 sync_note='Pose/time/config checked. Dynamic animation and independent pixel sync require separate evidence.',
                 pair_validation=pair,
                 merged_passes={key:dict(directory=str(d),metadata_sha256=sha256(d/f'frame_{i:06d}.json'),
                                        stage_path=src.get('stage_path'),source_stage_sha256=src.get('source_stage_sha256'),
                                        trajectory_sha256=src.get('trajectory_sha256'))
                                for key,d,src in [('rgb',rgb_dir,a),('thermal',ir_dir,b)]})
        checked.append((i,m))
    out.mkdir(parents=True);(out/'quality').mkdir()
    pending=out/'failure.txt'
    pending.write_text('Merge has not finalized; do not use partial outputs.',encoding='utf-8')
    writer=FrameMetadataWriter(out,checked[0][1])
    for i,m in checked:
        stem=f'frame_{i:06d}'
        for directory,suffixes in ((rgb_dir,RGB_FILES),(ir_dir,THERMAL_FILES)):
            for suffix in suffixes:shutil.copy2(directory/f'{stem}{suffix}',out/f'{stem}{suffix}')
        sat=np.load(out/f'{stem}_dn16.npy',allow_pickle=False)==65535
        mask=f'quality/{stem}_saturation.png';Image.fromarray(sat.astype('uint8')*255).save(out/mask)
        m['quality']['saturation_mask']=mask;writer.write(i,m)
    notes=dict(method='validated two-pass merge; raw planes copied verbatim',rgb_run=str(rgb_dir),thermal_run=str(ir_dir),
               frames=len(ids),strict_pixel_sync_verified=False)
    (out/'merge_notes.json').write_text(json.dumps(notes,indent=2),encoding='utf-8')
    # 我们自己链路的视频脚本(scripts/generate_replay_video.py)要求目录里有 replay_metadata.json。
    # 合并产物原本没有，这里根据两次回放各自的 replay_metadata.json 合成一份：
    # source_stage_sha256 取 RGB 那一遍（轨迹就是在那个 stage 上录的）；
    # 红外那一遍刻意用烘焙了火焰代理的 stage，差异写进 merge 段，不隐藏。
    synthesized=write_replay_metadata(rgb_dir,ir_dir,out,len(ids))
    if synthesized:notes['replay_metadata']=synthesized
    pending.unlink()
    return notes

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rgb-run',required=True,type=Path);ap.add_argument('--thermal-run',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path);args=ap.parse_args(argv)
    print(json.dumps(merge_runs(args.rgb_run,args.thermal_run,args.out),indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
