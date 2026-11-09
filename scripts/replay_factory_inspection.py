"""Factory-only frozen-pose RGB/thermal replay and Isaac-independent checks.

Route samples are sparse, but every distance/speed-qualified observation sample
is retained. This preserves the evaluator's frame continuity and never creates
manual T confirmation or claims independent pixel synchronization evidence.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/isaac-thermal-camera-simulation/src'))
from scripts.collection_capture_defaults import RGB_SIZE, THERMAL_SIZE, replay_capture_contract
DEFAULT_TRAJECTORY = ROOT / 'validation/factory-inspection/integration-03/reference_run/trajectory.jsonl'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')


def development_labels(provenance, *, manual_only=False):
    source = provenance.get('trajectory_source', 'legacy_unclassified')
    result = copy.deepcopy(provenance['task_result'])
    if source.startswith('synthetic') and (result.get('status') == 'success' or result.get('marked_by') == 'keyboard:T'):
        raise ValueError('synthetic trajectory cannot carry human confirmation')
    labels = {
        'trajectory_source': source, 'task_result': result,
        'qualification': 'manual_flight' if source == 'manual_driving' else 'development',
        'data_usage': 'pilot/development data; synthetic sources are not human teacher data',
        'manual_confirmation_created': False, 'strict_sync_verified': False,
        'strict_per_frame_pixel_sync_verified': False,
        'sync_status': 'factory_frozen_capture_runtime_checks_only',
        'sync_validation_status': 'not_verified',
        'sync_validation_scope': 'runtime render/pose/time/coverage checks only; no independent factory pixel-marker probe',
        'sync_validation_evidence': None,
    }
    if manual_only:
        labels.update(confirmation_mode='manual_only',qualification='manual_confirmation_preview',
            automatic_observation_gate_applied=False,
            data_usage='preview_only; recorded human confirmation, no automatic inspection qualification')
    return labels


def manual_confirmation_report(provenance):
    labels=development_labels(provenance,manual_only=True)
    result=labels['task_result']
    confirmed=(labels['trajectory_source']=='manual_driving' and result.get('status')=='success'
               and result.get('marked_by')=='keyboard:T')
    return {'result':'MANUAL_CONFIRMED' if confirmed else 'UNCONFIRMED',
            'automatic_rules_evaluated':False,'manual_confirmation_created':False,
            'recorded_task_result':result,'trajectory_sha256':provenance.get('trajectory_sha256'),
            'note':'Preview/manual label only; distance, speed, hold, route and hotspot visibility are not acceptance gates.'}


def select_records(records, distances, requirements, route_interval=1., *, manual_only=False, confirmation_frame_id=None):
    """Keep all observation candidates, never interpolate or bridge source gaps."""
    def longest_hold(eligible):
        start = previous_time = previous_frame = None
        longest = 0.
        for row, good in zip(records, eligible):
            t, frame = float(row['simulation_time']), row['frame_id']
            contiguous = (good and previous_time is not None and frame == previous_frame + 1
                          and 0. < t - previous_time <= maximum_gap + 1e-9)
            if good and (start is None or not contiguous):
                start = t
            if good and start is not None:
                longest = max(longest, t - start)
            if not good:
                start = None
            previous_time, previous_frame = t, frame
        return longest
    if not math.isfinite(route_interval) or route_interval <= 0:
        raise ValueError('route interval must be finite and positive')
    if len(records) != len(distances) or not records:
        raise ValueError('nonempty records and matching mesh distances are required')
    low, high = requirements.get('observation_distance_range_m', [6., 8.])
    speed_limit = requirements.get('maximum_observation_speed_m_s', .3)
    maximum_gap = requirements.get('maximum_sample_gap_s', .25)
    duration = requirements.get('hold_duration_s', 6.)
    eligible, times, ids = [], [], []
    previous = None
    for row, distance in zip(records, distances):
        t = float(row['simulation_time']); position = np.asarray(row['position'], dtype=float)
        velocity = np.asarray(row.get('linear_velocity', [math.inf] * 3), dtype=float)
        if (not math.isfinite(t) or position.shape != (3,) or not np.isfinite(position).all()
                or type(row['frame_id']) is not int or (times and (t <= times[-1] or row['frame_id'] <= ids[-1]))):
            raise ValueError('invalid trajectory pose, timestamp, or frame order')
        speed = float(np.linalg.norm(velocity)) if velocity.shape == (3,) and np.isfinite(velocity).all() else math.inf
        if previous is not None:
            speed = max(speed, float(np.linalg.norm(position - previous)) / (t - times[-1]))
        eligible.append(math.isfinite(float(distance)) and low <= distance <= high and speed <= speed_limit + 1e-9)
        previous = position; times.append(t); ids.append(row['frame_id'])
    if not manual_only and longest_hold(eligible) + 1e-9 < duration:
        raise ValueError('source has no continuous distance/speed-qualified observation hold')
    observations = set() if manual_only else {r['frame_id'] for r, good in zip(records, eligible) if good}
    selected, next_time = [], times[0]
    for row, t in zip(records, times):
        route_due = t >= next_time - 1e-9
        if route_due or row['frame_id'] in observations or (manual_only and row['frame_id']==confirmation_frame_id) or row is records[-1]:
            selected.append(row)
        if route_due:
            next_time = t + route_interval
    return selected, observations


def validate_capture(meta, record):
    """Runtime checks are deliberately weaker than independent pixel sync."""
    for key in ('explicit_render_triggered', 'explicit_render_completed', 'frozen_state_capture'):
        if meta.get(key) is not True:
            raise ValueError(f'capture check failed: {key}')
    for key, expected, tolerance in (
        ('pair_pose_max_error', 0., 1e-6), ('simulation_capture_drift_s', 0., 1e-8),
        ('simulation_time_before', record['simulation_time'], 1e-8),
        ('simulation_time_after', record['simulation_time'], 1e-8),
    ):
        value = float(meta[key])
        if not math.isfinite(value) or abs(value - expected) > tolerance:
            raise ValueError(f'capture check failed: {key}')
    matrices = [np.asarray(meta.get(key), dtype=float) for key in ('camera_world_matrix', 'thermal_camera_world_matrix')]
    recorded = np.asarray(record.get('rgb_camera_world_matrix'), dtype=float)
    if any(m.shape != (4, 4) or not np.isfinite(m).all() for m in [*matrices, recorded]):
        raise ValueError('camera matrices missing or nonfinite')
    if not np.allclose(matrices[0], recorded, atol=1e-6, rtol=0.) or not np.allclose(matrices[0], matrices[1], atol=1e-6, rtol=0.):
        raise ValueError('recorded/aligned camera pose differs')
    coverage = meta.get('thermal_coverage', {})
    if coverage.get('complete_for_this_view') is not True or coverage.get('apparent_t_nonfinite_pixels') != 0:
        raise ValueError('thermal coverage incomplete or nonfinite')


def hotspot_evidence(instance_ids, labels, target_path, minimum_pixels=20, *, target_paths=None):
    if target_paths is not None and (not isinstance(target_paths,list) or not target_paths or
            any(not isinstance(p,str) or not p.startswith('/') for p in target_paths) or
            len(set(target_paths))!=len(target_paths) or target_path not in target_paths):
        raise ValueError('Explicit hotspot surface paths must be unique and include the representative target')
    paths = {target_path} if target_paths is None else set(target_paths)
    ids = sorted(int(key) for key, path in labels.items() if path in paths)
    pixels = int(np.isin(np.asarray(instance_ids), ids).sum())
    visible = bool(ids and pixels >= minimum_pixels)
    return {**({} if target_paths is None else {'hotspot_prim_paths':target_paths}),
            'hotspot_instance_ids': ids, 'hotspot_visible_pixels': pixels,
            'hotspot_visible': visible, 'unobstructed': visible,
            'minimum_hotspot_pixels': minimum_pixels,
            'method': 'actual thermal renderer surface IDs under aligned RGB/IR cameras; partial surface visibility only'}


def hotspot_labels(labels,paths):
    wanted=set(paths)
    return {str(key):path for key,path in labels.items() if path in wanted}


def configure_replay_memory(settings,budget=.35):
    if not math.isfinite(budget) or not 0<budget<=1:
        raise ValueError('texture streaming budget must be in (0,1]')
    prefix='/rtx-transient/resourcemanager/texturestreaming/'
    settings.set(prefix+'enabled',True)
    settings.set(prefix+'memoryBudget',float(budget))
    return {'texture_streaming_enabled':settings.get(prefix+'enabled'),
            'texture_budget_fraction':settings.get(prefix+'memoryBudget'),
            'scope':'process-local texture residency budget; scene, temperature and output resolution unchanged'}


def write_frame_order(out,records):
    times=[float(r['simulation_time']) for r in records]
    if not times or any(not math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('frame order must be strictly chronological')
    path=Path(out)/'frame_order.csv'
    fields=['frame_index','rgb_file','thermal_file','simulation_time_s','time_from_start_s',
            'trajectory_frame_id','source_frame_id','source_event_ref']
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,r in enumerate(records):
            writer.writerow({'frame_index':i,'rgb_file':f'frame_{i:06d}_rgb.png',
                'thermal_file':f'frame_{i:06d}_thermal.png','simulation_time_s':f'{times[i]:.9f}',
                'time_from_start_s':f'{times[i]-times[0]:.9f}','trajectory_frame_id':r['frame_id'],
                'source_frame_id':r.get('source_frame_id',r['frame_id']),
                'source_event_ref':r.get('source_event_ref','')})
    return path


def validate_saved_frame(root, index, *, rgb_size=RGB_SIZE, thermal_size=THERMAL_SIZE):
    from PIL import Image
    root = Path(root); stem = f'frame_{index:06d}'
    checked = []
    for suffix, size in (('rgb', rgb_size), ('display8', thermal_size), ('radiance', thermal_size),
                         ('apparent_t', thermal_size), ('dn16', thermal_size)):
        path = root / f'{stem}_{suffix}.npy'
        array = np.load(path, allow_pickle=False)
        if array.shape[:2] != (size[1], size[0]):
            raise ValueError(f'{path.name}: incorrect resolution')
        if not np.isfinite(array).all():
            raise ValueError(f'{path.name}: nonfinite array values')
        checked.append(path.name)
    for suffix, size in (('rgb', rgb_size), ('thermal', thermal_size)):
        path = root / f'{stem}_{suffix}.png'
        with Image.open(path) as image:
            image.load()
            if image.size != size:
                raise ValueError(f'{path.name}: incorrect PNG resolution')
            expected = np.load(root / f'{stem}_{"rgb" if suffix == "rgb" else "display8"}.npy', allow_pickle=False)
            if not np.array_equal(np.asarray(image), expected):
                raise ValueError(f'{path.name}: PNG differs from saved array')
        checked.append(path.name)
    return {'files_checked': checked, 'nonfinite_values': 0}


def manual_replay_records(records, provenance):
    if provenance.get('terminal_snapshot') is None:
        return records  # Historical runs retain their original manual-preview semantics.
    from scripts.trajectory_recorder import build_training_sequence
    return build_training_sequence(records, provenance)


def write_terminal_replay_source(out, original, records):
    """Persist the derived sequence so the shared video reader sees exact event time.

    Copy the genuine recorded label and retain the original hash; never edit raw inputs.
    """
    original=Path(original).resolve()
    folder=Path(out)/'source_sequence';folder.mkdir(parents=True,exist_ok=False)
    path=folder/'trajectory.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    metadata=json.loads((original.parent/'run_metadata.json').read_text(encoding='utf-8'))
    metadata.update(record_count=len(records),trajectory_file=str(path.resolve()),
        source_trajectory_path=str(original),source_trajectory_sha256=sha256(original),
        source_run_metadata_sha256=sha256(original.parent/'run_metadata.json'),
        derivation='exact recorded terminal snapshot; no post-confirmation flight; original label copied unchanged')
    write_json(folder/'run_metadata.json',metadata)
    return path.resolve()


def prepare(config_path, trajectory, route_interval, *, manual_only=False):
    from scripts.scene_config import load_scene_config
    from scripts.trajectory_recorder import read_trajectory, trajectory_provenance
    from scripts.evaluate_factory_inspection import point_to_mesh_distances
    config = load_scene_config(config_path, include_task=False)
    provenance = trajectory_provenance(trajectory)
    labels = development_labels(provenance,manual_only=manual_only)
    if provenance['recording_scene_config_sha256'] != config.sha256 or provenance['recording_stage_sha256'] != sha256(config.scene.stage):
        raise ValueError('recorded scene configuration or stage differs from current source')
    from scripts.runtime_parameter_provenance import validate_thermal_inputs
    parameter_snapshot = validate_thermal_inputs(config,provenance.get('thermal_input_snapshot'))
    task = provenance['task']
    inspection = (task or {}).get('environment_evidence', {}).get('factory_inspection')
    if not inspection:
        raise ValueError('recorded factory task snapshot is required')
    relative = Path(inspection['geometry_file']); geometry = (ROOT / relative).resolve()
    if relative.is_absolute() or not geometry.is_relative_to(ROOT) or sha256(geometry) != inspection['geometry_sha256']:
        raise ValueError('factory geometry provenance mismatch')
    records = read_trajectory(trajectory)
    if manual_only:
        records = manual_replay_records(records, provenance)
    # Both exact body and recorded camera poses are mandatory for this new pipeline.
    for row in records:
        for key in ('recorded_body_world_matrix', 'rgb_camera_world_matrix'):
            matrix = np.asarray(row.get(key), dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f'{key}: finite recorded 4x4 transform required')
    requirements = inspection['inspection_requirements']
    if manual_only:
        distances=np.zeros(len(records))
    else:
        with np.load(geometry, allow_pickle=False) as mesh:
            distances = point_to_mesh_distances(np.asarray([r['position'] for r in records]),
                mesh['equipment_vertices'], mesh['equipment_triangles'],
                body_radius_m=requirements['body_envelope_radius_m'])
    selected, observations = select_records(records, distances, requirements, route_interval,
        manual_only=manual_only,confirmation_frame_id=provenance['task_result'].get('nearest_trajectory_frame_id'))
    files = [config.source_path, config.scene.stage, config.sensors.thermal_config,
             config.sensors.sensor_config, config.scene.thermal_scene, geometry,
             Path(trajectory), Path(trajectory).parent / 'run_metadata.json']
    if config.sensors.display_config:
        files.append(config.sensors.display_config)
    if parameter_snapshot is not None:
        files.extend(Path(p) for p in parameter_snapshot['file_sha256'])
    hashes = {str(p.resolve()): sha256(p) for p in files}
    plan = {'source_records': len(records), 'selected_frames': len(selected),
            'source_recording_trajectory':str(Path(trajectory).resolve()),
            'terminal_crop_applied':bool(manual_only and provenance.get('terminal_snapshot')),
            'observation_frames': len(observations), 'route_interval_s': route_interval,
            'observation_selection': ('uniform time samples plus recorded manual T frame; no automatic observation gate' if manual_only else
                                     'every source distance/speed-qualified record; no interpolation'),
            'observation_maximum_gap_s': None if manual_only else requirements['maximum_sample_gap_s'],
            'observation_required_hold_s': None if manual_only else requirements['hold_duration_s'],
            'rgb_resolution': list(RGB_SIZE), 'thermal_resolution': list(THERMAL_SIZE),
            'source_input_hashes': hashes, **labels}
    return config, provenance, records, selected, observations, plan


def validate_bundle(root):
    """Factory validation supplements the unchanged public video reader."""
    from scripts.generate_replay_video import load_replay
    from scripts.frame_metadata import FrameMetadataReader
    from scripts.trajectory_recorder import read_trajectory
    from scripts.evaluate_factory_inspection import _longest_hold
    root = Path(root); bundle = load_replay(root)
    metadata = bundle.metadata
    manual_only=metadata.get('confirmation_mode')=='manual_only'
    labels = development_labels(metadata['trajectory_provenance'],manual_only=manual_only)
    if metadata.get('strict_sync_verified') is not False or metadata.get('strict_per_frame_pixel_sync_verified') is not False:
        raise ValueError('factory replay lacks independent strict synchronization evidence')
    if metadata.get('task_result') != labels['task_result']:
        raise ValueError('recorded task result differs')
    records = read_trajectory(bundle.trajectory_path)
    if manual_only:
        records = manual_replay_records(records, metadata['trajectory_provenance'])
    by_id = {r['frame_id']: r for r in records}
    confirmation_id=labels['task_result'].get('nearest_trajectory_frame_id')
    if (manual_only and confirmation_id in by_id and
            confirmation_id not in {frame.trajectory_frame_id for frame in bundle.frames}):
        raise ValueError('recorded manual confirmation frame is missing from replay')
    rows = [json.loads(line) for line in (root / 'visual_evidence.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    visual = {row['frame_id']: row for row in rows}
    if len(visual) != len(rows):
        raise ValueError('duplicate visual evidence frame')
    reader = FrameMetadataReader(); checked = 0
    for frame in bundle.frames:
        record = by_id[frame.trajectory_frame_id]; meta = reader.read(frame.metadata_path)
        validate_capture(meta, record)
        validate_saved_frame(root, frame.index, rgb_size=tuple(metadata['rgb_resolution']),
                             thermal_size=tuple(metadata['thermal_resolution']))
        if meta.get('strict_sync_verified') is not False or meta.get('strict_per_frame_pixel_sync_verified') is not False:
            raise ValueError('frame has unsupported strict sync claim')
        if frame.trajectory_frame_id in metadata['observation_frame_ids']:
            evidence = visual.get(frame.trajectory_frame_id)
            if not evidence or evidence.get('evidence_image') != frame.rgb_path.name or evidence.get('thermal_evidence_image') != frame.thermal_path.name:
                raise ValueError('observation image evidence missing or mismatched')
            if (evidence.get('trajectory_sha256') != metadata['trajectory_provenance']['trajectory_sha256']
                    or evidence.get('scene_config_sha256') != metadata['scene_config_sha256']
                    or evidence.get('hotspot_prim_path') != metadata['task']['target_prim_path']
                    or abs(float(evidence['simulation_time']) - record['simulation_time']) > 1e-8
                    or not np.allclose(evidence['camera_world_matrix'], record['rgb_camera_world_matrix'], atol=1e-6, rtol=0.)):
                raise ValueError('visual evidence provenance/time/pose mismatch')
            ids = np.load(root / evidence['hotspot_instance_image'], allow_pickle=False)
            expected_paths = metadata['task'].get('environment_evidence',{}).get('factory_inspection',{}).get('hotspot_prim_paths')
            actual = hotspot_evidence(ids, meta['hotspot_renderer_labels'], evidence['hotspot_prim_path'], evidence['minimum_hotspot_pixels'],target_paths=expected_paths)
            if any(evidence.get(key) != value for key, value in actual.items()):
                raise ValueError('hotspot renderer evidence differs from saved instance IDs')
            checked += 1
    expected_ids = set(metadata['observation_frame_ids'])
    if set(visual) != expected_ids or not expected_ids.issubset({f.trajectory_frame_id for f in bundle.frames}):
        raise ValueError('observation evidence is incomplete')
    requirement = metadata['task']['environment_evidence']['factory_inspection']['inspection_requirements']
    hold = _longest_hold(records, [r['frame_id'] in expected_ids and visual[r['frame_id']]['hotspot_visible']
                                  and visual[r['frame_id']]['unobstructed'] for r in records], requirement['maximum_sample_gap_s'])
    if not manual_only and hold + 1e-9 < requirement['hold_duration_s']:
        raise ValueError('rendered observation does not establish the required visible hold')
    for path, digest in metadata['source_input_hashes'].items():
        if sha256(path) != digest:
            raise ValueError(f'source input changed: {path}')
    report = {'result': 'PASS', 'frames_checked': len(bundle.frames), 'observation_frames_checked': checked,
              'visible_observation_duration_s': hold, 'strict_sync_verified': False,
              'strict_per_frame_pixel_sync_verified': False, 'manual_confirmation_created': False,
              'tested_scope': ['saved RGB/thermal PNG versus NPY equality and resolutions',
                  'raw radiance/apparent-temperature/DN arrays finite', 'recorded timestamps and aligned camera matrices',
                  'thermal geometry coverage per sampled view', 'saved actual hotspot renderer instance IDs',
                  'source trajectory/task/config/geometry hashes and continuous visible observation hold'],
              'not_tested': ['independent pixel-buffer freshness/synchronization', 'continuous collision between samples',
                             'human recognition or T confirmation'], 'trajectory_source': labels['trajectory_source']}
    if manual_only:
        report.update(confirmation_mode='manual_only',automatic_observation_gate_applied=False,
            task_result=labels['task_result'],data_usage=labels['data_usage'])
        report['tested_scope']=report['tested_scope'][:4]+['source trajectory/task/config/geometry hashes']
    write_json(root / 'validation_summary.json', report)
    return report


def render(args):
    from scripts.replay_pose import FrozenPoseRestorer, compare_pose_matrices, pose_within_tolerance, world_matrix
    from scripts.trajectory_recorder import trajectory_provenance
    config, provenance, records, selected, observations, plan = prepare(args.config, args.trajectory, args.route_interval,manual_only=args.manual_only)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)); return 0
    if args.out is None:
        raise ValueError('--out is required for rendering')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.manual_only and provenance.get('terminal_snapshot') is not None:
        args.trajectory=write_terminal_replay_source(args.out,args.trajectory,records)
        provenance=trajectory_provenance(args.trajectory)
        plan['source_input_hashes'].update({str(args.trajectory):sha256(args.trajectory),
            str(args.trajectory.parent/'run_metadata.json'):sha256(args.trajectory.parent/'run_metadata.json')})
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'PathTracing', 'multi_gpu': False,
                         'samples_per_pixel_per_frame': 1})
    preview = None; visual = []; frames = []
    try:
        import carb
        memory_policy=configure_replay_memory(carb.settings.get_settings(),getattr(args,'texture_streaming_budget',.35))
        import omni.timeline
        from pxr import UsdGeom, UsdPhysics
        from scripts.start_crazyflie_dual_camera_view import open_preview
        print('[factory replay] Opening scene and camera render products...',flush=True)
        preview, _ = open_preview(app, scene_config=config, width=RGB_SIZE[0], height=RGB_SIZE[1],
            thermal_width=THERMAL_SIZE[0], thermal_height=THERMAL_SIZE[1], fast=False,
            start_controller=False, show_ui=False,disable_external_viewport=True)
        memory_policy=configure_replay_memory(carb.settings.get_settings(),getattr(args,'texture_streaming_budget',.35))
        preview.metadata['offline_memory_policy']=memory_policy
        print('[factory replay] Camera render products ready; freezing source pose...',flush=True)
        stage = preview.stage; stage.SetEditTarget(stage.GetSessionLayer())
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        timeline = omni.timeline.get_timeline_interface(); timeline.pause(); timeline.set_auto_update(False)
        restorer = FrozenPoseRestorer(stage, config)
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        source_indices = {r['frame_id']: i for i, r in enumerate(records)}
        target = provenance['task']['target_prim_path']
        for index, record in enumerate(selected):
            timestamp = float(record['simulation_time']); timeline.set_current_time(timestamp); timeline.commit()
            desired, quat = restorer.restore(record); app.update()
            if index==0:print('[factory replay] Capturing first frozen frame...',flush=True)
            rgb, thermal, meta = preview.capture(render=True, simulation_time=timestamp, frozen_state=True)
            validate_capture(meta, record)
            errors = compare_pose_matrices(world_matrix(restorer.body), desired, meters_per_unit)
            if not pose_within_tolerance(errors):
                raise ValueError(f'body pose restore failed: {errors}')
            last = preview.thermal._last
            renderer_labels = hotspot_labels(last.labels,
                provenance['task'].get('environment_evidence',{}).get('factory_inspection',{}).get('hotspot_prim_paths') or [target])
            meta.update({
                'frame_id': index, 'trajectory_frame_id': record['frame_id'],
                'trajectory_index': source_indices[record['frame_id']],
                'replay_simulation_time': timestamp, 'replay_source': str(args.trajectory.resolve()),
                'trajectory_sha256': provenance['trajectory_sha256'], 'scene_config_sha256': config.sha256,
                'source_stage_sha256': sha256(config.scene.stage),
                'task_id': provenance['task']['task_id'], 'task_result_status': provenance['task_result']['status'],
                'action_type': record.get('action_type'),
                'position': desired[3, :3].tolist(), 'orientation': quat,
                'recorded_body_world_matrix': record['recorded_body_world_matrix'],
                'restored_body_world_matrix': world_matrix(restorer.body).tolist(),
                'recorded_rgb_camera_world_matrix': record['rgb_camera_world_matrix'],
                'source_trajectory_pose': {k: record[k] for k in ('position', 'orientation') if k in record},
                'replay_pose_source': restorer.last_pose_source, 'pose_restore_validation': {'body': errors},
                'pose_restore_mode': 'session_layer_frozen_usd_transform',
                'hotspot_renderer_labels': renderer_labels, **development_labels(provenance,manual_only=args.manual_only),
            })
            if record['frame_id'] in observations:
                id_path = f'frame_{index:06d}_instance_id.npy'
                np.save(args.out / id_path, np.asarray(last.instance_id))
                evidence = {'frame_id': record['frame_id'], 'simulation_time': timestamp,
                    'trajectory_sha256': provenance['trajectory_sha256'], 'scene_config_sha256': config.sha256,
                    'hotspot_prim_path': target, 'camera_world_matrix': meta['camera_world_matrix'],
                    'evidence_image': f'frame_{index:06d}_rgb.png',
                    'thermal_evidence_image': f'frame_{index:06d}_thermal.png',
                    'hotspot_instance_image': id_path,
                    'strict_sync_verified': False,
                    **hotspot_evidence(last.instance_id, renderer_labels, target, args.minimum_hotspot_pixels,
                        target_paths=provenance['task'].get('environment_evidence',{}).get('factory_inspection',{}).get('hotspot_prim_paths'))}
                visual.append(evidence)
                meta['factory_visual_evidence'] = evidence
            preview.save_frame(args.out, index, rgb, thermal, meta)
            validate_saved_frame(args.out, index)
            frames.append(index)  # Keep counts, not hundreds of copies of renderer label tables.
            write_json(args.out/'progress.json',{'status':'IN_PROGRESS','frames_written':len(frames),
                'expected_frames':len(selected),'last_simulation_time_s':timestamp})
            if index % 10 == 0:
                print(f'[factory replay] {index+1}/{len(selected)} t={timestamp:.3f}s', flush=True)
        if trajectory_provenance(args.trajectory) != provenance:
            raise ValueError('trajectory/task source changed during rendering')
        if config.sensors.material_directory is not None:
            from scripts.runtime_parameter_provenance import validate_thermal_inputs
            validate_thermal_inputs(config,provenance.get('thermal_input_snapshot'))
        report = {'result': 'PASS', 'pipeline': 'factory_inspection_offline_v1',
            'offline_memory_policy':memory_policy,
            'trajectory': str(args.trajectory.resolve()), 'trajectory_provenance': provenance,
            'capture_contract': provenance.get('capture_contract'),
            'replay_capture_contract': replay_capture_contract(args.route_interval),
            'task': provenance['task'], 'frames': len(frames), 'resolution': list(RGB_SIZE),
            'scene_config_path': str(config.source_path), 'scene_config_sha256': config.sha256,
            'stage_path': str(config.scene.stage), 'source_stage_sha256': sha256(config.scene.stage),
            'sensor_config_path': str(config.sensors.sensor_config),
            'sensor_config_sha256': sha256(config.sensors.sensor_config),
            'thermal_config_path': str(config.sensors.thermal_config),
            'thermal_config_sha256': sha256(config.sensors.thermal_config),
            'thermal_environment_config_path': str(config.scene.thermal_scene),
            'thermal_environment_config_sha256': sha256(config.scene.thermal_scene),
            'display_config_path': (str(config.sensors.display_config) if config.sensors.display_config else None),
            'display_config_sha256': (sha256(config.sensors.display_config) if config.sensors.display_config else None),
            'sample_interval_s': args.route_interval, 'sampling_plan': plan,
            'source_input_hashes': plan['source_input_hashes'],
            'observation_frame_ids': sorted(observations),
            'frame_metadata': preview.frame_metadata_summary(args.out),
            'experiment_conditions': preview.metadata.get('experiment_conditions'),
            'note': 'PASS means sampled offline replay/output checks; task success and strict sync remain separate.',
            'rgb_resolution': list(RGB_SIZE), 'thermal_resolution': list(THERMAL_SIZE),
            **development_labels(provenance,manual_only=args.manual_only)}
        (args.out / 'visual_evidence.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in visual), encoding='utf8')
        write_json(args.out / 'replay_metadata.json', report)
        write_frame_order(args.out,selected)
        validate_bundle(args.out)
        write_json(args.out/'progress.json',{'status':'COMPLETE','frames_written':len(frames),'expected_frames':len(selected)})
        print(f'[factory replay] PASS: {args.out}', flush=True)
        return 0
    except Exception:
        (args.out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf8')
        raise
    finally:
        if preview:
            preview.close()
        app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/factory_inspection_scene.yaml')
    parser.add_argument('--trajectory', type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--route-interval', type=float, default=1.)
    parser.add_argument('--texture-streaming-budget',type=float,default=.35)
    parser.add_argument('--minimum-hotspot-pixels', type=int, default=20)
    parser.add_argument('--manual-only',action='store_true',help='preview replay with recorded manual labels; no inspection observation gate')
    parser.add_argument('--manual-summary',action='store_true',help='write a recorded manual-confirmation summary without automatic inspection evaluation')
    parser.add_argument('--plan-only', action='store_true', help='pure source/provenance/selection checks; never imports Kit')
    parser.add_argument('--validate-only', type=Path, help='pure checks of a completed factory replay')
    args = parser.parse_args(argv)
    if args.minimum_hotspot_pixels <= 0:
        parser.error('--minimum-hotspot-pixels must be positive')
    if args.manual_summary:
        if args.out is None:parser.error('--out is required with --manual-summary')
        from scripts.trajectory_recorder import trajectory_provenance
        report=manual_confirmation_report(trajectory_provenance(args.trajectory))
        args.out.mkdir(parents=True,exist_ok=False)
        write_json(args.out/'manual_confirmation.json',report)
        print(json.dumps(report,ensure_ascii=False));return 0
    if args.validate_only:
        print(json.dumps(validate_bundle(args.validate_only), indent=2)); return 0
    return render(args)


if __name__ == '__main__':
    raise SystemExit(main())
