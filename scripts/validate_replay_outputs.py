"""Validate file, array, timeline, camera-follow and source-hash checks for a replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_replay_video import load_replay
from scripts.frame_metadata import FrameMetadataReader
from scripts.sync_validation_metadata import (
    _load_verified_evidence, frame_sync_contract_passes, frame_sync_validation_passes,
)


def validate_replay(output: Path):
    output = output.resolve()
    replay = json.loads((output/'replay_metadata.json').read_text(encoding='utf-8'))
    frames = int(replay['frames'])
    legacy_resolution = replay.get('resolution')
    rgb_resolution = replay.get('rgb_resolution', legacy_resolution)
    thermal_resolution = replay.get('thermal_resolution', legacy_resolution or rgb_resolution)
    if rgb_resolution is None or thermal_resolution is None:
        raise ValueError('replay metadata must contain RGB and thermal resolutions')
    rgb_width, rgb_height = map(int, rgb_resolution)
    thermal_width, thermal_height = map(int, thermal_resolution)
    failures = []
    provenance = None
    try:
        bundle = load_replay(output)
        provenance = bundle.metadata['trajectory_provenance']
    except (ValueError, OSError, KeyError) as exc:
        failures.append(f'trajectory/task/image correspondence: {exc}')
    times, matrices = [], []
    finite_counts = {'dn16': 0, 'radiance': 0, 'apparent_t': 0}
    missing = []
    ids = []
    frame_sync_checks = []
    frame_metadata_reader = FrameMetadataReader()
    for index in range(frames):
        stem = f'frame_{index:06d}'
        required = [f'{stem}_rgb.png', f'{stem}_thermal.png', f'{stem}_rgb.npy',
                    f'{stem}_display8.npy', f'{stem}_dn16.npy', f'{stem}_radiance.npy',
                    f'{stem}_apparent_t.npy', f'{stem}.json']
        missing.extend(name for name in required if not (output/name).is_file())
        meta_path = output/f'{stem}.json'
        if not meta_path.is_file():
            continue
        try:
            meta = frame_metadata_reader.read(meta_path)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f'{stem}: frame metadata could not be read: {exc}')
            continue
        ids.append(meta.get('frame_id'))
        frame_sync_checks.append(meta)
        if meta.get('frame_id') != index:
            failures.append(f'{stem}: frame_id mismatch')
        times.append(float(meta['replay_simulation_time']))
        matrices.append(np.asarray(meta['camera_world_matrix'], dtype=np.float64))
        frame_rgb = meta.get('rgb_resolution', meta.get('resolution', [rgb_width, rgb_height]))
        frame_thermal = meta.get('thermal_resolution', meta.get('resolution', [thermal_width, thermal_height]))
        if list(frame_rgb) != [rgb_width, rgb_height]:
            failures.append(f'{stem}: RGB resolution metadata does not match replay metadata')
        if list(frame_thermal) != [thermal_width, thermal_height]:
            failures.append(f'{stem}: thermal resolution metadata does not match replay metadata')
        for kind in ('display8', 'dn16', 'radiance', 'apparent_t'):
            path = output/f'{stem}_{kind}.npy'
            if not path.is_file():
                continue
            arr = np.load(path, mmap_mode='r')
            expected_2d = (thermal_height, thermal_width)
            if tuple(arr.shape[:2]) != expected_2d:
                failures.append(f'{stem}_{kind}: shape {arr.shape} != {expected_2d}')
            if kind in finite_counts:
                finite = bool(np.isfinite(arr).all())
                finite_counts[kind] += int(finite)
                if not finite:
                    failures.append(f'{stem}_{kind}: non-finite values')
        rgb_path = output/f'{stem}_rgb.npy'
        if rgb_path.is_file():
            rgb = np.load(rgb_path, mmap_mode='r')
            expected_rgb_2d = (rgb_height, rgb_width)
            if tuple(rgb.shape[:2]) != expected_rgb_2d:
                failures.append(f'{stem}_rgb: shape {rgb.shape} != {expected_rgb_2d}')
        from PIL import Image
        for kind, expected_size in (('rgb', (rgb_width, rgb_height)),
                                    ('thermal', (thermal_width, thermal_height))):
            image_path = output/f'{stem}_{kind}.png'
            if image_path.is_file():
                with Image.open(image_path) as image:
                    if image.size != expected_size:
                        failures.append(f'{stem}_{kind}.png: size {image.size} != {expected_size}')
    if missing:
        failures.extend(f'missing file: {name}' for name in missing)
    times_increasing = len(times) == frames and all(b > a for a, b in zip(times, times[1:]))
    if not times_increasing:
        failures.append('simulation timestamps are not strictly increasing')
    camera_change = (float(np.max(np.abs(matrices[-1]-matrices[0])))
                     if len(matrices) > 1 else 0.0)
    stage_path = Path(replay['scene_config_path']).parent.parent / 'school_vln_01.usd'
    # Prefer the path encoded in each frame, which is the actual opened stage path.
    first_meta_path = output/'frame_000000.json'
    if first_meta_path.is_file():
        try:
            first_meta = frame_metadata_reader.read(first_meta_path)
            stage_path = Path(first_meta['stage_path'])
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f'first frame metadata could not provide the source stage path: {exc}')
    stage_hash_now = hashlib.sha256(stage_path.read_bytes()).hexdigest() if stage_path.is_file() else None
    source_unchanged = stage_hash_now == replay['source_stage_sha256']
    if not source_unchanged:
        failures.append('source USD hash does not match replay metadata')
    recording_stage_hash = (provenance or {}).get('recording_stage_sha256')
    recorded_stage_matches = (None if recording_stage_hash is None else
                              recording_stage_hash == replay['source_stage_sha256'])
    if recorded_stage_matches is False:
        failures.append('replay USD differs from the recorded source USD')
    sync_evidence = replay.get('sync_validation_evidence')
    sync_basis = replay.get('sync_basis', 'not_verified')
    try:
        current_sync_evidence = _load_verified_evidence()
        evidence_error = None
    except RuntimeError as exc:
        current_sync_evidence = None
        evidence_error = str(exc)
    # Project decision: offline pairing produced in a scene the school marker probe never
    # covered is used as synchronised data by convention. It must still pass its own
    # frame-level contract for every frame, and the strict claim stays false.
    convention_verified = bool(
        sync_basis == 'project_convention'
        and replay.get('sync_validation_status') == 'convention_only'
        and len(frame_sync_checks) == frames
        and all(frame_sync_contract_passes(frame) for frame in frame_sync_checks)
    )
    if sync_basis != 'project_convention' and current_sync_evidence is None:
        failures.append(f'validated synchronization evidence is unavailable: {evidence_error}')
    sync_verified = bool(
        replay.get('strict_sync_verified') is True
        and replay.get('sync_validation_status') == 'validated'
        and isinstance(sync_evidence, dict)
        and sync_evidence.get('result') == 'PASS'
        and current_sync_evidence is not None
        and len(frame_sync_checks) == frames
        and all(frame_sync_validation_passes(frame, current_sync_evidence) for frame in frame_sync_checks)
        and sync_evidence.get('report_sha256') == current_sync_evidence.get('report_sha256')
        and sync_evidence.get('report') == current_sync_evidence.get('report')
        and not failures
    )
    report = {
        'result': 'PASS_FILE_AND_ARRAY_CHECKS' if not failures else 'FAIL',
        'frames_expected': frames,
        'frames_validated': len(times),
        'resolution': [rgb_width, rgb_height],
        'rgb_resolution': [rgb_width, rgb_height],
        'thermal_resolution': [thermal_width, thermal_height],
        'frame_ids_contiguous': ids == list(range(frames)),
        'simulation_times_strictly_increasing': times_increasing,
        'camera_world_matrix_change_max_abs': camera_change,
        'camera_follow_motion_observed': camera_change > 1e-6,
        'finite_array_frames': finite_counts,
        'all_required_files_present': not missing,
        'source_usd_path': str(stage_path),
        'source_usd_sha256_metadata': replay['source_stage_sha256'],
        'source_usd_sha256_after': stage_hash_now,
        'source_usd_unchanged': source_unchanged,
        'recorded_source_usd_sha256': recording_stage_hash,
        'recorded_source_usd_matches_replay': recorded_stage_matches,
        'strict_sync_verified': sync_verified,
        'strict_per_frame_pixel_sync_verified': False,
        'sync_basis': sync_basis,
        'convention_sync_verified': convention_verified,
        'sync_status': (replay.get('sync_status', 'not_verified') if (sync_verified or convention_verified) else
                        'capture_validation_failed' if replay.get('sync_validation_status') == 'validated' else
                        replay.get('sync_status', 'not_verified')),
        'sync_validation_status': (replay.get('sync_validation_status', 'not_verified') if (sync_verified or convention_verified) else
                                   'capture_validation_failed' if replay.get('sync_validation_status') == 'validated' else
                                   replay.get('sync_validation_status', 'not_verified')),
        'sync_scope_failures': replay.get('sync_scope_failures', []),
        'sync_validation_scope': replay.get('sync_validation_scope'),
        'sync_validation_evidence': sync_evidence,
        'sync_scope': ('This report inherits capture-pipeline validation from replay metadata; '
                       'it does not run a new per-frame marker pixel test.'),
        'data_usage': replay.get('data_usage', 'pilot/development data'),
        'experiment_conditions': replay.get('experiment_conditions'),
        'trajectory_provenance': provenance,
        'task': provenance['task'] if provenance else None,
        'task_result': provenance['task_result'] if provenance else None,
        'trajectory_correspondence_verified': provenance is not None,
        'frame_metadata': replay.get('frame_metadata'),
        'failures': failures,
    }
    (output/'validation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    args = parser.parse_args(argv)
    report = validate_replay(args.replay)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['result'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
