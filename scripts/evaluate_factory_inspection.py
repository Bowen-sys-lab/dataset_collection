"""Evaluate a recorded factory flight without importing Isaac/Kit.

``evaluate_records`` measures body clearance from supplied world-space triangle
geometry. ``evaluate_samples`` is the pure boundary for an upstream geometry
measurement: its ``equipment_distance_m`` values already include body radius.
Neither API infers camera visibility from a body position or a manual T mark.

Visual JSONL has one object per trajectory sample (no header), with frame_id,
simulation_time, trajectory_sha256, scene_config_sha256, hotspot_prim_path,
hotspot_visible (bool), unobstructed (bool), camera_world_matrix (4x4 USD
row-vector transform), and evidence_image (existing path relative to the JSONL).
Rows must match the recorded time, hashes, target, and RGB camera transform.
This is a measured/reviewed visibility attestation, not image interpretation or
cryptographic certification. Missing/invalid rows cannot certify observation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.scene_config import load_scene_config


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _array(value, shape=None):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        return None
    return array


def _inspection(task):
    evidence = task.get('environment_evidence', {}) if isinstance(task, dict) else {}
    result = evidence.get('factory_inspection', {}) if isinstance(evidence, dict) else {}
    return result if isinstance(result, dict) else {}


def _requirements(inspection):
    return inspection.get('inspection_requirements', inspection.get('requirements', {}))


def _validated_requirements(requirements):
    result = {
        'minimum_equipment_distance_m': 5.0,
        'observation_distance_range_m': [6.0, 8.0],
        'hold_duration_s': 6.0,
        'reference_length_range_m': [180.0, 220.0],
        'body_envelope_radius_m': 0.18,
        'maximum_sample_gap_s': 0.25,
        'maximum_observation_speed_m_s': 0.3,
        'require_path_length': True,
    }
    if not isinstance(requirements, dict):
        raise ValueError('inspection requirements must be an object')
    result.update(requirements)
    if type(result['require_path_length']) is not bool:
        raise ValueError('require_path_length: expected boolean')
    for field in ('observation_distance_range_m', 'reference_length_range_m'):
        pair = _array(result[field], (2,))
        if pair is None or pair[0] < 0 or pair[1] < pair[0]:
            raise ValueError(f'{field}: expected finite nonnegative increasing pair')
        result[field] = pair.tolist()
    for field in ('minimum_equipment_distance_m', 'hold_duration_s', 'body_envelope_radius_m',
                  'maximum_sample_gap_s', 'maximum_observation_speed_m_s'):
        value = result[field]
        if not _finite_number(value) or value < 0:
            raise ValueError(f'{field}: expected finite nonnegative number')
    if result['hold_duration_s'] <= 0 or result['maximum_sample_gap_s'] <= 0:
        raise ValueError('hold duration and maximum sample gap must be positive')
    return result


def point_to_mesh_distances(points, vertices, triangles, *, body_radius_m=0.0):
    """Exact unsigned point-to-triangle distance minus a spherical body radius.

    Projection onto a triangle interior and all three bounded edges are tested.
    Degenerate faces reduce to edges/points. This is mesh surface clearance,
    not bounding-box/centroid distance or a claim of a swept-volume collision
    check between recorded samples. Geometry/points must be in world metres.
    """
    points = _array(points)
    vertices = _array(vertices)
    indices = np.asarray(triangles)
    if points is None or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must be a finite Nx3 array')
    if vertices is None or vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError('vertices must be a nonempty finite Nx3 array')
    if (indices.ndim != 2 or indices.shape[1] != 3 or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= len(vertices))):
        raise ValueError('triangles must be nonempty integer Mx3 vertex indices')
    if not _finite_number(body_radius_m) or body_radius_m < 0:
        raise ValueError('body radius must be finite and nonnegative')
    faces = vertices[indices]
    output = np.empty(len(points), dtype=np.float64)
    for point_index, point in enumerate(points):
        best = float('inf')
        # Bound temporary allocations for a full industrial mesh.
        for offset in range(0, len(faces), 8192):
            face = faces[offset:offset + 8192]
            a, b, c = face[:, 0], face[:, 1], face[:, 2]
            ab, ac = b - a, c - a
            normal = np.cross(ab, ac)
            normal_squared = np.einsum('ij,ij->i', normal, normal)
            nondegenerate = normal_squared > np.finfo(float).tiny
            ap = point - a
            numerator = np.einsum('ij,ij->i', ap, normal)
            factor = np.divide(numerator, normal_squared, out=np.zeros_like(numerator),
                               where=nondegenerate)
            projection = point - factor[:, None] * normal
            # Signed edge tests use the same triangle normal for either winding.
            inside = nondegenerate.copy()
            for start, end in ((a, b), (b, c), (c, a)):
                side = np.einsum('ij,ij->i', np.cross(end - start, projection - start), normal)
                inside &= side >= 0.0
                edge = end - start
                edge_squared = np.einsum('ij,ij->i', edge, edge)
                along = np.divide(np.einsum('ij,ij->i', point - start, edge), edge_squared,
                                  out=np.zeros(len(face)), where=edge_squared > 0)
                closest = start + np.clip(along, 0.0, 1.0)[:, None] * edge
                difference = point - closest
                best = min(best, float(np.min(np.einsum('ij,ij->i', difference, difference))))
            if np.any(inside):
                best = min(best, float(np.min(numerator[inside] ** 2 / normal_squared[inside])))
        output[point_index] = math.sqrt(max(0.0, best)) - body_radius_m
    return output


def _manual_confirmation(metadata):
    source = metadata.get('trajectory_source', 'legacy_unclassified')
    result = metadata.get('task_result', {})
    result = result if isinstance(result, dict) else {}
    marked = result.get('marked_by', metadata.get('success_marked_by'))
    status = result.get('status')
    if status is None:
        status = ('success' if metadata.get('success') is True else
                  'failed' if metadata.get('success') is False else 'unjudged')
    success = status == 'success'
    confirmed = source == 'manual_driving' and success and marked == 'keyboard:T'
    return {'confirmed': confirmed, 'trajectory_source': source, 'marked_by': marked,
            'status': 'PASS' if confirmed else 'FAIL'
            if source == 'manual_driving' and status == 'failed' else 'UNJUDGED'}


def _visual_rows(records, visual_evidence, trajectory_sha256, config_sha256, hotspot,
                 verified_image_paths):
    visual = {}
    duplicate_ids = set()
    for row in visual_evidence:
        if not isinstance(row, dict) or type(row.get('frame_id')) is not int:
            continue
        frame_id = row['frame_id']
        if frame_id in visual:
            duplicate_ids.add(frame_id)
        visual[frame_id] = row
    accepted, unknown, diagnostics = {}, 0, []
    for sample in records:
        frame_id = sample.get('frame_id')
        row = visual.get(frame_id)
        reason = None
        if row is None:
            reason = 'missing visual sample'
        elif frame_id in duplicate_ids:
            reason = 'duplicate visual frame'
        elif not trajectory_sha256 or not config_sha256:
            reason = 'missing trusted trajectory/config hashes'
        elif (row.get('trajectory_sha256') != trajectory_sha256
              or row.get('scene_config_sha256') != config_sha256):
            reason = 'visual trajectory/config hash mismatch'
        elif (not _finite_number(row.get('simulation_time'))
              or not _finite_number(sample.get('simulation_time'))
              or abs(row['simulation_time'] - sample['simulation_time']) > 1e-8):
            reason = 'visual simulation timestamp mismatch'
        elif not hotspot or row.get('hotspot_prim_path') != hotspot:
            reason = 'visual hotspot mismatch'
        elif type(row.get('hotspot_visible')) is not bool or type(row.get('unobstructed')) is not bool:
            reason = 'visibility/occlusion must be explicit booleans'
        else:
            recorded_camera = _array(sample.get('rgb_camera_world_matrix'), (4, 4))
            camera = _array(row.get('camera_world_matrix'), (4, 4))
            if (recorded_camera is None or camera is None
                    or not np.allclose(camera, recorded_camera, rtol=0.0, atol=1e-6)):
                reason = 'camera transform missing or mismatched'
            elif row.get('evidence_image') not in verified_image_paths:
                reason = 'evidence image missing or not verified'
        if reason:
            unknown += 1
            diagnostics.append({'frame_id': frame_id, 'reason': reason})
        else:
            accepted[frame_id] = row['hotspot_visible'] and row['unobstructed']
    return accepted, unknown, diagnostics


def _longest_hold(records, eligible, maximum_gap):
    start, previous_time, previous_frame, longest = None, None, None, 0.0
    for sample, good in zip(records, eligible):
        time, frame = sample.get('simulation_time'), sample.get('frame_id')
        finite = _finite_number(time) and type(frame) is int
        contiguous = (finite and previous_time is not None and frame == previous_frame + 1
                      and 0.0 < time - previous_time <= maximum_gap + 1e-9)
        if not finite or not good:
            start = None
        elif start is None or not contiguous:
            start = float(time)
        if start is not None:
            longest = max(longest, float(time) - start)
        previous_time = float(time) if finite else None
        previous_frame = frame if finite else None
    return longest


def evaluate_samples(records: Iterable[dict], requirements: dict, *, task=None,
                     run_metadata=None, visual_evidence=None, trajectory_sha256=None,
                     scene_config_sha256=None, verified_image_paths=None):
    """Evaluate explicit per-record mesh clearance; do not perform filesystem I/O.

    ``verified_image_paths`` is a set of evidence_image strings already checked
    by the caller. An empty set keeps visual evidence UNJUDGED. Frame spacing
    and timestamps must be strictly increasing; missing frame IDs/gaps reset
    holds. Actual recorded position differences and recorded velocity (if
    available) must both satisfy the speed limit. Units are metres/seconds.
    """
    samples = list(records)
    if any(not isinstance(row, dict) for row in samples):
        raise ValueError('every trajectory sample must be an object')
    req = _validated_requirements(requirements)
    task, metadata = task or {}, run_metadata or {}
    inspection = _inspection(task)
    issues, invalid, positions, distances, times, frames = [], [], [], [], [], []
    for index, row in enumerate(samples):
        position = _array(row.get('position'), (3,))
        distance = row.get('equipment_distance_m')
        time, frame = row.get('simulation_time'), row.get('frame_id')
        if position is None or not _finite_number(time) or type(frame) is not int:
            invalid.append(index)
        if index and (_finite_number(time) and _finite_number(times[-1])) and time <= times[-1]:
            invalid.append(index)
        if index and type(frame) is int and type(frames[-1]) is int and frame <= frames[-1]:
            invalid.append(index)
        if (scene_config_sha256 and row.get('scene_config_sha256') is not None
                and row['scene_config_sha256'] != scene_config_sha256):
            invalid.append(index)
        positions.append(position)
        distances.append(float(distance) if _finite_number(distance) else None)
        times.append(time)
        frames.append(frame)
    if not samples:
        invalid.append(-1)
    if invalid:
        issues.append('invalid/non-finite pose, timestamp, frame order, or record config hash')
    actual_length = sum(float(np.linalg.norm(b - a)) for a, b in zip(positions, positions[1:])
                        if a is not None and b is not None)
    waypoints = _array(inspection.get('reference_waypoints_m'))
    reference_length = None
    if waypoints is not None and waypoints.ndim == 2 and waypoints.shape[1] == 3 and len(waypoints) >= 2:
        reference_length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
    reference_declared = inspection.get('reference_path_length_m')
    declaration_matches = (_finite_number(reference_declared) and reference_length is not None
                           and abs(reference_declared - reference_length) <= 1e-3)
    length_min, length_max = req['reference_length_range_m']
    actual_pass = not invalid and length_min <= actual_length <= length_max
    reference_pass = bool(reference_length is not None and declaration_matches
                          and length_min <= reference_length <= length_max)
    finite_distances = [distance for distance in distances if distance is not None]
    safety_known = len(finite_distances) == len(samples) and bool(samples)
    safety = not invalid and safety_known and all(
        value >= req['minimum_equipment_distance_m'] for value in finite_distances)
    if not safety_known:
        issues.append('mesh distance missing/non-finite for one or more trajectory samples')
    low, high = req['observation_distance_range_m']
    eligible, speeds = [], []
    for index, row in enumerate(samples):
        speed = 0.0
        if index and positions[index] is not None and positions[index - 1] is not None:
            delta = times[index] - times[index - 1] if (
                _finite_number(times[index]) and _finite_number(times[index - 1])) else -1
            speed = float(np.linalg.norm(positions[index] - positions[index - 1])) / delta if delta > 0 else math.inf
        velocity = _array(row.get('linear_velocity'), (3,)) if 'linear_velocity' in row else None
        if 'linear_velocity' in row:
            speed = max(speed, float(np.linalg.norm(velocity))) if velocity is not None else math.inf
        speeds.append(speed if math.isfinite(speed) else None)
        eligible.append(index not in invalid and distances[index] is not None
                        and low <= distances[index] <= high
                        and speed <= req['maximum_observation_speed_m_s'] + 1e-9)
    measured_hold = _longest_hold(samples, eligible, req['maximum_sample_gap_s'])
    visual, unknown_count, visual_diagnostics = _visual_rows(
        samples, visual_evidence or [], trajectory_sha256, scene_config_sha256,
        inspection.get('hotspot_prim_path', task.get('target_prim_path')),
        set(verified_image_paths or []))
    verified_hold = _longest_hold(samples, [good and visual.get(row.get('frame_id')) is True
                                         for row, good in zip(samples, eligible)], req['maximum_sample_gap_s'])
    measured_hold_pass = measured_hold + 1e-9 >= req['hold_duration_s']
    verified_hold_pass = verified_hold + 1e-9 >= req['hold_duration_s']
    mileage_pass = not req['require_path_length'] or (actual_pass and reference_pass)
    distance_hold = bool(safety and mileage_pass and verified_hold_pass)
    manual = _manual_confirmation(metadata)
    visual_status = ('PASS' if verified_hold_pass else 'UNJUDGED'
                     if unknown_count or not safety_known else 'FAIL')
    # A measured failure is decisive; absent geometric/visual data stays unknown.
    known_failure = bool(invalid or not mileage_pass
                         or (safety_known and not safety)
                         or (safety_known and not measured_hold_pass)
                         or visual_status == 'FAIL' or manual['status'] == 'FAIL')
    result = ('FAIL' if known_failure else 'PASS' if distance_hold and manual['confirmed']
              else 'UNJUDGED')
    if req['require_path_length'] and not actual_pass:
        issues.append('actual flown path length does not meet the required range')
    if req['require_path_length'] and not reference_pass:
        issues.append('reference waypoint path/declared length does not meet the required range')
    if safety_known and not safety:
        issues.append('minimum equipment clearance was violated or trajectory is invalid')
    if not measured_hold_pass:
        issues.append('no continuous stationary distance-qualified hold of the required duration')
    if visual_status != 'PASS':
        issues.append('camera visibility and unobstructed target hold are not verified')
    return {
        'result': result, 'task_id': task.get('task_id'), 'task_provenance': 'recorded_run_snapshot',
        'qualification': 'manual_flight' if metadata.get('trajectory_source') == 'manual_driving' else 'development',
        'manual_confirmation': manual, 'distance_and_hold_pass': distance_hold,
        'safety_pass': bool(safety), 'safety_status': 'PASS' if safety else 'FAIL' if safety_known else 'UNJUDGED',
        'minimum_equipment_clearance_m': min(finite_distances) if finite_distances else None,
        'actual_path_length_m': actual_length, 'actual_path_length_pass': bool(actual_pass),
        'path_length_required': req['require_path_length'],
        'reference_path_length_m': reference_length, 'declared_reference_path_length_m': reference_declared,
        'reference_path_length_pass': reference_pass, 'reference_declaration_matches': declaration_matches,
        'distance_observation_duration_s': measured_hold,
        'verified_observation_duration_s': verified_hold,
        'visual_evidence': {'status': visual_status, 'accepted_sample_count': len(visual),
                            'missing_or_invalid_sample_count': unknown_count,
                            'diagnostics': visual_diagnostics},
        'requirements': req, 'invalid_sample_indices': sorted(set(invalid)), 'issues': issues,
        'measurement': {'distance': 'equipment_mesh_surface_minus_body_envelope_radius',
                        'observation_distance': 'body_envelope_to_equipment_surface',
                        'speed': 'maximum_of_position_difference_and_recorded_linear_velocity',
                        'time': 'recorded_simulation_time', 'continuous_between_samples': 'not_verified',
                        'whole_scene_collision': 'UNJUDGED',
                        'fov_and_occlusion': 'per_frame_external_visual_attestation'},
    }


def evaluate_records(records, task, equipment_vertices, equipment_triangles, **kwargs):
    """Pure mesh-backed evaluation using the recorded task snapshot."""
    samples = copy.deepcopy(list(records))
    requirements = _validated_requirements(_requirements(_inspection(task)))
    finite_indices, points = [], []
    for index, row in enumerate(samples):
        point = _array(row.get('position'), (3,))
        if point is not None:
            finite_indices.append(index)
            points.append(point)
        row.pop('equipment_distance_m', None)
    # Validate geometry even when every trajectory position is invalid.
    measured = point_to_mesh_distances(
        np.asarray(points).reshape((-1, 3)), equipment_vertices, equipment_triangles,
        body_radius_m=requirements['body_envelope_radius_m'])
    for index, distance in zip(finite_indices, measured):
        samples[index]['equipment_distance_m'] = float(distance)
    return evaluate_samples(samples, requirements, task=task, **kwargs)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path):
    result = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}: invalid JSON at line {number}') from exc
        if not isinstance(item, dict):
            raise ValueError(f'{path}: line {number} must be a JSON object')
        result.append(item)
    return result


def _geometry_path(root, inspection):
    value = inspection.get('geometry_file')
    if not isinstance(value, str) or not value.strip():
        return None
    relative = Path(value)
    candidate = (root / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(root):
        raise ValueError('recorded geometry_file must be repository-relative and remain inside repository')
    return candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--visual-evidence', type=Path)
    parser.add_argument('--repository-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        run, output, root = args.run.resolve(), args.out.resolve(), args.repository_root.resolve()
        if output.exists():
            raise ValueError(f'output directory already exists: {output}')
        config = load_scene_config(args.config, include_task=False)
        metadata_path, trajectory_path = run / 'run_metadata.json', run / 'trajectory.jsonl'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
        if not isinstance(metadata, dict) or not isinstance(metadata.get('task'), dict):
            raise ValueError('recorded run must contain its task snapshot')
        if metadata.get('scene_config_sha256') != config.sha256:
            raise ValueError('current scene config hash differs from recording or is missing')
        stage_hash = _sha256(config.scene.stage)
        if metadata.get('stage_sha256') != stage_hash:
            raise ValueError('current stage hash differs from recording or is missing')
        trajectory_hash = _sha256(trajectory_path)
        if metadata.get('trajectory_sha256', trajectory_hash) != trajectory_hash:
            raise ValueError('trajectory hash differs from recorded provenance')
        records = _read_jsonl(trajectory_path)
        task = metadata['task']
        inspection = _inspection(task)
        if not inspection:
            raise ValueError('recorded task lacks environment_evidence.factory_inspection metadata')
        visual, image_paths = [], set()
        if args.visual_evidence:
            visual_path = args.visual_evidence.resolve()
            visual = _read_jsonl(visual_path)
            for row in visual:
                image_path = row.get('evidence_image')
                if isinstance(image_path, str) and (visual_path.parent / image_path).is_file():
                    image_paths.add(image_path)
        kwargs = {'run_metadata': metadata, 'visual_evidence': visual,
                  'trajectory_sha256': trajectory_hash, 'scene_config_sha256': config.sha256,
                  'verified_image_paths': image_paths}
        geometry = _geometry_path(root, inspection)
        if geometry is not None and geometry.is_file():
            geometry_hash = _sha256(geometry)
            if inspection.get('geometry_sha256', geometry_hash) != geometry_hash:
                raise ValueError('geometry hash differs from recorded task provenance')
            with np.load(geometry, allow_pickle=False) as mesh:
                report = evaluate_records(records, task, mesh['equipment_vertices'], mesh['equipment_triangles'], **kwargs)
            geometry_info = {'path': str(geometry), 'sha256': geometry_hash, 'status': 'PASS'}
        else:
            # Never fall back to an AABB, centroid, or unverified recorded distances.
            clean_records = [{key: value for key, value in row.items() if key != 'equipment_distance_m'}
                             for row in records]
            report = evaluate_samples(clean_records, _requirements(inspection), task=task, **kwargs)
            report['issues'].append('equipment triangle geometry unavailable; clearance is UNJUDGED')
            geometry_info = {'path': str(geometry) if geometry else None, 'status': 'UNJUDGED'}
        report['provenance'] = {
            'trajectory_sha256': trajectory_hash, 'run_metadata_sha256': _sha256(metadata_path),
            'scene_config_sha256': config.sha256, 'stage_sha256': stage_hash,
            'recorded_task_config_sha256': metadata.get('task_config_sha256'),
            'task_snapshot_sha256': hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest(),
            'geometry': geometry_info,
        }
        output.mkdir(parents=True, exist_ok=False)
        (output / 'factory_evaluation.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        print(f"{report['result']}: {output / 'factory_evaluation.json'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'Factory evaluation refused: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
