"""Technical checks of raw records and the explicitly recovered collection branch."""
import math
import numpy as np
from scripts.collection_capture_defaults import maximum_record_gap
from scripts.recording_undo import active_branch


def recording_quality(records, *, sample_period_s=.1, metadata=None):
    errors = []
    if not records:
        return {'result': 'FAIL', 'issues': ['empty recording'], 'record_count': 0}
    metadata = metadata or {}
    times = np.asarray([r.get('simulation_time', math.nan) for r in records], dtype=float)
    gaps = np.diff(times)
    maximum_gap = maximum_record_gap(sample_period_s)
    if not np.isfinite(times).all() or (gaps <= 0).any():
        errors.append('nonfinite or non-increasing simulation time')
    if [r.get('frame_id') for r in records] != list(range(len(records))):
        errors.append('trajectory frame IDs not contiguous')
    for row in records:
        for key in ('recorded_body_world_matrix', 'rgb_camera_world_matrix', 'thermal_camera_world_matrix'):
            value = np.asarray(row.get(key), dtype=float)
            if value.shape != (4, 4) or not np.isfinite(value).all():
                errors.append(f'invalid {key}')

    exempt = set()
    branch = records
    events = metadata.get('collection_edit_events', [])
    if metadata.get('collision_pending'):
        errors.append('unresolved collision')
    if events:
        try:
            pending = None
            previous_stamp = -math.inf
            previous_boundary = 0
            resume_boundaries = {}
            for index, event in enumerate(events):
                stamp = float(event['simulation_time_s'])
                if not math.isfinite(stamp) or stamp < previous_stamp or event['event_id'] != index:
                    raise ValueError('invalid edit event clock or identity')
                previous_stamp = stamp
                if event['type'] == 'collision' and pending is None:
                    pending = stamp
                elif event['type'] == 'undo' and pending is not None:
                    boundary = event['next_raw_frame_id']
                    if type(boundary) is not int or not previous_boundary <= boundary <= len(records) or boundary < 1:
                        raise ValueError('invalid recovery sample boundary')
                    # Another collision can happen during the one-second
                    # sampling delay, before any new raw row exists.
                    before = resume_boundaries.get(boundary, times[boundary-1])
                    if not 0 <= pending-before <= maximum_gap+1e-9 or stamp < before:
                        raise ValueError('collision is not adjacent to its last raw sample')
                    anchor = event['anchor_frame_id']
                    if type(anchor) is not int or not 0 <= anchor < boundary:
                        raise ValueError('invalid recovery anchor')
                    if abs(times[anchor]-event['anchor_source_time_s']) > 1e-8:
                        raise ValueError('recovery anchor source clock differs')
                    resume_boundaries[boundary] = stamp
                    previous_boundary = boundary
                    pending = None
                else:
                    raise ValueError('collision and undo events must alternate')
            if pending is not None:
                raise ValueError('unresolved collision')
            for boundary, stamp in resume_boundaries.items():
                if boundary < len(records):
                    if not sample_period_s-1e-8 <= times[boundary]-stamp <= maximum_gap+1e-8:
                        raise ValueError('invalid first sample after recovery')
                    exempt.add(boundary-1)
            branch, _ = active_branch(records, metadata)
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            errors.append(f'invalid recovery: {exc}')
    if any(gap > maximum_gap+1e-9 and i not in exempt for i, gap in enumerate(gaps)):
        errors.append(f'trajectory sampling gap exceeds {maximum_gap:g} simulation seconds')
    corrected_gaps = np.diff([r['simulation_time'] for r in branch])
    if len(corrected_gaps) and (not np.isfinite(corrected_gaps).all() or
                               (corrected_gaps <= 0).any() or (corrected_gaps > maximum_gap+1e-8).any()):
        errors.append('invalid corrected branch sampling')
    return {'result': 'FAIL' if errors else 'PASS', 'issues': sorted(set(errors)),
            'record_count': len(records), 'corrected_record_count': len(branch),
            'sample_gap_mean_s': float(gaps.mean()) if len(gaps) else None,
            'sample_gap_max_s': float(gaps.max()) if len(gaps) else None,
            'corrected_sample_gap_max_s': float(corrected_gaps.max()) if len(corrected_gaps) else None,
            'recovery_gap_count': len(exempt), 'task_acceptance': 'NOT_EVALUATED'}
