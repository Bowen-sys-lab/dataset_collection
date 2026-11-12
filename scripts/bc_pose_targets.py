"""Future realized pose targets for BC; independent of sampled keyboard commands."""
from bisect import bisect_left
import math
from scripts.collection_capture_defaults import maximum_record_gap
import numpy as np


def wrap_yaw(value):
    return math.atan2(math.sin(value), math.cos(value))


def validate_pose(state):
    matrix = np.asarray(state.get('recorded_body_world_matrix'), dtype=float)
    position = np.asarray(state.get('position'), dtype=float)
    clock, yaw = state['simulation_time'], state['yaw']
    if (isinstance(clock, bool) or isinstance(yaw, bool) or not math.isfinite(clock) or not math.isfinite(yaw)
        or matrix.shape != (4,4) or position.shape != (3,) or not np.isfinite(matrix).all()
        or not np.isfinite(position).all() or not np.allclose(matrix[3,:3], position, rtol=0, atol=1e-5)
        or not np.allclose(matrix[:,3], [0,0,0,1], rtol=0, atol=1e-5)):
        raise ValueError('Recorded body pose/time/yaw is invalid or inconsistent')
    rotation = matrix[:3,:3]
    if (not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-5)
        or abs(np.linalg.det(rotation)-1.) > 1e-5
        or abs(wrap_yaw(math.atan2(rotation[0,1], rotation[0,0])-yaw)) > 1e-5):
        raise ValueError('Require a rigid metre body transform matching recorded heading')
    return position


def future_pose_target(first, later, *, initial_yaw):
    start, end = validate_pose(first), validate_pose(later)
    duration = later['simulation_time']-first['simulation_time']
    if duration <= 0 or not math.isfinite(initial_yaw):
        raise ValueError('Future endpoint must strictly follow observation time')
    delta = end-start
    c, s = math.cos(first['yaw']), math.sin(first['yaw'])
    forward, left, up = float(c*delta[0]+s*delta[1]), float(-s*delta[0]+c*delta[1]), float(delta[2])
    yaw_delta = wrap_yaw(later['yaw']-first['yaw'])
    if not all(math.isfinite(v) for v in [duration,forward,left,up,yaw_delta]):
        raise ValueError('Realized motion exceeds supported numeric range')
    hold = float(np.linalg.norm(delta)) <= 1e-4 and abs(yaw_delta) <= 1e-4
    return {'forward_m': forward, 'left_m': left, 'up_m': up, 'yaw_delta_rad': yaw_delta,
            'duration_s': duration, 'relative_start_yaw_rad': wrap_yaw(first['yaw']-initial_yaw),
            'motion_kind': 'HOLD' if hold else 'MOVE', 'stop': False}


def select_future_endpoint(states, index, *, horizon_s=.5, sample_period_s=.1):
    if not math.isfinite(horizon_s) or not 0 < horizon_s <= 2:
        raise ValueError('Horizon must be in (0,2] simulation seconds')
    if (not states or states[-1].get('action_type') != 'STOP'
        or states[-1].get('source_event_ref') != 'run_metadata.json#terminal_snapshot'):
        raise ValueError('Require exact terminal T; no invented endpoint')
    maximum_gap = maximum_record_gap(sample_period_s)
    times = [s['simulation_time'] for s in states]
    if any(not math.isfinite(t) for t in times) or any(b <= a or b-a > maximum_gap+1e-8 for a,b in zip(times,times[1:])):
        raise ValueError('Source clock is not contiguous and strictly increasing')
    if not 0 <= index < len(states)-1:
        raise ValueError('No future action exists after exact terminal event')
    target = min(times[-1], times[index]+horizon_s)
    endpoint = bisect_left(times, target-1e-8)
    return states[max(index+1, min(endpoint,len(states)-1))]
