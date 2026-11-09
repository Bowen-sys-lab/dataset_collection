"""Translate AeroVLA's reference NED action policy into sampled Z-up body poses."""
import math
import numpy as np


def plan_aerovla_action(position, yaw, action, *, cruise_speed_m_s=1.,
                       large_turn_vertical_speed_m_s=2., yaw_speed_rad_s=math.pi/4,
                       translation_yaw_threshold_rad=.25, step_s=1/60, settle_s=.5):
    position = np.asarray(position, dtype=float)
    forward, down, turn = [action[key] for key in ('forward_m', 'down_m', 'yaw_rad')]
    scalars = [yaw, forward, down, turn, cruise_speed_m_s, large_turn_vertical_speed_m_s,
               yaw_speed_rad_s, translation_yaw_threshold_rad, step_s, settle_s]
    if (position.shape != (3,) or not np.isfinite(position).all()
        or any(isinstance(value, (bool, np.bool_)) or not math.isfinite(value) for value in scalars)):
        raise ValueError('Action, position and motion parameters must be finite numbers')
    if not (0 <= forward <= 5 and -5 <= down <= 5 and -1.1 <= turn <= 1.1):
        raise ValueError('Action exceeds the public AeroVLA physical ranges')
    if (min(cruise_speed_m_s, large_turn_vertical_speed_m_s, yaw_speed_rad_s, step_s) <= 0
        or not 0 <= translation_yaw_threshold_rad <= math.pi or settle_s < 0):
        raise ValueError('Positive speeds/step and nonnegative settle time required')
    poses = [{'time_s': 0., 'position': position.tolist(), 'yaw': float(yaw)}]
    phases = []

    def append_phase(name, target_position, target_yaw, duration):
        if duration == 0:
            return
        if not math.isfinite(duration) or duration / step_s > 10000 - len(poses):
            raise ValueError('Motion would exceed the 10000-pose sampling budget')
        count = math.ceil(duration / step_s)
        origin = poses[-1]
        first = len(poses) - 1
        start_position = np.asarray(origin['position'])
        for index in range(1, count + 1):
            fraction = index / count
            with np.errstate(over='raise', invalid='raise'):
                point = start_position + fraction * (target_position - start_position)
            if not np.isfinite(point).all():
                raise ValueError('Motion produced nonfinite coordinates')
            poses.append({'time_s': origin['time_s'] + fraction * duration,
                          'position': point.tolist(),
                          'yaw': origin['yaw'] + fraction * (target_yaw - origin['yaw'])})
        phases.append({'name': name, 'first_pose': first, 'last_pose': len(poses)-1, 'duration_s': duration})

    target_yaw = yaw - turn  # NED positive yaw is clockwise; Isaac +Z yaw is counter-clockwise.
    if abs(turn) > .01 or forward > .01 or abs(down) > .01:
        append_phase('rotate', position, target_yaw, abs(turn) / yaw_speed_rad_s)
        if abs(turn) < translation_yaw_threshold_rad:
            offset = np.array([forward * math.cos(target_yaw), forward * math.sin(target_yaw), -down])
            distance = float(np.linalg.norm(offset))
            if distance > .01:
                duration = 1. if distance < 1 else distance / cruise_speed_m_s
                append_phase('translate', position + offset, target_yaw, duration)
        elif abs(down) > 0:
            append_phase('vertical_after_large_turn', position + np.array([0., 0., -down]),
                         target_yaw, abs(down) / large_turn_vertical_speed_m_s)
    last = poses[-1]
    append_phase('settle', np.asarray(last['position']), last['yaw'], settle_s)
    return {'poses': poses, 'phases': phases,
            'translation_executed': any(phase['name'] == 'translate' for phase in phases),
            'source_action': {'forward_m': forward, 'down_m': down, 'yaw_rad': turn},
            'policy': f'reference rotate-first; translate only for abs(turn)<{translation_yaw_threshold_rad}; otherwise vertical only',
            'coordinate_conversion': 'NED down and yaw signs inverted for Isaac Z-up'}
