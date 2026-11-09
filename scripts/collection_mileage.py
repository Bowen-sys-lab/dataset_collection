"""Scene-independent sampled body mileage. Length is guidance, never a task gate."""
import math
from scripts.collection_capture_defaults import maximum_record_gap


def length_band(length):
    return 'BELOW_TARGET' if length < 100 else 'ABOVE_TARGET' if length > 200 else 'IN_RANGE'


class MileageTracker:
    def __init__(self, max_gap_s=.25, max_speed_m_s=8.):
        self.max_gap_s = max_gap_s
        self.max_speed_m_s = max_speed_m_s
        self.length_3d_m = self.length_xy_m = 0.
        self.previous = None
        self.issues = {}

    def issue(self, reason):
        self.issues[reason] = self.issues.get(reason, 0) + 1

    def update(self, sim_time, position):
        try:
            stamp = float(sim_time)
            point = tuple(float(v) for v in position)
            valid = len(point) == 3 and math.isfinite(stamp) and all(map(math.isfinite, point))
        except (ValueError, TypeError, OverflowError):
            valid = False
        if not valid:
            self.issue('nonfinite_or_invalid_pose')
            self.previous = None
            return
        if self.previous is not None:
            old_time, old_point = self.previous
            dt = stamp - old_time
            distance = math.dist(point, old_point)
            if dt == 0 and distance <= 1e-8:
                return
            reason = ('time_rollback' if dt < 0 else 'motion_without_time' if dt == 0
                      else 'sampling_gap' if dt > self.max_gap_s + 1e-8
                      else 'implausible_motion' if distance > self.max_speed_m_s * dt + .1 else None)
            if reason:
                self.issue(reason)
            else:
                self.length_3d_m += distance
                self.length_xy_m += math.dist(point[:2], old_point[:2])
        self.previous = stamp, point

    def summary(self):
        return {'length_3d_m': self.length_3d_m, 'length_xy_m': self.length_xy_m,
                'length_band': length_band(self.length_3d_m), 'target_range_m': [100, 200],
                'measurement_status': 'INCOMPLETE' if self.issues or self.previous is None else 'COMPLETE',
                'issues': dict(self.issues), 'task_gate_applied': False,
                'near_upper_target': self.length_3d_m >= 180,
                'semantics': 'sampled body polyline; incomplete measurements are partial sums',
                'maximum_gap_s': self.max_gap_s, 'maximum_plausible_speed_m_s': self.max_speed_m_s}

    def status_text(self, target=None):
        label = 'INCOMPLETE' if self.issues else length_band(self.length_3d_m)
        warning = ' | approaching 200m' if 180 <= self.length_3d_m <= 200 else ''
        anchor = ''
        if target is not None and self.previous is not None:
            try:
                value=math.dist(self.previous[1],target)
                if math.isfinite(value):anchor=f' | Target anchor {value:.1f}m'
            except (ValueError,TypeError):pass
        return (f'Body path {self.length_3d_m:.1f}m | XY {self.length_xy_m:.1f}m | '
                f'{label}{warning}{anchor} | goal 100-200m; T always manual')


def summarize_mileage(records, metadata):
    gap = maximum_record_gap(metadata.get('sample_period_s', .1))
    raw = MileageTracker(max_gap_s=gap)
    for row in records:
        raw.update(row.get('simulation_time'), row.get('position'))
    result = metadata.get('task_result') or {}
    confirmed = None
    if result.get('status') == 'success' and result.get('marked_by') == 'keyboard:T':
        tracker = MileageTracker(max_gap_s=gap)
        try:
            stamp = float(result['simulation_time_s'])
            if not math.isfinite(stamp):
                raise ValueError('nonfinite confirmation')
            for row in records:
                if float(row['simulation_time']) <= stamp:
                    tracker.update(row['simulation_time'], row.get('position'))
            tracker.update(stamp, result.get('drone_position_m'))
        except (KeyError, ValueError, TypeError, OverflowError):
            tracker.issue('invalid_confirmation_event')
        confirmed = dict(tracker.summary(), endpoint_source='recorded_keyboard_T_event')
    return {'schema_version': 1, 'raw_recording': raw.summary(),
            'to_confirmation': confirmed, 'automatic_task_acceptance': 'NOT_EVALUATED'}
