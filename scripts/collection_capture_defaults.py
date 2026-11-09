"""Defaults for new collections; historical data retain their recorded settings."""
import math

# Offline outputs only; live preview dimensions remain scene-controlled.
RGB_SIZE = (256, 256)
THERMAL_SIZE = (256, 256)
TRAJECTORY_PERIOD_S = 1.0
SAMPLE_PERIOD_S = 1.0  # Offline image sampling only; independent of raw recording.


def capture_contract(trajectory_period_s=TRAJECTORY_PERIOD_S):
    """Snapshot the requested offline format separately from raw state cadence."""
    maximum_record_gap(trajectory_period_s)
    return {'schema_version': 1, 'contract_id': 'paired_offline_v3',
            'trajectory_period_s': trajectory_period_s,
            'offline_sample_interval_s': SAMPLE_PERIOD_S,
            'rgb_resolution': list(RGB_SIZE), 'thermal_resolution': list(THERMAL_SIZE),
            'action_schema': 'factory_pose_action_v2',
            'terminal_policy': 'exact_keyboard_T_snapshot_when_present',
            'preview_policy': 'scene_preview_unchanged'}


def replay_capture_contract(sample_interval_s=SAMPLE_PERIOD_S, *, rgb_size=RGB_SIZE, thermal_size=THERMAL_SIZE):
    """Describe output settings for this replay, independent of source history."""
    if (isinstance(sample_interval_s, bool) or not isinstance(sample_interval_s, (int, float))
            or not math.isfinite(sample_interval_s) or sample_interval_s <= 0):
        raise ValueError('sample_interval_s must be a finite positive number')
    return {'schema_version': 1,
            'contract_id': ('paired_offline_v3' if tuple(rgb_size) == RGB_SIZE
                            and tuple(thermal_size) == THERMAL_SIZE else 'paired_offline_custom'),
            'offline_sample_interval_s': sample_interval_s,
            'rgb_resolution': list(rgb_size), 'thermal_resolution': list(thermal_size),
            'preview_policy': 'scene_preview_unchanged'}


def maximum_record_gap(sample_period_s=.1):
    """Allow scheduling jitter, but reject a missing 1 Hz sample.

    Missing metadata retains the legacy 0.25 s continuity threshold.
    """
    if (isinstance(sample_period_s, bool) or not isinstance(sample_period_s, (int, float))
            or not math.isfinite(sample_period_s) or sample_period_s <= 0):
        raise ValueError('sample_period_s must be a finite positive number')
    return max(.25, 1.25 * sample_period_s)


if __name__ == '__main__':
    import json
    print(json.dumps(capture_contract()))
