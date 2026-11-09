"""Validate actual restored body transforms for the new Episode replay contract."""
import numpy as np


def check_body_replay(actual,record):
    expected=np.asarray(record.get('recorded_body_world_matrix'),dtype=float)
    actual=np.asarray(actual,dtype=float)
    if (expected.shape!=(4,4) or actual.shape!=(4,4) or not np.isfinite(expected).all()
        or not np.isfinite(actual).all() or not np.allclose(actual,expected,rtol=0,atol=1e-5)):
        raise ValueError('Rendered body pose differs from the recorded body transform')
    return {'restored':True,'source':'recorded_body_world_transform',
            'maximum_transform_abs_error':float(np.max(np.abs(actual-expected))),
            'tolerance':1e-5}
