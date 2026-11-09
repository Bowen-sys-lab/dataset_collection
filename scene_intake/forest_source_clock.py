"""Keep thermal evolution on source time while forest USD animation loops."""
from contextlib import contextmanager
import math


def recorded_animation_time_code(seconds,fps):
    if not math.isfinite(seconds) or not math.isfinite(fps) or fps<=0:
        raise ValueError('Recorded source time and positive frame rate must be finite')
    return seconds*fps


def recorded_animation_time_code(seconds,fps):
    if not math.isfinite(seconds) or not math.isfinite(fps) or fps<=0:
        raise ValueError('Recorded source time and positive frame rate must be finite')
    return seconds*fps


@contextmanager
def source_thermal_clock(preview, source_time):
    source_time=float(source_time)
    if not math.isfinite(source_time):raise ValueError('Source thermal time must be finite')
    original=preview.clock
    class SourceClock:
        def sample(self, animation_time):
            return original.sample(source_time)
        def __getattr__(self,name):
            return getattr(original,name)
    preview.clock=SourceClock()
    try:yield
    finally:preview.clock=original
