"""Live-only sparse transform-stack optimization; projection math stays vendor-owned."""
import numpy as np
from contextlib import contextmanager


def compact_motion(points, ids, previous, current, camera_previous, camera_current,
                   intrinsics, distortion):
    from irsim.optics.motion import image_plane_motion
    ids = np.asarray(ids)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError('instance IDs must be integers')
    clipped = np.clip(ids, 0, len(previous)-1)
    # Keep background at index zero. Only invert transforms used by this frame;
    # many instance IDs are sparse, so the vendor's full stack includes thousands
    # of unused identity matrices. Camera movement still reaches every pixel.
    present = np.unique(np.concatenate((np.array([0], dtype=clipped.dtype), clipped.ravel())))
    remapped = np.searchsorted(present, clipped)
    return image_plane_motion(points, remapped, np.asarray(previous)[present],
                              np.asarray(current)[present], camera_previous, camera_current,
                              intrinsics, distortion)


@contextmanager
def compact_motion_scope():
    from irsim_isaac.pipeline import motion_isaac
    original = motion_isaac.image_plane_motion
    motion_isaac.image_plane_motion = compact_motion
    try:
        yield
    finally:
        motion_isaac.image_plane_motion = original
