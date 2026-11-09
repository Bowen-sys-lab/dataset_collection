"""Restore recorded body world poses in a session layer, without PhysX control."""
import numpy as np


MAX_POSE_TRANSLATION_ERROR_M = 1e-4
MAX_POSE_ROTATION_ERROR_DEGREES = 0.01


def compare_pose_matrices(actual, expected, meters_per_unit=1.0):
    """Measure pose difference in meters/degrees, independent of transform scale."""
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    if actual.shape != (4, 4) or expected.shape != (4, 4):
        raise ValueError('pose matrices must both be 4x4')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('pose matrices must contain only finite values')
    meters_per_unit = float(meters_per_unit)
    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError('meters_per_unit must be finite and positive')

    def rigid_rotation(matrix):
        # USD transforms may include scale. The orthogonal polar factor removes
        # that scale before comparing orientation.
        left, _, right = np.linalg.svd(matrix[:3, :3])
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        return rotation

    translation_error_m = float(
        np.linalg.norm(actual[3, :3] - expected[3, :3]) * meters_per_unit)
    relative_rotation = rigid_rotation(actual).T @ rigid_rotation(expected)
    cosine = np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0)
    rotation_error_degrees = float(np.degrees(np.arccos(cosine)))
    return {
        'translation_error_m': translation_error_m,
        'rotation_error_degrees': rotation_error_degrees,
        'matrix_max_abs_difference': float(np.max(np.abs(actual - expected))),
    }


def pose_within_tolerance(errors, max_translation_error_m=MAX_POSE_TRANSLATION_ERROR_M,
                          max_rotation_error_degrees=MAX_POSE_ROTATION_ERROR_DEGREES):
    return (errors['translation_error_m'] <= max_translation_error_m and
            errors['rotation_error_degrees'] <= max_rotation_error_degrees)


def _validated_transform(matrix, name):
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f'{name} must be a 4x4 matrix')
    if not np.isfinite(matrix).all():
        raise ValueError(f'{name} must contain only finite values')
    return matrix


def body_world_from_camera_world(camera_world, camera_in_body):
    """Recover body world pose from a recorded camera pose and fixed mount.

    USD uses row-vector transforms: camera_world = camera_in_body @ body_world.
    This fallback keeps older trajectories usable when they already stored the
    RGB camera matrix but not the body matrix.
    """
    camera_world = _validated_transform(camera_world, 'camera world transform')
    camera_in_body = _validated_transform(camera_in_body, 'camera-in-body transform')
    try:
        return np.linalg.inv(camera_in_body) @ camera_world
    except np.linalg.LinAlgError as exc:
        raise ValueError('camera-in-body transform is singular') from exc


def resolve_recorded_body_world(record, camera_in_body=None):
    """Return an exact or recoverable body pose, preserving legacy fallback."""
    body_matrix = record.get('recorded_body_world_matrix')
    if body_matrix is not None:
        return (_validated_transform(body_matrix, 'recorded body world transform').copy(),
                'recorded_body_world_transform')
    camera_matrix = record.get('rgb_camera_world_matrix')
    if camera_matrix is not None and camera_in_body is not None:
        return (body_world_from_camera_world(camera_matrix, camera_in_body),
                'legacy_rgb_camera_world_transform')
    return None, 'legacy_position_orientation'


def world_matrix(prim):
    from pxr import UsdGeom
    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0), dtype=float)


class FrozenPoseRestorer:
    def __init__(self, stage, config):
        self.drone = stage.GetPrimAtPath(config.drone.prim_path)
        self.body = stage.GetPrimAtPath(config.drone.body_path)
        original_body = world_matrix(self.body)
        self.root_to_body = world_matrix(self.drone) @ np.linalg.inv(original_body)
        self.body_scale = np.linalg.norm(original_body[:3, :3], axis=1)
        camera = stage.GetPrimAtPath(config.drone.camera_path)
        self.camera_in_body = None
        if camera and camera.IsValid():
            self.camera_in_body = world_matrix(camera) @ np.linalg.inv(original_body)
        self.last_pose_source = None

    def restore(self, record):
        from pxr import Gf, UsdGeom
        from scripts.drone_control_math import orientation_wxyz_from_rpy
        from scripts.trajectory_recorder import orientation_wxyz_from_world_transform

        desired, self.last_pose_source = resolve_recorded_body_world(record, self.camera_in_body)
        if desired is not None:
            quat = orientation_wxyz_from_world_transform(desired)
        else:
            quat = record.get('orientation') or orientation_wxyz_from_rpy(
                float(record.get('roll', 0.)), float(record.get('pitch', 0.)),
                float(record.get('yaw', 0.)))
            desired_gf = Gf.Matrix4d(1)
            desired_gf.SetRotate(Gf.Quatd(float(quat[0]), Gf.Vec3d(*map(float, quat[1:]))))
            desired = np.asarray(desired_gf, dtype=float)
            desired[:3, :3] *= self.body_scale[:, None]
            desired[3, :3] = record['position']
        root_world = self.root_to_body @ desired
        local = root_world @ np.linalg.inv(world_matrix(self.drone.GetParent()))
        UsdGeom.Xformable(self.drone).MakeMatrixXform().Set(Gf.Matrix4d(local.tolist()))
        return desired, list(quat)
