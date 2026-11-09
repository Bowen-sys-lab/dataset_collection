"""Pure helpers for the isolated RGB color-chain and UV diagnostics."""
from __future__ import annotations

import math
import re

import numpy as np


_UV_WARNING = re.compile(
    r"Mesh '(?P<mesh>/[^']+)' has corrupted data in primvar '(?P<primvar>[^']+)': "
    r"buffer size (?P<actual>\d+) (?:does not|doesn't) match expected size (?P<expected>\d+) "
    r"in faceVarying primvars"
)


def parse_uv_warning(line: str) -> dict | None:
    """Parse one RTX/Hydra UV warning without guessing its source asset."""
    match = _UV_WARNING.search(str(line))
    if match is None:
        return None
    mesh_path = match.group('mesh')
    prototype_path = mesh_path.split('/', 2)
    if len(prototype_path) < 2:
        return None
    return {
        'mesh_prim_path': mesh_path,
        'prototype_path': '/' + prototype_path[1],
        'primvar_name': match.group('primvar'),
        'reported_buffer_size': int(match.group('actual')),
        'expected_face_varying_size': int(match.group('expected')),
    }


def parse_uv_warnings(lines) -> list[dict]:
    """Return parsed warnings in log order; repeated lines remain observable."""
    records = []
    for line in lines:
        record = parse_uv_warning(line)
        if record is not None:
            records.append(record)
    return records


def world_prim_range(stage, usd_module=None, root='/World'):
    """Traverse the authored world subtree, including USD instance proxies."""
    if usd_module is None:
        from pxr import Usd as usd_module
    start = stage.GetPrimAtPath(root)
    if not start or not start.IsValid():
        raise ValueError(f'USD traversal root is not valid: {root}')
    return usd_module.PrimRange(start, usd_module.TraverseInstanceProxies())


def color_patch_channels(rgb, mode, emission_scale=0.5):
    """Return USD Preview Surface diffuse/emissive inputs for a test swatch."""
    color = tuple(float(channel) for channel in rgb)
    if len(color) != 3 or any(not math.isfinite(channel) for channel in color):
        raise ValueError('RGB swatches must contain three finite channels')
    if mode == 'diffuse':
        return color, (0.0, 0.0, 0.0)
    if mode == 'emissive':
        scale = float(emission_scale)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError('emission_scale must be finite and non-negative')
        return (0.0, 0.0, 0.0), tuple(channel * scale for channel in color)
    raise ValueError(f'Unknown color patch mode: {mode}')


def project_patch_roi(*, local_xyz, patch_size, focal_length_mm,
                       horizontal_aperture_mm, vertical_aperture_mm,
                       width, height, inner_fraction=0.5) -> tuple[int, int, int, int]:
    """Project a camera-local square patch to a center ROI, excluding its edges.

    USD camera local forward is -Z, local +X projects right, and local +Y projects up.
    The returned bounds follow NumPy/PIL half-open ``(x0, y0, x1, y1)`` convention.
    """
    x, y, z = (float(value) for value in local_xyz)
    patch_width, patch_height = (float(value) for value in patch_size)
    focal = float(focal_length_mm)
    aperture_x, aperture_y = float(horizontal_aperture_mm), float(vertical_aperture_mm)
    if z >= 0:
        raise ValueError('color-chart patch must be in front of the camera (local z < 0)')
    if min(patch_width, patch_height, focal, aperture_x, aperture_y, width, height) <= 0:
        raise ValueError('patch, camera, and image dimensions must be positive')
    if not math.isfinite(inner_fraction) or not 0 < inner_fraction <= 1:
        raise ValueError('inner_fraction must be in (0, 1]')

    depth = -z
    center_x = width / 2 + x * focal / aperture_x / depth * width
    center_y = height / 2 - y * focal / aperture_y / depth * height
    half_x = patch_width / 2 * focal / aperture_x / depth * width * inner_fraction
    half_y = patch_height / 2 * focal / aperture_y / depth * height * inner_fraction
    x0 = max(0, int(math.floor(center_x - half_x)))
    y0 = max(0, int(math.floor(center_y - half_y)))
    x1 = min(int(width), int(math.ceil(center_x + half_x)))
    y1 = min(int(height), int(math.ceil(center_y + half_y)))
    if x0 >= x1 or y0 >= y1:
        raise ValueError('projected color-chart patch is outside the image')
    return x0, y0, x1, y1


def region_statistics(image, roi) -> dict:
    """Summarize a known RGB image region using channel and saturation metrics."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f'Expected an RGB/RGBA image, got {rgb.shape}')
    x0, y0, x1, y1 = (int(value) for value in roi)
    height, width = rgb.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f'ROI {roi} is outside image size {(width, height)}')
    values = rgb[y0:y1, x0:x1, :3].astype(np.float64)
    unit = values / 255.0 if np.issubdtype(rgb.dtype, np.integer) else values
    high, low = unit.max(axis=2), unit.min(axis=2)
    saturation = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)
    luma = values @ np.asarray([0.2126, 0.7152, 0.0722])
    return {
        'roi_xyxy': [x0, y0, x1, y1],
        'pixel_count': int(values.shape[0] * values.shape[1]),
        'mean_rgb': [float(value) for value in values.mean(axis=(0, 1))],
        'median_rgb': [float(value) for value in np.median(values, axis=(0, 1))],
        'mean_hsv_saturation': float(saturation.mean()),
        'mean_luma': float(luma.mean()),
    }


def compare_rgb_buffers(rendered, decoded_png) -> dict:
    """Compare the annotator buffer with the decoded saved PNG pixel-for-pixel."""
    left, right = np.asarray(rendered), np.asarray(decoded_png)
    same_shape = left.shape == right.shape
    same_dtype = left.dtype == right.dtype
    if not same_shape:
        return {
            'same_shape': False,
            'same_dtype': same_dtype,
            'exact_equal': False,
            'rendered_shape': list(left.shape),
            'decoded_png_shape': list(right.shape),
            'max_absolute_difference': None,
        }
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        'same_shape': True,
        'same_dtype': same_dtype,
        'exact_equal': bool(same_dtype and np.array_equal(left, right)),
        'rendered_shape': list(left.shape),
        'decoded_png_shape': list(right.shape),
        'max_absolute_difference': float(delta.max(initial=0)),
    }


def local_transform_from_world(wanted_world, parent_world):
    """Convert USD row-vector world transforms to a parent's local transform."""
    wanted = np.asarray(wanted_world, dtype=np.float64)
    parent = np.asarray(parent_world, dtype=np.float64)
    if wanted.shape != (4, 4) or parent.shape != (4, 4):
        raise ValueError('world transforms must both be 4x4 matrices')
    return wanted @ np.linalg.inv(parent)


def project_world_bbox_to_image(*, bbox_min, bbox_max, camera_world, focal_length_mm,
                                horizontal_aperture_mm, vertical_aperture_mm,
                                width, height) -> dict:
    """Project an axis-aligned world-space bounds box to an RGB camera image."""
    minimum = np.asarray(bbox_min, dtype=np.float64)
    maximum = np.asarray(bbox_max, dtype=np.float64)
    camera = np.asarray(camera_world, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,) or camera.shape != (4, 4):
        raise ValueError('expected 3D bounds and a 4x4 camera world matrix')
    if np.any(minimum > maximum):
        raise ValueError('bbox_min cannot exceed bbox_max')
    if min(float(focal_length_mm), float(horizontal_aperture_mm),
           float(vertical_aperture_mm), int(width), int(height)) <= 0:
        raise ValueError('camera and image dimensions must be positive')

    corners = np.asarray([
        (x, y, z, 1.0)
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ], dtype=np.float64)
    camera_points = corners @ np.linalg.inv(camera)
    depth = -camera_points[:, 2]
    if np.all(depth <= 1e-9):
        return {'in_camera_frustum': False, 'projected_bounds_xyxy': None,
                'camera_space_depth_min_m': float(depth.min()),
                'camera_space_depth_max_m': float(depth.max())}
    positive = depth > 1e-9
    camera_points = camera_points[positive]
    depth = depth[positive]
    focal_x = float(focal_length_mm) / float(horizontal_aperture_mm) * int(width)
    focal_y = float(focal_length_mm) / float(vertical_aperture_mm) * int(height)
    projected_x = int(width) / 2 + camera_points[:, 0] * focal_x / depth
    projected_y = int(height) / 2 - camera_points[:, 1] * focal_y / depth
    x0 = int(np.floor(projected_x.min()))
    y0 = int(np.floor(projected_y.min()))
    x1 = int(np.ceil(projected_x.max()))
    y1 = int(np.ceil(projected_y.max()))
    visible = x1 > 0 and y1 > 0 and x0 < int(width) and y0 < int(height)
    if not visible:
        bounds = None
    else:
        bounds = [max(0, x0), max(0, y0), min(int(width), x1), min(int(height), y1)]
    return {
        'in_camera_frustum': bool(visible),
        'projected_bounds_xyxy': bounds,
        'camera_space_depth_min_m': float(depth.min()),
        'camera_space_depth_max_m': float(depth.max()),
    }


def uv_signature_matches(*, expected_corners, values_count, indices_count,
                         warning_expected, warning_buffer) -> bool:
    """Check whether a USD Vec2 UV primvar shape can explain one RTX buffer warning.

    RTX reports UV element counts, as verified against the composed Spot mesh
    primvars in Kit. This is only a signature match, not proof that a
    render delegate prototype and a USD scene prim are the same object.
    """
    if int(expected_corners) != int(warning_expected):
        return False
    observed_sizes = {int(values_count)}
    if indices_count is not None:
        observed_sizes.add(int(indices_count))
    return int(warning_buffer) in observed_sizes


def uv_warning_sources_inspected(mesh_records):
    """Account for each warning group without equating signatures with identity."""
    if not mesh_records:
        return None
    return all(
        any(candidate.get('mesh_name_matches_warning') and
            candidate.get('all_warning_uv_primvars_matched')
            for candidate in record.get('signature_candidates', [])) or
        bool(record.get('mesh_geometry') and
             record['mesh_geometry'].get('warning_primvars') and
             all(detail.get('exists') for detail in
                 record['mesh_geometry']['warning_primvars'].values()))
        for record in mesh_records
    )
