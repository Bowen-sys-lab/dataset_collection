"""Image-state checks independent of Kit, clocks and self-assigned frame ids."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label


def diagnostic_target_path(task):
    return (task or {}).get('target_prim_path') or '/World/Target_01'


def largest_component(mask):
    labels, count = label(np.asarray(mask, dtype=bool))
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == np.argmax(sizes)


def centroid(mask):
    y, x = np.nonzero(mask)
    return None if len(x) < 4 else [float(x.mean()), float(y.mean())]


def compare_recorded_pose(actual, expected, meters_per_unit=1.0):
    """Compare a restored transform using physical pose tolerances, not scaled matrix elements."""
    from scripts.replay_pose import compare_pose_matrices, pose_within_tolerance

    errors = compare_pose_matrices(actual, expected, meters_per_unit=meters_per_unit)
    return {**errors, 'within_tolerance': pose_within_tolerance(errors)}


def rgb_marker_mask(rgb):
    # Detect the saturated green marker core, excluding its dim indirect
    # illumination/bloom on surrounding surfaces. uint8 threshold is explicit.
    c = np.asarray(rgb)[..., :3].astype(float)
    return largest_component((c[..., 1] >= 128) & (c[..., 1] > c[..., 0]*1.5+10) &
                             (c[..., 1] > c[..., 2]*1.5+10))


def compare_masks(rgb_mask, thermal_mask, expected_uv, *, pixel_tolerance=3.0):
    a, b = np.asarray(rgb_mask, bool), np.asarray(thermal_mask, bool)
    ca, cb = centroid(a), centroid(b)
    if a.shape != b.shape or ca is None or cb is None:
        return {'pass': False, 'reason': 'marker missing or incompatible masks',
                'rgb_centroid': ca, 'thermal_centroid': cb}
    ea, eb = a & ~binary_erosion(a), b & ~binary_erosion(b)
    distances = np.concatenate((distance_transform_edt(~eb)[ea],
                                distance_transform_edt(~ea)[eb]))
    rgb_error = float(np.linalg.norm(np.asarray(ca) - expected_uv))
    thermal_error = float(np.linalg.norm(np.asarray(cb) - expected_uv))
    edge_p95 = float(np.percentile(distances, 95))
    iou = float((a & b).sum() / (a | b).sum())
    return {'pass': bool(max(rgb_error, thermal_error, edge_p95) <= pixel_tolerance and iou >= .65),
            'rgb_centroid': ca, 'thermal_centroid': cb,
            'rgb_current_position_error_px': rgb_error,
            'thermal_current_position_error_px': thermal_error,
            'edge_p95_error_px': edge_p95, 'mask_iou': iou,
            'pixel_tolerance': pixel_tolerance}


def sample_on_grid(image, width, height):
    """Nearest pixel-centre sampling for diagnostics only; never resave source images."""
    image = np.asarray(image)
    if image.ndim not in (2, 3) or min(*image.shape[:2], width, height) <= 0:
        raise ValueError('image and comparison dimensions must be non-empty 2D/3D arrays')
    y = np.minimum(((np.arange(height) + .5) * image.shape[0] / height).astype(int), image.shape[0] - 1)
    x = np.minimum(((np.arange(width) + .5) * image.shape[1] / width).astype(int), image.shape[1] - 1)
    return image[np.ix_(y, x)] if image.ndim == 2 else image[y[:, None], x[None, :]]


def mask_on_grid(mask, width, height):
    return sample_on_grid(np.asarray(mask, dtype=bool), width, height)


def registered_overlay(rgb_color, rgb_mask, thermal_mask):
    """Build a diagnostic overlay on the thermal grid without altering source images."""
    rgb_color = np.asarray(rgb_color)
    rgb_mask = np.asarray(rgb_mask, dtype=bool)
    thermal_mask = np.asarray(thermal_mask, dtype=bool)
    if rgb_color.ndim != 3 or rgb_color.shape[2] < 3:
        raise ValueError('RGB color must be an HxWx3 image')
    if rgb_mask.shape != rgb_color.shape[:2] or thermal_mask.ndim != 2:
        raise ValueError('marker masks must match their image dimensions')
    height, width = thermal_mask.shape
    common_color = sample_on_grid(rgb_color[..., :3], width, height)
    common_rgb_mask = mask_on_grid(rgb_mask, width, height)
    overlay = (common_color.astype(float) * .45).astype(np.uint8)
    overlay[common_rgb_mask & ~thermal_mask] = (0, 255, 0)
    overlay[thermal_mask & ~common_rgb_mask] = (255, 0, 0)
    overlay[thermal_mask & common_rgb_mask] = (255, 255, 0)
    return overlay


def compare_paired_masks(rgb_mask, thermal_mask, expected_rgb_uv, expected_thermal_uv, *, pixel_tolerance=3.0):
    """Compare registered masks on the smaller grid and check both native centroids."""
    a, b = np.asarray(rgb_mask, bool), np.asarray(thermal_mask, bool)
    ca, cb = centroid(a), centroid(b)
    ha, wa = a.shape
    hb, wb = b.shape
    if abs(wa / ha - wb / hb) / max(wa / ha, wb / hb) > .005:
        return {'pass': False, 'reason': 'camera aspect ratios differ'}
    width, height = min(wa, wb), min(ha, hb)
    scale_a, scale_b = np.array([width / wa, height / ha]), np.array([width / wb, height / hb])
    # Same camera projection expressed in two native pixel-centre grids.
    expected = (np.asarray(expected_thermal_uv) + .5) * scale_b - .5
    result = compare_masks(mask_on_grid(a, width, height), mask_on_grid(b, width, height),
                           expected, pixel_tolerance=pixel_tolerance)
    error_a = None if ca is None else float(np.linalg.norm((np.asarray(ca) - expected_rgb_uv) * scale_a))
    error_b = None if cb is None else float(np.linalg.norm((np.asarray(cb) - expected_thermal_uv) * scale_b))
    result.update(rgb_native_centroid=ca, thermal_native_centroid=cb,
                  rgb_native_expected_uv=list(expected_rgb_uv), thermal_native_expected_uv=list(expected_thermal_uv),
                  native_rgb_error_in_comparison_px=error_a, native_thermal_error_in_comparison_px=error_b,
                  comparison_resolution=[width, height],
                  comparison_scope='diagnostic masks sampled on smaller grid; original outputs unchanged')
    result['pass'] = bool(result['pass'] and error_a is not None and error_b is not None and
                          max(error_a, error_b) <= pixel_tolerance)
    return result


def unique_marker_positions(count, width, height, *, spacing_px=8.0):
    """Non-repeating, well-separated positions prevent a multi-frame lag from aliasing."""
    if count <= 0 or min(width, height) < 32 or spacing_px < 8:
        raise ValueError('positive frame count, dimensions >=32 and spacing >=8 required')
    x = np.arange(width * .16, width * .84 + 1e-9, spacing_px)
    y = np.arange(height * .28, height * .76 + 1e-9, spacing_px)
    positions = np.array([(u, v) for v in y for u in x])
    if count > len(positions):
        raise ValueError(f'marker position capacity is {len(positions)} at {width}x{height}; requested {count}')
    return positions[np.random.default_rng(0).permutation(len(positions))[:count]]


def match_marker_position(observed_uv, expected_positions, *, pixel_tolerance=3.0):
    if observed_uv is None:
        return {'matched_frame_index': None, 'position_error_px': None, 'candidate_count': 0}
    distances = np.linalg.norm(np.asarray(expected_positions) - observed_uv, axis=1)
    candidates = np.flatnonzero(distances <= pixel_tolerance)
    matched = int(candidates[0]) if len(candidates) == 1 else None
    return {'matched_frame_index': matched, 'position_error_px': float(distances.min()),
            'candidate_count': int(len(candidates))}


def expected_uv(point_world, camera_world, width, height, focal, aperture):
    # USD matrices use row vectors; camera looks along -Z with +Y up.
    point = np.append(point_world, 1.0) @ np.linalg.inv(camera_world)
    if point[2] >= 0:
        raise ValueError('marker is behind the camera')
    fx = width * focal / aperture
    return [float(width / 2 + fx * point[0] / -point[2]),
            float(height / 2 - fx * point[1] / -point[2])]
