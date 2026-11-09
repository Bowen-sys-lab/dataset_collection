"""Independent moving-marker evidence. Hash changes alone never prove freshness."""
import hashlib
import numpy as np

from scripts.sync_validation_support import centroid, compare_masks, largest_component, rgb_marker_mask


def image_hash(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


class PreviewContentTracker:
    """Pixel hashes are diagnostics only; identical pixels may be a new static frame."""
    def __init__(self):
        self.previous = None
        self.rejected = 0
        self.repeated_pairs = 0

    def inspect(self, rgb, thermal_display):
        hashes = (image_hash(rgb), image_hash(thermal_display))
        repeated = tuple(self.previous is not None and old == new
                         for old, new in zip(self.previous or (None, None), hashes))
        self.previous = hashes
        if any(repeated):
            self.repeated_pairs += 1
        return {'rgb_sha256': hashes[0], 'thermal_sha256': hashes[1],
                'rgb_hash_repeated': repeated[0], 'thermal_hash_repeated': repeated[1],
                'pair_content_changed': not any(repeated),
                'reject_on_hash': False,
                'freshness_status': 'unknown_from_hash',
                'fresh_current_state_verified': False,
                'content_check_scope': 'Hash diagnostics only; equal static pixels and changing old-frame noise cannot prove age',
                'rgb_internal_frame_id': None, 'thermal_internal_frame_id': None,
                'rgb_internal_timestamp': None, 'thermal_geometry_internal_timestamp': None}

    def report(self):
        return {'hash_repeated_pair_diagnostics': self.repeated_pairs,
                'old_aov_frames_verified': False}


def match_pair_history(rgb_uv, thermal_uv, geometry_uv, history, read_wall_s, tolerance_px=2.):
    """Match a unique marker state, without inventing a render completion timestamp."""
    matches = {}
    for name, uv in (('rgb', rgb_uv), ('thermal', thermal_uv), ('geometry', geometry_uv)):
        candidates = [h for h in history if uv is not None and
                      np.linalg.norm(np.asarray(uv)-np.asarray(h['expected_uv'])) <= tolerance_px]
        matches[name] = candidates[0] if len(candidates) == 1 else None
    steps = [None if matches[k] is None else matches[k]['physics_step']
             for k in ('rgb', 'thermal', 'geometry')]
    shared = bool(history) and None not in steps and len(set(steps)) == 1
    matched = matches['rgb'] if shared else None
    return {'same_historical_marker_state_verified': shared,
            'current_marker_state_verified': shared and steps[0] == history[-1]['physics_step'],
            'matched_marker_physics_step': None if matched is None else matched['physics_step'],
            'matched_marker_simulation_time_s': None if matched is None else matched['simulation_time_s'],
            'marker_edit_to_read_ms': None if matched is None else (read_wall_s-matched['wall_s'])*1000.,
            'marker_latency_scope': 'Known marker edit to CPU image read; not renderer completion to monitor scanout',
            'rgb_matched_step': steps[0], 'thermal_matched_step': steps[1], 'geometry_matched_step': steps[2]}


class LiveFrameValidator:
    def __init__(self):
        self.previous = None
        self.accepted = 0
        self.stale = 0

    def inspect(self, rgb, thermal, geometry_mask, expected, *, marker_state_changed=True):
        rgb_mask = rgb_marker_mask(rgb)
        rgb_hash = image_hash(rgb)
        rgb_centroid = centroid(rgb_mask)
        sample = {'rgb_sha256': rgb_hash, 'rgb_marker_uv': rgb_centroid,
                  'expected_marker_uv': list(expected), 'rgb_internal_frame_id': None,
                  'thermal_internal_frame_id': None, 'rgb_internal_timestamp': None,
                  'thermal_geometry_internal_timestamp': None,
                  'internal_timestamp_scope': 'No per-AOV timestamp exposed; thermal solver time is not a renderer timestamp'}
        duplicate_rgb = self.previous is not None and self.previous['rgb_sha256'] == rgb_hash
        sample['rgb_hash_repeated'] = duplicate_rgb
        sample['rgb_mask_repeated'] = bool(self.previous is not None and
            np.array_equal(self.previous['rgb_mask'], rgb_mask))
        if thermal is not None:
            ir_hash = image_hash(thermal.display8)
            ir_mask = largest_component(np.isfinite(thermal.apparent_t) & (thermal.apparent_t > 330.))
            geometry = compare_masks(rgb_mask, geometry_mask, expected)
            physical_output = compare_masks(rgb_mask, ir_mask, expected)
            sample.update(thermal_sha256=ir_hash, thermal_marker_uv=centroid(ir_mask),
                geometry_marker_uv=centroid(geometry_mask), geometry_pixel_check=geometry,
                thermal_pixel_check=physical_output,
                thermal_hash_repeated=bool(self.previous is not None and self.previous.get('ir_hash') == ir_hash),
                thermal_mask_repeated=bool(self.previous is not None and
                    np.array_equal(self.previous.get('ir_mask'), ir_mask)),
                dimensions_match=rgb.shape[:2] == thermal.display8.shape[:2],
                temperatures_finite=bool(np.isfinite(thermal.apparent_t).all()))
            passed = bool(geometry['pass'] and physical_output['pass'] and
                sample['dimensions_match'] and sample['temperatures_finite'])
        else:
            ir_mask = ir_hash = None
            error = None if rgb_centroid is None else float(np.linalg.norm(np.asarray(rgb_centroid)-expected))
            sample.update(thermal_sha256=None, thermal_marker_uv=None,
                rgb_current_position_error_px=error, thermal_disabled=True)
            passed = error is not None and error <= 3.
        # A changing noisy image may still contain old geometry. Compare masks too.
        repeated = sample['rgb_mask_repeated'] or bool(sample.get('thermal_mask_repeated'))
        passed = bool(passed and not (marker_state_changed and (duplicate_rgb or repeated)))
        sample['fresh_current_state_verified'] = passed
        sample['classification'] = 'fresh_current_state' if passed else 'stale_or_unverified'
        if passed:
            sample['frame_id'] = self.accepted
            self.accepted += 1
        else:
            sample['frame_id'] = None
            self.stale += 1
        sample['pixel_direction_consistent'] = None
        if passed and self.previous is not None and self.previous['passed']:
            delta = np.asarray(expected) - self.previous['expected']
            measured = np.asarray(rgb_centroid) - self.previous['rgb_centroid']
            directions = [float(np.dot(delta, measured)) > 0] if np.linalg.norm(delta) > 3 else []
            if ir_mask is not None and self.previous.get('ir_centroid') is not None and directions:
                directions.append(float(np.dot(delta, np.asarray(sample['thermal_marker_uv'])-
                    self.previous['ir_centroid'])) > 0)
            sample['pixel_direction_consistent'] = bool(all(directions)) if directions else None
        self.previous = {'rgb_sha256': rgb_hash, 'rgb_mask': rgb_mask.copy(), 'ir_hash': ir_hash,
            'ir_mask': None if ir_mask is None else ir_mask.copy(), 'expected': np.asarray(expected),
            'rgb_centroid': rgb_centroid, 'ir_centroid': sample.get('thermal_marker_uv'), 'passed': passed}
        return sample


def latency_summary(values):
    from scripts.preview_scheduler import percentile_summary
    summary = percentile_summary(values)
    summary['max'] = float(max(values)) if values else None
    return summary


def driving_gates(*, viewport_enabled, latencies, loop_intervals, samples, has_thermal):
    # No samples, no measured human input, or unknown image state never become PASS.
    return {'external_viewport_enabled': bool(viewport_enabled),
        'all_observed_control_latencies_below_50ms': bool(latencies) and max(latencies) < .05,
        'all_measured_loop_intervals_below_100ms': bool(loop_intervals) and max(loop_intervals) < .1,
        'rgb_and_thermal_current_state': has_thermal and bool(samples) and
            all(s['fresh_current_state_verified'] for s in samples),
        'motion_direction_consistent': any(s['pixel_direction_consistent'] is True for s in samples) and
            all(s['pixel_direction_consistent'] is not False for s in samples),
        'human_end_to_end_latency_verified': False}
