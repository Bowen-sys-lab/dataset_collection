"""Single-flight, wall-clock preview scheduling, independent of physics time."""
from __future__ import annotations

import math


def validate_split_options(mode, save_preview):
    if mode=='split' and save_preview:
        raise ValueError('split preview cannot save paired frames; use paired diagnostics or offline replay')


class SplitPreviewScheduler:
    """Independent wall-clock deadlines; no backlog after stalls or pauses."""
    def __init__(self, rgb_hz=15., thermal_hz=2.):
        if any(not math.isfinite(v) or v <= 0 for v in (rgb_hz, thermal_hz)):
            raise ValueError('split preview rates must be positive and finite')
        self.periods = {'rgb': 1/rgb_hz, 'thermal': 1/thermal_hz}
        self.deadlines = dict.fromkeys(self.periods)

    def poll(self, now):
        result = {}
        for name, period in self.periods.items():
            due = self.deadlines[name] is None or now + 1e-9 >= self.deadlines[name]
            result[name] = due
            if due:
                self.deadlines[name] = now + period
        return result


class PreviewScheduler:
    def __init__(self, hz=12.0, max_age_s=0.25):
        if not math.isfinite(hz) or hz < 0 or not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('preview rate must be finite and non-negative; age limit must be positive')
        self.uncapped = hz == 0
        self.period = None if self.uncapped else 1.0 / hz
        self.max_age_s = max_age_s
        self.next_due = None
        self.in_flight = False
        self.dropped = 0
        self.published = 0
        self._last_source = None

    def due(self, now):
        return not self.in_flight and (self.next_due is None or now >= self.next_due)

    def begin(self, now, source_id=None):
        if not self.due(now):
            return False
        if self.next_due is not None and not self.uncapped:
            self.dropped += max(0, int((now - self.next_due) / self.period))
        # No backlog and no catch-up burst after a slow frame/pause.
        self.next_due = None if self.uncapped else now + self.period
        if source_id is not None and source_id == self._last_source:
            self.dropped += 1
            return False
        self.in_flight = True
        return True

    def finish(self, started, finished, source_id=None, valid=True):
        if not self.in_flight:
            raise RuntimeError('no preview capture in flight')
        self.in_flight = False
        if not valid or finished - started > self.max_age_s:
            self.dropped += 1
            return None
        self._last_source = source_id
        frame_id = self.published
        self.published += 1
        return frame_id

    def reset(self):
        self.next_due = None
        self._last_source = None


def percentile_summary(values):
    import numpy as np
    if not values:
        return {'samples': 0, 'mean': None, 'p95': None, 'p99': None}
    a = np.asarray(values, dtype=float)
    return {'samples': len(values), 'mean': float(a.mean()),
            'p95': float(np.percentile(a, 95)), 'p99': float(np.percentile(a, 99))}


def verify_renderer_evidence(path, config, renderer, width, height):
    """Non-default rendering is opt-in only after a matching marker diagnostic."""
    if renderer == 'PathTracing':
        return
    import json
    if path is None:
        raise ValueError('Realtime RTX requires --renderer-proof from a passing matching marker test')
    proof = json.loads(path.read_text(encoding='utf-8'))
    conditions = proof.get('experiment_conditions', {})
    if (proof.get('result') != 'PASS' or proof.get('strict_per_frame_pixel_sync_verified') is not True
            or proof.get('renderer') != renderer or proof.get('resolution') != [width, height]
            or proof.get('scene_config_sha256') != config.sha256):
        raise ValueError('Renderer proof is failed, incomplete or for different conditions')
    import hashlib
    for key, file in (('usd', config.scene.stage), ('thermal_config', config.sensors.thermal_config),
                      ('sensor_config', config.sensors.sensor_config)):
        if conditions.get('files', {}).get(key, {}).get('sha256') != hashlib.sha256(file.read_bytes()).hexdigest():
            raise ValueError('Renderer proof does not match current ' + key)
