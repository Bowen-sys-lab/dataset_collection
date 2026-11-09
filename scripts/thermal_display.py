"""Project-owned display profiles for the software thermal camera.

Display profiles only change the 8-bit presentation image. They never modify
``radiance``, ``apparent_t``, ``dn16`` or sensor history.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class DisplayProfileError(ValueError):
    """Invalid project-owned display configuration."""


class _UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise DisplayProfileError('display profile keys must be text')
            if key in mapping:
                raise DisplayProfileError(f'duplicate display profile key {key!r}')
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class DisplayProfile:
    name: str
    mode: str
    dde_gain: float | None = None
    agc: str | None = None
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    polarity: str | None = None
    palette: str | None = None
    source_path: str | None = None
    source_sha256: str | None = None

    def metadata(self) -> dict[str, Any]:
        values = {
            'name': self.name,
            'mode': self.mode,
            'dde_gain': self.dde_gain,
            'agc': self.agc,
            'temperature_min_c': self.temperature_min_c,
            'temperature_max_c': self.temperature_max_c,
            'polarity': self.polarity,
            'palette': self.palette,
            'source_path': self.source_path,
            'source_sha256': self.source_sha256,
        }
        values['profile_sha256'] = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        return values


def _finite(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise DisplayProfileError(f'{field} must be a finite number')
    return float(value)


def _validate(data: dict[str, Any], *, path: Path | None = None) -> DisplayProfile:
    allowed = {
        'schema_version', 'name', 'mode', 'dde_gain', 'agc',
        'temperature_min_c', 'temperature_max_c', 'polarity', 'palette',
    }
    unknown = set(data) - allowed
    if unknown:
        raise DisplayProfileError(f'display profile has unknown fields: {sorted(unknown)}')
    if type(data.get('schema_version')) is not int or data['schema_version'] != 1:
        raise DisplayProfileError('display profile schema_version must be 1')
    name = data.get('name', 'display_profile')
    if not isinstance(name, str) or not name.strip():
        raise DisplayProfileError('display profile name must be non-empty text')
    mode = data.get('mode')
    if mode not in ('vendor', 'linear', 'fixed_apparent_temperature'):
        raise DisplayProfileError(
            "display profile mode must be vendor, linear or fixed_apparent_temperature"
        )
    dde = data.get('dde_gain')
    if dde is not None:
        dde = _finite(dde, 'dde_gain')
        if dde < 0:
            raise DisplayProfileError('dde_gain must be non-negative')
    agc = data.get('agc')
    if agc is not None and agc not in ('linear', 'plateau_equalization', 'none'):
        raise DisplayProfileError('agc must be linear, plateau_equalization or none')
    low = data.get('temperature_min_c')
    high = data.get('temperature_max_c')
    if mode == 'fixed_apparent_temperature':
        if low is None or high is None:
            raise DisplayProfileError('fixed temperature mode requires min and max Celsius')
        low, high = _finite(low, 'temperature_min_c'), _finite(high, 'temperature_max_c')
        if not low < high:
            raise DisplayProfileError('temperature_min_c must be lower than temperature_max_c')
    elif low is not None or high is not None:
        raise DisplayProfileError('temperature range is only valid for fixed temperature mode')
    polarity = data.get('polarity', 'white_hot' if mode == 'fixed_apparent_temperature' else None)
    palette = data.get('palette', 'gray' if mode == 'fixed_apparent_temperature' else None)
    if polarity is not None and polarity not in ('white_hot', 'black_hot'):
        raise DisplayProfileError('polarity must be white_hot or black_hot')
    if palette is not None and palette not in ('gray', 'ironbow', 'rainbow', 'lava', 'arctic'):
        raise DisplayProfileError('unsupported display palette')
    if mode == 'fixed_apparent_temperature' and (dde not in (None, 0.0) or agc is not None):
        raise DisplayProfileError('fixed temperature display uses no DDE or AGC; set dde_gain: 0 and omit agc')
    source_path = None if path is None else str(path.resolve())
    source_hash = None if path is None else hashlib.sha256(path.read_bytes()).hexdigest()
    return DisplayProfile(name, mode, dde, agc, low, high, polarity, palette,
                          source_path, source_hash)


def load_display_profile(path: str | Path) -> DisplayProfile:
    path = Path(path).resolve()
    try:
        data = yaml.load(path.read_text(encoding='utf-8-sig'), Loader=_UniqueLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DisplayProfileError(f'{path}: {exc}') from exc
    if not isinstance(data, dict):
        raise DisplayProfileError(f'{path}: expected a mapping')
    return _validate(data, path=path)


def profile_variants(min_c: float, max_c: float) -> tuple[DisplayProfile, ...]:
    """Return the one-pass comparison set used by the diagnostic tool."""
    min_c, max_c = _finite(min_c, 'temperature_min_c'), _finite(max_c, 'temperature_max_c')
    if not min_c < max_c:
        raise DisplayProfileError('temperature_min_c must be lower than temperature_max_c')
    return (
        DisplayProfile('current', 'vendor'),
        DisplayProfile('no_detail', 'vendor', dde_gain=0.0),
        DisplayProfile('linear', 'linear', dde_gain=0.0, agc='linear'),
        DisplayProfile('fixed_range', 'fixed_apparent_temperature', dde_gain=0.0,
                       temperature_min_c=min_c, temperature_max_c=max_c,
                       polarity='white_hot', palette='gray'),
    )


def fixed_temperature_display(apparent_t: Any, min_c: float, max_c: float, *,
                              polarity: str = 'white_hot', palette: str = 'gray') -> np.ndarray:
    """Map apparent temperature in Kelvin to an RGBA8 fixed-range display."""
    min_c, max_c = _finite(min_c, 'temperature_min_c'), _finite(max_c, 'temperature_max_c')
    if not min_c < max_c:
        raise DisplayProfileError('temperature_min_c must be lower than temperature_max_c')
    from irsim.isp.palette import to_display8
    temperature_c = np.asarray(apparent_t, dtype=np.float32) - np.float32(273.15)
    if temperature_c.ndim != 2 or not temperature_c.size:
        raise DisplayProfileError('apparent_t must be a non-empty (H, W) image in Kelvin')
    normalized = (temperature_c - np.float32(min_c)) / np.float32(max_c - min_c)
    invalid = ~np.isfinite(normalized)
    normalized = np.where(invalid, 0.0, normalized).astype(np.float32, copy=False)
    display = to_display8(normalized, polarity, palette)
    display[invalid] = (255, 0, 255, 255)
    return display


def profile_isp(sensor_isp: Any, profile: DisplayProfile) -> Any:
    """Create an ISP copy for vendor/linear profiles without changing sensor config."""
    updates: dict[str, Any] = {}
    if profile.dde_gain is not None:
        updates['dde_gain'] = profile.dde_gain
    if profile.mode == 'linear':
        updates['agc'] = 'linear'
    elif profile.agc is not None:
        updates['agc'] = profile.agc
    if profile.polarity is not None:
        updates['polarity'] = profile.polarity
    if profile.palette is not None:
        updates['palette'] = profile.palette
    return sensor_isp if not updates else sensor_isp.model_copy(update=updates)


def display_range_metadata(profile: DisplayProfile | None, apparent_t: Any) -> dict[str, Any] | None:
    """Report finite/clipped pixels without altering the underlying temperatures."""
    if profile is None or profile.mode != 'fixed_apparent_temperature':
        return None
    image = np.asarray(apparent_t)
    finite = np.isfinite(image)
    values = image[finite].astype(np.float64) - 273.15
    return {
        'finite_pixels': int(values.size),
        'nonfinite_pixels': int(image.size - values.size),
        'below_range_pixels': int(np.count_nonzero(values < profile.temperature_min_c)),
        'above_range_pixels': int(np.count_nonzero(values > profile.temperature_max_c)),
        'clipped_fraction_of_finite': (float(np.mean(
            (values < profile.temperature_min_c) | (values > profile.temperature_max_c)
        )) if values.size else None),
        'display_temperature_min_c': profile.temperature_min_c,
        'display_temperature_max_c': profile.temperature_max_c,
        'clipping_scope': '8-bit display only; apparent_t remains unbounded by this display range',
    }
