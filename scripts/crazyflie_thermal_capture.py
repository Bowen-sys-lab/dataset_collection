"""Small, engine-free checks shared by the Crazyflie thermal capture script."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _require_plane(name: str, value: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        raise ValueError(f"{name} output is missing")
    array = np.asarray(value)
    if array.shape[:2] != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match {expected_shape}")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    measured = array[..., :3] if array.ndim == 3 else array
    if measured.size == 0 or np.ptp(measured) == 0:
        raise ValueError(f"{name} is constant or empty")
    return array


def validate_frame_contract(
    *,
    rgb: Any,
    radiance: Any,
    apparent_t: Any,
    dn16: Any,
    display8: Any,
    expected_shape: tuple[int, int],
) -> dict[str, Any]:
    """Validate one registered RGB/thermal frame before it is written to disk."""
    thermal = {
        name: _require_plane(name, value, expected_shape)
        for name, value in {
            "radiance": radiance,
            "apparent_t": apparent_t,
            "dn16": dn16,
            "display8": display8,
        }.items()
    }
    rgb_array = _require_plane("rgb", rgb, expected_shape)
    return {
        "shape": list(expected_shape),
        "rgb_shape": list(rgb_array.shape),
        "planes": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in thermal.items()},
    }


def camera_metadata(
    *,
    camera_path: str,
    stage_path: str,
    sensor_config: str,
    camera_position: Any,
    capture_time_s: float | None,
) -> Mapping[str, Any]:
    """Return explicit provenance fields for a static capture sidecar."""
    return {
        "camera_path": camera_path,
        "stage_path": stage_path,
        "sensor_config": sensor_config,
        "camera_position_world": [float(v) for v in np.asarray(camera_position).reshape(-1)[:3]],
        "capture_time_s": None if capture_time_s is None else float(capture_time_s),
        "capture_time_domain": "third_party_thermal_weather_scene_seconds",
        "integration_mode": "shared_camera_rgb_and_thermal",
        "note": (
            "IrCamera authors sensor optics on the existing USD camera in this process; the USD "
            "stage is not saved. This clock is not asserted to be Isaac physics/timeline time."
        ),
    }
