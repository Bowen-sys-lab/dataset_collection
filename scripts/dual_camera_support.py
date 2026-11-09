"""Small, Isaac-independent helpers shared by the dual-camera preview."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from typing import Callable

import numpy as np


class ExplicitRenderIncomplete(RuntimeError):
    """A render did not finish; the live loop may discard an interrupted pair."""


def capture_interruption_reason(*, app_running, window_visible, timeline_playing):
    """Only user lifecycle transitions may interrupt live capture without error."""
    if not app_running or not window_visible:
        return 'window_closed'
    if not timeline_playing:
        return 'timeline_paused_or_stopped'
    return None


def preview_sensor_data(source: dict, width: int, height: int,
                        focal_length: float, horizontal_aperture: float) -> dict:
    """Make an independent preview configuration, preserving source HFOV.

    Square pixels and the requested aspect ratio determine the vertical FOV.
    This is an illustrative sensor variant, not a calibrated Boson model.
    """
    if (not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0
            or not all(math.isfinite(v) and v > 0 for v in (focal_length, horizontal_aperture))):
        raise ValueError('resolution and optics must be finite and positive')
    result = deepcopy(source)
    spec = result['sensor']
    spec['fpa'].update(width=width, height=height)
    spec['optics']['supersample_factor'] = 1
    pitch_mm = spec['fpa']['pitch_um'] * .001
    spec['optics']['focal_length_mm'] = width * pitch_mm * focal_length / horizontal_aperture
    return result


def scale_camera_optics_for_resolution(optics, width: int, height: int):
    """Scale pixel intrinsics for a different output grid without changing lens/FOV."""
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('output width and height must be positive integers')
    intrinsics = optics.intrinsics
    scale_x = width / intrinsics.width
    scale_y = height / intrinsics.height
    if abs(scale_x - scale_y) / max(scale_x, scale_y) > 0.005:
        raise ValueError('output aspect ratio must match the source camera aspect ratio')
    scaled_intrinsics = replace(
        intrinsics,
        fx_px=intrinsics.fx_px * scale_x,
        fy_px=intrinsics.fy_px * scale_y,
        cx_px=intrinsics.cx_px * scale_x,
        cy_px=intrinsics.cy_px * scale_y,
        width=width,
        height=height,
    )
    return replace(optics, intrinsics=scaled_intrinsics)


def rgba_bytes(image) -> tuple[bytes, list[int]]:
    """Convert a uint8 grayscale/RGB/RGBA image to tightly packed RGBA bytes.

    The image is copied through a contiguous array so that flipped NumPy views
    retain their visual pixel order. Raw floating-point thermal values are
    deliberately rejected: they must first be mapped to a display image.
    """
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image must not be empty")
    if array.dtype != np.uint8:
        raise ValueError("rgba_bytes expects uint8 display data")
    if array.ndim == 2:
        rgb = np.repeat(array[..., None], 3, axis=2)
        rgba = np.concatenate((rgb, np.full((*array.shape, 1), 255, dtype=np.uint8)), axis=2)
    elif array.ndim == 3 and array.shape[2] == 3:
        rgba = np.concatenate((array, np.full((*array.shape[:2], 1), 255, dtype=np.uint8)), axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        rgba = array
    else:
        raise ValueError("image must be HxW grayscale, HxWx3 RGB, or HxWx4 RGBA")
    height, width = rgba.shape[:2]
    return np.ascontiguousarray(rgba).tobytes(), [int(width), int(height)]


def ui_rgba_payload(image) -> tuple[list[int], list[int]]:
    """Kit 6.1 ByteImageProvider accepts a sequence of integers, not bytes."""
    data, size = rgba_bytes(image)
    return list(data), size


def ui_rgba_array(image) -> tuple[np.ndarray, list[int]]:
    """Use Kit's NumPy upload API, avoiding a Python integer per channel."""
    data, size = rgba_bytes(image)
    return np.frombuffer(data, dtype=np.uint8), size


class PreviewClock:
    """Map simulation timestamps to a monotonic thermal-model time.

    A timestamp that moves backwards denotes Stop/Play replay and starts a new
    epoch. The returned time remains continuous so a solver is never asked to
    run backwards.
    """

    def __init__(self) -> None:
        self._last_sim: float | None = None
        self._base_sim = 0.0
        self._elapsed = 0.0
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def sample(self, simulation_time: float) -> tuple[float, int]:
        value = float(simulation_time)
        if not math.isfinite(value):
            raise ValueError("simulation_time must be finite")
        if self._last_sim is None:
            self._base_sim = value
            self._last_sim = value
            self._elapsed = 0.0
            return self._elapsed, self._epoch
        if value < self._last_sim:
            self._epoch += 1
            self._base_sim = value
            self._last_sim = value
            return self._elapsed, self._epoch
        self._elapsed += value - self._last_sim
        self._last_sim = value
        return self._elapsed, self._epoch


class CleanupStack:
    """Run resource finalizers in reverse order, safely and only once."""

    def __init__(self) -> None:
        self._callbacks: list[Callable[[], object]] = []
        self._closed = False

    def add(self, callback: Callable[[], object]) -> Callable[[], object]:
        if self._closed:
            raise RuntimeError("cannot add cleanup callback after close")
        self._callbacks.append(callback)
        return callback

    def close(self) -> list[str]:
        if self._closed:
            return []
        self._closed = True
        errors: list[str] = []
        while self._callbacks:
            callback = self._callbacks.pop()
            try:
                callback()
            except Exception as exc:  # cleanup must continue for remaining resources
                errors.append(f"{type(exc).__name__}: {exc}")
        return errors
