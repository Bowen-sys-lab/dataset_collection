"""Bounded, time-interpolated sky tables for interactive preview only.

The reference sky is evaluated at 30 s knots on its native 0.5 degree grid.
Only this slow environmental term is interpolated; scene geometry and sensor
state still advance every captured frame. No USD/Kit calls run in the worker.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
import numpy as np


class PreviewSky:
    def __init__(self, source, start_s, interval_s=30.0):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError('sky interval must be finite and positive')
        self.source = source
        self.start_s = float(start_s)
        self.interval_s = float(interval_s)
        self.elevation = np.linspace(0., np.pi/2, 181)
        self.tilt = np.linspace(0., np.pi, 181)
        self._tables = {}
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='preview-sky')
        self._pending = None
        self._closed = False
        try:
            self._table(0)
            self._table(1)
            self._prefetch(2)
        except Exception:
            self.close()
            raise

    def __getattr__(self, name):
        return getattr(self.source, name)

    @property
    def cached_samples(self):
        return len(self._tables)

    def _evaluate(self, index):
        # Private model copy: the upstream model has mutable, unbounded caches.
        sky = deepcopy(self.source)
        t = self.start_s + index*self.interval_s
        return (np.asarray(sky.radiance(t, self.elevation)),
                np.asarray(sky.effective_radiance(t, self.tilt)))

    def _table(self, index):
        if index not in self._tables:
            if self._pending is not None and self._pending[0] == index:
                self._tables[index] = self._pending[1].result()
                self._pending = None
            else:
                self._tables[index] = self._evaluate(index)
        return self._tables[index]

    def _prefetch(self, index):
        if self._pending is None:
            self._pending = (index, self._pool.submit(self._evaluate, index))
        elif self._pending[0] != index:
            # A time jump may make a prediction obsolete. Keep at most one job.
            if self._pending[1].done() or self._pending[1].cancel():
                self._pending = None
                self._prefetch(index)

    def _sample(self, t, angles, channel):
        if self._closed:
            raise RuntimeError('sky cache is closed')
        if not math.isfinite(t):
            raise ValueError('sky time must be finite')
        fraction = (float(t)-self.start_s)/self.interval_s
        index = math.floor(fraction)
        weight = fraction-index
        left, right = self._table(index)[channel], self._table(index+1)[channel]
        values = left*(1-weight) + right*weight
        self._tables = {k: v for k, v in self._tables.items() if index <= k <= index+1}
        self._prefetch(index+2)
        grid = self.elevation if channel == 0 else self.tilt
        return np.interp(angles, grid, values)

    def radiance(self, t_s, elevation_rad):
        return self._sample(t_s, elevation_rad, 0)

    def apparent_temperature_k(self, t_s, elevation_rad):
        return self.source.lut.apparent_temperature(self.radiance(t_s, elevation_rad),
                                                     self.source.quantity)

    def effective_radiance_from_sky_view(self, t_s, sky_view_factor):
        tilt = np.arccos(np.clip(2*np.asarray(sky_view_factor)-1, -1., 1.))
        return self._sample(t_s, tilt, 1)

    def close(self):
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._tables.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
