"""Extend a checked prefix, retaining the original exterior-start certificate."""
from copy import copy
import numpy as np

from scripts.collision_scene import check_path


class CollisionPathCursor:
    def __init__(self, scene):
        # Share immutable triangles/AABB caches; keep the certified endpoint private.
        self._scene = copy(scene)
        self._scene.start = np.array(scene.start, copy=True)
        self._segments = 0

    def extend(self, positions):
        result = check_path(self._scene, positions)
        if result['status'] == 'CLEAR':
            self._scene.start = np.asarray(positions[-1], dtype=float).copy()
            self._segments += result['segments_checked']
        return {**result, 'certified_segments_total': self._segments}
