"""Project-owned integration policy for optional unsupported RTX channels."""
from contextlib import contextmanager
from collections import OrderedDict
from functools import partial
import sys

import numpy as np


_MATERIAL_LOOKUP_CACHE_LIMIT = 8


def _approximate_bytes(value):
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(_approximate_bytes(key) + _approximate_bytes(item)
                    for key, item in value.items())
    elif isinstance(value, tuple):
        size += sum(_approximate_bytes(item) for item in value)
    return size


class MaterialLookupCache:
    """Bounded, content-sensitive lookup storage owned by one camera.

    Only normalized ID/path/material metadata is retained, never frame planes
    or temperature/geometry/sensor history. Counters are cumulative until an
    explicit ``clear(reset_stats=True)``.
    """

    def __init__(self, max_entries=_MATERIAL_LOOKUP_CACHE_LIMIT):
        if isinstance(max_entries, bool) or int(max_entries) != max_entries or max_entries <= 0:
            raise ValueError('max_entries must be a positive integer')
        self.max_entries = int(max_entries)
        self._entries = OrderedDict()
        self._hits = self._builds = self._evictions = 0

    def clear(self, *, reset_stats=False):
        """Release all lookup storage, optionally reset diagnostic counters."""
        self._entries.clear()
        if reset_stats:
            self._hits = self._builds = self._evictions = 0

    def stats(self):
        """Return a new dict suitable for diagnostic JSON output."""
        return {'hits': self._hits, 'builds': self._builds,
                'evictions': self._evictions, 'entries': len(self._entries),
                'approximate_bytes': sum(item[1] for item in self._entries.values())}

    def lookup(self, by_path, paths):
        from irsim_isaac.pipeline.material_ids import UNMAPPED_MATERIAL_ID

        # Content keys observe ID reuse, nested label edits, and resolution
        # path/material mutations, including when callers reuse their objects.
        key = (tuple(by_path.items()), tuple(paths.items()))
        cached = self._entries.get(key)
        if cached is not None:
            self._hits += 1
            self._entries.move_to_end(key)
            return cached[0]
        lookup = {ident: by_path.get(path, UNMAPPED_MATERIAL_ID)
                  for ident, path in paths.items()}
        self._builds += 1
        self._entries[key] = (lookup, _approximate_bytes(key) + _approximate_bytes(lookup))
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._evictions += 1
        return lookup


def cached_labels_to_paths(id_to_labels):
    """Compatibility wrapper; normalization always observes current content."""
    from irsim_isaac.pipeline.material_ids import labels_to_paths

    return labels_to_paths(id_to_labels)


def _material_lookup(id_to_labels, resolutions, cache=None):
    # Keep resolution validation before label normalization, as in the vendor.
    by_path = {item.path: int(item.material_id) for item in resolutions}
    paths = cached_labels_to_paths(id_to_labels)
    cache = cache if cache is not None else MaterialLookupCache()
    return cache.lookup(by_path, paths), paths


def material_id_plane_cached(instance_ids, id_to_labels, resolutions, *,
                             background_id=0, strict=False, cache=None):
    """Project-owned exact ID remap with a cached label/material lookup table."""
    from irsim_isaac.pipeline.material_ids import UNMAPPED_MATERIAL_ID

    ids = np.asarray(instance_ids)
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(f'instance_ids must be an integer plane, got {ids.dtype}')
    if ids.ndim != 2:
        raise ValueError(f'instance_ids must be (H, W), got {ids.shape}')
    lookup, paths = _material_lookup(id_to_labels, resolutions, cache)
    present = np.unique(ids)
    missing = [int(ident) for ident in present.tolist()
               if ident != background_id and int(ident) not in paths]
    if strict and missing:
        raise KeyError(
            f'instance ids {sorted(missing)} appear in the image but not in idToLabels; '
            'every rendered prim must be resolvable or the pixel silently becomes UNMAPPED'
        )
    out = np.zeros(ids.shape, dtype=np.int32)
    for ident in present.tolist():
        if ident == background_id:
            continue
        material = lookup.get(int(ident), UNMAPPED_MATERIAL_ID)
        if material != UNMAPPED_MATERIAL_ID:
            out[ids == ident] = np.int32(material)
    return out


@contextmanager
def cached_material_mapping(module, cache=None):
    """Temporarily route one vendor module's mapping call through the project cache."""
    import importlib

    modules = [module]
    for name in ('aerial_bridge', 'motion_isaac', 'point_bridge', 'mesh_bridge'):
        try:
            candidate = importlib.import_module('irsim_isaac.pipeline.' + name)
        except ImportError:
            continue
        if candidate not in modules:
            modules.append(candidate)
    originals = []
    try:
        for target in modules:
            material = getattr(target, 'material_id_plane', None)
            labels = getattr(target, 'labels_to_paths', None)
            originals.append((target, material, labels))
            if material is not None:
                target.material_id_plane = (partial(material_id_plane_cached, cache=cache)
                                            if cache is not None else material_id_plane_cached)
            if labels is not None:
                target.labels_to_paths = cached_labels_to_paths
        yield
    finally:
        for target, material, labels in originals:
            if material is not None:
                target.material_id_plane = material
            if labels is not None:
                target.labels_to_paths = labels


@contextmanager
def supported_aov_names(module):
    # IrCamera.open has no channel-selection argument. Scope the runtime override
    # to its one-time setup, then restore the vendor module immediately.
    original = module.AOV_NAMES
    module.AOV_NAMES = {key: value for key, value in original.items() if key != 'occlusion'}
    try:
        yield
    finally:
        module.AOV_NAMES = original


def camera_class(*, material_cache_enabled=False):
    from irsim_isaac.pipeline import ir_camera

    class SupportedIrCamera(ir_camera.IrCamera):
        mapping_cache_enabled = material_cache_enabled
        compact_motion_enabled = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._material_mapping_cache = (MaterialLookupCache()
                                            if self.mapping_cache_enabled else None)

        def __setattr__(self, name, value):
            # These are vendor-owned public identities. Clearing here also
            # releases metadata immediately if a replacement precedes planes().
            if name in ('scene', 'config', '_stage'):
                if name in self.__dict__ and self.__dict__[name] is not value:
                    cache = self.__dict__.get('_material_mapping_cache')
                    if cache is not None:
                        cache.clear()
            super().__setattr__(name, value)

        @property
        def material_mapping_cache(self):
            """Camera-owned storage, or None while the opt-in cache is unused."""
            return self._material_mapping_cache

        def material_cache_stats(self):
            """Cumulative lookup counters and current storage size for profiling."""
            if self._material_mapping_cache is None:
                return {'hits': 0, 'builds': 0, 'evictions': 0, 'entries': 0,
                        'approximate_bytes': 0}
            return self._material_mapping_cache.stats()

        def clear_material_cache(self, *, reset_stats=False):
            """Release lookup storage; leave diagnostic counters unless requested."""
            if self._material_mapping_cache is not None:
                self._material_mapping_cache.clear(reset_stats=reset_stats)

        def open(self, **kwargs):
            self.clear_material_cache()
            with supported_aov_names(ir_camera):
                return super().open(**kwargs)

        def close(self):
            try:
                return super().close()
            finally:
                self.clear_material_cache()

        def planes(self, **kwargs):
            from contextlib import nullcontext
            from scripts.live_motion_optimization import compact_motion_scope
            if self.mapping_cache_enabled and self._material_mapping_cache is None:
                self._material_mapping_cache = MaterialLookupCache()
            with (cached_material_mapping(ir_camera, self._material_mapping_cache)
                  if self.mapping_cache_enabled else nullcontext()), \
                    (compact_motion_scope() if self.compact_motion_enabled else nullcontext()):
                return super().planes(**kwargs)

    return SupportedIrCamera
