"""Lossless shared metadata for new replays, with legacy JSON compatibility.

Use ``load_frame_metadata(path)`` for one frame or reuse ``FrameMetadataReader``
when loading a sequence. Both return the original complete metadata dictionary;
the transport fields used for deduplication are removed.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 2
COMMON_FILENAME = 'common_metadata.json'
# Only run-level configuration/provenance is eligible. Sensor state,
# temperatures, poses, timestamps, and frame-specific sync checks remain per-frame.
COMMON_FIELDS = frozenset({
    'thermal_configuration', 'experiment_conditions', 'sensor_config',
    'rgb_render_options', 'display_profile',
    'scene_config_path', 'scene_config_sha256', 'stage_path', 'source_stage_sha256',
    'thermal_environment_config_path', 'thermal_environment_config_sha256',
    'display_config_path', 'display_config_sha256',
    'static_target_paths', 'drone_prim_path', 'drone_body_path',
    'rgb_camera_path', 'thermal_camera_path', 'rgb_render_camera_path',
    'resolution', 'rgb_resolution', 'thermal_resolution',
    'rgb_intrinsics_px', 'thermal_intrinsics_px',
    'sensor_config_path', 'sensor_config_sha256', 'thermal_config_path', 'thermal_config_sha256',
    'fallback_material', 'fallback_material_records',
    'temperature_unit', 'thermal_target_binding', 'sky_time_interpolation_s', 'limitation',
    'ao_aov_enabled', 'ao_policy', 'renderer', 'disabled_viewports',
    'shared_live_render_product', 'material_cache_enabled',
    'task_id', 'task_result_status', 'trajectory_sha256', 'replay_source',
    'flat_field_override', 'flat_field_enabled', 'display_only_correction',
    'raw_dn_unchanged', 'radiance_unchanged', 'apparent_temperature_unchanged',
    'sync_status', 'sync_validation_status', 'sync_validation_scope',
    'sync_validation_evidence', 'sync_note',
})
TRANSPORT_FIELDS = frozenset({
    'metadata_schema_version', 'common_metadata_ref', 'common_metadata_sha256',
    'common_metadata_omitted_fields',
})


def _json_object(raw, path):
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected a JSON object')
    return value


class FrameMetadataWriter:
    """Write configuration once without discarding later changes.

    Frame overrides/removals are explicit, so configuration changes cannot
    silently inherit an obsolete value. Existing files are never overwritten.
    """

    def __init__(self, output: str | Path, initial_metadata: dict):
        if TRANSPORT_FIELDS.intersection(initial_metadata):
            raise ValueError('frame metadata contains reserved transport fields')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        existing_frames = list(self.output.glob('frame_*.json'))
        if existing_frames:
            raise FileExistsError(f'frame metadata already exists in {self.output}')
        self._common = deepcopy({key: value for key, value in initial_metadata.items()
                                 if key in COMMON_FIELDS})
        payload = json.dumps({'metadata_schema_version': SCHEMA_VERSION, 'metadata': self._common},
                             ensure_ascii=False, indent=2).encode('utf-8')
        with (self.output / COMMON_FILENAME).open('xb') as stream:
            stream.write(payload)
        self.reference = {
            'metadata_schema_version': SCHEMA_VERSION,
            'common_metadata_ref': COMMON_FILENAME,
            'common_metadata_sha256': hashlib.sha256(payload).hexdigest(),
        }

    @property
    def summary(self):
        return {
            'schema_version': SCHEMA_VERSION,
            'common_metadata_file': COMMON_FILENAME,
            'common_metadata_sha256': self.reference['common_metadata_sha256'],
            'common_metadata_fields': sorted(self._common),
        }

    def ensure_available(self, index: int):
        frame_path = self.output / f'frame_{index:06d}.json'
        if frame_path.exists():
            raise FileExistsError(frame_path)

    def write(self, index: int, metadata: dict):
        if TRANSPORT_FIELDS.intersection(metadata):
            raise ValueError('frame metadata contains reserved transport fields')
        frame = {key: value for key, value in metadata.items()
                 if key not in self._common or value != self._common[key]}
        omitted = sorted(self._common.keys() - metadata.keys())
        frame.update(self.reference)
        if omitted:
            frame['common_metadata_omitted_fields'] = omitted
        with (self.output / f'frame_{index:06d}.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(frame, ensure_ascii=False, indent=2))


class FrameMetadataReader:
    """Resolve shared files once per unchanged file, checking their saved SHA-256."""

    def __init__(self):
        self._cache = {}

    def read(self, path: str | Path, *, fields=None) -> dict:
        """Optionally select fields after validating the complete transport contract."""
        path = Path(path).resolve()
        frame = _json_object(path.read_text(encoding='utf-8'), path)
        if not TRANSPORT_FIELDS.intersection(frame):
            return frame if fields is None else {key: value for key, value in frame.items() if key in fields}
        if type(frame.get('metadata_schema_version')) is not int or \
                frame['metadata_schema_version'] != SCHEMA_VERSION:
            raise ValueError(f'{path}: unsupported frame metadata schema version')
        ref = frame.get('common_metadata_ref')
        if not isinstance(ref, str) or not ref or '/' in ref or '\\' in ref or ':' in ref:
            raise ValueError(f'{path}: common metadata reference must name a file in the frame directory')
        common_path = (path.parent / ref).resolve()
        if common_path.parent != path.parent or not common_path.is_file():
            raise ValueError(f'{path}: missing or invalid common metadata reference')
        expected_hash = frame.get('common_metadata_sha256')
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or \
                any(c not in '0123456789abcdef' for c in expected_hash):
            raise ValueError(f'{path}: invalid common metadata hash')
        stat = common_path.stat()
        signature = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size, stat.st_ino)
        cached = self._cache.get(common_path)
        if cached is None or cached[0] != signature:
            raw = common_path.read_bytes()
            actual_hash = hashlib.sha256(raw).hexdigest()
            document = _json_object(raw, common_path)
            if document.get('metadata_schema_version') != SCHEMA_VERSION or \
                    not isinstance(document.get('metadata'), dict):
                raise ValueError(f'{common_path}: invalid common metadata schema')
            common = document['metadata']
            if TRANSPORT_FIELDS.intersection(common):
                raise ValueError(f'{common_path}: reserved transport fields in common metadata')
            cached = (signature, actual_hash, common)
            self._cache[common_path] = cached
        if cached[1] != expected_hash:
            raise ValueError(f'{path}: common metadata hash mismatch')
        omitted = frame.get('common_metadata_omitted_fields', [])
        if not isinstance(omitted, list) or any(not isinstance(key, str) or key not in cached[2]
                                                for key in omitted):
            raise ValueError(f'{path}: invalid omitted common metadata fields')
        complete = deepcopy(cached[2] if fields is None else
                            {key: value for key, value in cached[2].items() if key in fields})
        for key in omitted:
            complete.pop(key, None)
        complete.update({key: value for key, value in frame.items() if key not in TRANSPORT_FIELDS
                         and (fields is None or key in fields)})
        return complete


def load_frame_metadata(path: str | Path) -> dict:
    """Read a complete frame dictionary in either old or compact format."""
    return FrameMetadataReader().read(path)
