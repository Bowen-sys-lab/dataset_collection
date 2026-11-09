"""Unified, validated configuration for reusable UAV scenes.

This module has no Isaac/Kit imports, so configuration errors are reported before
the relatively expensive SimulationApp startup.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml


class SceneConfigError(ValueError):
    pass


class _UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in result:
                raise SceneConfigError(
                    f"duplicate YAML key {key!r}, line {key_node.start_mark.line + 1}"
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class ScenePaths:
    stage: Path
    thermal_scene: Path
    static_targets: tuple[str, ...] = ()
    collision_exclude_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class DroneConfig:
    prim_path: str
    body_path: str
    camera_path: str
    minimum_height: float
    #: Optional per-scene cruise speed (m/s); CLI --horizontal-speed still wins.
    horizontal_speed: float | None = None


@dataclass(frozen=True)
class SensorRuntimeConfig:
    width: int
    height: int
    sensor_config: Path
    thermal_config: Path
    preview_fast: bool
    align_thermal_to_rgb: bool = False
    flat_field_enabled: bool = False
    sensor_seed: int = 0
    display_config: Path | None = None
    material_directory: Path | None = None


@dataclass(frozen=True)
class SceneConfig:
    schema_version: int
    source_path: Path
    sha256: str
    scene: ScenePaths
    drone: DroneConfig
    sensors: SensorRuntimeConfig
    task: dict[str, Any] | None = None
    task_path: Path | None = None
    task_sha256: str | None = None


def _mapping(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise SceneConfigError(f"{name}: expected a mapping")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SceneConfigError(f"{name}: expected non-empty text")
    return value


def _path(value: Any, name: str, base: Path, *, must_exist: bool = True) -> Path:
    text = _text(value, name)
    result = Path(text)
    if not result.is_absolute():
        result = (base / result).resolve()
    else:
        result = result.resolve()
    if must_exist and not result.is_file():
        raise SceneConfigError(f"{name}: file does not exist: {result}")
    return result


def _prim(value: Any, name: str) -> str:
    text = _text(value, name)
    if not text.startswith('/') or text == '//':
        raise SceneConfigError(f"{name}: expected an absolute USD Prim path")
    parts = text[1:].split('/')
    if any(not part or not part.replace('_', 'a').isalnum() or part[0].isdigit() for part in parts):
        raise SceneConfigError(f"{name}: invalid USD Prim path: {text}")
    return text.rstrip('/') or '/'


def _fields(value: dict, allowed: set[str], required: set[str], name: str) -> dict:
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise SceneConfigError(f"{name}: unknown fields {sorted(unknown)}")
    if missing:
        raise SceneConfigError(f"{name}: missing fields {sorted(missing)}")
    return value


_TASK_FIELDS = {
    'task_id', 'instruction', 'target_prim_path', 'target_description',
    'environment_description', 'environment_landmarks', 'environment_evidence',
    'start_position', 'target_position', 'target_position_semantics', 'notes',
}


def _validate_task_json(task: dict[str, Any]) -> None:
    try:
        json.dumps(task, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SceneConfigError(f'task must contain finite JSON-compatible values: {exc}') from exc


def _load_task_file(path: Path, *, require_complete: bool = True) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        task = yaml.load(raw.decode('utf-8-sig'), Loader=_UniqueLoader)
        task = _mapping(task, 'task')
        # Old templates carried null result placeholders. Results belong to a
        # run, so accept those placeholders but never reuse a previous label.
        for result_field in ('success', 'success_time'):
            if result_field in task:
                if task.pop(result_field) is not None:
                    raise SceneConfigError(f'task.{result_field}: results must not be set in a task template')
        _fields(task, _TASK_FIELDS, {'task_id', 'instruction'} if require_complete else set(), 'task')
        for required_or_present in ('task_id', 'instruction'):
            if required_or_present in task:
                _text(task[required_or_present], f'task.{required_or_present}')
        if 'target_prim_path' in task:
            _prim(task['target_prim_path'], 'task.target_prim_path')
        for field in ('target_description', 'environment_description', 'target_position_semantics', 'notes'):
            if field in task and (not isinstance(task[field], str) or task[field] != task[field].strip()):
                raise SceneConfigError(f'task.{field}: expected trimmed text')
        if 'environment_landmarks' in task:
            landmarks = task['environment_landmarks']
            if not isinstance(landmarks, list) or any(
                    not isinstance(item, str) or not item.strip() or item != item.strip()
                    for item in landmarks):
                raise SceneConfigError('task.environment_landmarks must be a list of non-empty text')
        _validate_task_json(task)
        return copy.deepcopy(task), hashlib.sha256(raw).hexdigest()
    except (OSError, UnicodeError, yaml.YAMLError, SceneConfigError) as exc:
        raise SceneConfigError(f'{path}: {exc}') from exc


def resolve_task_config(default_task: dict[str, Any] | None, *, task_file=None,
                        cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge scene task defaults, optional task YAML, then supplied CLI overrides."""
    task = copy.deepcopy(default_task or {})
    if task_file is not None:
        path = Path(task_file).expanduser().resolve()
        loaded, _ = _load_task_file(path, require_complete=False)
        task.update(loaded)
    if cli_overrides:
        task.update({key: value for key, value in cli_overrides.items() if value is not None})
    if task:
        _fields(task, _TASK_FIELDS, set(), 'task')
        for required_or_present in ('task_id', 'instruction'):
            if required_or_present in task:
                _text(task[required_or_present], f'task.{required_or_present}')
        if 'target_prim_path' in task:
            _prim(task['target_prim_path'], 'task.target_prim_path')
        _validate_task_json(task)
    return task


def load_scene_config(path: str | Path, *, include_task: bool = True) -> SceneConfig:
    """Load runtime settings; offline replay obtains its task from the recorded run."""
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
        data = yaml.load(raw.decode('utf-8-sig'), Loader=_UniqueLoader)
        root = _mapping(data, 'config')
        _fields(root, {'schema_version', 'scene', 'drone', 'sensors', 'task'},
                {'schema_version', 'scene', 'drone', 'sensors'}, 'config')
        if type(root['schema_version']) is not int or root['schema_version'] != 1:
            raise SceneConfigError('schema_version must be 1')
        base = source.parent
        scene = _mapping(root['scene'], 'scene')
        _fields(scene, {'stage', 'thermal_scene', 'static_targets', 'collision_exclude_paths'}, {'stage', 'thermal_scene'}, 'scene')
        exclude_raw = scene.get('collision_exclude_paths', [])
        if not isinstance(exclude_raw, list):
            raise SceneConfigError('scene.collision_exclude_paths must be a list of Prim paths')
        collision_exclude_paths = tuple(_prim(p, 'scene.collision_exclude_paths') for p in exclude_raw)
        if len(set(collision_exclude_paths)) != len(collision_exclude_paths):
            raise SceneConfigError('scene.collision_exclude_paths contains duplicate paths')
        static_raw = scene.get('static_targets', [])
        if not isinstance(static_raw, list):
            raise SceneConfigError('scene.static_targets must be a list of Prim paths')
        static_targets = tuple(_prim(p, 'scene.static_targets') for p in static_raw)
        if len(set(static_targets)) != len(static_targets):
            raise SceneConfigError('scene.static_targets contains duplicate paths')
        drone = _mapping(root['drone'], 'drone')
        _fields(drone, {'prim_path', 'body_path', 'camera_path', 'minimum_height',
                         'horizontal_speed'},
                {'prim_path', 'body_path', 'camera_path', 'minimum_height'}, 'drone')
        sensors = _mapping(root['sensors'], 'sensors')
        _fields(sensors, {'width', 'height', 'sensor_config', 'thermal_config', 'preview_fast', 'align_thermal_to_rgb',
                         'flat_field_enabled', 'sensor_seed', 'display_config', 'material_directory'},
                {'width', 'height', 'sensor_config', 'thermal_config'}, 'sensors')
        width, height = sensors['width'], sensors['height']
        if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
            raise SceneConfigError('sensors.width and sensors.height must be positive integers')
        minimum = drone['minimum_height']
        horizontal_speed = drone.get('horizontal_speed')
        if horizontal_speed is not None:
            if (type(horizontal_speed) not in (int, float)
                    or not math.isfinite(float(horizontal_speed))
                    or not 0.1 <= float(horizontal_speed) <= 20.0):
                raise SceneConfigError('drone.horizontal_speed must be within 0.1..20 m/s')
            horizontal_speed = float(horizontal_speed)
        if type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum < 0:
            raise SceneConfigError('drone.minimum_height must be a finite non-negative number')
        preview_fast = sensors.get('preview_fast', True)
        if type(preview_fast) is not bool:
            raise SceneConfigError('sensors.preview_fast must be true or false')
        align = sensors.get('align_thermal_to_rgb', False)
        if type(align) is not bool:
            raise SceneConfigError('sensors.align_thermal_to_rgb must be true or false')
        flat_field = sensors.get('flat_field_enabled', False)
        if type(flat_field) is not bool:
            raise SceneConfigError('sensors.flat_field_enabled must be true or false')
        seed = sensors.get('sensor_seed', 0)
        if type(seed) is not int or seed < 0:
            raise SceneConfigError('sensors.sensor_seed must be a non-negative integer')
        display_path = None
        material_directory = None
        if sensors.get('material_directory') is not None:
            material_directory = _path(sensors['material_directory'], 'sensors.material_directory', base, must_exist=False)
            if not material_directory.is_dir() or not any(material_directory.glob('*.yaml')):
                raise SceneConfigError('sensors.material_directory: expected a directory containing material YAML files')
        if sensors.get('display_config') is not None:
            from scripts.thermal_display import DisplayProfileError, load_display_profile
            display_path = _path(sensors['display_config'], 'sensors.display_config', base)
            try:
                load_display_profile(display_path)
            except DisplayProfileError as exc:
                raise SceneConfigError(f'sensors.display_config: {exc}') from exc
        task = None
        task_path = None
        task_sha256 = None
        if root.get('task') is not None:
            task_section = _mapping(root['task'], 'task')
            _fields(task_section, {'file'}, {'file'}, 'task')
            task_path = _path(task_section['file'], 'task.file', base, must_exist=include_task)
            if include_task:
                task, task_sha256 = _load_task_file(task_path)
        return SceneConfig(
            schema_version=1,
            source_path=source,
            sha256=hashlib.sha256(raw).hexdigest(),
            scene=ScenePaths(_path(scene['stage'], 'scene.stage', base),
                             _path(scene['thermal_scene'], 'scene.thermal_scene', base), static_targets, collision_exclude_paths),
            drone=DroneConfig(_prim(drone['prim_path'], 'drone.prim_path'),
                              _prim(drone['body_path'], 'drone.body_path'),
                              _prim(drone['camera_path'], 'drone.camera_path'), float(minimum),
                                     horizontal_speed),
            sensors=SensorRuntimeConfig(width, height,
                                        _path(sensors['sensor_config'], 'sensors.sensor_config', base),
                                        _path(sensors['thermal_config'], 'sensors.thermal_config', base),
                                        preview_fast, align, flat_field, seed, display_path, material_directory),
            task=task,
            task_path=task_path,
            task_sha256=task_sha256,
        )
    except (OSError, UnicodeError, yaml.YAMLError, SceneConfigError) as exc:
        if isinstance(exc, SceneConfigError) and str(exc).startswith(str(source)):
            raise
        raise SceneConfigError(f"{source}: {exc}") from exc


DEFAULT_SCENE_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'crazyflie_scene.yaml'
