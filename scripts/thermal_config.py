"""User-owned temperature/material configuration. No Kit or USD imports.

YAML uses Celsius; irsim solvers use kelvin. Bindings match prim paths,
not display names. The most specific matching path wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import math
from pathlib import Path

import yaml

DEFAULT_THERMAL_CONFIG = Path(__file__).resolve().parents[1] / 'configs/crazyflie_thermal.yaml'


class ThermalConfigError(ValueError):
    """Invalid user configuration; do not silently substitute another value."""


class _UniqueKeyLoader(getattr(yaml,'CSafeLoader',yaml.SafeLoader)):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ThermalConfigError('YAML keys must be text')
            if key in result:
                raise ThermalConfigError(f'duplicate YAML key {key!r}, line {key_node.start_mark.line + 1}')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _fields(value, allowed, required, where):
    if not isinstance(value, dict):
        raise ThermalConfigError(f'{where}: expected a mapping')
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ThermalConfigError(f'{where}: unknown fields {sorted(unknown)}; missing fields {sorted(missing)}')
    return value


def _text(value, where):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ThermalConfigError(f'{where}: expected non-empty text without surrounding spaces')
    return value


def _number(value, where):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ThermalConfigError(f'{where}: expected a finite number, not text or true/false')
    return float(value)


@dataclass(frozen=True)
class ThermalTarget:
    name: str
    mode: str
    temperature_k: float | None = None
    offset_k: float = 0.0


@dataclass(frozen=True)
class ThermalBinding:
    name: str
    prim_path: str
    include_children: bool
    target: str
    material: str

    def matches(self, prim_path):
        return (prim_path == self.prim_path or
                (self.include_children and prim_path.startswith(self.prim_path.rstrip('/') + '/')))


@dataclass(frozen=True)
class ThermalConfig:
    schema_version: int
    default_target: str
    default_material: str
    targets: tuple[ThermalTarget, ...]
    bindings: tuple[ThermalBinding, ...]
    source_path: str
    sha256: str

    def target(self, name):
        for target in self.targets:
            if target.name == name:
                return target
        raise ThermalConfigError(f'unknown target {name!r}')

    def binding_for(self, prim_path):
        exact = self._binding_index.get(prim_path)
        if exact is not None:
            return exact
        ancestor = prim_path
        while ancestor != '/':
            ancestor = ancestor.rpartition('/')[0] or '/'
            binding = self._binding_index.get(ancestor)
            if binding is not None and binding.include_children:
                return binding
        return None

    @cached_property
    def _binding_index(self):
        return {binding.prim_path: binding for binding in self.bindings}

    def target_for(self, prim_path):
        binding = self.binding_for(prim_path)
        return self.default_target if binding is None else binding.target

    def material_for(self, prim_path):
        binding = self.binding_for(prim_path)
        return self.default_material if binding is None else binding.material


def load_thermal_config(path=DEFAULT_THERMAL_CONFIG):
    path = Path(path).resolve()
    try:
        raw = path.read_bytes()
        data = yaml.load(raw.decode('utf-8-sig'), Loader=_UniqueKeyLoader)
        _fields(data, ('schema_version', 'defaults', 'targets', 'bindings'),
                ('schema_version', 'defaults', 'targets', 'bindings'), 'config')
        if type(data['schema_version']) is not int or data['schema_version'] != 1:
            raise ThermalConfigError('schema_version must be 1')
        defaults = _fields(data['defaults'], ('target', 'material'), ('target', 'material'), 'defaults')
        default_target = _text(defaults['target'], 'defaults.target')
        default_material = _text(defaults['material'], 'defaults.material')
        if not isinstance(data['targets'], dict) or not data['targets']:
            raise ThermalConfigError('targets: expected a non-empty mapping')
        targets = []
        for name, spec in data['targets'].items():
            _text(name, 'target name')
            _fields(spec, ('mode', 'temperature_c', 'offset_c'), ('mode',), f'targets.{name}')
            mode = spec['mode']
            if mode == 'fixed':
                _fields(spec, ('mode', 'temperature_c'), ('mode', 'temperature_c'), name)
                temperature = _number(spec['temperature_c'], f'{name}.temperature_c') + 273.15
                if temperature <= 0:
                    raise ThermalConfigError(f'{name}: temperature must be above -273.15 C')
                targets.append(ThermalTarget(name, mode, temperature_k=temperature))
            elif mode == 'offset':
                _fields(spec, ('mode', 'offset_c'), ('mode', 'offset_c'), name)
                targets.append(ThermalTarget(name, mode, offset_k=_number(spec['offset_c'], f'{name}.offset_c')))
            elif mode == 'ambient':
                _fields(spec, ('mode',), ('mode',), name)
                targets.append(ThermalTarget(name, mode))
            else:
                raise ThermalConfigError(f'{name}: mode must be fixed, ambient or offset')
        names = {target.name for target in targets}
        if default_target not in names:
            raise ThermalConfigError(f'defaults.target: unknown target {default_target!r}')
        if not isinstance(data['bindings'], list):
            raise ThermalConfigError('bindings: expected a list')
        bindings, seen_names, seen_paths = [], set(), set()
        fields = ('name', 'prim_path', 'include_children', 'target', 'material')
        for item in data['bindings']:
            _fields(item, fields, fields, 'binding')
            name = _text(item['name'], 'binding.name')
            prim = _text(item['prim_path'], f'{name}.prim_path')
            if prim != '/' and (not prim.startswith('/') or
                               not all(segment.isidentifier() for segment in prim[1:].split('/'))):
                raise ThermalConfigError(f'{name}: use an absolute Prim path, e.g. /World/Cube')
            if type(item['include_children']) is not bool:
                raise ThermalConfigError(f'{name}.include_children: use true or false without quotes')
            target = _text(item['target'], f'{name}.target')
            material = _text(item['material'], f'{name}.material')
            if target not in names:
                raise ThermalConfigError(f'{name}: unknown target {target!r}')
            if name in seen_names or prim in seen_paths:
                raise ThermalConfigError(f'{name}: duplicate binding name or prim_path {prim!r}')
            seen_names.add(name)
            seen_paths.add(prim)
            bindings.append(ThermalBinding(name, prim, item['include_children'], target, material))
        return ThermalConfig(1, default_target, default_material, tuple(targets), tuple(bindings),
                             str(path), hashlib.sha256(raw).hexdigest())
    except (OSError, UnicodeError, yaml.YAMLError, ThermalConfigError) as exc:
        raise ThermalConfigError(f'{path}: {exc}') from exc
