"""城市场景的 target / instruction_variant / start_group 选择与可回放配置。

只读配方 `configs/city_200m_bc_plan.yaml`，不写磁盘；`runtime_configuration`
只返回字节和路径，`--CheckOnly` 因此保持非破坏性。
"""
from __future__ import annotations

import copy
import hashlib
import yaml
from pathlib import Path
from scripts.collection_capture_defaults import TRAJECTORY_PERIOD_S

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/city_200m_bc_plan.yaml'

TARGETS = ('van', 'person', 'black_car', 'fake_person')
START_GROUPS = ('A', 'B', 'D')


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan() -> dict:
    return yaml.safe_load(PLAN.read_text(encoding='utf-8'))


def resolve_selection(target: str = 'van', instruction_variant: int = 1,
                      start_group: str = 'A') -> dict:
    """Resolve one checked (target, description, start) combination into a task."""
    if target not in TARGETS:
        raise ValueError(f'Target must be one of {TARGETS}')
    if type(instruction_variant) is not int or instruction_variant not in (1, 2, 3):
        raise ValueError('InstructionVariant must be 1..3')
    plan = load_plan()
    item = next((entry for entry in plan['targets'] if entry['target_id'] == target), None)
    if item is None:
        raise ValueError(f'Target {target} is not in the plan')
    group = next((entry for entry in plan['start_groups'] if entry['group'] == start_group), None)
    if group is None:
        raise ValueError(f'StartGroup must be one of {[g["group"] for g in plan["start_groups"]]}')
    variants = item.get('instruction_variants') or []
    if len(variants) < instruction_variant:
        raise ValueError(f'Target {target} has fewer than {instruction_variant} instruction variants')
    variant = variants[instruction_variant - 1]

    task = {
        'task_id': f'city-bc-{target}',
        'instruction': variant['text'],
        'target_prim_path': item['target_prim_path'],
        'target_description': item['description'],
        'environment_description': item['environment_description'],
        'environment_landmarks': list(item.get('landmarks') or []),
        'start_position': list(group['position_m']),
        'target_position': list(item['target_position']),
        'target_position_semantics': '目标包围盒中心，米制、Z-up、世界坐标',
        'notes': '人工按 T 确认；自动评分独立进行，T 不等于自动任务通过。',
        'environment_evidence': {
            'city_target': {
                'version': 1,
                'target': target,
                'instruction_variant': instruction_variant,
                'start_group': start_group,
                'split': group['split'],
                'confirmation_mode': 'manual_only',
                'record_period_s': TRAJECTORY_PERIOD_S,
                'automatic_task_acceptance': 'NOT_EVALUATED',
                'stage': item['stage'],
            }
        },
    }
    return {'task': task, 'yaw_degrees': float(group['yaw_degrees']), 'split': group['split'],
            'target': target, 'instruction_variant': instruction_variant,
            'start_group': start_group, 'stage': item['stage'],
            'scene_template': item['scene_template']}


def runtime_configuration(selection: dict, runtime_dir: Path):
    """Absolute-path scene + task bytes for one run; nothing is written here."""
    source = ROOT / selection['scene_template']
    data = yaml.safe_load(source.read_text(encoding='utf-8'))
    data['scene']['stage'] = str((ROOT / selection['stage']).resolve())
    for section, keys in (('scene', ('thermal_scene',)),
                          ('sensors', ('sensor_config', 'thermal_config',
                                       'display_config', 'material_directory'))):
        for key in keys:
            if data[section].get(key):
                data[section][key] = str((source.parent / data[section][key]).resolve())
    task_path = Path(runtime_dir) / 'task.yaml'
    config_path = Path(runtime_dir) / 'scene.yaml'
    data['task'] = {'file': str(task_path)}
    task_bytes = yaml.safe_dump(selection['task'], allow_unicode=True, sort_keys=False).encode('utf-8')
    config_bytes = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode('utf-8')
    return config_path, config_bytes, task_path, task_bytes


def selection_summary(selection: dict) -> dict:
    task = copy.deepcopy(selection['task'])
    task.pop('environment_evidence', None)
    return {'target': selection['target'], 'instruction_variant': selection['instruction_variant'],
            'start_group': selection['start_group'], 'split': selection['split'],
            'yaw_degrees': selection['yaw_degrees'], 'task': task}
