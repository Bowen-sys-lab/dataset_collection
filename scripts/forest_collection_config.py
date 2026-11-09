"""Versioned forest scene and target selection; legacy snapshots stay independent."""
from pathlib import Path
import copy
import yaml

ROOT=Path(__file__).resolve().parents[1]
VERSION='forest_two_target_v1'
REGISTRY=ROOT/'configs/forest_pilot_targets.yaml'


def is_two_target(row):
    return row.get('scene')=='forest' and row.get('start_version')==VERSION


def selection(row):
    registry=yaml.safe_load(REGISTRY.read_text(encoding='utf-8'))
    if registry.get('version')!=VERSION or row.get('target') not in registry['targets']:
        raise ValueError('Unsupported forest target/version')
    return registry,registry['targets'][row['target']]


def scene_config_path(row):
    if not is_two_target(row):return ROOT/'configs/forest_fire_flight_scene_night.yaml'
    registry,_=selection(row)
    return (ROOT/registry['scene_config']).resolve()


def target_config_path(row):
    if not is_two_target(row):return ROOT/'configs/forest_fire_target.yaml'
    _,target=selection(row)
    return (ROOT/target['config']).resolve()


def task_for(row):
    if not is_two_target(row):
        return yaml.safe_load((ROOT/'configs/forest_fire_task.yaml').read_text(encoding='utf-8'))
    _,item=selection(row)
    target=yaml.safe_load(target_config_path(row).read_text(encoding='utf-8'))['target']
    return dict(task_id='forest_two_target',instruction=row['instruction'],
                target_prim_path=target['prim_path'],target_position=copy.deepcopy(target['position_m']),
                target_description=item['description'],target_position_semantics=item['anchor_semantics'],
                start_position=copy.deepcopy(row['candidate_position_m']),
                environment_description='Dim forest with a dirt road, slopes, trees and separated static targets. Prescribed temperatures; RGB smoke and flame surface surrogate.',
                environment_landmarks=['forest dirt road','tree trunks and conifers','roadside barrels'])


def source_asset_root(row):
    return ROOT/('forest_fire_scene_20261103_local' if is_two_target(row) else 'forest_fire_scene_local')
