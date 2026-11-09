"""Version the remaining City starts, preserving pilots and historical inputs.

This prepares candidate configurations only. Spawn geometry, complete routes and
manual terminal visibility require separate preflight and flight review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collection_capture_defaults import capture_contract

VERSION = 'city20_long_start_remaining_v1'
STATUS = 'candidate_pending_spawn_preflight'
PILOTS = ('C01', 'C06', 'C11', 'C16')
START_X = {
    'C02': 195., 'C03': 210., 'C04': 240., 'C05': 250.,
    'C07': 110., 'C08': 125., 'C09': 145., 'C10': 160.,
    'C12': 55., 'C13': 85., 'C14': 100., 'C15': 115.,
    'C17': 110., 'C18': 130., 'C19': 150., 'C20': 170.,
}
SEQUENCE = [f'C{group * 5 + start:02d}' for start in range(2, 6) for group in range(4)]
TABLE = ROOT / 'configs/city_200m_v2_task_table.json'
EPISODES = ROOT / 'configs/collection_episodes/city_v2.json'
PREVIOUS = ROOT / 'docs/city20-task-manifest-2026-11-07-long-start-trial.json'
MANIFEST = ROOT / 'docs/city20-task-manifest-2026-11-08-long-start-all.json'
TASK_DIR = ROOT / 'configs' / VERSION
BACKUP = ROOT / 'validation/diagnostics/city20-remaining-long-start-20261108/backup'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def validate(table, episodes, manifest, task_documents, task_hashes):
    rows = {r['episode']: r for r in table['tasks']}
    launches = {r['episode']: r for r in episodes['episodes']}
    items = {r['episode']: r for r in manifest['tasks']}
    expected = {f'C{i:02d}' for i in range(1, 21)}
    assert len(table['tasks']) == len(episodes['episodes']) == len(manifest['tasks']) == 20
    assert set(rows) == set(launches) == set(items) == set(task_documents) == expected
    assert len({r['task_id'] for r in rows.values()}) == 20
    splits = dict(Counter(r['split'] for r in rows.values()))
    assert splits == manifest['split_counts'] == {'train': 12, 'validation': 4, 'test': 4}
    assert manifest['collection_sequence'] == SEQUENCE
    assert manifest['collection_mode'] == 'record_all_then_serial_replay'
    for ep in sorted(expected):
        row, launch, item, task = rows[ep], launches[ep], items[ep], task_documents[ep]
        evidence = task['environment_evidence']
        collection = evidence['city_collection']
        position = task['start_position']
        target = task['target_position']
        distance = round(math.dist(position[:2], target[:2]), 2)
        bearing = round(math.degrees(math.atan2(target[1]-position[1], target[0]-position[0])) % 360, 2)
        assert 100 <= distance <= 200, (ep, distance)
        assert row['start'] == launch['start_position_m'] == item['start_position_m'] == position
        assert row['yaw'] == launch['initial_yaw_degrees'] == item['initial_yaw_degrees'] == evidence['initial_yaw_degrees']
        assert row['direction'] == launch['relative_direction'] == item['relative_direction_zh'] == evidence['relative_direction']
        assert row['task_id'] == launch['task_id'] == item['task_id'] == task['task_id']
        assert row['target'] == launch['target'] == item['target'] == evidence['target']
        assert row['block'] == launch['block_index'] == evidence['block_index']
        assert row['split'] == launch['split'] == item['split'] == collection['split']
        assert row['language'] == launch['language'] == collection['language'] == 'en'
        assert row['file'] == launch['task_file'] == item['task_file']
        assert row['task_version'] == launch['task_version'] == item['task_version'] == collection['version']
        assert item['task_sha256'] == task_hashes[ep]
        assert item['instruction_en'] == task['instruction']
        assert row['start_distance_m'] == item['start_horizontal_distance_m'] == evidence['start_distance_m'] == distance
        assert launch['target_bearing_degrees'] == evidence['target_bearing_degrees'] == bearing
        assert item['source_task']['path'] == collection['source_task_path']
        assert item['source_task']['sha256'] == collection['source_task_sha256'] == digest(collection['source_task_path'])
        assert collection['episode'] == ep
        assert collection['previous_task_sha256'] == digest(collection['previous_task_path'])
        assert item['previous_task_file'] == collection['previous_task_path']
        assert item['previous_task_sha256'] == collection['previous_task_sha256']
        if ep in START_X:
            assert position == [START_X[ep], -22., 6.5]
            east = evidence['target'] == 'ExhaustVent'
            yaw = 0. if east else 180.
            delta = (bearing-yaw+180) % 360-180
            assert (0 < delta < 45) if east else (-45 < delta < 0), (ep, delta)
            assert item['collection_order'] == SEQUENCE.index(ep)+1
            assert collection['start_review_status'] == row['start_review_status'] == launch['start_review_status'] == item['status'] == STATUS
            assert not re.search(r'\d|°|℃', task['instruction']), ep
        else:
            assert item['collection_order'] is None
    for source in manifest['preserved_inputs']:
        assert digest(source['path']) == source['sha256'], source['path']
    return {'result': 'PASS', 'tasks': 20, 'new_candidates': 16, 'preserved_pilots': 4,
            'split_counts': splits, 'minimum_horizontal_distance_m': min(r['start_distance_m'] for r in rows.values()),
            'maximum_horizontal_distance_m': max(r['start_distance_m'] for r in rows.values()),
            'source_hashes_verified': True, 'spawn_route_terminal_review': 'PENDING'}


def check_written():
    table, episodes, manifest = read_json(TABLE), read_json(EPISODES), read_json(MANIFEST)
    documents = {r['episode']: yaml.safe_load(Path(r['file']).read_text(encoding='utf-8')) for r in table['tasks']}
    hashes = {r['episode']: digest(r['file']) for r in table['tasks']}
    report = validate(table, episodes, manifest, documents, hashes)
    for source in manifest['active_indexes']:
        assert digest(source['path']) == source['sha256']
    for source in manifest['previous_active_sources']:
        assert digest(source['path']) == source['sha256']
    return report


def build():
    if TASK_DIR.exists() or BACKUP.exists() or MANIFEST.exists():
        raise FileExistsError('Remaining long starts already prepared; use --check-only, preserve this version.')
    table, episodes, previous_manifest = read_json(TABLE), read_json(EPISODES), read_json(PREVIOUS)
    rows = {r['episode']: r for r in table['tasks']}
    launches = {r['episode']: r for r in episodes['episodes']}
    manifest = copy.deepcopy(previous_manifest)
    items = {r['episode']: r for r in manifest['tasks']}
    preserved_paths = (set(ROOT.glob('configs/city20_english_v1/*.yaml')) |
                       set(ROOT.glob('configs/city20_long_start_trial_v1/*.yaml')) |
                       set(ROOT.glob('configs/city_200m_v2_task_*.yaml')) |
                       set(ROOT.glob('docs/city20-task-manifest-2026-11-07*.json')))
    preserved = [{'path': str(p), 'sha256': digest(p)} for p in sorted(preserved_paths)]
    texts, documents, hashes = {}, {}, {}
    for ep, row in rows.items():
        old_path = Path(row['file'])
        task = yaml.safe_load(old_path.read_text(encoding='utf-8'))
        documents[ep] = task
        item = items[ep]
        if ep in PILOTS:
            assert row['task_version'] == 'city20_long_start_trial_v1'
            hashes[ep] = digest(old_path)
            item['collection_order'] = None
            item['collection_phase'] = 'existing_pilot_config_preserved'
            continue
        assert ep in START_X and row['task_version'] == 'city20_english_v1'
        east = row['target'] == 'ExhaustVent'
        side, travel, direction = ('left', 'eastward', '前方偏左') if east else ('right', 'westward', '前方偏右')
        yaw = 0. if east else 180.
        position = [START_X[ep], -22., 6.5]
        target = task['target_position']
        distance = round(math.dist(position[:2], target[:2]), 2)
        bearing = round(math.degrees(math.atan2(target[1]-position[1], target[0]-position[0])) % 360, 2)
        body = task['instruction'].split(' Find ', 1)[1]
        task['instruction'] = (f'The target is several blocks ahead, slightly to your {side}. '
            f'Head {travel} along the southern edge of the district, then search the streets near the target. Find {body}')
        task['start_position'] = position
        evidence = task['environment_evidence']
        evidence.update(initial_yaw_degrees=yaw, target_bearing_degrees=bearing,
                        relative_direction=direction, start_distance_m=distance)
        evidence['city_collection'].update(version=VERSION, previous_task_path=str(old_path),
            previous_task_sha256=digest(old_path), start_review_status=STATUS)
        new_path = TASK_DIR / f'{ep}.yaml'
        text = yaml.safe_dump(task, allow_unicode=True, sort_keys=False, width=100)
        texts[new_path] = text
        hashes[ep] = hashlib.sha256(text.encode('utf-8')).hexdigest()
        row.update(start=position, yaw=yaw, direction=direction, start_distance_m=distance,
                   file=str(new_path), task_version=VERSION, start_review_status=STATUS)
        launches[ep].update(start_position_m=position, initial_yaw_degrees=yaw,
            target_bearing_degrees=bearing, relative_direction=direction,
            task_file=str(new_path), task_version=VERSION, start_review_status=STATUS)
        item.update(status=STATUS, start_position_m=position, initial_yaw_degrees=yaw,
            relative_direction_zh=direction, start_horizontal_distance_m=distance,
            target_bearing_degrees=bearing, instruction_en=task['instruction'], task_file=str(new_path),
            task_sha256=hashes[ep], task_version=VERSION, previous_task_file=str(old_path),
            previous_task_sha256=digest(old_path), collection_order=SEQUENCE.index(ep)+1,
            collection_phase='remaining_s2_through_s5')
        item['instruction_zh'] = (f'目标位于前方数个街区，方向{direction}。沿城市南侧向'
            f'{"东" if east else "西"}飞行，进入目标所在街区后按英文任务识别外观与红外特征，停稳并人工确认。')
    manifest.update(plan_id='city20-long-start-all-20261108', status=STATUS,
        task_version='mixed_preserved_pilots_and_remaining_v1', remaining_task_version=VERSION,
        collection_sequence=SEQUENCE, collection_mode='record_all_then_serial_replay',
        pilot_episodes_preserved=list(PILOTS), capture_contract=capture_contract(),
        previous_manifest={'path': str(PREVIOUS), 'sha256': digest(PREVIOUS)},
        preserved_inputs=preserved, visibility_review='pending_spawn_route_and_manual_terminal_review',
        route_assumption='Southern-edge corridor at y=-22; full routes and terminal visibility remain unverified.')
    report = validate(table, episodes, manifest, documents, hashes)
    BACKUP.mkdir(parents=True, exist_ok=False)
    backups = []
    for path in (TABLE, EPISODES):
        saved = BACKUP / path.relative_to(ROOT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, saved)
        backups.append({'path': str(saved), 'sha256': digest(saved), 'original_path': str(path)})
    manifest['previous_active_sources'] = backups
    TASK_DIR.mkdir(parents=True, exist_ok=False)
    for path, text in texts.items():
        path.write_bytes(text.encode('utf-8'))
    TABLE.write_bytes(json_text(table).encode('utf-8'))
    EPISODES.write_bytes(json_text(episodes).encode('utf-8'))
    manifest['active_indexes'] = [{'path': str(p), 'sha256': digest(p)} for p in (TABLE, EPISODES)]
    manifest['configuration_validation'] = report
    MANIFEST.write_bytes(json_text(manifest).encode('utf-8'))
    return check_written()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify generated files without rewriting them.')
    args = parser.parse_args()
    print(json.dumps(check_written() if args.check_only else build(), indent=2))
