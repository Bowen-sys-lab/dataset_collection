"""Resolve a scene profile + plan selection into JSON for the unified record.ps1.

No Isaac import. Reads configs/collection_profiles/<scene>.yaml and the profile's
plan YAML, validates target / instruction variant / start group, and prints one
JSON object. The PowerShell entry only dispatches; scene data stays in config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / 'configs' / 'collection_profiles'


def fail(message: str) -> dict:
    return {'ok': False, 'error': message}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--target', default='')
    parser.add_argument('--start-group', default='')
    parser.add_argument('--variant', type=int, default=1)
    args = parser.parse_args()

    profile_path = PROFILES / f'{args.scene}.yaml'
    if not profile_path.is_file():
        print(json.dumps(fail(f'unknown scene {args.scene!r}; expected {profile_path}')))
        return 0
    profile = yaml.safe_load(profile_path.read_text(encoding='utf-8'))
    plan_path = ROOT / profile['plan']
    if not plan_path.is_file():
        print(json.dumps(fail(f'plan missing for scene {args.scene}: {plan_path}')))
        return 0
    plan = yaml.safe_load(plan_path.read_text(encoding='utf-8'))

    targets = [item['target_id'] for item in plan.get('targets', [])]
    starts = [group['group'] for group in plan.get('start_groups', [])]
    # Profile may map user-facing CLI names to plan target ids (factory: tank -> tank_GC075).
    target_map = profile.get('targets') or {name: name for name in targets}
    target = args.target or profile.get('default_target') or next(iter(target_map), '')
    start = args.start_group or profile.get('default_start_group') or (starts[0] if starts else '')

    if target not in target_map:
        print(json.dumps(fail(f'scene {args.scene} has no target {target!r}; allowed {sorted(target_map)}')))
        return 0
    plan_target = target_map[target]
    if plan_target not in targets:
        print(json.dumps(fail(f'profile maps target {target!r} to {plan_target!r}, absent from plan {plan_path}')))
        return 0
    if start not in starts:
        print(json.dumps(fail(f'scene {args.scene} has no start group {start!r}; allowed {starts}')))
        return 0
    item = next(entry for entry in plan['targets'] if entry['target_id'] == plan_target)
    variants = item.get('instruction_variants') or []
    if not 1 <= args.variant <= len(variants):
        print(json.dumps(fail(
            f'target {target} has {len(variants)} instruction variants; got {args.variant}')))
        return 0

    entry = (ROOT / profile['backend_entry']).resolve()
    if not entry.is_file():
        print(json.dumps(fail(f'backend entry missing: {entry}')))
        return 0

    split = next(group['split'] for group in plan['start_groups'] if group['group'] == start)
    print(json.dumps({
        'ok': True, 'scene': args.scene, 'display_name': profile.get('display_name', args.scene),
        'backend': profile.get('backend', ''), 'entry': str(entry),
        'plan': str(plan_path), 'target': target, 'plan_target': plan_target,
        'start_group': start,
        'instruction_variant': args.variant, 'split': split,
        'targets': sorted(target_map), 'start_groups': starts,
        'instruction_count': len(variants),
    }, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

