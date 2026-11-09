"""Check received package hashes and the separately acquired thermal dependency."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dataset_handoff import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--dependency', action='store_true')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    issues, checked = [], []
    for kind in ['code', 'assets', 'sample']:
        path = args.root/f'handoff/receiver/{kind}-manifest.json'
        if kind == 'sample' and not path.exists(): continue
        if not path.is_file():
            issues.append({'path': str(path), 'reason': 'missing_manifest'})
            continue
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        issues.extend(verify_manifest(args.root, data['files']))
        checked.append({'kind': kind, 'files': len(data['files'])})
    if args.dependency:
        path = args.root/'handoff/receiver/dependency-lock.json'
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        issues.extend(verify_manifest(args.root/data['directory'], data['files']))
        checked.append({'kind': 'thermal_dependency', 'files': len(data['files'])})
    result = {'result': 'PASS' if not issues else 'FAIL', 'checked': checked, 'issues': issues,
              'scope': 'File integrity only. Run scene/GPU/live recording acceptance separately.'}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(result, ensure_ascii=True))
    raise SystemExit(0 if not issues else 1)


if __name__ == '__main__': main()
