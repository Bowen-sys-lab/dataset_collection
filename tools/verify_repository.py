"""Verify this repository and optionally the separately installed assets/dependency."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.dataset_handoff import verify_manifest


def check(root, *, assets=False, dependency=False):
    root = Path(root).resolve()
    receiver = root/'handoff/receiver'
    checked, issues, not_checked = [], [], []
    tasks = [('repository', receiver/'repository-manifest.json', root)]
    if assets:
        tasks.append(('scene_assets', receiver/'assets-manifest.json', root))
    else:
        not_checked.append('scene_assets')
    if dependency:
        path = receiver/'dependency-lock.json'
        if path.is_file():
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            dep_root = (root/data['directory']).resolve()
            if not dep_root.is_relative_to(root):
                issues.append({'path':data['directory'],'reason':'outside_root'})
            else:
                tasks.append(('thermal_dependency', path, dep_root))
        else:
            issues.append({'path':str(path),'reason':'missing_manifest'})
    else:
        not_checked.append('thermal_dependency')
    for kind, path, content_root in tasks:
        if not path.is_file():
            issues.append({'path':str(path),'reason':'missing_manifest'})
            continue
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        issues.extend(verify_manifest(content_root, data['files']))
        checked.append({'kind':kind,'files':len(data['files'])})
    return {'result':'PASS' if not issues else 'FAIL','checked':checked,'issues':issues,
            'not_checked':not_checked,
            'scope':'File hashes only. Does not certify installed Isaac, GPU, live flight, task success or formal readiness.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--assets',action='store_true')
    parser.add_argument('--dependency',action='store_true')
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    result = check(args.root,assets=args.assets,dependency=args.dependency)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=True))
    return 0 if result['result']=='PASS' else 1


if __name__=='__main__': raise SystemExit(main())
