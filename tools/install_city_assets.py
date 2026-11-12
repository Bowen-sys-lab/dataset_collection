"""Verify and extract a maintainer-provided City asset package into the project root."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'resources/city/city-assets-manifest.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True,
                        help='Path to the private City-only ZIP supplied by the project owner')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    package = args.archive.expanduser().resolve()
    data = json.loads(MANIFEST.read_text(encoding='utf8'))
    if not package.is_file():
        raise FileNotFoundError(package)
    if package.stat().st_size != data['bytes'] or sha(package) != data['sha256']:
        raise ValueError('City archive SHA256 or byte size does not match city-assets-manifest.json')
    if not args.check_only:
        with zipfile.ZipFile(package) as archive:
            for item in archive.infolist():
                target = (ROOT/Path(item.filename)).resolve()
                if not target.is_relative_to(ROOT) or item.filename.replace('\\', '/') != item.filename:
                    raise ValueError(f'Unsafe archive path: {item.filename}')
            archive.extractall(ROOT)
    failures = []
    for row in data['files']:
        path = ROOT/row['path']
        if not path.is_file():
            failures.append({'path': row['path'], 'reason': 'missing'})
        elif path.stat().st_size != row['size'] or sha(path) != row['sha256']:
            failures.append({'path': row['path'], 'reason': 'changed'})
    if failures:
        raise ValueError(json.dumps(failures[:10], ensure_ascii=False))
    print(json.dumps({'result': 'PASS', 'scene': data['scene'], 'files': len(data['files']),
                      'archive': package.name, 'check_only': args.check_only}, ensure_ascii=False))


if __name__ == '__main__':
    main()
