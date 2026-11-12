"""Verify and extract the committed City asset package into the project root."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'resources/city/city-assets-20261112.zip'
MANIFEST = ROOT/'resources/city/city-assets-manifest.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text(encoding='utf8'))
    if not PACKAGE.is_file():
        raise FileNotFoundError(PACKAGE)
    if PACKAGE.stat().st_size != data['bytes'] or sha(PACKAGE) != data['sha256']:
        raise ValueError('City archive SHA256 or byte size does not match city-assets-manifest.json')
    if not args.check_only:
        with zipfile.ZipFile(PACKAGE) as archive:
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
                      'archive': PACKAGE.name, 'check_only': args.check_only}, ensure_ascii=False))


if __name__ == '__main__':
    main()
