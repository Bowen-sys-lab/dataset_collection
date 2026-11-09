"""Prepare versioned local delivery archives; never publish or alter source assets."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT/'handoff/reproduction-20261109'
SKIP = {'.git', '__pycache__', '.venv', '.venvs', 'out', 'node_modules'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def allowed_file(path):
    return (not any(part in SKIP for part in path.parts)
            and path.suffix.lower() not in {'.lnk', '.pyc', '.pyo', '.log', '.bak'}
            and '.bak_' not in path.name and not path.name.startswith('.env'))


def add_file(root, files, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    if not source.is_relative_to(root):
        raise ValueError(f'Outside source root: {source}')
    if not source.is_file():
        raise FileNotFoundError(source)
    rel = source.relative_to(root)
    if allowed_file(rel):
        files[rel.as_posix()] = source


def add_tree(root, files, tree):
    base = root/tree
    if not base.is_dir(): raise FileNotFoundError(base)
    for source in base.rglob('*'):
        if source.is_file(): add_file(root, files, source)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf8')


def manifest(root, paths):
    return [{'path': p, 'size': (root/p).stat().st_size, 'sha256': sha(root/p)} for p in sorted(paths)]


def verify_archive(archive_path, rows):
    expected = {r['path']: r for r in rows}
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None: raise ValueError('ZIP CRC failed')
        seen = set()
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if ('\\' in item.filename or ':' in item.filename or path.is_absolute()
                    or '..' in path.parts or not path.parts or path.parts[0] != 'IsaacSim_Demo'):
                raise ValueError(f'Unsafe archive path: {item.filename}')
            if item.filename in seen: raise ValueError('Duplicate archive member')
            seen.add(item.filename)
            rel = path.relative_to('IsaacSim_Demo').as_posix()
            if rel in expected:
                row = expected.pop(rel)
                with archive.open(item) as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if item.file_size != row['size'] or digest != row['sha256']:
                    raise ValueError(f'Changed archive member: {rel}')
            elif not rel.startswith('handoff/receiver/'):
                raise ValueError(f'Unexpected archive member: {rel}')
        if expected: raise ValueError(f'Missing archive members: {list(expected)[:3]}')
    return True


def prepare():
    stage = RELEASE/'staging/IsaacSim_Demo'
    if stage.exists(): raise FileExistsError(stage)
    code, assets = {}, {}
    for tree in ['scripts', 'configs', 'tests']:
        add_tree(ROOT, code, tree)
    for source in (ROOT/'scene_intake').glob('*.py'):
        add_file(ROOT, code, source)
    for source in (ROOT/'scene_intake').glob('*.bat'):
        add_file(ROOT, code, source)
    for folder in ['templates', 'examples']:
        add_tree(ROOT, code, 'handoff/dataset-workflow-kit/'+folder)
    for name in ['check_environment.ps1', 'create_review.ps1', 'environment-baseline.json']:
        add_file(ROOT, code, ROOT/'handoff/dataset-workflow-kit'/name)
    for name in ['build_reproduction_delivery.py', 'verify_reproduction_delivery.py',
                 'build_practical_dataset_guide.py', 'build_personal_dataset_guide.py']:
        add_file(ROOT, code, ROOT/'tools'/name)
    for name in ['dataset-workflow-personal-guide.docx', 'dataset-workflow-practical-guide.docx']:
        add_file(ROOT, code, ROOT/'docs'/name)
    for source in ROOT.glob('run_*.bat'):
        add_file(ROOT, code, source)

    audits = {}
    for name in ['factory', 'forest', 'city', 'city-legacy']:
        report = ROOT/f'validation/handoff/{name}-assets-20261109.json'
        data = json.loads(report.read_text(encoding='utf-8-sig'))
        if data['outside_project']: raise ValueError('Unclassified external scene resources')
        for source in data['resolved_files']: add_file(ROOT, assets, Path(source))
        audits[name] = data
    # YAML references outside the USD closure, plus source attribution.
    for tree in ['factory/model02_thermal_v7m/configuration/materials',
                 'forest_fire_scene_local/environment',
                 'forest_fire_scene_20261103_local/configs',
                 'forest_fire_scene_20261103_local/environment',
                 'city_200m_v2_local/environment']:
        add_tree(ROOT, assets, tree)
    add_file(ROOT, assets, ROOT/'factory/model02_thermal_v7m/configuration/night_weather.csv')
    for base in ['forest_fire_scene_20261103_local', 'forest_fire_scene_local',
                 'city_200m_v2_local', 'city_night_200m_person_20261028', 'factory/model02_thermal_v7m']:
        for source in (ROOT/base).glob('*'):
            if source.is_file() and source.name.upper().startswith(('SOURCES', 'LICENSE', 'COPYRIGHT')):
                add_file(ROOT, assets, source)

    sample_id = 'city-pilot30-C20-20261108T135934150505Z'
    sample = {}
    add_tree(ROOT, sample, 'validation/trajectory_runs/'+sample_id)
    add_tree(ROOT, sample, 'validation/episodes/'+sample_id)
    # Processing logs are part of the actual sample evidence, unlike development logs.
    for source in (ROOT/'validation/episodes'/sample_id).rglob('*.log'):
        sample[source.relative_to(ROOT).as_posix()] = source
    add_file(ROOT, sample, ROOT/f'validation/city-v2/launches/{sample_id}.json')

    stage.mkdir(parents=True)
    groups = {'code': list(code), 'assets': list(assets), 'sample': list(sample)}
    for collection in [code, assets, sample]:
        for rel, source in collection.items():
            target = stage/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    dep = ROOT/'third_party/isaac-thermal-camera-simulation'
    commit = subprocess.check_output(['git', '-C', str(dep), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != '563a51b82938d7e03f4b41dfbbf8a6476b9e0be3':
        raise ValueError('Unexpected thermal dependency revision')
    if subprocess.check_output(['git', '-C', str(dep), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Thermal dependency has unrecorded local modifications')
    dep_files = [p for p in dep.rglob('*') if p.is_file() and allowed_file(p.relative_to(dep))
                 and p.relative_to(dep).parts[0] in {'src', 'configs', 'data'}]
    dep_rows = manifest(dep, [p.relative_to(dep).as_posix() for p in dep_files])
    write_json(stage/'handoff/receiver/dependency-lock.json', {
        'repository': 'https://github.com/reza-shahriari/isaac-thermal-camera-simulation',
        'commit': commit, 'directory': 'third_party/isaac-thermal-camera-simulation',
        'redistribution': 'Source excluded. Obtain this exact revision from upstream under its terms.',
        'files': dep_rows})
    write_json(stage/'handoff/receiver/source-asset-audits.json', audits)
    write_json(RELEASE/'groups.json', groups)
    print(json.dumps({'stage': str(stage), 'files': {k: len(v) for k, v in groups.items()},
                      'dependency_files': len(dep_rows)}))


def archive_all():
    stage = RELEASE/'staging/IsaacSim_Demo'
    groups = json.loads((RELEASE/'groups.json').read_text(encoding='utf8'))
    # Receiver scripts/docs and generated metadata are authored after prepare.
    groups['code'] = sorted(set(groups['code']) | {
        p.relative_to(stage).as_posix() for p in (stage/'handoff/receiver').rglob('*')
        if p.is_file() and not p.name.endswith('-manifest.json')})
    extra_assets = list((stage/'assets/isaac').rglob('*'))
    groups['assets'] = sorted(set(groups['assets']) | {
        p.relative_to(stage).as_posix() for p in extra_assets if p.is_file()})
    report = {'scope': 'Local controlled handoff; no GitHub upload', 'archives': []}
    for kind, paths in groups.items():
        rows = manifest(stage, paths)
        meta_rel = f'handoff/receiver/{kind}-manifest.json'
        write_json(stage/meta_rel, {'schema_version': 1, 'kind': kind, 'files': rows})
        target = RELEASE/f'dataset-{kind}-20261109.zip'
        if target.exists(): raise FileExistsError(target)
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
            for rel in sorted(set(paths) | {meta_rel}): out.write(stage/rel, 'IsaacSim_Demo/'+rel)
        verify_archive(target, rows)
        report['archives'].append({'path': target.name, 'files': len(rows)+1,
                                   'bytes': target.stat().st_size, 'sha256': sha(target)})
    write_json(RELEASE/'archives.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'archive'])
    args = parser.parse_args()
    prepare() if args.operation == 'prepare' else archive_all()
