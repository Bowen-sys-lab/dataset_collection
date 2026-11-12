"""Build the private City asset package used by the collection workflow."""
from pathlib import Path
import hashlib
import json
import zipfile

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parents[1]
STAGING = PROJECT / 'handoff/reproduction-20261109/staging/IsaacSim_Demo'
AUDIT = PROJECT / 'validation/handoff/city-assets-20261109.json'
# Keep the private archive outside the GitHub checkout so a build cannot
# accidentally stage the restricted scene asset.
OUTPUT = PROJECT / 'handoff/private-city-assets/city-assets-20261112.zip'
MANIFEST = REPO / 'resources/city/city-assets-manifest.json'
RECEIVER_MANIFEST = REPO / 'handoff/receiver/assets-manifest.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit = json.loads(AUDIT.read_text(encoding='utf8'))
    files = {}
    for value in audit['resolved_files']:
        source = Path(value)
        relative = source.relative_to(PROJECT)
        if relative.parts[0].startswith(('factory', 'forest')):
            continue
        if relative.name == 'lesson_crazyflie.usd' or relative.parts[0].startswith('city_'):
            files[relative.as_posix()] = source
    for relative, source in {
        'assets/isaac/cf2x/cf2x.usd': STAGING/'assets/isaac/cf2x/cf2x.usd',
        'assets/isaac/cf2x/configuration/cf2x_robot_schema.usd': STAGING/'assets/isaac/cf2x/configuration/cf2x_robot_schema.usd',
        'city_200m_v2_local/environment/weather.csv': PROJECT/'city_200m_v2_local/environment/weather.csv',
        'lesson_crazyflie.usd': STAGING/'lesson_crazyflie.usd',
    }.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        files[relative] = source
    rows = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        OUTPUT.unlink()
    with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative, source in sorted(files.items()):
            archive.write(source, relative)
            rows.append({'path': relative, 'size': source.stat().st_size, 'sha256': digest(source)})
    manifest = {
        'schema_version': 1,
        'scene': 'city_200m_v3',
        'package': OUTPUT.name,
        'bytes': OUTPUT.stat().st_size,
        'sha256': digest(OUTPUT),
        'extract_to': 'D:/IsaacSim_Demo',
        'files': rows,
        'scope': 'City-only runtime scene closure plus localized Crazyflie payload and weather CSV.',
        'not_included': ['factory assets', 'forest assets', 'Isaac Sim installation', 'thermal simulator source'],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    receiver_manifest = {
        'schema_version': 1,
        'kind': 'assets',
        'scene': 'city_200m_v3',
        'files': rows,
        'scope': manifest['scope'],
        'not_included': manifest['not_included'],
    }
    RECEIVER_MANIFEST.write_text(
        json.dumps(receiver_manifest, ensure_ascii=False, indent=2), encoding='utf8'
    )
    print(json.dumps({'package': str(OUTPUT), 'files': len(rows), 'bytes': OUTPUT.stat().st_size,
                      'sha256': manifest['sha256']}))


if __name__ == '__main__':
    main()
