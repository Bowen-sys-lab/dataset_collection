"""Refresh the repository file inventory after intentional maintainer changes."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dataset_handoff import file_manifest


def main():
    data = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT)
    paths = {Path(name.decode('utf8')) for name in data.split(b'\0') if name}
    output = ROOT/'handoff/receiver/repository-manifest.json'
    paths.discard(output.relative_to(ROOT))
    forbidden = {'third_party','validation','output','assets','factory','.git'}
    if any(path.parts[0] in forbidden for path in paths):
        raise ValueError('External assets or local outputs unexpectedly selected for repository inventory')
    files = [ROOT/path for path in paths if (ROOT/path).is_file()]
    manifest = {'schema_version':1,'kind':'repository_files','baseline':'collection-20261109',
                'scope':'Repository file hashes only; external assets and dependency have separate manifests.',
                'files':file_manifest(ROOT,files)}
    output.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
    print('Repository manifest files:',len(manifest['files']))


if __name__=='__main__': main()
