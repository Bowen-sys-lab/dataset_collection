"""Local diagnostic of one self-generated assembly checkpoint (never external pickle)."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.thermal_snapshot import load_snapshot, run_assembly_checkpoint, state_digest
from scripts.compare_thermal_snapshots import compare_assembly_results
s = load_snapshot(Path(sys.argv[1]))
r = run_assembly_checkpoint(s['assembly_checkpoint'], cache_enabled=False)
comparison = compare_assembly_results(r, {'planes': s['planes'], 'outputs': s['outputs'],
                                         'history': s['assembly_history_after']})
print(json.dumps(comparison, indent=2))
def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            yield from differences(a.get(k), b.get(k), path+'/'+k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a,b)):
            yield from differences(x, y, path+'/'+str(i))
    elif a != b:
        yield path, str(a)[:120], str(b)[:120]
print('HISTORY_DIFFERENCES', json.dumps(list(differences(r['history'], s['assembly_history_after']))[:30]))
