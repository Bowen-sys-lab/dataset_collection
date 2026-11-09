"""Inventory source USD dependencies without launching Kit. Unresolved != proven active."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--isaac-root',type=Path,default=Path(r'D:\isaacsim'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    libs=sorted((args.isaac_root/'extscache').glob('omni.usd.libs-*/pxr'))
    if len(libs)!=1: raise ValueError('Expected exactly one pinned Isaac USD library; do not guess versions')
    usd=libs[0].parent; sys.path.insert(0,str(usd))
    handles=[os.add_dll_directory(str(p)) for p in [usd/'bin',args.isaac_root/'kit',args.isaac_root/'kit/bin'] if p.is_dir()]
    from pxr import UsdUtils
    layers,assets,missing=UsdUtils.ComputeAllDependencies(str(args.stage.resolve()))
    files={Path(layer.realPath).resolve() for layer in layers if layer.realPath}
    files.update(Path(p).resolve() for p in assets if Path(p).is_file())
    root=Path(__file__).resolve().parents[1]
    report={'stage':str(args.stage.resolve()),'resolved_files':[str(p) for p in sorted(files)],
            'resolved_bytes':sum(p.stat().st_size for p in files),'unresolved':sorted(set(str(p) for p in missing)),
            'outside_project':[str(p) for p in sorted(files) if not p.is_relative_to(root)],
            'result':'NEEDS_ENGINEERING_REVIEW' if missing else 'STATIC_DEPENDENCIES_RESOLVED',
            'limitations':'Scans source layers, including inactive branches. Kit built-in MDL may appear unresolved. Does not certify the active render, config/weather/thermal closure or licenses.'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'layers':len(layers),'resolved_files':len(files),'unresolved':len(report['unresolved']),
                      'result':report['result'],'report':str(args.out)},ensure_ascii=True))


if __name__=='__main__': main()
