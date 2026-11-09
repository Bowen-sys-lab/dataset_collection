"""列出材质库并检查温度配置里引用的材质是否都存在。

不启动 Isaac Sim：只读 YAML + 加载 irsim 材料库，几秒出结果。
输出用英文，避免 Windows 控制台中文乱码。

用法：
    & D:\\isaacsim\\python.bat scripts\\check_materials.py
    & D:\\isaacsim\\python.bat scripts\\check_materials.py --materials configs\\materials
    & D:\\isaacsim\\python.bat scripts\\check_materials.py --thermal configs\\school_vln_thermal.yaml

退出码 0 表示全部通过；1 表示有材质加载失败或配置引用了不存在的材质。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from project_materials import load_project_material_library  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--materials', default=str(ROOT / 'configs' / 'materials'),
                        help='project material directory merged over the vendor library')
    parser.add_argument('--thermal', action='append', default=[],
                        help='thermal config YAML to check, repeatable')
    args = parser.parse_args()

    directory = Path(args.materials).resolve()
    try:
        library = load_project_material_library(directory if directory.is_dir() else None)
    except Exception as exc:  # noqa: BLE001 - show the validator's own message
        print(f'FAIL material library: {exc}')
        return 1

    vendor = load_project_material_library(None)
    own = sorted(set(library) - set(vendor))
    print(f'materials: {len(library)} total '
          f'({len(vendor)} vendor + {len(own)} project)')
    if own:
        print('project additions: ' + ', '.join(own))
    print()
    print(f'{"material":34} {"source":11} {"treatment":16} {"LWIR eps":>8} {"LWIR tau":>8}')
    print('-' * 82)
    for name in sorted(library):
        spec = library[name].spec
        try:
            props = library[name].band_properties('lwir')
            eps, tau = f'{props.emissivity:.3f}', f'{props.transmittance:.3f}'
        except Exception as exc:  # noqa: BLE001
            eps, tau = 'ERR', str(exc)[:16]
        print(f'{name:34} {spec.source:11} {spec.surface_treatment:16} {eps:>8} {tau:>8}')

    status = 0
    for thermal in args.thermal:
        path = Path(thermal).resolve()
        try:
            data = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        except Exception as exc:  # noqa: BLE001
            print(f'\nFAIL {path}: unreadable ({exc})')
            status = 1
            continue
        used = {data.get('defaults', {}).get('material')}
        used |= {item.get('material') for item in data.get('bindings') or []}
        used.discard(None)
        missing = sorted(used - set(library))
        print(f'\n{path.name}: {len(used)} materials', end='')
        if missing:
            print(f' -> FAIL unknown: {missing}')
            status = 1
        else:
            print(' -> OK')
    return status


if __name__ == '__main__':
    raise SystemExit(main())
