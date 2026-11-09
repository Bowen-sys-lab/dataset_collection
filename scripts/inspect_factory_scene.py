"""Read-only inventory of the repaired factory USD for inspection planning.

The script never authors a USD layer.  It is intentionally usable with the
Isaac Sim Python launcher because ``pxr`` is supplied by Kit, not by the
system Python installation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _vec(values):
    return [float(values[i]) for i in range(3)]


def _bbox(stage, prim):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                               UsdGeom.Tokens.proxy], useExtentsHint=True)
    try:
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    except Exception:
        return None
    if box.IsEmpty():
        return None
    minimum, maximum = _vec(box.GetMin()), _vec(box.GetMax())
    if not all(math.isfinite(value) for value in (*minimum, *maximum)):
        return None
    return {
        "min": minimum,
        "max": maximum,
        "center": [(a + b) / 2.0 for a, b in zip(minimum, maximum)],
        "size": [b - a for a, b in zip(minimum, maximum)],
    }


def _material_bindings(prim):
    from pxr import UsdShade

    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    if not material:
        return []
    return [str(material.GetPath())]


def inspect(stage_path: Path, out_dir: Path) -> dict:
    from pxr import Usd, UsdGeom

    stage_path = stage_path.resolve()
    stage = Usd.Stage.Open(str(stage_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {stage_path}")
    stage.Load()
    root = stage.GetPseudoRoot()
    report = {
        "stage": str(stage_path),
        "source_sha256": hashlib.sha256(stage_path.read_bytes()).hexdigest(),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "root_children": [str(child.GetPath()) for child in root.GetChildren()],
        "geometry": [],
        "materials": {},
        "root_bbox": _bbox(stage, root),
    }
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        bbox = _bbox(stage, prim)
        if not bbox:
            continue
        bindings = _material_bindings(prim)
        item = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "kind": str(prim.GetMetadata("kind") or ""),
            "bbox": bbox,
            "material_paths": bindings,
            "visibility": str(UsdGeom.Imageable(prim).ComputeVisibility()),
            "instance_proxy": bool(prim.IsInstanceProxy()),
        }
        report["geometry"].append(item)
        for material_path in bindings:
            report["materials"].setdefault(material_path, []).append(str(prim.GetPath()))
    report["geometry_count"] = len(report["geometry"])
    report["material_count"] = len(report["materials"])
    report["bounds"] = {
        "min": report["root_bbox"]["min"] if report["root_bbox"] else None,
        "max": report["root_bbox"]["max"] if report["root_bbox"] else None,
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "scene_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # A compact candidate list is easier to inspect manually than the full
    # geometry report.  It does not assert semantic labels; selection remains
    # evidence-based and is recorded separately after rendering.
    candidates = sorted(
        (item for item in report["geometry"] if max(item["bbox"]["size"]) >= 0.25),
        key=lambda item: (-max(item["bbox"]["size"]), item["path"]))
    (out_dir / "geometry_candidates.json").write_text(
        json.dumps(candidates[:500], ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = inspect(args.stage, args.out)
    print(json.dumps({
        "stage": report["stage"],
        "geometry_count": report["geometry_count"],
        "material_count": report["material_count"],
        "bounds": report["bounds"],
        "out": str(args.out.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
