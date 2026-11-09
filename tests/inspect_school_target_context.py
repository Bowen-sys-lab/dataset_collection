"""Read-only report of world-space scenery nearest the school Spot target."""

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path

from scripts.scene_config import load_scene_config

ROOT = Path(__file__).resolve().parents[1]


def _bbox_record(prim, cache):
    from pxr import UsdGeom

    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return None
    lo, hi = box.GetMin(), box.GetMax()
    minimum = [float(lo[i]) for i in range(3)]
    maximum = [float(hi[i]) for i in range(3)]
    center = [(minimum[i] + maximum[i]) / 2.0 for i in range(3)]
    size = [maximum[i] - minimum[i] for i in range(3)]
    return {"min": minimum, "max": maximum, "center": center, "size": size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/school_vln_scene.yaml")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = load_scene_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    source_hash = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": True,
            "renderer": "PathTracing",
            "multi_gpu": False,
            "samples_per_pixel_per_frame": 1,
        }
    )
    report = {
        "result": "FAIL",
        "stage_path": str(config.scene.stage),
        "source_usd_sha256_before": source_hash,
        "units": "USD stage units (expected meters); Z-up; horizontal separation uses X/Y",
    }
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        context = omni.usd.get_context()
        if not context.open_stage(str(config.scene.stage)):
            raise RuntimeError(f"Could not open stage: {config.scene.stage}")
        for _ in range(240):
            app.update()
            if not any(context.get_stage_loading_status()):
                break
        stage = context.get_stage()
        if stage is None:
            raise RuntimeError("USD context has no open stage")

        time_code = Usd.TimeCode.Default()
        cache = UsdGeom.BBoxCache(
            time_code,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        target_path = "/World/Target_01"
        target = stage.GetPrimAtPath(target_path)
        if not target.IsValid():
            raise RuntimeError(f"Target prim does not exist: {target_path}")
        target_box = _bbox_record(target, cache)
        if target_box is None:
            raise RuntimeError("Target has no composed world-space bounds")

        candidates = []
        world = stage.GetPrimAtPath("/World")
        for prim in world.GetChildren():
            if prim.GetPath().pathString == target_path:
                continue
            box = _bbox_record(prim, cache)
            if box is None:
                continue
            # Minimum horizontal edge-to-edge gap between the aligned bounds.
            dx = max(box["min"][0] - target_box["max"][0],
                     target_box["min"][0] - box["max"][0], 0.0)
            dy = max(box["min"][1] - target_box["max"][1],
                     target_box["min"][1] - box["max"][1], 0.0)
            center_dx = box["center"][0] - target_box["center"][0]
            center_dy = box["center"][1] - target_box["center"][1]
            candidates.append(
                {
                    "prim_path": prim.GetPath().pathString,
                    "horizontal_bbox_gap_m": math.hypot(dx, dy),
                    "center_horizontal_distance_m": math.hypot(center_dx, center_dy),
                    "center_delta_xy_m": [center_dx, center_dy],
                    "bbox": box,
                }
            )
        candidates.sort(
            key=lambda item: (item["horizontal_bbox_gap_m"],
                              item["center_horizontal_distance_m"], item["prim_path"])
        )
        detailed_roots = [
            path for path in ("/World/hospital", "/World/office", "/World/E_school3_010_12")
            if stage.GetPrimAtPath(path).IsValid()
        ]
        nearby_meshes = {}
        for root_path in detailed_roots:
            root_prim = stage.GetPrimAtPath(root_path)
            root_candidates = []
            for prim in Usd.PrimRange(root_prim):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                box = _bbox_record(prim, cache)
                if box is None:
                    continue
                dx = max(box["min"][0] - target_box["max"][0],
                         target_box["min"][0] - box["max"][0], 0.0)
                dy = max(box["min"][1] - target_box["max"][1],
                         target_box["min"][1] - box["max"][1], 0.0)
                center_dx = box["center"][0] - target_box["center"][0]
                center_dy = box["center"][1] - target_box["center"][1]
                root_candidates.append(
                    {
                        "prim_path": prim.GetPath().pathString,
                        "horizontal_bbox_gap_m": math.hypot(dx, dy),
                        "center_horizontal_distance_m": math.hypot(center_dx, center_dy),
                        "center_delta_xy_m": [center_dx, center_dy],
                        "bbox": box,
                    }
                )
            root_candidates.sort(
                key=lambda item: (item["horizontal_bbox_gap_m"],
                                  item["center_horizontal_distance_m"], item["prim_path"])
            )
            nearby_meshes[root_path] = root_candidates[:20]

        report.update(
            {
                "result": "PASS",
                "target_path": target_path,
                "target_bbox": target_box,
                "nearest_world_children": candidates[:40],
                "nearest_meshes_within_named_building_roots": nearby_meshes,
                "composed_world_child_count_with_bounds": len(candidates),
                "interpretation_limit": (
                    "Distances are computed from composed axis-aligned bounds. They establish spatial "
                    "proximity only; prim names and bounding boxes do not prove a semantic scene label."
                ),
            }
        )
    except Exception:
        report["error"] = traceback.format_exc()
        (args.out / "failure.txt").write_text(report["error"], encoding="utf-8")
    finally:
        report["source_usd_sha256_after"] = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
        report["source_unchanged"] = report["source_usd_sha256_after"] == source_hash
        (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        app.close()
    print(json.dumps(report, indent=2))
    return 0 if report.get("result") == "PASS" and report.get("source_unchanged") else 1


if __name__ == "__main__":
    raise SystemExit(main())
