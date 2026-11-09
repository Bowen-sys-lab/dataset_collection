"""USD 级探针：场景里到底有什么、热绑定覆盖多少、相机/动画还在不在。

这是接包判定的核心：文件级检查只能说明"东西在"，探针说明"东西还能用"。
同时带 `--subtree` 用来摸某个子树的结构、API schema、引用来源和变换，
排查"能不能把某台机器人塞进这个场景"这类问题。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\probe_scene.py ^
        --stage D:\\IsaacSim_Demo\\forest_fire_scene_local\\scene.usda ^
        --thermal-config D:\\IsaacSim_Demo\\configs\\forest_fire_thermal.yaml ^
        --subtree /World/Crazyflie ^
        --json D:\\IsaacSim_Demo\\scene_intake\\out\\probe_fire.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
import yaml  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402


def subtree(stage, root: str, limit: int = 80):
    prim = stage.GetPrimAtPath(root)
    if not prim or not prim.IsValid():
        return {"root": root, "missing": True}
    rows = []
    for p in Usd.PrimRange(prim):
        refs = p.GetMetadata("references")
        payloads = p.GetMetadata("payload")
        xf = UsdGeom.Xformable(p)
        row = {
            "path": str(p.GetPath()),
            "type": p.GetTypeName(),
            "schemas": list(p.GetAppliedSchemas()),
            "active": bool(p.IsActive()),
            "defined": bool(p.IsDefined()),
            "kind": p.GetMetadata("kind"),
            "references": str(refs) if refs else None,
            "payload": str(payloads) if payloads else None,
            "translate": None,
        }
        if xf:
            ops = [op.GetOpName() for op in xf.GetOrderedXformOps()]
            row["xform_ops"] = ops
        rows.append(row)
        if len(rows) >= limit:
            rows.append({"truncated": True})
            break
    return {"root": root, "prim_count": len(rows), "prims": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--subtree", action="append", default=[])
    ap.add_argument("--thermal-config", default="")
    ap.add_argument("--camera", default="/World/Crazyflie/body/Camera_mount/Camera",
                    help="用于写 inspection_camera.json 的相机 Prim")
    ap.add_argument("--write-inspection-camera", default="",
                    help="把该相机的世界矩阵写到这个 JSON（author 的 --saved-view 需要）")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    stage_path = Path(args.stage).resolve()
    ctx = omni.usd.get_context()
    ctx.open_stage(str(stage_path))
    for _ in range(60):
        app.update()
    stage = ctx.get_stage()

    prims = list(stage.Traverse())
    meshes = [p for p in prims if p.IsA(UsdGeom.Mesh)]
    cameras = [p for p in prims if p.IsA(UsdGeom.Camera)]
    lights = [p for p in prims if p.GetTypeName() in ("SphereLight", "DistantLight", "DomeLight", "RectLight")]
    articulations = [p for p in prims if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    rigid_bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    physics_scenes = [p for p in prims if p.IsA(UsdPhysics.Scene)]

    with_thermal = [p for p in meshes if p.HasAttribute("thermal:temperatureC")]

    report = {
        "stage": str(stage_path),
        "default_prim": (stage.GetDefaultPrim().GetPath().pathString
                         if stage.GetDefaultPrim() else None),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "time_range": [stage.GetStartTimeCode(), stage.GetEndTimeCode()],
        "counts": {"prims": len(prims), "meshes": len(meshes), "cameras": len(cameras),
                   "lights": len(lights), "articulation_roots": len(articulations),
                   "rigid_bodies": len(rigid_bodies), "physics_scenes": len(physics_scenes)},
        "cameras": [str(p.GetPath()) for p in cameras],
        "lights": [str(p.GetPath()) for p in lights],
        "physics": {"articulation_roots": [str(p.GetPath()) for p in articulations],
                    "rigid_bodies": [str(p.GetPath()) for p in rigid_bodies][:10],
                    "physics_scenes": [str(p.GetPath()) for p in physics_scenes]},
        "thermal": {"meshes_with_temperature": len(with_thermal),
                    "meshes_without": len(meshes) - len(with_thermal)},
        "animation": {
            "meshes_with_point_samples": sum(
                1 for p in meshes if p.GetAttribute("points") and p.GetAttribute("points").GetNumTimeSamples() > 1),
            "prims_with_any_samples": sum(
                1 for p in prims if any(a.GetNumTimeSamples() > 1 for a in p.GetAttributes())),
        },
    }

    # 与热配置对照：每个绑定路径是否还在、命中多少网格
    if args.write_inspection_camera:
        camera_prim = stage.GetPrimAtPath(args.camera)
        if not camera_prim or not camera_prim.IsValid():
            raise SystemExit(f"camera prim missing: {args.camera}")
        m = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        matrix = [[float(m[r][c]) for c in range(4)] for r in range(4)]
        payload = {"camera": args.camera, "stage": str(stage_path), "camera_to_world": matrix}
        target = Path(args.write_inspection_camera)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report["inspection_camera"] = str(target)

    if args.thermal_config:
        cfg = yaml.safe_load(Path(args.thermal_config).read_text(encoding="utf-8"))
        binding_report = []
        for b in cfg.get("bindings", []):
            path = b["prim_path"]
            prim = stage.GetPrimAtPath(path)
            hit = 0
            if prim and prim.IsValid():
                for p in Usd.PrimRange(prim):
                    if p.IsA(UsdGeom.Mesh):
                        hit += 1
            binding_report.append({"name": b.get("name"), "prim_path": path,
                                   "exists": bool(prim and prim.IsValid()), "meshes": hit})
        report["thermal_bindings"] = binding_report
        report["thermal"]["binding_misses"] = [b["name"] for b in binding_report
                                               if not b["exists"] or b["meshes"] == 0]

    if args.subtree:
        report["subtrees"] = {root: subtree(stage, root) for root in args.subtree}

    out = json.dumps(report, indent=2, default=str)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(out, encoding="utf-8")

    print(json.dumps({k: report[k] for k in report if k != "subtrees"}, indent=1, default=str))
    if args.subtree:
        for root, data in report["subtrees"].items():
            print(f"\n--- subtree {root} ---")
            if data.get("missing"):
                print("  MISSING")
                continue
            for row in data["prims"][:40]:
                if "truncated" in row:
                    print("  ... truncated")
                    continue
                ref = f" ref={row['references']}" if row["references"] else ""
                print(f"  {row['path']:52s} {row['type']:14s} {','.join(row['schemas'])}{ref}")
    print(f"\nSTATUS: measured ({report['counts']['meshes']} meshes)")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
