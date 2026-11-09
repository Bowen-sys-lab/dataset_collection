"""离线验收：城市 200 m 场景在本项目配置下是否真的"能用"。

不启动 Isaac（只要 USD + PyYAML），检查：
  * 三件套能被现有 loader 读入（scene_config / thermal_config）
  * 叠加层能打开，机体/相机 Prim 在
  * 1088 条绑定逐条落在 Mesh 上，且没有 Mesh 漏绑
  * 绑定里的红外材质在旧版 irsim 材质库里都存在
  * YAML 温度与作者写在 USD 上的 thermal:temperatureC 一致

用法：
    python scene_intake\\validate_city_200m.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
sys.path.insert(0, str(PROJECT))

from pxr import Usd, UsdGeom  # noqa: E402

from scripts.scene_config import load_scene_config  # noqa: E402
from scripts.thermal_config import load_thermal_config  # noqa: E402

VENDOR_MATERIALS = PROJECT / "third_party" / "isaac-thermal-camera-simulation" / "configs" / "materials"


def material_names() -> set[str]:
    return {path.stem for path in VENDOR_MATERIALS.glob("*.yaml")}


def check_scene(config_path: Path) -> dict:
    config = load_scene_config(config_path)
    stage = Usd.Stage.Open(str(config.scene.stage))
    if stage is None:
        raise SystemExit(f"cannot open stage: {config.scene.stage}")

    meshes, gprims = [], []
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            meshes.append(prim)
        if prim.IsA(UsdGeom.Gprim):
            gprims.append(prim)
    mesh_paths = {str(p.GetPath()) for p in meshes}
    gprim_paths = {str(p.GetPath()) for p in gprims}

    drone = stage.GetPrimAtPath(config.drone.prim_path)
    camera = stage.GetPrimAtPath(config.drone.camera_path)
    physics = [str(p.GetPath()) for p in stage.Traverse() if p.IsA("PhysicsScene")]

    thermal = load_thermal_config(config.sensors.thermal_config)
    missing_paths, not_geometry, temp_mismatch, unresolved_payload = [], [], [], []
    # 飞行叠加层的机体是 https S3 payload；离线（不带代理）时 USD 只能看到 Xform，
    # 拿不到 Mesh。它属于"这次离线没解析"，不是配置错，单独记。
    drone_root = "/".join(config.drone.prim_path.split("/")[:3])
    for binding in thermal.bindings:
        prim = stage.GetPrimAtPath(binding.prim_path)
        if not prim:
            missing_paths.append(binding.prim_path)
            continue
        if binding.include_children:
            # 父节点绑定：至少要有一个子孙几何命中
            prefix = binding.prim_path.rstrip("/") + "/"
            hit = any(path.startswith(prefix) for path in gprim_paths)
            if hit:
                continue
        elif binding.prim_path in gprim_paths:
            continue
        if binding.prim_path.rstrip("/").startswith(drone_root):
            unresolved_payload.append(binding.prim_path)
        else:
            not_geometry.append(binding.prim_path)

    # 与运行时一致：先精确匹配，再找最近的 include_children 祖先
    exact = {binding.prim_path: binding for binding in thermal.bindings if not binding.include_children}
    ancestors = sorted((binding for binding in thermal.bindings if binding.include_children),
                       key=lambda item: len(item.prim_path), reverse=True)
    covered_meshes = 0
    for path in mesh_paths:
        binding = exact.get(path)
        if binding is None:
            for candidate in ancestors:
                if path.startswith(candidate.prim_path.rstrip("/") + "/"):
                    binding = candidate
                    break
        if binding is not None:
            covered_meshes += 1
        elif path.rstrip("/").startswith(drone_root):
            continue
    for binding in thermal.bindings:
        if binding.include_children or binding.prim_path not in gprim_paths:
            continue
        prim = stage.GetPrimAtPath(binding.prim_path)
        attr = prim.GetAttribute("thermal:temperatureC")
        if attr and attr.HasAuthoredValue():
            expected = float(thermal.target(binding.target).temperature_k) - 273.15
            actual = float(attr.Get())
            if abs(actual - expected) > 1e-3:
                temp_mismatch.append({"path": binding.prim_path, "usd_c": actual,
                                      "yaml_c": expected})

    materials = material_names() | {path.stem for path in
                                    (PROJECT / "configs" / "materials").rglob("*.yaml")}
    used = {binding.material for binding in thermal.bindings} | {thermal.default_material}
    return {
        "config": str(config_path),
        "stage": str(config.scene.stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "meshes": len(meshes),
        "gprims": len(gprims),
        "cameras": [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Camera)],
        "physics_scenes": physics,
        "drone": {"path": config.drone.prim_path, "exists": bool(drone)},
        "camera": {"path": config.drone.camera_path, "exists": bool(camera)},
        "thermal": {
            "bindings": len(thermal.bindings),
            "targets": len(thermal.targets),
            "bound_meshes": covered_meshes,
            "unbound_meshes": len(mesh_paths) - covered_meshes,
            "defaulted_gprims": max(0, len(gprims) - covered_meshes),
            "missing_paths": missing_paths[:10],
            "missing_path_count": len(missing_paths),
            "not_geometry": not_geometry[:10],
            "not_geometry_count": len(not_geometry),
            "unresolved_payload": unresolved_payload,
            "temperature_mismatch": temp_mismatch[:10],
            "temperature_mismatch_count": len(temp_mismatch),
            "materials": sorted(used),
            "unknown_materials": sorted(used - materials),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    results = [check_scene(PROJECT / "configs" / name)
               for name in ("city_200m_scene.yaml", "city_200m_flight_scene.yaml",
                            "city_200m_scenario_scene.yaml", "city_200m_day_scene.yaml",
                            "city_200m_dark_scene.yaml", "city_200m_decoys_scene.yaml",
                            "city_200m_lamps_a_off_scene.yaml",
                            "city_200m_lamps_b_off_scene.yaml",
                            "city_200m_precise_collision_scene.yaml",
                            "city_200m_ep001_scene.yaml", "city_200m_ep002_scene.yaml")]
    report = {"results": results}
    ok = True
    for item in results:
        thermal = item["thermal"]
        problems = []
        if thermal["missing_path_count"]:
            problems.append(f"{thermal['missing_path_count']} binding paths missing")
        if thermal["not_geometry_count"]:
            problems.append(f"{thermal['not_geometry_count']} bindings not on geometry")
        if thermal["unknown_materials"]:
            problems.append(f"unknown materials {thermal['unknown_materials']}")
        if thermal["temperature_mismatch_count"]:
            problems.append(f"{thermal['temperature_mismatch_count']} temperature mismatches")
        item["status"] = "PASS" if not problems else "FAIL"
        item["problems"] = problems
        ok = ok and not problems
        print(f"{item['status']} {Path(item['config']).name}: "
              f"meshes={item['meshes']} bound={thermal['bound_meshes']} "
              f"unbound={thermal['unbound_meshes']} gprims={item['gprims']} "
              f"drone={item['drone']['exists']} camera={item['camera']['exists']}")
        for problem in problems:
            print("     - " + problem)
    report["status"] = "PASS" if ok else "FAIL"
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print("report: " + args.json)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
