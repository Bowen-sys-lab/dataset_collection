"""外部依赖自检：irsim 版本、API、材质覆盖、波段哈希与 LUT 缓存。

用法（需要 Isaac 的 Python，因为它带 pyyaml/numpy/pydantic）：
    & 'D:\\isaacsim\\python.bat' scene_intake\\check_irsim.py --scene D:\\IsaacSim_Demo\\forest_fire_scene_local

退出码：0 全部通过；2 有阻断项；1 脚本自身错误。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

DEFAULT_ROOT = Path(r"D:\IsaacSim_Demo\third_party\irsim-15015cd")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="irsim checkout 路径")
    ap.add_argument("--scene", required=True, help="场景包根目录（含 configs/ 与 scene.usda）")
    ap.add_argument("--json", default="", help="把结果写入该 JSON 文件")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    scene = Path(args.scene).resolve()
    checks: list[dict] = []

    def add(name: str, ok: bool, blocking: bool, detail="") -> None:
        checks.append({"check": name, "ok": bool(ok), "blocking": blocking, "detail": str(detail)})

    # ---- 1. checkout 本身 -------------------------------------------------
    add("irsim.root_exists", root.is_dir(), True, str(root))
    pyproject = root / "pyproject.toml"
    add("irsim.pyproject", pyproject.is_file(), True)
    add("irsim.license", (root / "LICENSE").is_file(), False,
        "use-only / non-commercial / no modification / no redistribution")
    add("irsim.src_package", (root / "src" / "irsim" / "__init__.py").is_file(), True)

    table_py = read_text(root / "src" / "irsim" / "materials" / "table.py")
    cam_py = read_text(root / "src" / "irsim_isaac" / "pipeline" / "ir_camera.py")
    api = {
        "MaterialTable.for_sensor": "def for_sensor" in table_py,
        "MaterialTable.from_library": "def from_library" in table_py,
        "IrCamera.companion_only_prim_paths": "companion_only_prim_paths" in cam_py,
        "IrCamera.truth()": "def truth(" in cam_py,
    }
    for name, ok in api.items():
        add("api." + name, ok, name in ("MaterialTable.for_sensor", "IrCamera.companion_only_prim_paths"), )

    # ---- 2. 场景配置 ------------------------------------------------------
    scene_yaml = scene / "configs" / "scene.yaml"
    thermal_yaml = scene / "configs" / "thermal.yaml"
    add("scene.configs.scene_yaml", scene_yaml.is_file(), True)
    add("scene.configs.thermal_yaml", thermal_yaml.is_file(), True)
    add("scene.stage", (scene / "scene.usda").is_file(), True)

    scene_cfg = yaml.safe_load(scene_yaml.read_text(encoding="utf-8")) if scene_yaml.is_file() else {}
    thermal_cfg = yaml.safe_load(thermal_yaml.read_text(encoding="utf-8")) if thermal_yaml.is_file() else {}

    sensor_rel = (scene_cfg.get("sensors") or {}).get("sensor_config", "")
    sensor_path = (scene / "configs" / sensor_rel).resolve() if sensor_rel else None
    add("scene.sensor_config", bool(sensor_path and sensor_path.is_file()), True, str(sensor_path))

    mat_rel = (scene_cfg.get("sensors") or {}).get("material_directory", "")
    mat_dir = (scene / "configs" / mat_rel).resolve() if mat_rel else None
    add("scene.material_directory", bool(mat_dir and mat_dir.is_dir()), True, str(mat_dir))

    # ---- 3. 材质覆盖（场景要的每一个材质都得在 checkout 里） -----------------
    used = sorted({b.get("material") for b in (thermal_cfg.get("bindings") or []) if b.get("material")})
    missing = [m for m in used if not (root / "configs" / "materials" / f"{m}.yaml").is_file()]
    add("materials.used", True, False, ", ".join(used))
    add("materials.missing", not missing, True, ", ".join(missing) if missing else "none")

    # ---- 4. 波段哈希与 LUT 缓存 -------------------------------------------
    band = {}
    if sensor_path and sensor_path.is_file():
        sys.path.insert(0, str(root / "src"))
        try:
            from irsim.config.loader import band_hash, load_sensor_config

            sensor = load_sensor_config(sensor_path, data_dir=root / "data")
            h = band_hash(sensor, data_dir=root / "data")
            band = {"band": str(sensor.sensor.band.band_id), "band_hash": h}
            add("band.hash", True, False, f"{band['band']} {h[:12]}")
            cache = scene / "cache" / "luts"
            hit = sorted(cache.glob(f"{h[:12]}*.json")) if cache.is_dir() else []
            add("luts.cache_matches_pin", bool(hit), False,
                str(hit[0].name) if hit else f"no cache for {h[:12]} in {cache} (rebuild ~10 s)")
        except Exception as exc:  # noqa: BLE001
            add("band.hash", False, True, f"{type(exc).__name__}: {exc}")

    blocking_failures = [c for c in checks if c["blocking"] and not c["ok"]]
    result = {
        "irsim_root": str(root),
        "scene": str(scene),
        "api": api,
        "materials": {"used": used, "missing": missing},
        "band": band,
        "checks": checks,
        "blocking_failures": [c["check"] for c in blocking_failures],
        "status": "PASS" if not blocking_failures else "FAIL",
    }

    for c in checks:
        mark = "ok  " if c["ok"] else ("FAIL" if c["blocking"] else "warn")
        print(f"[{mark}] {c['check']:38s} {c['detail']}")
    print(f"\nSTATUS: {result['status']}  ({len(blocking_failures)} blocking failure(s))")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"json: {out}")
    return 0 if not blocking_failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
