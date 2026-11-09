"""用一个场景配置走一遍本项目自己的加载器，确认"能被现有流程吃下去"。

不需要 Isaac，只需要本项目现有的 Python 环境（pyyaml/pydantic/numpy）。
它校验的是"流程约定"，不是画面：

  1. scene_config.load_scene_config    —— 路径、Prim 路径、显示配置
  2. thermal_config.load_thermal_config —— 目标/绑定语义
  3. irsim SceneConfig                  —— 热环境 YAML（含 weather 文件可解析）
  4. project_materials + MaterialTable  —— 场景要的每个红外材质都真的能加载

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\validate_project_scene.py --config D:\\IsaacSim_Demo\\configs\\forest_fire_scene.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
VENDOR = PROJECT / "third_party" / "isaac-thermal-camera-simulation"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(VENDOR / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT / "configs" / "forest_fire_scene.yaml"))
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from scripts.scene_config import load_scene_config
    from scripts.thermal_config import load_thermal_config
    from scripts.thermal_display import load_display_profile

    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    cfg = load_scene_config(args.config)
    add("scene_config.load", True, f"{cfg.source_path.name} sha256={cfg.sha256[:12]}")
    add("scene.stage_exists", cfg.scene.stage.is_file(), cfg.scene.stage)
    add("scene.thermal_scene_exists", cfg.scene.thermal_scene.is_file(), cfg.scene.thermal_scene)
    add("sensors.sensor_config_exists", cfg.sensors.sensor_config.is_file(), cfg.sensors.sensor_config)
    add("sensors.thermal_config_exists", cfg.sensors.thermal_config.is_file(), cfg.sensors.thermal_config)
    add("sensors.material_directory", bool(cfg.sensors.material_directory), cfg.sensors.material_directory)

    if cfg.sensors.display_config:
        profile = load_display_profile(cfg.sensors.display_config)
        add("display.profile", True, f"{cfg.sensors.display_config.name} "
                                     f"[{profile.temperature_min_c}, {profile.temperature_max_c}] C")

    thermal = load_thermal_config(cfg.sensors.thermal_config)
    add("thermal_config.targets", len(thermal.targets) > 0,
        ", ".join(f"{t.name}={t.temperature_k:.2f}K" if t.temperature_k else t.name for t in thermal.targets))
    add("thermal_config.bindings", len(thermal.bindings) > 0, f"{len(thermal.bindings)} bindings")

    # ---- 热环境 YAML ------------------------------------------------------
    import yaml
    from irsim.config.scene import SceneConfig
    from irsim.scene import Scene  # noqa: F401  (import side effects: pydantic schema load)

    env_raw = yaml.safe_load(cfg.scene.thermal_scene.read_text(encoding="utf-8"))
    env = SceneConfig.model_validate(env_raw)
    weather = Path(env.scene.weather_file)
    if not weather.is_absolute():
        weather = (cfg.scene.thermal_scene.parent / weather).resolve()
    add("thermal_scene.schema", True, f"schema_version={env_raw.get('schema_version')}")
    add("thermal_scene.weather_file", weather.is_file(), weather)

    # ---- 材质：场景要的每一个都得能加载 --------------------------------------
    from irsim.materials.table import MaterialTable
    from scripts.project_materials import load_project_material_library

    library = load_project_material_library(cfg.sensors.material_directory)
    from irsim.config.loader import load_sensor_config

    sensor = load_sensor_config(cfg.sensors.sensor_config)
    table = MaterialTable.from_library(library, sensor.sensor.band.band_id)
    materials = set(table.names)
    wanted = {thermal.default_material} | {b.material for b in thermal.bindings}
    missing = sorted(wanted - materials)
    add("materials.available", not missing, f"missing={missing}" if missing else f"{len(wanted)} materials ok")
    add("materials.wood_pine_loaded", "wood_pine" in materials, "custom directory merged")

    failed = [c for c in checks if not c["ok"]]
    for c in checks:
        print(f"[{'ok  ' if c['ok'] else 'FAIL'}] {c['check']:34s} {c['detail']}")
    print(f"\nSTATUS: {'PASS' if not failed else 'FAIL'}  ({len(failed)} failure(s))")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": str(cfg.source_path), "checks": checks,
                                   "status": "PASS" if not failed else "FAIL"}, indent=2),
                       encoding="utf-8")
        print(f"json: {out}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
