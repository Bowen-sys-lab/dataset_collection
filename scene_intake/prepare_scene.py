"""把一个外部场景包变成"本机可直接用"的工作副本，并把重复劳动一次做完。

以前每来一个包都要手工做这几件事，现在一条命令：

  1. 复制出一份工作副本（原包只读不动）
  2. 修 configs/scene.yaml 里指向作者机器的路径（../../external/irsim）
  3. 检查本地 irsim 缺哪些红外材质，并报告项目自备材质目录的覆盖情况
  4. 检查 LUT 缓存是否与 pin 版本匹配，不匹配就挪走以便自动重算
  5. 生成 records/inspection_camera.json（作者的 --saved-view 需要）
  6. 跑 USD 级探针，出结构/热绑定/相机/动画的报告

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\prepare_scene.py ^
        --package D:\\IsaacSim_Demo\\forest_fire_scene_20261026\\forest_fire_scene_20261026 ^
        --work    D:\\IsaacSim_Demo\\forest_fire_scene_work ^
        --irsim   D:\\IsaacSim_Demo\\third_party\\irsim-15015cd
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import yaml

PROJECT = Path(r"D:\IsaacSim_Demo")
PYTHON_BAT = Path(r"D:\isaacsim\python.bat")
DEFAULT_IRSIM = PROJECT / "third_party" / "irsim-15015cd"
EXTRA_MATERIALS = PROJECT / "configs" / "materials" / "forest_fire"
THERMAL_CONFIG = PROJECT / "configs" / "forest_fire_thermal.yaml"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_package(src: Path, dst: Path) -> dict:
    if dst.exists():
        raise SystemExit(f"working copy already exists: {dst} (choose another --work)")
    shutil.copytree(src, dst)
    files = [p for p in dst.rglob("*") if p.is_file()]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def fix_irsim_paths(work: Path, irsim: Path) -> dict:
    """把作者机器上的 ../../external/irsim 指到本机 checkout。"""
    cfg = work / "configs" / "scene.yaml"
    text = cfg.read_text(encoding="utf-8")
    rel = Path("..", "..", "third_party", irsim.name).as_posix()
    new = re.sub(r"\.\./\.\./external/irsim", rel, text)
    if new != text:
        cfg.write_text(new, encoding="utf-8")
    resolved = {}
    data = yaml.safe_load(new)
    for key in ("sensor_config", "material_directory"):
        value = (data.get("sensors") or {}).get(key)
        if value:
            path = (work / "configs" / value).resolve()
            resolved[key] = {"path": str(path), "exists": path.exists()}
    return {"rewritten": new != text, "resolved": resolved}


def material_coverage(work: Path, irsim: Path) -> dict:
    """场景要的每个红外材质，在 irsim 或项目自备目录里是否都有。"""
    thermal = yaml.safe_load(THERMAL_CONFIG.read_text(encoding="utf-8"))
    used = sorted({b.get("material") for b in thermal.get("bindings", []) if b.get("material")})
    in_irsim = {p.stem for p in (irsim / "configs" / "materials").glob("*.yaml")}
    local_dir = work / "configs" / "materials" / "local"
    local_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(EXTRA_MATERIALS.glob("*.yaml")):
        shutil.copy2(src, local_dir / src.name)
    in_local = {p.stem for p in local_dir.glob("*.yaml")}
    missing = [m for m in used if m not in in_irsim and m not in in_local]
    return {"used": used, "in_irsim": sorted(m for m in used if m in in_irsim),
            "from_project": sorted(m for m in used if m not in in_irsim and m in in_local),
            "missing": missing, "project_material_dir": str(local_dir)}


def lut_status(work: Path, irsim: Path, sensor_config: Path) -> dict:
    """用 irsim 自己的 band_hash 判断缓存是否属于当前 pin 版本。"""
    import sys

    sys.path.insert(0, str(irsim / "src"))
    from irsim.config.loader import band_hash, load_sensor_config

    sensor = load_sensor_config(sensor_config, data_dir=irsim / "data")
    digest = band_hash(sensor, data_dir=irsim / "data")
    cache = work / "cache" / "luts"
    hit = sorted(cache.glob(f"{digest[:12]}*.json")) if cache.is_dir() else []
    moved = 0
    if cache.is_dir() and not hit:
        stale_dir = work / "cache" / "luts_stale"
        stale_dir.mkdir(exist_ok=True)
        for path in cache.iterdir():
            if path.is_file():
                shutil.move(str(path), str(stale_dir / path.name))
                moved += 1
    return {"band_hash": digest[:12], "matching_bundle": bool(hit), "moved_stale": moved}


def run_probe(stage: Path, extra: list[str]) -> dict:
    out_json = PROJECT / "scene_intake" / "out" / f"{stage.parent.name}_probe.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(PYTHON_BAT), "-X", "utf8", str(PROJECT / "scene_intake" / "probe_scene.py"),
           "--stage", str(stage), "--json", str(out_json), *extra]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {"returncode": result.returncode,
            "report": str(out_json) if out_json.is_file() else None,
            "stderr_tail": (result.stderr or "")[-400:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--irsim", default=str(DEFAULT_IRSIM))
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    package = Path(args.package).resolve()
    work = Path(args.work).resolve()
    irsim = Path(args.irsim).resolve()
    if not (package / "scene.usda").is_file():
        raise SystemExit(f"not a scene package (no scene.usda): {package}")
    if not (irsim / "src" / "irsim").is_dir():
        raise SystemExit(f"irsim checkout not found: {irsim}")

    report = {"package": str(package), "work": str(work), "irsim": str(irsim),
              "package_scene_sha256": sha256(package / "scene.usda")}
    report["copy"] = copy_package(package, work)
    report["paths"] = fix_irsim_paths(work, irsim)
    report["materials"] = material_coverage(work, irsim)

    sensor = Path(report["paths"]["resolved"]["sensor_config"]["path"])
    report["lut"] = lut_status(work, irsim, sensor)

    if not args.skip_probe:
        camera_json = work / "records" / "inspection_camera.json"
        report["inspection_camera"] = run_probe(
            work / "scene.usda", ["--write-inspection-camera", str(camera_json)])
        report["inspection_camera"]["written"] = camera_json.is_file()
        report["probe"] = run_probe(work / "scene.usda",
                                    ["--thermal-config", str(THERMAL_CONFIG)])

    out = Path(args.json or (PROJECT / "scene_intake" / "out" / f"{work.name}_prepare.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"PREPARE work={work}")
    print(f"  copy        : files={report['copy']['files']} ({report['copy']['bytes']/1048576:.1f} MB)")
    print(f"  paths       : rewritten={report['paths']['rewritten']} "
          f"sensor_exists={report['paths']['resolved']['sensor_config']['exists']}")
    mats = report["materials"]
    print(f"  materials   : used={mats['used']}")
    print(f"                from_irsim={mats['in_irsim']} from_project={mats['from_project']} missing={mats['missing']}")
    lut = report["lut"]
    print(f"  lut         : band_hash={lut['band_hash']} matching={lut['matching_bundle']} "
          f"moved_stale={lut['moved_stale']}")
    if "probe" in report:
        print(f"  probe       : rc={report['probe']['returncode']} report={report['probe']['report']}")
        print(f"  inspection  : written={report['inspection_camera']['written']}")
    print(f"  report      : {out}")
    return 0 if not mats["missing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
