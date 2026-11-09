"""场景包接入：校验完整性、扫描结构、和上一包做差异。

只读扫描，绝不修改被接入的包。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\ingest.py <包目录> [--probe probe.json]

输出：
    scene_intake/out/<包名>/report.json    本次报告
    scene_intake/out/<包名>/delta.md       与上一次接入的差异（首次为空）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml

REQUIRED = ["scene.usda", "configs", "environment", "assets"]
ASSET_REF = re.compile(r"@([^@\r\n]+)@")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recover_name(name: str) -> str:
    """SHA256SUMS 里可能有被 GBK 误解码的中文名，还原后好找文件。"""
    try:
        return name.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def check_manifest(pkg: Path) -> dict:
    manifest_path = pkg / "SHA256SUMS.json"
    if not manifest_path.is_file():
        return {"present": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ok, bad, missing, renamed = 0, [], [], []
    for name, digest in manifest.items():
        target = pkg / name
        if not target.is_file():
            alt = recover_name(name)
            if alt != name and (pkg / alt).is_file():
                renamed.append({"manifest": name, "actual": alt})
                target = pkg / alt
            else:
                missing.append(name)
                continue
        if sha256(target) == digest:
            ok += 1
        else:
            bad.append(name)
    return {
        "present": True,
        "declared": len(manifest),
        "verified": ok,
        "mismatched": bad,
        "missing": missing,
        "name_encoding_issues": renamed,
    }


def scan_assets(pkg: Path) -> dict:
    """解析 @...@ 资产引用。

    USD 的资产路径按"所在层"解析；但包里的 records/ 备份层是从别处拷来的，
    引用要按包根才找得到。所以逐级尝试，并区分"真缺"和"归档层"。
    """
    refs, missing, resolved_elsewhere, unparsed = [], [], [], []
    for layer in pkg.rglob("*.usda"):
        text = layer.read_text(encoding="utf-8", errors="replace")
        for ref in ASSET_REF.findall(text):
            if ref.startswith(("http://", "https://", "omniverse:", "file:")):
                continue
            rel_layer = layer.relative_to(pkg).as_posix()
            candidates = [layer.parent] + [p for p in layer.parents if pkg in p.parents or p == pkg]
            hit = next((c / ref for c in candidates if (c / ref).is_file()), None)
            refs.append({"layer": rel_layer, "asset": ref})
            if hit is None:
                missing.append({"layer": rel_layer, "asset": ref,
                                "archival": rel_layer.startswith("records/")})
            elif hit.parent != layer.parent:
                resolved_elsewhere.append({"layer": rel_layer, "asset": ref})
    for layer in pkg.rglob("*.usdc"):
        unparsed.append(layer.relative_to(pkg).as_posix())
    hard_missing = [m for m in missing if not m["archival"]]
    return {
        "usda_refs": len(refs),
        "missing": hard_missing,
        "archival_missing": [m for m in missing if m["archival"]],
        "resolved_against_package_root": resolved_elsewhere,
        "binary_layers": unparsed,
    }


def read_yaml(path: Path):
    if not path.is_file():
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"__parse_error__": True}


def scan_configs(pkg: Path) -> dict:
    scene = read_yaml(pkg / "configs" / "scene.yaml") or {}
    thermal = read_yaml(pkg / "configs" / "thermal.yaml") or {}
    display = read_yaml(pkg / "configs" / "display.yaml") or {}
    env = read_yaml(pkg / "environment" / "night_environment.yaml") or {}
    weather = (pkg / "environment" / "weather.csv")
    rows = []
    if weather.is_file():
        rows = [ln for ln in weather.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    bindings = thermal.get("bindings") or []
    sensors = scene.get("sensors") or {}
    scene_section = scene.get("scene") or {}

    def resolve(rel: str):
        if not rel:
            return None
        p = (pkg / "configs" / rel).resolve()
        return {"path": str(p), "exists": p.exists()}

    stage_entry = resolve(scene_section.get("stage", ""))
    return {
        "stage": scene_section.get("stage"),
        "stage_resolved": stage_entry,
        "stage_exists": bool(stage_entry and stage_entry["exists"]),
        "sensor_config": sensors.get("sensor_config"),
        "sensor_config_resolved": resolve(sensors.get("sensor_config", "")),
        "material_directory_resolved": resolve(sensors.get("material_directory", "")),
        "display_range_c": [display.get("temperature_min_c"), display.get("temperature_max_c")],
        "thermal_targets": {k: v.get("temperature_c") for k, v in (thermal.get("targets") or {}).items()},
        "thermal_bindings": len(bindings),
        "thermal_materials": sorted({b.get("material") for b in bindings if b.get("material")}),
        "environment_site": (env.get("scene") or {}).get("site"),
        "weather_rows": max(0, len(rows) - 2),
    }


def scan_records(pkg: Path) -> dict:
    out = {}
    for name in ("validation.json", "fire_validation.json", "portability_validation.json", "build_report.json"):
        path = pkg / "records" / name
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out[name] = {k: data[k] for k in list(data)[:8]}
            except Exception:  # noqa: BLE001
                out[name] = "unreadable"
    return out


def file_inventory(pkg: Path) -> dict:
    files = [p for p in pkg.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return {
        "files": len(files),
        "bytes": sum(p.stat().st_size for p in files),
        "top_level": sorted({p.relative_to(pkg).parts[0] for p in files}),
        "list": sorted(p.relative_to(pkg).as_posix() for p in files),
    }


def build_report(pkg: Path, probe: dict | None) -> dict:
    inventory = file_inventory(pkg)
    report = {
        "package": pkg.name,
        "path": str(pkg),
        "required_present": {k: (pkg / k).exists() for k in REQUIRED},
        "inventory": {k: inventory[k] for k in ("files", "bytes", "top_level")},
        "manifest": check_manifest(pkg),
        "assets": scan_assets(pkg),
        "configs": scan_configs(pkg),
        "records": scan_records(pkg),
        "probe": probe or {},
        "_file_list": inventory["list"],
    }
    blockers, warnings = [], []
    if not all(report["required_present"].values()):
        blockers.append("required_paths_missing: " + ", ".join(k for k, v in report["required_present"].items() if not v))
    if report["manifest"].get("present") and report["manifest"].get("mismatched"):
        blockers.append(f"sha256_mismatch: {len(report['manifest']['mismatched'])}")
    if report["assets"]["missing"]:
        blockers.append(f"missing_assets: {len(report['assets']['missing'])}")
    if report["assets"]["archival_missing"]:
        warnings.append(f"archival_layer_refs: {len(report['assets']['archival_missing'])} (records/ 备份层，不算缺失)")
    cfg = report["configs"]
    for key in ("sensor_config_resolved", "material_directory_resolved"):
        entry = cfg.get(key)
        if entry and not entry["exists"]:
            warnings.append(f"{key} 指向不存在的路径: {entry['path']}")
    if cfg.get("stage") and not cfg.get("stage_exists"):
        blockers.append(f"stage_missing: {cfg['stage']}")
    if probe and probe.get("status") == "FAIL":
        blockers.append("probe_failed")
    report["warnings"] = warnings
    report["blockers"] = blockers
    report["status"] = "PASS" if not blockers else "FAIL"
    return report


def delta_markdown(prev: dict, cur: dict) -> str:
    if not prev:
        return "# 首次接入\n\n没有可比的上一次，报告见 `report.json`。\n"
    lines = ["# 与上次接入的差异", ""]
    a, b = prev.get("inventory", {}), cur.get("inventory", {})
    lines += [
        "| 指标 | 上次 | 本次 |",
        "|---|---:|---:|",
        f"| 文件数 | {a.get('files')} | {b.get('files')} |",
        f"| 体积 | {a.get('bytes', 0)/1048576:.2f} MB | {b.get('bytes', 0)/1048576:.2f} MB |",
    ]
    pa, pb = prev.get("configs", {}), cur.get("configs", {})
    lines += [
        f"| 热绑定 | {pa.get('thermal_bindings')} | {pb.get('thermal_bindings')} |",
        "",
    ]
    old_files, new_files = set(prev.get("_file_list") or []), set(cur.get("_file_list") or [])
    added, removed = sorted(new_files - old_files), sorted(old_files - new_files)
    lines.append(f"新增文件 {len(added)}，删除文件 {len(removed)}")
    for p in added[:15]:
        lines.append(f"- + {p}")
    for p in removed[:15]:
        lines.append(f"- − {p}")

    old_t, new_t = pa.get("thermal_targets") or {}, pb.get("thermal_targets") or {}
    changed = {k: (old_t.get(k), new_t.get(k)) for k in set(old_t) | set(new_t) if old_t.get(k) != new_t.get(k)}
    if changed:
        lines += ["", "温度改动："]
        for k, (x, y) in sorted(changed.items()):
            lines.append(f"- {k}: {x} → {y}")
    if pa.get("display_range_c") != pb.get("display_range_c"):
        lines.append(f"- 显示范围: {pa.get('display_range_c')} → {pb.get('display_range_c')}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("package")
    ap.add_argument("--probe", default="", help="probe_scene.py 输出的 JSON，可选")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "out"))
    args = ap.parse_args()

    pkg = Path(args.package).resolve()
    if not pkg.is_dir():
        raise SystemExit(f"not a directory: {pkg}")
    probe = json.loads(Path(args.probe).read_text(encoding="utf-8")) if args.probe else None

    out_dir = Path(args.out) / pkg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    prev = None
    history = sorted(out_dir.glob("report-*.json"))
    if history:
        prev = json.loads(history[-1].read_text(encoding="utf-8"))

    report = build_report(pkg, probe)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "delta.md").write_text(delta_markdown(prev, report), encoding="utf-8")

    stamp = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    from datetime import datetime

    (out_dir / f"report-{datetime.now().strftime('%Y%m%dT%H%M%S')}.json").write_text(
        json.dumps(stamp, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    inv, cfg = report["inventory"], report["configs"]
    print(f"package : {report['package']}")
    print(f"status  : {report['status']}")
    print(f"files   : {inv['files']}  ({inv['bytes']/1048576:.1f} MB)")
    man = report["manifest"]
    if man.get("present"):
        print(f"manifest: {man['verified']}/{man['declared']} verified, "
              f"{len(man['mismatched'])} mismatch, {len(man['missing'])} missing, "
              f"{len(man['name_encoding_issues'])} name-encoding")
    print(f"assets  : {report['assets']['usda_refs']} refs, {len(report['assets']['missing'])} missing")
    print(f"thermal : {cfg['thermal_bindings']} bindings, materials {', '.join(cfg['thermal_materials'] or [])}")
    for b in report["blockers"]:
        print(f"BLOCKER : {b}")
    print(f"report  : {out_dir / 'report.json'}")
    print(f"delta   : {out_dir / 'delta.md'}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
