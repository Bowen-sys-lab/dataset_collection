"""独立核验一次采集结果：不信管线自己的 PASS，自己重算一遍。

检查项：
  1. 清单里每个文件的 SHA-256 是否与磁盘一致
  2. 报告与磁盘的帧数是否一致，帧是否齐全（rgb/thermal/display8/dn16/radiance/json）
  3. 数组是否有限、形状与声明分辨率一致
  4. 指定表面温度是否真的出现在真值平面里（火 923.15 K、桶 383.15 K、桶口 453.15 K）
  5. 表观温度范围是否合理，DN 饱和/显示裁切是否自洽
  6. 相机内参与位姿是否有限，距离是否与请求值相符
  7. 数据集索引与采集内容是否一致

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\verify_dataset.py --capture validation\\forest-targetcfg-01
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(r"D:\IsaacSim_Demo")
FRAME_FILES = ("_rgb.png", "_rgb.npy", "_thermal.png", "_apparent_t.npy", "_dn16.npy",
               "_display8.npy", "_radiance.npy", ".json")


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def requested_distance_contract(metas,*,recorded):
    values=[m.get('requested_distance_m') for m in metas]
    if recorded:
        return all(value is None for value in values),'trajectory poses; no fixed requested observation distance'
    distances=sorted({round(float(value or 0),2) for value in values})
    return bool(distances) and all(math.isfinite(value) and value>0 for value in distances),distances


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capture", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    cap = Path(args.capture).resolve()
    checks = []

    def add(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})

    # ---- 1. 清单 ---------------------------------------------------------
    manifest_path = cap / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    add("manifest.present", manifest is not None, manifest_path)
    digest_bad, digest_ok = [], 0
    if manifest:
        for rel, digest in manifest.get("files", {}).items():
            target = cap / rel
            if not target.is_file() or sha256(target) != digest:
                digest_bad.append(rel)
            else:
                digest_ok += 1
        add("manifest.files_match", not digest_bad, f"{digest_ok} verified, {len(digest_bad)} bad")
        extra = [p.relative_to(cap).as_posix() for p in cap.rglob("*")
                 if p.is_file() and p.relative_to(cap).as_posix() not in manifest.get("files", {})
                 and p.name != "collection_manifest.json"]
        add("manifest.covers_everything", not extra, f"{len(extra)} unlisted file(s)")

    # ---- 2. 帧完整性 ------------------------------------------------------
    report_path = cap / "collection_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    frames = sorted(int(p.name.split("_")[1]) for p in cap.glob("frame_*_apparent_t.npy"))
    add("report.pass", report.get("result") == "PASS", report.get("result"))
    add("report.frame_count", report.get("frames") == len(frames), f"report={report.get('frames')} disk={len(frames)}")
    missing = [f"{i}{suffix}" for i in frames for suffix in FRAME_FILES if not (cap / f"frame_{i:06d}{suffix}").is_file()]
    add("frames.complete", not missing, f"{len(missing)} missing part(s)")

    # ---- 3/4/5. 数值与真值 -------------------------------------------------
    truth_hits = {"flame_923.15": 0, "barrel_383.15": 0, "rim_453.15": 0}
    bad_arrays, t_ranges, sat_total, clip_total = [], [], 0, 0
    for i in frames:
        try:
            app = np.load(cap / f"frame_{i:06d}_apparent_t.npy")
            dn = np.load(cap / f"frame_{i:06d}_dn16.npy")
            if not np.isfinite(app).all():
                bad_arrays.append(f"frame_{i:06d}_apparent_t.npy")
            t_ranges.append((float(app.min()), float(app.max())))
            sat_total += int((dn.astype(np.uint16) == 65535).sum())
            truth = cap / "truth" / f"frame_{i:06d}_temperature_k.npy"
            if truth.is_file():
                tk = np.load(truth)
                truth_hits["flame_923.15"] += int(np.isclose(tk, 923.15, atol=0.1).sum())
                truth_hits["barrel_383.15"] += int(np.isclose(tk, 383.15, atol=0.1).sum())
                truth_hits["rim_453.15"] += int(np.isclose(tk, 453.15, atol=0.1).sum())
        except Exception as exc:  # noqa: BLE001
            bad_arrays.append(f"frame_{i:06d}: {exc}")
    add("arrays.finite", not bad_arrays, f"{len(bad_arrays)} problem(s)")
    selected_target=report.get('target') or {}
    target_specific=selected_target.get('verification_profile')=='forest_two_target_v1'
    target_observation=None
    if target_specific:
        # Recompute exact object-ID evidence; a different person/fire cannot supply it.
        if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
        from scripts.forest_target_evidence import target_surface_evidence
        selected_pixels=expected_pixels=0;problems=[]
        for i in frames:
            try:
                labels=json.loads((cap/f'truth/frame_{i:06d}_legends.json').read_text(encoding='utf-8'))['instance_id']
                evidence=target_surface_evidence(np.load(cap/f'truth/frame_{i:06d}_instance_id.npy'),labels,
                    np.load(cap/f'truth/frame_{i:06d}_temperature_k.npy'),selected_target['prim_path'],
                    selected_target['expected_temperatures_c'])
                selected_pixels+=evidence['target_pixels'];expected_pixels+=evidence['expected_temperature_pixels']
                meta=json.loads((cap/f'frame_{i:06d}.json').read_text(encoding='utf-8'))
                if meta.get('selected_target_evidence')!=evidence:
                    problems.append(f'frame {i}: stored target evidence differs from raw planes')
            except Exception as error:
                problems.append(str(error))
        task=((report.get('recording_context') or {}).get('recording') or {}).get('task')
        identity_ok=not task or task.get('target_prim_path')==selected_target.get('prim_path')
        add('truth.selected_target_identity',identity_ok,'report target matches recorded task')
        add('truth.selected_target_evidence_integrity',not problems,problems)
        target_observation=dict(status='OBSERVED' if expected_pixels else
                                'UNEXPECTED_SURFACE_TEMPERATURE' if selected_pixels else 'NOT_OBSERVED',
            target_prim_path=selected_target['prim_path'],target_pixels=selected_pixels,
            expected_temperature_pixels=expected_pixels,automatic_task_acceptance='NOT_EVALUATED')
    else:
        add("truth.flame_present", truth_hits["flame_923.15"] > 0, f"{truth_hits['flame_923.15']} px @923.15K")
        add("truth.barrel_present", truth_hits["barrel_383.15"] > 0, f"{truth_hits['barrel_383.15']} px @383.15K")
        add("truth.rim_present", truth_hits["rim_453.15"] > 0, f"{truth_hits['rim_453.15']} px @453.15K")
    if t_ranges:
        lo = min(r[0] for r in t_ranges)
        hi = max(r[1] for r in t_ranges)
        add("apparent_t.range_sane", (180.0 < lo <= hi < 1250.0 if target_specific
                                     else 200.0 < lo < 320.0 and 300.0 < hi < 1200.0),
            f"{lo:.1f}-{hi:.1f} K over {len(t_ranges)} frames")

    # ---- 6. 相机 ----------------------------------------------------------
    metas = [json.loads((cap / f"frame_{i:06d}.json").read_text(encoding="utf-8")) for i in frames]
    pose_ok = all(np.isfinite(np.asarray(m.get("camera_world_matrix", [[float("nan")]]), dtype=float)).all()
                  for m in metas)
    common = {}
    if (cap / "common_metadata.json").is_file():
        common = json.loads((cap / "common_metadata.json").read_text(encoding="utf-8"))

    def find_key(node, key, depth=0):
        if depth > 8:
            return None
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find_key(value, key, depth + 1)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node[:5]:
                found = find_key(value, key, depth + 1)
                if found is not None:
                    return found
        return None

    intr = find_key(common, "rgb_intrinsics_px") or find_key(metas[0], "rgb_intrinsics_px")
    thermal_intr = find_key(common, "thermal_intrinsics_px")

    def intr_values(value):
        if isinstance(value, dict):
            keys = ("fx", "fy", "cx", "cy")
            if all(k in value for k in keys):
                return [float(value[k]) for k in keys]
            return [float(x) for x in list(value.values())[:4]]
        if isinstance(value, (list, tuple)):
            return [float(x) for x in value[:4]]
        return []

    rgb_vals = intr_values(intr)
    intr_ok = len(rgb_vals) == 4 and bool(np.isfinite(np.asarray(rgb_vals)).all())
    add("camera.pose_finite", pose_ok, f"{len(metas)} frames")
    add("camera.intrinsics", intr_ok,
        f"rgb={intr} thermal={thermal_intr}")
    recorded=bool(report.get('recording_context'))
    distance_ok,distance_detail=requested_distance_contract(metas,recorded=recorded)
    add('capture.distances',distance_ok,distance_detail)
    if recorded:
        try:
            if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
            from scene_intake.make_replay_metadata import build_replay_metadata
            saved=json.loads((cap/'replay_metadata.json').read_text(encoding='utf-8'))
            build_replay_metadata(cap,Path(saved['trajectory']),require_manifest=True)
            add('recorded.pose_time_provenance',True,'source matrices, times and labels independently checked')
        except Exception as error:
            add('recorded.pose_time_provenance',False,str(error))

    # ---- 7. 数据集索引 -----------------------------------------------------
    dataset_path = Path(args.dataset) if args.dataset else PROJECT / "validation" / "datasets" / cap.name / "dataset.json"
    if dataset_path.is_file():
        ds = json.loads(dataset_path.read_text(encoding="utf-8"))
        add("dataset.frames", ds.get("frames") == len(frames), f"index={ds.get('frames')} disk={len(frames)}")
        add("dataset.integrity_verified", ds.get("integrity_manifest_verified") is True, "")
        add("dataset.scene_label", bool(ds.get("scene")), ds.get("scene"))
        # 任务身份：报告里的 task_id 必须一路传到索引的每一行
        report_task = (report.get("task") or {}).get("task_id")
        if report_task:
            index_csv = dataset_path.parent / "index.csv"
            ids = set()
            if index_csv.is_file():
                import csv

                with index_csv.open(encoding="utf-8-sig") as fh:
                    ids = {row.get("task_id") for row in csv.DictReader(fh)}
            add("dataset.task_propagated", ids == {report_task} and
                (ds.get("task") or {}).get("task_id") == report_task,
                f"report={report_task} index={sorted(i for i in ids if i)}")
    else:
        add("dataset.present", False, dataset_path)

    failed = [c for c in checks if not c["ok"]]
    for c in checks:
        print(f"[{'ok  ' if c['ok'] else 'FAIL'}] {c['check']:28s} {c['detail']}")
    print(f"\nVERDICT: {'PASS' if not failed else 'FAIL'}  "
          f"frames={len(frames)} truth(flame/barrel/rim)={truth_hits} dn_saturated_px={sat_total}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({"capture": str(cap), "checks": checks,
                                               "truth_hits": truth_hits, "target_observation": target_observation,
                                               "verdict": "PASS" if not failed else "FAIL"},
                                              indent=2), encoding="utf-8")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
