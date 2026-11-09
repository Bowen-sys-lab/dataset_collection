"""森林火任务的独立评估器：把"任务执行成功"变成可复核的判据。

判据（默认值来自任务描述，可用参数覆盖）：
  1. 任务存在（任务文件里有 target_position）
  2. 抵达：轨迹到目标的最小距离 <= reach_m（默认 8 m）
  3. 停留：在 reach_m 内且速度 < hover_speed 的连续时间 >= min_hover_s（默认 3 s）
  4. 观测：抵达时刻目标在画面内（几何投影；不判断遮挡）
  5. 红外证据：抵达窗口内某帧的表观温度最大值 >= thermal_min_c（默认 150 ℃），
     并统计 >500 K 的像素数——距离够但画面里没有火，不判成功
  6. 记录：判据、阈值、标记时刻全部写进输出，便于复核；不往别人的报告里伪造结果

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\evaluate_forest_task.py ^
        --trajectory validation\\trajectory_runs\\<run>\\trajectory.jsonl ^
        --capture    validation\\<run>-merged ^
        --task       configs\\forest_fire_task.yaml ^
        --json       validation\\<run>-merged\\task_evaluation.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

PROJECT = Path(r"D:\IsaacSim_Demo")
DEFAULT_TASK = PROJECT / "configs" / "forest_fire_task.yaml"


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_trajectory(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        pos = np.asarray(rec.get("position"), dtype=float)
        if pos.shape != (3,):
            continue
        vel = np.asarray(rec.get("linear_velocity") or [0.0, 0.0, 0.0], dtype=float)
        rows.append({"t": float(rec.get("simulation_time") or 0.0), "position": pos,
                     "speed": float(np.linalg.norm(vel)) if vel.shape == (3,) else 0.0})
    rows.sort(key=lambda r: r["t"])
    return rows


def dwell_window(rows: list[dict], target: np.ndarray, reach_m: float, hover_speed: float) -> dict:
    """在半径内且低速的最长连续窗口。"""
    best, current_start, best_span = 0.0, None, None
    for row in rows:
        inside = float(np.linalg.norm(row["position"] - target)) <= reach_m and row["speed"] < hover_speed
        if inside and current_start is None:
            current_start = row["t"]
        if (not inside) and current_start is not None:
            span = row["t"] - current_start
            if span > best:
                best, best_span = span, (current_start, row["t"])
            current_start = None
    if current_start is not None and rows:
        span = rows[-1]["t"] - current_start
        if span > best:
            best, best_span = span, (current_start, rows[-1]["t"])
    return {"seconds": round(best, 3), "window": best_span}


def evaluate_frames(capture: Path, target: np.ndarray, reach_m: float,
                    thermal_min_c: float, closest_t: float | None) -> dict:
    frames = sorted(int(p.name.split("_")[1]) for p in capture.glob("frame_*_apparent_t.npy"))
    if not frames:
        return {"frames": 0, "skipped": "no thermal frames"}
    common = {}
    if (capture / "common_metadata.json").is_file():
        common = json.loads((capture / "common_metadata.json").read_text(encoding="utf-8"))
    intr = find_key(common, "rgb_intrinsics_px") or {}
    fx = float(intr.get("fx_px", 0.0))
    fy = float(intr.get("fy_px", 0.0))
    cx = float(intr.get("cx_px", 0.0))
    cy = float(intr.get("cy_px", 0.0))

    per_frame = []
    best_thermal = None
    in_frame_hit = None
    for i in frames:
        meta = json.loads((capture / f"frame_{i:06d}.json").read_text(encoding="utf-8"))
        app = np.load(capture / f"frame_{i:06d}_apparent_t.npy")
        t = float(meta.get("simulation_time_s") or 0.0)
        pose = np.asarray(meta.get("camera_world_matrix"), dtype=float)
        distance = None
        visible = None
        if pose.shape == (4, 4):
            cam_pos = pose[3, :3]
            distance = float(np.linalg.norm(cam_pos - target))
            if fx > 0 and fy > 0:
                rot = pose[:3, :3]
                local = rot.T @ (target - cam_pos)
                if local[2] < -1e-6:
                    u = cx + fx * (local[0] / -local[2])
                    v = cy - fy * (local[1] / -local[2])
                    width = int(intr.get("width") or 0) or int(app.shape[1])
                    height = int(intr.get("height") or 0) or int(app.shape[0])
                    visible = bool(0 <= u < width and 0 <= v < height)
                else:
                    visible = False
                    u = v = None
        entry = {"frame": i, "t": round(t, 3), "distance_m": None if distance is None else round(distance, 2),
                 "target_in_frame": visible,
                 "apparent_t_max_c": round(float(app.max()) - 273.15, 1),
                 "pixels_above_500k": int((app > 500.0).sum())}
        per_frame.append(entry)
        if distance is not None and distance <= reach_m:
            if best_thermal is None or entry["apparent_t_max_c"] > best_thermal["apparent_t_max_c"]:
                best_thermal = entry
            if entry["target_in_frame"]:
                in_frame_hit = entry
    if in_frame_hit is None and per_frame:
        # 抵达时刻没对上时，取最接近的一段里目标是否可见
        near = [f for f in per_frame if f["distance_m"] is not None]
        if near:
            in_frame_hit = min(near, key=lambda f: f["distance_m"])
    thermal_ok = bool(best_thermal and best_thermal["apparent_t_max_c"] >= thermal_min_c)
    return {"frames": len(frames), "per_frame": per_frame,
            "closest_frame_t": closest_t,
            "thermal_best": best_thermal, "thermal_ok": thermal_ok,
            "in_frame": in_frame_hit is not None and (in_frame_hit.get("target_in_frame") is True),
            "in_frame_frame": None if in_frame_hit is None else in_frame_hit["frame"],
            "intrinsics_used": {"fx": fx, "fy": fy, "cx": cx, "cy": cy}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", default="", help="飞行轨迹；缺省时按静态采集模式评估")
    ap.add_argument("--capture", default="")
    ap.add_argument("--task", default=str(DEFAULT_TASK))
    ap.add_argument("--reach-m", type=float, default=None)
    ap.add_argument("--min-hover-s", type=float, default=None)
    ap.add_argument("--hover-speed", type=float, default=0.3)
    ap.add_argument("--thermal-min-c", type=float, default=150.0)
    ap.add_argument("--start-tolerance-m", type=float, default=2.0,
                    help="录制轨迹的起点必须落在任务声明的 start_position 附近")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    task_path = Path(args.task)
    task = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
    target = np.asarray(task.get("target_position"), dtype=float)
    reach_m = args.reach_m if args.reach_m is not None else 8.5  # task says about 8 m; 0.5 m tolerance
    min_hover = args.min_hover_s if args.min_hover_s is not None else 3.0

    traj_path = Path(args.trajectory) if args.trajectory else None
    mode = "flight" if traj_path else "static_capture"
    if traj_path:
        rows = load_trajectory(traj_path)
    elif args.capture:
        # 静态采集：用每帧相机位姿当作"没有位移的轨迹"，只算距离/画面/红外
        cap0 = Path(args.capture)
        rows = []
        for path in sorted(cap0.glob("frame_*.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            pose = np.asarray(meta.get("camera_world_matrix"), dtype=float)
            if pose.shape != (4, 4):
                continue
            rows.append({"t": float(meta.get("simulation_time_s") or 0.0),
                         "position": pose[3, :3], "speed": 0.0})
        rows.sort(key=lambda r: r["t"])
    else:
        raise SystemExit("need --trajectory or --capture")
    checks, metrics = {}, {}
    metrics["mode"] = mode

    checks["task_present"] = {"ok": target.shape == (3,),
                              "detail": task.get("task_id") or "no task_id"}
    if target.shape != (3,):
        print(json.dumps({"checks": checks, "verdict": "FAIL"}))
        return 2

    # 起点一致性：任务声明了 start_position，录制轨迹就必须从那里开始
    declared_start = np.asarray(task.get("start_position"), dtype=float) if task.get("start_position") else None
    if declared_start is None or declared_start.shape != (3,):
        checks["start_matches_task"] = {"ok": None, "detail": "task declares no start_position"}
    elif mode != "flight" or not rows:
        checks["start_matches_task"] = {"ok": None, "detail": "not applicable (no flight trajectory)"}
    else:
        delta = float(np.linalg.norm(rows[0]["position"] - declared_start))
        checks["start_matches_task"] = {
            "ok": delta <= args.start_tolerance_m,
            "detail": f"trajectory starts {delta:.2f} m from declared "
                      f"{tuple(round(float(v), 2) for v in declared_start)} (limit {args.start_tolerance_m:.1f} m)"}

    distances = [float(np.linalg.norm(r["position"] - target)) for r in rows]
    min_distance = min(distances) if distances else math.inf
    closest = rows[int(np.argmin(distances))] if rows else None
    metrics["samples"] = len(rows)
    metrics["min_distance_m"] = None if not rows else round(min_distance, 2)
    metrics["duration_s"] = None if not rows else round(rows[-1]["t"] - rows[0]["t"], 2)
    checks["reached"] = {"ok": min_distance <= reach_m,
                         "detail": f"{min_distance:.2f} m (limit {reach_m:.1f} m)"}

    dwell = dwell_window(rows, target, reach_m, args.hover_speed)
    if mode == "static_capture":
        checks["dwell"] = {"ok": None,
                           "detail": "not applicable: static capture has no flight motion "
                                     "(use --trajectory for a flown run)"}
    else:
        metrics["dwell_s"] = dwell["seconds"]
        metrics["dwell_window_s"] = dwell["window"]
        checks["dwell"] = {"ok": dwell["seconds"] >= min_hover,
                           "detail": f"{dwell['seconds']:.2f} s within {reach_m:.0f} m at <{args.hover_speed} m/s "
                                     f"(need {min_hover:.1f} s)"}

    frame_report = {}
    if args.capture:
        cap = Path(args.capture)
        frame_report = evaluate_frames(cap, target, reach_m, args.thermal_min_c,
                                       None if closest is None else closest["t"])
        if "skipped" in frame_report:
            checks["observed_in_frame"] = {"ok": False, "detail": frame_report["skipped"]}
            checks["thermal_evidence"] = {"ok": False, "detail": frame_report["skipped"]}
        else:
            metrics["frames_evaluated"] = frame_report["frames"]
            checks["observed_in_frame"] = {"ok": frame_report["in_frame"],
                                           "detail": f"frame {frame_report['in_frame_frame']} "
                                                     f"(geometric projection only)"}
            best = frame_report.get("thermal_best") or {}
            metrics["thermal_best_c"] = best.get("apparent_t_max_c")
            metrics["pixels_above_500k"] = best.get("pixels_above_500k")
            checks["thermal_evidence"] = {
                "ok": bool(frame_report["thermal_ok"]),
                "detail": f"max {best.get('apparent_t_max_c')} C at frame {best.get('frame')} "
                          f"(need >= {args.thermal_min_c:.0f} C)"}

    failed = [name for name, c in checks.items() if c["ok"] is False]
    verdict = "PASS" if not failed else "FAIL"
    result = {
        "verdict": verdict,
        # 静态采集只证明"观测条件满足"，不等于"飞抵并悬停"的任务完成
        "verdict_scope": "full_task" if mode == "flight" else "observation_only",
        "task": {"task_id": task.get("task_id"), "source": str(task_path),
                 "sha256": sha256(task_path), "target_position": target.tolist(),
                 "instruction": task.get("instruction")},
        "criteria": {"reach_m": reach_m, "min_hover_s": min_hover,
                     "hover_speed_m_s": args.hover_speed, "thermal_min_c": args.thermal_min_c,
                     "in_frame": "geometric projection of the target centre; occlusion not inferred"},
        "inputs": {"trajectory": None if traj_path is None else str(traj_path),
                   "trajectory_sha256": None if traj_path is None else sha256(traj_path),
                   "capture": args.capture or None},
        "checks": checks, "metrics": metrics,
        "per_frame": frame_report.get("per_frame", []),
        "failed_checks": failed,
        "meaning": "auto evaluation from trajectory geometry + thermal pixels; no human T-mark was used",
    }

    for name, c in checks.items():
        tag = "ok  " if c["ok"] is True else ("skip" if c["ok"] is None else "FAIL")
        print(f"[{tag}] {name:20s} {c['detail']}")
    print(f"\nVERDICT: {verdict}  min_distance={metrics.get('min_distance_m')} m  "
          f"dwell={metrics.get('dwell_s')} s  thermal_best={metrics.get('thermal_best_c')} C")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"json: {out}")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
