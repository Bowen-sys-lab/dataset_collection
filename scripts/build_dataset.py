"""Pack one replay run into a queryable dataset entry.

A replay directory is a flat pile of ~1000 files: per frame an RGB PNG, a thermal PNG,
five .npy planes and an 85-field JSON. Nothing links a frame to the task it serves, to the
inspection target, or to whether the episode passed, so answering "give me every frame
between 6 and 8 m from the hotspot with the target on screen" means writing a bespoke
parser every time.

This script emits instead:
  dataset.json      provenance (scene/config hashes, sensor, counts, versions)
  index.csv/.jsonl  one row per frame: pose, image paths, temperature stats, target geometry
  episodes.csv/.jsonl  one row per run: task, instruction, target truth, evaluation verdict

Parquet is not used because neither the Isaac interpreter nor the bundled one ships pyarrow;
CSV + JSONL load straight into pandas either way.

Run with the Isaac interpreter (it has numpy and yaml):
  python build_dataset.py --run  <trajectory run dir> \
                          --replay <replay output dir> \
                          --config <scene yaml> \
                          [--eval <factory_evaluation.json>] \
                          --out <dataset dir>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import yaml

# Camera geometry of the drone RGB camera, read from the stage's Camera prim. Used only to
# project the inspection target into the image; distances stay metric and need no intrinsics.
FOCAL_MM = 18.0
HAPERTURE_MM = 20.955
VAPERTURE_MM = 16.764
HOT_C = 60.0
VERY_HOT_C = 70.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def camera_basis(matrix: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotation, translation) for a pxr row-vector world matrix."""
    m = np.asarray(matrix, dtype=float)
    return m[:3, :3], m[3, :3]


def project_target(matrix: list[list[float]], target: np.ndarray,
                   width: int, height: int) -> tuple[float, float, bool, float]:
    """Pinhole-project a world point; return (u, v, in_view, depth_m)."""
    rot, tvec = camera_basis(matrix)
    cam = (target - tvec) @ rot.T          # world -> camera (row-vector convention)
    depth = -float(cam[2])                 # the camera looks down -Z
    if depth <= 1e-6:
        return float("nan"), float("nan"), False, depth
    fx = FOCAL_MM / HAPERTURE_MM * width
    fy = FOCAL_MM / VAPERTURE_MM * height
    u = fx * (cam[0] / depth) + width / 2.0
    v = height / 2.0 - fy * (cam[1] / depth)
    return u, v, (0.0 <= u < width and 0.0 <= v < height), depth


def collect_frames(replay: Path) -> list[dict]:
    rows = []
    for meta_path in sorted(replay.glob("frame_*.json")):
        meta = load_json(meta_path)
        stem = meta_path.with_suffix("")
        t_path = replay / f"{stem.name}_apparent_t.npy"
        stats = {}
        if t_path.exists():
            t = np.load(t_path).astype(np.float64) - 273.15
            stats = {
                "t_min_c": float(t.min()),
                "t_max_c": float(t.max()),
                "t_median_c": float(np.median(t)),
                "t_hot_px_60c": int((t > HOT_C).sum()),
                "t_hot_px_70c": int((t > VERY_HOT_C).sum()),
            }
        pose = meta.get("source_trajectory_pose") or {}
        rows.append({
            "frame_id": int(meta.get("frame_id", -1)),
            "sim_time": float(meta.get("simulation_time_s", meta.get("simulation_time_before", 0.0))),
            "position": pose.get("position"),
            "orientation": pose.get("orientation"),
            "camera_world_matrix": meta.get("camera_world_matrix"),
            "rgb_png": f"{stem.name}_rgb.png",
            "thermal_png": f"{stem.name}_thermal.png",
            "apparent_t_npy": f"{stem.name}_apparent_t.npy",
            "source_stage_sha256": meta.get("source_stage_sha256"),
            "thermal_config_sha256": meta.get("thermal_config_sha256"),
            **stats,
        })
    return rows


def build(args: argparse.Namespace) -> int:
    run_dir: Path = args.run
    replay: Path = args.replay
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    task_path = (args.config.parent / config["task"]["file"]).resolve()
    task = yaml.safe_load(task_path.read_text(encoding="utf-8-sig"))
    target = np.asarray(task.get("target_position") or task.get("environment_evidence", {})
                        .get("factory_inspection", {}).get("hotspot_position_m"), dtype=float)
    if target.size != 3:
        raise SystemExit("task file carries no target_position; cannot label frames")

    frames = collect_frames(replay)
    if not frames:
        raise SystemExit(f"no frame_*.json under {replay}")

    first_png = next(replay.glob("frame_*_rgb.png"))
    from PIL import Image
    with Image.open(first_png) as img:
        width, height = img.size

    run_id = run_dir.name
    # Provenance lives in the recorder's run metadata, not in the per-frame JSON (which only
    # carries the common-metadata hash). Recompute the stage hash so a dataset can prove the
    # scene it was rendered from is still the one on disk.
    run_meta = {}
    run_meta_path = run_dir / "run_metadata.json"
    if run_meta_path.exists():
        run_meta = load_json(run_meta_path)
    scene_sha = run_meta.get("stage_sha256") or next(
        (f["source_stage_sha256"] for f in frames if f["source_stage_sha256"]), None)
    stage_path = Path(run_meta.get("stage_path") or "")
    scene_sha_now = sha256_file(stage_path) if stage_path.is_file() else None
    scene_hash_verified = bool(scene_sha and scene_sha_now and scene_sha == scene_sha_now)

    index_rows = []
    for f in frames:
        pos = np.asarray(f["position"] or [float("nan")] * 3, dtype=float)
        u = v = float("nan")
        in_view = False
        if f["camera_world_matrix"] is not None:
            u, v, in_view, _ = project_target(f["camera_world_matrix"], target, width, height)
        index_rows.append({
            "run_id": run_id,
            "task_id": task.get("task_id"),
            "frame_id": f["frame_id"],
            "sim_time_s": round(f["sim_time"], 4),
            "pos_x": round(float(pos[0]), 4), "pos_y": round(float(pos[1]), 4),
            "pos_z": round(float(pos[2]), 4),
            "quat_w": (f["orientation"] or [None] * 4)[0],
            "quat_x": (f["orientation"] or [None] * 4)[1],
            "quat_y": (f["orientation"] or [None] * 4)[2],
            "quat_z": (f["orientation"] or [None] * 4)[3],
            "dist_to_target_m": round(float(np.linalg.norm(pos - target)), 4),
            "target_u_px": None if math.isnan(u) else round(u, 2),
            "target_v_px": None if math.isnan(v) else round(v, 2),
            "target_in_view": bool(in_view),
            "t_min_c": round(f["t_min_c"], 3) if "t_min_c" in f else None,
            "t_max_c": round(f["t_max_c"], 3) if "t_max_c" in f else None,
            "t_median_c": round(f["t_median_c"], 3) if "t_median_c" in f else None,
            "t_hot_px_60c": f.get("t_hot_px_60c"),
            "t_hot_px_70c": f.get("t_hot_px_70c"),
            "rgb_png": f["rgb_png"],
            "thermal_png": f["thermal_png"],
            "apparent_t_npy": f["apparent_t_npy"],
            "instruction": task.get("instruction"),
        })

    evaluation = {}
    if args.eval and Path(args.eval).exists():
        evaluation = load_json(Path(args.eval))

    traj = [json.loads(line) for line in
            (run_dir / "trajectory.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    path_len = sum(math.dist(traj[i - 1]["position"], traj[i]["position"]) for i in range(1, len(traj)))
    speeds = [
        math.dist(traj[i - 1]["position"], traj[i]["position"])
        / max(1e-9, traj[i]["simulation_time"] - traj[i - 1]["simulation_time"])
        for i in range(1, len(traj))
    ]
    fi = task.get("environment_evidence", {}).get("factory_inspection", {})
    episode = {
        "run_id": run_id,
        "task_id": task.get("task_id"),
        "instruction": task.get("instruction"),
        "target_prim_path": task.get("target_prim_path"),
        "target_description": task.get("target_description"),
        "target_x": float(target[0]), "target_y": float(target[1]), "target_z": float(target[2]),
        "target_temperature_c": fi.get("thermal_truth", {}).get("hotspot_temperature_c"),
        "frames": len(index_rows),
        "duration_s": round(traj[-1]["simulation_time"] - traj[0]["simulation_time"], 3) if traj else None,
        "path_length_m": round(path_len, 3),
        "max_speed_m_s": round(max(speeds), 3) if speeds else None,
        "median_speed_m_s": round(sorted(speeds)[len(speeds) // 2], 3) if speeds else None,
        "frames_target_in_view": sum(1 for r in index_rows if r["target_in_view"]),
        "eval_result": evaluation.get("result"),
        "eval_distance_and_hold_pass": evaluation.get("distance_and_hold_pass"),
        "eval_safety_pass": evaluation.get("safety_pass"),
        "eval_observation_duration_s": evaluation.get("distance_observation_duration_s"),
        "eval_manual_confirmation": (evaluation.get("manual_confirmation") or {}).get("status"),
        "required_hold_s": fi.get("inspection_requirements", {}).get("hold_duration_s"),
        "required_observation_range_m": fi.get("inspection_requirements", {}).get("observation_distance_range_m"),
        "scene_sha256": scene_sha,
        "scene_stage": config["scene"]["stage"],
        "thermal_config": config["sensors"]["thermal_config"],
        "task_file": task_path.name,
        "image_width": width,
        "image_height": height,
    }

    if args.copy_images:
        dest = out / "frames"
        dest.mkdir(exist_ok=True)
        for f in frames:
            for key in ("rgb_png", "thermal_png"):
                src = replay / f[key]
                if src.exists():
                    shutil.copy2(src, dest / src.name)
        for row in index_rows:
            row["rgb_png"] = str(Path("frames") / row["rgb_png"])
            row["thermal_png"] = str(Path("frames") / row["thermal_png"])
            row["apparent_t_npy"] = str(replay / row["apparent_t_npy"])

    def write_table(name: str, rows: list[dict]) -> None:
        if not rows:
            return
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_table("index", index_rows)
    write_table("episodes", [episode])

    dataset = {
        "dataset_version": 1,
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        .isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_id": run_id,
        "episodes": 1,
        "frames": len(index_rows),
        "frames_target_in_view": episode["frames_target_in_view"],
        "scene_sha256": scene_sha,
        "scene_sha256_now": scene_sha_now,
        "scene_hash_verified": scene_hash_verified,
        "scene_stage": run_meta.get("stage_path") or config["scene"]["stage"],
        "task_file": task_path.name,
        "task_sha256": sha256_file(task_path),
        "scene_config_sha256": run_meta.get("scene_config_sha256"),
        "sensor_config_sha256": run_meta.get("sensor_config_sha256"),
        "thermal_config_sha256": run_meta.get("thermal_config_sha256"),
        "thermal_config": config["sensors"]["thermal_config"],
        "camera": {"focal_mm": FOCAL_MM, "h_aperture_mm": HAPERTURE_MM,
                   "v_aperture_mm": VAPERTURE_MM, "width": width, "height": height},
        "target": {"prim": task.get("target_prim_path"), "position": target.tolist(),
                   "temperature_c": episode["target_temperature_c"]},
        "evaluation": {k: episode[k] for k in episode if k.startswith("eval_")},
        "tables": {"index": "index.csv", "episodes": "episodes.csv"},
    }
    (out / "dataset.json").write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    hot = sum(1 for r in index_rows if r["target_in_view"])
    print(f"[dataset] {out}")
    print(f"  frames            : {len(index_rows)}")
    print(f"  target in view    : {hot}")
    print(f"  eval result       : {episode['eval_result']}")
    print(f"  scene sha256      : {(scene_sha or '')[:16]}...")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--eval", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--copy-images", action="store_true")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
