"""Version the four City v2 pilot starts without changing prior task files."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
VERSION = "city20_long_start_trial_v1"
PILOTS = {
    "C01": ((225.0, -22.0, 6.5), 180.0, "前方偏右", "right", "westward"),
    "C06": ((135.0, -22.0, 6.5), 180.0, "前方偏右", "right", "westward"),
    "C11": ((70.0, -22.0, 6.5), 0.0, "前方偏左", "left", "eastward"),
    "C16": ((135.0, -22.0, 6.5), 180.0, "前方偏右", "right", "westward"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    table_path = ROOT / "configs/city_200m_v2_task_table.json"
    episodes_path = ROOT / "configs/collection_episodes/city_v2.json"
    manifest_path = ROOT / "docs/city20-task-manifest-2026-11-07.json"
    new_manifest_path = ROOT / "docs/city20-task-manifest-2026-11-07-long-start-trial.json"
    task_dir = ROOT / "configs" / VERSION
    backup = ROOT / "validation/diagnostics/city20-long-start-20261107/backup"
    if task_dir.exists() or backup.exists() or new_manifest_path.exists():
        raise FileExistsError("Long-start trial already prepared; preserve the existing version.")

    table = json.loads(table_path.read_text(encoding="utf-8"))
    episodes = json.loads(episodes_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["episode"]: row for row in table["tasks"]}
    launch_rows = {row["episode"]: row for row in episodes["episodes"]}
    manifest_rows = {row["episode"]: row for row in manifest["tasks"]}
    assert len(rows) == len(launch_rows) == len(manifest_rows) == 20
    assert all(rows[ep]["task_version"] == "city20_english_v1" for ep in PILOTS)

    backup.mkdir(parents=True, exist_ok=False)
    old_sources = []
    for original in (table_path, episodes_path, manifest_path):
        saved = backup / original.relative_to(ROOT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, saved)
        old_sources.append({"path": str(saved), "sha256": sha256(saved),
                            "original_path": str(original)})
    task_dir.mkdir(parents=True, exist_ok=False)

    revised = copy.deepcopy(manifest)
    revised_rows = {row["episode"]: row for row in revised["tasks"]}
    revised["plan_id"] = "city20-long-start-trial-20261107"
    revised["status"] = "four_pilot_starts_pending_live_validation"
    revised["task_version"] = "mixed_until_remaining_starts_validated"
    revised["pilot_task_version"] = VERSION
    revised["previous_active_sources"] = old_sources
    revised["visibility_review"] = "pending_pilot_spawn_route_and_terminal_review"
    for ep, (position, yaw, direction, side, travel) in PILOTS.items():
        row, launch, item = rows[ep], launch_rows[ep], revised_rows[ep]
        previous = Path(row["file"])
        task = yaml.safe_load(previous.read_text(encoding="utf-8"))
        target = task["target_position"]
        distance = round(math.dist(position[:2], target[:2]), 2)
        assert 100.0 <= distance <= 200.0, (ep, distance)
        bearing = round(math.degrees(math.atan2(target[1] - position[1], target[0] - position[0])) % 360, 2)
        body = task["instruction"].split(" Find ", 1)[1]
        task["instruction"] = (
            f"The target is several blocks ahead, slightly to your {side}. "
            f"Head {travel} along the southern edge of the district, then search the streets near the target. "
            f"Find {body}"
        )
        task["start_position"] = list(position)
        task["environment_evidence"].update(
            initial_yaw_degrees=yaw, target_bearing_degrees=bearing,
            relative_direction=direction, start_distance_m=distance,
        )
        source = task["environment_evidence"]["city_collection"]
        source.update(version=VERSION, previous_task_path=str(previous),
                      previous_task_sha256=sha256(previous),
                      start_review_status="candidate_pending_live_validation")
        new_task = task_dir / f"{ep}.yaml"
        new_task.write_text(yaml.safe_dump(task, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8")

        row.update(start=list(position), yaw=yaw, direction=direction, start_distance_m=distance,
                   file=str(new_task), task_version=VERSION,
                   start_review_status="candidate_pending_live_validation")
        launch.update(start_position_m=list(position), initial_yaw_degrees=yaw,
                      target_bearing_degrees=bearing, relative_direction=direction,
                      task_file=str(new_task), task_version=VERSION,
                      start_review_status="candidate_pending_live_validation")
        item.update(status="pending_spawn_route_and_manual_terminal_review",
                    start_position_m=list(position), initial_yaw_degrees=yaw,
                    relative_direction_zh=direction, start_horizontal_distance_m=distance,
                    instruction_en=task["instruction"], task_file=str(new_task),
                    task_sha256=sha256(new_task), task_version=VERSION,
                    previous_task_file=str(previous), previous_task_sha256=sha256(previous))
        item["instruction_zh"] = (
            f"目标位于前方数个街区，方向{direction}。沿城市南侧向{'西' if travel == 'westward' else '东'}飞行，"
            "进入目标所在街区后按英文任务识别外观与红外特征，停稳并人工确认。"
        )
        print(f"{ep}: start={position}, yaw={yaw:g}, target bearing={bearing:g}, horizontal={distance:.2f} m")

    for ep, item in revised_rows.items():
        if ep not in PILOTS:
            item["status"] = "pending_long_start_design_after_pilots"
    save_json(table_path, table)
    save_json(episodes_path, episodes)
    revised["active_indexes"] = [
        {"path": str(path), "sha256": sha256(path)} for path in (episodes_path, table_path)
    ]
    save_json(new_manifest_path, revised)
    print(f"Prior indexes and manifest preserved at {backup}")
    print(f"New pilot manifest: {new_manifest_path}")


if __name__ == "__main__":
    main()
