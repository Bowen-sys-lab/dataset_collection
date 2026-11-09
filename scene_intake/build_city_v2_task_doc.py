import io
import json
import math

import yaml

ROOT = r"D:\IsaacSim_Demo"
table = json.load(open(ROOT + r"\configs\city_200m_v2_task_table.json", encoding="utf-8"))
episodes = json.load(open(ROOT + r"\configs\collection_episodes\city_v2.json", encoding="utf-8"))
episode_of = {row["task_id"]: row["episode"] for row in episodes["episodes"]}
extras = json.load(open(ROOT + r"\configs\city_200m_v2_task_extras.json", encoding="utf-8"))
extra_of = {row["task_id"]: row for row in extras["tasks"]}

SECTORS = ["东", "东北", "北", "西北", "西", "西南", "南", "东南"]


def sector(dx, dy):
    angle = math.degrees(math.atan2(dy, dx)) % 360.0
    return SECTORS[int(round(angle / 45.0)) % 8]


lines = []
lines.append("# 城市 v2 夜间任务表（C01–C20）\n")
lines.append("日期：2026-11-06。场景：主场景夜间（暗光）。统一入口：`record.ps1 -Scene city -Episode Cxx`。\n")
lines.append("| 任务 | 起点 | 初始方向 | 去哪里找、找什么 |")
lines.append("|---|---|---|---|")
for task in table["tasks"]:
    doc = yaml.safe_load(open(task["file"], encoding="utf-8"))
    sx, sy, sz = task["start"]
    tx, ty, tz = doc["target_position"]
    approach = sector(sx - tx, sy - ty)
    block = task["block"]
    center_x = 2.7 + 45.0 * block
    inside = abs(sx - center_x) <= 22.5 and -18.0 <= sy <= 21.2
    direction = task["direction"]
    suffix = ""
    if inside:
        suffix = "（初始遮挡）"
    text = f"从{approach}侧进入，" + extra_of[task["task_id"]]["search_description"]
    lines.append(
        f"| `{episode_of[task['task_id']]}` | 街区{block} {approach}侧（{task['start_distance_m']:.0f} m） | "
        f"{direction}{suffix} | {text} |"
    )
lines.append("\n起点坐标、yaw、任务 id 见 `configs/city_200m_v2_task_table.json`；"
             "“距目标”只是起点到目标的水平距离，不是判定条件。"
             "“初始遮挡”按起点是否落在街区内部判定，试飞时逐条确认。\n")
lines.append("## 采集\n")
lines.append("```powershell")
lines.append(r'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "D:\IsaacSim_Demo\scripts\record.ps1" -Scene city -Episode C01')
lines.append("```")
lines.append("\n预检加 `-CheckOnly`。\n")
io.open(ROOT + r"\docs\city-v2-night-tasks.md", "w", encoding="utf-8", newline="\n").write("\n".join(lines))
print("wrote", len(table["tasks"]), "rows")
