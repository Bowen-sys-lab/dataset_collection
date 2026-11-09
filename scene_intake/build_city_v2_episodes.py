"""Map the 20 city v2 night tasks to the unified C01..C20 Episode ids."""
import json
import pathlib

ROOT = pathlib.Path(r"D:\IsaacSim_Demo")
table = json.loads((ROOT / "configs" / "city_200m_v2_task_table.json").read_text(encoding="utf-8"))
episodes = []
for index, task in enumerate(table["tasks"], start=1):
    episodes.append({
        "episode": f"C{index:02d}",
        "scene": "city",
        "task_id": task["task_id"],
        "target": task["target"],
        "block_index": task["block"],
        "start_position_m": task["start"],
        "initial_yaw_degrees": task["yaw"],
        "target_bearing_degrees": None,
        "relative_direction": task["direction"],
        "task_file": task["file"],
    })
doc = {
    "schema_version": 1,
    "scene": "city",
    "scheme": "city_v2_night_low_light",
    "count": len(episodes),
    "source_table": "configs/city_200m_v2_task_table.json",
    "episodes": episodes,
}
out = ROOT / "configs" / "collection_episodes" / "city_v2.json"
out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
print("wrote", out, len(episodes))
