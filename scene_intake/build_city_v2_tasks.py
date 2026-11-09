"""Build the city v2 night task table: 4 targets x 5 starts.

Starts are sampled on rings around each target, rejected when they fall inside
building bounding boxes or when the straight line to the target is blocked.
Each task gets a fixed non-forward initial yaw and an instruction carrying the
target direction relative to that yaw (TravelUAV-style).

Outputs:
  configs/city_200m_v2_task_<target>_s<N>.yaml
  configs/city_200m_v2_task_table.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml
from pxr import Gf, Usd, UsdGeom

PROJECT = Path(r"D:\IsaacSim_Demo")
STAGE = PROJECT / "city_200m_v2_flight.usda"
LAYOUT = PROJECT / "city_200m_v2_local" / "records" / "target_layout.json"
OUT = PROJECT / "configs"

TARGETS = ["HeatMainCover", "HVAC", "ExhaustVent", "Person_02"]
PRIM = {
    "HeatMainCover": "/World/Targets/HeatMainCover",
    "HVAC": "/World/Targets/HVAC",
    "ExhaustVent": "/World/Targets/ExhaustVent",
    "Person_02": "/World/Targets/Person_02/CC_Base_Body",
}
FACTS = {
    "HeatMainCover": {
        "object": "一块直径约 0.7 米的圆形金属井盖，盖面有同心圆纹路，平嵌在沥青路面上",
        "environment": "位于两侧楼体夹出的内部窄街，路边有低矮绿植和停放的车辆，这段窄街只有这一块圆形井盖",
        "thermal": "热像上盖面是一整块亮圆盘（约 38 ℃），比周围沥青明显更热",
        "landmarks": ["内部窄街", "圆形井盖", "路边绿植"],
    },
    "HVAC": {
        "object": "一台约 1 米宽、0.8 米高的白色空调外机，正面有圆形风扇格栅，落在矮基座上",
        "environment": "在东北楼后的服务带里靠墙摆放，旁边是铁门和落水管，这条服务带只有这一台外机",
        "thermal": "热像上外机盘管位置有一小块明显热斑（约 43 ℃），机壳其余部分接近环境温度",
        "landmarks": ["楼后服务带", "白色外机", "铁门"],
    },
    "ExhaustVent": {
        "object": "一个约 1 米高的金属排风基座，正面是横向百叶，顶部接短风管",
        "environment": "立在南侧建筑服务前缘的路缘旁，旁边有行道树和路缘石，这一侧只有这一个百叶基座",
        "thermal": "热像上百叶口呈条状热异常（约 40 ℃），比周围地面和墙面都亮",
        "landmarks": ["南侧服务区", "百叶基座", "行道树"],
    },
    "Person_02": {
        "object": "一名身高约 1.76 米的行人，穿深色外套、长裤，戴帽子，站在人行道上",
        "environment": "位于西侧楼旁，身旁有绿植和一层阳台，这段人行道只有这一名行人",
        "thermal": "热像上呈暖色人形，皮肤约 33 ℃、衣物 28–31 ℃，比背景墙面明显更暖",
        "landmarks": ["西侧人行道", "绿植", "阳台"],
    },
}

# 每个目标 5 条轨迹用 5 种不同说法：目标特征、环境、唯一性、热特征都要保留。
ROW_TEXTS = {
    "HeatMainCover": [
        "沿内部窄街找唯一一块圆形金属井盖：直径约 0.7 米，盖面有同心圆纹路，平嵌在沥青里；热像上是一整块亮圆盘（约 38 ℃），比周围路面明显更热。",
        "目标是一块平嵌在窄街沥青上的圆形金属井盖，盖面同心圆纹路、直径约 0.7 米；这段窄街只有这一块，路边是低矮绿植和停放的车辆。红外下盖面约 38 ℃，呈完整亮圆盘。",
        "在两侧楼体夹出的窄街里，找路面中央唯一的圆形金属盖板：同心圆纹路、直径约 0.7 米；热像显示盖面为亮圆盘（约 38 ℃），是这段街最热的地面目标。",
        "沿窄街向里搜索，目标是一块圆形金属井盖，盖面有同心圆纹路、直径约 0.7 米，周围是绿植和停车；这段街没有第二块圆形井盖，红外下盖面约 38 ℃、明显亮于沥青。",
        "找窄街路面上唯一的圆形金属井盖：直径约 0.7 米、盖面同心圆纹路；热像上是一整块约 38 ℃ 的亮圆盘，与两侧楼体和路面的冷背景分得很开。",
    ],
    "HVAC": [
        "沿楼后服务带找唯一一台白色空调外机：约 1 米宽、0.8 米高，正面是圆形风扇格栅，落在矮基座上；旁边是铁门和落水管。热像上盘管位置有一块约 43 ℃ 的热斑，机壳其余部分接近环境温度。",
        "目标是一台靠墙摆放的白色空调外机，正面圆形风扇格栅、约 1 米宽，底部有矮基座；它是东北楼后服务带里唯一一台白色外机，旁边是铁门和落水管。红外下只有盘管处一小块 43 ℃ 热斑。",
        "绕到东北楼后，在服务带里找唯一的白色外机：约 0.8 米高、1 米宽，圆形风扇格栅很显眼；热像上盘管位置约 43 ℃，比机壳和墙面都亮。",
        "沿服务带靠墙一侧搜索，找一台白色空调外机：圆形风扇格栅、矮基座，约 1 米宽；这条服务带只有这一台，旁边是铁门和落水管。红外下盘管处有一小块 43 ℃ 热斑，其余部分不热。",
        "找楼后服务带里唯一的空调外机：白色机身、圆形风扇格栅、约 0.8 米高；热像上盘管位置约 43 ℃，是小而集中的热斑，机壳其余部分接近环境温度。",
    ],
    "ExhaustVent": [
        "沿南侧建筑服务前缘找唯一一个金属排风基座：约 1 米高，正面是横向百叶，顶部接短风管；旁边有行道树和路缘石。热像上百叶口呈条状热异常（约 40 ℃），比地面和墙面都亮。",
        "目标是一个立在南侧路缘旁的排风基座，约 1 米高，正面横向百叶、顶部短风管；这一侧只有这一个百叶基座，旁边是行道树。红外下百叶口约 40 ℃，呈条状亮带。",
        "在南侧服务前缘搜索，找这一侧唯一的百叶金属基座：约 1 米高，顶部接短风管；热像上百叶口是一条约 40 ℃ 的热带，明显亮于周围地面。",
        "沿路缘找唯一的排风基座：金属外壳、约 1 米高，正面横向百叶；旁边有行道树和路缘石，这一侧没有第二个百叶基座。红外下百叶口约 40 ℃，呈条状热异常。",
        "在南侧服务区里找顶部带短风管的金属排风基座：约 1 米高，正面横向百叶；热像上百叶口约 40 ℃，是这一片最明显的条状热源。",
    ],
    "Person_02": [
        "沿西侧楼旁人行道找唯一一名行人：身高约 1.76 米，穿深色外套、长裤，戴帽子；身旁有绿植和一层阳台。热像上呈暖色人形，皮肤约 33 ℃、衣物 28–31 ℃，比背景墙面更暖。",
        "目标是一名站在人行道上的行人，约 1.76 米高，深色外套、长裤、戴帽子；这段人行道只有这一名行人，旁边是绿植和一层阳台。红外下人体皮肤约 33 ℃，衣物 28–31 ℃。",
        "在西侧楼旁的人行道上找唯一的行人：身高约 1.76 米，深色外套、长裤、帽子；热像上是一个约 33 ℃ 的暖色人形，比墙体和绿植都亮。",
        "沿西侧人行道搜索，找一名穿深色外套、戴帽子的行人，身高约 1.76 米；身旁有绿植和阳台，这段路只有这一名行人。红外下皮肤约 33 ℃、衣物 28–31 ℃，呈完整人形热轮廓。",
        "找西侧人行道上唯一的行人：约 1.76 米高，深色外套、长裤、帽子；热像上皮肤约 33 ℃、衣物 28–31 ℃，暖色人形和冷背景分得很开。",
    ],
}
SECTORS = ["正前方", "右前方", "右侧", "右后方", "正后方", "左后方", "左侧", "左前方"]


def load_layout() -> dict:
    data = json.loads(LAYOUT.read_text(encoding="utf-8"))
    return {t["name"]: t for t in data["targets"]}


def building_boxes(stage) -> list[tuple[str, Gf.Range3d]]:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, "render"])
    boxes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if not path.startswith("/World/Blocks/"):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not box.IsEmpty():
            boxes.append((path, box))
    return boxes


def inside(point: Gf.Vec3d, box: Gf.Range3d, margin: float = 0.35) -> bool:
    return all(box.GetMin()[i] - margin <= point[i] <= box.GetMax()[i] + margin for i in range(3))


def blocked(start: Gf.Vec3d, end: Gf.Vec3d, boxes, skip_prefix: str) -> bool:
    steps = max(8, int((end - start).GetLength() / 0.5))
    for index in range(1, steps):
        point = start + (end - start) * (index / steps)
        for path, box in boxes:
            if path.startswith(skip_prefix):
                continue
            if inside(point, box, margin=0.15):
                return True
    return False


def sector_for(yaw_deg: float, bearing_deg: float) -> str:
    delta = (bearing_deg - yaw_deg + 540.0) % 360.0 - 180.0
    index = int(round(delta / 45.0)) % 8
    return SECTORS[index]


def choose_starts(target: dict, boxes) -> list[dict]:
    """Sample free points on rings around the target (10-26 m, 15-degree steps).

    Bounding-box blocking is intentionally not used: the asset's assembly boxes
    overstate occlusion and rejected every street point. Free space at 6.5 m plus
    later visual review is the acceptance gate.
    """
    tx, ty, tz = target["position_m"]
    candidates = []
    for radius in (10.0, 14.0, 18.0, 22.0, 26.0):
        for step in range(24):
            angle = math.radians(step * 15.0)
            point = Gf.Vec3d(tx + radius * math.cos(angle), ty + radius * math.sin(angle), 6.5)
            if any(inside(point, box) for _, box in boxes):
                continue
            bearing = math.degrees(math.atan2(ty - point[1], tx - point[0])) % 360.0
            candidates.append({"position": [round(point[i], 2) for i in range(3)],
                               "bearing": round(bearing, 1), "radius": radius,
                               "angle": step * 15})
    if len(candidates) < 5:
        raise RuntimeError(f"not enough free starts for {target['name']}: {len(candidates)}")
    candidates.sort(key=lambda c: (c["bearing"], c["radius"]))
    chosen = [candidates[0]]
    while len(chosen) < 5:
        best = None
        best_gap = -1.0
        for candidate in candidates:
            if candidate in chosen:
                continue
            gap = min(min(abs(candidate["bearing"] - other["bearing"]),
                          360 - abs(candidate["bearing"] - other["bearing"]))
                      for other in chosen)
            if gap > best_gap:
                best, best_gap = candidate, gap
        chosen.append(best)
    chosen.sort(key=lambda c: c["bearing"])
    return chosen


def build() -> dict:
    stage = Usd.Stage.Open(str(STAGE))
    boxes = building_boxes(stage)
    layout = load_layout()
    offsets = [100, -100, 140, -140, 80]
    table = []
    extras = []
    for target_name in TARGETS:
        target = layout[target_name]
        starts = choose_starts(target, boxes)
        for index, start in enumerate(starts, start=1):
            yaw = round((start["bearing"] + offsets[index - 1]) % 360.0, 1)
            direction = sector_for(yaw, start["bearing"])
            facts = FACTS[target_name]
            row_text = ROW_TEXTS[target_name][index - 1]
            instruction = (f"目标在你的{direction}。{row_text} "
                           f"先转向目标方向，靠近后确认热异常并按 T。")
            variants = [
                instruction,
                (f"目标在你的{direction}；{row_text} 先转向再靠近确认。"),
                (f"先看热像：{facts['thermal']}。目标在你的{direction}；{row_text} 转向后靠近确认。"),
            ]
            task_id = f"city-v2-{target_name.lower()}-s{index}"
            task = {
                "task_id": task_id,
                "instruction": instruction,
                "target_prim_path": PRIM[target_name],
                "target_description": facts["object"],
                "environment_description": facts["environment"],
                "environment_landmarks": facts["landmarks"],
                "start_position": start["position"],
                "target_position": [round(v, 3) for v in target["position_m"]],
                "target_position_semantics": "目标底部放置点，米制、Z-up、世界坐标",
                "notes": "人工按 T 确认；自动评分独立进行，T 不等于自动任务通过。",
                "environment_evidence": {
                    "scene": "city_200m_v2",
                    "scheme": "night_low_light",
                    "target": target_name,
                    "block_index": target["block_index"],
                    "initial_yaw_degrees": yaw,
                    "target_bearing_degrees": start["bearing"],
                    "relative_direction": direction,
                    "start_distance_m": start["radius"],
                    "start_index": index,
                },
            }
            path = OUT / f"city_200m_v2_task_{target_name.lower()}_s{index}.yaml"
            path.write_text(yaml.safe_dump(task, allow_unicode=True, sort_keys=False, width=200),
                            encoding="utf-8", newline="\n")
            table.append({"task_id": task_id, "target": target_name, "block": target["block_index"],
                          "start": start["position"], "yaw": yaw, "direction": direction,
                          "start_distance_m": start["radius"], "file": str(path)})
            extras.append({"task_id": task_id, "search_description": row_text,
                           "thermal_description": facts["thermal"],
                           "instruction_variants": variants})
    (OUT / "city_200m_v2_task_extras.json").write_text(
        json.dumps({"tasks": extras}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (OUT / "city_200m_v2_task_table.json").write_text(
        json.dumps({"scheme": "night_low_light", "targets": TARGETS, "tasks": table},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"tasks": len(table), "table": str(OUT / "city_200m_v2_task_table.json")}


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
