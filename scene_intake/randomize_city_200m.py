"""按 episode 随机化候选，生成一套可直接跑的场景/热/任务配置。

每个 episode 随机三件事：
  1. 三辆白车里哪一辆引擎盖热（65 ℃）
  2. 三个人里哪一个是低体温（28 ℃）
  3. 整组道具落在哪一段街区（Scenario 沿 +X 平移 45 m × 0..2）

产物（都在 configs/ 下，扁平命名，和既有配置一致）：
  city_200m_ep007.usda                  episode 叠加层（只平移 Scenario）
  city_200m_ep007_scene.yaml
  city_200m_ep007_thermal.yaml
  city_200m_ep007_task.yaml
  city_200m_ep007_meta.json             episode 参数（目标、温度、平移）

用法：
    python scene_intake\\randomize_city_200m.py --episode 7 --kind n3
    python scene_intake\\randomize_city_200m.py --episode 8 --kind n5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

PROJECT = Path(r"D:\IsaacSim_Demo")
CONFIGS = PROJECT / "configs"
SCENARIO = PROJECT / "city_200m_scenario.usda"

VANS = [(84.0, -19.9), (96.0, -21.8), (108.0, -19.9)]
PERSONS = [
    ("Person_01", "/World/Targets/Person_01", 105.0, -27.0),
    ("Person_02", "/World/Scenario/Person_02/Rig/Body", 28.0, -27.0),
    ("Person_03", "/World/Scenario/Person_03/Rig/Body", 135.0, -27.0),
]
PARTS = [
    ("CC_Base_Body", "person_skin", "cold_skin", "human_skin"),
    ("CC_Base_Eye", "person_eyes", "cold_eyes", "human_skin"),
    ("V_Neck_Scrub_Shirt", "person_shirt", "cold_clothes", "cotton_clothing"),
    ("Field_Jacket", "person_jacket", "cold_clothes", "polyester_clothing"),
    ("Cargo_pant", "person_pants", "cold_pants", "cotton_clothing"),
    ("Basketball_Shoes", "person_shoes", "cold_shoes", "shoe_rubber"),
    ("Baseball_Cap", "person_cap", "cold_cap", "cotton_clothing"),
]
HOOD_LOCAL = (1.72, 0.0, 2.17)

PART_TEMPS = {"CC_Base_Body": 33.0, "CC_Base_Eye": 33.0, "V_Neck_Scrub_Shirt": 31.0,
              "Field_Jacket": 29.5, "Cargo_pant": 29.0, "Basketball_Shoes": 28.0,
              "Baseball_Cap": 29.0}
COLD_TEMPS = {"CC_Base_Body": 28.0, "CC_Base_Eye": 28.0, "V_Neck_Scrub_Shirt": 26.0,
              "Field_Jacket": 26.0, "Cargo_pant": 25.5, "Basketball_Shoes": 24.0,
              "Baseball_Cap": 26.0}


def episode_paths(index: int) -> dict:
    stem = f"city_200m_ep{index:03d}"
    return {
        "stem": stem,
        "stage": PROJECT / f"{stem}.usda",
        "scene": CONFIGS / f"{stem}_scene.yaml",
        "thermal": CONFIGS / f"{stem}_thermal.yaml",
        "task": CONFIGS / f"{stem}_task.yaml",
        "meta": CONFIGS / f"{stem}_meta.json",
    }


def make_thermal(hot_van: int, cold_person: int) -> dict:
    base = yaml.safe_load((CONFIGS / "city_200m_thermal_scenario.yaml").read_text(encoding="utf-8"))
    drop_prefixes = (
        "/World/Targets/Person_01/",
        "/World/Scenario/Person_02/",
        "/World/Scenario/Person_03/",
        "/World/Scenario/Van_01/Hood",
        "/World/Scenario/Van_02/Hood",
        "/World/Scenario/Van_03/Hood",
    )
    bindings = [item for item in base["bindings"]
                if not item["prim_path"].startswith(drop_prefixes)]
    counter = 0

    def add(prim, target, material, children=False):
        nonlocal counter
        counter += 1
        bindings.append({"name": f"ep_{counter:04d}", "prim_path": prim,
                         "include_children": children, "target": target,
                         "material": material})

    for index in range(3):
        root = f"/World/Scenario/Van_{index + 1:02d}"
        add(f"{root}/Hood", "van_hood_hot" if index == hot_van else "van_hood_cold",
            "car_paint_white")
    for index, (_, root, _, _) in enumerate(PERSONS):
        cold = index == cold_person
        for part, normal_target, cold_target, material in PARTS:
            add(f"{root}/{part}", cold_target if cold else normal_target, material)
    return {"schema_version": 1, "defaults": dict(base["defaults"]),
            "targets": dict(base["targets"]), "bindings": bindings}


def stage_text(shift: int, cold_person: int) -> str:
    if True:
        attrs = []
        for index, (_, _, _, _) in enumerate(PERSONS):
            cold = index == cold_person
            temps = {part: (COLD_TEMPS[part] if cold else PART_TEMPS[part])
                     for part, _, _, _ in PARTS}
            body = "\n".join(
                f'                    over "{part}"\n'
                "                    {\n"
                f"                        custom float thermal:temperatureC = {temp}\n"
                "                    }"
                for part, temp in temps.items())
            if index == 0:
                attrs.append('    over "Targets"\n    {\n        over "Person_01"\n        {\n'
                             + body + "\n        }\n    }")
            else:
                attrs.append(
                    f'        over "Person_0{index + 1}"\n'
                    "        {\n"
                    '            over "Rig"\n'
                    "            {\n"
                    '                over "Body"\n'
                    "                {\n"
                    + body + "\n                }\n            }\n        }")
        return "\n".join([
            "#usda 1.0",
            "(",
            '    defaultPrim = "World"',
            "    metersPerUnit = 1",
            '    upAxis = "Z"',
            "    startTimeCode = 0",
            "    endTimeCode = 2400",
            "    timeCodesPerSecond = 24",
            "    framesPerSecond = 24",
            "    subLayers = [",
            "        @./city_200m_scenario.usda@",
            "    ]",
            "    customLayerData = {",
            '        string comment = "City scenario episode: props shifted along +X by 45 m x block."',
            "    }",
            ")",
            'over "World"',
            "{",
            '    over "Scenario"',
            "    {",
            f"        double3 xformOp:translate = ({45.0 * shift}, 0, 0)",
            '        uniform token[] xformOpOrder = ["xformOp:translate"]',
            "\n".join(attrs[1:]),
            "    }",
            "",
            attrs[0],
            "}",
            "",
        ])


def scene_text(stem: str, shift: int) -> str:
    return f"""\
# episode {stem}：道具平移 {45 * shift} m（第 {shift} 段街区）
schema_version: 1

scene:
  stage: ../{stem}.usda
  thermal_scene: city_200m_night_environment.yaml
  static_targets: []

drone:
  prim_path: /World/CityFlight/Crazyflie
  body_path: /World/CityFlight/Crazyflie/body
  camera_path: /World/CityFlight/Crazyflie/body/CityCam
  minimum_height: 1.5

sensors:
  width: 320
  height: 256
  sensor_config: ../third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml
  thermal_config: {stem}_thermal.yaml
  preview_fast: true
  align_thermal_to_rgb: true
  flat_field_enabled: true
  sensor_seed: 741
  display_config: city_200m_display_scenario.yaml
  material_directory: materials/city_200m
"""


def task_text(kind: str, hot_van: int, cold_person: int, shift: int) -> str:
    offset = 45.0 * shift
    start = f"[{70.0 + offset}, -22.0, 6.5]"
    if kind == "n3":
        x, y = VANS[hot_van]
        target = f"[{x + HOOD_LOCAL[0] + offset:.2f}, {y}, {HOOD_LOCAL[2]}]"
        return f"""\
# S1/N3 episode：三辆同款白车，只有一辆引擎盖热（目标随机落在第 {hot_van + 1} 辆）
task_id: city-ep{shift}-n3-van{hot_van + 1}
instruction: 街边停着三辆同款白色面包车，找到发动机刚熄火、引擎盖还热的那一辆。
target_prim_path: /World/Scenario/Van_{hot_van + 1:02d}/Hood
target_description: 三辆同款白色面包车中，引擎盖温度明显偏高的那一辆
environment_description: 夜间城市主街，三辆同款白色面包车错开停放，附近有空调外机和排风口
environment_landmarks:
  - 主街北侧三辆白色面包车
  - 路缘上的空调外机
start_position: {start}
target_position: {target}
target_position_semantics: 目标车辆引擎盖中心，米制、Z-up、世界坐标
notes: 车身 26 ℃，目标引擎盖 65 ℃；空调外机 50 ℃ 为干扰热源
"""
    _, _, px, py = PERSONS[cold_person]
    target = f"[{px + offset}, {py}, 0.9]"
    return f"""\
# S1/N5 episode：三个人外观一致，只有一个人体温偏低（目标随机落在 Person_0{cold_person + 1}）
task_id: city-ep{shift}-n5-person{cold_person + 1}
instruction: 街上有三个人，找到体温明显偏低、可能已经失温的那一个。
target_prim_path: {PERSONS[cold_person][1]}
target_description: 三名同款行人中，皮肤温度只有 28 ℃ 的那一个
environment_description: 夜间城市人行道，三名行人分散站立，附近的空调外机和排风口也有热辐射
environment_landmarks:
  - 人行道上分散站立的三名行人
  - 路缘上的空调外机
start_position: {start}
target_position: {target}
target_position_semantics: 目标人物包围盒中心，米制、Z-up、世界坐标
notes: 另外两人 33 ℃；目标皮肤 28 ℃、衣物 24–26 ℃
"""


def build(index: int, kind: str, seed: int | None = None) -> dict:
    rng = random.Random(seed if seed is not None else index * 7919)
    hot_van = rng.randrange(3)
    cold_person = rng.randrange(3)
    shift = rng.randrange(3)
    paths = episode_paths(index)

    paths["stage"].write_text(stage_text(shift, cold_person), encoding="utf-8")
    paths["scene"].write_text(scene_text(paths["stem"], shift), encoding="utf-8")
    thermal = make_thermal(hot_van, cold_person)
    paths["thermal"].write_text(
        f"# 由 scene_intake/randomize_city_200m.py 生成：episode {index}\n"
        + yaml.safe_dump(thermal, sort_keys=False, allow_unicode=True,
                         default_flow_style=False, width=200), encoding="utf-8")
    paths["task"].write_text(task_text(kind, hot_van, cold_person, shift), encoding="utf-8")
    meta = {"episode": index, "kind": kind, "seed": seed, "hot_van": hot_van + 1,
            "cold_person": cold_person + 1, "block_shift": shift,
            "stage": str(paths["stage"]), "scene": str(paths["scene"]),
            "thermal": str(paths["thermal"]), "task": str(paths["task"])}
    paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def check(index: int) -> int:
    sys.path.insert(0, str(PROJECT))
    from scripts.scene_config import load_scene_config
    from scripts.thermal_config import load_thermal_config

    paths = episode_paths(index)
    failures = []
    try:
        config = load_scene_config(paths["scene"])
        print(f"OK scene   {config.scene.stage}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"scene: {exc}")
    try:
        thermal = load_thermal_config(paths["thermal"])
        print(f"OK thermal bindings={len(thermal.bindings)}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"thermal: {exc}")
    if not failures:
        try:
            from pxr import Usd
            stage = Usd.Stage.Open(str(paths["stage"]))
            missing = [item.prim_path for item in thermal.bindings
                       if not stage.GetPrimAtPath(item.prim_path)]
            if missing:
                failures.append(f"missing binding prims: {missing[:3]} ({len(missing)})")
            else:
                print("OK prims   all binding paths resolve")
        except ImportError:
            print("SKIP prims (pxr not on PYTHONPATH)")
    for item in failures:
        print("FAIL " + item)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--kind", choices=("n3", "n5"), default="n3")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        meta = build(args.episode, args.kind, args.seed)
        print(f"WROTE episode {meta['episode']}: hot_van={meta['hot_van']} "
              f"cold_person={meta['cold_person']} shift={meta['block_shift']}")
    return check(args.episode)


if __name__ == "__main__":
    raise SystemExit(main())
