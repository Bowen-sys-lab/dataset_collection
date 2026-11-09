"""给出题版加"黑色车余热诱饵"和"假人"两个 M3 道具。

只新增叠加层，不改 scenario：

  * BlackVan_01  黑色面包车，车身/引擎盖 35 ℃（刚熄火的余热，比白车热、比热引擎盖冷）
  * FakePerson   服装店假人（引用 Person_01 的几何），全身 26 ℃（没有体温）

产物：
  city_200m_decoys.usda
  configs/city_200m_decoys_scene.yaml
  configs/city_200m_thermal_decoys.yaml
  configs/city_200m_task_black_car.yaml
  configs/city_200m_task_fake_person.yaml

用法：
    python scene_intake\\add_city_200m_decoys.py
    python scene_intake\\add_city_200m_decoys.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT = Path(r"D:\IsaacSim_Demo")
CONFIGS = PROJECT / "configs"
STAGE = PROJECT / "city_200m_decoys.usda"
GROUND_Z = -0.2974635
BLACK_VAN = (120.0, -21.6)
FAKE_PERSON = (62.0, -27.0)
PERSON_SRC = (105.0, -27.0)

BLACK_MATERIAL = """\
            def Material "paint_black"
            {
                token outputs:surface.connect = </World/Scenario/Looks/paint_black/Surface.outputs:surface>
                def Shader "Surface"
                {
                    uniform token info:id = "UsdPreviewSurface"
                    color3f inputs:diffuseColor = (0.035, 0.037, 0.04)
                    float inputs:roughness = 0.35
                    token outputs:surface
                }
            }"""


def cube(name, center, size, material, *, collision=False, indent=12) -> str:
    pad = " " * indent
    api = ' (\n{p}    prepend apiSchemas = ["PhysicsCollisionAPI"]\n{p})'.format(p=pad) if collision else ""
    return "\n".join([
        f'{pad}def Cube "{name}"{api}',
        f"{pad}{{",
        f"{pad}    double size = 1",
        f"{pad}    double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})",
        f"{pad}    double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})",
        f'{pad}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
        f"{pad}    rel material:binding = </World/Scenario/Looks/{material}>",
        f"{pad}}}",
    ])


def cylinder(name, center, radius, height, material, indent=12) -> str:
    pad = " " * indent
    return "\n".join([
        f'{pad}def Cylinder "{name}"',
        f"{pad}{{",
        f"{pad}    double radius = {radius}",
        f"{pad}    double height = {height}",
        f"{pad}    float3 xformOp:rotateXYZ = (90, 0, 0)",
        f"{pad}    double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})",
        f'{pad}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]',
        f"{pad}    rel material:binding = </World/Scenario/Looks/{material}>",
        f"{pad}}}",
    ])


def black_van() -> str:
    x, y = BLACK_VAN
    return "\n".join([
        '        def Xform "BlackVan_01"',
        "        {",
        f"            double3 xformOp:translate = ({x}, {y}, {GROUND_Z})",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        cube("Body", (0, 0, 1.15), (4.6, 1.95, 1.9), "paint_black", collision=True),
        cube("Windshield", (2.02, 0, 2.0), (0.10, 1.78, 0.55), "glass_dark"),
        cube("Hood", (1.72, 0, 2.17), (1.15, 1.86, 0.14), "paint_black"),
        cube("Bumper", (2.33, 0, 0.55), (0.10, 1.9, 0.35), "paint_grey"),
        cylinder("Wheel_FL", (1.45, 0.98, 0.35), 0.35, 0.24, "rubber"),
        cylinder("Wheel_FR", (1.45, -0.98, 0.35), 0.35, 0.24, "rubber"),
        cylinder("Wheel_RL", (-1.45, 0.98, 0.35), 0.35, 0.24, "rubber"),
        cylinder("Wheel_RR", (-1.45, -0.98, 0.35), 0.35, 0.24, "rubber"),
        "        }",
    ])


def fake_person() -> str:
    x, y = FAKE_PERSON
    dx, dy = x - PERSON_SRC[0], y - PERSON_SRC[1]
    temps = {"CC_Base_Body": 26.0, "CC_Base_Eye": 26.0, "V_Neck_Scrub_Shirt": 26.0,
             "Field_Jacket": 26.0, "Cargo_pant": 26.0, "Basketball_Shoes": 26.0,
             "Baseball_Cap": 26.0}
    overrides = "\n".join(
        f'                    over "{part}"\n'
        "                    {\n"
        f"                        custom float thermal:temperatureC = {temp}\n"
        "                    }"
        for part, temp in temps.items())
    return "\n".join([
        '        def Xform "FakePerson"',
        "        {",
        f"            double3 xformOp:translate = ({dx}, {dy}, 0)",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        '            def Xform "Rig"',
        "            {",
        "                double3 xformOp:translate = (0, 0, 0)",
        '                uniform token[] xformOpOrder = ["xformOp:translate"]',
        '                def "Body" (',
        "                    prepend references = @./city_200m_local/city_200m.usda@</World/Targets/Person_01>",
        "                )",
        "                {",
        overrides,
        "                }",
        "            }",
        "        }",
    ])


def stage_text() -> str:
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
        '        string comment = "Scenario decoys: one warm black van and one ambient mannequin."',
        "    }",
        ")",
        'over "World"',
        "{",
        '    over "Scenario"',
        "    {",
        '        over "Looks"',
        "        {",
        BLACK_MATERIAL,
        "        }",
        "",
        black_van(),
        "",
        fake_person(),
        "    }",
        "",
        "    # 假人在 x=62、黑车在 x=120；起点前移到 x=40，两个目标都在前方。",
        '    over "CityFlight"',
        "    {",
        "        double3 xformOp:translate = (40, -22, 6.5)",
        "    }",
        "}",
        "",
    ])


def thermal_doc() -> dict:
    base = yaml.safe_load((CONFIGS / "city_200m_thermal_scenario.yaml").read_text(encoding="utf-8"))
    targets = dict(base["targets"])
    targets.update({
        "black_car_warm": {"mode": "fixed", "temperature_c": 35.0},
        "mannequin": {"mode": "fixed", "temperature_c": 26.0},
    })
    bindings = list(base["bindings"])
    counter = 0

    def add(prim, target, material, children=False):
        nonlocal counter
        counter += 1
        bindings.append({"name": f"decoy_{counter:04d}", "prim_path": prim,
                         "include_children": children, "target": target,
                         "material": material})

    add("/World/Scenario/BlackVan_01", "black_car_warm", "car_paint_black", children=True)
    add("/World/Scenario/BlackVan_01/Hood", "black_car_warm", "car_paint_black")
    add("/World/Scenario/BlackVan_01/Windshield", "glass", "glass_windshield")
    for part in ("CC_Base_Body", "CC_Base_Eye", "V_Neck_Scrub_Shirt", "Field_Jacket",
                 "Cargo_pant", "Basketball_Shoes", "Baseball_Cap"):
        add(f"/World/Scenario/FakePerson/Rig/Body/{part}", "mannequin", "human_skin"
            if part.startswith("CC_Base") else "cotton_clothing")
    return {"schema_version": 1,
            "defaults": dict(base["defaults"]),
            "targets": targets, "bindings": bindings}


SCENE = """\
# 出题版 + 两个 M3 道具：黑色车（余热）与假人（无体温）
schema_version: 1

scene:
  stage: ../city_200m_decoys.usda
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
  thermal_config: city_200m_thermal_decoys.yaml
  preview_fast: true
  align_thermal_to_rgb: true
  flat_field_enabled: true
  sensor_seed: 741
  display_config: city_200m_display_scenario.yaml
  material_directory: materials/city_200m
"""

TASK_BLACK = """\
# M3/控制集：黑色车刚熄火，RGB 用颜色定位，IR 确认余热。
task_id: city-black-car
instruction: 街边停着一辆黑色面包车，刚熄火不久，找到它。
target_prim_path: /World/Scenario/BlackVan_01
target_description: 黑色面包车，车身和引擎盖还带着余热（35 ℃）
environment_description: 夜间城市主街，三辆白色面包车与一辆黑色面包车沿街停放
environment_landmarks:
  - 主街北侧并排的白色面包车
  - 黑色面包车在它们东侧
start_position: [40.0, -22.0, 6.5]
target_position: [120.0, -21.6, 0.9]
target_position_semantics: 黑色车包围盒中心，米制、Z-up、世界坐标
notes: 车身 35 ℃、白车 26 ℃、热引擎盖 65 ℃，三个温度档位
"""

TASK_FAKE = """\
# M3：人形目标里有真有假；假人没有体温，RGB 只能看到人形。
task_id: city-fake-person
instruction: 街边的人形目标里有一个是服装店假人，找到那个假人。
target_prim_path: /World/Scenario/FakePerson/Rig/Body
target_description: 人形假人，全身 26 ℃，没有体温
environment_description: 夜间人行道上有四个人形目标，其中一个是假人
environment_landmarks:
  - 人行道上分散站立的人形目标
start_position: [40.0, -22.0, 6.5]
target_position: [62.0, -27.0, 0.9]
target_position_semantics: 假人包围盒中心，米制、Z-up、世界坐标
notes: 三个真人 33/33/28 ℃，假人 26 ℃；只有 IR 能区分"冷"和"没体温"，RGB 只能确认人形
"""


def write_all() -> None:
    STAGE.write_text(stage_text(), encoding="utf-8")
    (CONFIGS / "city_200m_decoys_scene.yaml").write_text(SCENE, encoding="utf-8")
    doc = thermal_doc()
    header = ("# 由 scene_intake/add_city_200m_decoys.py 生成：出题版热配置 + 黑色车/假人。\n")
    (CONFIGS / "city_200m_thermal_decoys.yaml").write_text(
        header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                                default_flow_style=False, width=200), encoding="utf-8")
    (CONFIGS / "city_200m_task_black_car.yaml").write_text(TASK_BLACK, encoding="utf-8")
    (CONFIGS / "city_200m_task_fake_person.yaml").write_text(TASK_FAKE, encoding="utf-8")


def check() -> int:
    sys.path.insert(0, str(PROJECT))
    from scripts.scene_config import load_scene_config
    from scripts.thermal_config import load_thermal_config

    failures = []
    try:
        config = load_scene_config(CONFIGS / "city_200m_decoys_scene.yaml")
        print(f"OK scene   {config.scene.stage}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"scene: {exc}")
    try:
        thermal = load_thermal_config(CONFIGS / "city_200m_thermal_decoys.yaml")
        print(f"OK thermal targets={len(thermal.targets)} bindings={len(thermal.bindings)}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"thermal: {exc}")
    for item in failures:
        print("FAIL " + item)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    write_all()
    print(f"WROTE {STAGE}")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
