"""给夜间城市 200 m 补上"能出题"的内容：候选、热源、碰撞。

转换后的 `city_200m_local` 是作者原样：1088 个 Mesh、温度 24–28 ℃、没有热源、
没有碰撞体。这个脚本只加一个叠加层和配套配置，不改原场景、不改既有脚本：

  * 3 辆同款白色面包车（同一套几何，只有中间那辆引擎盖 65 ℃）      → S1/N3 候选
  * 2 个新增行人（一个正常 33 ℃、一个偏低 28 ℃）                    → S1/N5 候选
  * 3 台落地空调外机（50 ℃）+ 1 个地面排风口（40 ℃）                → 热源/干扰
  * 地面碰撞 + 6 段街区的不可见碰撞盒                                → 撞楼不再穿模
  * 场景/热/显示/两条任务配置

用法：
    python scene_intake\\add_city_200m_scenario.py
    python scene_intake\\add_city_200m_scenario.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT = Path(r"D:\IsaacSim_Demo")
CONFIGS = PROJECT / "configs"
LOCAL = PROJECT / "city_200m_local"
SCENARIO_STAGE = PROJECT / "city_200m_scenario.usda"

GROUND_Z = -0.2974635
VANS = [  # x, y, 引擎盖温度
    (84.0, -19.9, 26.0),
    (96.0, -21.8, 65.0),
    (108.0, -19.9, 26.0),
]
PERSONS = [  # name, x, y, rotate_z, 冷?
    ("Person_02", 28.0, -27.0, 0.0, False),
    ("Person_03", 135.0, -27.0, 180.0, True),
]
# 原始人物包围盒中心，用于把复制体挪到原点再放到目标位置
PERSON_SRC = (105.0, -27.0, 0.0)
PERSON_PARTS = [  # 子 Mesh 名, 正常目标, 冷目标, 材质
    ("CC_Base_Body", "person_skin", "cold_skin", "human_skin"),
    ("CC_Base_Eye", "person_eyes", "cold_eyes", "human_skin"),
    ("V_Neck_Scrub_Shirt", "person_shirt", "cold_clothes", "cotton_clothing"),
    ("Field_Jacket", "person_jacket", "cold_clothes", "polyester_clothing"),
    ("Cargo_pant", "person_pants", "cold_pants", "cotton_clothing"),
    ("Basketball_Shoes", "person_shoes", "cold_shoes", "shoe_rubber"),
    ("Baseball_Cap", "person_cap", "cold_cap", "cotton_clothing"),
]
AC_UNITS = [(70.0, -19.2), (100.0, -19.2), (130.0, -19.2)]
VENT = (55.0, -19.2)

MATERIALS = {
    "paint_white": ("(0.78, 0.79, 0.80)", 0.45),
    "paint_grey": ("(0.32, 0.33, 0.34)", 0.60),
    "glass_dark": ("(0.04, 0.05, 0.06)", 0.12),
    "rubber": ("(0.03, 0.03, 0.035)", 0.85),
}


def cube(name: str, center, size, material: str, *, invisible: bool = False,
         collision: bool = False, indent: int = 8) -> str:
    pad = " " * indent
    api = ' (\n{p}    prepend apiSchemas = ["PhysicsCollisionAPI"]\n{p})'.format(p=pad) if collision else ""
    lines = [
        f'{pad}def Cube "{name}"{api}',
        f"{pad}{{",
        f"{pad}    double size = 1",
        f"{pad}    double3 xformOp:scale = ({size[0]}, {size[1]}, {size[2]})",
        f"{pad}    double3 xformOp:translate = ({center[0]}, {center[1]}, {center[2]})",
        f'{pad}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
    ]
    if invisible:
        lines.append(f'{pad}    token visibility = "invisible"')
    else:
        lines.append(f"{pad}    rel material:binding = </World/Scenario/Looks/{material}>")
    lines.append(f"{pad}}}")
    return "\n".join(lines)


def cylinder(name: str, center, radius: float, height: float, material: str,
             indent: int = 8) -> str:
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


def van(name: str, x: float, y: float, hot: bool) -> str:
    z = GROUND_Z
    lines = [
        f'        def Xform "{name}"',
        "        {",
        f"            double3 xformOp:translate = ({x}, {y}, {z})",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        "",
        cube("Body", (0, 0, 1.15), (4.6, 1.95, 1.9), "paint_white", indent=12,
             collision=True),
        cube("Windshield", (2.02, 0, 2.0), (0.10, 1.78, 0.55), "glass_dark", indent=12),
        cube("Hood", (1.72, 0, 2.17), (1.15, 1.86, 0.14), "paint_white", indent=12),
        cube("Bumper", (2.33, 0, 0.55), (0.10, 1.9, 0.35), "paint_grey", indent=12),
        cylinder("Wheel_FL", (1.45, 0.98, 0.35), 0.35, 0.24, "rubber", indent=12),
        cylinder("Wheel_FR", (1.45, -0.98, 0.35), 0.35, 0.24, "rubber", indent=12),
        cylinder("Wheel_RL", (-1.45, 0.98, 0.35), 0.35, 0.24, "rubber", indent=12),
        cylinder("Wheel_RR", (-1.45, -0.98, 0.35), 0.35, 0.24, "rubber", indent=12),
        "        }",
    ]
    return "\n".join(lines)


def person(name: str, x: float, y: float, rotate_z: float, cold: bool) -> str:
    import math

    cx, cy = PERSON_SRC[0], PERSON_SRC[1]
    angle = math.radians(rotate_z)
    rx = cx * math.cos(angle) - cy * math.sin(angle)
    ry = cx * math.sin(angle) + cy * math.cos(angle)
    # xformOpOrder = translate, rotate：点先绕 Rig 原点旋转再平移。
    # 要让旋转绕人物中心 c，平移量取 c - R·c；根节点再把人放到目标位置。
    tx, ty = cx - rx, cy - ry
    overrides = ""
    if cold:
        temps = {"CC_Base_Body": 28.0, "CC_Base_Eye": 28.0, "V_Neck_Scrub_Shirt": 26.0,
                 "Field_Jacket": 26.0, "Cargo_pant": 25.5, "Basketball_Shoes": 24.0,
                 "Baseball_Cap": 26.0}
        overrides = "\n".join(
            f'                    over "{part}"\n'
            "                    {\n"
            f"                        custom float thermal:temperatureC = {temp}\n"
            "                    }"
            for part, temp in temps.items())
    return "\n".join([
        f'        def Xform "{name}"',
        "        {",
        f"            double3 xformOp:translate = ({x - cx:.6f}, {y - cy:.6f}, 0)",
        '            uniform token[] xformOpOrder = ["xformOp:translate"]',
        '            def Xform "Rig"',
        "            {",
        f"                double3 xformOp:translate = ({tx:.6f}, {ty:.6f}, 0)",
        f"                float3 xformOp:rotateXYZ = (0, 0, {rotate_z})",
        '                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]',
        '                def "Body" (',
        "                    prepend references = @./city_200m_local/city_200m.usda@</World/Targets/Person_01>",
        "                )",
        "                {",
        overrides,
        "                }",
        "            }",
        "        }",
    ])


def scenario_usda() -> str:
    materials = []
    for name, (color, roughness) in MATERIALS.items():
        materials.append(
            f'        def Material "{name}"\n'
            "        {\n"
            f'            token outputs:surface.connect = </World/Scenario/Looks/{name}/Surface.outputs:surface>\n'
            f'            def Shader "Surface"\n'
            "            {\n"
            '                uniform token info:id = "UsdPreviewSurface"\n'
            f"                color3f inputs:diffuseColor = {color}\n"
            f"                float inputs:roughness = {roughness}\n"
            "                token outputs:surface\n"
            "            }\n"
            "        }"
        )

    props = [van(f"Van_{index + 1:02d}", x, y, hot) for index, (x, y, hot) in enumerate(VANS)]
    for index, (x, y) in enumerate(AC_UNITS, start=1):
        props.append(cube(f"ACUnit_{index:02d}", (x, y, 0.42), (1.0, 0.5, 0.84),
                          "paint_grey", indent=8))
    props.append(cube("Vent", (VENT[0], VENT[1], 0.10), (1.6, 0.8, 0.2),
                      "paint_grey", indent=8))
    props.extend(person(name, x, y, rot, cold) for name, x, y, rot, cold in PERSONS)

    colliders = []
    for index in range(6):
        colliders.append(cube(f"Collider_Block_{index:02d}",
                              (2.7 + 45.0 * index, 1.65, 7.0), (42.0, 39.1, 14.0),
                              "paint_grey", invisible=True, collision=True, indent=8))

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
        "        @./city_200m_flight.usda@",
        "    ]",
        "    customLayerData = {",
        '        string comment = "City scenario: 3 candidate vans, 3 people, heat sources, collision proxies, lamp groups."',
        "    }",
        ")",
        'over "World"',
        "{",
        '    def Xform "Scenario"',
        "    {",
        '        def "Looks"',
        "        {",
        "\n".join(materials),
        "        }",
        "",
        "\n\n".join(props),
        "    }",
        "",
        "    # 街区粗碰撞与道具分开：episode 叠加层只平移 Scenario，不会把碰撞盒带偏。",
        '    def "ScenarioCollision"',
        "    {",
        "\n".join(colliders),
        "    }",
        "",
        "    # 地面参与碰撞：无人机落下来/擦地时不会直接穿过去。",
        '    over "Ground" (',
        '        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]',
        "    )",
        "    {",
        '        token physics:approximation = "none"',
        "    }",
        "",
        lamp_overrides(),
        "}",
        "",
    ])


def lamp_overrides() -> str:
    """给 336 盏路灯标上分组（按街区奇偶分 A/B），方便后续按组开关。"""
    blocks = []
    for block in range(6):
        group = "A" if block % 2 == 0 else "B"
        lamps = []
        for lamp in range(56):
            lamps.append(
                f'            over "Lamp_{lamp:03d}"\n'
                "            {\n"
                f'                custom string city:lamp_group = "{group}"\n'
                f"                custom int city:lamp_block = {block}\n"
                "            }")
        blocks.append(
            f'        over "Block_{block:02d}"\n'
            "        {\n" + "\n".join(lamps) + "\n        }")
    return '    over "Lighting"\n    {\n' + "\n".join(blocks) + "\n    }"


def thermal_config() -> dict:
    base = yaml.safe_load((CONFIGS / "city_200m_thermal_flight.yaml").read_text(encoding="utf-8"))
    targets = dict(base["targets"])
    targets.update({
        "van_hood_hot": {"mode": "fixed", "temperature_c": 65.0},
        "van_hood_cold": {"mode": "fixed", "temperature_c": 26.0},
        "ac_unit_on": {"mode": "fixed", "temperature_c": 60.0},
        "ac_unit_mid": {"mode": "fixed", "temperature_c": 51.0},
        "ac_unit_off": {"mode": "fixed", "temperature_c": 29.0},
        "lamp_bulb": {"mode": "fixed", "temperature_c": 90.0},
        "glass": {"mode": "fixed", "temperature_c": 33.0},
        "vent": {"mode": "fixed", "temperature_c": 40.0},
        "cold_skin": {"mode": "fixed", "temperature_c": 28.0},
        "cold_eyes": {"mode": "fixed", "temperature_c": 28.0},
        "cold_clothes": {"mode": "fixed", "temperature_c": 26.0},
        "cold_pants": {"mode": "fixed", "temperature_c": 25.5},
        "cold_shoes": {"mode": "fixed", "temperature_c": 24.0},
        "cold_cap": {"mode": "fixed", "temperature_c": 26.0},
    })
    bindings = list(base["bindings"])
    counter = 0

    def add(prim, target, material, children=False):
        nonlocal counter
        counter += 1
        bindings.append({"name": f"scenario_{counter:04d}", "prim_path": prim,
                         "include_children": children, "target": target,
                         "material": material})

    for index, (x, y, hot) in enumerate(VANS, start=1):
        root = f"/World/Scenario/Van_{index:02d}"
        add(root, "van_hood_cold", "car_paint_white", children=True)
        add(root + "/Hood", "van_hood_hot" if hot > 30 else "van_hood_cold", "car_paint_white")
        add(root + "/Windshield", "glass", "glass_windshield")
    for index, target in enumerate(("ac_unit_on", "ac_unit_off", "ac_unit_mid"), start=1):
        add(f"/World/Scenario/ACUnit_{index:02d}", target, "painted_composite")
    add("/World/LampBulbs", "lamp_bulb", "painted_composite", children=True)
    add("/World/Scenario/Vent", "vent", "rusted_steel")

    for name, x, y, rot, cold in PERSONS:
        root = f"/World/Scenario/{name}/Rig/Body"
        for part, normal_target, cold_target, material in PERSON_PARTS:
            add(f"{root}/{part}", cold_target if cold else normal_target, material)
    return {"schema_version": 1,
            "defaults": {"target": "unknown", "material": "concrete"},
            "targets": targets, "bindings": bindings}


SCENE_TEMPLATE = """\
# 夜间城市 200 m 的"出题版"：3 辆同款白车 + 3 个人 + 空调/排风热源 + 碰撞体。
# 叠加链：city_200m_scenario.usda -> city_200m_flight.usda -> city_200m_local/city_200m.usda
schema_version: 1

scene:
  stage: ../city_200m_scenario.usda
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
  thermal_config: city_200m_thermal_scenario.yaml
  preview_fast: true
  align_thermal_to_rgb: true
  flat_field_enabled: true
  sensor_seed: 741
  display_config: city_200m_display_scenario.yaml
  material_directory: materials/city_200m
"""

TASK_N3 = """\
# S1/N3：三辆同款白车，只有一辆引擎盖还热。外观在 RGB 里完全一致，热像给出唯一交集。
task_id: city-n3-warm-engine
instruction: 街边停着三辆同款白色面包车，找到发动机刚熄火、引擎盖还热的那一辆。
target_prim_path: /World/Scenario/Van_02/Hood
target_description: 三辆同款白色面包车中，引擎盖温度明显偏高的那一辆
environment_description: 夜间城市主街南侧，三辆白色面包车平行停放，附近有空调外机和地面排风口
environment_landmarks:
  - 主街北侧三辆并排的白色面包车
  - 路缘上的空调外机
start_position: [70.0, -22.0, 6.5]
target_position: [97.7, -21.8, 2.2]
target_position_semantics: 目标车辆引擎盖中心，米制、Z-up、世界坐标
notes: 三辆车身都是 26 ℃，只有中间一辆引擎盖 65 ℃；空调外机 50 ℃ 是干扰热源
"""

TASK_N5 = """\
# S1/N5：三个人外观完全一致，只有一个体温偏低（28 ℃）。RGB 无法区分，热像可区分。
task_id: city-n5-cold-person
instruction: 街上有三个人，找到体温明显偏低、可能已经失温的那一个。
target_prim_path: /World/Scenario/Person_03/Rig/Body
target_description: 三名同款行人中，皮肤温度只有 28 ℃ 的那一个
environment_description: 夜间城市人行道，三名行人分散站立，附近的空调外机和排风口也有热辐射
environment_landmarks:
  - 人行道上分散站立的三名行人
  - 路缘上的空调外机
start_position: [70.0, -22.0, 6.5]
target_position: [135.0, -27.0, 0.9]
target_position_semantics: 目标人物包围盒中心，米制、Z-up、世界坐标
notes: Person_02 正常 33 ℃；Person_03 皮肤 28 ℃、衣物 24–26 ℃；三人模型与姿态一致
"""


BAT_ENV = """\
setlocal
set ISAAC=D:\\isaacsim
set PROJECT=D:\\IsaacSim_Demo
set USDROOT=%ISAAC%\\extscache\\omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312
set PXR_USD_WINDOWS_DLL_PATH=%ISAAC%\\kit;%ISAAC%\\kit\\bin;%ISAAC%\\kit\\python;%USDROOT%\\bin
set PYTHONPATH=%PROJECT%;%PROJECT%\\third_party\\isaac-thermal-camera-simulation\\src
set WARP_CACHE_PATH=%PROJECT%\\validation\\isaac_cache\\warp
set OMNI_WARP_CACHE_PATH=%WARP_CACHE_PATH%
rem 机体 payload（cf2x.usd）已缓存到本机；换机器时先设一次 HTTP_PROXY/HTTPS_PROXY，
rem 让 Isaac 能拉取 https S3 资源，缓存后可去掉。 
set PATH=%ISAAC%\\kit;%ISAAC%\\kit\\bin;%ISAAC%\\kit\\python;C:\\Windows\\System32;C:\\Windows;C:\\Windows\\System32\\Wbem
"""

BAT_STATIC = """\
@echo off
rem 出题版静态取图（3 辆白车 + 3 个人 + 空调/排风热源 + 碰撞体）
rem 用法: run_city_scenario_capture.bat [out-name] [config]
rem 默认 configs\\city_200m_scenario_scene.yaml；也可传 decoys / day / dark / epNNN 的配置
""" + BAT_ENV + """\
set RUNID=%1
if "%RUNID%"=="" set RUNID=city-scenario-static-%RANDOM%
set CONFIG=%2
if "%CONFIG%"=="" set CONFIG=configs\\city_200m_scenario_scene.yaml
cd /d %PROJECT%
"%ISAAC%\\python.bat" -B "scripts\\capture_crazyflie_thermal_static.py" ^
  --config "%CONFIG%" ^
  --out "%PROJECT%\\validation\\%RUNID%"
echo.
echo output: %PROJECT%\\validation\\%RUNID%
"""

BAT_FLIGHT = """\
@echo off
rem 出题版飞一小段 + 高分辨率回放；可带一条任务文件（默认 N3）
rem 用法: run_city_scenario_flight.bat [run-id] [task-file] [frames] [speed]
""" + BAT_ENV + """\
set RUNID=%1
if "%RUNID%"=="" set RUNID=city-scenario-flight-%RANDOM%
set TASKFILE=%2
if "%TASKFILE%"=="" set TASKFILE=configs\\city_200m_task_n3.yaml
set FRAMES=%3
if "%FRAMES%"=="" set FRAMES=180
set SPEED=%4
if "%SPEED%"=="" set SPEED=3.0
cd /d %PROJECT%
echo [1/2] record trajectory with task %TASKFILE%
"%ISAAC%\\python.bat" -B "scripts\\start_crazyflie_dual_camera_view.py" ^
  --config "configs\\city_200m_scenario_scene.yaml" --headless --frames %FRAMES% ^
  --record-trajectory --run-id "%RUNID%" --probe-control --horizontal-speed %SPEED% ^
  --task-file "%TASKFILE%"
if errorlevel 1 exit /b %errorlevel%
echo [2/2] replay high resolution
"%ISAAC%\\python.bat" -X utf8 "scripts\\replay_trajectory_highres.py" ^
  --config "configs\\city_200m_scenario_scene.yaml" ^
  --trajectory "validation\\trajectory_runs\\%RUNID%\\trajectory.jsonl" ^
  --out "validation\\%RUNID%" --rgb-width 640 --rgb-height 512 --thermal-width 320 --thermal-height 256
if errorlevel 1 exit /b %errorlevel%
echo.
echo trajectory: %PROJECT%\\validation\\trajectory_runs\\%RUNID%
echo frames:     %PROJECT%\\validation\\%RUNID%
"""


LIGHT_VARIANT = """\
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 2400
    timeCodesPerSecond = 24
    framesPerSecond = 24
    subLayers = [
        @./city_200m_scenario.usda@
    ]
    customLayerData = {{
        string comment = "{comment}"
    }}
)
over "World"
{{
    # 只改天光强度做光照档；路灯仍开着，白天相对天光可以忽略。
    over "Sky"
    {{
        float inputs:intensity = {intensity}
        color3f inputs:color = {color}
    }}
}}
"""

LIGHT_SCENE_TEMPLATE = """\
# {comment}
schema_version: 1

scene:
  stage: ../{stage}
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
  thermal_config: city_200m_thermal_scenario.yaml
  preview_fast: true
  align_thermal_to_rgb: true
  flat_field_enabled: true
  sensor_seed: 741
  display_config: city_200m_display_scenario.yaml
  material_directory: materials/city_200m
"""


LAMP_VARIANT = """\
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 2400
    timeCodesPerSecond = 24
    framesPerSecond = 24
    subLayers = [
        @./city_200m_scenario.usda@
    ]
    customLayerData = {{
        string comment = "{comment}"
    }}
)
over "World"
{{
{overrides}
}}
"""


def lamp_off_overrides(group: str) -> str:
    """把某一组路灯的 intensity 设为 0，做“路灯可分组开关”的光照档。"""
    blocks = []
    for block in range(6):
        if ("A" if block % 2 == 0 else "B") != group:
            continue
        lamps = []
        for lamp in range(56):
            lamps.append(
                f'            over "Lamp_{lamp:03d}"\n'
                "            {\n"
                "                float inputs:intensity = 0\n"
                "            }")
        blocks.append(
            f'        over "Block_{block:02d}"\n'
            "        {\n" + "\n".join(lamps) + "\n        }")
    return '    over "Lighting"\n    {\n' + "\n".join(blocks) + "\n    }"


def write_light_variants() -> None:
    day = LIGHT_VARIANT.format(comment="Scenario at L0 daylight (dome light only).",
                               intensity=1400, color="(1, 1, 1)")
    dark = LIGHT_VARIANT.format(comment="Scenario at L3 near-dark (RGB nearly blind).",
                                intensity=40, color="(0.55, 0.62, 0.8)")
    (PROJECT / "city_200m_day.usda").write_text(day, encoding="utf-8")
    (PROJECT / "city_200m_dark.usda").write_text(dark, encoding="utf-8")
    (CONFIGS / "city_200m_day_scene.yaml").write_text(LIGHT_SCENE_TEMPLATE.format(
        comment="L0 白天对照：天光 1400，热/道具/碰撞与出题版一致",
        stage="city_200m_day.usda"), encoding="utf-8")
    (CONFIGS / "city_200m_dark_scene.yaml").write_text(LIGHT_SCENE_TEMPLATE.format(
        comment="L3 近黑：天光 40，RGB 几乎失效，红外仍可用，用于 M1/D5 降级",
        stage="city_200m_dark.usda"), encoding="utf-8")

    # 路灯分组开关：A 组 = 0/2/4 段，B 组 = 1/3/5 段
    for group in ("A", "B"):
        stage = f"city_200m_lamps_{group.lower()}_off.usda"
        (PROJECT / stage).write_text(LAMP_VARIANT.format(
            comment=f"Scenario with lamp group {group} switched off.",
            overrides=lamp_off_overrides(group)), encoding="utf-8")
        (CONFIGS / f"city_200m_lamps_{group.lower()}_off_scene.yaml").write_text(
            LIGHT_SCENE_TEMPLATE.format(
                comment=f"路灯 {group} 组（{'0/2/4' if group == 'A' else '1/3/5'} 段）关闭，其余不变",
                stage=stage), encoding="utf-8")


def write_all() -> dict:
    SCENARIO_STAGE.write_text(scenario_usda(), encoding="utf-8")
    (CONFIGS / "city_200m_scenario_scene.yaml").write_text(SCENE_TEMPLATE, encoding="utf-8")
    (PROJECT / "run_city_scenario_capture.bat").write_text(BAT_STATIC, encoding="utf-8")
    (PROJECT / "run_city_scenario_flight.bat").write_text(BAT_FLIGHT, encoding="utf-8")
    write_light_variants()

    doc = thermal_config()
    header = (
        "# 由 scene_intake/add_city_200m_scenario.py 生成：飞行版热配置 + 出题道具。\n"
        "# 新增：面包车引擎盖 65/26 ℃、空调外机 50 ℃、排风口 40 ℃、新增行人 33/28 ℃。\n"
    )
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                          default_flow_style=False, width=200)
    (CONFIGS / "city_200m_thermal_scenario.yaml").write_text(header + body, encoding="utf-8")

    (CONFIGS / "city_200m_display_scenario.yaml").write_text(
        "# 出题版显示：外墙/地面 24 ℃ 左右保留层次，窗户 33 ℃ 偏亮，引擎盖 65 ℃、空调 60 ℃、灯头 90 ℃ 顶部饱和。\n"
        "schema_version: 1\n"
        "name: city_200m_scenario\n"
        "mode: fixed_apparent_temperature\n"
        "temperature_min_c: 18.0\n"
        "temperature_max_c: 42.0\n"
        "dde_gain: 0.0\n"
        "polarity: white_hot\n"
        "palette: gray\n", encoding="utf-8")

    (CONFIGS / "city_200m_task_n3.yaml").write_text(TASK_N3, encoding="utf-8")
    (CONFIGS / "city_200m_task_n5.yaml").write_text(TASK_N5, encoding="utf-8")

    return {"stage": str(SCENARIO_STAGE), "bindings": len(doc["bindings"]),
            "targets": len(doc["targets"])}


def check() -> int:
    sys.path.insert(0, str(PROJECT))
    from scripts.scene_config import load_scene_config
    from scripts.thermal_config import load_thermal_config

    failures = []
    try:
        config = load_scene_config(CONFIGS / "city_200m_scenario_scene.yaml")
        print(f"OK scene   stage={config.scene.stage}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"scene: {exc}")
    try:
        thermal = load_thermal_config(CONFIGS / "city_200m_thermal_scenario.yaml")
        print(f"OK thermal targets={len(thermal.targets)} bindings={len(thermal.bindings)}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"thermal: {exc}")
    for name in ("city_200m_task_n3.yaml", "city_200m_task_n5.yaml"):
        try:
            text = (CONFIGS / name).read_text(encoding="utf-8")
            data = yaml.safe_load(text)
            for key in ("task_id", "instruction", "target_prim_path", "start_position",
                        "target_position"):
                if key not in data:
                    raise ValueError(f"missing {key}")
            print(f"OK task    {name}: {data['task_id']}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
    for item in failures:
        print("FAIL " + item)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    result = write_all()
    print(f"WROTE {result['stage']}")
    print(f"      bindings={result['bindings']} targets={result['targets']}")
    print(f"      configs city_200m_{{scenario_scene,thermal_scenario,display_scenario,task_n3,task_n5}}.yaml")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
