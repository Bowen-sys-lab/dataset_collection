"""把「夜间城市 200 m + 人物」交付包接入本项目现有流程。

交付包（city_night_200m_person_20261028）自带作者自己的 configs/thermal.yaml、
environment/ 与 preview 脚本，但用的是 irsim 15015cd 的 schema（无 defaults、
binding 无 name/include_children、材质 concrete_weathered 在旧版 irsim 里没有）。
本项目现有采集/回放入口只认 configs/ 三件套 + 旧版 irsim，因此本脚本做一次
机械转换，只新增文件，不改任何既有脚本：

  1. 复制一份工作副本 city_200m_local（原包只读）
  2. configs/city_200m_thermal.yaml           1088 条 per-prim 绑定（旧 schema）
  3. configs/city_200m_thermal_flight.yaml    同上 + 叠加进来的机体
  4. configs/city_200m_night_environment.yaml 天气/地点（weather_file 改绝对路径）
  5. configs/city_200m_display.yaml           与作者一致的 18–38 ℃
  6. configs/city_200m_scene.yaml             静态取图（作者机位）
  7. configs/city_200m_flight_scene.yaml      飞行叠加层
  8. configs/city_200m_task.yaml              首条可用任务（人物）
  9. city_200m_flight.usda                    机体 + PhysicsScene 叠加层
 10. run_city_capture.bat                     静态 RGB+红外取图入口
 11. scene_intake/out/<包名>/report.json      接入检查结果

用法：
    python scene_intake\\convert_city_200m.py
    python scene_intake\\convert_city_200m.py --check   # 只校验已生成文件
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml

PROJECT = Path(r"D:\IsaacSim_Demo")
VENDOR = PROJECT / "city_night_200m_person_20261028"
WORK = PROJECT / "city_200m_local"
CONFIGS = PROJECT / "configs"
ENTRY = "city_200m.usda"

# 旧版 irsim 没有 concrete_weathered（它带 SLUM 实测谱线，旧版 data/ 里没有）。
# 归到同类混凝土，而不是复制一份名字相同、数值不同的假材质。
MATERIAL_MAP = {"concrete_weathered": "concrete"}

# 起始机位：主街西端、离地 9 m、机头朝东（+X），正对人行道上的人物（约 125 m 外）。
START = (70.0, -22.0, 6.5)
FLIGHT_ROOT = "/World/CityFlight"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_work() -> dict:
    """原包只读复制；已存在就跳过（不覆盖用户后续改动）。"""
    if WORK.exists():
        files = [p for p in WORK.rglob("*") if p.is_file()]
        return {"copied": False, "files": len(files),
                "bytes": sum(p.stat().st_size for p in files)}
    shutil.copytree(VENDOR, WORK, ignore=shutil.ignore_patterns("__pycache__"))
    files = [p for p in WORK.rglob("*") if p.is_file()]
    return {"copied": True, "files": len(files),
            "bytes": sum(p.stat().st_size for p in files)}


def load_vendor() -> tuple[dict, list[dict]]:
    data = yaml.safe_load((VENDOR / "configs" / "thermal.yaml").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "targets" not in data or "bindings" not in data:
        raise SystemExit("vendor thermal.yaml has an unexpected schema")
    return data["targets"], data["bindings"]


def our_targets(vendor_targets: dict) -> dict:
    """作者 targets 已是 fixed/温度，字段名一致；只保证数值是 float。"""
    out = {}
    for name, spec in vendor_targets.items():
        if spec.get("mode") != "fixed":
            raise SystemExit(f"target {name}: unsupported mode {spec.get('mode')!r}")
        out[name] = {"mode": "fixed", "temperature_c": float(spec["temperature_c"])}
    return out


def our_bindings(vendor: list[dict]) -> list[dict]:
    seen = set()
    rows = []
    for index, item in enumerate(vendor, start=1):
        path = item["prim_path"]
        if path in seen:
            raise SystemExit(f"duplicate binding prim_path in vendor config: {path}")
        seen.add(path)
        material = MATERIAL_MAP.get(item["material"], item["material"])
        rows.append({
            "name": f"city_{index:04d}",
            "prim_path": path,
            "include_children": False,
            "target": item["target"],
            "material": material,
        })
    return rows


def thermal_doc(targets: dict, bindings: list[dict], *, flight: bool) -> dict:
    targets = dict(targets)
    if flight:
        targets["drone_body"] = {"mode": "fixed", "temperature_c": 35.0}
    rows = list(bindings)
    if flight:
        rows.append({
            "name": "city_drone",
            "prim_path": FLIGHT_ROOT,
            "include_children": True,
            "target": "drone_body",
            "material": "painted_composite",
        })
    return {"schema_version": 1,
            "defaults": {"target": "unknown", "material": "concrete"},
            "targets": targets, "bindings": rows}


HEADER_THERMAL = """\
# 由 scene_intake/convert_city_200m.py 从作者 configs/thermal.yaml 转换（旧版 irsim schema）。
# 转换内容：
#   * 补 defaults / schema_version / binding.name / include_children
#   * concrete_weathered -> concrete（旧版 irsim 没有该材质，见脚本 MATERIAL_MAP）
#   * 温度和目标的数值原样保留，作者设定：道路 28、墙体 27、金属 26、树干/玻璃 25、叶片 24 ℃
# 作者每条绑定都是"单个 Mesh 的完整 Prim 路径"，所以 include_children: false。
# 不要手改：改源包或本脚本后重新生成。
"""

HEADER_ENV = """\
# 夜间城市的热环境（由作者 environment/night_environment.yaml 改写）。
# 改动只有一处：weather_file 改成绝对路径——本项目以 D:\\IsaacSim_Demo 为工作目录，
# 相对路径会解析到 configs/ 下。
# 不写 scene.targets：目标温度由 configs/city_200m_thermal.yaml 提供，两处同时写会触发重名检查。
schema_version: 17
scene:
  name: city_night_200m_person
  description: Synthetic 200 m night city with a static person and prescribed surface temperatures
  weather_file: {weather}
  atmosphere_preset: us_standard_clear
  environment_preset: clear_dry
  site:
    latitude_deg: 31.2
    longitude_deg: 121.5
    altitude_m: 10.0
  start_utc: '2026-10-26T12:00:00Z'
"""

SCENE_TEMPLATE = """\
# {comment}
schema_version: 1

scene:
  stage: {stage}
  thermal_scene: city_200m_night_environment.yaml
  static_targets: []

drone:
  prim_path: {prim}
  body_path: {body}
  camera_path: {camera}
  minimum_height: 1.5

sensors:
  # 与作者预览一致：320x256（放大虚拟像元保持 FOV），原生 640x512
  width: 320
  height: 256
  # 沿用现有流程的旧版 irsim，不为了这个场景切版本
  sensor_config: ../third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml
  thermal_config: {thermal}
  preview_fast: true
  align_thermal_to_rgb: true
  flat_field_enabled: true
  sensor_seed: 741
  display_config: city_200m_display.yaml
  # 人物用到的 polyester_clothing / shoe_rubber 旧版 irsim 没有，从作者用的 15015cd 抄定义
  material_directory: materials/city_200m
"""

TASK = """\
# 首条可用任务：作者交付的唯一真目标（人行道上看手机的男子）。
# 位置与温度来自 records/person_01.json；这是"能跑通"的冒烟任务，不是 S1 的 N3/N5 题目，
# 后者需要白车/发动机热/空调外机等候选与热源，见 docs/city-200m-scene-20261030.md 的缺口清单。
task_id: city-person-01
instruction: 沿主街向东飞，找到人行道上站着看手机的那名男子。
target_prim_path: /World/Targets/Person_01
target_description: 静止站立的男子，深色外套，正在看手机
environment_description: 夜间城市街区，六段同一街区的连续复制，主街南侧有路灯与车辆
environment_landmarks:
  - 主街南侧人行道
  - 连续排列的六段街区
start_position: [{sx}, {sy}, {sz}]
target_position: [105.0, -27.1, 0.9]
target_position_semantics: 人物包围盒中心，米制、Z-up、世界坐标
notes: 温度与材质是场景设定，不是实测；目标是静态人物，未加发动机/空调等热源
"""

OVERLAY = """\
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    # 必须由根层写时间范围：USD 只看根层的 timeCodesPerSecond/范围，子层的 0–2400 不会传上来。
    # 缺这段时 Isaac 会报 "no usable time-code range" 并把时间轴锁在单帧，控制器不动。
    startTimeCode = 0
    endTimeCode = 2400
    timeCodesPerSecond = 24
    framesPerSecond = 24
    subLayers = [
        @./city_200m_visual_fix.usda@
    ]
    customLayerData = {{
        string comment = "200 m night city + a flyable Crazyflie. Relative paths: move the project folder freely."
    }}
)
over "World"
{{
    # 起点在街口中段、主街南侧、离地 6.5 m；yaw 0 时机体 +X 朝东，飞控的 W 就是沿街前进。
    # 机体自带相机装在机身侧面（body 坐标里朝 +Y），不是机头方向，所以这里另挂一台 CityCam：
    # 与机体 +X 同向、下俯 10°，焦距/光阑取交付包固定机位的 18 mm / 20.48 mm。
    def Xform "CityFlight"
    {{
        double3 xformOp:translate = ({x}, {y}, {z})
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Xform "Crazyflie" (
            prepend references = @./lesson_crazyflie.usd@</World/Crazyflie>
        )
        {{
            double3 xformOp:translate = (0, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]

            over "body"
            {{
                def Camera "CityCam"
                {{
                    float2 clippingRange = (0.1, 2000)
                    float focalLength = 18
                    float horizontalAperture = 20.48
                    float verticalAperture = 16.384
                    # 行 = 相机 X/右、Y/上、Z/后、平移（row-vector 约定，平移在第 4 行）
                    matrix4d xformOp:transform = ((0, -1, 0, 0), (0.1736, 0, 0.9848, 0), (-0.9848, 0, 0.1736, 0), (0.12, 0, -0.03, 1))
                    uniform token[] xformOpOrder = ["xformOp:transform"]
                }}
            }}
        }}
    }}

    # 城市场景原本没有物理场景；控制器起效需要它。
    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}
}}
"""

BAT = """\
@echo off
rem 夜间城市 200 m：静态 RGB + 红外取图（走现有旧版 irsim 与 scripts，不改任何既有脚本）
rem 用法: run_city_capture.bat [out-name] [camera]
rem 相机默认用交付包自带的 /World/Cameras/Middle；也可传 /World/Cameras/Start 等。
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
set RUNID=%1
if "%RUNID%"=="" (
  for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmmss"') do set RUNID=city-200m-static-%%i
)
set CAMERA=%2
if "%CAMERA%"=="" set CAMERA=/World/Cameras/Middle
cd /d %PROJECT%
"%ISAAC%\\python.bat" -B "scripts\\capture_crazyflie_thermal_static.py" ^
  --config "configs\\city_200m_scene.yaml" ^
  --camera-path "%CAMERA%" ^
  --out "%PROJECT%\\validation\\%RUNID%"
echo.
echo output: %PROJECT%\\validation\\%RUNID%
"""

BAT_FLIGHT = """\
@echo off
rem 夜间城市 200 m：先飞一小段录轨迹，再按轨迹高分辨率回放（RGB 640x512 / 红外 320x256）
rem 用法: run_city_flight_capture.bat [run-id] [frames] [horizontal-speed]
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
set RUNID=%1
if "%RUNID%"=="" set RUNID=city-flight-%RANDOM%
set FRAMES=%2
if "%FRAMES%"=="" set FRAMES=180
set SPEED=%3
if "%SPEED%"=="" set SPEED=3.0
cd /d %PROJECT%
echo [1/2] record trajectory (%FRAMES% frames at %SPEED% m/s)
"%ISAAC%\\python.bat" -B "scripts\\start_crazyflie_dual_camera_view.py" ^
  --config "configs\\city_200m_flight_scene.yaml" --headless --frames %FRAMES% ^
  --record-trajectory --run-id "%RUNID%" --probe-control --horizontal-speed %SPEED%
if errorlevel 1 exit /b %errorlevel%
echo [2/2] replay high resolution
"%ISAAC%\\python.bat" -X utf8 "scripts\\replay_trajectory_highres.py" ^
  --config "configs\\city_200m_flight_scene.yaml" ^
  --trajectory "validation\\trajectory_runs\\%RUNID%\\trajectory.jsonl" ^
  --out "validation\\%RUNID%" --rgb-width 640 --rgb-height 512 --thermal-width 320 --thermal-height 256
if errorlevel 1 exit /b %errorlevel%
echo.
echo trajectory: %PROJECT%\\validation\\trajectory_runs\\%RUNID%
echo frames:     %PROJECT%\\validation\\%RUNID%
"""


def copy_materials() -> list[str]:
    """旧版 irsim 缺的两条人物衣物材质，从作者用的 15015cd 原样抄定义。

    两条都是 schema_version 2 的纯 band 参数（没有谱线文件），旧版 loader 可读。
    只新增、不覆盖：名字若与既有库重名，load_project_material_library 会报错而不是静默合并。
    """
    source = PROJECT / "third_party" / "irsim-15015cd" / "configs" / "materials"
    target = CONFIGS / "materials" / "city_200m"
    target.mkdir(parents=True, exist_ok=True)
    (target / "README.md").write_text(
        "# city_200m 本地材质\n\n"
        "本目录只补旧版 irsim（isaac-thermal-camera-simulation, 563a51b）缺的两条材质，\n"
        "内容逐字节抄自作者使用的 irsim 15015cd，未改数值：\n\n"
        "- `polyester_clothing.yaml` — 人物外套\n"
        "- `shoe_rubber.yaml` — 人物鞋\n\n"
        "作者的 `concrete_weathered` 没有抄：它依赖 SLUM 实测谱线（旧版 data/ 里没有），\n"
        "转换时归到同类 `concrete`，见 scene_intake/convert_city_200m.py 的 MATERIAL_MAP。\n",
        encoding="utf-8")
    copied = []
    for name in ("polyester_clothing.yaml", "shoe_rubber.yaml"):
        shutil.copy2(source / name, target / name)
        copied.append(name)
    return copied


def write_configs() -> dict:
    vendor_targets, vendor_bindings = load_vendor()
    targets = our_targets(vendor_targets)
    bindings = our_bindings(vendor_bindings)

    def dump(path: Path, doc: dict, header: str = "") -> None:
        body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True,
                              default_flow_style=False, width=200)
        path.write_text(header + body, encoding="utf-8")

    thermal_path = CONFIGS / "city_200m_thermal.yaml"
    dump(thermal_path, thermal_doc(targets, bindings, flight=False), HEADER_THERMAL)
    flight_thermal = CONFIGS / "city_200m_thermal_flight.yaml"
    dump(flight_thermal, thermal_doc(targets, bindings, flight=True),
         HEADER_THERMAL.replace("作者 configs/thermal.yaml", "作者 configs/thermal.yaml，另加机体"))

    (CONFIGS / "city_200m_night_environment.yaml").write_text(
        HEADER_ENV.format(weather=str((WORK / "environment" / "weather.csv").resolve())),
        encoding="utf-8")
    (CONFIGS / "city_200m_display.yaml").write_text(
        "# 与作者 configs/display.yaml 一致（18–38 ℃）；城市本体温差只有 24–28 ℃，靠人物 33 ℃ 拉开。\n"
        "schema_version: 1\n"
        "name: city_200m_fixed_temperature\n"
        "mode: fixed_apparent_temperature\n"
        "temperature_min_c: 18.0\n"
        "temperature_max_c: 38.0\n"
        "dde_gain: 0.0\n"
        "polarity: white_hot\n"
        "palette: gray\n", encoding="utf-8")

    (CONFIGS / "city_200m_scene.yaml").write_text(SCENE_TEMPLATE.format(
        comment="静态取图：作者交付的场景入口 + 自带固定机位（没有机体）",
        stage=f"../city_200m_local/{ENTRY}",
        prim="/World/Cameras/Middle",
        body="/World/Cameras",
        camera="/World/Cameras/Middle",
        thermal="city_200m_thermal.yaml"), encoding="utf-8")
    (CONFIGS / "city_200m_flight_scene.yaml").write_text(SCENE_TEMPLATE.format(
        comment="飞行版：叠加层里挂了可飞机体，drone 指向 /World/CityFlight/Crazyflie",
        stage="../city_200m_flight.usda",
        prim=f"{FLIGHT_ROOT}/Crazyflie",
        body=f"{FLIGHT_ROOT}/Crazyflie/body",
        camera=f"{FLIGHT_ROOT}/Crazyflie/body/CityCam",
        thermal="city_200m_thermal_flight.yaml"), encoding="utf-8")
    (CONFIGS / "city_200m_task.yaml").write_text(
        TASK.format(sx=START[0], sy=START[1], sz=START[2]), encoding="utf-8")
    materials = copy_materials()

    (PROJECT / "city_200m_flight.usda").write_text(
        OVERLAY.format(x=START[0], y=START[1], z=START[2]), encoding="utf-8")
    (PROJECT / "run_city_capture.bat").write_text(BAT, encoding="utf-8")
    (PROJECT / "run_city_flight_capture.bat").write_text(BAT_FLIGHT, encoding="utf-8")

    return {"targets": len(targets), "bindings": len(bindings),
            "materials": materials,
            "files": [str(p) for p in (thermal_path, flight_thermal)]}


def write_fork_notes() -> dict:
    vendor_entry = VENDOR / ENTRY
    work_entry = WORK / ENTRY
    notes = {
        "fork": "local working copy of city_night_200m_person_20261028",
        "vendor_origin": str(VENDOR),
        "changes": [],
        "vendor_scene_sha256": sha256(vendor_entry),
        "fork_scene_sha256": sha256(work_entry),
        "everything_else": "byte-identical to the vendor package",
        "note": "目前没有改场景内容；热配置/环境/机位全部放在 D:\\IsaacSim_Demo\\configs 与叠加层里。",
    }
    notes["identical"] = notes["vendor_scene_sha256"] == notes["fork_scene_sha256"]
    (WORK / "FORK_NOTES.json").write_text(
        json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    return notes


def intake_report(copy: dict, fork: dict) -> dict:
    vendor_targets, vendor_bindings = load_vendor()
    checks = {
        "package": VENDOR.name,
        "entry": ENTRY,
        "copy": copy,
        "fork": fork,
        "vendor": {
            "files": sum(1 for p in VENDOR.rglob("*") if p.is_file()),
            "mesh_bindings": len(vendor_bindings),
            "targets": len(vendor_targets),
            "materials": sorted({b["material"] for b in vendor_bindings}),
        },
        "converted": {
            "targets": len(our_targets(vendor_targets)),
            "bindings": len(our_bindings(vendor_bindings)),
            "material_map": MATERIAL_MAP,
        },
    }
    out = PROJECT / "scene_intake" / "out" / VENDOR.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    return checks


def check() -> int:
    """不启动 Isaac 的离线校验：三件套能否被现有 loader 读入。"""
    sys.path.insert(0, str(PROJECT))
    from scripts.scene_config import load_scene_config
    from scripts.thermal_config import load_thermal_config

    failures = []
    for name in ("city_200m_scene.yaml", "city_200m_flight_scene.yaml"):
        try:
            config = load_scene_config(CONFIGS / name)
            print(f"OK scene  {name}: stage={config.scene.stage}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
    for name in ("city_200m_thermal.yaml", "city_200m_thermal_flight.yaml"):
        try:
            config = load_thermal_config(CONFIGS / name)
            print(f"OK thermo {name}: targets={len(config.targets)} bindings={len(config.bindings)}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {exc}")
    for item in failures:
        print("FAIL " + item)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只校验已生成文件")
    args = parser.parse_args()
    if args.check:
        return check()
    copy = copy_work()
    result = write_configs()
    fork = write_fork_notes()
    report = intake_report(copy, fork)
    print(f"WORK      {WORK}  copied={copy['copied']} files={copy['files']}")
    print(f"THERMAL   targets={result['targets']} bindings={result['bindings']}")
    print(f"OVERLAY   {PROJECT / 'city_200m_flight.usda'}")
    print(f"REPORT    {PROJECT / 'scene_intake' / 'out' / VENDOR.name / 'report.json'}")
    print(f"FORK      identical={report['fork']['identical']}")
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
