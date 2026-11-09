"""Integrate the 2026-11-03 city delivery as a separate v2 scene family.

The new package is self-contained (``city_200m.usda`` + ``city_block_original.usda``)
and ships six built-in targets, 1176 meshes and a 20-target thermal config.
This script:

1. builds ``city_200m_v2_flight.usda`` (night_clear overlay + main entry + CityFlight rig),
2. builds the thermal config from the scenario's ``thermalBoundary`` custom data and
   the vendor's prim bindings,
3. builds the display and scene YAMLs,
4. validates that every binding path and material preset resolves.

Run with plain Python 3.12 + PyYAML + USD (no Isaac import for --check).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from pxr import Sdf, Usd, UsdGeom

PROJECT = Path(r"D:\IsaacSim_Demo")
LOCAL = PROJECT / "city_200m_v2_local"
CONFIGS = PROJECT / "configs"
MATERIALS = PROJECT / "third_party" / "irsim-15015cd" / "configs" / "materials"
SENSOR = (PROJECT / "third_party" / "isaac-thermal-camera-simulation" / "configs"
          / "sensors" / "flir_boson_640_lwir.yaml")

FLIGHT = PROJECT / "city_200m_v2_flight.usda"
THERMAL = CONFIGS / "city_200m_v2_thermal.yaml"
DISPLAY = CONFIGS / "city_200m_v2_display.yaml"
SCENE = CONFIGS / "city_200m_v2_scene.yaml"

FLIGHT_TEMPLATE = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    startTimeCode = 0
    endTimeCode = 2400
    timeCodesPerSecond = 24
    framesPerSecond = 24
    subLayers = [
        @./city_200m_v2_local/city_200m.usda@
    ]
    customLayerData = {{
        string comment = "2026-11-03 city delivery + Crazyflie rig (night clear)."
    }}
)
over "World"
{{
    def Xform "CityFlight"
    {{
        double3 xformOp:translate = (-11.81, 5.81, 3.27)
        float3 xformOp:rotateXYZ = (0, 0, -50)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]

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
                    # 行 = 相机 X/右、Y/上、Z/后、平移（row-vector，平移在第 4 行）
                    matrix4d xformOp:transform = ((0, -1, 0, 0), (0.1736, 0, 0.9848, 0), (-0.9848, 0, 0.1736, 0), (0.12, 0, -0.03, 1))
                    uniform token[] xformOpOrder = ["xformOp:transform"]
                }}
            }}
        }}
    }}

    def PhysicsScene "PhysicsScene"
    {{
        vector3f physics:gravityDirection = (0, 0, -1)
        float physics:gravityMagnitude = 9.81
    }}
}}
"""


def scenario_boundary(scenario: Path) -> dict:
    layer = Sdf.Layer.FindOrOpen(str(scenario))
    data = layer.customLayerData
    boundary = data["thermalBoundary"]
    targets = boundary["targets"]
    temperatures = {}
    for name, values in targets.items():
        temperatures[name] = float(values["temperature_c"])
    return {"air": float(boundary["air_temperature_c"]), "targets": temperatures}


def build_flight() -> dict:
    FLIGHT.write_text(FLIGHT_TEMPLATE.format(), encoding="utf-8", newline="\n")
    return {"flight": str(FLIGHT)}


def build_thermal() -> dict:
    vendor = yaml.safe_load((LOCAL / "configs" / "thermal.yaml").read_text(encoding="utf-8"))
    targets = {name: {"mode": "fixed", "temperature_c": float(spec["temperature_c"])}
               for name, spec in vendor["targets"].items()}
    bindings = []
    for index, binding in enumerate(vendor["bindings"], start=1):
        bindings.append({
            "name": f"v2_{index:04d}",
            "prim_path": binding["prim_path"],
            "include_children": False,
            "target": binding["target"],
            "material": binding["material"],
        })
    doc = {
        "schema_version": 1,
        "defaults": {"target": "unknown", "material": "concrete"},
        "targets": targets,
        "bindings": bindings,
    }
    header = "# 2026-11-03 城市 v2：目标温度取交付包 thermal.yaml（主场景 night），绑定为逐 Mesh。\n"
    body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, default_flow_style=False, width=200)
    THERMAL.write_text(header + body, encoding="utf-8", newline="\n")
    return {"thermal": str(THERMAL), "targets": len(targets), "bindings": len(bindings)}


def build_materials() -> dict:
    """Convert the 12 new irsim presets to the older runtime's band schema.

    The 2026-11 delivery materials author ``spectral_emissivity`` (sometimes a
    list of curves) together with a partial ``emissivity_per_band``. The thermal
    runtime pinned in this project accepts exactly one optical source, so keep
    four-band emissivity/roughness and drop the spectral files.
    """
    out = CONFIGS / "materials" / "city_200m_v2"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.yaml"):
        stale.unlink()
    base = PROJECT / "third_party" / "isaac-thermal-camera-simulation" / "configs" / "materials"
    names = sorted({b["material"] for b in yaml.safe_load(
        (LOCAL / "configs" / "thermal.yaml").read_text(encoding="utf-8"))["bindings"]})
    converted, from_base = [], []
    for name in names:
        if (base / f"{name}.yaml").is_file():
            # The runtime treats a custom file with a base name as an override and
            # requires identical content; reuse the project preset instead.
            from_base.append(name)
            continue
        doc = yaml.safe_load((MATERIALS / f"{name}.yaml").read_text(encoding="utf-8"))
        optical = doc["material"]["optical"]
        optical.pop("spectral_emissivity", None)
        optical.pop("spectral_reflectance", None)
        optical.pop("reflectance_per_band", None)
        bands = dict(optical.get("emissivity_per_band") or {})
        for band in ("nir", "swir", "mwir"):
            bands.setdefault(band, 0.90)
        bands.setdefault("lwir", bands["mwir"])
        optical["emissivity_per_band"] = {band: bands[band] for band in ("nir", "swir", "mwir", "lwir")}
        roughness = dict(optical.get("roughness_per_band") or {})
        for band in ("nir", "swir", "mwir"):
            roughness.setdefault(band, 0.75)
        roughness.setdefault("lwir", roughness["mwir"])
        optical["roughness_per_band"] = {band: roughness[band] for band in ("nir", "swir", "mwir", "lwir")}
        (out / f"{name}.yaml").write_text(
            yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, default_flow_style=False, width=120),
            encoding="utf-8", newline="\n")
        converted.append(name)
    return {"materials": str(out), "count": len(converted), "converted": converted,
            "from_base": from_base}


def build_display_scene() -> dict:
    environment = CONFIGS / "city_200m_v2_night_environment.yaml"
    environment.write_text(
        "# 2026-11-03 城市 v2 夜间热环境；weather_file 改绝对路径（运行时 cwd 是项目根）。\n"
        "schema_version: 17\n"
        "scene:\n"
        "  name: city_night_200m_person\n"
        "  description: Synthetic 200 m night city with six targets and prescribed surface temperatures\n"
        "  weather_file: D:\\IsaacSim_Demo\\city_200m_v2_local\\environment\\weather.csv\n"
        "  atmosphere_preset: us_standard_clear\n"
        "  environment_preset: clear_dry\n"
        "  site:\n"
        "    latitude_deg: 31.2\n"
        "    longitude_deg: 121.5\n"
        "    altitude_m: 10.0\n"
        "  start_utc: '2026-10-26T12:00:00Z'\n", encoding="utf-8", newline="\n")
    DISPLAY.write_text(
        "# 2026-11-03 城市 v2：夜间 10–70 ℃，热目标 43–62 ℃ 留在线性区。\n"
        "schema_version: 1\n"
        "name: city_200m_v2_night_clear\n"
        "mode: fixed_apparent_temperature\n"
        "temperature_min_c: 10.0\n"
        "temperature_max_c: 70.0\n"
        "dde_gain: 0.0\n"
        "polarity: white_hot\n"
        "palette: gray\n", encoding="utf-8", newline="\n")
    SCENE.write_text(
        "# 2026-11-03 城市 v2（night_clear）+ 飞行 rig。\n"
        "schema_version: 1\n"
        "scene:\n"
        "  stage: ../city_200m_v2_flight.usda\n"
        "  thermal_scene: city_200m_v2_night_environment.yaml\n"
        "  static_targets: []\n"
        "drone:\n"
        "  prim_path: /World/CityFlight/Crazyflie\n"
        "  body_path: /World/CityFlight/Crazyflie/body\n"
        "  camera_path: /World/CityFlight/Crazyflie/body/CityCam\n"
        "  minimum_height: 1.5\n"
        "sensors:\n"
        "  width: 320\n"
        "  height: 256\n"
        "  sensor_config: ../third_party/isaac-thermal-camera-simulation/configs/sensors/flir_boson_640_lwir.yaml\n"
        "  thermal_config: city_200m_v2_thermal.yaml\n"
        "  preview_fast: true\n"
        "  align_thermal_to_rgb: true\n"
        "  flat_field_enabled: true\n"
        "  sensor_seed: 741\n"
        "  display_config: city_200m_v2_display.yaml\n"
        "  material_directory: materials/city_200m_v2\n",
        encoding="utf-8", newline="\n")
    return {"scene": str(SCENE), "display": str(DISPLAY), "environment": str(environment)}


def check() -> dict:
    doc = yaml.safe_load(THERMAL.read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(str(FLIGHT))
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    bound = set()
    missing = []
    for binding in doc["bindings"]:
        path = binding["prim_path"]
        prim = stage.GetPrimAtPath(path)
        if not prim:
            missing.append(path)
            continue
        bound.add(path)
    materials = {b["material"] for b in doc["bindings"]}
    converted = CONFIGS / "materials" / "city_200m_v2"
    base = PROJECT / "third_party" / "isaac-thermal-camera-simulation" / "configs" / "materials"
    absent = sorted(name for name in materials
                    if not (converted / f"{name}.yaml").is_file()
                    and not (base / f"{name}.yaml").is_file())
    unbound = [str(p.GetPath()) for p in meshes if str(p.GetPath()) not in bound]
    return {
        "meshes": len(meshes),
        "bindings": len(doc["bindings"]),
        "missing_prim_paths": missing[:5],
        "missing_prim_count": len(missing),
        "unbound_meshes": unbound[:5],
        "unbound_count": len(unbound) if unbound else 0,
        "missing_material_presets": absent,
        "targets": len(doc["targets"]),
        "target_names": sorted(doc["targets"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(), ensure_ascii=False, indent=2))
        return 0
    result = {}
    result.update(build_flight())
    result.update(build_thermal())
    result.update(build_materials())
    result.update(build_display_scene())
    result["check"] = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
