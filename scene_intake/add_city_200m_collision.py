"""把街区粗碰撞盒换成逐 Mesh 的凸包碰撞（可选的高精度版本）。

粗盒只有一个 42×39×14 m 的盒子，无人机贴着骑楼飞行时会被"整段街区"挡住。
这个脚本另存一份 stage：关掉粗盒，给 540 个建筑/街道 Mesh 逐个加
PhysicsCollisionAPI + PhysicsMeshCollisionAPI（convexHull），碰撞贴合几何。

540 个凸包会增加 PhysX 启动/内存开销，所以它是可选版本，默认场景仍用粗盒。

用法（需要 USD 的 PYTHONPATH）：
    python scene_intake\\add_city_200m_collision.py
"""
from __future__ import annotations

from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics

PROJECT = Path(r"D:\IsaacSim_Demo")
SOURCE = PROJECT / "city_200m_scenario.usda"
TARGET = PROJECT / "city_200m_precise_collision.usda"
CONFIG = PROJECT / "configs" / "city_200m_precise_collision_scene.yaml"

SCENE = """\
# 高精度碰撞版：粗盒关闭，540 个建筑/街道 Mesh 逐个凸包碰撞。
# PhysX 启动和内存开销比粗盒版高，按需使用。
schema_version: 1

scene:
  stage: ../city_200m_precise_collision.usda
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


def build() -> dict:
    stage = Usd.Stage.Open(str(SOURCE))
    root = stage.GetRootLayer()
    layer = Sdf.Layer.CreateNew(str(TARGET))
    layer.subLayerPaths.append("./city_200m_scenario.usda")
    layer.defaultPrim = "World"
    layer.customLayerData = {
        "comment": "Scenario with per-mesh convex-hull collision and coarse boxes disabled."
    }
    stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
    stage.SetEditTarget(layer)
    try:
        coarse = stage.GetPrimAtPath("/World/ScenarioCollision")
        if coarse:
            coarse.SetActive(False)
        count = 0
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/World/Blocks/"):
                continue
            if prim.GetTypeName() != "Mesh":
                continue
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            count += 1
    finally:
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetSessionLayer().subLayerPaths.remove(layer.identifier)
    layer.Save()
    CONFIG.write_text(SCENE, encoding="utf-8")
    return {"target": str(TARGET), "colliders": count}


def main() -> int:
    result = build()
    print(f"WROTE {result['target']}  colliders={result['colliders']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
