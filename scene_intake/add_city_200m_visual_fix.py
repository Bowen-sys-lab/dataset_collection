"""把街区 Mesh 改成双面渲染，修掉从室内/背面看时墙面消失的问题。

城市素材是单面 GLTF 网格（`doubleSided=false`）。从外面看正常，一旦飞进街区内部，
背面被剔除，墙就"缺一块"。这里用一个只读叠加层把 /World/Blocks 下所有 Mesh 的
`doubleSided` 打开，不改原资产。

产物：`city_200m_visual_fix.usda`（sublayer 原城市入口，只含 doubleSided 覆盖）。
之后 `city_200m_flight.usda` 改为 sublayer 这一层。

用法（需要 USD 的 PYTHONPATH）：
    python scene_intake\\add_city_200m_visual_fix.py
"""
from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

PROJECT = Path(r"D:\IsaacSim_Demo")
SOURCE = PROJECT / "city_200m_local" / "city_200m.usda"
TARGET = PROJECT / "city_200m_visual_fix.usda"


def build() -> dict:
    stage = Usd.Stage.Open(str(SOURCE))
    layer = Sdf.Layer.CreateNew(str(TARGET))
    layer.subLayerPaths.append("./city_200m_local/city_200m.usda")
    layer.defaultPrim = "World"
    layer.customLayerData = {
        "comment": "City visual fix: every block mesh renders double-sided."
    }
    stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
    stage.SetEditTarget(layer)
    count = 0
    try:
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if not str(prim.GetPath()).startswith("/World/Blocks/"):
                continue
            prim.GetAttribute("doubleSided").Set(True)
            count += 1
        bulbs = add_lamp_bulbs(stage)
    finally:
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.GetSessionLayer().subLayerPaths.remove(layer.identifier)
    layer.Save()
    return {"target": str(TARGET), "meshes": count, "bulbs": bulbs}


def add_lamp_bulbs(stage: Usd.Stage) -> int:
    """在每盏已点亮的 StreetLight 灯头位置放一颗小发光球，供 RGB/红外显示。"""
    material = UsdShade.Material.Define(stage, "/World/Looks/LampGlow")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/LampGlow/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 0.72, 0.42))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    created = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() != "SphereLight":
            continue
        path = str(prim.GetPath())
        if not path.startswith("/World/Lighting/"):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = transform.ExtractTranslation()
        bulb = UsdGeom.Sphere.Define(stage, "/World/LampBulbs/" + path[len("/World/Lighting/"):])
        bulb.GetRadiusAttr().Set(0.22)
        # The light sits at the fixture head; drop the proxy just below it so the
        # housing does not occlude the glow from normal flight altitudes.
        bulb.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], position[2] - 0.55))
        UsdShade.MaterialBindingAPI.Apply(bulb.GetPrim()).Bind(material)
        created += 1
    return created


def main() -> int:
    result = build()
    print(f"WROTE {result['target']}  doubleSided meshes={result['meshes']}  lamp bulbs={result['bulbs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
