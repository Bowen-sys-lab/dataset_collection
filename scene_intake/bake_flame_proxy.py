"""把作者的热代面烘成一层 USD，供回放使用。

作者的做法是：渲染红外前，按当前帧的火焰透明轮廓生成不透明面片，绑一个平材质，
只存在于会话层。回放脚本（既有流程）没有这一步，所以红外里看不到火焰。

这里不改任何脚本：在会话层里跑一次作者的工具，把那一层导出成独立 USD，
再用它和飞行叠加层组合成一个"火焰可见"的场景，专供红外回放。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\bake_flame_proxy.py --time-code 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
for entry in (PROJECT, PROJECT / "third_party" / "isaac-thermal-camera-simulation" / "src",
              PROJECT / "forest_fire_scene_local" / "scripts"):
    sys.path.insert(0, str(entry))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Sdf, Usd  # noqa: E402

try:  # 优先用本项目自己的副本，脱离作者脚本目录
    from scene_intake.fire_thermal_proxy import thermal_flame_surfaces  # noqa: E402
except ImportError:  # noqa: E402
    from fire_thermal_proxy import thermal_flame_surfaces  # noqa: E402

SCENE = PROJECT / "forest_fire_scene_local" / "scene.usda"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-code", type=float, default=0.0)
    ap.add_argument("--out", default=str(PROJECT / "forest_fire_flame_proxy.usda"))
    args = ap.parse_args()
    out = Path(args.out)

    ctx = omni.usd.get_context()
    ctx.open_stage(str(SCENE))
    for _ in range(60):
        app.update()
    stage = ctx.get_stage()

    with thermal_flame_surfaces(stage, args.time_code):
        session = stage.GetSessionLayer()
        if not session.subLayerPaths:
            raise SystemExit("no session sublayer was created")
        layer = Sdf.Layer.Find(session.subLayerPaths[0])
        if layer is None:
            raise SystemExit("session sublayer could not be found")
        layer.Export(str(out))

    check = Usd.Stage.Open(str(out))
    if check is None:
        raise SystemExit(f"exported layer does not open: {out}")
    meshes = [p for p in check.Traverse() if p.GetTypeName() == "Mesh"]
    looks = [p for p in check.Traverse() if p.GetTypeName() == "Material"]
    print(f"BAKED {out}  meshes={len(meshes)} materials={len(looks)} bytes={out.stat().st_size}")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
