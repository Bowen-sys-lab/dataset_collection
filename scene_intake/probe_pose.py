"""打印某个 prim（或相机）的世界矩阵与朝向，用来排查姿态问题。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\probe_pose.py --stage forest_fire_flight.usda
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
from isaacsim import SimulationApp  # noqa: E402

ASSETS = PROJECT / "validation" / "isaac_local_assets"
app = SimulationApp({"headless": True,
                     "extra_args": ["--/persistent/isaac/asset_root/default=" + str(ASSETS).replace("\\", "/")]})

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default=str(PROJECT / "forest_fire_flight.usda"))
ap.add_argument("--prim", default="/World/FireFlight")
ap.add_argument("--camera", default="/World/FireFlight/Crazyflie/body/Camera_mount/Camera")
args = ap.parse_args()

ctx = omni.usd.get_context()
ctx.open_stage(str(Path(args.stage).resolve()))
for _ in range(60):
    app.update()
stage = ctx.get_stage()
t = Usd.TimeCode.Default()

def dump(path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print(f"{path}: MISSING")
        return None
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(t)
    trans = m.ExtractTranslation()
    rot = m.ExtractRotationMatrix()
    ops = [op.GetOpName() for op in xf.GetOrderedXformOps()]
    print(f"{path}")
    print(f"  ops     : {ops}")
    print(f"  translate: {tuple(round(v, 3) for v in trans)}")
    print("  rot rows : " + " | ".join(str(tuple(round(rot[r][c], 3) for c in range(3))) for r in range(3)))
    return rot

rot = dump(args.prim)
dump(args.camera)
if rot is not None:
    fwd = -(rot * Gf.Vec3d(0, 0, 1))
    print("  (if this prim were the camera) forward:", tuple(round(v, 4) for v in fwd))
app.close()
