"""算出森林无人机相机的局部矩阵，使它在机体坐标系里落在指定位姿。

机体坐标约定（与工厂 model02 一致）：+X 前进、+Y 右、+Z 上。
目标位姿：机头前上方，光轴沿 +X、下俯 pitch_deg。

打印可直接粘进 USDA 的 matrix4d 行。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default=str(PROJECT / "forest_fire_flight.usda"))
ap.add_argument("--body", default="/World/FireFlight/Crazyflie/body")
ap.add_argument("--camera", default="/World/FireFlight/Crazyflie/body/Camera_mount/Camera")
ap.add_argument("--ahead", type=float, default=0.25)
ap.add_argument("--up", type=float, default=0.03)
ap.add_argument("--pitch-deg", type=float, default=15.0)
args = ap.parse_args()

ctx = omni.usd.get_context()
ctx.open_stage(str(Path(args.stage).resolve()))
for _ in range(60):
    app.update()
stage = ctx.get_stage()

def world(path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise SystemExit(f"missing prim: {path}")
    return UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

body_w = world(args.body)
cam_w = world(args.camera)
mount_w = world(args.camera.rsplit("/", 1)[0])

# 目标：机体坐标系下的相机位姿
pitch = math.radians(args.pitch_deg)
f = Gf.Vec3d(math.cos(pitch), 0.0, -math.sin(pitch))   # 光轴（前 + 下）
u = Gf.Vec3d(math.sin(pitch), 0.0, math.cos(pitch))    # 上
z_c = -f
x_c = Gf.Cross(u, z_c)                                 # 右 = up × z
target = Gf.Matrix4d(
    x_c[0], u[0], z_c[0], 0.0,
    x_c[1], u[1], z_c[1], 0.0,
    x_c[2], u[2], z_c[2], 0.0,
    args.ahead, 0.0, args.up, 1.0)

# 当前：mount 相对机体
mount_rel = mount_w * body_w.GetInverse()
local = mount_rel.GetInverse() * target

def rows(m):
    return "\n".join("        (" + ", ".join(f"{m[r][c]:.6f}" for c in range(4)) + ")" for r in range(4))

print("--- 目标（机体坐标系） ---")
print(rows(target))
print("--- 需要写到相机 prim 的局部矩阵 ---")
print(rows(local))
print("--- 校验：合成后光轴 ---")
world_target = body_w * target
fwd = -(world_target.ExtractRotationMatrix() * Gf.Vec3d(0, 0, 1))
print("  forward:", tuple(round(v, 4) for v in fwd))
print("  position:", tuple(round(v, 3) for v in world_target.ExtractTranslation()))
print("--- 当前相机 ---")
cur_fwd = -(cam_w.ExtractRotationMatrix() * Gf.Vec3d(0, 0, 1))
print("  current forward:", tuple(round(v, 4) for v in cur_fwd))
print("  current ops:", [op.GetOpName() for op in UsdGeom.Xformable(stage.GetPrimAtPath(args.camera)).GetOrderedXformOps()])
app.close()
