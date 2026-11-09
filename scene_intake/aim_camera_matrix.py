"""反解相机局部矩阵：让相机朝向指定世界方向（默认指向火桶），并可沿该方向前移。

输出可直接粘进 USDA 的 matrix4d 行。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default=str(PROJECT / "forest_fire_flight.usda"))
ap.add_argument("--camera", default="/World/FireFlight/Crazyflie/body/Camera_mount/Camera")
ap.add_argument("--look-at", default="50.0, 6.864, 15.31", help="目标点 x,y,z")
ap.add_argument("--ahead", type=float, default=0.25)
ap.add_argument("--up-bias", type=float, default=0.0, help="额外上移，米")
args = ap.parse_args()

ctx = omni.usd.get_context()
ctx.open_stage(str(Path(args.stage).resolve()))
for _ in range(60):
    app.update()
stage = ctx.get_stage()

cam_prim = stage.GetPrimAtPath(args.camera)
if not cam_prim or not cam_prim.IsValid():
    raise SystemExit(f"missing camera: {args.camera}")
mount_path = args.camera.rsplit("/", 1)[0]
mount_w = UsdGeom.Xformable(stage.GetPrimAtPath(mount_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
cam_w = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

target = Gf.Vec3d(*[float(v) for v in args.look_at.split(",")])
cam_pos = cam_w.ExtractTranslation()
f = (target - cam_pos)
f.Normalize()
up_hint = Gf.Vec3d(0, 0, 1)
z_c = -f
x_c = Gf.Cross(up_hint, z_c)
x_c.Normalize()
y_c = Gf.Cross(z_c, x_c)
pos = cam_pos + f * args.ahead + Gf.Vec3d(0, 0, args.up_bias)

T = Gf.Matrix4d(
    x_c[0], y_c[0], z_c[0], 0.0,
    x_c[1], y_c[1], z_c[1], 0.0,
    x_c[2], y_c[2], z_c[2], 0.0,
    pos[0], pos[1], pos[2], 1.0)
local = T * mount_w.GetInverse()

print("--- look_at ---")
print("  fwd:", tuple(round(v, 4) for v in f))
print("  cam world pos now:", tuple(round(v, 3) for v in cam_pos))
print("  cam world pos new:", tuple(round(v, 3) for v in pos))
print("--- matrix4d（粘进 USDA）---")
print("\n".join("        (" + ", ".join(f"{local[r][c]:.6f}" for c in range(4)) + ")"
                for r in range(4)))
print("--- 校验 ---")
world_check = mount_w * local
fwd_check = -(world_check.ExtractRotationMatrix() * Gf.Vec3d(0, 0, 1))
fwd_check.Normalize()
print("  forward:", tuple(round(v, 4) for v in fwd_check),
      "| error deg:", round(Gf.Acos(Gf.Clamp(Gf.Dot(fwd_check, f), -1, 1)) * 180 / 3.141592653589793, 2))
print("  position:", tuple(round(v, 3) for v in world_check.ExtractTranslation()))
app.close()
