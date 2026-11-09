"""算出让机载相机对准火桶的机体姿态修正（世界系前乘）。

不猜相机装法：直接读相机当前的世界朝向，算它与目标方向的夹角，
给出需要叠加在 FireFlight 上的四元数。若已有姿态不完美，再跑一次即可收敛。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\aim_camera.py
"""
from __future__ import annotations

import math
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

STAGE = PROJECT / "forest_fire_flight.usda"
FLIGHT = "/World/FireFlight"
CAMERA = "/World/FireFlight/Crazyflie/body/Camera_mount/Camera"
BARREL = Gf.Vec3d(55.0, 6.86, 14.71)

ctx = omni.usd.get_context()
ctx.open_stage(str(STAGE))
for _ in range(60):
    app.update()
stage = ctx.get_stage()
time = Usd.TimeCode.Default()

flight_prim = stage.GetPrimAtPath(FLIGHT)
flight = UsdGeom.Xformable(flight_prim)
camera = UsdGeom.Xformable(stage.GetPrimAtPath(CAMERA))

m_cam = camera.ComputeLocalToWorldTransform(time)
pos = m_cam.ExtractTranslation()
f_world = (-(m_cam.ExtractRotationMatrix() * Gf.Vec3d(0, 0, 1))).GetNormalized()
target = (BARREL - pos).GetNormalized()
angle = math.degrees(math.acos(max(-1.0, min(1.0, Gf.Dot(f_world, target)))))

print("camera world pos:", tuple(round(v, 3) for v in pos))
print("camera world forward:", tuple(round(v, 4) for v in f_world))
print("desired forward:", tuple(round(v, 4) for v in target))
print(f"current aim error: {angle:.2f} deg")

axis = Gf.Vec3d(Gf.Cross(f_world, target))
sin_a = axis.GetLength()
cos_a = Gf.Dot(f_world, target)
if sin_a < 1e-9:
    q_corr = Gf.Quatd(1, 0, 0, 0)
else:
    axis = axis.GetNormalized()
    q_corr = Gf.Quatd(math.cos(math.acos(cos_a) / 2.0), axis * math.sin(math.acos(cos_a) / 2.0))

current = flight_prim.GetAttribute("xformOp:orient")
q_current = current.Get() if current and current.HasAuthoredValue() else Gf.Quatd(1, 0, 0, 0)
q_new = q_corr * q_current
print("current orient:", tuple(round(v, 6) for v in (q_current.GetReal(), *q_current.GetImaginary())))
print("correction    :", tuple(round(v, 6) for v in (q_corr.GetReal(), *q_corr.GetImaginary())))
print("NEW orient    :", tuple(round(v, 6) for v in (q_new.GetReal(), *q_new.GetImaginary())))
print()
print("usda: quatd xformOp:orient = (%.6f, %.6f, %.6f, %.6f)" %
      (q_new.GetReal(), *q_new.GetImaginary()))
app.close()
