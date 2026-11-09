"""从指定相机渲一帧，用来肉眼检查朝向和是否拍到机身。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\look_through_camera.py --out scene_intake\\out\\cam_check.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
from isaacsim import SimulationApp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default=str(PROJECT / "forest_fire_flight.usda"))
ap.add_argument("--camera", default="/World/FireFlight/Crazyflie/body/Camera_mount/Camera")
ap.add_argument("--out", default=str(PROJECT / "scene_intake" / "out" / "cam_check.png"))
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=512)
args = ap.parse_args()

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(str(Path(args.stage).resolve()))
for _ in range(90):
    app.update()
stage = ctx.get_stage()
cam = stage.GetPrimAtPath(args.camera)
if not cam or not cam.IsValid():
    raise SystemExit(f"missing camera: {args.camera}")

rp = rep.create.render_product(args.camera, (args.width, args.height))
writer = rep.WriterRegistry.get("BasicWriter")
out = Path(args.out).resolve()
out.parent.mkdir(parents=True, exist_ok=True)
writer.initialize(output_dir=str(out.parent), rgb=True)
writer.attach([rp])
for _ in range(6):
    app.update()
rep.orchestrator.step()
writer.detach()
rp.destroy()
print("wrote:", out)
app.close()
