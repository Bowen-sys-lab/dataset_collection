"""飞行可行性测试：火场景 + 你们的 Crazyflie，能不能起控制器并稳定悬停。

只读场景；物理改动（关重力等）由他们的运行时写在会话层，不落盘。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\test_fire_flight.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "third_party" / "isaac-thermal-camera-simulation" / "src"))

from isaacsim import SimulationApp  # noqa: E402
from irsim_isaac.env import simulation_app_config  # noqa: E402

app = SimulationApp(simulation_app_config())

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

STAGE = PROJECT / "forest_fire_flight.usda"
DRONE = "/World/FireFlight/Crazyflie"

report = {"stage": str(STAGE), "drone": DRONE}
ctx = omni.usd.get_context()
ctx.open_stage(str(STAGE))
for _ in range(90):
    app.update()

from scripts.drone_physics_runtime import DronePhysicsRuntime  # noqa: E402
from scripts.start_crazyflie_controller import start_crazyflie  # noqa: E402

runtime = DronePhysicsRuntime(DRONE)
report["articulation_valid"] = runtime.valid
report["articulation_ready"] = runtime.ready
before = runtime.snapshot()
report["pose_before"] = before.get("position")

timeline = omni.timeline.get_timeline_interface()
timeline.play()
app.update()

controller = start_crazyflie(prim_path=DRONE, minimum_height=1.5)
report["controller_started"] = controller is not None

samples = []
t0 = time.time()
for _ in range(300):  # ~5 s at 60 Hz
    app.update()
    if len(samples) < 5 and (time.time() - t0) > 0.8 * (len(samples) + 1):
        snap = runtime.snapshot()
        if snap.get("ready"):
            samples.append({"t_s": round(time.time() - t0, 2), "z": round(snap["position"][2], 3)})

after = runtime.snapshot()
report["pose_after"] = after.get("position")
report["samples"] = samples
report["physics_ready_after"] = after.get("ready")
report["gravity_enabled_after"] = after.get("gravity_enabled")

ok = bool(after.get("ready")) and bool(samples) and all(abs(s["z"] - samples[0]["z"]) < 1.0 for s in samples)
report["hover_stable_within_1m"] = ok
report["verdict"] = "PASS" if ok else "FAIL"
print("FLIGHT_TEST", json.dumps(report, ensure_ascii=False), flush=True)

try:
    from scripts.start_crazyflie_controller import stop_controller
    stop_controller()
except Exception:  # noqa: BLE001
    pass
timeline.stop()
app.close()
