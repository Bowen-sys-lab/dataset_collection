"""光照探针：同一机位下，月光/天光强度与曝光对画面亮度的影响。

只读场景；强度与曝光只改会话状态，不写回 USD。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\lighting_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
from isaacsim import SimulationApp  # noqa: E402

ASSETS = PROJECT / "validation" / "isaac_local_assets"
app = SimulationApp({
    "headless": True,
    "renderer": "PathTracing",
    "extra_args": ["--/persistent/isaac/asset_root/default=" + str(ASSETS).replace("\\", "/")],
})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd  # noqa: E402

STAGE = PROJECT / "forest_fire_scene_local" / "scene.usda"
CAM = "/World/Cameras/TargetClose"
OUT = PROJECT / "validation" / "lighting_probe"
OUT.mkdir(parents=True, exist_ok=True)

ctx = omni.usd.get_context()
ctx.open_stage(str(STAGE))
for _ in range(60):
    app.update()
stage = ctx.get_stage()

moon = stage.GetPrimAtPath("/World/Environment/Moon")
sky = stage.GetPrimAtPath("/World/Environment/Sky")
base = {
    "moon": moon.GetAttribute("inputs:intensity").Get(),
    "sky": sky.GetAttribute("inputs:intensity").Get(),
}

product = rep.create.render_product(CAM, (480, 480))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(product)

rows = []
settings = [
    ("as-delivered", base["moon"], base["sky"], 0.0),
    ("x3 lights", base["moon"] * 3, base["sky"] * 3, 0.0),
    ("x6 lights", base["moon"] * 6, base["sky"] * 6, 0.0),
    ("as-delivered +2EV", base["moon"], base["sky"], 2.0),
    ("x3 lights +1EV", base["moon"] * 3, base["sky"] * 3, 1.0),
]
for name, moon_i, sky_i, ev in settings:
    moon.GetAttribute("inputs:intensity").Set(float(moon_i))
    sky.GetAttribute("inputs:intensity").Set(float(sky_i))
    from pxr import UsdRender  # noqa: F401  (kept for symmetry; exposure via carb)
    import carb
    carb.settings.get_settings().set("/rtx/post/tonemap/exposure", float(ev))
    for _ in range(6):
        rep.orchestrator.step(rt_subframes=4)
    data = np.asarray(ann.get_data())[..., :3].astype(np.float32)
    lum = data @ np.array([0.2126, 0.7152, 0.0722])
    rows.append({
        "case": name,
        "moon": round(float(moon_i), 1),
        "sky": round(float(sky_i), 1),
        "exposure_ev": ev,
        "mean_luma": round(float(lum.mean()), 2),
        "p50": round(float(np.percentile(lum, 50)), 2),
        "p95": round(float(np.percentile(lum, 95)), 2),
        "below_10_pct": round(float((lum < 10).mean() * 100), 1),
        "above_200_pct": round(float((lum > 200).mean() * 100), 2),
    })
    Image.fromarray(data.astype(np.uint8)).save(OUT / (name.replace(" ", "_") + ".png"))

(OUT / "lighting_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("LIGHTING_SWEEP", json.dumps(rows), flush=True)
app.close()
