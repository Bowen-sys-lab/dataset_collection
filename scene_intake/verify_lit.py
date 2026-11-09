"""量一版"提亮多少"，用与照度探针相同的机位和统计口径。

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\verify_lit.py --stage forest_fire_lit.usda
"""
from __future__ import annotations

import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default=str(PROJECT / "forest_fire_lit.usda"))
ap.add_argument("--out", default=str(PROJECT / "validation" / "lighting_probe"))
args = ap.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
ctx = omni.usd.get_context()
ctx.open_stage(args.stage)
for _ in range(60):
    app.update()
stage = ctx.get_stage()

moon = stage.GetPrimAtPath("/World/Environment/Moon").GetAttribute("inputs:intensity").Get()
sky = stage.GetPrimAtPath("/World/Environment/Sky").GetAttribute("inputs:intensity").Get()

product = rep.create.render_product("/World/Cameras/TargetClose", (480, 480))
ann = rep.AnnotatorRegistry.get_annotator("rgb")
ann.attach(product)
for _ in range(8):
    rep.orchestrator.step(rt_subframes=4)
data = np.asarray(ann.get_data())[..., :3].astype(np.float32)
lum = data @ np.array([0.2126, 0.7152, 0.0722])
row = {
    "stage": Path(args.stage).name,
    "moon": float(moon),
    "sky": float(sky),
    "mean_luma": round(float(lum.mean()), 2),
    "p50": round(float(np.percentile(lum, 50)), 2),
    "p95": round(float(np.percentile(lum, 95)), 2),
    "below_10_pct": round(float((lum < 10).mean() * 100), 1),
    "above_200_pct": round(float((lum > 200).mean() * 100), 2),
}
Image.fromarray(data.astype(np.uint8)).save(out / "lit_1p8.png")
(out / "lit_measure.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
print("LIT_MEASURE", json.dumps(row), flush=True)
app.close()
