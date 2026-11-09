"""把我们的观感调优应用到新的场景副本（光照 + 常温材质温度分布）。

与森林旧副本保持一致：
  - Moon 140 -> 210、Sky 90 -> 135（×1.5，夜间不至于全黑）
  - vegetation_leaf 13.5–15.5、wood_pine 17.0–18.2、soil_dry(常温) 15.5
  - 火相关与干扰目标（rusted_steel / human_skin / ...）一律不动

用法：
    & 'D:\\isaacsim\\python.bat' scene_intake\\tune_scene_copy.py --scene <path to scene.usda>
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from isaacsim import SimulationApp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--light-scale", type=float, default=1.5)
ap.add_argument("--no-light", action="store_true")
ap.add_argument("--no-thermal", action="store_true")
args = ap.parse_args()

SCENE = Path(args.scene).resolve()
backup = SCENE.with_suffix(SCENE.suffix + ".bak_before_tuning")
if not backup.exists():
    shutil.copy2(SCENE, backup)
print("backup:", backup.name, "sha256:", hashlib.sha256(backup.read_bytes()).hexdigest()[:16])

app = SimulationApp({"headless": True})
import omni.usd  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(str(SCENE))
for _ in range(30):
    app.update()
stage = ctx.get_stage()
layer = stage.GetRootLayer()
stage.SetEditTarget(layer)

LIGHTS = {"Moon": 140.0, "Sky": 90.0}
HV, MAT = "thermal:temperatureC", "thermal:material"
edited = {"lights": {}, "leaf": 0, "wood": 0, "soil": 0}


def jitter(path: str, lo: float, hi: float) -> float:
    h = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return round(lo + (hi - lo) * h, 1)


with Usd.EditContext(stage, layer):
    if not args.no_light:
        for name, original in LIGHTS.items():
            prim = stage.GetPrimAtPath(f"/World/Environment/{name}")
            if not prim or not prim.IsValid():
                prim = stage.GetPrimAtPath(f"/World/Lights/{name}")
            attr = prim.GetAttribute("inputs:intensity") if prim and prim.IsValid() else None
            if attr and attr.HasAuthoredValue():
                new = round(float(attr.Get()) * args.light_scale, 1)
                attr.Set(new)
                edited["lights"][name] = [float(attr.Get()), new]
    if not args.no_thermal:
        for prim in stage.Traverse():
            mat = prim.GetAttribute(MAT)
            tmp = prim.GetAttribute(HV)
            if not (mat and mat.HasAuthoredValue() and tmp and tmp.HasAuthoredValue()):
                continue
            name = mat.Get()
            cur = float(tmp.Get())
            p = prim.GetPath().pathString
            if name == "vegetation_leaf" and cur < 100:
                tmp.Set(jitter(p, 13.5, 15.5))
                edited["leaf"] += 1
            elif name == "wood_pine" and cur < 100:
                tmp.Set(jitter(p, 17.0, 18.2))
                edited["wood"] += 1
            elif name == "soil_dry" and cur < 100:
                tmp.Set(15.5)
                edited["soil"] += 1
layer.Save()
print("edited:", edited)
app.close()
