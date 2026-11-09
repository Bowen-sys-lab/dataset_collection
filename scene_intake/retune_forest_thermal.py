"""把森林场景的常温材质温度按夜间物理合理值分布，并记录改动。

规则（气温 18 C、晴夜、风 2 m/s）：
  vegetation_leaf  13.5 - 15.5 C   （树冠对天辐射冷却）
  wood_pine        17.0 - 18.2 C   （树干热惯量）
  soil_dry(常温)   15.5 C          （干土半遮阴）
火相关（110/180/500/650 C）一律不动。
温度用路径哈希做确定性抖动，重跑得到同样结果。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT = Path(r"D:\IsaacSim_Demo")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import omni.usd  # noqa: E402
from pxr import Sdf, Usd  # noqa: E402

SCENE = PROJECT / "forest_fire_scene_local" / "scene.usda"

ctx = omni.usd.get_context()
ctx.open_stage(str(SCENE))
for _ in range(30):
    app.update()
stage = ctx.get_stage()

HV = "thermal:temperatureC"
MAT = "thermal:material"

def jitter(path: str, lo: float, hi: float) -> float:
    h = int(hashlib.sha256(path.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return round(lo + (hi - lo) * h, 1)

edited = {"leaf": 0, "wood": 0, "soil": 0}
stage.SetEditTarget(stage.GetRootLayer())
with Usd.EditContext(stage, stage.GetRootLayer()):
    for prim in stage.Traverse():
        mat = prim.GetAttribute(MAT)
        tmp = prim.GetAttribute(HV)
        if not (mat and mat.HasAuthoredValue() and tmp and tmp.HasAuthoredValue()):
            continue
        name = mat.Get()
        cur = float(tmp.Get())
        if name == "vegetation_leaf" and cur < 100:
            tmp.Set(jitter(prim.GetPath().pathString, 13.5, 15.5))
            edited["leaf"] += 1
        elif name == "wood_pine" and cur < 100:
            tmp.Set(jitter(prim.GetPath().pathString, 17.0, 18.2))
            edited["wood"] += 1
        elif name == "soil_dry" and cur < 100:
            tmp.Set(15.5)
            edited["soil"] += 1
stage.GetRootLayer().Save()
print("edited:", edited)
app.close()
