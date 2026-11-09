"""检查"火焰代面"是不是真的合成进场景了。"""
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.usd
from pxr import Usd, UsdGeom, UsdShade

STAGE = Path(r"D:\IsaacSim_Demo\forest_fire_flight_flame.usda")
ctx = omni.usd.get_context()
ctx.open_stage(str(STAGE))
for _ in range(60):
    app.update()
stage = ctx.get_stage()

rows = []
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Site/Target_BurningBarrel_001/Flames")):
    if prim.GetTypeName() != "Mesh":
        continue
    mesh = UsdGeom.Mesh(prim)
    faces = len(mesh.GetFaceVertexCountsAttr().Get() or [])
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    rows.append((prim.GetName(), faces, mat.GetPath().pathString if mat else None))

total_faces = sum(r[1] for r in rows)
materials = {r[2] for r in rows}
print(f"FLAME_CHECK meshes={len(rows)} faces={total_faces} materials={sorted(m for m in materials if m)}")
for name, faces, mat in rows:
    print(f"  {name:12s} faces={faces:5d} material={mat}")

drone = stage.GetPrimAtPath("/World/FireFlight/Crazyflie")
print("drone articulation:", bool(drone and drone.HasAPI(Usd.PhysicsArticulationRootAPI)) if drone else False)
print("terrain meshes:", len([p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/Site/Forest"))
                              if p.GetTypeName() == "Mesh"]))
app.close()
