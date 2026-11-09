"""Check actual exported single-material bindings against Blender's visible material."""
from pathlib import Path


def validate_export():
    import bpy
    from pxr import Usd, UsdShade
    path = Path('D:/IsaacSim_Demo/factory/model02_thermal_v2/model02_refined_asset.usdc')
    stage = Usd.Stage.Open(str(path))
    stage.GetRootLayer().Reload()
    problems = []
    checked = 0
    for prim in stage.Traverse():
        attr = prim.GetAttribute('userProperties:blender:object_name')
        name = attr.Get() if attr else None
        obj = bpy.data.objects.get(name) if name else None
        if not obj or obj.type!='MESH':
            continue
        used = set(p.material_index for p in obj.data.polygons)
        if len(used)!=1:
            continue
        expected = obj.data.materials[next(iter(used))].name.replace('-','_').replace('.','_')
        if expected[0].isdigit():
            expected = '_'+expected
        meshes = [p for p in prim.GetChildren() if p.GetTypeName()=='Mesh']
        for mesh in meshes:
            material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
            actual = material.GetPrim().GetName() if material else None
            checked += 1
            if actual!=expected:
                problems.append({'object':name,'expected':expected,'actual':actual})
    if checked == 0:
        problems.append({'error':'No matching single-material Mesh was checked'})
    return {'result':'FAIL' if problems else 'PASS','checked_meshes':checked,'problems':problems}
