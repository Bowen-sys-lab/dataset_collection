"""Scene-only opaque silhouettes for the IR geometry AOV.

Originally written by the scene author inside the delivered package
(``forest_fire_scene_local/scripts/fire_thermal_proxy.py``) and copied here so this
project no longer depends on the vendor script tree. Behaviour unchanged.

RTX geometry AOVs omit the translucent visual fire. Sample the CURRENT atlas
alpha into surface geometry, preserving mesh paths and their 650 C bindings.
This remains a surface surrogate, not combustion gas radiometry.
"""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom, UsdShade, Sdf


@contextmanager
def thermal_flame_surfaces(stage, time_code):
    originals=[]
    images={}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or not prim.GetName().startswith('Tongue_'):continue
        mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        texture=stage.GetPrimAtPath(str(mat.GetPath())+'/Texture')
        if not texture or not stage.GetPrimAtPath(str(mat.GetPath())+'/CardMDL'):continue
        asset=texture.GetAttribute('inputs:file').Get()
        filename=asset.resolvedPath
        if not filename:raise ValueError('Unresolved fire atlas: '+str(asset))
        if filename not in images:images[filename]=np.asarray(Image.open(filename))[...,3]
        originals.append((str(prim.GetPath()),np.asarray(prim.GetAttribute('points').Get(time_code)),
                          np.asarray(prim.GetAttribute('primvars:st').Get(time_code)),images[filename]))
    layer=Sdf.Layer.CreateAnonymous('fire-thermal-silhouettes.usda')
    session=stage.GetSessionLayer()
    session.subLayerPaths.insert(0,layer.identifier)
    try:
        with Usd.EditContext(stage,layer):
            material=UsdShade.Material.Define(stage,'/World/Looks/FireThermalSilhouette')
            shader=UsdShade.Shader.Define(stage,str(material.GetPath())+'/Surface')
            shader.CreateIdAttr('UsdPreviewSurface')
            shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set((.3,.3,.3))
            shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(1.)
            shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
            total=0
            for path,original,uv,alpha in originals:
                ny,nx=96,48
                def interpolate(values,u,v):
                    row=np.minimum((v*12).astype(int),11)
                    f=(v*12-row)[...,None]
                    left=values[row*2]*(1-f)+values[(row+1)*2]*f
                    right=values[row*2+1]*(1-f)+values[(row+1)*2+1]*f
                    return left*(1-u[...,None])+right*u[...,None]
                cu,cv=np.meshgrid((np.arange(nx)+.5)/nx,(np.arange(ny)+.5)/ny)
                tc=interpolate(uv,cu,cv)
                ix=np.clip((tc[...,0]*alpha.shape[1]).astype(int),0,alpha.shape[1]-1)
                iy=np.clip(((1-tc[...,1])*alpha.shape[0]).astype(int),0,alpha.shape[0]-1)
                rows,cols=np.nonzero(alpha[iy,ix]>64)
                if not len(rows):raise ValueError('Empty thermal flame silhouette: '+path)
                u=np.stack([cols,cols+1,cols+1,cols],axis=1)/nx
                v=np.stack([rows,rows,rows+1,rows+1],axis=1)/ny
                points=interpolate(original,u,v).reshape(-1,3)
                mesh=UsdGeom.Mesh(stage.GetPrimAtPath(path))
                mesh.GetPointsAttr().Set([tuple(p) for p in points])
                mesh.GetFaceVertexCountsAttr().Set([4]*len(rows))
                mesh.GetFaceVertexIndicesAttr().Set(list(range(len(points))))
                mesh.CreateExtentAttr().Set([tuple(points.min(axis=0)),tuple(points.max(axis=0))])
                UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Set([(0.,0.)]*len(points))
                UsdShade.MaterialBindingAPI(mesh).Bind(material)
                total+=len(rows)
            print('THERMAL_FLAME_SILHOUETTES',len(originals),'meshes',total,'faces',flush=True)
        yield len(originals)
    finally:
        session.subLayerPaths.remove(layer.identifier)
