"""Derive visible meshes from authored material subsets without changing source USD."""


def verify_overlay_source(stage):
    """Reject a changed direct source USD in the versioned City overlay.

    Legacy scenes have no pin. This pin covers the direct v2 layer, not every
    transitive texture/reference; retain the full original asset bundle too.
    """
    import hashlib
    from pathlib import Path
    layer=stage.GetRootLayer()
    if layer.customLayerData.get('thermalRevision')!='city_material_partition_v3':
        return None
    expected=layer.customLayerData.get('sourceStageSha256')
    if not layer.realPath or len(layer.subLayerPaths)!=1 or not isinstance(expected,str) or len(expected)!=64:
        raise ValueError('City material-partition overlay requires one pinned source USD')
    source=(Path(layer.realPath).parent/layer.subLayerPaths[0]).resolve()
    actual=hashlib.sha256(source.read_bytes()).hexdigest()
    if actual!=expected:
        raise ValueError(f'City overlay source USD changed: {source}; expected {expected}, got {actual}. Restore the original source or build a new version.')
    return {'source_stage_path':str(source),'source_stage_sha256':actual,'verified':True,
            'scope':'direct source USD only; preserve transitive source assets separately'}


def partition_mesh(stage, path, classify):
    """Partition static polygon meshes by thermal class; keep RGB subset bindings.

    Writes only to the caller's edit layer. Original invisible mesh remains for
    provenance; siblings inherit the same parent and local transform. Fail closed
    on unsupported topology rather than silently dropping a face or primvar.
    """
    from collections import defaultdict
    import numpy as np
    from pxr import Usd, UsdGeom, UsdShade
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
    prim = mesh.GetPrim()
    subsets = UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets()
    if not subsets:
        return []
    if mesh.GetSubdivisionSchemeAttr().Get() != 'none' or mesh.GetHoleIndicesAttr().Get():
        raise ValueError(f'unsupported subdivision/holes: {path}')
    if any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()):
        raise ValueError(f'animated mesh: {path}')
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    points = mesh.GetPointsAttr().Get()
    offsets = np.concatenate(([0], np.cumsum(counts)))
    groups = defaultdict(list)
    seen = set()
    for sub in subsets:
        faces = list(sub.GetIndicesAttr().Get())
        if any(f < 0 or f >= len(counts) or f in seen for f in faces) or len(set(faces)) != len(faces):
            raise ValueError(f'overlap or invalid subset indices: {path}')
        seen.update(faces)
        mat, _ = UsdShade.MaterialBindingAPI(sub).ComputeBoundMaterial()
        groups[classify(mat)].append((sub.GetPrim().GetName(), faces, mat))
    remaining = sorted(set(range(len(counts))) - seen)
    if remaining:
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        groups[classify(mat)].append(('Remainder', remaining, mat))
    parts = []
    for category, assignments in sorted(groups.items()):
        faces = [f for _, fs, _ in assignments for f in fs]
        if not faces:
            continue
        corners = np.concatenate([np.arange(offsets[f], offsets[f+1]) for f in faces])
        vertices, remapped = np.unique(indices[corners], return_inverse=True)
        new_path = prim.GetPath().GetParentPath().AppendChild(prim.GetName()+'__thermal_'+category)
        if stage.GetPrimAtPath(new_path):
            raise ValueError(f'partition already exists: {new_path}')
        part = UsdGeom.Mesh.Define(stage, new_path)
        part.CreateSubdivisionSchemeAttr('none')
        part.CreatePointsAttr(type(points)([points[int(i)] for i in vertices]))
        part.CreateFaceVertexCountsAttr(counts[faces].tolist())
        part.CreateFaceVertexIndicesAttr(remapped.tolist())
        part.MakeMatrixXform().Set(mesh.GetLocalTransformation())
        part.CreateDoubleSidedAttr(mesh.GetDoubleSidedAttr().Get())
        part.CreateOrientationAttr(mesh.GetOrientationAttr().Get())
        part.CreatePurposeAttr(mesh.GetPurposeAttr().Get())
        normals = mesh.GetNormalsAttr().Get()
        arrays = []
        if normals is not None and len(normals):
            arrays.append((mesh.GetNormalsInterpolation(), normals, part.CreateNormalsAttr(), None))
            part.SetNormalsInterpolation(mesh.GetNormalsInterpolation())
        for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if not pv.HasValue():
                continue
            value = pv.ComputeFlattened()
            out = UsdGeom.PrimvarsAPI(part).CreatePrimvar(pv.GetPrimvarName(), pv.GetTypeName(), pv.GetInterpolation(), pv.GetElementSize())
            arrays.append((pv.GetInterpolation(), value, out.GetAttr(), pv.GetElementSize()))
        for interp, value, attr, elem in arrays:
            if elem not in (None,1):
                raise ValueError(f'unsupported primvar element size: {path}')
            selection = {'faceVarying': corners, 'uniform': faces, 'vertex': vertices, 'varying': vertices}.get(interp)
            attr.Set(value if selection is None else type(value)([value[int(i)] for i in selection]))
        start = 0
        for name, fs, mat in assignments:
            sub = UsdShade.MaterialBindingAPI(part).CreateMaterialBindSubset(name, list(range(start, start+len(fs))))
            if mat:
                UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(mat)
            start += len(fs)
        parts.append({'path': str(new_path), 'category': category, 'faces': len(faces), 'source': str(path)})
    mesh.CreateVisibilityAttr('invisible')
    return parts
