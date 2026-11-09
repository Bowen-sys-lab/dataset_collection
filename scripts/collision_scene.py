"""Build a bounded static collision scene and prove an exterior starting point."""
from dataclasses import dataclass, field
import math
import numpy as np
from scripts.continuous_collision import sweep_surface_segment


@dataclass
class CollisionScene:
    triangles: np.ndarray
    triangle_owners: list
    bounds: np.ndarray
    start: np.ndarray
    radius: float
    report: dict
    triangle_min: np.ndarray = field(init=False, repr=False)
    triangle_max: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.triangle_min = self.triangles.min(1)
        self.triangle_max = self.triangles.max(1)


def build_collision_scene(stage, *, start, bounds, radius, drone_root='/World/Crazyflie'):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdSkel
    start = np.asarray(start, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if (start.shape != (3,) or bounds.shape != (2, 3) or not np.isfinite(start).all()
        or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1])
        or not math.isfinite(radius) or radius <= 0):
        raise ValueError('Finite start, ordered 2x3 metre bounds and positive radius required')
    if UsdGeom.GetStageMetersPerUnit(stage) != 1 or UsdGeom.GetStageUpAxis(stage) != 'Z':
        raise ValueError('Collision scene requires world metres and Z-up')
    issues, ignored = [], []
    faces, owners, escape_faces = [], [], []
    maximum_z = -math.inf
    meshes = 0
    checked_ancestors = set()
    cache = UsdGeom.XformCache()
    for error in stage.GetCompositionErrors():
        path = str(getattr(getattr(error, 'rootSite', None), 'path', ''))
        if path == drone_root or path.startswith(drone_root + '/'):
            ignored.append({'path': path, 'reason': 'excluded robot asset; envelope measured separately'})
        else:
            issues.append({'path': path, 'reason': 'unresolved scene composition', 'detail': str(error)})
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path == drone_root or path.startswith(drone_root + '/'):
            continue
        if prim.HasAPI(UsdLux.LightAPI):
            ignored.append({'path': path, 'reason': 'light emitter'})
            continue
        if not prim.IsA(UsdGeom.Boundable):
            continue
        collider = prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            ancestor_path = str(ancestor.GetPath())
            if ancestor_path in checked_ancestors:
                break
            checked_ancestors.add(ancestor_path)
            if ancestor.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(ancestor).TransformMightBeTimeVarying():
                issues.append({'path': ancestor_path, 'reason': 'animated obstacle transform'})
            if ancestor.IsA(UsdGeom.Imageable) and UsdGeom.Imageable(ancestor).GetVisibilityAttr().GetNumTimeSamples():
                issues.append({'path': ancestor_path, 'reason': 'time-varying obstacle visibility'})
            ancestor = ancestor.GetParent()
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible and not collider:
            continue
        if not prim.IsA(UsdGeom.Mesh):
            issues.append({'path': path, 'reason': 'unsupported non-mesh obstacle', 'type': prim.GetTypeName()})
            continue
        mesh = UsdGeom.Mesh(prim)
        attributes = [mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr()]
        if any(attribute.GetNumTimeSamples() for attribute in attributes) or prim.HasAPI(UsdSkel.BindingAPI):
            issues.append({'path': path, 'reason': 'animated or skinned mesh'})
            continue
        if mesh.GetSubdivisionSchemeAttr().Get() != 'none':
            issues.append({'path': path, 'reason': 'subdivision surface requires evaluated geometry'})
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        if points.size == 0 and counts.size == 0 and indices.size == 0:
            ignored.append({'path': path, 'reason': 'authored empty mesh; no points or faces'})
            continue
        if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all()
            or not len(counts) or not np.all(counts == 3) or len(indices) != int(counts.sum())
            or np.any(indices < 0) or np.any(indices >= len(points))):
            issues.append({'path': path, 'reason': 'invalid or non-triangulated geometry'})
            continue
        with np.errstate(over='raise', invalid='raise'):
            world = (np.c_[points, np.ones(len(points))] @ np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]
        if not np.isfinite(world).all():
            raise ValueError(f'Nonfinite world geometry: {path}')
        maximum_z = max(maximum_z, float(world[:, 2].max()))
        meshes += 1
        triangles = world[indices.reshape(-1, 3)]
        lower, upper = triangles.min(1), triangles.max(1)
        keep = np.all(upper >= bounds[0], axis=1) & np.all(lower <= bounds[1], axis=1)
        faces.append(triangles[keep])
        owners.extend([path] * int(keep.sum()))
        escape = (np.all(upper[:, :2] >= start[:2] - radius - 1e-9, axis=1)
                  & np.all(lower[:, :2] <= start[:2] + radius + 1e-9, axis=1)
                  & (upper[:, 2] >= start[2] - radius - 1e-9))
        escape_faces.append(triangles[escape])
    triangles = np.concatenate(faces) if faces else np.empty((0, 3, 3))
    escape_triangles = np.concatenate(escape_faces) if escape_faces else np.empty((0, 3, 3))
    if not meshes:
        issues.append({'reason': 'no supported scene obstacle geometry'})
    if np.any(start - radius < bounds[0]) or np.any(start + radius > bounds[1]):
        issues.append({'reason': 'start sphere outside trial bounds'})
    escape_end = start.copy()
    escape_end[2] = max(start[2], maximum_z) + radius + 1
    if not np.isfinite(escape_end).all() or escape_end[2] <= maximum_z + radius:
        issues.append({'reason': 'cannot construct a finite exterior endpoint above the scene'})
    escape_result = {'status': 'NOT_CHECKED'}
    if not issues:
        escape_result = (sweep_surface_segment(start, escape_end, escape_triangles, radius=radius)
                         if len(escape_triangles) else {'status': 'CLEAR_SURFACES', 'distance_queries': 0})
    ready = not issues and escape_result['status'] == 'CLEAR_SURFACES'
    report = {'ready': ready, 'bounds_m': bounds.tolist(), 'start_m': start.tolist(), 'radius_m': radius,
              'supported_meshes': meshes, 'domain_triangles': len(triangles),
              'exterior_escape_triangles': len(escape_triangles), 'escape_end_m': escape_end.tolist(),
              'start_escape': escape_result, 'issues': issues, 'explicit_exclusions': ignored,
              'scope': 'Static composed mesh world, robot excluded and lights ignored; all other included geometry treated as hard surfaces'}
    return CollisionScene(triangles, owners, bounds, start, radius, report)


def check_path(scene, positions):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions) or not np.isfinite(positions).all():
        raise ValueError('Path must be a nonempty finite Nx3 array')
    if not scene.report['ready']:
        return {'status': 'NOT_READY', 'issues': scene.report['issues']}
    if not np.allclose(positions[0], scene.start, rtol=0, atol=1e-8):
        return {'status': 'START_MISMATCH'}
    if np.any(positions - scene.radius < scene.bounds[0]) or np.any(positions + scene.radius > scene.bounds[1]):
        return {'status': 'OUT_OF_BOUNDS'}
    for index, (start, end) in enumerate(zip(positions, positions[1:])):
        low = np.minimum(start, end) - scene.radius - 1e-9
        high = np.maximum(start, end) + scene.radius + 1e-9
        candidates = np.flatnonzero(np.all(scene.triangle_max >= low, axis=1)
                                   & np.all(scene.triangle_min <= high, axis=1))
        result = (sweep_surface_segment(start, end, scene.triangles[candidates], radius=scene.radius)
                  if len(candidates) else {'status': 'CLEAR_SURFACES'})
        if result['status'] != 'CLEAR_SURFACES':
            result['path_interval'] = [index, index + 1]
            if 'contact_triangle_index' in result:
                result['contact_triangle_index'] = int(candidates[result['contact_triangle_index']])
                result['object_prim_path'] = scene.triangle_owners[result['contact_triangle_index']]
            return result
    return {'status': 'CLEAR', 'segments_checked': max(0, len(positions) - 1)}
