"""Measure the composed drone and conservatively bound its four rotor rotations."""
import hashlib
import itertools
import json
import numpy as np


def measure_drone_envelope(stage, drone_root, body_path):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdSkel
    from scripts.replay_pose import world_matrix
    root = stage.GetPrimAtPath(drone_root)
    body = stage.GetPrimAtPath(body_path)
    if not root or not body or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('A composed drone body in metre units is required')
    body_world = world_matrix(body)
    rotation = body_world[:3, :3]
    if (not np.isfinite(body_world).all() or not np.allclose(rotation @ rotation.T, np.eye(3), rtol=0, atol=1e-8)
        or np.linalg.det(rotation) < 0):
        raise ValueError('Composed body must have a rigid unit-scale world transform')
    inverse = np.linalg.inv(body_world)
    cache = UsdGeom.XformCache()
    geometry = {}
    joints = []
    digest = hashlib.sha256()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(prim).TransformMightBeTimeVarying():
            raise ValueError(f'Authored drone animation is not bounded: {path}')
        if prim.IsA(UsdPhysics.Joint):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                joints.append(UsdPhysics.RevoluteJoint(prim))
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError(f'Unsupported moving drone joint: {path}')
        if not prim.IsA(UsdGeom.Boundable) or prim.HasAPI(UsdLux.LightAPI):
            continue
        if any(attr.GetNumTimeSamples() for attr in prim.GetAttributes()) or prim.HasAPI(UsdSkel.BindingAPI):
            raise ValueError(f'Animated or skinned drone geometry is not bounded: {path}')
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        else:
            extent = UsdGeom.Boundable.ComputeExtentFromPlugins(UsdGeom.Boundable(prim), Usd.TimeCode.Default())
            if extent is None or len(extent) != 2:
                raise ValueError(f'Unsupported drone boundable: {path}')
            points = np.array(list(itertools.product(*zip(extent[0], extent[1]))), dtype=float)
        if points.size == 0:
            continue
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
            raise ValueError(f'Invalid drone geometry: {path}')
        with np.errstate(over='raise', invalid='raise'):
            local = (np.c_[points, np.ones(len(points))] @ np.asarray(cache.GetLocalToWorldTransform(prim)) @ inverse)[:, :3]
        if not np.isfinite(local).all():
            raise ValueError(f'Nonfinite composed drone geometry: {path}')
        geometry[path] = local
        digest.update(path.encode('utf-8'))
        digest.update(local.astype('<f8').tobytes())
    if not geometry or len(joints) != 4:
        raise ValueError('Complete composed geometry and exactly four revolute rotor joints required')
    rotors = []
    for joint in joints:
        base = joint.GetBody0Rel().GetTargets()
        targets = joint.GetBody1Rel().GetTargets()
        if list(map(str, base)) != [body_path] or len(targets) != 1:
            raise ValueError('Rotor must rotate directly relative to the measured body')
        target = str(targets[0])
        vertices = [points for path, points in geometry.items() if path == target or path.startswith(target + '/')]
        if not vertices:
            raise ValueError(f'Rotor geometry is missing: {target}')
        rotor_transform = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target))) @ inverse
        pivot_local = np.asarray(joint.GetLocalPos1Attr().Get(), dtype=float)
        pivot_body = (np.r_[pivot_local, 1.] @ rotor_transform)[:3]
        points_body = np.concatenate(vertices)
        rotor_points = (np.c_[points_body, np.ones(len(points_body))] @ np.linalg.inv(rotor_transform))[:, :3]
        local_radius = float(np.linalg.norm(rotor_points - pivot_local, axis=1).max())
        maximum_scale = float(np.linalg.svd(rotor_transform[:3, :3], compute_uv=False).max())
        radius = float(np.linalg.norm(pivot_body)) + maximum_scale * local_radius
        rotors.append({'path': target, 'joint': str(joint.GetPath()), 'pivot_body_m': pivot_body.tolist(),
                       'local_radius_m': local_radius, 'maximum_scale': maximum_scale,
                       'rotation_sphere_bound_m': radius})
    paths = [r['path'] for r in rotors]
    if len(set(paths)) != 4 or any(a.startswith(b + '/') for a in paths for b in paths if a != b):
        raise ValueError('Overlapping or nested rotor bodies are not supported')
    digest.update(json.dumps(rotors, sort_keys=True).encode('utf-8'))
    static_radius = float(np.linalg.norm(np.concatenate(list(geometry.values())), axis=1).max())
    bound = max([static_radius] + [rotor['rotation_sphere_bound_m'] for rotor in rotors])
    if not np.isfinite(bound):
        raise ValueError('Composed envelope arithmetic is nonfinite')
    return {'result': 'PASS', 'geometry_sha256': digest.hexdigest(), 'body_path': body_path,
            'geometry_prims': len(geometry), 'rotors': rotors,
            'static_geometry_radius_m': static_radius, 'rotation_envelope_radius_m': bound}
