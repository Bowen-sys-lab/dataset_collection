"""Shared static-scene protection and append-only recovery for manual teaching."""
import copy
import math
from scripts.recording_undo import active_branch

RECOVERY_CONTRACT = 'static_sphere_undo_v2'


def collision_undo_enabled(*, record_trajectory=True, probe_control=False, requested=None):
    """Manual recordings inherit protection, independent of scene name."""
    if requested and not record_trajectory:
        raise ValueError('--collision-undo requires --record-trajectory')
    return bool(record_trajectory and (not probe_control if requested is None else requested))


class SphereGuard:
    def __init__(self, query, drone_root, radius, vector=None):
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError('Positive finite collision radius required')
        if vector is None:
            import carb
            vector=lambda xyz:carb.Float3(*xyz)
        self.query,self.drone_root,self.radius,self.vector=query,drone_root,radius,vector

    def __call__(self, old, new):
        if not all(math.isfinite(float(v)) for v in (*old,*new)):
            raise ValueError('Nonfinite collision query pose')
        hits=[]
        def report(hit):
            path=hit.collision
            if path != self.drone_root and not path.startswith(self.drone_root+'/'):
                hits.append({'object_prim_path':path,'rigid_body_path':hit.rigid_body,
                             'distance_m':float(getattr(hit,'distance',0.))})
            return True
        self.query.overlap_sphere(self.radius,self.vector(old),report,False)
        delta=[b-a for a,b in zip(old,new)]
        distance=math.sqrt(sum(v*v for v in delta))
        if not hits and distance>1e-10:
            self.query.sweep_sphere_all(self.radius,self.vector(old),
                self.vector([v/distance for v in delta]),distance,report,True)
        return min(hits,key=lambda hit:hit['distance_m']) if hits else None


def prepare_static_colliders(stage, drone_root, *, exclude_paths=()):
    """Cover static meshes and native solid shapes, using session overrides only."""
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf
    root = Sdf.Path(drone_root)
    excluded = [Sdf.Path(path) for path in exclude_paths]
    for path in excluded:
        if not stage.GetPrimAtPath(path) or path.HasPrefix(root) or root.HasPrefix(path):
            raise ValueError(f'Invalid collision exclusion: {path}')
    meshes, shapes = [], []
    native_shapes = (UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cylinder, UsdGeom.Cone)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        while True:
            instances = [p for p in stage.Traverse() if p.IsInstance() and not p.GetPath().HasPrefix(root)
                         and not any(p.GetPath().HasPrefix(path) for path in excluded)]
            if not instances:
                break
            for prim in instances:
                prim.SetInstanceable(False)
        for prim in stage.Traverse():
            path = prim.GetPath()
            if path.HasPrefix(root):
                continue
            if any(path.HasPrefix(exclude) for exclude in excluded):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
                continue
            mesh = prim.IsA(UsdGeom.Mesh)
            if not mesh and not any(prim.IsA(shape) for shape in native_shapes):
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
                continue
            # Animated UVs/appearance do not move surfaces; animated geometry or
            # transforms (including ancestors) cannot use this static guard.
            current = prim
            while current and not current.IsPseudoRoot():
                spatial = ('points', 'faceVertexCounts', 'faceVertexIndices', 'size', 'radius', 'height')
                if any(attr.GetNumTimeSamples() for attr in current.GetAttributes()
                       if attr.GetName().startswith('xformOp:') or current == prim and attr.GetName() in spatial):
                    raise ValueError(f'Animated obstacle unsupported by static collision guard: {path}')
                if current.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(current).GetRigidBodyEnabledAttr().Get():
                    raise ValueError(f'Dynamic obstacle unsupported by static collision guard: {path}')
                current = current.GetParent()
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
            if mesh:
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set('none')
                meshes.append(str(path))
            else:
                shapes.append(str(path))
    if not meshes and not shapes:
        raise ValueError('No visible static collision geometry found')
    return {'mesh_count': len(meshes), 'mesh_paths': meshes,
            'shape_count': len(shapes), 'shape_paths': shapes,
            'excluded_visual_paths': list(exclude_paths)}


def prepare_guard(stage, drone_root, body_path, *, exclude_paths=()):
    """Never save source layers or convexify streets; full-body swept sphere."""
    import numpy as np
    from pxr import UsdGeom
    from omni.physx import get_physx_scene_query_interface
    from scripts.drone_envelope import measure_drone_envelope
    envelope=measure_drone_envelope(stage,drone_root,body_path)
    root=stage.GetPrimAtPath(drone_root)
    root_position=np.asarray(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(0))[3,:3]
    body_position=np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(body_path)).ComputeLocalToWorldTransform(0))[3,:3]
    radius=envelope['rotation_envelope_radius_m']+float(np.linalg.norm(body_position-root_position))+.02
    coverage=prepare_static_colliders(stage,drone_root,exclude_paths=exclude_paths)
    metadata={'schema':RECOVERY_CONTRACT,
              'scope':'visible static meshes and solid shapes plus existing colliders; configured visual effects excluded; not independent collision certification',
              'runtime_session_only':True,**coverage,
              'sphere_radius_m':radius,'drone_envelope':envelope,'undo_seconds':3.,
              'time_semantics':'monotonic raw clock; corrected branch compresses discarded/wait time'}
    return SphereGuard(get_physx_scene_query_interface(),drone_root,radius),metadata


class RecordingRecovery:
    def __init__(self,recorder,clock,guard):
        self.recorder,self.clock,self.guard=recorder,clock,guard
        self.resume_time=-math.inf
        self.pending_anchor=None

    def collision(self,detail):
        self.recorder.record_collision(float(self.clock()),detail)
        self.recorder.flush()
        print('[collision] held at obstacle; click Undo collision: back 3 seconds',flush=True)

    def prepare_undo(self):
        r=self.recorder
        if not r.metadata.get('collision_pending'):raise ValueError('No pending collision')
        branch,_=active_branch(r.records,r.metadata)
        collision=next(e for e in reversed(r.metadata['collection_edit_events']) if e['type']=='collision')
        cutoff=collision['simulation_time_s']-3.
        eligible=[row for row in branch if row.get('source_simulation_time_s',row['simulation_time'])<=cutoff]
        candidates=list(reversed(eligible)) if eligible else branch[:1]
        for row in candidates:
            pose=row.get('controller_target_pose')
            if not pose:raise ValueError('Safe anchor has no recorded controller root pose')
            if self.guard(pose['position'],pose['position']) is None:
                self.pending_anchor=row['frame_id']
                return copy.deepcopy(pose)
        raise ValueError('No safe recorded anchor; end this attempt and start a new RunId')

    def commit_undo(self):
        stamp=float(self.clock())
        self.recorder.undo_to(self.pending_anchor,stamp)
        self.pending_anchor=None
        self.resume_time=stamp+float(self.recorder.metadata['sample_period_s'])
        self.recorder.flush()
        print(f'[collision] restored; next saved sample at {self.resume_time:.3f}s',flush=True)

    def can_sample(self,stamp):
        return not self.recorder.metadata.get('collision_pending') and stamp>=self.resume_time-1e-9
