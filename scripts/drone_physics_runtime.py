"""Isaac Sim 6.1 adapter: all live poses and velocities use PhysX tensors."""
from contextlib import contextmanager


@contextmanager
def tensor_backend():
    from isaacsim.core.experimental.utils.backend import use_backend
    with use_backend('tensor', raise_on_fallback=True):
        yield


class DronePhysicsRuntime:
    def __init__(self, prim_path):
        import omni.usd
        from pxr import Usd, UsdPhysics, PhysxSchema
        from isaacsim.core.experimental.prims import Articulation
        self.context = omni.usd.get_context()
        self.stage = self.context.get_stage()
        self.prim = self.stage.GetPrimAtPath(prim_path) if self.stage else None
        if not self.prim or not self.prim.IsValid():
            raise RuntimeError(f'Cannot find drone: {prim_path}')
        if not self.prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            raise RuntimeError(f'No ArticulationRootAPI: {prim_path}')
        # Temporary overrides; do not save physics settings into the asset.
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            for prim in Usd.PrimRange(self.prim):
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(False)
                    PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(True)
        self.articulation = Articulation(prim_path)

    @property
    def valid(self):
        return self.context.get_stage() == self.stage and self.prim.IsValid()

    @property
    def ready(self):
        return self.valid and self.articulation.is_physics_tensor_entity_valid()

    def snapshot(self):
        if not self.ready:
            return {'valid': self.valid, 'ready': False}
        with tensor_backend():
            p, q = self.articulation.get_world_poses()
            v, w = self.articulation.get_velocities()
            gravity = self.articulation.get_link_enabled_gravities()
        return {'valid': True, 'ready': True,
                'position': p.numpy()[0].tolist(), 'orientation': q.numpy()[0].tolist(),
                'linear_velocity': v.numpy()[0].tolist(), 'angular_velocity': w.numpy()[0].tolist(),
                'gravity_enabled': gravity.numpy()[0].tolist()}

    def prepare_hover(self):
        with tensor_backend():
            self.articulation.set_link_enabled_gravities(False)
            self.articulation.set_velocities(linear_velocities=[[0., 0., 0.]],
                                            angular_velocities=[[0., 0., 0.]])
            if self.articulation.get_link_enabled_gravities().numpy().any():
                raise RuntimeError('Gravity is still enabled on a drone link')

    def write_pose(self, position, orientation):
        with tensor_backend():
            self.articulation.set_world_poses(positions=[list(position)], orientations=[list(orientation)])
            self.articulation.set_velocities(linear_velocities=[[0., 0., 0.]],
                                            angular_velocities=[[0., 0., 0.]])
