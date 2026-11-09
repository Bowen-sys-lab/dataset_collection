"""Read-only physics diagnostics for Isaac Sim's Script Editor."""

from __future__ import annotations


def print_drone_physics(root_path: str = "/Quadcopter") -> None:
    """Print rigid-body schemas and relevant attributes below ``root_path``."""
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Cannot find USD prim: {root_path}")

    print(f"[drone-diagnose] root={root.GetPath()}")
    found = False
    for prim in Usd.PrimRange(root):
        schemas = [str(schema) for schema in prim.GetAppliedSchemas()]
        relevant = [
            schema for schema in schemas
            if "RigidBody" in schema or "Mass" in schema or "Collision" in schema
        ]
        attributes = []
        for attribute in prim.GetAttributes():
            name = str(attribute.GetName()).lower()
            if any(word in name for word in ("gravity", "kinematic", "mass")):
                attributes.append(f"{attribute.GetName()}={attribute.Get()}")
        if not relevant and not attributes:
            continue
        found = True
        print(f"[drone-diagnose] prim={prim.GetPath()}")
        print(f"  schemas={relevant or 'none'}")
        print(f"  attributes={attributes or 'none'}")

    if not found:
        print("[drone-diagnose] no rigid-body-related schema or attribute found below root")
    print("[drone-diagnose] read-only check complete")


def enable_direct_pose_mode(root_path: str = "/Quadcopter") -> int:
    """Configure the referenced drone for pose control while preserving articulation."""
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open.")
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Cannot find USD prim: {root_path}")

    changed = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        # Keep every body dynamic so the asset's articulation remains valid.
        # Disable gravity because the teaching controller writes the pose.
        api.CreateKinematicEnabledAttr().Set(False)
        physx_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_api.CreateDisableGravityAttr().Set(True)
        print(f"[drone-setup] dynamic/no-gravity on {prim.GetPath()}")
        changed += 1
    print(f"[drone-setup] direct-pose mode configured for {changed} rigid bodies")
    return changed
