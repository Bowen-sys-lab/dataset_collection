"""Runtime-only USD camera rig. Importing this module does not boot Isaac."""
import re


class CameraMount:
    """Use the existing camera as RGB and reuse/create one thermal camera.

    A camera nested under a mount has more than its own local transform. We
    compose its world transform relative to the moving link before reparenting.
    """

    def __init__(self, stage, source_path, body_path, *, align_thermal_to_rgb=False):
        from pxr import Sdf, Usd, UsdGeom, UsdRender
        source = UsdGeom.Camera(stage.GetPrimAtPath(source_path))
        body = stage.GetPrimAtPath(body_path)
        if not source or not body or not UsdGeom.Xformable(body):
            raise ValueError('source Camera and moving body must both exist')
        if not Sdf.Path(source_path).HasPrefix(Sdf.Path(body_path)):
            raise ValueError('source camera must be below the moving body link')
        self.root_path = body_path.rstrip('/') + '/Sensors'
        self.rgb_path = source_path
        self.thermal_path = self.root_path + '/ThermalCamera'
        if source_path == self.thermal_path:
            raise ValueError('RGB source must differ from the thermal camera')
        self.stage = stage
        self.old_target = stage.GetEditTarget()
        self.layer = Sdf.Layer.CreateAnonymous('dual_camera_preview.usda')
        self.closed = False
        self.source_focal = source.GetFocalLengthAttr().Get()
        self.source_aperture = source.GetHorizontalApertureAttr().Get()
        root = stage.GetPrimAtPath(self.root_path)
        if root and root.GetTypeName() not in ('', 'Xform', 'Scope'):
            raise ValueError(f'{self.root_path} must be a sensor container')
        thermal_prim = stage.GetPrimAtPath(self.thermal_path)
        if thermal_prim and not thermal_prim.IsA(UsdGeom.Camera):
            raise ValueError(f'{self.thermal_path} exists but is not a Camera')
        # An empty Kit viewport placeholder has no saved installation transform.
        keep_pose = bool(thermal_prim and
                         UsdGeom.Xformable(thermal_prim).GetXformOpOrderAttr().HasAuthoredValueOpinion())
        source_world = source.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self.camera_redirects = {}
        for child in body.GetChildren():
            if child.GetName() == 'Sensors' or re.fullmatch(r'Sensors_preview(?:_\d+)?', child.GetName()):
                for name in ('RGBCamera', 'ThermalCamera'):
                    old = str(child.GetPath()) + '/' + name
                    new = self.rgb_path if name == 'RGBCamera' else self.thermal_path
                    if old != new:
                        self.camera_redirects[old] = new
        stage.GetSessionLayer().subLayerPaths.insert(0, self.layer.identifier)
        try:
            stage.SetEditTarget(self.layer)
            if not root:
                root = UsdGeom.Xform.Define(stage, self.root_path).GetPrim()
            elif not root.GetTypeName():
                root.SetTypeName('Xform')
            camera = UsdGeom.Camera.Define(stage, self.thermal_path)
            if not keep_pose or align_thermal_to_rgb:
                parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(root)
                camera.MakeMatrixXform().Set(source_world * parent_world.GetInverse())
            for name in UsdGeom.Camera.GetSchemaAttributeNames(False):
                attr = camera.GetPrim().GetAttribute(name)
                value = source.GetPrim().GetAttribute(name).Get()
                if value is not None and not attr.HasAuthoredValueOpinion():
                    attr.Set(value)
            for prim in list(stage.Traverse()):
                if prim.IsA(UsdRender.Product):
                    # Saved preview render products are runtime leftovers. Avoid
                    # rendering extra sensor buffers and invalid camera targets.
                    if align_thermal_to_rgb and prim.GetName().startswith('Replicator'):
                        prim.SetActive(False)
                        continue
                    rel = UsdRender.Product(prim).GetCameraRel()
                    targets = rel.GetTargets()
                    updated = [Sdf.Path(self.camera_redirects.get(str(p), str(p))) for p in targets]
                    if updated != targets:
                        rel.SetTargets(updated)
            if align_thermal_to_rgb:
                for prim in stage.Traverse():
                    if prim.IsA(UsdRender.Settings):
                        products = UsdRender.Settings(prim).GetProductsRel()
                        products.SetTargets([p for p in products.GetTargets()
                                             if stage.GetPrimAtPath(p) and stage.GetPrimAtPath(p).IsActive()])
            for old in self.camera_redirects:
                prim = stage.GetPrimAtPath(old)
                if prim and prim.IsA(UsdGeom.Camera):
                    prim.SetActive(False)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stage.SetEditTarget(self.old_target)
        paths = self.stage.GetSessionLayer().subLayerPaths
        if self.layer.identifier in paths:
            paths.remove(self.layer.identifier)
