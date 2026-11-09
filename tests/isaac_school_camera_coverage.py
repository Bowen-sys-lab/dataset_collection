"""Audit the school's real view without marker/backdrop or UAV controller."""
import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.scene_config import load_scene_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/school_vln_scene.yaml')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=None, help='temporary inspection render width')
    parser.add_argument('--height', type=int, default=None, help='temporary inspection render height')
    parser.add_argument('--spot-view-distance', type=float, default=None,
                        help='optional overview distance in meters for the temporary Spot inspection view')
    args = parser.parse_args()
    config = load_scene_config(args.config)
    args.out.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'PathTracing', 'multi_gpu': False,
                         'samples_per_pixel_per_frame': 1})
    preview = stage = layer = original_edit = None
    report = {'result': 'FAIL', 'strict_sync_verified': False, 'source_usd_sha256_before': before}
    status = 1
    try:
        import omni.usd
        import omni.timeline
        from pxr import UsdPhysics, UsdGeom, Sdf, Gf
        import numpy as np
        context = omni.usd.get_context()
        if not context.open_stage(str(config.scene.stage)):
            raise RuntimeError('Could not open school stage')
        for _ in range(180):
            app.update()
            if not any(context.get_stage_loading_status()):
                break
        stage = context.get_stage()
        original_edit = stage.GetEditTarget()
        layer = Sdf.Layer.CreateAnonymous('school_coverage.usda')
        stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
        stage.SetEditTarget(layer)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
            if prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
        from scripts.start_crazyflie_dual_camera_view import open_preview
        preview, _ = open_preview(app, scene_config=config, stage_path=config.scene.stage,
                                  width=args.width, height=args.height,
                                  start_controller=False)
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
        timeline.set_auto_update(False)
        timeline.set_current_time(0.)
        timeline.commit()
        rgb, thermal, meta = preview.capture(render=True, frozen_state=True)
        preview.save_frame(args.out, 0, rgb, thermal, meta)
        report['thermal_coverage'] = meta.get('thermal_coverage', {})
        # A separate, temporary inspection viewpoint proves Spot is renderable.
        # It is not the starting flight view and does not change the saved rig.
        spot_path = '/World/Target_01'
        spot = stage.GetPrimAtPath(spot_path)
        bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                    UsdGeom.Tokens.proxy]).ComputeWorldBound(spot).ComputeAlignedRange()
        center = bounds.GetMidpoint()
        extent = bounds.GetSize()
        distance = (max(3., float(extent.GetLength()) * 1.5)
                    if args.spot_view_distance is None else max(3., args.spot_view_distance))
        eye = center + Gf.Vec3d(distance*.6, -distance*.8, distance*.35)
        world = Gf.Matrix4d().SetLookAt(eye, center, Gf.Vec3d(0, 0, 1)).GetInverse()
        for camera_path in (preview.mount.rgb_path, preview.mount.thermal_path):
            prim = stage.GetPrimAtPath(camera_path)
            parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(world * parent.GetInverse())
        timeline.set_current_time(.2)
        timeline.commit()
        rgb, thermal, meta = preview.capture(render=True, frozen_state=True)
        preview.save_frame(args.out, 1, rgb, thermal, meta)
        last = preview.thermal._last
        spot_ids = [i for i, path in last.labels.items() if path == spot_path or path.startswith(spot_path+'/')]
        visible = np.isin(last.instance_id, spot_ids)
        material_mapped = bool(visible.any() and np.all(last.material_id[visible] > 0))
        report['spot_inspection'] = {
            'target_path': spot_path, 'world_bbox_min': list(bounds.GetMin()),
            'world_bbox_max': list(bounds.GetMax()), 'world_center': list(center),
            'visible_pixels': int(visible.sum()), 'infrared_material_mapped': material_mapped,
            'thermal_coverage': meta['thermal_coverage'],
            'view_mode': 'temporary camera aimed at Spot; not initial drone view',
            'camera_distance_m': distance,
        }
        report['source_usd_sha256_after'] = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
        if report['source_usd_sha256_after'] != before:
            raise RuntimeError('Source USD changed')
        if not report['thermal_coverage'].get('complete_for_this_view'):
            raise RuntimeError('Actual school view contains unmapped infrared surfaces')
        if not material_mapped or not meta['thermal_coverage']['complete_for_this_view']:
            raise RuntimeError('Spot inspection failed: not visible/mapped or incomplete thermal view')
        report['result'] = 'PASS'
        status = 0
    except Exception:
        report['error'] = traceback.format_exc()
        (args.out/'failure.txt').write_text(report['error'], encoding='utf-8')
        print(report['error'], flush=True)
    finally:
        report['source_usd_sha256_after'] = hashlib.sha256(config.scene.stage.read_bytes()).hexdigest()
        report['source_unchanged'] = report['source_usd_sha256_after'] == before
        if not report['source_unchanged']:
            report['result'] = 'FAIL'
            status = 1
        if preview:
            preview.close()
        if layer:
            stage.SetEditTarget(original_edit)
            stage.GetSessionLayer().subLayerPaths.remove(layer.identifier)
        (args.out/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
        app.close()
    return status


if __name__ == '__main__':
    raise SystemExit(main())
