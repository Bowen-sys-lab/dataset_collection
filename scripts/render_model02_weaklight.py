"""Capture actual Isaac RGB/IR scene checks; never creates a manual-success label."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def assess_views(views):
    """Image usability checks, deliberately not a lux or sensor calibration."""
    minimum_visible = {'overview':.08,'building':.2,'tank':.15}
    by_name = {view['view']:view for view in views}
    issues = []
    for name,minimum in minimum_visible.items():
        if name not in by_name:
            issues.append(name+': missing view')
            continue
        view = by_name[name]
        if view['nonblack_fraction'] < minimum:
            issues.append(name+': nearly black; insufficient visible structure')
        if view['saturated_fraction'] > .02:
            issues.append(name+': excessive saturation')
        if not view['thermal_apparent_temperature_finite']:
            issues.append(name+': non-finite thermal output')
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/model02_weaklight_scene.yaml')
    parser.add_argument('--refined',action='store_true',help='Include refined facade/window diagnostics at higher thermal resolution')
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless':True,'renderer':'PathTracing','multi_gpu':False,
                         'samples_per_pixel_per_frame':16})
    preview = None
    try:
        import carb
        import numpy as np
        import omni.timeline
        from pxr import Gf, UsdGeom
        from scripts.scene_config import load_scene_config
        from scripts.start_crazyflie_dual_camera_view import open_preview
        config = load_scene_config(args.config)
        preview,_ = open_preview(app,scene_config=config,width=960,height=768,
            thermal_width=640 if args.refined else 320,thermal_height=512 if args.refined else 256,
            fast=False,start_controller=False,show_ui=False)
        assert preview.metadata['authored_lighting']['configuration_verified']
        stage = preview.stage
        stage.SetEditTarget(stage.GetSessionLayer())
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause();timeline.set_auto_update(False)
        views = [('overview',(646.,-867.,44.),(700.,-809.,3.)),
                 ('building',(693.,-847.,7.),(716.,-825.,2.8)),
                 ('tank',(693.37,-797.25,3.5),(693.672056966,-789.8881293,2.8340754))]
        if args.refined:
            views.extend([('facade',(740.,-850.,8.),(720.,-825.,2.8)),
                          ('window_detail',(718.,-836.,4.),(718.3,-828.8,3.65))])
        summary = []
        for index,(name,eye,target) in enumerate(views):
            transform = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*target),Gf.Vec3d(0,0,1)).GetInverse()
            for path in [preview.mount.rgb_path,preview.mount.thermal_path]:
                prim = stage.GetPrimAtPath(path)
                parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(0)
                UsdGeom.Xformable(prim).MakeMatrixXform().Set(transform*parent.GetInverse())
            timeline.set_current_time(0.);timeline.commit()
            rgb,thermal,meta = preview.capture(render=True,simulation_time=0.,frozen_state=True)
            if args.refined and not meta['thermal_coverage']['complete_for_this_view']:
                raise RuntimeError('Refined diagnostic has unmapped thermal pixels: '+name)
            meta.update(view_name=name,data_usage='scene validation only',manual_success=False)
            preview.save_frame(args.out,index,rgb,thermal,meta)
            pixels = np.asarray(rgb)[...,:3].astype(float)
            luminance = pixels@np.array([.2126,.7152,.0722])
            summary.append({'view':name,'frame':index,'eye':list(eye),'target':list(target),
                'rgb_luma_percentiles_8bit':np.percentile(luminance,[5,50,95,99]).tolist(),
                'nonblack_fraction':float((luminance>3).mean()),
                'saturated_fraction':float((pixels>=254).all(axis=-1).mean()),
                'thermal_apparent_temperature_finite':bool(np.isfinite(thermal.apparent_t).all())})
            print('[weaklight] captured '+name,flush=True)
        issues = assess_views(summary)
        report = {'result':'FAIL' if issues else 'IMAGE_CHECKS_PASS_VISUAL_REVIEW_REQUIRED',
                  'issues':issues,'purpose':'Actual RGB/IR lighting diagnostics, not manual trajectory collection',
                  'authored_lighting':preview.metadata['authored_lighting'],
                  'conditions':preview.metadata['experiment_conditions'],'views':summary,
                  'fixture_geometry':'75 fixture mesh parts' if args.refined else '41 independently bound fixture mesh parts',
                  'remaining_model_refinement':('Other preserved source components are listed in the v2 backlog'
                      if args.refined else 'Building/tank fine thermal partitions are a separate remaining plan task')}
        (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print('[weaklight] '+report['result'],flush=True)
        if issues:
            raise RuntimeError('Weak-light image acceptance failed: '+'; '.join(issues))
    finally:
        if preview:
            preview.close()
        app.close()


if __name__ == '__main__':
    main()
