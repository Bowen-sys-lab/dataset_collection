"""Render temporary, read-only factory selection views with Isaac Sim."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, default=ROOT/'factory/import_fixed/factory_scene_fixed.usd')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--views', type=Path, required=True,
                        help='JSON list of name, eye, target, optional focal_length')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'renderer': 'PathTracing',
                         'multi_gpu': False, 'samples_per_pixel_per_frame': 8})
    try:
        import omni.usd
        import omni.replicator.core as rep
        import numpy as np
        from PIL import Image
        from pxr import Gf, UsdGeom, Sdf
        context = omni.usd.get_context()
        if not context.open_stage(str(args.stage.resolve())):
            raise RuntimeError('Could not open factory')
        for _ in range(90):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        camera = UsdGeom.Camera.Define(stage, '/FactoryInspectionTemporaryCamera')
        camera.CreateClippingRangeAttr(Gf.Vec2f(.1, 10000))
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(27.)
        color = rep.AnnotatorRegistry.get_annotator('LdrColor')
        product = rep.create.render_product(str(camera.GetPath()), (1280, 960))
        color.attach(product)
        views = json.loads(args.views.read_text(encoding='utf-8-sig'))
        for view in views:
            eye, target = Gf.Vec3d(*view['eye']), Gf.Vec3d(*view['target'])
            up = Gf.Vec3d(*view.get('up', [0, 0, 1]))
            camera.CreateFocalLengthAttr(float(view.get('focal_length', 35.)))
            UsdGeom.Xformable(camera).MakeMatrixXform().Set(
                Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse())
            for _ in range(3):
                rep.orchestrator.step(rt_subframes=8, delta_time=0, pause_timeline=True,
                                      wait_for_render=True)
            pixels = np.asarray(color.get_data())
            Image.fromarray(pixels[..., :3]).save(args.out/(view['name']+'.png'))
            print('[factory view] '+view['name'], flush=True)
        color.detach(product)
        product.destroy()
    finally:
        app.close()


if __name__ == '__main__':
    main()
