"""Evaluate one recorded neighbourhood task run: T mark plus three automatic checks.

The human T press only says "I think I found it".  This adds the parts a report can defend:

1. **success marked** -- the run actually has a T press;
2. **distance** -- the marked position is within ``--max-distance`` of the task target;
3. **target in frame** -- the target's bounding box projects inside the RGB image at the marked
   moment (a geometric check, no occlusion test);
4. **hover** -- the aircraft stayed within the distance and below a speed threshold for at least
   ``--min-hover`` seconds before marking.

Usage:
    python scripts/evaluate_neighbourhood_task.py --run validation/trajectory_runs/night_ac_001
    python scripts/evaluate_neighbourhood_task.py --run <dir> --report out.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_run(run_dir: Path):
    meta = json.loads((run_dir / 'run_metadata.json').read_text(encoding='utf-8'))
    samples = [json.loads(line) for line in
               (run_dir / 'trajectory.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    return meta, samples


def _camera_intrinsics(scene_config_path: str | None):
    """Focal length and apertures from the scene's RGB camera, in millimetres."""
    from pxr import Usd, UsdGeom

    if not scene_config_path:
        return 18.0, 20.955, 16.764
    import yaml
    cfg = yaml.safe_load(Path(scene_config_path).read_text(encoding='utf-8'))
    stage_path = (Path(scene_config_path).parent / cfg['scene']['stage']).resolve()
    stage = Usd.Stage.Open(str(stage_path))
    camera = UsdGeom.Camera(stage.GetPrimAtPath(cfg['drone']['camera_path']))
    return (float(camera.GetFocalLengthAttr().Get()),
            float(camera.GetHorizontalApertureAttr().Get()),
            float(camera.GetVerticalApertureAttr().Get()))


def _project(point, matrix, focal, aperture_h, aperture_v, width, height):
    import numpy as np

    m = np.asarray(matrix, dtype=float)
    rotation, translation = m[:3, :3], m[3, :3]
    local = rotation.T @ (np.asarray(point, dtype=float) - translation)
    if local[2] >= -1e-6:
        return None
    u = (local[0] / -local[2]) * (focal / (aperture_h / 2.0))
    v = (local[1] / -local[2]) * (focal / (aperture_v / 2.0))
    return (width / 2.0 * (1.0 + u), height / 2.0 * (1.0 - v))


def evaluate(run_dir: Path, max_distance=5.0, min_hover=3.0, hover_speed=0.3):
    meta, samples = _load_run(run_dir)
    task = meta.get('task') or {}
    target = task.get('target_position')
    result = {'run_dir': str(run_dir), 'task_id': task.get('task_id'), 'checks': {}, 'metrics': {}}
    if not target:
        result['checks']['task_present'] = {'ok': False, 'detail': 'run has no task attached'}
        return result
    result['checks']['task_present'] = {'ok': True,
                                        'detail': task.get('target_prim_path', '')[-60:]}

    marked = meta.get('success') is True and meta.get('success_position') is not None
    result['checks']['success_marked'] = {'ok': marked,
                                          'detail': 'T press recorded' if marked else 'no T press'}
    if not marked:
        return result

    position = meta['success_position']
    distance = math.dist(position, target)
    result['metrics']['success_distance_m'] = round(distance, 2)
    result['checks']['distance'] = {'ok': distance <= max_distance,
                                    'detail': f'{distance:.2f} m (limit {max_distance} m)'}

    # Target in frame: project the target bbox corners with the camera at the marked instant.
    stamp = meta.get('success_simulation_time_s')
    sample = min(samples, key=lambda s: abs(s['simulation_time'] - (stamp or 0.0)))
    resolution = sample.get('resolution') or meta.get('resolution') or [320, 256]
    focal, ap_h, ap_v = _camera_intrinsics(meta.get('scene_config_path'))
    from pxr import Usd, UsdGeom
    import yaml
    stage_path = None
    if meta.get('scene_config_path'):
        cfg = yaml.safe_load(Path(meta['scene_config_path']).read_text(encoding='utf-8'))
        stage_path = (Path(meta['scene_config_path']).parent / cfg['scene']['stage']).resolve()
    in_frame = 0
    if stage_path:
        stage = Usd.Stage.Open(str(stage_path))
        box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            stage.GetPrimAtPath(task['target_prim_path'])).ComputeAlignedRange()
        if not box.IsEmpty():
            lo, hi = box.GetMin(), box.GetMax()
            corners = [[lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1], lo[2] if i & 4 else hi[2]]
                       for i in range(8)]
            for corner in corners:
                pixel = _project(corner, sample['rgb_camera_world_matrix'], focal,
                                 ap_h, ap_v, resolution[0], resolution[1])
                if pixel and 0 <= pixel[0] < resolution[0] and 0 <= pixel[1] < resolution[1]:
                    in_frame += 1
    result['metrics']['target_corners_in_frame'] = in_frame
    result['checks']['target_in_frame'] = {'ok': in_frame > 0,
                                           'detail': f'{in_frame}/8 corners inside the image'}

    # Hover: a window of at least min_hover seconds within distance and below the speed threshold,
    # ending no later than the mark.
    end = stamp if stamp is not None else samples[-1]['simulation_time']
    window = [s for s in samples if s['simulation_time'] <= end + 1e-6]
    best, start_time = 0.0, None
    for s in window:
        close = math.dist(s['position'], target) <= max_distance
        slow = math.sqrt(sum(v * v for v in s.get('linear_velocity', [0, 0, 0]))) <= hover_speed
        if close and slow:
            if start_time is None:
                start_time = s['simulation_time']
            best = max(best, s['simulation_time'] - start_time)
        else:
            start_time = None
    result['metrics']['hover_duration_s'] = round(best, 2)
    result['checks']['hover'] = {'ok': best >= min_hover,
                                 'detail': f'{best:.1f} s within {max_distance} m at '
                                           f'≤{hover_speed} m/s (need {min_hover} s)'}

    result['passed'] = all(check['ok'] for check in result['checks'].values())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=None)
    parser.add_argument('--max-distance', type=float, default=5.0)
    parser.add_argument('--min-hover', type=float, default=3.0)
    parser.add_argument('--hover-speed', type=float, default=0.3)
    args = parser.parse_args()

    result = evaluate(args.run, args.max_distance, args.min_hover, args.hover_speed)
    report = args.report or (args.run / 'task_evaluation.json')
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"task: {result.get('task_id')}")
    for name, check in result['checks'].items():
        print(f"  [{'PASS' if check['ok'] else 'FAIL'}] {name:18} {check['detail']}")
    if 'passed' in result:
        print(f"result: {'PASS' if result['passed'] else 'FAIL'}")
    print(f"written: {report}")
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
