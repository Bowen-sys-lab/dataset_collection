"""Compare paired frozen captures with flat-field display correction off and on."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.frame_metadata import FrameMetadataReader


def _regions(array):
    gray = np.asarray(array)
    if gray.ndim == 3:
        gray = gray[..., 0]
    h, w = gray.shape
    boxes = {
        'center': gray[h//4:3*h//4, w//4:3*w//4],
        'top': gray[:max(1, h//8), :],
        'bottom': gray[-max(1, h//8):, :],
        'left': gray[:, :max(1, w//8)],
        'right': gray[:, -max(1, w//8):],
        'top_left': gray[:max(1, h//4), :max(1, w//4)],
        'top_right': gray[:max(1, h//4), -max(1, w//4):],
        'bottom_left': gray[-max(1, h//4):, :max(1, w//4)],
        'bottom_right': gray[-max(1, h//4):, -max(1, w//4):],
    }
    return {name: {
        'mean': float(np.mean(box)), 'std': float(np.std(box)),
        'saturated_fraction': float(np.mean(box == 255)),
    } for name, box in boxes.items()}


def compare(off_dir: Path, on_dir: Path, out_dir: Path):
    off_dir, on_dir, out_dir = off_dir.resolve(), on_dir.resolve(), out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    off_frames = sorted(off_dir.glob('frame_*_thermal.png'))
    on_frames = sorted(on_dir.glob('frame_*_thermal.png'))
    if not off_frames or [p.name for p in off_frames] != [p.name for p in on_frames]:
        raise ValueError('off/on folders must contain matching thermal PNG frame names')
    results = []
    off_reader, on_reader = FrameMetadataReader(), FrameMetadataReader()
    for off_png, on_png in zip(off_frames, on_frames):
        stem = off_png.name.removesuffix('_thermal.png')
        rgb_off = np.load(off_dir/f'{stem}_rgb.npy')
        rgb_on = np.load(on_dir/f'{stem}_rgb.npy')
        display_off = np.load(off_dir/f'{stem}_display8.npy')
        display_on = np.load(on_dir/f'{stem}_display8.npy')
        arrays = {}
        for suffix in ('dn16', 'radiance', 'apparent_t'):
            left = np.load(off_dir/f'{stem}_{suffix}.npy')
            right = np.load(on_dir/f'{stem}_{suffix}.npy')
            arrays[suffix] = {
                'same_shape': left.shape == right.shape,
                'max_abs_difference': float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64)))),
                'exactly_equal': bool(np.array_equal(left, right)),
                'all_finite': bool(np.isfinite(left).all() and np.isfinite(right).all()),
            }
        if rgb_off.shape != rgb_on.shape:
            raise ValueError(f'{stem}: RGB shapes differ')
        rgb_delta = np.abs(rgb_off.astype(np.int16) - rgb_on.astype(np.int16))
        rgb_delta_report = {
            'same_shape': rgb_off.shape == rgb_on.shape,
            'exactly_equal_across_processes': bool(np.array_equal(rgb_off, rgb_on)),
            'changed_pixel_fraction': float(np.mean(np.any(rgb_delta != 0, axis=-1))),
            'mean_absolute_channel_difference': float(np.mean(rgb_delta)),
            'max_absolute_channel_difference': int(np.max(rgb_delta)),
            'note': 'Separate RTX processes may have small visible-render variation; flat-field acts only on thermal display8.',
        }
        off_meta = off_reader.read(off_dir/f'{stem}.json')
        on_meta = on_reader.read(on_dir/f'{stem}.json')
        pose_off = np.asarray(off_meta['camera_world_matrix'], dtype=float)
        pose_on = np.asarray(on_meta['camera_world_matrix'], dtype=float)
        pose_error = float(np.max(np.abs(pose_off-pose_on)))
        rgb_image = Image.fromarray(rgb_off).convert('RGB')
        off_image = Image.fromarray(display_off).convert('RGB')
        on_image = Image.fromarray(display_on).convert('RGB')
        tile = Image.new('RGB', (rgb_image.width*3, rgb_image.height+28), 'white')
        tile.paste(rgb_image, (0, 28)); tile.paste(off_image, (rgb_image.width, 28))
        tile.paste(on_image, (rgb_image.width*2, 28))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 7), 'RGB', fill='black')
        draw.text((rgb_image.width+8, 7), 'Thermal: flat-field OFF', fill='black')
        draw.text((rgb_image.width*2+8, 7), 'Thermal: flat-field ON', fill='black')
        tile.save(out_dir/f'{stem}_comparison.png')
        results.append({
            'frame': stem, 'rgb_comparison': rgb_delta_report,
            'camera_pose_max_abs_difference': pose_error,
            'flat_field_off_regions': _regions(display_off),
            'flat_field_on_regions': _regions(display_on),
            'physical_arrays': arrays,
            'strict_sync_verified': False,
        })
    report = {
        'result': 'PASS_DISPLAY_BRANCH_COMPARISON' if all(
            row['rgb_comparison']['same_shape'] and row['camera_pose_max_abs_difference'] <= 1e-8 and
            all(item['exactly_equal'] and item['all_finite'] for item in row['physical_arrays'].values())
            for row in results) else 'FAIL',
        'flat_field_scope': 'display8 only; RGB, dn16, radiance, apparent_t must remain identical',
        'frames': results,
    }
    (out_dir/'comparison_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--off', type=Path, required=True)
    parser.add_argument('--on', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.off, args.on, args.out)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['result'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
