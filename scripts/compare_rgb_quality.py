"""Create a quantitative same-pose RGB render-quality A/B report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.frame_metadata import load_frame_metadata


def image_statistics(image):
    rgb = np.asarray(image)[..., :3]
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f'Expected an RGB/RGBA image, got {rgb.shape}')
    values = rgb.astype(np.float64)
    unit = values / 255.0
    high, low = unit.max(axis=2), unit.min(axis=2)
    saturation = np.divide(high - low, high, out=np.zeros_like(high), where=high > 0)
    luma = values @ np.asarray([0.2126, 0.7152, 0.0722])
    padded = np.pad(luma, 1, mode='edge')
    laplacian = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                 padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * luma)
    return {
        'size': [int(rgb.shape[1]), int(rgb.shape[0])],
        'mean_rgb': [float(x) for x in values.mean(axis=(0, 1))],
        'std_rgb': [float(x) for x in values.std(axis=(0, 1))],
        'mean_luma': float(luma.mean()),
        'luma_percentiles_1_50_99': [float(x) for x in np.percentile(luma, [1, 50, 99])],
        'mean_hsv_saturation': float(saturation.mean()),
        'clipped_channel_fraction': float(np.mean((rgb == 0) | (rgb == 255))),
        'high_frequency_luma_variance': float(laplacian.var()),
    }


def array_difference(left, right):
    a, b = np.asarray(left), np.asarray(right)
    same_shape = a.shape == b.shape
    same_dtype = a.dtype == b.dtype
    if not same_shape:
        return {'exact_equal': False, 'same_shape': False, 'same_dtype': same_dtype,
                'left_shape': list(a.shape), 'right_shape': list(b.shape),
                'max_absolute_difference': None, 'mean_absolute_difference': None}
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        'exact_equal': bool(same_dtype and np.array_equal(a, b)),
        'same_shape': True,
        'same_dtype': same_dtype,
        'left_shape': list(a.shape),
        'right_shape': list(b.shape),
        'max_absolute_difference': float(delta.max(initial=0)),
        'mean_absolute_difference': float(delta.mean()),
    }


def _load_run(path):
    root = Path(path)
    summary = json.loads((root / 'replay_metadata.json').read_text(encoding='utf-8'))
    frame = load_frame_metadata(root / 'frame_000000.json')
    rgb = np.load(root / 'frame_000000_rgb.npy')
    return root, summary, frame, rgb


def compare_runs(spp1_path, spp8_path, locked_path, output_dir):
    runs = {
        'spp1_default': _load_run(spp1_path),
        'spp8_default': _load_run(spp8_path),
        'spp8_tonemap_locked': _load_run(locked_path),
    }
    reference = runs['spp1_default']
    alignment = {'same_source_stage': True, 'same_scene_config': True,
                 'same_trajectory': True, 'same_resolution': True,
                 'same_simulation_time': True, 'same_camera_pose': True,
                 'same_thermal_camera_pose': True}
    for _, summary, frame, _ in runs.values():
        alignment['same_source_stage'] &= summary['source_stage_sha256'] == reference[1]['source_stage_sha256']
        alignment['same_scene_config'] &= summary['scene_config_sha256'] == reference[1]['scene_config_sha256']
        alignment['same_trajectory'] &= summary['trajectory'] == reference[1]['trajectory']
        alignment['same_resolution'] &= summary['resolution'] == reference[1]['resolution']
        alignment['same_simulation_time'] &= frame['replay_simulation_time'] == reference[2]['replay_simulation_time']
        alignment['same_camera_pose'] &= frame['camera_world_matrix'] == reference[2]['camera_world_matrix']
        alignment['same_thermal_camera_pose'] &= (
            frame['thermal_camera_world_matrix'] == reference[2]['thermal_camera_world_matrix'])
    if not all(alignment.values()):
        raise ValueError(f'Runs do not share the same scene state: {alignment}')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    image_metrics = {}
    for name, (_, summary, _, rgb) in runs.items():
        image_metrics[name] = {
            **image_statistics(rgb),
            'samples_per_pixel': summary['rgb_render_options']['samples_per_pixel_per_frame'],
            'tonemap_lock_requested': summary['rgb_render_options']['tonemap_lock_requested'],
        }

    rgb_differences = {}
    thermal_differences = {}
    baseline_rgb = reference[3]
    baseline_root = reference[0]
    thermal_names = ('display8', 'radiance', 'apparent_t', 'dn16')
    for name, (root, _, _, rgb) in list(runs.items())[1:]:
        rgb_differences[f'spp1_default_vs_{name}'] = array_difference(baseline_rgb, rgb)
        for thermal_name in thermal_names:
            a = np.load(baseline_root / f'frame_000000_{thermal_name}.npy')
            b = np.load(root / f'frame_000000_{thermal_name}.npy')
            thermal_differences.setdefault(thermal_name, {})[name] = array_difference(a, b)

    tile_size = image_metrics['spp1_default']['size']
    tile_w, tile_h = tile_size
    caption_h = 32
    sheet = Image.new('RGB', (tile_w * 3, tile_h + caption_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (name, (_, _, _, rgb)) in enumerate(runs.items()):
        tile = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8), mode='RGB')
        sheet.paste(tile, (tile_w * index, caption_h))
        draw.text((tile_w * index + 8, 8), name, fill=(255, 255, 255))
    sheet.save(output_dir / 'rgb_ab_montage.png')

    report = {
        'result': 'PASS',
        'diagnostic_only': True,
        'data_usage': 'pilot/development data',
        'alignment_checks': alignment,
        'rgb_metrics': image_metrics,
        'rgb_differences': rgb_differences,
        'thermal_data_differences': thermal_differences,
        'interpretation_limits': [
            'Higher spatial variation may mean recovered detail or residual noise; it is not a sharpness verdict by itself.',
            'A same-pose software A/B is not camera calibration, strict RGB/thermal synchronization proof, or benchmark qualification.',
            'Tonemap-lock comparison only tests explicitly reapplying the already recorded values.',
        ],
        'montage': str((output_dir / 'rgb_ab_montage.png').resolve()),
    }
    (output_dir / 'rgb_quality_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spp1', type=Path, required=True)
    parser.add_argument('--spp8', type=Path, required=True)
    parser.add_argument('--spp8-tonemap-locked', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_runs(args.spp1, args.spp8, args.spp8_tonemap_locked, args.out)
    print(f"RGB_QUALITY_COMPARISON_{report['result']} {args.out}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
