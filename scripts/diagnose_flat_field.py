"""Compare the thermal display chain on a uniform-temperature detector plane.

This is a pipeline diagnostic, not a scene capture. It does not open or save USD.
The physical outputs are written unchanged alongside the display output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
THERMAL_ROOT = ROOT / 'third_party' / 'isaac-thermal-camera-simulation'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THERMAL_ROOT / 'src'))


def _stats(array):
    values = np.asarray(array, dtype=np.float64)
    h, w = values.shape[:2]
    gray = values[..., 0] if values.ndim == 3 else values
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
    result = {}
    for name, box in boxes.items():
        result[name] = {
            'mean': float(np.mean(box)), 'std': float(np.std(box)),
            'min': float(np.min(box)), 'max': float(np.max(box)),
            'saturated_fraction': float(np.mean(gray == np.iinfo(gray.dtype).max))
            if np.issubdtype(gray.dtype, np.integer) else None,
        }
    result['global'] = {
        'mean': float(np.mean(gray)), 'std': float(np.std(gray)),
        'min': float(np.min(gray)), 'max': float(np.max(gray)),
    }
    return result


def _capture(sensor, materials, lut, flat, width, height, temperature_k):
    from irsim.pipeline.core import PipelineConfig, PipelineState
    from irsim.pipeline.frame import run_frame
    config = PipelineConfig.from_sensor(sensor, materials, lut=lut, noise_enabled=True,
                                        psf_enabled=False, flat_field_enabled=flat,
                                        sensor_seed=0)
    k = sensor.sensor.optics.supersample_factor
    render_height, render_width = height * k, width * k
    planes = {
        'temperature_k': np.full((render_height, render_width), temperature_k, dtype=np.float32),
        'material_id': np.ones((render_height, render_width), dtype=np.int32),
        'distance_m': np.full((render_height, render_width), 20.0, dtype=np.float32),
        'sky_mask': np.zeros((render_height, render_width), dtype=bool),
        'normal_dot_view': np.ones((render_height, render_width), dtype=np.float32),
    }
    outputs = run_frame(planes, config, PipelineState())
    if flat:
        from irsim.detector.quantise import quantise
        from irsim.isp.display import run_display_branch
        corrected = quantise(config.flat_field.apply(outputs.signal_dn), sensor.sensor.fpa.bit_depth)
        display = run_display_branch(corrected, sensor.sensor.isp, sensor.sensor.fpa.bit_depth)
        from dataclasses import replace
        outputs = replace(outputs, display8=display.display8)
    return config, outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sensor', type=Path, default=THERMAL_ROOT/'configs/sensors/flir_boson_640_lwir.yaml')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--temperature-c', type=float, default=20.0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    from irsim.config.loader import load_sensor_config
    from irsim.radiometry.lut_files import load_band_lut_for_config
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.table import MaterialTable
    sensor = load_sensor_config(args.sensor)
    materials = MaterialTable.from_library(MaterialLibrary.load(), sensor.sensor.band.band_id)
    lut = load_band_lut_for_config(sensor, THERMAL_ROOT/'data/lut')
    width, height = sensor.sensor.fpa.width, sensor.sensor.fpa.height
    result = {
        'temperature_c': args.temperature_c, 'temperature_k': args.temperature_c + 273.15,
        'resolution': [width, height], 'sensor_seed': 0,
        'sensor_config_path': str(args.sensor.resolve()),
        'sensor_config_sha256': hashlib.sha256(args.sensor.read_bytes()).hexdigest(),
        'vignetting_cos4': sensor.sensor.optics.vignetting_cos4,
        'comparison': {},
    }
    for name, flat in (('flat_field_off', False), ('flat_field_on', True)):
        config, outputs = _capture(sensor, materials, lut, flat, width, height, args.temperature_c + 273.15)
        display = np.asarray(outputs.display8)
        np.save(args.out/f'{name}_dn16.npy', outputs.dn16)
        np.save(args.out/f'{name}_radiance.npy', outputs.radiance)
        np.save(args.out/f'{name}_apparent_t.npy', outputs.apparent_t)
        np.save(args.out/f'{name}_display8.npy', display)
        from PIL import Image
        Image.fromarray(display).save(args.out/f'{name}_display8.png')
        result['comparison'][name] = {
            'flat_field_enabled': flat,
            'stats_display8': _stats(display),
            'stats_dn16': _stats(outputs.dn16),
            'stats_radiance': _stats(outputs.radiance),
            'stats_apparent_t': _stats(outputs.apparent_t),
            'display_input_is_physical_dn': True,
            'physical_arrays_finite': bool(np.isfinite(outputs.radiance).all() and
                                           np.isfinite(outputs.apparent_t).all()),
            'flat_field_coefficients_present': config.flat_field is not None,
        }
    off = result['comparison']['flat_field_off']['stats_display8']['global']
    on = result['comparison']['flat_field_on']['stats_display8']['global']
    result['display_uniformity_improvement'] = {
        'off_std': off['std'], 'on_std': on['std'],
        'std_ratio_on_over_off': None if off['std'] == 0 else on['std'] / off['std'],
        'off_corner_center_span': max(result['comparison']['flat_field_off']['stats_display8'][k]['mean']
                                      for k in ('top_left','top_right','bottom_left','bottom_right')) -
                                  result['comparison']['flat_field_off']['stats_display8']['center']['mean'],
        'on_corner_center_span': max(result['comparison']['flat_field_on']['stats_display8'][k]['mean']
                                     for k in ('top_left','top_right','bottom_left','bottom_right')) -
                                 result['comparison']['flat_field_on']['stats_display8']['center']['mean'],
    }
    (args.out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
