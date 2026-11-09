"""Compare display profiles on owned saved data, without starting Isaac or GPU.

Snapshot comparison uses the exact saved unquantised signal and flat-field
coefficients. Redisplay of normal replay frames uses apparent_t only; those
frames do not contain the signal needed for an exact vendor ISP replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/isaac-thermal-camera-simulation/src'))
sys.path.insert(0, 'D:/isaacsim/extscache/omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312')

from scripts.thermal_display import fixed_temperature_display, profile_variants
from scripts.thermal_imaging import display_from_outputs
from scripts.thermal_snapshot import load_snapshot, state_digest
from scripts.frame_metadata import FrameMetadataReader, FrameMetadataWriter

RAW_NAMES = ('radiance', 'apparent_t', 'dn16', 'signal_dn', 'flux')


def array_record(value):
    if value is None:
        return None
    array = np.asarray(value)
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def temperature_statistics(apparent_t, min_c, max_c, mask=None):
    values = np.asarray(apparent_t, dtype=np.float64) - 273.15
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    finite = values[np.isfinite(values)]
    count = int(finite.size)
    return {
        'finite_pixels': count, 'nonfinite_pixels': int(values.size - count),
        'min_c': float(finite.min()) if count else None,
        'max_c': float(finite.max()) if count else None,
        'std_c': float(finite.std()) if count else None,
        'below_range_fraction': float(np.mean(finite < min_c)) if count else None,
        'above_range_fraction': float(np.mean(finite > max_c)) if count else None,
    }


def gray_statistics(display, apparent_t, roi=None):
    gray = np.asarray(display)[..., :3].astype(np.float64).mean(axis=-1)
    temperature = np.asarray(apparent_t, dtype=np.float64)
    if roi is None:
        height, width = gray.shape
        x0 = int(width * 270 / 640)
        y0 = int(height * 370 / 512)
        x1 = max(x0 + 1, int(width * 370 / 640))
        y1 = max(y0 + 1, int(height * 430 / 512))
        roi = (slice(y0, y1), slice(x0, x1))
    patch, thermal = gray[roi], temperature[roi]
    finite = np.isfinite(thermal)
    std_c = float(thermal[finite].std()) if finite.any() else None
    std_gray = float(patch[finite].std()) if finite.any() else None
    return {
        'gray_mean': float(patch[finite].mean()) if finite.any() else None,
        'gray_std': std_gray, 'apparent_t_std_c': std_c,
        'gray_levels_per_temperature_std': (std_gray / std_c if std_c else None),
        'black_fraction': float(np.mean(gray == 0)),
        'white_fraction': float(np.mean(gray == 255)),
    }


def compare_display_frame(config, outputs, profiles, *, state=None, sky_mask=None):
    before = {name: array_record(getattr(outputs, name)) for name in RAW_NAMES}
    history = state_digest((config, state))
    variants, frames = {}, {}
    raw_exact = True
    for profile in profiles:
        started = time.perf_counter()
        result = display_from_outputs(config, outputs, profile)
        elapsed = time.perf_counter() - started
        current = {name: array_record(getattr(result, name)) for name in RAW_NAMES}
        exact = current == before
        raw_exact = raw_exact and exact
        frames[profile.name] = result.display8
        values = {'profile': profile.metadata(), 'raw_arrays_byte_exact': exact,
                  'display_s': elapsed, 'roi': gray_statistics(result.display8, outputs.apparent_t),
                  'display_hash': array_record(result.display8)}
        if profile.mode == 'fixed_apparent_temperature':
            values['temperature_range'] = temperature_statistics(
                outputs.apparent_t, profile.temperature_min_c, profile.temperature_max_c)
            if sky_mask is not None and sky_mask.shape == outputs.apparent_t.shape:
                values['foreground_temperature_range'] = temperature_statistics(
                    outputs.apparent_t, profile.temperature_min_c, profile.temperature_max_c,
                    mask=~sky_mask)
        variants[profile.name] = values
    untouched = history == state_digest((config, state))
    original_untouched = before == {name: array_record(getattr(outputs, name)) for name in RAW_NAMES}
    return frames, {'raw_arrays_byte_exact': bool(raw_exact and original_untouched),
                    'sensor_config_and_state_unchanged': untouched, 'raw_input_hashes': before,
                    'variants': variants}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def montage(path, images):
    from PIL import Image, ImageDraw
    width, height = np.asarray(images[0][1]).shape[:2][::-1]
    canvas = Image.new('RGB', (width * len(images), height + 26), 'black')
    draw = ImageDraw.Draw(canvas)
    for index, (name, array) in enumerate(images):
        canvas.paste(Image.fromarray(np.asarray(array)).convert('RGB'), (index * width, 26))
        draw.text((index * width + 5, 5), name, fill='white')
    canvas.save(path)


def compare_snapshots(directory, output, min_c, max_c):
    from PIL import Image
    profiles = profile_variants(min_c, max_c)
    paths = sorted(directory.glob('frame_*.pkl'))
    if not paths:
        raise ValueError('No local thermal snapshots found')
    records = []
    for path in paths:
        snapshot = load_snapshot(path)
        config, raw = snapshot['config'], snapshot['raw_outputs']
        state = (snapshot['state_before'], snapshot.get('history_after'))
        images, report = compare_display_frame(
            config, raw, profiles, state=state, sky_mask=snapshot['planes'].get('sky_mask'))
        legacy = display_from_outputs(config, raw)
        report['current_matches_existing_project_display'] = (
            array_record(legacy.display8) == array_record(images['current']))
        report['current_matches_saved_display'] = (
            array_record(snapshot['outputs'].display8) == array_record(images['current']))
        report['snapshot'] = str(path)
        report['case'] = snapshot['metadata']['case']
        report['sensor_chain_attached'] = config.chain is not None
        for name, pixels in images.items():
            target = output / name
            target.mkdir(exist_ok=True)
            Image.fromarray(pixels).save(target / f'{path.stem}_thermal.png')
        ordered = [('RGB', snapshot['rgb'])] + list(images.items())
        montage(output / f'{path.stem}_comparison.png', ordered)
        write_json(output / f'{path.stem}.json', report)
        records.append(report)
    return records


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redisplay_replay(source, output, min_c, max_c):
    """Copy a saved replay into a new directory, changing only display files."""
    from PIL import Image
    profile = profile_variants(min_c, max_c)[-1].metadata()
    records = []
    paths = sorted(source.glob('frame_*_apparent_t.npy'))
    if not paths:
        raise ValueError('No saved apparent-temperature replay frames found')
    metadata_reader = FrameMetadataReader()
    metadata_writer = None
    for path in paths:
        stem = path.name.removesuffix('_apparent_t.npy')
        apparent_t = np.load(path, allow_pickle=False)
        original = np.load(source / f'{stem}_display8.npy', allow_pickle=False)
        rgb = np.load(source / f'{stem}_rgb.npy', allow_pickle=False)
        display = fixed_temperature_display(apparent_t, min_c, max_c)
        raw_hashes = {}
        for kind in ('dn16', 'radiance', 'apparent_t', 'rgb'):
            filename = f'{stem}_{kind}.npy'
            shutil.copyfile(source / filename, output / filename)
            raw_hashes[kind] = {'source': file_hash(source / filename),
                                'output': file_hash(output / filename)}
        shutil.copyfile(source / f'{stem}_rgb.png', output / f'{stem}_rgb.png')
        np.save(output / f'{stem}_display8.npy', display)
        Image.fromarray(display).save(output / f'{stem}_thermal.png')
        if stem in ('frame_000000', 'frame_000021', 'frame_000042'):
            montage(output / f'{stem}_comparison.png',
                    [('RGB', rgb), ('original', original), (f'fixed {min_c:g}..{max_c:g} C', display)])
        meta = metadata_reader.read(source / f'{stem}.json')
        meta['source_display_provenance'] = {
            key: meta[key] for key in ('thermal_imaging', 'display_profile', 'experiment_conditions')
            if key in meta
        }
        meta['display_profile'] = {
            **profile,
            'input_quantity': 'saved apparent_t in Kelvin',
            'meaning': 'Fixed apparent-temperature visualization, not factory sensor ISP',
        }
        imaging = dict(meta.get('thermal_imaging', {}))
        if 'vendor_isp_parameters' not in imaging and 'display_parameters' in imaging:
            imaging['vendor_isp_parameters'] = imaging['display_parameters']
        imaging.update({
            'display_profile': meta['display_profile'],
            'display_parameters': profile,
            'display_parameters_sha256': hashlib.sha256(
                json.dumps(profile, sort_keys=True).encode()).hexdigest(),
            'display_input_quantity': 'saved apparent_t (K)',
            'project_display_flat_field_applied': False,
            'display_correction_source': 'none_fixed_apparent_temperature',
            'display_meaning': 'Fixed apparent-temperature visualization; separate from vendor sensor ISP',
        })
        meta['thermal_imaging'] = imaging
        meta['experiment_conditions_scope'] = 'Source capture conditions; new display is described by display_profile'
        meta['derived_display_source'] = str(source.resolve())
        # Redisplay is not new rendering or a new synchronization verification.
        meta['redisplay_only'] = True
        if metadata_writer is None:
            metadata_writer = FrameMetadataWriter(output, meta)
        metadata_writer.write(int(stem.removeprefix('frame_')), meta)
        record = {
            'frame_id': meta['frame_id'], 'raw_file_hashes': raw_hashes,
            'raw_files_byte_exact': all(pair['source'] == pair['output'] for pair in raw_hashes.values()),
            'original_roi': gray_statistics(original, apparent_t),
            'fixed_roi': gray_statistics(display, apparent_t),
            'temperature_range': temperature_statistics(apparent_t, min_c, max_c),
            'simulation_time': meta['simulation_time_s'],
        }
        records.append(record)
    replay = json.loads((source / 'replay_metadata.json').read_text(encoding='utf-8'))
    replay['source_display_provenance'] = {
        key: replay[key] for key in ('display_profile', 'experiment_conditions') if key in replay
    }
    replay['experiment_conditions_scope'] = 'Source capture conditions; new display is described by display_profile'
    replay['redisplay_only'] = True
    replay['derived_display_source'] = str(source.resolve())
    replay['display_profile'] = profile
    write_json(output / 'replay_metadata.json', replay)
    return records


def saved_target_view(source, output, min_c, max_c):
    """Show the previously aimed Spot view; no new target-detection claim."""
    stem = 'frame_000001'
    rgb = np.load(source / f'{stem}_rgb.npy', allow_pickle=False)
    original = np.load(source / f'{stem}_display8.npy', allow_pickle=False)
    apparent = np.load(source / f'{stem}_apparent_t.npy', allow_pickle=False)
    fixed = fixed_temperature_display(apparent, min_c, max_c)
    montage(output / 'spot_comparison.png', [('RGB Spot view', rgb), ('original', original),
                                             (f'fixed {min_c:g}..{max_c:g} C', fixed)])
    # The saved target inspection independently established that this camera
    # was aimed at Spot. A warm mask is only a display diagnostic, not a new
    # semantic target detector or ground-truth silhouette.
    temperature_c = np.asarray(apparent, dtype=np.float64) - 273.15
    warm = temperature_c > 30.
    background = (temperature_c > 17.) & (temperature_c < 25.)
    gray = fixed[..., 0].astype(np.float64)
    contrast = (float(gray[warm].mean() - gray[background].mean())
                if warm.any() and background.any() else None)
    return {'source': str(source), 'scope': 'Previously aimed Spot view; warm mask is not semantic ground truth',
            'warm_mask_pixels': int(warm.sum()), 'warm_background_gray_difference': contrast,
            'temperature_range': temperature_statistics(apparent, min_c, max_c)}


def uniform_display_diagnostic(config, *, frames=30, seed=170, min_c=-30., max_c=45.):
    """Synthetic same-input display check; not a physical NETD calibration."""
    from irsim.pipeline.frame import Outputs
    generator = np.random.default_rng(seed)
    profiles = profile_variants(min_c, max_c)
    samples = {profile.name: [] for profile in profiles}
    for index in range(frames):
        apparent = (293.15 + generator.normal(0., .05, (128, 160))).astype(np.float32)
        signal = (20000. + (apparent - np.float32(293.15)) * np.float32(820.)).astype(np.float32)
        synthetic = Outputs(radiance=np.ones_like(apparent), apparent_t=apparent,
                            dn16=np.rint(signal).astype(np.uint16),
                            display8=np.zeros((*apparent.shape, 4), dtype=np.uint8),
                            signal_dn=signal, flux=np.ones_like(apparent))
        # The scene here is a synthetic uniform input; do not apply school
        # optics flat-field coefficients to this already-uniform fixture.
        from types import SimpleNamespace
        diagnostic_config = SimpleNamespace(flat_field=None, chain=None, sensor=config.sensor)
        images, _ = compare_display_frame(diagnostic_config, synthetic, profiles)
        for name, pixels in images.items():
            gray = pixels[..., 0].astype(float)
            samples[name].append({'mean': float(gray.mean()), 'std': float(gray.std())})
    return {'scope': 'Synthetic uniform arrays, not a rendered blackbody or physical noise calibration',
            'fixed_range_c': [min_c, max_c],
            'seed': seed, 'frames': frames, 'prescribed_noise_std_c': .05,
            'variants': {name: {'mean_spatial_gray_std': float(np.mean([s['std'] for s in values])),
                                'frame_mean_gray_std': float(np.std([s['mean'] for s in values]))}
                         for name, values in samples.items()}}


def protected_file_records():
    names = [ROOT / 'lesson_crazyflie.usd', ROOT / 'school_vln_01.usd',
             ROOT / 'scripts/drone_keyboard_controller.py', ROOT / 'scripts/drone_control_math.py',
             ROOT / 'configs/school_vln_scene.yaml']
    names.extend(sorted((ROOT / 'third_party/isaac-thermal-camera-simulation/src').rglob('*.py')))
    names.extend(sorted((ROOT / 'third_party/isaac-thermal-camera-simulation/configs').rglob('*.yaml')))
    return {str(path): file_hash(path) for path in names}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshots', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--spot-view', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--min-c', type=float, default=-30.)
    parser.add_argument('--max-c', type=float, default=45.)
    parser.add_argument('--trusted-local-snapshots', action='store_true', required=True)
    args = parser.parse_args(argv)
    profile_variants(args.min_c, args.max_c)
    args.out.mkdir(parents=True, exist_ok=False)
    before = protected_file_records()
    report = {'result': 'INCOMPLETE', 'data_usage': 'pilot/development data',
              'strict_sync_verified': False, 'gui_visual_test': False,
              'scope': 'Same-input display comparison and saved-data redisplay, no new render'}
    try:
        snapshot_out = args.out / 'snapshot_comparison'
        replay_out = args.out / 'replay_fixed'
        snapshot_out.mkdir()
        replay_out.mkdir()
        report['snapshot_frames'] = compare_snapshots(args.snapshots, snapshot_out, args.min_c, args.max_c)
        report['replay_frames'] = redisplay_replay(args.replay, replay_out, args.min_c, args.max_c)
        if args.spot_view:
            report['spot_view'] = saved_target_view(args.spot_view, args.out, args.min_c, args.max_c)
        report['uniform_display_diagnostic'] = uniform_display_diagnostic(
            load_snapshot(sorted(args.snapshots.glob('frame_*.pkl'))[0])['config'],
            min_c=args.min_c, max_c=args.max_c)
        exact = all(item['raw_arrays_byte_exact'] and item['sensor_config_and_state_unchanged'] and
                    item['current_matches_existing_project_display'] and item['current_matches_saved_display']
                    for item in report['snapshot_frames'])
        unchanged = all(item['raw_files_byte_exact'] for item in report['replay_frames'])
        report['original_files_unchanged'] = before == protected_file_records()
        report['fixed_range_c'] = [args.min_c, args.max_c]
        report['sample_count'] = len(report['snapshot_frames'])
        report['replay_frame_count'] = len(report['replay_frames'])
        report['display_acceptance'] = {
            'saved_trajectory_no_clipping': all(
                item['temperature_range']['nonfinite_pixels'] == 0 and
                item['temperature_range']['below_range_fraction'] == 0. and
                item['temperature_range']['above_range_fraction'] == 0.
                for item in report['replay_frames']),
            'ground_roi_gray_std_reduced_all_frames': all(
                item['fixed_roi']['gray_std'] < item['original_roi']['gray_std']
                for item in report['replay_frames']),
            'semantic_target_detection_verified': False,
        }
        report['result'] = 'PASS_DISPLAY_AND_RAW_DATA_CHECKS' if exact and unchanged and report['original_files_unchanged'] else 'FAIL'
        report['full_shutter_drift_history_covered'] = False
        print(f"DISPLAY_COMPARISON_{report['result']} snapshots={report['sample_count']} replay={report['replay_frame_count']}", flush=True)
    except Exception:
        import traceback
        report['result'] = 'FAIL'
        report['failure'] = traceback.format_exc()
        (args.out / 'failure.txt').write_text(report['failure'], encoding='utf-8')
        raise
    finally:
        write_json(args.out / 'summary.json', report)
    return 0 if report['result'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
