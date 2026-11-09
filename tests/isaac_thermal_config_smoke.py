"""Render before/after a YAML edit, using two fresh previews; never save USD."""
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party/isaac-thermal-camera-simulation/src'))


def main():
    import numpy as np
    import yaml
    from scripts.start_crazyflie_dual_camera_view import open_preview
    from scripts.thermal_config import DEFAULT_THERMAL_CONFIG
    from isaacsim import SimulationApp

    stage_path = ROOT / 'lesson_crazyflie.usd'
    source_hash = hashlib.sha256(stage_path.read_bytes()).hexdigest()
    config_hash = hashlib.sha256(DEFAULT_THERMAL_CONFIG.read_bytes()).hexdigest()
    out = ROOT / 'validation' / time.strftime('thermal-yaml-smoke-%Y%m%d-%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    app = SimulationApp({'headless': True, 'renderer': 'PathTracing',
                         'samples_per_pixel_per_frame': 1, 'multi_gpu': False})
    preview = controller = None
    status = 1
    samples = []
    try:
        import omni.timeline
        config = yaml.safe_load(DEFAULT_THERMAL_CONFIG.read_text(encoding='utf-8'))
        config_path = out / 'test_thermal.yaml'
        for index, ground_c in enumerate((20.0, 40.0)):
            # Only edit the test copy. The default user config stays byte-identical.
            config['targets']['ground']['temperature_c'] = ground_c
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            preview, controller = open_preview(app, width=160, height=128,
                                               thermal_config_path=config_path)
            report = preview.metadata['thermal_configuration']
            assert not report['unmatched_bindings'], report
            assert report['binding_counts']['ground'] > 0, report
            assert report['binding_counts']['cube'] > 0, report
            assert report['sha256'] == hashlib.sha256(config_path.read_bytes()).hexdigest()
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            for _ in range(180):
                app.update()
                if controller._ready:
                    break
            assert controller._ready, controller.diagnostics()
            for _ in range(3):
                app.update()
            rgb, thermal, meta = preview.capture(render=True)
            surface = np.asarray(preview.thermal.last_frame.planes['temperature_k'])
            mask = np.isclose(surface, ground_c + 273.15, atol=0.005, rtol=0)
            assert int(mask.sum()) > 10, 'ground temperature did not reach visible pixels'
            assert np.isfinite(thermal.radiance).all()
            assert np.isfinite(thermal.apparent_t).all()
            assert np.ptp(rgb[..., :3]) > 0
            assert np.ptp(thermal.display8) > 0
            assert meta['pair_pose_max_error'] < 1e-6
            assert abs(meta['thermal_target_temperatures_k']['ground'] - (ground_c+273.15)) < 0.005
            preview.save_frame(out, index, rgb, thermal, meta)
            np.save(out / f'frame_{index:06d}_input_temperature_k.npy', surface)
            samples.append({'ground_c': ground_c, 'matched_pixels': int(mask.sum()),
                            'ground_radiance_mean': float(thermal.radiance[mask].mean()),
                            'config_sha256': report['sha256']})
            controller.stop()
            controller = None
            preview.close()
            preview = None
            app.update()
        assert samples[1]['config_sha256'] != samples[0]['config_sha256']
        assert samples[1]['ground_radiance_mean'] > samples[0]['ground_radiance_mean'] + 0.1, samples
        assert hashlib.sha256(stage_path.read_bytes()).hexdigest() == source_hash
        assert hashlib.sha256(DEFAULT_THERMAL_CONFIG.read_bytes()).hexdigest() == config_hash
        result = {'result': 'PASS', 'samples': samples, 'source_unchanged': True,
                  'user_config_unchanged': True, 'resolution': [160, 128],
                  'manual_gui_test': False, 'radiance_responds_to_yaml': True}
        (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print('THERMAL_YAML_SMOKE_PASS ' + str(out), flush=True)
        status = 0
    except Exception:
        (out / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        traceback.print_exc()
    finally:
        if controller:
            controller.stop()
        if preview:
            preview.close()
        app.close(exit_code=status)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
