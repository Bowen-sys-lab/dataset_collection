import importlib.util
from contextlib import redirect_stderr
import io
import hashlib
import json
import unittest
from types import SimpleNamespace


def fixture():
    reports = []
    for condition, mean in zip(('material_cache_off', 'material_cache_on',
                                'material_cache_on', 'material_cache_off',
                                'material_cache_off', 'material_cache_on'),
                               (.1, .09, .091, .101, .099, .089)):
        samples = [dict(loop_s=mean, app_update_s=.01, capture_total_s=.07,
                        thermal_get_outputs_s=.06, rgb_read_s=.01, ui_upload_s=.01,
                        main_thread_cpu_s=.08) for _ in range(600)]
        poses = [{'position': [0., 0., 2.], 'yaw': 0.,
                  'camera_world_matrix': [[1., 0., 0., 0.], [0., 1., 0., 0.],
                                          [0., 0., 1., 0.], [0., 0., 2., 1.]]}
                 for _ in range(600)]
        pose_digest = hashlib.sha256(json.dumps(poses, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        reports.append(dict(result='PASS', source_unchanged=True, vendor_source_unchanged=True,
            file_hashes_before={'scene': 'abc', 'vendor': 'def', 'controller': 'ghi'},
            file_hashes_after={'scene': 'abc', 'vendor': 'def', 'controller': 'ghi'},
            experiment_conditions={'flat_field': True},
            baseline=dict(resolution=[160, 128], renderer='PathTracing', enabled_viewport_count=1,
                preview_rate_mode='uncapped', preconversion=False, shared_live_render_product=False,
                warmup_frames=30, motion_sequence_sha256='motion', motion_baseline={'position': [0, 0, 2], 'yaw': 0},
                rounds=[dict(condition=condition, profiled=False, samples=samples, segments={},
                    actual_capture_poses=poses, actual_pose_sequence_sha256=pose_digest,
                    cache_stats_after={'approximate_bytes': 123})])))
    return reports


class MaterialCacheExperimentTests(unittest.TestCase):
    def test_missing_incomplete_or_corrupt_actual_pose_evidence_cannot_accept(self):
        for field in ('actual_capture_poses', 'actual_pose_sequence_sha256'):
            with self.subTest(field=field):
                reports = fixture()
                del reports[1]['baseline']['rounds'][0][field]
                result = self.module().summarize(reports, {'result': 'PASS'})
                self.assertEqual(result['result'], 'FAIL')
                self.assertFalse(result['retention'])
        for change in ('missing_frame', 'corrupt_digest'):
            reports = fixture()
            row = reports[1]['baseline']['rounds'][0]
            if change == 'missing_frame':
                row['actual_capture_poses'].pop()
            else:
                row['actual_pose_sequence_sha256'] = '0'*64
            with self.subTest(change=change):
                result = self.module().summarize(reports, {'result': 'PASS'})
                self.assertEqual(result['result'], 'FAIL')
                self.assertFalse(result['retention'])

    def test_changed_actual_pose_with_correct_digest_cannot_accept(self):
        for field in ('position', 'yaw', 'camera_world_matrix'):
            reports = fixture()
            row = reports[1]['baseline']['rounds'][0]
            pose = row['actual_capture_poses'][19]
            if field == 'position':
                pose['position'][0] = .00000001
            elif field == 'yaw':
                pose['yaw'] = .00000001
            else:
                pose['camera_world_matrix'][3][0] = .00000001
            row['actual_pose_sequence_sha256'] = hashlib.sha256(json.dumps(row['actual_capture_poses'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(field=field):
                result = self.module().summarize(reports, {'result': 'PASS'})
                self.assertEqual(result['result'], 'FAIL')
                self.assertFalse(result['retention'])

    def test_bad_shape_or_nonfinite_actual_pose_cannot_accept_even_if_matching(self):
        bad_values = (('position', [1., 2.]), ('yaw', [0.]),
                      ('camera_world_matrix', [[1., 0., 0., 0.]]*3),
                      ('camera_world_matrix', [[1., 0., 0.]]*4),
                      ('position', [0., 0., float('nan')]), ('yaw', float('inf')),
                      ('position', [0., 0., '2']), ('yaw', True))
        for field, value in bad_values:
            reports = fixture()
            for report in reports:
                row = report['baseline']['rounds'][0]
                row['actual_capture_poses'][19][field] = value
                row['actual_pose_sequence_sha256'] = hashlib.sha256(json.dumps(row['actual_capture_poses'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            with self.subTest(field=field, value=value):
                result = self.module().summarize(reports, {'result': 'PASS'})
                self.assertEqual(result['result'], 'FAIL')
                self.assertFalse(result['retention'])

    def test_preflight_rejects_extra_rounds_or_profile_in_throughput(self):
        import isaac_thermal_baseline as baseline
        parse = getattr(baseline, 'parse_arguments', None)
        self.assertTrue(callable(parse), 'preflight parser must exist')
        common = ['--condition', 'material_cache_on', '--phase', 'baseline', '--out', 'unused',
                  '--rounds', '1', '--warmup', '30', '--frames', '600']
        args = parse(common)
        self.assertFalse(args.profile_only)
        self.assertEqual(baseline.conditions_for_run(args.condition, profile_only=True), ['material_cache_on_profiled'])
        for suffix in (['--rounds', '2'], ['--warmup', '29'], ['--frames', '599'], ['--phase', 'both']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse(common+suffix)
        self.assertTrue(parse(common+['--profile-only']).profile_only)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.summarize_material_cache'),
                             'material cache summarizer must exist')
        from scripts import summarize_material_cache
        return summarize_material_cache

    def test_accepts_three_reproducible_improvements(self):
        report = self.module().summarize(fixture(), {'result': 'PASS'})
        self.assertTrue(report['retention'])
        self.assertAlmostEqual(report['aggregate']['mean_loop_reduction'], .1)
        self.assertFalse(report['valid_new_pair_fps_verified'])

    def test_rejects_one_harmed_pair_and_consistently_worse_tails(self):
        reports = fixture()
        for sample in reports[1]['baseline']['rounds'][0]['samples']:
            sample['loop_s'] = .11
        self.assertFalse(self.module().summarize(reports, {'result': 'PASS'})['retention'])
        reports = fixture()
        for index in (1, 2):
            for sample in reports[index]['baseline']['rounds'][0]['samples'][-40:]:
                sample['loop_s'] = .12
        result = self.module().summarize(reports, {'result': 'PASS'})
        self.assertFalse(result['retention'])
        self.assertGreaterEqual(result['gates']['p95_worse_pairs'], 2)

    def test_noise_gain_cannot_accept(self):
        reports = fixture()
        for index, mean in enumerate((.08, .075, .095, .1, .12, .115)):
            for sample in reports[index]['baseline']['rounds'][0]['samples']:
                sample['loop_s'] = mean
        report = self.module().summarize(reports, {'result': 'PASS'})
        self.assertFalse(report['retention'])
        self.assertFalse(report['gates']['gain_exceeds_variation'])

    def test_mismatched_or_unverified_evidence_cannot_accept(self):
        module = self.module()
        for mutate in (lambda r: r[1]['baseline'].update(renderer='RayTracedLighting'),
                       lambda r: r[1]['file_hashes_after'].update(controller='changed'),
                       lambda r: r[1]['baseline']['rounds'][0].update(profiled=True),
                       lambda r: r[1]['baseline']['rounds'][0]['samples'].pop(),
                       lambda r: r[1]['baseline'].update(motion_sequence_sha256='other')):
            reports = fixture()
            mutate(reports)
            self.assertFalse(module.summarize(reports, {'result': 'PASS'})['retention'])
        self.assertFalse(module.summarize(fixture(), {'result': 'FAIL'})['retention'])

    def test_motion_is_bounded_deterministic_and_resets_controller_state(self):
        import isaac_thermal_baseline as baseline
        motion = getattr(baseline, 'commanded_pose', None)
        self.assertTrue(callable(motion), 'deterministic commanded motion must exist')
        origin = {'position': [1., 2., 3.], 'yaw': .4}
        first = motion(origin, 0)
        self.assertEqual(first, {'position': [1., 2., 3.], 'yaw': .4})
        poses = [motion(origin, i) for i in range(600)]
        self.assertEqual(poses, [motion(origin, i) for i in range(600)])
        self.assertTrue(any(p != first for p in poses))
        for p in poses:
            self.assertLessEqual(max(abs(a-b) for a, b in zip(p['position'], origin['position'])), .2)
            self.assertLessEqual(abs(p['yaw']-.4), .06)
        controller = SimpleNamespace(keys=SimpleNamespace(pressed={'W'}), position=(0,0,0),
            yaw=0, velocity=(1,1,1,1), roll=1, pitch=1, roll_rate=1, pitch_rate=1)
        baseline.apply_commanded_pose(controller, poses[12])
        self.assertEqual(controller.position, tuple(poses[12]['position']))
        self.assertFalse(controller.keys.pressed)
        self.assertEqual(controller.velocity, (0.,0.,0.,0.))
        self.assertEqual((controller.roll, controller.pitch, controller.roll_rate, controller.pitch_rate), (0,0,0,0))


if __name__ == '__main__':
    unittest.main()
