"""Regression tests for scene-independent manual recovery and its quality gate."""
import argparse
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.trajectory_recorder import TrajectoryRecorder
from tests.test_recording_undo import state


class SharedCollectionRecoveryTests(unittest.TestCase):
    def parse_live(self, argv):
        from scripts import start_crazyflie_dual_camera_view as live
        original = argparse.ArgumentParser.parse_args
        parsed = []
        def capture(parser, values):
            result = original(parser, values)
            parsed.append(result)
            return result
        class BeforeStartup(Exception):
            pass
        with patch.object(argparse.ArgumentParser, 'parse_args', capture), \
             patch.object(live, 'load_scene_config', side_effect=BeforeStartup):
            with self.assertRaises(BeforeStartup):
                live.main(argv)
        return parsed[0]

    def test_manual_recording_defaults_to_recovery_for_any_config(self):
        args = self.parse_live(['--record-trajectory', '--config', 'future-scene.yaml'])
        self.assertTrue(args.collision_undo)
        self.assertEqual((args.trajectory_period, args.preview_fps, args.thermal_preview_fps), (1., 15., 2.))

    def test_preview_and_synthetic_probe_keep_existing_default(self):
        self.assertFalse(self.parse_live([]).collision_undo)
        self.assertFalse(self.parse_live(['--record-trajectory', '--probe-control']).collision_undo)

    def test_explicit_opt_out_and_opt_in(self):
        self.assertFalse(self.parse_live(['--record-trajectory', '--no-collision-undo']).collision_undo)
        self.assertTrue(self.parse_live(['--record-trajectory', '--probe-control', '--collision-undo']).collision_undo)

    def test_factory_and_forest_episode_arguments_inherit_recovery(self):
        from scripts.collection_episode import episode_preview_args, recording_identity, load_episodes
        for scene in ('factory', 'forest'):
            row = next(r for r in load_episodes() if r['scene'] == scene)
            identity = recording_identity(row, 'contract')
            args = episode_preview_args(row, Path('launch'), identity)
            parsed = self.parse_live(args[2:])
            self.assertTrue(parsed.collision_undo)
            self.assertEqual((parsed.trajectory_period, parsed.preview_fps, parsed.thermal_preview_fps), (1., 15., 2.))
            off = episode_preview_args(row, Path('launch'), identity, collision_undo=False)
            self.assertFalse(self.parse_live(off[2:]).collision_undo)

    def test_pilot_snapshot_matches_preview_recovery_policy(self):
        from scripts import factory_bc_pilot as factory, city_bc_pilot as city
        from scripts.factory_collection_config import resolve_selection
        with patch.object(factory, 'resolve_selection',
                          side_effect=lambda *a: resolve_selection(*a, require_ready=False)):
            for scene, module in (('factory', factory), ('city', city)):
                for requested in (None, False, True):
                    launch = module.build_launch(f'{scene}-bc-pilot-recovery-contract', collision_undo=requested)
                    enabled = requested is not False
                    self.assertEqual(launch['collision_undo'], enabled)
                    self.assertEqual(self.parse_live(launch['preview_args']).collision_undo, enabled)

    def recovered_run(self, folder):
        recorder = TrajectoryRecorder.create(folder, {'trajectory_source': 'manual_driving', 'sample_period_s': 1.}, 'recovery')
        for i in range(7):
            recorder.append(state(i, float(i), float(i)))
        recorder.record_collision(6.2, {'object_prim_path': '/Wall'})
        recorder.undo_to(3, 8.)
        recorder.append(state(7, 9., 4.))
        recorder.append(state(8, 10., 5.))
        return recorder.records, recorder.metadata

    def test_quality_accepts_only_documented_recovery_gap_and_keeps_raw_count(self):
        from scripts.factory_bc_pilot import recording_quality
        with tempfile.TemporaryDirectory() as folder:
            rows, metadata = self.recovered_run(folder)
            original = copy.deepcopy(rows)
            report = recording_quality(rows, sample_period_s=1., metadata=metadata)
            self.assertEqual(report['result'], 'PASS', report)
            self.assertEqual(report['record_count'], 9)
            self.assertEqual(report['corrected_record_count'], 6)
            self.assertEqual(report['corrected_sample_gap_max_s'], 1.)
            self.assertEqual(rows, original)

    def test_quality_still_rejects_missing_samples_and_unresolved_collision(self):
        from scripts.factory_bc_pilot import recording_quality
        with tempfile.TemporaryDirectory() as folder:
            rows, metadata = self.recovered_run(folder)
            for kind in ('gap', 'clock', 'unresolved', 'fake_undo'):
                with self.subTest(kind=kind):
                    r, m = copy.deepcopy(rows), copy.deepcopy(metadata)
                    if kind == 'gap': r[-1]['simulation_time'] += 2.
                    if kind == 'clock': r[1]['simulation_time'] = 0.
                    if kind == 'unresolved': m['collision_pending'] = True
                    if kind == 'fake_undo': m['collection_edit_events'].pop(0)
                    self.assertEqual(recording_quality(r, sample_period_s=1., metadata=m)['result'], 'FAIL')

    def test_second_collision_before_next_one_hz_sample_is_valid(self):
        from scripts.factory_bc_pilot import recording_quality
        with tempfile.TemporaryDirectory() as folder:
            r = TrajectoryRecorder.create(folder, {'trajectory_source': 'manual_driving', 'sample_period_s': 1.}, 'chain')
            for i in range(7):
                r.append(state(i, float(i), float(i)))
            r.record_collision(6.2, {'object_prim_path': '/Wall'})
            r.undo_to(3, 8.)
            r.record_collision(8.5, {'object_prim_path': '/OtherWall'})
            r.undo_to(3, 10.)
            for resumed in (False, True):
                with self.subTest(resumed=resumed):
                    if resumed:
                        r.append(state(7, 11., 4.))
                    report = recording_quality(r.records, sample_period_s=1., metadata=r.metadata)
                    self.assertEqual(report['result'], 'PASS', report)


if __name__ == '__main__':
    unittest.main()
