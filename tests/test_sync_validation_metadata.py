import unittest
from pathlib import Path
import tempfile

from scripts.sync_validation_metadata import (
    SYNC_EVIDENCE_PATH, PROJECT_ROOT, SYNC_CONTRACT_CHECKS,
    build_sync_validation_metadata,
    frame_sync_contract_passes,
    frame_sync_validation_passes,
)


class SyncValidationMetadataTests(unittest.TestCase):
    def setUp(self):
        self.valid_frame = {
            'capture_mode': 'offline_frozen_state_render',
            'explicit_render_triggered': True,
            'explicit_render_completed': True,
            'frozen_state_capture': True,
            'pair_pose_max_error': 1e-12,
            'simulation_capture_drift_s': 0.0,
            'simulation_time_before': 12.0,
            'simulation_time_after': 12.0,
            'rgb_resolution': [512, 410],
            'thermal_resolution': [256, 205],
            'thermal_coverage': {'complete_for_this_view': True},
            'scene_config_sha256': '1407c3cd3d590f5f44debce0086510e5c754d31f69d660e3edb8e79417cb6166',
            'source_stage_sha256': 'c312ef134aad9d88e66c225c8e03982a0fa1a9292793c0e63dd9e39f87685e06',
            'thermal_config_sha256': 'a1bdc184bbdfa24d4bad1dbdf1a905638a63058dc2bcb79cd8898c660428fac0',
            'sensor_config_sha256': '66f653994c3d597fd772fcffed6f466e409374f58cad713ad61f124b9977c7b2',
            'thermal_environment_config_sha256': '5986cccbcfad36a2e5fb6e9ed78be63f469050004481328286180cb0fdcedd35',
            'display_config_sha256': '4747592b44a0d765de6e136bad0eaf2eb5d5290fcd2afb2e19c979f091f22070',
            'rgb_camera_path': '/World/Crazyflie/body/Camera_mount/Camera',
            'thermal_camera_path': '/World/Crazyflie/body/Sensors/ThermalCamera',
            'rgb_render_options': {'renderer': 'PathTracing'},
            'rgb_intrinsics_px': {'fx_px': 309.71839192708336, 'fy_px': 309.71839192708336,
                                  'cx_px': 256.0, 'cy_px': 205.0, 'width': 512, 'height': 410},
            'thermal_intrinsics_px': {'fx_px': 154.85919596354168, 'fy_px': 154.85919596354168,
                                      'cx_px': 128.0, 'cy_px': 102.5, 'width': 256, 'height': 205},
        }

    def test_verified_capture_pipeline_and_passing_frame_contract_are_marked_validated(self):
        metadata = build_sync_validation_metadata(self.valid_frame)

        self.assertEqual(metadata['sync_basis'], 'strict_probe')
        self.assertTrue(metadata['strict_sync_verified'])
        self.assertTrue(metadata['sync_validation_status'] == 'validated')
        self.assertEqual(metadata['sync_validation_scope'], 'pipeline_and_frame_capture_contract')
        self.assertEqual(metadata['sync_status'], 'synchronized_pipeline_validated')
        self.assertFalse(metadata['strict_per_frame_pixel_sync_verified'])
        self.assertEqual(metadata['sync_validation_evidence']['result'], 'PASS')
        self.assertEqual(metadata['sync_validation_evidence']['frames_tested'], 16)
        self.assertTrue(all(metadata['sync_validation_checks'].values()))

    def test_failed_frame_contract_cannot_receive_a_verified_sync_label(self):
        invalid = dict(self.valid_frame, explicit_render_completed=False)

        with self.assertRaisesRegex(RuntimeError, 'sync contract'):
            build_sync_validation_metadata(invalid)

    def test_live_preview_is_never_promoted_to_verified_offline_sync(self):
        metadata = build_sync_validation_metadata({'capture_mode': 'live_preview'})

        self.assertFalse(metadata['strict_sync_verified'])
        self.assertFalse(metadata['strict_per_frame_pixel_sync_verified'])
        self.assertEqual(metadata['sync_validation_status'], 'not_verified')
        self.assertEqual(metadata['sync_status'], 'live_preview_not_strict')

    def test_frame_sync_status_requires_all_runtime_checks_and_matching_evidence(self):
        metadata = build_sync_validation_metadata(self.valid_frame)
        frame = {**self.valid_frame, **metadata}
        self.assertTrue(frame_sync_validation_passes(frame, metadata['sync_validation_evidence']))
        frame['sync_validation_checks'] = dict(metadata['sync_validation_checks'], camera_pose_match=False)
        self.assertFalse(frame_sync_validation_passes(frame, metadata['sync_validation_evidence']))

    def test_changed_scene_or_resolution_cannot_inherit_previous_validation(self):
        changes = (
            {'scene_config_sha256': 'different-scene-config'},
            {'source_stage_sha256': 'different-usd'},
            {'rgb_resolution': [640, 512], 'thermal_resolution': [640, 512]},
            {'sensor_config_sha256': 'different-sensor'},
            {'thermal_environment_config_sha256': 'different-environment'},
            {'thermal_intrinsics_px': {'fx_px': 1}},
        )
        for change in changes:
            with self.subTest(change=change):
                metadata = build_sync_validation_metadata({**self.valid_frame, **change})
                # A different scene must not inherit the school probe's strict label.
                self.assertEqual(metadata['sync_basis'], 'project_convention')
                self.assertFalse(metadata['strict_sync_verified'])
                self.assertFalse(metadata['strict_per_frame_pixel_sync_verified'])
                self.assertEqual(metadata['sync_validation_status'], 'convention_only')
                self.assertEqual(metadata['sync_validation_scope'], 'frame_capture_contract_only')
                self.assertTrue(metadata['sync_scope_failures'])
                # The frame-level contract still passed and is still reported.
                self.assertTrue(frame_sync_contract_passes(metadata))

    def test_convention_frames_pass_their_contract_but_never_strict_validation(self):
        metadata = build_sync_validation_metadata(
            {**self.valid_frame, 'scene_config_sha256': 'different-scene-config'})
        frame = {**self.valid_frame, **metadata}
        self.assertTrue(frame_sync_contract_passes(frame))
        self.assertFalse(frame_sync_validation_passes(frame, None))

    def test_missing_evidence_falls_back_to_project_convention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = build_sync_validation_metadata(self.valid_frame, root)
            self.assertEqual(metadata['sync_basis'], 'project_convention')
            self.assertFalse(metadata['strict_sync_verified'])
            self.assertIsNone(metadata['sync_validation_evidence'])

    def test_modified_evidence_is_an_integrity_error_not_a_scope_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / SYNC_EVIDENCE_PATH
            path.parent.mkdir(parents=True)
            path.write_bytes((PROJECT_ROOT / SYNC_EVIDENCE_PATH).read_bytes() + b' ')
            with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                build_sync_validation_metadata(self.valid_frame, root)


if __name__ == '__main__':
    unittest.main()
