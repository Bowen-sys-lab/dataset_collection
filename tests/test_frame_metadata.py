"""Tests for compact, backward-compatible replay frame metadata."""
import json
from pathlib import Path
import tempfile
import unittest


class SelectedFrameMetadataTests(unittest.TestCase):
    def test_selected_fields_resolve_common_values_without_returning_large_payload(self):
        import tempfile
        from pathlib import Path
        from scripts.frame_metadata import FrameMetadataReader, FrameMetadataWriter
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            writer = FrameMetadataWriter(root, {'resolution': [10, 20], 'thermal_configuration': {'large': [1]*10000}})
            writer.write(0, {'resolution': [10, 20], 'thermal_configuration': {'large': [1]*10000}, 'time': 3})
            reader = FrameMetadataReader()
            selected = reader.read(root/'frame_000000.json', fields={'resolution', 'time'})
            self.assertEqual(selected, {'resolution': [10, 20], 'time': 3})
            selected['resolution'][0] = -1
            self.assertEqual(reader.read(root/'frame_000000.json', fields={'resolution'}), {'resolution': [10, 20]})

    def test_filter_cannot_bypass_common_hash_validation(self):
        import tempfile
        from pathlib import Path
        from scripts.frame_metadata import FrameMetadataReader, FrameMetadataWriter
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            writer = FrameMetadataWriter(root, {'resolution': [10, 20]})
            writer.write(0, {'resolution': [10, 20], 'time': 3})
            (root/'common_metadata.json').write_text('{}')
            with self.assertRaises(ValueError):
                FrameMetadataReader().read(root/'frame_000000.json', fields={'time'})

    def test_selected_legacy_and_overridden_or_omitted_fields(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.frame_metadata import FrameMetadataReader, FrameMetadataWriter
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            writer = FrameMetadataWriter(root, {'resolution': [10, 20], 'renderer': 'first'})
            writer.write(0, {'resolution': [30, 40], 'time': 3})
            reader = FrameMetadataReader()
            self.assertEqual(reader.read(root/'frame_000000.json', fields={'resolution','renderer','time'}),
                             {'resolution': [30, 40], 'time': 3})
            legacy = root/'legacy.json'
            legacy.write_text(json.dumps({'time': 2, 'unused': [1]*100}))
            self.assertEqual(reader.read(legacy, fields={'time'}), {'time': 2})
from copy import deepcopy


class FrameMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.frame_metadata import FrameMetadataWriter, load_frame_metadata
        cls.FrameMetadataWriter = FrameMetadataWriter
        cls.load_frame_metadata = staticmethod(load_frame_metadata)

    def test_shared_configuration_is_stored_once_and_each_frame_round_trips(self):
        common = {
            "thermal_configuration": {"assignments": {"/World/Object": {"material": "paint", "temperature_k": 301.0}}},
            "experiment_conditions": {"renderer": "PathTracing", "target": "/World/Target_01"},
            "sensor_config": {"width": 256, "height": 205},
            "rgb_intrinsics_px": {"fx": 250.0, "fy": 250.0},
            "thermal_environment_config_sha256": "env-sha",
            "display_config_sha256": "display-sha",
        }
        frames = [
            {**common, "frame_id": 0, "replay_simulation_time": 2.0,
             "position": [1.0, 2.0, 3.0], "thermal_target_temperatures_k": {"spot": 301.0}},
            {**common, "frame_id": 1, "replay_simulation_time": 2.2,
             "position": [1.1, 2.0, 3.0], "thermal_target_temperatures_k": {"spot": 302.0}},
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.FrameMetadataWriter(root, frames[0])
            writer.write(0, frames[0])
            writer.write(1, frames[1])

            common_file = json.loads((root / "common_metadata.json").read_text(encoding="utf-8"))
            raw_frame = json.loads((root / "frame_000000.json").read_text(encoding="utf-8"))
            self.assertEqual(common_file["metadata"]["thermal_configuration"], common["thermal_configuration"])
            self.assertEqual(common_file["metadata"]["thermal_environment_config_sha256"], "env-sha")
            self.assertEqual(common_file["metadata"]["display_config_sha256"], "display-sha")
            self.assertNotIn("thermal_configuration", raw_frame)
            self.assertNotIn("experiment_conditions", raw_frame)
            self.assertEqual(self.load_frame_metadata(root / "frame_000000.json"), frames[0])
            self.assertEqual(self.load_frame_metadata(root / "frame_000001.json"), frames[1])

    def test_frame_override_and_removal_of_common_field_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = {"thermal_configuration": {"version": 1}, "frame_id": 0}
            updated = {"thermal_configuration": {"version": 2}, "frame_id": 1}
            writer = self.FrameMetadataWriter(root, initial)
            writer.write(0, initial)
            writer.write(1, updated)

            self.assertEqual(self.load_frame_metadata(root / "frame_000001.json"), updated)

            removed = {"frame_id": 2}
            writer.write(2, removed)
            self.assertEqual(self.load_frame_metadata(root / "frame_000002.json"), removed)

    def test_reader_rejects_modified_common_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.FrameMetadataWriter(root, {"thermal_configuration": {"version": 1}, "frame_id": 0})
            writer.write(0, {"thermal_configuration": {"version": 1}, "frame_id": 0})
            common_path = root / "common_metadata.json"
            common_path.write_text(common_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "hash"):
                self.load_frame_metadata(root / "frame_000000.json")

    def test_reader_keeps_support_for_legacy_full_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame_000000.json"
            legacy = {"frame_id": 4, "thermal_configuration": {"assignments": [1, 2]},
                      "simulation_time_s": 12.5}
            path.write_text(json.dumps(legacy), encoding="utf-8")

            self.assertEqual(self.load_frame_metadata(path), legacy)

    def test_sync_validation_evidence_is_shared_without_losing_frame_status(self):
        common = {
            'sync_status': 'synchronized_pipeline_validated',
            'sync_validation_status': 'validated',
            'sync_validation_scope': 'pipeline_and_frame_capture_contract',
            'sync_validation_evidence': {'result': 'PASS', 'frames_tested': 16},
        }
        frame = {**common, 'frame_id': 0, 'strict_sync_verified': True,
                 'strict_per_frame_pixel_sync_verified': False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = self.FrameMetadataWriter(root, frame)
            writer.write(0, frame)

            shared = json.loads((root / 'common_metadata.json').read_text(encoding='utf-8'))['metadata']
            stored_frame = json.loads((root / 'frame_000000.json').read_text(encoding='utf-8'))
            restored = self.load_frame_metadata(root / 'frame_000000.json')

            self.assertEqual(shared['sync_validation_evidence'], common['sync_validation_evidence'])
            self.assertNotIn('sync_validation_evidence', stored_frame)
            self.assertEqual(restored, frame)

    def test_cached_reader_does_not_share_mutable_configuration_between_frames(self):
        from scripts.frame_metadata import FrameMetadataReader
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"sensor_config": {"gain": [1.0, 2.0]}, "frame_id": 0}
            writer = self.FrameMetadataWriter(root, metadata)
            writer.write(0, metadata)
            reader = FrameMetadataReader()
            first = reader.read(root / "frame_000000.json")
            first["sensor_config"]["gain"][0] = 99.0
            self.assertEqual(reader.read(root / "frame_000000.json"), metadata)

            path = root / "common_metadata.json"
            path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                reader.read(root / "frame_000000.json")

    def test_reader_rejects_missing_common_file_and_reference_outside_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"sensor_config": {"gain": 1}, "frame_id": 0}
            writer = self.FrameMetadataWriter(root, metadata)
            writer.write(0, metadata)
            path = root / "frame_000000.json"
            frame = json.loads(path.read_text(encoding="utf-8"))
            frame["common_metadata_ref"] = "../common_metadata.json"
            path.write_text(json.dumps(frame), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "reference"):
                self.load_frame_metadata(path)
            frame["common_metadata_ref"] = "common_metadata.json"
            path.write_text(json.dumps(frame), encoding="utf-8")
            (root / "common_metadata.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                self.load_frame_metadata(path)

    def test_writer_snapshots_common_configuration_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"sensor_config": {"gain": 1}, "frame_id": 0}
            writer = self.FrameMetadataWriter(root, metadata)
            writer.write(0, metadata)
            metadata["sensor_config"]["gain"] = 2
            metadata["frame_id"] = 1
            writer.write(1, metadata)
            self.assertEqual(self.load_frame_metadata(root / "frame_000000.json")["sensor_config"], {"gain": 1})
            self.assertEqual(self.load_frame_metadata(root / "frame_000001.json"), metadata)
            with self.assertRaises(FileExistsError):
                writer.write(1, metadata)
            with self.assertRaises(FileExistsError):
                self.FrameMetadataWriter(root, metadata)


class CompactCaptureTests(unittest.TestCase):
    def test_compact_save_preserves_image_and_raw_array_files(self):
        import numpy as np
        from scripts.frame_metadata import FrameMetadataWriter, load_frame_metadata
        from scripts.start_crazyflie_dual_camera_view import DualCameraPreview, ThermalFrame
        rgb = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        thermal = ThermalFrame(
            display8=rgb.copy(), radiance=np.ones((2, 3), dtype=np.float32),
            apparent_t=np.full((2, 3), 293.15, dtype=np.float32),
            dn16=np.arange(6, dtype=np.uint16).reshape(2, 3))
        original = {"frame_id": 0, "sensor_config": {"gain": 1}, "simulation_time_s": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / "full"
            compact = Path(directory) / "compact"
            preview = DualCameraPreview.__new__(DualCameraPreview)
            preview.save_frame(full, 0, rgb, thermal, deepcopy(original))
            writer = FrameMetadataWriter(compact, original)
            preview.save_frame(compact, 0, rgb, thermal, deepcopy(original), metadata_writer=writer)
            for source in full.glob('frame_*'):
                if source.suffix != '.json':
                    self.assertEqual(source.read_bytes(), (compact / source.name).read_bytes())
            restored = load_frame_metadata(compact / 'frame_000000.json')
            self.assertGreaterEqual(restored.pop('file_write_s'), 0.0)
            self.assertEqual(restored, original)


if __name__ == "__main__":
    unittest.main()
