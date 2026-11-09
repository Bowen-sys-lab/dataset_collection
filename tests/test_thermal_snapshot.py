import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts.thermal_snapshot import (owned_copy, compare_arrays, state_digest,
                                     save_snapshot, load_snapshot, FrozenPreviewSky)


class SnapshotTests(unittest.TestCase):
    def test_sky_snapshot_preserves_shared_atmosphere_identity(self):
        from scripts.thermal_snapshot import preview_sky_memo
        sky = SimpleNamespace(source=SimpleNamespace(_atm=SimpleNamespace(cache={}),
            weather=SimpleNamespace()), start_s=0., interval_s=30.,
            elevation=np.array([0., 1.]), tilt=np.array([0., 1.]),
            _table=lambda i: (np.array([2., 4.]), np.array([2., 4.])))
        memo = preview_sky_memo(sky, 0.)
        from copy import deepcopy
        clone = deepcopy({'sky': sky, 'atmosphere': sky.source._atm}, memo)
        self.assertIs(clone['sky'].source._atm, clone['atmosphere'])

    def test_frozen_sky_temperature_uses_interpolated_radiance_not_source(self):
        class Lut:
            def apparent_temperature(self, radiance, quantity):
                return radiance * 2
        class Source:
            quantity = 'lb'
            lut = Lut()
            def apparent_temperature_k(self, *args):
                raise AssertionError('Must not delegate to non-interpolated source')
        sky = SimpleNamespace(source=Source(), start_s=0., interval_s=30.,
            elevation=np.array([0., 1.]), tilt=np.array([0., 1.]),
            _table=lambda i: (np.array([2., 4.])+i, np.array([2., 4.])+i))
        frozen = FrozenPreviewSky(sky, 15.)
        np.testing.assert_array_equal(frozen.apparent_temperature_k(15., np.array([0., 1.])),
                                      np.array([5., 9.]))

    def test_assembly_comparison_rejects_wrong_material_even_with_equal_outputs(self):
        from scripts.compare_thermal_snapshots import compare_assembly_results
        value = np.array([[1.]], np.float32)
        output = SimpleNamespace(**{k: value.copy() for k in
            ('radiance', 'apparent_t', 'dn16', 'display8', 'signal_dn', 'flux')},
            isp_hash='test', report={})
        left = {'planes': {'material_id': np.array([[3]], np.int32)},
                'outputs': output, 'history': {'frame_index': 1}}
        right = owned_copy(left)
        self.assertTrue(compare_assembly_results(left, right)['exact'])
        right['planes']['material_id'][0, 0] = 5
        self.assertFalse(compare_assembly_results(left, right)['exact'])
        right = owned_copy(left)
        right['history']['frame_index'] = 2
        self.assertFalse(compare_assembly_results(left, right)['exact'])

    def test_material_cache_sequence_rejects_captured_planes_with_equal_outputs(self):
        from scripts.compare_thermal_snapshots import compare_material_cache_sequence
        from irsim.materials.mapping import Resolution
        from irsim_isaac.pipeline.material_ids import material_id_plane
        from scripts.thermal_runtime import material_id_plane_cached

        value = np.array([[1.]], np.float32)
        output = SimpleNamespace(**{k: value.copy() for k in
            ('radiance', 'apparent_t', 'dn16', 'display8', 'signal_dn', 'flux')},
            isp_hash='test', report={})
        planes = {'material_id': np.array([[1]], np.int32)}
        history = {'frame_index': 1, 'lag': value.copy()}
        checkpoint = {
            'attributes': {
                'config': SimpleNamespace(materials=SimpleNamespace(names=['unmapped', 'A', 'B'])),
                'resolutions': [Resolution('/World/A', 'A', 1, 'override')]},
            'raw_aovs': SimpleNamespace(instance_id=np.array([[7]], np.uint32),
                device_handles={'instance': {'info': {'idToLabels': {'7': '/World/A'}}}}),
            'result': {'planes': planes, 'outputs': output, 'history': history}}
        snapshot = {'metadata': {'case': 'static'}, 'assembly_checkpoint': checkpoint,
                    'config': SimpleNamespace(chain=None), 'planes': owned_copy(planes),
                    'outputs': owned_copy(output), 'assembly_history_after': owned_copy(history)}

        # Replace only the external camera execution; retain real mapping,
        # exact comparisons, captured gates and wrong-mapping sensitivity.
        def run_owned_checkpoint(checkpoint, *, cache_enabled, cache=None):
            result = owned_copy(checkpoint['result'])
            raw = checkpoint['raw_aovs']
            args = (raw.instance_id, raw.device_handles['instance']['info']['idToLabels'],
                    checkpoint['attributes']['resolutions'])
            result['planes']['material_id'] = (material_id_plane_cached(*args, cache=cache)
                                              if cache_enabled else material_id_plane(*args))
            return result

        corruptions = {
            'bytes': {'material_id': np.array([[2]], np.int32)},
            'dtype': {'material_id': np.array([[1]], np.int64)},
            'shape': {'material_id': np.array([1], np.int32)},
            'missing': {},
            'extra': {**owned_copy(planes), 'extra': value.copy()}}
        with TemporaryDirectory() as folder, \
                patch('scripts.thermal_snapshot.run_assembly_checkpoint', run_owned_checkpoint), \
                patch('scripts.compare_thermal_snapshots.scenario_checks',
                      return_value={'single_checkpoint_fixture': True}):
            path = Path(folder)/'frame.pkl'
            save_snapshot(path, snapshot)
            self.assertEqual(compare_material_cache_sequence([path])['result'], 'PASS')
            for change, changed_planes in corruptions.items():
                with self.subTest(change=change):
                    corrupted = owned_copy(snapshot)
                    corrupted['planes'] = changed_planes
                    save_snapshot(path, corrupted)
                    report = compare_material_cache_sequence([path])
                    frame = report['frames'][0]
                    self.assertTrue(frame['comparison']['exact'])
                    self.assertTrue(frame['matches_captured']['exact'])
                    self.assertTrue(frame['captured_history_exact'])
                    self.assertTrue(report['wrong_mapping_detected'])
                    self.assertEqual(report['result'], 'FAIL')
                    self.assertFalse(frame['pass'])
                    name = 'extra' if change == 'extra' else 'material_id'
                    self.assertFalse(frame['captured_planes'][name]['exact'])

    def test_sequence_rejects_corrupt_captured_output_or_history(self):
        from scripts.compare_thermal_snapshots import compare_sequence
        from scripts.thermal_snapshot import sensor_history
        def kernel(planes, config, state, points, rotors):
            state.frame_index += 1
            value = planes['temperature_k'].copy()
            return SimpleNamespace(**{k: value.copy() for k in
                ('radiance', 'apparent_t', 'dn16', 'display8', 'signal_dn', 'flux')},
                isp_hash='test', report={})
        config = SimpleNamespace(flat_field=None, chain=None, noise=None)
        state = SimpleNamespace(t_s=0., frame_index=0)
        planes = {'temperature_k': np.array([[300.]], dtype=np.float32)}
        output = kernel(planes, config, owned_copy(state), (), ())
        after = SimpleNamespace(t_s=0., frame_index=1)
        snapshot = {'metadata': {'case': 'static', 'snapshot_id': 0},
                    'config': config, 'state_before': state, 'planes': planes,
                    'point_targets': (), 'rotor_veils': (), 'outputs': output,
                    'history_after': sensor_history(config, after)}
        with TemporaryDirectory() as folder:
            path = Path(folder)/'frame.pkl'
            save_snapshot(path, snapshot)
            self.assertEqual(compare_sequence([path], frame_function=kernel,
                                             require_scenarios=False)['result'], 'PASS')
            snapshot['outputs'].radiance[0, 0] = 301.
            save_snapshot(path, snapshot)
            self.assertEqual(compare_sequence([path], frame_function=kernel,
                                             require_scenarios=False)['result'], 'FAIL')
            snapshot['outputs'] = output = kernel(planes, config, owned_copy(state), (), ())
            snapshot['history_after'] = sensor_history(config, state)
            save_snapshot(path, snapshot)
            self.assertEqual(compare_sequence([path], frame_function=kernel,
                                             require_scenarios=False)['result'], 'FAIL')

    def test_snapshot_owns_arrays_and_independent_sensor_history(self):
        source = {'plane': np.arange(4, dtype=np.float32),
                  'state': SimpleNamespace(buffers={'lag': np.ones(4)})}
        left, right = owned_copy(source), owned_copy(source)
        left['plane'][0] = 99
        left['state'].buffers['lag'][0] = 8
        self.assertEqual(source['plane'][0], 0)
        self.assertEqual(right['state'].buffers['lag'][0], 1)
        self.assertNotEqual(state_digest(left['state']), state_digest(right['state']))

    def test_one_float_bit_or_dn_change_fails_without_tolerance(self):
        original = np.array([300., 301.], np.float32)
        changed = original.copy()
        changed[0] = np.nextafter(changed[0], np.float32(400.))
        self.assertFalse(compare_arrays(original, changed)['exact'])
        self.assertGreater(compare_arrays(original, changed)['max_abs_error'], 0.)
        self.assertTrue(compare_arrays(original, original.copy())['exact'])
        self.assertFalse(compare_arrays(np.array([1], np.uint16), np.array([2], np.uint16))['exact'])
        self.assertFalse(compare_arrays(original, original.astype(np.float64))['exact'])

    def test_snapshot_roundtrip_preserves_arrays_and_rng_history(self):
        value = {'array': np.arange(6).reshape(2, 3), 'rng': np.random.default_rng(7)}
        with TemporaryDirectory() as folder:
            path = Path(folder)/'frame.pkl'
            save_snapshot(path, value)
            read = load_snapshot(path)
            self.assertEqual(state_digest(value), state_digest(read))
            np.testing.assert_array_equal(value['rng'].normal(size=5), read['rng'].normal(size=5))

    def test_frozen_sky_uses_exact_tables_and_refuses_unrecorded_time(self):
        class Sky:
            start_s = 10.
            interval_s = 30.
            elevation = np.linspace(0., np.pi/2, 181)
            tilt = np.linspace(0., np.pi, 181)
            source = SimpleNamespace(quantity='lb')
            def _table(self, index):
                return np.arange(181, dtype=float)+index, np.arange(181, dtype=float)*2+index
        source = Sky()
        frozen = FrozenPreviewSky(source, 11.)
        view = np.array([0., .5, 1.])
        angle = np.arccos(np.clip(2*view-1, -1., 1.))
        weight = 1/30
        expected = np.interp(angle, source.tilt,
                             source._table(0)[1]*(1-weight)+source._table(1)[1]*weight)
        np.testing.assert_array_equal(frozen.effective_radiance_from_sky_view(11., view), expected)
        with self.assertRaises(ValueError):
            frozen.effective_radiance_from_sky_view(100., view)

    def test_changed_snapshot_bytes_are_not_deserialized(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'frame.pkl'
            save_snapshot(path, {'x': 1})
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaises(ValueError):
                load_snapshot(path)
