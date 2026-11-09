import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.thermal_runtime import (MaterialLookupCache, cached_material_mapping,
                                     cached_labels_to_paths, material_id_plane_cached,
                                     supported_aov_names, camera_class)


class SupportedAovTests(unittest.TestCase):
    def test_only_optional_ao_is_removed_and_original_object_restored(self):
        original = {'normals': ('Normals',), 'occlusion': ('AmbientOcclusion',),
                    'depth': ('DepthLinearized',)}
        module = SimpleNamespace(AOV_NAMES=original)
        with supported_aov_names(module):
            self.assertEqual(module.AOV_NAMES, {'normals': ('Normals',), 'depth': ('DepthLinearized',)})
            self.assertIn('occlusion', original)
        self.assertIs(module.AOV_NAMES, original)


class MaterialLookupCacheTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.array([[0, 4, 4], [7, 0, 7]], dtype=np.uint32)
        self.resolutions = [SimpleNamespace(path='/World/A', material_id=3),
                            SimpleNamespace(path='/World/B', material_id=5)]

    def test_nested_label_mutation_invalidates_lookup(self):
        cache = MaterialLookupCache(max_entries=8)
        labels = {4: {'class': '/World/A'}, 7: '/World/B'}
        first = material_id_plane_cached(self.ids, labels, self.resolutions, cache=cache)
        labels[4]['class'] = '/World/B'
        changed = material_id_plane_cached(self.ids, labels, self.resolutions, cache=cache)
        self.assertEqual(first.tolist(), [[0, 3, 3], [5, 0, 5]])
        self.assertEqual(changed.tolist(), [[0, 5, 5], [5, 0, 5]])
        self.assertEqual(cache.stats()['builds'], 2)

    def test_resolution_mutation_invalidates_lookup(self):
        cache = MaterialLookupCache(max_entries=8)
        material_id_plane_cached(self.ids, {4: '/World/A', 7: '/World/B'},
                                 self.resolutions, cache=cache)
        self.resolutions[0].material_id = 9
        changed = material_id_plane_cached(self.ids, {4: '/World/A', 7: '/World/B'},
                                           self.resolutions, cache=cache)
        self.assertEqual(changed.tolist(), [[0, 9, 9], [5, 0, 5]])
        self.assertEqual(cache.stats()['builds'], 2)

    def test_resolution_path_mutation_and_reused_id_change_material(self):
        cache = MaterialLookupCache()
        labels = {4: '/World/A', 7: '/World/B'}
        material_id_plane_cached(self.ids, labels, self.resolutions, cache=cache)
        self.resolutions[0].path = '/World/C'
        moved = material_id_plane_cached(self.ids, labels, self.resolutions, cache=cache)
        self.assertEqual(moved.tolist(), [[0, 0, 0], [5, 0, 5]])
        labels[4] = '/World/C'
        reused = material_id_plane_cached(self.ids, labels, self.resolutions, cache=cache)
        self.assertEqual(reused.tolist(), [[0, 3, 3], [5, 0, 5]])
        self.assertEqual(cache.stats()['builds'], 3)

    def test_integer_validation_background_and_unmapped_match_vendor(self):
        from irsim_isaac.pipeline.material_ids import material_id_plane
        cache = MaterialLookupCache()
        cases = [
            (np.array([[4, 4, 0]], dtype=np.int64), None, self.resolutions, {}),
            (np.array([[4, 7, 9]], dtype=np.uint32),
             {'4': {'class': '/World/A'}, '7': '/World/Unknown', '9': [], 'bad': '/World/A'},
             self.resolutions, {}),
            (np.array([[-1, 4, 7]], dtype=np.int64), {4: '/World/A', 7: '/World/B'},
             self.resolutions, {'background_id': -1, 'strict': True}),
            (np.array([[4, 7]], dtype=np.uint64), {4: '/World/A', 7: '/World/B'},
             self.resolutions, {'background_id': 4, 'strict': True}),
            (np.empty((0, 3), dtype=np.int32), {}, self.resolutions, {'strict': True}),
            # Vendor keys retain their type: coercing resolution path 4 to '4'
            # would incorrectly produce material 8 for this fixture.
            (np.array([[4]], dtype=np.int32), {4: '4'},
             [SimpleNamespace(path=4, material_id=8)], {}),
        ]
        for ids, labels, resolutions, kwargs in cases:
            with self.subTest(dtype=ids.dtype, labels=labels, kwargs=kwargs):
                expected = material_id_plane(ids, labels, iter(resolutions), **kwargs)
                actual = material_id_plane_cached(ids, labels, iter(resolutions), cache=cache, **kwargs)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.tobytes(), expected.tobytes())
        for invalid in (np.array([4], dtype=np.int32), np.ones((1, 1, 1), dtype=np.int32),
                        np.array([[True]], dtype=bool), np.array([[4]], dtype=object)):
            with self.subTest(invalid=invalid):
                try:
                    material_id_plane(invalid, {}, self.resolutions)
                except Exception as error:
                    with self.assertRaises(type(error)) as actual:
                        material_id_plane_cached(invalid, {}, self.resolutions, cache=cache)
                    self.assertEqual(str(actual.exception), str(error))

    def test_bad_resolution_is_validated_before_bad_labels_even_on_a_hit(self):
        from irsim_isaac.pipeline.material_ids import material_id_plane
        cache = MaterialLookupCache()
        material_id_plane_cached(self.ids, {4: '/World/A'}, self.resolutions, cache=cache)
        self.resolutions[0].material_id = 'not-an-integer'
        with self.assertRaises(ValueError) as expected:
            material_id_plane(self.ids, ['bad-label-payload'], self.resolutions)
        with self.assertRaises(ValueError) as actual:
            material_id_plane_cached(self.ids, ['bad-label-payload'], self.resolutions, cache=cache)
        self.assertEqual(str(actual.exception), str(expected.exception))

    def test_invalid_first_material_precedes_later_resolution_generator_failure(self):
        from irsim_isaac.pipeline.material_ids import material_id_plane

        def resolutions():
            yield SimpleNamespace(path='/World/A', material_id='not-an-integer')
            raise RuntimeError('later generator failure')

        with self.assertRaises(ValueError) as expected:
            material_id_plane(self.ids, {4: '/World/A'}, resolutions())
        with self.assertRaises(ValueError) as actual:
            material_id_plane_cached(self.ids, {4: '/World/A'}, resolutions(),
                                     cache=MaterialLookupCache())
        self.assertEqual(str(actual.exception), str(expected.exception))

    def test_unhashable_first_path_precedes_invalid_later_material(self):
        from irsim_isaac.pipeline.material_ids import material_id_plane

        def resolutions():
            yield SimpleNamespace(path=['unhashable'], material_id=3)
            yield SimpleNamespace(path='/World/A', material_id='not-an-integer')

        with self.assertRaises(TypeError) as expected:
            material_id_plane(self.ids, {4: '/World/A'}, resolutions())
        with self.assertRaises(TypeError) as actual:
            material_id_plane_cached(self.ids, {4: '/World/A'}, resolutions(),
                                     cache=MaterialLookupCache())
        self.assertEqual(str(actual.exception), str(expected.exception))

    def test_clear_can_reset_stats_for_measured_frame_interval(self):
        cache = MaterialLookupCache()
        args = (self.ids, {4: '/World/A'}, self.resolutions)
        material_id_plane_cached(*args, cache=cache)
        material_id_plane_cached(*args, cache=cache)
        self.assertEqual(cache.stats()['hits'], 1)
        cache.clear(reset_stats=True)
        self.assertEqual(cache.stats(), {'hits': 0, 'builds': 0, 'evictions': 0,
                                        'entries': 0, 'approximate_bytes': 0})

    def test_mutating_returned_label_table_cannot_corrupt_next_normalization(self):
        labels = {4: {'class': '/World/A'}}
        first = cached_labels_to_paths(labels)
        first[4] = '/World/B'
        self.assertEqual(cached_labels_to_paths(labels), {4: '/World/A'})

    def test_bounded_eviction_and_stats_are_reported(self):
        cache = MaterialLookupCache(max_entries=2)
        for ident in (4, 5, 6):
            material_id_plane_cached(np.array([[ident]], dtype=np.uint32),
                                     {ident: f'/World/{ident}'},
                                     [SimpleNamespace(path=f'/World/{ident}', material_id=ident)],
                                     cache=cache)
        stats = cache.stats()
        self.assertEqual(stats['entries'], 2)
        self.assertEqual(stats['builds'], 3)
        self.assertEqual(stats['evictions'], 1)
        self.assertGreater(stats['approximate_bytes'], 0)

    def test_cache_instances_are_independent_and_clear_is_supported(self):
        first, second = MaterialLookupCache(), MaterialLookupCache()
        args = (self.ids, {4: '/World/A', 7: '/World/B'}, self.resolutions)
        material_id_plane_cached(*args, cache=first)
        material_id_plane_cached(*args, cache=second)
        self.assertEqual(first.stats()['builds'], 1)
        self.assertEqual(second.stats()['builds'], 1)
        first.clear()
        self.assertEqual(first.stats()['entries'], 0)
        self.assertEqual(second.stats()['entries'], 1)

    def test_cached_mapping_preserves_vendor_validation_and_strict_exception(self):
        from irsim_isaac.pipeline.material_ids import material_id_plane
        invalid = np.array([[1.5]], dtype=np.float32)
        with self.assertRaises(TypeError) as vendor_error:
            material_id_plane(invalid, {}, self.resolutions)
        with self.assertRaises(TypeError) as cached_error:
            material_id_plane_cached(invalid, {}, self.resolutions,
                                     cache=MaterialLookupCache())
        self.assertEqual(str(cached_error.exception), str(vendor_error.exception))
        with self.assertRaises(KeyError) as vendor_error:
            material_id_plane(self.ids, {4: '/World/A'}, self.resolutions, strict=True)
        with self.assertRaises(KeyError) as cached_error:
            material_id_plane_cached(self.ids, {4: '/World/A'}, self.resolutions,
                                     strict=True, cache=MaterialLookupCache())
        self.assertEqual(str(cached_error.exception), str(vendor_error.exception))

    def test_cached_material_mapping_reuses_resolved_id_table(self):
        from irsim_isaac.pipeline import ir_camera
        original = ir_camera.material_id_plane
        cache = MaterialLookupCache()
        ids = np.array([[0, 4, 4], [7, 0, 7]], dtype=np.uint32)
        resolutions = (SimpleNamespace(path='/World/A', material_id=3),
                       SimpleNamespace(path='/World/B', material_id=5))
        with cached_material_mapping(ir_camera, cache):
            first = ir_camera.material_id_plane(ids, {4: '/World/A', 7: '/World/B'}, resolutions)
            second = ir_camera.material_id_plane(ids, {4: '/World/A', 7: '/World/B'}, resolutions)
            self.assertTrue(np.array_equal(first, [[0, 3, 3], [5, 0, 5]]))
            self.assertTrue(np.array_equal(second, first))
        self.assertIs(ir_camera.material_id_plane, original)
        self.assertEqual(cache.stats()['builds'], 1)
        self.assertEqual(cache.stats()['hits'], 1)

    def test_nested_mapping_scopes_restore_the_outer_camera_cache(self):
        from irsim_isaac.pipeline import ir_camera
        original = ir_camera.material_id_plane
        first, second = MaterialLookupCache(), MaterialLookupCache()
        args = (self.ids, {4: '/World/A', 7: '/World/B'}, self.resolutions)
        with cached_material_mapping(ir_camera, first):
            ir_camera.material_id_plane(*args)
            with cached_material_mapping(ir_camera, second):
                ir_camera.material_id_plane(*args)
            ir_camera.material_id_plane(*args)
        self.assertEqual(first.stats()['hits'], 1)
        self.assertEqual(second.stats()['hits'], 0)
        self.assertEqual(second.stats()['builds'], 1)
        self.assertIs(ir_camera.material_id_plane, original)

    def test_cached_mapping_matches_vendor_mapping(self):
        from irsim_isaac.pipeline.material_ids import material_id_plane
        ids = np.array([[0, 4, 4], [7, 0, 7]], dtype=np.uint32)
        resolutions = (SimpleNamespace(path='/World/A', material_id=3),
                       SimpleNamespace(path='/World/B', material_id=5))
        expected = material_id_plane(ids, {4: '/World/A', 7: '/World/B'}, resolutions)
        actual = material_id_plane_cached(ids, {4: '/World/A', 7: '/World/B'}, resolutions)
        self.assertTrue(np.array_equal(actual, expected))

    def test_cached_label_normalization_reuses_equal_payload(self):
        payload = {'4': '/World/A', '7': '/World/B'}
        first = cached_labels_to_paths(payload)
        second = cached_labels_to_paths(dict(payload))
        self.assertEqual(first, {4: '/World/A', 7: '/World/B'})
        self.assertEqual(second, first)

    def test_vendor_map_restored_even_when_camera_open_fails(self):
        module = SimpleNamespace(AOV_NAMES={'occlusion': ('AmbientOcclusion',)})
        original = module.AOV_NAMES
        with self.assertRaises(RuntimeError):
            with supported_aov_names(module):
                raise RuntimeError('camera initialization failed')
        self.assertIs(module.AOV_NAMES, original)


class CameraCacheLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from irsim.config.loader import load_sensor_config
        from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
        from irsim.scene import Scene
        from irsim.materials.library import MaterialLibrary
        from irsim.materials.table import MaterialTable
        from irsim.pipeline.core import PipelineConfig
        root = Path(__file__).resolve().parents[1] / 'third_party/isaac-thermal-camera-simulation'
        cls.sensor = load_sensor_config(root / 'configs/sensors/flir_boson_640_lwir.yaml')
        band = cls.sensor.sensor.band.band_id
        lut = load_band_lut_for_config(cls.sensor, root / 'data/lut')
        response = load_band_response_for_config(cls.sensor, root / 'data')
        cls.scene = Scene.from_file(root / 'configs/scenes/quad_flight_clear_noon.yaml',
                                    {band: lut}, responses={band: response})
        materials = MaterialTable.from_library(MaterialLibrary.load(), band)
        cls.pipeline = PipelineConfig.from_sensor(cls.sensor, materials, lut,
                                                 noise_enabled=False, psf_enabled=False)

    def make_camera(self, enabled=True):
        from irsim_isaac.pipeline.gbuffer_isaac import RawAovs
        camera = camera_class(material_cache_enabled=enabled)(
            self.sensor, deepcopy(self.scene), pipeline=self.pipeline,
            prim_to_target={'/World/A': 'airframe', '/World/B': 'airframe'},
            resolutions=[SimpleNamespace(path='/World/A', material_id=3),
                         SimpleNamespace(path='/World/B', material_id=5)])
        ids = np.array([[0, 4, 4], [7, 0, 7]], dtype=np.uint32)
        position = np.zeros((2, 3, 3), dtype=np.float32)
        position[..., 2] = -2.0
        normal = np.zeros_like(position)
        normal[..., 2] = 1.0
        raw = RawAovs(distance_m=np.full((2, 3), 2.0, dtype=np.float32),
                      normal=normal, position=position, instance_id=ids,
                      device_handles={'instance': {'info': {'idToLabels':
                                      {4: '/World/A', 7: '/World/B'}}}})
        # Only the renderer I/O is supplied by a CPU fixture. Vendor camera
        # construction, assembly, mapping and thermal bridge remain real.
        camera._reader = SimpleNamespace(read=lambda: raw, detach=lambda: None)
        return camera

    def test_actual_camera_planes_use_owned_cache_and_independent_cameras(self):
        first, second = self.make_camera(), self.make_camera()
        a = first.planes(step=False)
        b = second.planes(step=False)
        first.planes(step=False)
        self.assertEqual(a['material_id'].tolist(), [[0, 3, 3], [5, 0, 5]])
        np.testing.assert_array_equal(a['material_id'], b['material_id'])
        self.assertEqual(first.material_cache_stats()['hits'], 1)
        self.assertEqual(first.material_cache_stats()['builds'], 1)
        self.assertEqual(second.material_cache_stats()['hits'], 0)
        first.clear_material_cache()
        self.assertEqual(first.material_cache_stats()['entries'], 0)
        self.assertEqual(second.material_cache_stats()['entries'], 1)

    def test_camera_default_is_off_and_uses_vendor_mapping(self):
        default_class = camera_class()
        self.assertFalse(default_class.mapping_cache_enabled)
        camera = self.make_camera(enabled=False)
        planes = camera.planes(step=False)
        self.assertEqual(planes['material_id'].tolist(), [[0, 3, 3], [5, 0, 5]])
        self.assertEqual(camera.material_cache_stats()['entries'], 0)
        self.assertEqual(camera.material_cache_stats()['builds'], 0)

    def test_close_releases_camera_storage(self):
        camera = self.make_camera()
        camera.planes(step=False)
        camera.close()
        self.assertEqual(camera.material_cache_stats()['entries'], 0)
        self.assertEqual(camera.material_cache_stats()['approximate_bytes'], 0)

    def test_open_reopen_and_open_failure_clear_storage_and_restore_aov_names(self):
        from irsim_isaac.pipeline import ir_camera
        camera = self.make_camera()
        original = ir_camera.AOV_NAMES
        # Vendor open is exclusively Kit/render-product/annotator work. Mock
        # that boundary while running the actual project lifecycle wrapper.
        with patch.object(ir_camera.IrCamera, 'open', return_value=camera):
            for _ in range(2):
                camera.planes(step=False)
                camera.open(settle_frames=0)
                self.assertEqual(camera.material_cache_stats()['entries'], 0)
                self.assertIs(ir_camera.AOV_NAMES, original)
        camera.planes(step=False)
        with patch.object(ir_camera.IrCamera, 'open', side_effect=RuntimeError('open failed')):
            with self.assertRaisesRegex(RuntimeError, 'open failed'):
                camera.open()
        self.assertEqual(camera.material_cache_stats()['entries'], 0)
        self.assertIs(ir_camera.AOV_NAMES, original)

    def test_scene_and_config_replacement_clear_even_equal_mapping_storage(self):
        camera = self.make_camera()
        camera.planes(step=False)
        camera.scene = deepcopy(camera.scene)
        self.assertEqual(camera.material_cache_stats()['entries'], 0)
        camera.planes(step=False)
        camera.config = deepcopy(camera.config)
        self.assertEqual(camera.material_cache_stats()['entries'], 0)
        camera.planes(step=False)
        self.assertEqual(camera.material_cache_stats()['builds'], 3)

    def test_stage_replacement_releases_storage(self):
        camera = self.make_camera()
        camera.planes(step=False)
        camera._stage = object()
        self.assertEqual(camera.material_cache_stats()['entries'], 0)

    def test_failed_close_still_releases_storage(self):
        camera = self.make_camera()
        camera.planes(step=False)
        def failing_detach():
            raise RuntimeError('detach failed')
        camera._reader.detach = failing_detach
        with self.assertRaisesRegex(RuntimeError, 'detach failed'):
            camera.close()
        self.assertEqual(camera.material_cache_stats()['entries'], 0)

    def test_scope_restores_vendor_functions_after_strict_camera_failure(self):
        from irsim_isaac.pipeline import ir_camera, aerial_bridge, motion_isaac
        modules = (ir_camera, aerial_bridge, motion_isaac)
        originals = [(module, getattr(module, 'material_id_plane', None),
                      getattr(module, 'labels_to_paths', None)) for module in modules]
        camera = self.make_camera()
        camera.resolutions[0].material_id = 'invalid'
        with self.assertRaises(ValueError):
            camera.planes(step=False)
        for module, material, labels in originals:
            if material is not None:
                self.assertIs(module.material_id_plane, material)
            if labels is not None:
                self.assertIs(module.labels_to_paths, labels)


if __name__ == '__main__':
    unittest.main()
