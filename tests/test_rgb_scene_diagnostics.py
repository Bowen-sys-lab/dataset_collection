"""Behavior tests for the RGB/UV scene diagnostic helpers."""
import importlib.util
import unittest

import numpy as np


class RgbSceneDiagnosticTests(unittest.TestCase):
    def _diagnostics(self):
        spec = importlib.util.find_spec('scripts.rgb_scene_diagnostics')
        self.assertIsNotNone(spec, 'RGB scene diagnostic helpers have not been implemented')
        from scripts import rgb_scene_diagnostics
        return rgb_scene_diagnostics

    def test_uv_warning_parser_extracts_mesh_primvar_and_mismatch_sizes(self):
        diagnostics = self._diagnostics()
        record = diagnostics.parse_uv_warning(
            "[Warning] Mesh '/__Prototype_42/mesh_0' has corrupted data in primvar 'st_2': "
            'buffer size 6 does not match expected size 24 in faceVarying primvars'
        )

        self.assertEqual(record, {
            'mesh_prim_path': '/__Prototype_42/mesh_0',
            'prototype_path': '/__Prototype_42',
            'primvar_name': 'st_2',
            'reported_buffer_size': 6,
            'expected_face_varying_size': 24,
        })

    def test_patch_roi_uses_camera_projection_and_excludes_patch_edges(self):
        diagnostics = self._diagnostics()

        roi = diagnostics.project_patch_roi(
            local_xyz=(0.5, 0.2, -2.0), patch_size=(0.3, 0.3),
            focal_length_mm=18.0, horizontal_aperture_mm=36.0,
            vertical_aperture_mm=28.8, width=640, height=512,
            inner_fraction=0.5,
        )

        self.assertEqual(roi, (388, 212, 412, 236))

    def test_region_statistics_measure_color_of_the_requested_pixels(self):
        diagnostics = self._diagnostics()
        rgb = np.asarray([
            [[255, 0, 0], [0, 255, 0], [128, 128, 128]],
            [[0, 0, 255], [255, 255, 0], [0, 0, 0]],
        ], dtype=np.uint8)

        stats = diagnostics.region_statistics(rgb, (0, 0, 2, 2))

        self.assertEqual(stats['pixel_count'], 4)
        self.assertEqual(stats['median_rgb'], [127.5, 127.5, 0.0])
        self.assertEqual(stats['mean_rgb'], [127.5, 127.5, 63.75])
        self.assertAlmostEqual(stats['mean_hsv_saturation'], 1.0, places=6)

    def test_color_patch_modes_separate_lit_base_color_from_emission(self):
        diagnostics = self._diagnostics()
        self.assertTrue(hasattr(diagnostics, 'color_patch_channels'),
                        'lit and emissive color patch modes have not been implemented')

        diffuse, emission = diagnostics.color_patch_channels((1.0, 0.25, 0.0), 'diffuse')
        self.assertEqual(diffuse, (1.0, 0.25, 0.0))
        self.assertEqual(emission, (0.0, 0.0, 0.0))

        diffuse, emission = diagnostics.color_patch_channels(
            (1.0, 0.25, 0.0), 'emissive', emission_scale=0.5)
        self.assertEqual(diffuse, (0.0, 0.0, 0.0))
        self.assertEqual(emission, (0.5, 0.125, 0.0))

    def test_buffer_comparison_detects_one_changed_channel_and_shape_mismatch(self):
        diagnostics = self._diagnostics()
        source = np.zeros((2, 2, 4), dtype=np.uint8)

        self.assertTrue(diagnostics.compare_rgb_buffers(source, source.copy())['exact_equal'])
        changed = source.copy()
        changed[1, 1, 2] = 9
        difference = diagnostics.compare_rgb_buffers(source, changed)
        self.assertFalse(difference['exact_equal'])
        self.assertEqual(difference['max_absolute_difference'], 9.0)
        self.assertFalse(diagnostics.compare_rgb_buffers(source, source[:, :1])['same_shape'])

    def test_world_transform_is_converted_to_parent_local_before_authoring(self):
        diagnostics = self._diagnostics()
        self.assertTrue(hasattr(diagnostics, 'local_transform_from_world'),
                        'world-to-parent transform conversion has not been implemented')
        parent_world = np.eye(4)
        parent_world[3, :3] = [3.0, -2.0, 5.0]
        wanted_world = np.eye(4)
        wanted_world[3, :3] = [10.0, 4.0, 6.0]

        local = diagnostics.local_transform_from_world(wanted_world, parent_world)

        self.assertTrue(np.allclose(local @ parent_world, wanted_world))
        self.assertTrue(np.allclose(local[3, :3], [7.0, 6.0, 1.0]))

    def test_world_bbox_projection_marks_centered_geometry_inside_camera_frustum(self):
        diagnostics = self._diagnostics()
        self.assertTrue(hasattr(diagnostics, 'project_world_bbox_to_image'),
                        'world-bounds camera-frustum probe has not been implemented')
        result = diagnostics.project_world_bbox_to_image(
            bbox_min=(-0.5, -0.5, -2.0), bbox_max=(0.5, 0.5, -2.0),
            camera_world=np.eye(4), focal_length_mm=18.0,
            horizontal_aperture_mm=36.0, vertical_aperture_mm=28.8,
            width=640, height=512,
        )

        self.assertTrue(result['in_camera_frustum'])
        self.assertEqual(result['projected_bounds_xyxy'], [240, 176, 400, 336])

    def test_world_bbox_projection_marks_geometry_behind_camera_outside_frustum(self):
        diagnostics = self._diagnostics()
        result = diagnostics.project_world_bbox_to_image(
            bbox_min=(-0.5, -0.5, 2.0), bbox_max=(0.5, 0.5, 2.0),
            camera_world=np.eye(4), focal_length_mm=18.0,
            horizontal_aperture_mm=36.0, vertical_aperture_mm=28.8,
            width=640, height=512,
        )

        self.assertFalse(result['in_camera_frustum'])
        self.assertIsNone(result['projected_bounds_xyxy'])

    def test_uv_signature_match_uses_face_corner_count_and_buffer_count(self):
        diagnostics = self._diagnostics()
        self.assertTrue(hasattr(diagnostics, 'uv_signature_matches'),
                        'UV warning-to-source signature matching has not been implemented')

        self.assertTrue(diagnostics.uv_signature_matches(
            expected_corners=77241, values_count=4080, indices_count=None,
            warning_expected=77241, warning_buffer=4080,
        ))

    def test_world_prim_range_includes_instance_proxies(self):
        diagnostics = self._diagnostics()
        self.assertTrue(hasattr(diagnostics, 'world_prim_range'),
                        'USD scene traversal helper has not been implemented')

        class RootPrim:
            @staticmethod
            def IsValid():
                return True

        root = RootPrim()
        proxy_predicate = object()
        captured = {}

        class Stage:
            def GetPrimAtPath(self, path):
                captured['root_path'] = path
                return root

        class Usd:
            @staticmethod
            def TraverseInstanceProxies():
                return proxy_predicate

            @staticmethod
            def PrimRange(start, predicate):
                captured['start'] = start
                captured['predicate'] = predicate
                return ['world prim', 'instance proxy']

        result = diagnostics.world_prim_range(Stage(), Usd)

        self.assertEqual(result, ['world prim', 'instance proxy'])
        self.assertEqual(captured, {
            'root_path': '/World',
            'start': root,
            'predicate': proxy_predicate,
        })
        self.assertFalse(diagnostics.uv_signature_matches(
            expected_corners=77241, values_count=2040, indices_count=None,
            warning_expected=77241, warning_buffer=4080,
        ))

    def test_uv_source_inspection_requires_complete_mapping_for_every_warning_group(self):
        diagnostics = self._diagnostics()
        self.assertTrue(hasattr(diagnostics, 'uv_warning_sources_inspected'),
                        'UV source completeness has not been implemented')

        matched = [{
            'signature_candidates': [
                {'mesh_name_matches_warning': True, 'all_warning_uv_primvars_matched': True},
            ],
        }]
        unmatched = [{
            'signature_candidates': [
                {'mesh_name_matches_warning': True, 'all_warning_uv_primvars_matched': False},
            ],
        }]

        self.assertTrue(diagnostics.uv_warning_sources_inspected(matched))
        self.assertFalse(diagnostics.uv_warning_sources_inspected(matched + unmatched))


if __name__ == '__main__':
    unittest.main()
