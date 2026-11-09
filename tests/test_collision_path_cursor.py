import unittest
import numpy as np


class CollisionPathCursorTests(unittest.TestCase):
    def cursor(self):
        from scripts.collision_scene import CollisionScene
        from scripts.collision_path_cursor import CollisionPathCursor
        triangle = np.array([[[0., -5., -5.], [0., 5., -5.], [0., 0., 5.]]])
        scene = CollisionScene(triangle, ['/wall'], np.array([[-9., -9., -9.], [9., 9., 9.]]),
                               np.array([2., 0., 0.]), .1, {'ready': True, 'issues': []})
        return scene, CollisionPathCursor(scene)

    def test_safe_prefix_is_extended_without_resetting_original_certificate(self):
        scene, cursor = self.cursor()
        self.assertEqual(cursor.extend([[2, 0, 0], [2, 1, 0]])['status'], 'CLEAR')
        result = cursor.extend([[2, 1, 0], [2, 2, 0]])
        self.assertEqual(result['status'], 'CLEAR')
        self.assertEqual(result['certified_segments_total'], 2)
        self.assertEqual(scene.start.tolist(), [2., 0., 0.])

    def test_contact_or_discontinuous_extension_never_advances_cursor(self):
        _, cursor = self.cursor()
        hit = cursor.extend([[2, 0, 0], [-2, 0, 0]])
        self.assertEqual(hit['status'], 'CONTACT')
        self.assertEqual(hit['object_prim_path'], '/wall')
        self.assertEqual(cursor.extend([[3, 0, 0], [4, 0, 0]])['status'], 'START_MISMATCH')
        self.assertEqual(cursor.extend([[2, 0, 0], [3, 0, 0]])['status'], 'CLEAR')


if __name__ == '__main__':
    unittest.main()
