import unittest
from types import SimpleNamespace
import numpy as np
from scene_intake import forest_collection as forest


class ForestCollectionTests(unittest.TestCase):
    def test_legacy_camera_truth_is_copied_without_new_vendor_api(self):
        self.assertTrue(hasattr(forest,'capture_truth'),'legacy camera truth adapter is missing')
        t=np.array([[923.15]],dtype=np.float32)
        camera=SimpleNamespace(last_frame=SimpleNamespace(planes={'temperature_k':t},labels={1:'/World/Flame'}))
        planes,legends=forest.capture_truth(camera)
        t[0,0]=300.
        self.assertAlmostEqual(float(planes['temperature_k'][0,0]),923.15,places=3)
        self.assertEqual(legends['instance_id'],{1:'/World/Flame'})
    def test_loop_time_follows_scene_fps_and_wraps(self):
        self.assertEqual(forest.animation_time_code(1.,24.,0.,96.),24.)
        self.assertEqual(forest.animation_time_code(4.5,24.,0.,96.),12.)
        self.assertEqual(forest.animation_time_code(4.,24.,0.,96.),0.)
        with self.assertRaises(ValueError):forest.animation_time_code(float('nan'),24.,0.,96.)

    def test_native_and_readable_profiles_do_not_change_scene_or_temperature(self):
        a=forest.capture_profile('baseline');b=forest.capture_profile('native-readable')
        self.assertEqual(a['thermal_resolution'],[256,256])
        self.assertEqual(b['thermal_resolution'],[256,256])
        self.assertEqual(a['rgb_resolution'],[256,256])
        self.assertEqual(b['rgb_resolution'],[256,256])
        self.assertEqual(b['rgb_ev'],2.)
        self.assertEqual(a['scene_config'],b['scene_config'])
        self.assertEqual(a['sensor_range'],b['sensor_range'])

    def test_two_target_episode_replay_uses_the_factory_dimensions(self):
        from scripts.replay_factory_inspection import RGB_SIZE, THERMAL_SIZE

        task = {'environment_evidence': {'collection_episode': {
            'scene': 'forest', 'start_version': 'forest_two_target_v1',
        }}}
        profile = forest.capture_profile('native-readable', recorded_task=task)
        self.assertEqual(tuple(profile['rgb_resolution']), RGB_SIZE)
        self.assertEqual(tuple(profile['thermal_resolution']), THERMAL_SIZE)
        self.assertEqual(forest.capture_profile('native-readable')['rgb_resolution'], [256, 256])
        self.assertEqual(forest.capture_profile('native-readable')['thermal_resolution'], [256, 256])


if __name__=='__main__':unittest.main()
