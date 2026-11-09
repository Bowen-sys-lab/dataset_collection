import unittest
import numpy as np
from pathlib import Path
import tempfile
from scene_intake import forest_sync_probe as probe


class ForestSyncProbeTests(unittest.TestCase):
    def test_probe_positions_avoid_terrain_occlusion_and_remain_distinct(self):
        self.assertTrue(hasattr(probe,'probe_marker_positions'))
        xy=np.asarray(probe.probe_marker_positions(8,640,512))
        self.assertTrue((xy[:,1]<=225).all())
        distances=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=-1)+np.eye(8)*999
        self.assertGreater(float(distances.min()),40.)

    def test_probe_reads_utf8_scene_config_on_windows(self):
        self.assertTrue(hasattr(probe,'probe_thermal_config'))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'thermal.yaml'
            p.write_text('# 森林火焰\ntargets: {}\nbindings: []\n',encoding='utf-8')
            result=probe.probe_thermal_config(p)
            self.assertEqual(result['targets']['sync_hot']['temperature_c'],80.)
            self.assertEqual(result['bindings'][0]['prim_path'],'/World/ForestSyncMarker')

    def test_marker_must_match_independently_in_color_geometry_and_temperature(self):
        self.assertTrue(hasattr(probe,'evaluate_marker'))
        mask=np.zeros((20,20),bool);mask[8:12,8:12]=True
        rgb=np.zeros((20,20,3),np.uint8);rgb[mask]=[0,255,0]
        truth=np.where(mask,353.15,290.)
        apparent=np.where(mask,350.,290.)
        self.assertTrue(probe.evaluate_marker(rgb,apparent,truth,[9.5,9.5],[9.5,9.5])['pass'])
        self.assertFalse(probe.evaluate_marker(rgb,np.roll(apparent,6,axis=1),truth,[9.5,9.5],[9.5,9.5])['pass'])
        self.assertFalse(probe.evaluate_marker(rgb,apparent,truth,[2.,2.],[2.,2.])['pass'])

if __name__=='__main__':unittest.main()
