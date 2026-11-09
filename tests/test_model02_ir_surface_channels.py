import copy,importlib.util,unittest
import numpy as np


class SurfaceChannelTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.validate_model02_ir_surface_channels'))
        from scripts import validate_model02_ir_surface_channels
        return validate_model02_ir_surface_channels

    def fixture(self):
        return {'experiment':'normal_map','camera':'c','geometry':'g','ir_material':'m','environment':'e','clock':0.,'sensor_state':'s',
            'normal':np.ones((2,2,3)),'distance':np.ones((2,2)),
            'temperature':np.ones((2,2))*300.,'material_id':np.ones((2,2),dtype=int),
            'radiance':np.ones((2,2)),'rgb':np.ones((2,2,3),dtype=np.uint8)}

    def test_normal_comparison_rejects_changed_controls(self):
        m=self.module();before=self.fixture()
        for key in ['camera','geometry','ir_material','environment','sensor_state','temperature']:
            after=copy.deepcopy(before)
            after[key]=after[key]+1 if isinstance(after[key],np.ndarray) else 'changed'
            self.assertEqual(m.compare_controlled_pair(before,after)['result'],'inconclusive',key)

    def test_normal_observation_is_measured_not_forced(self):
        m=self.module();before=self.fixture();after=copy.deepcopy(before)
        self.assertEqual(m.compare_controlled_pair(before,after)['normal_effect'],'not_observed')
        after['normal'][0,0,0]+=.1
        self.assertEqual(m.compare_controlled_pair(before,after)['normal_effect'],'observed')

    def test_rgb_only_variation_keeps_true_temperature_and_ir_inputs(self):
        m=self.module();before=self.fixture();before['experiment']='rgb_color';after=copy.deepcopy(before)
        after['rgb']+=50
        report=m.compare_controlled_pair(before,after)
        self.assertEqual(report['result'],'controlled');self.assertEqual(report['radiance_max_abs_difference'],0.)
        self.assertEqual(report['temperature_max_abs_difference'],0.)

    def test_same_temperature_different_material_case_is_explicit(self):
        m=self.module();before=self.fixture();before['experiment']='ir_material';after=copy.deepcopy(before)
        after['ir_material']='bare_aluminium';after['material_id']+=1;after['radiance']+=.2
        report=m.compare_controlled_pair(before,after)
        self.assertEqual(report['result'],'controlled');self.assertEqual(report['temperature_max_abs_difference'],0.)
