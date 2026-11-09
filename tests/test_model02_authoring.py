import copy
import unittest
from scripts.model02_authoring import validate_authoring,resolve_temperature


class AuthoringTests(unittest.TestCase):
    def config(self):
        return {'schema_version':1,'phase':'prebinding',
            'condition':{'air_temperature_c':20.,'indoor_temperature_c':24.,'wind_m_s':1.,'hours_after_sunset':2.,'weather':'dry','equipment_state':'operating'},
            'targets':{'a':{'mode':'fixed','temperature_c':30.},'b':{'mode':'fixed','temperature_c':31.}},
            'surfaces':{'A':{'object':'objA','target':'a','material':'steel'},'B':{'object':'objB','target':'b','material':'steel'}}}

    def test_independent_edit_does_not_merge_same_material_regions(self):
        before=self.config();after=copy.deepcopy(before)
        after['targets']['a']['temperature_c']=60.
        validate_authoring(after,{'A','B'},{'steel'})
        self.assertEqual(after['targets']['b'],before['targets']['b'])
        self.assertEqual(after['surfaces'],before['surfaces'])

    def test_missing_region_or_reused_target_is_rejected(self):
        data=self.config();del data['surfaces']['B']
        with self.assertRaises(ValueError):validate_authoring(data,{'A','B'},{'steel'})
        data=self.config();data['surfaces']['B']['target']='a'
        with self.assertRaises(ValueError):validate_authoring(data,{'A','B'},{'steel'})

    def test_bad_material_and_temperature_fail_before_apply(self):
        data=self.config();data['surfaces']['A']['material']='unknown'
        with self.assertRaises(ValueError):validate_authoring(data,{'A','B'},{'steel'})
        for invalid in [float('nan'),-273.15,True]:
            data=self.config();data['targets']['a']['temperature_c']=invalid
            with self.assertRaises(ValueError):validate_authoring(data,{'A','B'},{'steel'})

    def test_existing_target_modes_use_explicit_current_air(self):
        self.assertEqual(resolve_temperature({'mode':'ambient'},20.),20.)
        self.assertEqual(resolve_temperature({'mode':'offset','offset_c':3.},20.),23.)
        self.assertEqual(resolve_temperature({'mode':'fixed','temperature_c':30.},20.),30.)

if __name__=='__main__':unittest.main()
