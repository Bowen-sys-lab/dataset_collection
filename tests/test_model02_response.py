import unittest
import numpy as np
from scripts.model02_response import check_local_temperature_response


class ResponseTests(unittest.TestCase):
    def test_only_selected_visible_pixels_may_change_temperature(self):
        before=np.full((4,4),293.15);mask=np.zeros((4,4),bool);mask[1,2]=True
        after=before.copy();after[mask]+=10
        result=check_local_temperature_response(before,after,mask,10.)
        self.assertEqual(result['changed_temperature_pixels'],1)
        after[0,0]+=1
        with self.assertRaises(ValueError):check_local_temperature_response(before,after,mask,10.)
        with self.assertRaises(ValueError):check_local_temperature_response(before,before,mask,10.)
        with self.assertRaises(ValueError):check_local_temperature_response(before,before,np.zeros_like(mask),10.)


if __name__=='__main__':unittest.main()
