import math
import unittest


class TinyFitCheckTests(unittest.TestCase):
    def rows(self,offset=0.):
        return [{'kind':k,'model_loss':loss+offset,'supervised_tokens':n} for k,loss,n in
                [('MOVE',4.,60),('HOLD',3.,60),('STOP',8.,3),('MOVE',2.,60),('HOLD',5.,60),('STOP',6.,3)]]

    def test_loss_weighted_by_supervised_tokens_not_prompt_or_batch_count(self):
        from scripts.bc_tinyfit_checks import summarize_losses
        result=summarize_losses(self.rows())
        self.assertAlmostEqual(result['token_weighted_loss'],(4*60+3*60+8*3+2*60+5*60+6*3)/246)
        self.assertAlmostEqual(result['by_kind']['STOP'],7.)
        self.assertEqual(result['supervised_tokens'],246)

    def test_nonfinite_or_missing_kind_or_invalid_counts_rejected(self):
        from scripts.bc_tinyfit_checks import summarize_losses
        for change in ['nan','negative','zero','bool','missing']:
            rows=self.rows()
            if change=='nan':rows[0]['model_loss']=math.nan
            if change=='negative':rows[0]['model_loss']=-1.
            if change=='zero':rows[0]['supervised_tokens']=0
            if change=='bool':rows[0]['supervised_tokens']=True
            if change=='missing':rows=[r for r in rows if r['kind']!='STOP']
            with self.subTest(change=change),self.assertRaises(ValueError):summarize_losses(rows)

    def test_predeclared_per_kind_improvement_and_exact_step_count(self):
        from scripts.bc_tinyfit_checks import assess_fit
        self.assertTrue(assess_fit(self.rows(),self.rows(-.2),steps=18)['passed'])
        self.assertFalse(assess_fit(self.rows(),self.rows(.1),steps=18)['passed'])
        bad=self.rows(-.2)
        for row in bad:
            if row['kind']=='STOP':row['model_loss']+=.3
        self.assertFalse(assess_fit(self.rows(),bad,steps=18)['passed'])
        for steps in [0,17,19,True]:
            with self.assertRaises(ValueError):assess_fit(self.rows(),self.rows(-.2),steps=steps)


if __name__=='__main__':unittest.main()
