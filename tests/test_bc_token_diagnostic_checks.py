import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'Requires independent model environment')
class TokenDiagnosisTests(unittest.TestCase):
    def test_action_scores_account_for_patches_after_bos_and_causal_shift(self):
        from scripts.bc_token_diagnostic_checks import score_action_tokens
        # Two prompt tokens, then target7 and EOS2; two visual patches after BOS.
        logits=torch.zeros(1,6,10);logits[0,3,7]=4.;logits[0,4,2]=5.
        labels=torch.tensor([[-100,-100,7,2]])
        rows=score_action_tokens(logits,labels,patch_tokens=2)
        self.assertEqual([r['logit_index'] for r in rows],[3,4])
        self.assertEqual([r['target_id'] for r in rows],[7,2])
        self.assertEqual([r['predicted_id'] for r in rows],[7,2])
        self.assertTrue(all(r['correct'] and r['target_rank']==1 for r in rows))

    def test_action_scores_exclude_padding_and_measure_wrong_token_rank(self):
        from scripts.bc_token_diagnostic_checks import score_action_tokens
        logits=torch.zeros(1,7,10);logits[0,3,8]=4.;logits[0,3,7]=2.;logits[0,4,2]=5.
        rows=score_action_tokens(logits,torch.tensor([[-100,-100,7,2,-100]]),patch_tokens=2)
        self.assertEqual(len(rows),2);self.assertFalse(rows[0]['correct'])
        self.assertEqual(rows[0]['target_rank'],2);self.assertEqual(rows[0]['predicted_id'],8)
        self.assertLess(rows[0]['target_probability'],rows[0]['predicted_probability'])

    def test_unprovable_shapes_and_nonfinite_values_are_rejected(self):
        from scripts.bc_token_diagnostic_checks import score_action_tokens
        labels=torch.tensor([[-100,-100,7,2]])
        for logits,patches in [(torch.zeros(1,5,10),2),(torch.zeros(1,6,10),True),
                               (torch.full((1,6,10),float('nan')),2)]:
            with self.assertRaises(ValueError):score_action_tokens(logits,labels,patch_tokens=patches)


if __name__=='__main__':unittest.main()
