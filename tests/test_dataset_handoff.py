import importlib
import json
from pathlib import Path
import tempfile
import unittest


class HandoffTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.dataset_handoff'),
                             'Missing the distributable handoff helper')
        return importlib.import_module('scripts.dataset_handoff')

    def test_manifest_detects_modified_and_missing_files(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'a.txt').write_text('original')
            (root/'b.txt').write_text('keep')
            manifest = api.file_manifest(root, [root/'a.txt', root/'b.txt'])
            self.assertEqual(api.verify_manifest(root, manifest), [])
            (root/'a.txt').write_text('changed')
            (root/'b.txt').unlink()
            self.assertEqual({x['path'] for x in api.verify_manifest(root, manifest)},
                             {'a.txt', 'b.txt'})

    def test_review_scaffold_preserves_sources_and_does_not_claim_success(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run = root/'run'; batch = root/'batch'
            run.mkdir(); (batch/'dataset').mkdir(parents=True)
            raw = b'{"task_result":{"status":"unjudged"}}'
            (run/'run_metadata.json').write_bytes(raw)
            (run/'trajectory.jsonl').write_text('{"frame_id":0}\n')
            (batch/'dataset/episode.json').write_text(json.dumps({'run_id':'run', 'instruction':'Find target'}))
            (batch/'dataset/frames.jsonl').write_text(json.dumps({'frame_index':0,'observation_time_s':1.,'rgb':'rgb.png','thermal':'ir.png','original_source_frame_id':0})+'\n')
            out = root/'review'
            api.create_review(run, batch, out)
            review = json.loads((out/'review.json').read_text(encoding='utf-8'))
            self.assertEqual(review['decision'], 'PENDING')
            self.assertEqual(review['task_success'], 'UNKNOWN')
            self.assertEqual((run/'run_metadata.json').read_bytes(), raw)
            self.assertTrue((out/'frame_review.csv').is_file())
            with self.assertRaises(FileExistsError):
                api.create_review(run, batch, out)

    def test_include_requires_independent_review_and_evidence(self):
        api = self.api()
        review = {'decision':'INCLUDE', 'annotator':'A', 'reviewer':'A',
                  'task_success':'UNKNOWN', 'target_correct':'UNKNOWN',
                  'technical_check':'UNKNOWN', 'terminal_check':'UNKNOWN',
                  'path_check':'UNKNOWN', 'evidence':[], 'issues':[]}
        self.assertTrue(api.review_issues(review))
        review.update(reviewer='B',task_success='YES',target_correct='YES',
                      technical_check='PASS',terminal_check='PASS',
                      path_check='PASS',evidence=['frame_000001.json'])
        self.assertEqual(api.review_issues(review), [])


if __name__ == '__main__':
    unittest.main()
