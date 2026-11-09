import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest


class ModelWorkerProtocolTests(unittest.TestCase):
    def test_request_reply_binds_image_and_observation_and_keeps_each_step(self):
        from scripts.model_worker_protocol import serve_requests, exchange_request, write_json
        from scripts.factory_closed_loop import observation_digest
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root/'rgb.png'
            image.write_bytes(b'first image')
            observation = {'index': 0, 'rgb_path': str(image),
                           'rgb_sha256': hashlib.sha256(image.read_bytes()).hexdigest()}
            seen = []
            def predict(request, destination):
                seen.append(request['observation']['index'])
                return {'raw_output': '98 49 49', 'inference_seconds': 1.}
            thread = threading.Thread(target=serve_requests, args=(root, predict),
                                      kwargs={'max_requests': 2, 'idle_timeout_s': 3})
            thread.start()
            try:
                first = exchange_request(root, observation, 'find tank', timeout_s=2)
                self.assertEqual(first['observation_digest'], observation_digest(observation))
                image.write_bytes(b'tampered')
                second = dict(observation, index=1)
                result = exchange_request(root, second, 'find tank', timeout_s=2)
                self.assertEqual(result['status'], 'FAIL')
                self.assertEqual(seen, [0])
                self.assertTrue((root/'step-000/response.json').exists())
                self.assertTrue((root/'step-001/response.json').exists())
            finally:
                write_json(root/'stop.json', {})
                thread.join(4)
            self.assertFalse(thread.is_alive())

    def test_missing_or_dead_worker_times_out_without_a_motion_default(self):
        from scripts.model_worker_protocol import exchange_request
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(TimeoutError):
                exchange_request(Path(temp), {'index': 0}, 'tank', timeout_s=.05)

    def test_existing_step_is_never_reused(self):
        from scripts.model_worker_protocol import exchange_request
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'step-000').mkdir()
            (root/'step-000/response.json').write_text('old result')
            with self.assertRaises(FileExistsError):
                exchange_request(root, {'index': 0}, 'tank', timeout_s=.05)
            self.assertEqual((root/'step-000/response.json').read_text(), 'old result')


if __name__ == '__main__':
    unittest.main()
