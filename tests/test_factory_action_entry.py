import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FactoryActionEntryTests(unittest.TestCase):
    def test_missing_config_saves_failure_before_any_renderer_starts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as temp:
            folder = Path(temp)
            output = folder/'new-output'
            result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/probe_factory_action.py'),
                                     '--config', str(folder/'missing.yaml'), '--out', str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['result'], 'FAIL')
            self.assertTrue((output/'failure.txt').is_file())

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as temp:
            output = Path(temp)
            marker = output/'report.json'
            marker.write_text('existing evidence', encoding='utf-8')
            result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/probe_factory_action.py'),
                                     '--out', str(output)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'existing evidence')


if __name__ == '__main__':
    unittest.main()
