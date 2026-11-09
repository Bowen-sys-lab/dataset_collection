import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FactoryClosedLoopEntryTests(unittest.TestCase):
    def test_missing_config_saves_failure_without_loading_simulator(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as temp:
            output = Path(temp)/'new'
            result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/run_factory_closed_loop.py'),
                                     '--config', str(Path(temp)/'missing.yaml'), '--out', str(output)],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((output/'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['termination']['reason'], 'system_error')
            self.assertTrue((output/'failure.txt').is_file())

    def test_old_evidence_directory_is_not_modified(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'validation') as temp:
            path = Path(temp)/'report.json'
            path.write_text('old evidence')
            result = subprocess.run([sys.executable, '-B', str(ROOT/'scripts/run_factory_closed_loop.py'),
                                     '--out', temp], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(path.read_text(), 'old evidence')


if __name__ == '__main__':
    unittest.main()
