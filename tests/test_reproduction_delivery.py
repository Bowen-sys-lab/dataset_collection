import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.build_reproduction_delivery import add_file, verify_archive, allowed_file


class DeliveryTests(unittest.TestCase):
    def test_missing_and_outside_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                add_file(root, {}, root / 'missing.py')
            with self.assertRaises(ValueError):
                add_file(root, {}, Path(__file__))

    def test_excludes_caches_and_machine_shortcuts(self):
        for path in ['configs/a.lnk', 'scripts/__pycache__/a.pyc', '.git/config',
                     'tools/.venv/key', 'scripts/old.log', 'configs/old.yaml.bak']:
            self.assertFalse(allowed_file(Path(path)), path)
        self.assertTrue(allowed_file(Path('configs/task.yaml')))

    def test_archive_verification_rejects_tampering_and_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('IsaacSim_Demo/data.txt', b'changed')
            with self.assertRaises(ValueError):
                verify_archive(archive, [{'path': 'data.txt', 'size': 1, 'sha256': '0'*64}])
            with zipfile.ZipFile(archive, 'w') as output:
                output.writestr('../escape', b'bad')
            with self.assertRaises(ValueError):
                verify_archive(archive, [])


if __name__ == '__main__':
    unittest.main()
