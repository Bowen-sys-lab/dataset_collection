import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.verify_repository import check


class RepositorySetupTests(unittest.TestCase):
    def prepare(self, folder):
        root = Path(folder)
        receiver = root/'handoff/receiver'
        receiver.mkdir(parents=True)
        file = root/'entry.py'
        file.write_bytes(b'code')
        row = {'path':'entry.py','size':4,'sha256':hashlib.sha256(b'code').hexdigest()}
        (receiver/'repository-manifest.json').write_text(json.dumps({'files':[row]}),encoding='utf8')
        return root, receiver

    def test_code_check_does_not_claim_assets_or_dependency_are_installed(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _ = self.prepare(folder)
            result = check(root)
            self.assertEqual(result['result'],'PASS')
            self.assertEqual(result['checked'],[{'kind':'repository','files':1}])
            self.assertEqual(result['not_checked'],['scene_assets','thermal_dependency'])
            self.assertEqual(check(root, assets=True)['result'],'FAIL')

    def test_missing_or_modified_repository_file_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _ = self.prepare(folder)
            (root/'entry.py').write_bytes(b'bad')
            self.assertEqual(check(root)['issues'][0]['reason'],'changed')
            (root/'entry.py').unlink()
            self.assertEqual(check(root)['issues'][0]['reason'],'missing')

    def test_dependency_hashes_are_checked_against_its_own_root(self):
        with tempfile.TemporaryDirectory() as folder:
            root, receiver = self.prepare(folder)
            dep = root/'third_party/thermal'
            dep.mkdir(parents=True)
            (dep/'x').write_bytes(b'x')
            row = {'path':'x','size':1,'sha256':hashlib.sha256(b'x').hexdigest()}
            (receiver/'dependency-lock.json').write_text(json.dumps({'directory':'third_party/thermal','files':[row]}))
            self.assertEqual(check(root,dependency=True)['result'],'PASS')
            (dep/'x').write_bytes(b'y')
            self.assertEqual(check(root,dependency=True)['result'],'FAIL')


if __name__ == '__main__': unittest.main()
