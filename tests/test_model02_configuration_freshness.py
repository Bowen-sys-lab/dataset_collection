import hashlib
from pathlib import Path
import tempfile
import unittest

from scripts.model02_configuration_freshness import verify_inputs


class FreshnessTests(unittest.TestCase):
    def test_changed_or_missing_input_and_added_material_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root=Path(folder)
            (root/'materials').mkdir()
            files={'source.blend':b'model','surfaces.json':b'inventory',
                   'materials/steel.yaml':b'material','spectrum.csv':b'optical data'}
            for path,content in files.items():(root/path).write_bytes(content)
            data={'input_sha256':{p:hashlib.sha256(v).hexdigest() for p,v in files.items()},
                  'material_directories':['materials']}
            verify_inputs(root,data)
            for path,content in files.items():
                (root/path).write_bytes(content+b'changed')
                with self.assertRaisesRegex(ValueError,'changed'):verify_inputs(root,data)
                (root/path).write_bytes(content)
            (root/'materials/new.yaml').write_text('added')
            with self.assertRaisesRegex(ValueError,'membership'):verify_inputs(root,data)
            (root/'materials/new.yaml').unlink()
            (root/'source.blend').unlink()
            with self.assertRaisesRegex(ValueError,'missing'):verify_inputs(root,data)


if __name__=='__main__':unittest.main()
