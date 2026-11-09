import os
import unittest
from unittest.mock import patch
from scripts.collection_shell import powershell_executable


class ShellTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows entrypoint')
    def test_shell_is_resolved_when_isaac_path_has_no_powershell_directory(self):
        with patch.dict(os.environ,{'PATH':'D:/isaacsim/kit'}):
            path=powershell_executable()
            self.assertTrue(path.is_file())
            self.assertTrue(path.is_absolute())
            self.assertEqual(path.name.lower(),'powershell.exe')


if __name__=='__main__':unittest.main()
