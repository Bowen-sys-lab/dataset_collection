import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.install_city_assets import MANIFEST, PACKAGE, ROOT, main


class CityAssetPackageTests(unittest.TestCase):
    def test_manifest_is_city_only_and_archive_is_below_github_file_limit(self):
        data = json.loads(MANIFEST.read_text(encoding='utf8'))
        self.assertEqual(data['scene'], 'city_200m_v3')
        self.assertLess(PACKAGE.stat().st_size, 100 * 1024 * 1024)
        self.assertTrue(all(not row['path'].startswith(('factory/', 'forest_fire_')) for row in data['files']))

    def test_archive_members_stay_inside_project(self):
        with zipfile.ZipFile(PACKAGE) as archive:
            for item in archive.infolist():
                target = (ROOT/Path(item.filename)).resolve()
                self.assertTrue(target.is_relative_to(ROOT))
                self.assertNotIn('..', Path(item.filename).parts)

    def test_check_only_verifies_installed_files(self):
        old = __import__('sys').argv
        try:
            __import__('sys').argv = ['install_city_assets.py', '--check-only']
            main()
        finally:
            __import__('sys').argv = old


if __name__ == '__main__':
    unittest.main()
