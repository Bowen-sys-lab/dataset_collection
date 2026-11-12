import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.install_city_assets import MANIFEST, ROOT, main


class CityAssetPackageTests(unittest.TestCase):
    def test_manifest_is_city_only_and_private_archive_is_below_github_file_limit(self):
        data = json.loads(MANIFEST.read_text(encoding='utf8'))
        self.assertEqual(data['scene'], 'city_200m_v3')
        self.assertLess(data['bytes'], 100 * 1024 * 1024)
        self.assertTrue(all(not row['path'].startswith(('factory/', 'forest_fire_')) for row in data['files']))

    def test_archive_members_stay_inside_project(self):
        with tempfile.TemporaryDirectory() as folder:
            archive_path = Path(folder) / 'fixture.zip'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('city/example.txt', 'ok')
                archive.writestr('../escape.txt', 'blocked')
            with zipfile.ZipFile(archive_path) as archive:
                members = archive.infolist()
                self.assertTrue((ROOT / Path(members[0].filename)).resolve().is_relative_to(ROOT))
                self.assertIn('..', Path(members[1].filename).parts)

    def test_missing_private_archive_is_reported(self):
        old = __import__('sys').argv
        try:
            __import__('sys').argv = ['install_city_assets.py', '--archive', str(ROOT / 'missing-city.zip')]
            with self.assertRaises(FileNotFoundError):
                main()
        finally:
            __import__('sys').argv = old

    def test_manifest_paths_are_relative(self):
        data = json.loads(MANIFEST.read_text(encoding='utf8'))
        for row in data['files']:
            self.assertTrue((ROOT / Path(row['path'])).resolve().is_relative_to(ROOT))
            self.assertNotIn('..', Path(row['path']).parts)

    def test_old_archive_is_not_required(self):
        self.assertFalse((ROOT / 'resources/city/city-assets-20261112.zip').exists())


if __name__ == '__main__':
    unittest.main()
