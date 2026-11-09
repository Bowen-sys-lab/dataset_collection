"""City v2 launchers must survive a non-UTF-8 Windows console."""

import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32' / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'


@unittest.skipUnless(os.name == 'nt' and POWERSHELL.is_file(), 'Windows PowerShell 5.1 required')
class CityV2WindowsEncodingTests(unittest.TestCase):
    def test_c01_check_only_parses_utf8_task_under_gbk_console(self):
        for launcher, args in (
            ('record_city_v2_task.ps1', '-Task city-v2-heatmaincover-s1'),
            ('record.ps1', '-Scene city -Episode C01'),
        ):
            with self.subTest(launcher=launcher):
                command = (
                    '[Console]::OutputEncoding = [Text.Encoding]::GetEncoding(936); '
                    f"& '{ROOT / 'scripts' / launcher}' {args} -CheckOnly; "
                    'if (-not $?) { exit 1 }'
                )
                result = subprocess.run(
                    [str(POWERSHELL), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', command],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    encoding='gbk', errors='replace', check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout)
                self.assertIn('task=city-v2-heatmaincover-s1', result.stdout)
                self.assertRegex(result.stdout, r'run=city-pilot30-C01-[0-9]{8}T[0-9]{12}Z snapshot=')
                self.assertIn('direction=', result.stdout)
                self.assertIn('instruction=The target is several blocks ahead, slightly to your right.', result.stdout)
                self.assertIn('task_version=city20_long_start_trial_v1', result.stdout)
                self.assertIn('raw trajectory=1.0 simulation seconds', result.stdout)
                self.assertIn('scene_config=', result.stdout)
                self.assertIn('city_200m_v3_scene.yaml', result.stdout)
                self.assertIn('thermal_revision=city_material_partition_v3', result.stdout)


if __name__ == '__main__':
    unittest.main()
