"""A separate project copy must use its own task file in CheckOnly mode."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class CityLaunchPortabilityTests(unittest.TestCase):
    def test_check_only_uses_task_inside_project_copy(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root / 'scripts'
            configs = root / 'configs'
            scripts.mkdir()
            configs.mkdir()
            for name in ('record_city_v2_task.ps1', 'collection_capture_defaults.py'):
                shutil.copy2(repo / 'scripts' / name, scripts / name)
            (configs / 'task.yaml').write_text(
                'start_position: [1, 2, 3]\nenvironment_evidence:\n  initial_yaw_degrees: 90\ninstruction: isolated-task\n',
                encoding='utf-8',
            )
            entry = dict(task_id='isolated', file=r'D:\IsaacSim_Demo\configs\task.yaml', episode='C01',
                         target='target', task_version='v1', direction='forward')
            (configs / 'city_200m_v2_task_table.json').write_text(
                json.dumps({'tasks': [entry]}), encoding='utf-8',
            )
            env = os.environ.copy()
            env['PATH'] = str(repo / 'handoff/receiver/bin') + os.pathsep + env['PATH']
            result = subprocess.run(
                ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                 str(scripts / 'record_city_v2_task.ps1'), '-Task', 'isolated', '-CheckOnly'],
                cwd=repo, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('isolated-task', result.stdout)
            self.assertIn(str(root.resolve()), result.stdout)


if __name__ == '__main__':
    unittest.main()
