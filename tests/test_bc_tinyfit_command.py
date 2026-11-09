import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class TinyFitCommandTests(unittest.TestCase):
    def test_gpu_busy_refuses_before_any_model_or_data_load(self):
        from scripts import verify_bc_tiny_fit as command
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out'
            with patch.object(command,'gpu_processes',return_value=['12,user.exe']),patch('sys.argv',
                ['probe','--native-report','missing','--cohort-report','missing','--out',str(out)]):
                self.assertEqual(command.main(),2)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'BLOCKED_GPU_BUSY')
            self.assertEqual(report['optimizer_steps'],0);self.assertEqual(report['model_load_count'],0)

    def test_existing_outputs_not_overwritten(self):
        from scripts import verify_bc_tiny_fit as command
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out';out.mkdir();(out/'report.json').write_text('HISTORY')
            with patch('sys.argv',['probe','--native-report','missing','--cohort-report','missing','--out',str(out)]),self.assertRaises(FileExistsError):
                command.main()
            self.assertEqual((out/'report.json').read_text(),'HISTORY')


if __name__=='__main__':unittest.main()
