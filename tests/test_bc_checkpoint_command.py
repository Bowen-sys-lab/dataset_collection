import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class CheckpointCommandTests(unittest.TestCase):
    def test_busy_gpu_does_not_load_native_report_or_model(self):
        from scripts import verify_bc_checkpoint_roundtrip as command
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out'
            with patch.object(command,'gpu_processes',return_value=['321,user.exe']), \
                 patch('sys.argv',['probe','--native-report',str(Path(temp)/'absent.json'),'--out',str(out)]):
                self.assertEqual(command.main(),2)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'BLOCKED_GPU_BUSY')
            self.assertEqual(report['optimizer_steps'],0)
            self.assertEqual(report['model_load_count'],0)

    def test_original_evidence_directory_is_never_overwritten(self):
        from scripts import verify_bc_checkpoint_roundtrip as command
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'out';out.mkdir();(out/'report.json').write_text('HISTORY')
            with patch('sys.argv',['probe','--native-report',temp,'--out',str(out)]),self.assertRaises(FileExistsError):
                command.main()
            self.assertEqual((out/'report.json').read_text(),'HISTORY')


if __name__=='__main__':unittest.main()
