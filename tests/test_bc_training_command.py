import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class BCTrainingCommandTests(unittest.TestCase):
    def test_busy_gpu_is_recorded_before_any_model_or_dataset_load(self):
        from scripts import verify_bc_training_interface as command
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'exclusive'
            with patch.object(command,'gpu_processes',return_value=['123,user-replay.exe']), \
                 patch('sys.argv',['probe','--batch',str(Path(temp)/'absent-data'),'--out',str(out),'--backward']):
                self.assertEqual(command.main(),2)
            report=json.loads((out/'report.json').read_text())
            self.assertEqual(report['status'],'BLOCKED_GPU_BUSY')
            self.assertEqual(report['model_load_count'],0)
            self.assertEqual(report['optimizer_steps'],0)
            self.assertFalse(report['training_ready'])

    def test_existing_evidence_directory_cannot_be_overwritten(self):
        from scripts import verify_bc_training_interface as command
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'existing';out.mkdir();(out/'report.json').write_text('HISTORY')
            with patch('sys.argv',['probe','--batch',temp,'--out',str(out)]),self.assertRaises(FileExistsError):
                command.main()
            self.assertEqual((out/'report.json').read_text(),'HISTORY')


if __name__=='__main__':unittest.main()
