"""The batch review entry point should be usable in a clean installation."""
import subprocess
import sys
import unittest
from pathlib import Path


class CityReviewEntryTests(unittest.TestCase):
    def test_review_command_exposes_help(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, '-m', 'scripts.review_city_batch', '--help'],
            cwd=root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--queue', result.stdout)
        self.assertIn('--manifest', result.stdout)


if __name__ == '__main__':
    unittest.main()
