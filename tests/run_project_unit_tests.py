"""Run project unit tests under python.bat, without starting Kit."""
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
# python.bat supplies NumPy/YAML but USD normally enters sys.path via Kit startup.
sys.path.insert(0, 'D:/isaacsim/extscache/omni.usd.libs-1.0.3+00c488ae.wx64.r.cp312')
suite = unittest.defaultTestLoader.discover(str(root/'tests'), pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
