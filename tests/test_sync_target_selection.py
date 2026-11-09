import unittest
from scripts import sync_validation_support as module

class SyncTargetSelectionTests(unittest.TestCase):
    def test_configured_factory_target_replaces_legacy_school_spot(self):
        self.assertTrue(hasattr(module,'diagnostic_target_path'))
        self.assertEqual(module.diagnostic_target_path({'target_prim_path':'/World/Factory/Hotspot'}),'/World/Factory/Hotspot')
        self.assertEqual(module.diagnostic_target_path(None),'/World/Target_01')

if __name__=='__main__': unittest.main()
