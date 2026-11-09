import unittest
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from pxr import Usd

from scripts.thermal_snapshot import anonymous_session_layer
from isaac_thermal_baseline import conditions_for_run, finalize_integrity, noise_with_seed


class ThermalBaselineHelperTests(unittest.TestCase):
    def test_material_cache_conditions_do_not_profile_throughput(self):
        self.assertEqual(conditions_for_run('material_cache_off'), ['material_cache_off'])
        self.assertEqual(conditions_for_run('material_cache_on'), ['material_cache_on'])

    def test_source_mutation_cannot_report_pass(self):
        report = {'result': 'PASS'}
        finalize_integrity(report, source_unchanged=False, vendor_unchanged=True)
        self.assertEqual(report['result'], 'FAIL')
        self.assertFalse(report['source_unchanged'])

    def test_noise_seed_change_rebuilds_fixed_pattern_and_stream_seed(self):
        import numpy as np
        from irsim.noise.stage import NoiseStage
        from irsim.noise.three_d import FixedPattern
        source = NoiseStage(0, (2, 3), (1., .1, .2, .1, .2, .1, 0.),
                            FixedPattern(np.zeros(2, np.float32), np.zeros(3, np.float32),
                                         np.zeros((2, 3), np.float32)))
        changed = noise_with_seed(source, 5)
        self.assertEqual(changed.sensor_seed, 5)
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
            (source.unit_fixed.v, source.unit_fixed.h, source.unit_fixed.vh),
            (changed.unit_fixed.v, changed.unit_fixed.h, changed.unit_fixed.vh))))

    def test_disabled_aov_run_cannot_enter_snapshots_in_same_process(self):
        script = Path(__file__).with_name('isaac_thermal_baseline.py')
        with TemporaryDirectory() as folder:
            output = Path(folder)/'should-not-start'
            run = subprocess.run([sys.executable, '-B', str(script), '--condition',
                                  'rgb_only_aovs_disabled', '--out', str(output)],
                                 capture_output=True)
            self.assertEqual(run.returncode, 2)
            self.assertFalse(output.exists())

    def test_each_process_runs_only_one_render_product_state(self):
        self.assertEqual(conditions_for_run('full_thermal'),
                         ['full_thermal', 'full_thermal_profiled'])
        self.assertEqual(conditions_for_run('cpu_bypass_aovs_retained'),
                         ['cpu_bypass_aovs_retained'])
        self.assertEqual(conditions_for_run('rgb_only_aovs_disabled'),
                         ['rgb_only_aovs_disabled'])

    def test_diagnostic_edits_are_removed_without_touching_root_layer(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim('/World')
        root = stage.GetRootLayer()
        before = root.ExportToString()
        original_target = stage.GetEditTarget()

        with anonymous_session_layer(stage) as layer:
            self.assertTrue(layer.anonymous)
            stage.DefinePrim('/World/DiagnosticOnly')
            self.assertEqual(before, stage.GetRootLayer().ExportToString())
            self.assertIn('DiagnosticOnly', layer.ExportToString())

        self.assertEqual(before, root.ExportToString())
        self.assertEqual(original_target.GetLayer(), stage.GetEditTarget().GetLayer())
        self.assertFalse(stage.GetPrimAtPath('/World/DiagnosticOnly'))


if __name__ == '__main__':
    unittest.main()
