import unittest
from types import SimpleNamespace
from scripts.thermal_diagnostics import StageTimings, summarize_segments


class StageTimingTests(unittest.TestCase):
    def test_profile_hooks_time_mapping_after_runtime_override_and_restore(self):
        from contextlib import contextmanager
        import scripts.thermal_diagnostics as diagnostics
        helper = getattr(diagnostics, 'profile_material_mapping_scope', None)
        self.assertTrue(callable(helper), 'inside-scope mapping profiler must exist')
        tick = [0.]
        module = SimpleNamespace(material_id_plane=lambda: 'vendor')
        def cached():
            tick[0] += 2
            return 'cached'
        @contextmanager
        def mapping_scope(*args, **kwargs):
            original = module.material_id_plane
            module.material_id_plane = cached
            try:
                yield
            finally:
                module.material_id_plane = original
        runtime = SimpleNamespace(cached_material_mapping=mapping_scope)
        original = module.material_id_plane
        timer = StageTimings(clock=lambda: tick[0], thread_clock=lambda: tick[0])
        with helper(timer, runtime=runtime, modules=[module]):
            with runtime.cached_material_mapping(module):
                self.assertEqual(module.material_id_plane(), 'cached')
        self.assertIs(runtime.cached_material_mapping, mapping_scope)
        self.assertIs(module.material_id_plane, original)
        self.assertEqual(timer.values['mapping_lookup']['calls'], 1)
        self.assertEqual(timer.values['mapping_lookup']['exclusive_wall_s'], 2)

    def test_call_counts_are_integers_not_time_percentiles(self):
        report = summarize_segments([{'read': {'calls': 2, 'inclusive_wall_s': .1}},
                                     {}, {'read': {'calls': 3, 'inclusive_wall_s': .2}}])
        self.assertEqual(report['read']['calls'], {'total': 5, 'per_frame': [2, 0, 3]})
        self.assertEqual(report['read']['inclusive_wall_s']['samples'], 3)

    def test_nested_timings_do_not_double_count_and_restore_on_exception(self):
        tick = [0.]
        def clock():
            return tick[0]
        def child():
            tick[0] += 2.
        module = SimpleNamespace(child=child)
        def parent():
            tick[0] += 1.
            module.child()
            tick[0] += 3.
            raise ValueError('intentional')
        module.parent = parent
        timings = StageTimings(clock=clock, thread_clock=clock)
        with self.assertRaises(ValueError):
            with timings.wrap([(module, 'parent', 'outer'), (module, 'child', 'inner')]):
                module.parent()
        self.assertIs(module.parent, parent)
        self.assertIs(module.child, child)
        self.assertEqual(timings.values['outer']['exclusive_wall_s'], 4.)
        self.assertEqual(timings.values['outer']['inclusive_wall_s'], 6.)
        self.assertEqual(timings.values['inner']['exclusive_wall_s'], 2.)
