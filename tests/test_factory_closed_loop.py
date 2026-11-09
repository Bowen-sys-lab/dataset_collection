import copy
import hashlib
from pathlib import Path
import tempfile
import unittest


class FactoryClosedLoopTests(unittest.TestCase):
    def run_case(self, *, outputs=None, guard_status='CLEAR', stale=False,
                 wrong_reply=False, abort=False, max_seconds=20, wrong_time=False,
                 execution_status='CLEAR', abort_during_motion=False, inference_timeout=False):
        from scripts.factory_closed_loop import run_closed_loop, observation_digest
        from scripts.aerovla_action_bridge import plan_aerovla_action
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        state = {'position': [0., 0., 3.], 'yaw': 0., 'time': 10.}
        seen, executed = [], []

        def observe(index):
            path = Path(temp.name)/f'{index}.png'
            path.write_bytes(f'view at {state["position"]}'.encode())
            body = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], state['position']+[1]]
            return {'index': 0 if stale else index, 'simulation_time_s': 10. if wrong_time else state['time'],
                    'position': list(state['position']), 'yaw': state['yaw'],
                    'body_world_matrix': body, 'camera_world_matrix': copy.deepcopy(body),
                    'rgb_path': str(path), 'rgb_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

        def infer(observation, instruction):
            seen.append(list(observation['position']))
            if inference_timeout:
                raise TimeoutError('worker response deadline')
            index = len(seen)-1
            return {'status': 'PASS', 'observation_digest': 'old' if wrong_reply else observation_digest(observation),
                    'instruction': instruction, 'raw_output': (outputs or ['98 49 49']*2)[index]}

        def execute(plan, index):
            executed.append(index)
            state.update(position=plan['poses'][-1]['position'], yaw=plan['poses'][-1]['yaw'],
                         time=state['time']+plan['poses'][-1]['time_s'])
            if abort_during_motion:
                report['steps'][index]['partial_execution'] = {'last_pose': {'position': list(state['position'])}}
                raise KeyboardInterrupt()
            return {'simulation_time_s': state['time'], 'position': list(state['position']),
                    'yaw': state['yaw'], 'actual_collision_check': {'status': execution_status, 'object_prim_path': '/wall'}}

        report = {'steps': []}
        run_closed_loop(report=report, observe=observe, infer=infer, execute=execute,
                        guard=lambda path: {'status': guard_status, 'object_prim_path': '/wall'},
                        planner=plan_aerovla_action, instruction='Find tank', motion={},
                        max_steps=2, max_sim_seconds=max_seconds, should_abort=lambda: abort)
        return report, seen, executed

    def test_feedback_uses_post_action_view_and_retains_final_view(self):
        report, seen, executed = self.run_case()
        self.assertEqual(seen, [[0., 0., 3.], [5., 0., 3.]])
        self.assertEqual(executed, [0, 1])
        self.assertEqual(report['final_observation']['position'], [10., 0., 3.])
        self.assertEqual(report['termination']['reason'], 'step_limit')
        self.assertTrue(report['feedback_verified'])
        self.assertEqual(report['task_result']['status'], 'unjudged')

    def test_stale_post_action_view_is_system_error(self):
        report, seen, executed = self.run_case(stale=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(executed, [0])
        self.assertEqual(report['termination']['reason'], 'system_error')
        self.assertFalse(report['feedback_verified'])

    def test_reply_for_different_observation_never_moves(self):
        report, _, executed = self.run_case(wrong_reply=True)
        self.assertEqual(executed, [])
        self.assertEqual(report['termination']['reason'], 'system_error')

    def test_inference_timeout_is_separate_from_bad_model_output(self):
        report, _, executed = self.run_case(inference_timeout=True)
        self.assertEqual(executed, [])
        self.assertEqual(report['termination']['reason'], 'inference_timeout')

    def test_malformed_output_is_error_instead_of_stop(self):
        report, _, executed = self.run_case(outputs=['fly north'])
        self.assertEqual(executed, [])
        self.assertEqual(report['termination']['reason'], 'model_error')

    def test_model_stop_is_not_task_success_or_motion(self):
        report, _, executed = self.run_case(outputs=['00 49 49'])
        self.assertEqual(executed, [])
        self.assertEqual(report['termination']['reason'], 'model_stop')
        self.assertEqual(report['task_result']['status'], 'unjudged')

    def test_stop_after_feedback_still_proves_new_view_was_consumed(self):
        report, seen, executed = self.run_case(outputs=['98 49 49', '00 49 49'])
        self.assertEqual(seen, [[0., 0., 3.], [5., 0., 3.]])
        self.assertEqual(executed, [0])
        self.assertTrue(report['feedback_verified'])
        self.assertEqual(report['termination']['reason'], 'model_stop')

    def test_unsafe_plan_is_retained_and_blocked_before_motion(self):
        for status, reason in [('CONTACT', 'collision_predicted'), ('OUT_OF_BOUNDS', 'out_of_bounds'),
                               ('INDETERMINATE', 'collision_uncertain'), ('NOT_READY', 'system_error')]:
            with self.subTest(status=status):
                report, _, executed = self.run_case(guard_status=status)
                self.assertEqual(executed, [])
                self.assertEqual(report['termination']['reason'], reason)
                self.assertEqual(report['steps'][0]['planned_collision_check']['object_prim_path'], '/wall')

    def test_time_budget_blocks_action_without_clipping_model_output(self):
        report, _, executed = self.run_case(max_seconds=1)
        self.assertEqual(executed, [])
        self.assertEqual(report['termination']['reason'], 'simulation_timeout')

    def test_manual_abort_never_calls_model_or_moves(self):
        report, seen, executed = self.run_case(abort=True)
        self.assertEqual(seen, [])
        self.assertEqual(executed, [])
        self.assertEqual(report['termination']['reason'], 'manual_abort')

    def test_new_index_with_old_simulation_time_is_rejected(self):
        report, seen, executed = self.run_case(wrong_time=True)
        self.assertEqual(executed, [0])
        self.assertEqual(len(seen), 1)
        self.assertEqual(report['termination']['reason'], 'system_error')
        self.assertFalse(report['feedback_verified'])

    def test_actual_contact_is_separate_from_predicted_collision_and_retains_object(self):
        report, seen, executed = self.run_case(execution_status='CONTACT')
        self.assertEqual(executed, [0])
        self.assertEqual(len(seen), 1)
        self.assertEqual(report['termination']['reason'], 'collision_actual')
        self.assertEqual(report['termination']['detail']['object_prim_path'], '/wall')

    def test_abort_after_movement_retains_partial_execution_instead_of_unexecuted_label(self):
        report, _, _ = self.run_case(abort_during_motion=True)
        step = report['steps'][0]
        self.assertEqual(report['termination']['reason'], 'manual_abort')
        self.assertTrue(step['executed'])
        self.assertFalse(step['execution_completed'])
        self.assertEqual(step['execution_state'], 'interrupted')
        self.assertEqual(step['partial_execution']['last_pose']['position'], [5., 0., 3.])


if __name__ == '__main__':
    unittest.main()
