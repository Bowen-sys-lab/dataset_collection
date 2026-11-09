"""Bounded feedback orchestration; sensor, inference and control are external boundaries."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from scripts.aerovla_offline_probe import parse_action


def observation_digest(observation):
    return hashlib.sha256(json.dumps(observation, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def validate_observation(observation, index, previous=None, execution=None):
    if observation['index'] != index or isinstance(observation['index'], bool):
        raise ValueError('Stale or out-of-order observation index')
    clock = observation['simulation_time_s']
    if not math.isfinite(clock) or not math.isfinite(observation['yaw']):
        raise ValueError('Nonfinite observation time/yaw')
    body = np.asarray(observation['body_world_matrix'], dtype=float)
    camera = np.asarray(observation['camera_world_matrix'], dtype=float)
    position = np.asarray(observation['position'], dtype=float)
    if (body.shape != (4, 4) or camera.shape != (4, 4) or position.shape != (3,)
        or not all(np.isfinite(value).all() for value in [body, camera, position])
        or not np.allclose(body[3, :3], position, rtol=0, atol=1e-4)):
        raise ValueError('Observation pose is unavailable or inconsistent')
    if hashlib.sha256(Path(observation['rgb_path']).read_bytes()).hexdigest() != observation['rgb_sha256']:
        raise ValueError('Observation image changed')
    if previous is not None:
        if clock <= previous['simulation_time_s']:
            raise ValueError('Post-action observation time did not advance')
        if (abs(clock-execution['simulation_time_s']) > 1e-6
            or not np.allclose(position, execution['position'], rtol=0, atol=1e-4)
            or abs(observation['yaw']-execution['yaw']) > 1e-6):
            raise ValueError('Observation does not represent the actual action endpoint')


def run_closed_loop(*, report, observe, infer, execute, guard, planner, instruction,
                    motion, max_steps, max_sim_seconds, should_abort=lambda: False,
                    persist=lambda: None):
    """Keep all proposals and failure reasons; predicted collisions never execute."""
    report.update(steps=[], feedback_verified=False, task_result={'status': 'unjudged'},
                  physics_response_certified=False)
    phase = 'system_error'
    try:
        if (isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= 20
            or not math.isfinite(max_sim_seconds) or max_sim_seconds <= 0 or not instruction.strip()):
            raise ValueError('Require 1..20 steps, positive simulation budget and instruction')
        observation = observe(0)
        validate_observation(observation, 0)
        report['initial_observation'] = observation
        report['final_observation'] = observation
        start_time = observation['simulation_time_s']
        history = [observation['position']]
        for index in range(max_steps):
            if should_abort():
                report['termination'] = {'reason': 'manual_abort', 'step': index}
                break
            step = {'index': index, 'observation': observation, 'executed': False}
            report['steps'].append(step)
            persist()
            phase = 'system_error'
            validate_observation(observation, index)
            reply = infer(observation, instruction)
            step['model_reply'] = reply
            if (reply.get('status') != 'PASS' or reply.get('observation_digest') != observation_digest(observation)
                or reply.get('instruction') != instruction):
                raise ValueError('Model response status or observation/instruction identity is invalid')
            validate_observation(observation, index)
            phase = 'model_error'
            action = parse_action(reply['raw_output'].replace('</s>', ''))
            step['action'] = action
            report['feedback_verified'] = index >= 1
            if action['official_termination']:
                report['termination'] = {'reason': 'model_stop', 'step': index,
                                         'model_reason': action['termination_reason']}
                break
            phase = 'system_error'
            plan = planner(observation['position'], observation['yaw'], action, **motion)
            step['plan'] = plan
            if observation['simulation_time_s']-start_time+plan['poses'][-1]['time_s'] > max_sim_seconds:
                report['termination'] = {'reason': 'simulation_timeout', 'step': index}
                break
            clearance = guard(history + [row['position'] for row in plan['poses']])
            step['planned_collision_check'] = clearance
            if clearance['status'] != 'CLEAR':
                reason = {'CONTACT': 'collision_predicted', 'OUT_OF_BOUNDS': 'out_of_bounds',
                          'INDETERMINATE': 'collision_uncertain'}.get(clearance['status'], 'system_error')
                report['termination'] = {'reason': reason, 'step': index, 'detail': clearance}
                break
            persist()
            if should_abort():
                report['termination'] = {'reason': 'manual_abort', 'step': index}
                break
            step.update(execution_state='started', execution_completed=False, executed=None)
            persist()
            execution = execute(plan, index)
            step['execution'] = execution
            step['executed'] = True
            actual_check = execution['actual_collision_check']
            if actual_check['status'] != 'CLEAR':
                step['execution_state'] = 'blocked_actual'
                report['termination'] = {'reason': {'CONTACT': 'collision_actual', 'OUT_OF_BOUNDS': 'out_of_bounds',
                    'INDETERMINATE': 'collision_uncertain'}.get(actual_check['status'], 'system_error'),
                    'step': index, 'detail': actual_check}
                break
            step.update(execution_state='completed', execution_completed=True)
            following = observe(index+1)
            validate_observation(following, index+1, observation, execution)
            step['post_observation'] = following
            report['final_observation'] = following
            history += [row['position'] for row in plan['poses']]
            observation = following
            persist()
        else:
            report['termination'] = {'reason': 'step_limit', 'step': max_steps}
    except KeyboardInterrupt:
        report['termination'] = {'reason': 'manual_abort', 'detail': 'keyboard or abort file'}
    except TimeoutError as error:
        report['termination'] = {'reason': 'inference_timeout', 'detail': str(error)}
    except Exception as error:
        report['termination'] = {'reason': phase, 'detail': f'{type(error).__name__}: {error}'}
    finally:
        if report['steps'] and report['steps'][-1].get('execution_state') == 'started':
            last = report['steps'][-1]
            last['execution_state'] = 'interrupted'
            last['executed'] = True if last.get('partial_execution') else None
        report['result'] = ('PASS_BOUNDED_KINEMATIC_FEEDBACK' if report['feedback_verified']
                            and report['termination']['reason'] not in ['model_error', 'system_error', 'inference_timeout']
                            else 'INCOMPLETE_OR_FAILED_FEEDBACK')
        persist()
    return report
