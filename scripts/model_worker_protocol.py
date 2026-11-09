"""Local immutable per-step requests between Isaac and a resident model process."""
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.factory_closed_loop import observation_digest


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    if path.exists():
        raise FileExistsError(path)
    os.rename(temporary, path)


def wait_json(path, *, timeout_s, alive=lambda: True, should_abort=lambda: False):
    deadline = time.monotonic()+timeout_s
    while not path.is_file():
        if should_abort():
            raise KeyboardInterrupt('Manual abort requested')
        if not alive():
            raise RuntimeError('Model worker exited before response')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Model worker did not produce {path.name}')
        time.sleep(.025)
    return json.loads(path.read_text(encoding='utf-8'))


def exchange_request(exchange, observation, instruction, *, timeout_s=180,
                     alive=lambda: True, should_abort=lambda: False):
    folder = Path(exchange)/f'step-{observation["index"]:03d}'
    folder.mkdir(exist_ok=False)
    write_json(folder/'request.json', {'observation': observation, 'instruction': instruction,
                                     'observation_digest': observation_digest(observation)})
    return wait_json(folder/'response.json', timeout_s=timeout_s, alive=alive, should_abort=should_abort)


def serve_requests(exchange, predict, *, max_requests, idle_timeout_s=300):
    exchange = Path(exchange)
    for index in range(max_requests):
        folder = exchange/f'step-{index:03d}'
        try:
            request = wait_json(folder/'request.json', timeout_s=idle_timeout_s,
                                should_abort=lambda: (exchange/'stop.json').exists())
        except KeyboardInterrupt:
            return
        response = {'status': 'FAIL'}
        try:
            observation = request['observation']
            response.update(observation_digest=request['observation_digest'], instruction=request['instruction'])
            if (observation['index'] != index
                or observation_digest(observation) != request['observation_digest']
                or hashlib.sha256(Path(observation['rgb_path']).read_bytes()).hexdigest() != observation['rgb_sha256']):
                raise ValueError('Model input observation or image identity changed')
            response.update(predict(request, folder))
            # Recheck after inference so a concurrent change cannot masquerade as this view.
            if hashlib.sha256(Path(observation['rgb_path']).read_bytes()).hexdigest() != observation['rgb_sha256']:
                raise ValueError('Image changed during inference')
            response['status'] = 'PASS'
        except Exception as error:
            response.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        write_json(folder/'response.json', response)
