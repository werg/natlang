import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

from scripts.create_model_swap_config import model_swap_config
from scripts.run_managed_improvement import ready, recipe_digest, run_managed, validate_spec


SERVER = '''
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys
port, role, log = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
log.write_text(log.read_text() + role + "\\n" if log.exists() else role + "\\n")
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = role.encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args): pass
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''
STAGE = '''
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
import sys
url, expected, output = sys.argv[1:]
try:
    with urlopen(url, timeout=2) as response: actual = response.read().decode()
except URLError: actual = "down"
if actual != expected: raise SystemExit(f"expected {expected}, got {actual}")
Path(output).write_text(expected)
'''


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def fixture(tmp_path):
    port = free_port()
    endpoint = f'http://127.0.0.1:{port}'
    run = tmp_path / 'round'
    def stage(name, expected):
        output = f'${{run}}/{name}.done'
        return {'id': name, 'command': [sys.executable, '-c', STAGE,
                                        endpoint + '/health', expected, output],
                'inputs': [], 'outputs': [output]}
    config = {'version': 'natlang.training_pipeline/1', 'repository': str(tmp_path),
              'run_directory': str(run), 'collection': {'student_server': endpoint,
                                                         'teacher_server': endpoint},
              'stages': [stage('build-hard-states', 'student'),
                         stage('combine-verified', 'teacher'),
                         stage('train-correction', 'down')]}
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps(config))
    launch_log = tmp_path / 'launches.txt'
    def service(role):
        return {'endpoint': endpoint, 'ready_url': endpoint + '/health',
                'start': [sys.executable, '-c', SERVER, str(port), role, str(launch_log)],
                'env': {}, 'startup_timeout_seconds': 10}
    spec = {'version': 'natlang.model_swap/1', 'pipeline_sha256': recipe_digest(config),
            'run_directory': str(run), 'student': service('student'), 'teacher': service('teacher')}
    services = tmp_path / 'services.json'
    services.write_text(json.dumps(spec))
    return recipe, run, services, launch_log


def test_managed_round_swaps_one_port_and_resumes_without_relaunch(tmp_path):
    recipe, run, services, launch_log = fixture(tmp_path)
    assert run_managed(recipe, run, services) == 0
    assert launch_log.read_text().splitlines() == ['student', 'teacher']
    assert not (run / 'model-swap-service.json').exists()
    assert (run / 'train-correction.done').read_text() == 'down'
    assert run_managed(recipe, run, services) == 0
    assert launch_log.read_text().splitlines() == ['student', 'teacher']


def test_managed_docker_config_uses_run_specific_names(tmp_path):
    recipe, run, _, _ = fixture(tmp_path)
    config = json.loads(recipe.read_text())
    base_adapter = tmp_path / 'base-run/train-teacher/checkpoint/weights/adapter_model.safetensors'
    base_adapter.parent.mkdir(parents=True)
    base_adapter.write_bytes(b'adapter')
    config['continuation'] = {'base_run': str(tmp_path / 'base-run')}
    config['collection']['teacher_model'] = 'Ternary-Bonsai-2-27B'
    recipe.write_text(json.dumps(config))
    model = tmp_path / 'models/student.gguf'
    model.parent.mkdir()
    model.write_bytes(b'fixture')
    result = model_swap_config(recipe, run, model.name)
    assert result['student']['env']['NATLANG_SERVER_NAME'].startswith('natlang-student-')
    assert result['teacher']['env']['BONSAI_SERVER_NAME'].startswith('natlang-teacher-')
    assert result['student']['stop'][-1] != 'natlang-llama'
    assert result['teacher']['stop'][-1] != 'natlang-bonsai'
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='serving artifact changed'):
        validate_spec(result, json.loads(recipe.read_text()), run)
    config['collection']['student_server'] = 'http://remote.example:8080'
    recipe.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='local HTTP endpoint'):
        model_swap_config(recipe, run, model.name)


def test_refuses_to_take_over_an_existing_unowned_service(tmp_path):
    recipe, run, services, launch_log = fixture(tmp_path)
    endpoint = json.loads(services.read_text())['student']['ready_url']
    port = int(endpoint.split(':')[2].split('/')[0])
    external = subprocess.Popen([sys.executable, '-c', SERVER, str(port), 'external', str(launch_log)],
                                start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not ready(endpoint) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready(endpoint)
        with pytest.raises(RuntimeError, match='already live but not owned'):
            run_managed(recipe, run, services)
        assert external.poll() is None
        assert not (run / 'model-swap-service.json').exists()
    finally:
        os.killpg(external.pid, signal.SIGTERM)
        external.wait()


def test_failed_stage_still_releases_owned_service(tmp_path):
    recipe, run, services, _ = fixture(tmp_path)
    config = json.loads(recipe.read_text())
    config['stages'][0]['command'][-2] = 'impossible-role'
    recipe.write_text(json.dumps(config))
    plan = json.loads(services.read_text())
    plan['pipeline_sha256'] = recipe_digest(config)
    services.write_text(json.dumps(plan))
    assert run_managed(recipe, run, services) != 0
    assert not ready(plan['student']['ready_url'])
    assert not (run / 'model-swap-service.json').exists()
