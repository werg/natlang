"""Exercise the durable round through real frozen Node collectors and HTTP models."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Thread

from scripts.create_training_pipeline import recipe
from scripts.create_student_improvement_pipeline import improvement_pipeline
from scripts.run_training_pipeline import run_pipeline


def server(replies):
    class Handler(BaseHTTPRequestHandler):
        calls = 0

        def do_POST(self):
            length = int(self.headers.get('Content-Length', '0'))
            self.rfile.read(length)
            index = min(type(self).calls, len(replies) - 1)
            type(self).calls += 1
            calls = replies[index]
            message = {'role': 'assistant', 'content': ''}
            if calls:
                message['tool_calls'] = [
                    {'id': f'call-{index}-{offset}', 'type': 'function',
                     'function': {'name': name, 'arguments': json.dumps(args)}}
                    for offset, (name, args) in enumerate(calls)]
            body = json.dumps({'choices': [{'message': message}],
                               'usage': {'prompt_tokens': 10, 'completion_tokens': 4}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=http.serve_forever, daemon=True)
    thread.start()
    return http, thread, Handler


def test_frozen_round_collects_and_admits_teacher_repair(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    base_config = recipe(repo, python=sys.executable)
    base_recipe = tmp_path / 'base-recipe.json'
    base_recipe.write_text(json.dumps(base_config))
    base_run = tmp_path / 'base-run'
    checkpoint = base_run / 'train-teacher/checkpoint'
    (checkpoint / 'weights').mkdir(parents=True)
    (checkpoint / 'weights/adapter_model.safetensors').write_bytes(b'fixture')
    (checkpoint / 'state.json').write_text(json.dumps({'trained_examples': 1,
                                                       'corpus': {'target_examples': 1}}))
    (base_run / 'prepared').mkdir()
    (base_run / 'prepared/splits.json').write_text('{"groups":{"runner-case":"train"}}')
    (base_run / 'training-readiness.json').write_text('{}')
    (base_run / 'pipeline-state.json').write_text(json.dumps({
        'config_sha256': hashlib.sha256(json.dumps(base_config, sort_keys=True).encode()).hexdigest(),
        'stages': {'train-teacher': {'status': 'complete'}}}))
    program = {'version': 'natlang.program/2', 'id': 'runner-case', 'kind': 'lambda_source',
               'source_groups': ['runner-case'], 'split': 'train',
               'semantics': {'root': 'one.nl',
                             'files': {'one.nl': '---\nargs: {}\nreturns: number\n---\nReturn one.\n'},
                             'inputs': {}, 'expected': 1, 'operation': 'exact'}}
    programs = tmp_path / 'programs.jsonl'
    programs.write_text(json.dumps(program) + '\n')
    student, student_thread, _ = server([[('eval', {'code': 'throw new Error("wrong branch")'})], None])
    teacher, teacher_thread, _ = server([[('return_result', {'value': 1})]])
    try:
        run = tmp_path / 'round'
        config = improvement_pipeline(base_recipe, base_run, programs, run,
                                      f'http://127.0.0.1:{student.server_port}', 'student',
                                      f'http://127.0.0.1:{teacher.server_port}', 'teacher', max_turns=2)
        config_path = tmp_path / 'round-recipe.json'
        config_path.write_text(json.dumps(config))
        assert run_pipeline(config_path, run, until='prepare-correction') == 0
        prepared = [json.loads(line) for line in (run / 'prepared-correction/teacher.jsonl').read_text().splitlines()]
        assert sum(row['training_admission']['approved'] for row in prepared) == 1
        assert json.loads((run / 'hard-state-queue.jsonl.manifest.json').read_text())['count'] == 1
        assert json.loads((run / 'preference-pairs.jsonl.manifest.json').read_text())['count'] == 1
        turns = [json.loads(line) for line in (run / 'verified-turns.jsonl').read_text().splitlines()]
        assert len(turns) == 2
        assert turns[0]['training_admission']['approved'] is False
        assert turns[1]['training_admission']['approved'] is True
        assert run_pipeline(config_path, run, until='combine-verified') == 0
    finally:
        student.shutdown()
        teacher.shutdown()
        student_thread.join()
        teacher_thread.join()
        student.server_close()
        teacher.server_close()
