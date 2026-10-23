"""Current hardware profiles, native selective checkpointing, and teacher deployment checks."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train import hardware_profile as hp
from natlang_neuralese.train.recipe import OPERATIONAL_STAGE_PARAMETERS

REPO = Path(__file__).resolve().parents[2]
SM120 = {'name': 'NVIDIA RTX PRO 6000 Blackwell Server Edition', 'capability': [12, 0], 'memory_gb': 95.0,
         'integrated': False}
GB10 = {'name': 'NVIDIA GB10', 'capability': [12, 1], 'memory_gb': 119.7, 'integrated': True}
ADA = {'name': 'NVIDIA GeForce RTX 4060 Laptop GPU', 'capability': [8, 9], 'memory_gb': 8.0, 'integrated': False}


def test_profile_file_is_valid_and_only_operational():
    document = hp.load()
    assert hp.PROFILE_STAGE_PARAMETERS <= OPERATIONAL_STAGE_PARAMETERS
    for profile in document['profiles']:
        for values in profile.get('stage_parameters', {}).values():
            assert set(values) <= hp.PROFILE_STAGE_PARAMETERS


def test_selection_by_device():
    document = hp.load()
    assert hp.select(document, SM120)['name'] == 'rtx-pro-6000-blackwell'
    assert hp.select(document, GB10)['name'] == 'dgx-gb10'
    assert hp.select(document, ADA)['name'] == 'pop-ada'
    assert hp.select(document, {**SM120, 'memory_gb': 32.0}) is None  # an RTX 5090 (sm_120, 32 GB) is not this part
    assert hp.select(document, None) is None
    assert hp.select(document, SM120, override='none') is None
    assert hp.select(document, GB10, override='rtx-pro-6000-blackwell')['name'] == 'rtx-pro-6000-blackwell'
    with pytest.raises(hp.ProfileError, match='names no profile'):
        hp.select(document, GB10, override='h100')


def test_training_math_is_refused():
    base = {'schema': hp.SCHEMA, 'profiles': [{'name': 'x', 'match': {'capability': [12, 0]},
                                                'stage_parameters': {'native_state_sft': {'batch_tokens': 65536}}}]}
    with pytest.raises(hp.ProfileError, match='Training math stays in the recipe'):
        hp.validate(base)
    base['profiles'][0]['stage_parameters'] = {}
    base['profiles'][0]['environment'] = {'NATLANG_MOE_FP8': '1'}
    with pytest.raises(hp.ProfileError, match='environment may set only'):
        hp.validate(base)


def test_apply_records_replaced_values(tmp_path):
    profile = {'name': 'p', 'match': {'capability': [12, 0]}, 'status': 'measured',
               'stage_parameters': {'native_state_sft': {'checkpoint_layers': False}},
               'environment': {'NATLANG_MAPLE_COMPILE': '1'}}
    merged, env, record = hp.apply(profile, 'native_state_sft', {'checkpoint_layers': True, 'epochs': 1.0})
    assert merged == {'checkpoint_layers': False, 'epochs': 1.0}
    assert env == {'NATLANG_MAPLE_COMPILE': '1'}
    assert record['applied'] == {'checkpoint_layers': {'recipe': True, 'profile': False}}
    assert hp.apply(profile, 'native_state_rl', {'checkpoint_layers': True})[0] == {'checkpoint_layers': True}
    path = tmp_path / 'profiles.json'
    path.write_text(json.dumps({'schema': hp.SCHEMA, 'profiles': [profile]}))
    merged, env, record = hp.for_stage('native_state_sft', {}, path=path, detected=SM120)
    assert record['profile'] == 'p' and record['device'] == SM120 and merged['checkpoint_layers'] is False
    assert hp.for_stage('native_state_sft', {'k': 1}, path=tmp_path / 'absent.json', detected=SM120) == (
        {'k': 1}, {}, None)


def test_gb10_profile_changes_nothing():
    """The running DGX lineages see no profile change: the GB10 profile is empty."""
    merged, env, record = hp.for_stage('native_state_sft', {'checkpoint_layers': True}, detected=GB10)
    assert merged == {'checkpoint_layers': True} and env == {} and record['applied'] == {}


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_native_checkpointed_graph_ce_and_gradients_match(family):
    """Operational activation checkpointing preserves the actual native graph objective."""
    import copy
    from test_shared_state import tiny
    from test_native_state_sft import graph
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.data.native_graph import graph_loss

    torch.manual_seed(817)
    model = SharedStateModel(tiny(family), (58, 59, 60, 61), rank=4)
    checkpointed = copy.deepcopy(model)
    example = parse_graph({**graph('hardware-native-checkpoint'),
                           'tokenizer_fingerprint': 'tokenizer-sha'}, tokenizer_fingerprint='tokenizer-sha')
    full_loss, _ = graph_loss(model, example, checkpoint_layers=False, chunk_tokens=4)
    checked_loss, _ = graph_loss(checkpointed, example, checkpoint_layers=True, chunk_tokens=4)
    torch.testing.assert_close(checked_loss, full_loss, atol=3e-6, rtol=2e-5)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    full_gradients = torch.autograd.grad(full_loss, parameters, allow_unused=True)
    checked_parameters = tuple(parameter for parameter in checkpointed.parameters() if parameter.requires_grad)
    assert len(checked_parameters) == len(parameters)
    checked_gradients = torch.autograd.grad(checked_loss, checked_parameters, allow_unused=True)
    for actual, expected in zip(checked_gradients, full_gradients):
        assert (actual is None) == (expected is None)
        if actual is not None:
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-4)


def _load_teacher_bench():
    sys.path.insert(0, str(REPO / 'scripts' / 'coreweave'))
    import teacher_bench
    return teacher_bench


def test_teacher_bench_against_a_fake_server(tmp_path):
    tb = _load_teacher_bench()
    prompts = tb.synthetic_prompts(4, prefix_tokens=60, tail_tokens=12)
    assert len({p['messages'][0]['content'] for p in prompts}) == 1  # one shared prefix
    assert len({p['messages'][1]['content'] for p in prompts}) == 4

    class Handler(BaseHTTPRequestHandler):
        hits = 0

        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/metrics':
                Handler.hits += 3
                body = (f'vllm:prefix_cache_hits_total{{model="m"}} {Handler.hits}\n'
                        f'vllm:prefix_cache_queries_total{{model="m"}} {Handler.hits * 2}\n').encode()
            else:
                body = b'{"data": []}'
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for token in ('a', 'b', 'c'):
                self.wfile.write(f'data: {json.dumps({"choices": [{"delta": {"content": token}}]})}\n\n'.encode())
            usage = {'choices': [], 'usage': {'prompt_tokens': 70, 'completion_tokens': 3}}
            self.wfile.write(f'data: {json.dumps(usage)}\n\ndata: [DONE]\n\n'.encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tmp_path / 'teacher.json'
        assert tb.main(['--base-url', f'http://127.0.0.1:{server.server_port}', '--model', 'm', '--out', str(out),
                        '--concurrency', '1,2', '--requests', '4', '--prefix-tokens', '60', '--tail-tokens', '12']) == 0
        document = json.loads(out.read_text())
        levels = document['configs'][0]['levels']
        assert [level['ok'] for level in levels] == [4, 4]
        assert levels[0]['output_tokens_per_second'] > 0 and levels[0]['prefix_cache_hit_rate'] == pytest.approx(0.5)
        assert document['ranking'][0]['name'] == 'running'
    finally:
        server.shutdown()


def test_coreweave_bench_jobs_validate():
    sys.path.insert(0, str(REPO / 'scripts' / 'coreweave'))
    import job
    for name in ('hw-bench-train', 'hw-bench-teacher', 'hw-bench-probe'):
        declared = job.validate(json.loads((REPO / 'scripts/coreweave/jobs' / f'{name}.json').read_text()))
        assert declared['resources']['gpu'] == 1
    configs = json.loads((REPO / 'scripts/coreweave/teacher-configs.json').read_text())
    assert len({c['name'] for c in configs}) == len(configs) and all(c['command'][0] == 'vllm' for c in configs)
