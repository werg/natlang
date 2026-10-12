"""Hardware profiles, selective checkpointing, the hw_bench matrix and the teacher bench
(plans/COREWEAVE_HARDWARE_OPTIMIZATION.md)."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train import hardware_profile as hp
from natlang_neuralese.train import hw_bench
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
                                                'stage_parameters': {'neuralese_direct': {'batch_tokens': 65536}}}]}
    with pytest.raises(hp.ProfileError, match='Training math stays in the recipe'):
        hp.validate(base)
    base['profiles'][0]['stage_parameters'] = {}
    base['profiles'][0]['environment'] = {'NATLANG_MOE_FP8': '1'}
    with pytest.raises(hp.ProfileError, match='environment may set only'):
        hp.validate(base)


def test_apply_records_replaced_values(tmp_path):
    profile = {'name': 'p', 'match': {'capability': [12, 0]}, 'status': 'measured',
               'stage_parameters': {'neuralese_direct': {'checkpoint_layers': False, 'window_execution': 'packed'}},
               'environment': {'NATLANG_MAPLE_COMPILE': '1'}}
    merged, env, record = hp.apply(profile, 'neuralese_direct', {'window_execution': 'packed', 'epochs': 1.0})
    assert merged == {'window_execution': 'packed', 'epochs': 1.0, 'checkpoint_layers': False}
    assert env == {'NATLANG_MAPLE_COMPILE': '1'}
    assert record['applied'] == {'checkpoint_layers': {'recipe': None, 'profile': False}}
    assert hp.apply(profile, 'core_text_warmup', {'a': 1})[0] == {'a': 1}  # other stage kinds untouched
    path = tmp_path / 'profiles.json'
    path.write_text(json.dumps({'schema': hp.SCHEMA, 'profiles': [profile]}))
    merged, env, record = hp.for_stage('neuralese_direct', {}, path=path, detected=SM120)
    assert record['profile'] == 'p' and record['device'] == SM120 and merged['checkpoint_layers'] is False
    assert hp.for_stage('neuralese_direct', {'k': 1}, path=tmp_path / 'absent.json', detected=SM120) == (
        {'k': 1}, {}, None)


def test_gb10_profile_changes_nothing():
    """The running DGX lineages see no profile change: the GB10 profile is empty."""
    merged, env, record = hp.for_stage('neuralese_direct', {'window_execution': 'packed'}, detected=GB10)
    assert merged == {'window_execution': 'packed'} and env == {} and record['applied'] == {}


def test_uncheckpointed_layers_are_exact():
    """Selective checkpointing (the top N layers keep their activations) equals full checkpointing."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from natlang_neuralese.model.hf_port import QwenPortBackbone
    from natlang_neuralese.model.lfm2_port import ControlTokens
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                         attn_implementation='eager')
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    port = QwenPortBackbone(model, ControlTokens(open_id=62, close_id=63))
    for p in port.layers[3].parameters():
        p.requires_grad_(True)
    ids = torch.randint(0, 60, (2, 12))

    def run(uncheckpointed):
        port.checkpoint_layers, port.uncheckpointed_layers = True, uncheckpointed
        for p in port.layers[3].parameters():
            p.grad = None
        x = port.embed(ids).detach().requires_grad_(True)
        h, cache = port.run_layers(x, range(0, port.num_layers), port_cache(port))
        (h.square().sum() + sum(s.k.sum() for s in cache.states)).backward()
        return h.detach(), x.grad, port.layers[3].mlp.down_proj.weight.grad.clone()

    def port_cache(p):
        from natlang_neuralese.model.lfm2_port import PortCache
        return PortCache.empty(p.num_layers)

    full = run(0)
    for n in (1, 2, 4):
        other = run(n)
        for a, b in zip(other, full):
            torch.testing.assert_close(a, b)


def test_hw_bench_arms_and_derive():
    for matrix in ('smoke', 'full'):
        names = [arm['name'] for arm in hw_bench.arms(matrix)]
        assert len(names) == len(set(names)) and 'm-ref' in names and 'l-ref' in names
    records = [
        {'name': 'm-ref', 'reference': 'm-ref', 'status': 'ok', 'losses': [15.2], 'median_seconds': 4.0,
         'supervised_per_update': 32000, 'supervised_tokens_per_second': 8000.0},
        {'name': 'm-unckpt4', 'reference': 'm-ref', 'status': 'ok', 'losses': [15.2005], 'median_seconds': 3.5,
         'supervised_per_update': 32000, 'supervised_tokens_per_second': 32000 / 3.5},
        {'name': 'm-bf16', 'reference': None, 'status': 'ok', 'losses': [15.3], 'median_seconds': 3.0,
         'supervised_per_update': 32000, 'supervised_tokens_per_second': 32000 / 3.0},
        {'name': 'm-unckpt14', 'reference': 'm-ref', 'status': 'oom'},
    ]
    derived = hw_bench.derive(records)
    assert derived['numerics']['m-unckpt4']['equal']
    assert derived['fastest_equal']['m-ref']['arm'] == 'm-unckpt4'
    assert derived['fastest_equal']['m-ref']['speedup_over_reference'] == pytest.approx(4.0 / 3.5)
    assert derived['fastest_any']['m']['arm'] == 'm-bf16'


def test_hw_bench_dry_run(capsys):
    assert hw_bench.main(['--out', '/tmp/unused', '--matrix', 'smoke', '--dry-run', '--only', '^l-',
                          '--lfm', '/models/lfm']) == 0
    arms = json.loads(capsys.readouterr().out)
    assert [a['name'] for a in arms] == ['l-ref', 'l-nockpt'] and arms[0]['args'][-2:] == ['--model', '/models/lfm']


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
    for name in ('hw-bench-train', 'hw-bench-teacher'):
        declared = job.validate(json.loads((REPO / 'scripts/coreweave/jobs' / f'{name}.json').read_text()))
        assert declared['resources']['gpu'] == 1
    configs = json.loads((REPO / 'scripts/coreweave/teacher-configs.json').read_text())
    assert len({c['name'] for c in configs}) == len(configs) and all(c['command'][0] == 'vllm' for c in configs)
