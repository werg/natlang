import json
import threading
from argparse import Namespace
import hashlib
from pathlib import Path

import pytest

from natlang_neuralese.rl import memetic_program as memetic
from natlang_neuralese.rl import memetic_batch
from natlang_neuralese.rl import memetic_task_runner


def case(ident, program='p', split='train'):
    return {'id': ident, 'program_id': program, 'split': split,
        'curriculum': {'reference': {'root': [['eval', {'code': 'return 1'}]]}}}


def test_repeated_program_slices_are_disjoint_train_only(tmp_path):
    dev, confirmation = tmp_path / 'dev.jsonl', tmp_path / 'confirm.jsonl'
    dev.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('a', 'b')))
    confirmation.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('c', 'd')))
    left, right = memetic.prepare_slices(dev, confirmation, 'p')
    assert [item['id'] for item in left + right] == ['a', 'b', 'c', 'd']
    confirmation.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('a', 'd')))
    with pytest.raises(ValueError, match='overlap'):
        memetic.prepare_slices(dev, confirmation, 'p')
    confirmation.write_text(''.join(json.dumps(case(i, split='test')) + '\n' for i in ('c', 'd')))
    with pytest.raises(ValueError, match='non-train'):
        memetic.prepare_slices(dev, confirmation, 'p')


def test_batch_derives_program_from_code_and_keeps_worlds_disjoint(tmp_path):
    source = tmp_path / 'source.jsonl'
    rows = []
    for index in range(6):
        row = case(f'case-{index}')
        del row['program_id']
        row['semantics'] = {'root': 'review.nl', 'files': {'review.nl': 'Return the bound evidence.'},
                            'services': {'packet': f'export function read() {{ return {index // 2}; }}'},
                            'service_scopes': {'packet': ['read']}, 'inputs': {'question': str(index)},
                            'expected': index}
        row['source_groups'] = [f'world-{index // 2}']
        rows.append(row)
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = memetic_batch.select(source, tmp_path / 'batch')
    assert manifest['world_components'] == 3
    assert len(manifest['jobs']) == 1 and manifest['jobs'][0]['ready']
    job = manifest['jobs'][0]
    dev, confirmation = memetic.prepare_slices(Path(job['development']['path']),
                                                Path(job['confirmation']['path']), job['program_id'])
    assert len(dev) >= 2 and len(confirmation) >= 2
    assert {item['memetic_source']['world_bindings'][0] for item in dev}.isdisjoint(
        {item['memetic_source']['world_bindings'][0] for item in confirmation})
    assert all(item['memetic_source']['source_sha256'] == manifest['source_sha256'] for item in dev + confirmation)
    with pytest.raises(ValueError, match='resume changed'):
        extra = {**rows[0], 'id': 'new-case'}
        source.write_text(source.read_text() + json.dumps(extra) + '\n')
        memetic_batch.select(source, tmp_path / 'batch')


def test_batch_baseline_runs_without_author_credentials(tmp_path, monkeypatch):
    source = tmp_path / 'cases.jsonl'
    rows = []
    for index in range(4):
        row = case(str(index)); del row['program_id']
        row['semantics'] = {'root': 'task.nl', 'files': {'task.nl': 'Return the bound result.'},
                            'inputs': {'value': index}, 'expected': index}
        row['source_groups'] = [f'world-{index}']
        rows.append(row)
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    seen = []
    def fake_program(arguments):
        seen.append(arguments)
        return {'status': 'baseline_complete'}
    monkeypatch.setattr(memetic_batch.memetic_program, 'main', fake_program)
    memetic_batch.main(['--source', str(source), '--out', str(tmp_path / 'batch'),
        '--server', 'http://student', '--model-id', 'lfm', '--baseline-only'])
    assert len(seen) == 1 and '--baseline-only' in seen[0] and '--author' not in seen[0]
    progress = json.loads((tmp_path / 'batch/progress.json').read_text())
    assert progress['results'][0]['status'] == 'baseline_complete'


def test_binding_requires_actual_native_id_and_oracle_gain(tmp_path):
    ident = 'nzs1_' + 'a' * 52
    assert memetic.state_guidance('Check the contract.', ident) == '\ue002' + ident + '\ue003'
    with pytest.raises(ValueError):
        memetic.state_guidance('Check.', 'nz1_' + 'a' * 52)
    assert memetic.acceptance({'scored': 4, 'successes': 1}, {'scored': 4, 'successes': 2})
    assert not memetic.acceptance({'scored': 4, 'successes': 1}, {'scored': 3, 'successes': 3})
    assert not memetic.acceptance({'scored': 4, 'successes': 1}, {'scored': 4, 'successes': 1})
    baseline = {'development': {'scored': 4, 'successes': 1},
                'confirmation': {'scored': 4, 'successes': 1}}
    assert memetic.qualification(baseline, {'development': {'scored': 4, 'successes': 2},
                                             'confirmation': {'scored': 4, 'successes': 2}}, .1)
    assert not memetic.qualification(baseline, {'development': {'scored': 4, 'successes': 2},
                                                 'confirmation': {'scored': 4, 'successes': 1}}, .1)
    output = tmp_path / 'row-000' / 'sample-00'
    output.mkdir(parents=True)
    (output / 'results.jsonl').write_text(json.dumps({'trajectory': [
        {'model_response': {'native_state_imports': [{'id': ident}]}}]}) + '\n')
    packet = tmp_path / 'packet.jsonl'; packet.write_text(json.dumps(case('one'))+'\n')
    (tmp_path/'queue.jsonl').write_text(json.dumps({'output':str(output/'results.jsonl'),
        'key':'one','index':0})+'\n')
    (tmp_path/'preparation.json').write_text(json.dumps({'packet_sha256':hashlib.sha256(packet.read_bytes()).hexdigest()}))
    (tmp_path/'journal.jsonl').write_text('')
    assert memetic.native_import_reached(tmp_path, ident)
    assert not memetic.native_import_reached(tmp_path, 'nzs1_' + 'b' * 52)


def test_author_requires_exact_self_hosted_source_receipt(tmp_path):
    from natlang_neuralese.rl import memetic_teacher_ferry as ferry
    path = tmp_path / 'teacher.json'
    receipt = {'schema': 'natlang.memetic-author-provenance/1',
        'model_id': ferry.TEACHER_MODEL_ID, 'serve_alias': ferry.TEACHER_SERVE_ALIAS,
        'revision': ferry.TEACHER_REVISION,
        'hosting': 'self-hosted-coreweave', 'endpoint': ferry.TEACHER_URL,
        'sandbox_id': ferry.TEACHER_SANDBOX_ID,
        'source_bindings': list(ferry.TEACHER_SOURCE_BINDINGS),
        'checkpoint_sha256': 'a' * 64, 'tokenizer_fingerprint': 'b' * 64,
        'runtime_sha256': 'c' * 64, 'source_receipt_sha256': 'd' * 64}
    path.write_text(json.dumps(receipt))
    assert memetic.author_provenance(path, receipt['serve_alias'], receipt['endpoint'])['receipt'] == receipt
    receipt['sandbox_id'] = 'unbound-teacher'
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='pinned self-hosted'):
        memetic.author_provenance(path, receipt['serve_alias'], receipt['endpoint'])


def test_writer_rejects_wrong_student_or_missing_producer(monkeypatch):
    ident = 'nzs1_' + 'a' * 52
    actor = {key: 'f' * 64 for key in memetic.IDENTITY}
    actor['architecture'] = 'lfm'
    good = {'id': ident, 'native_state': actor,
        'source_capture': {'policy': actor, 'segments': [
            {'producing': [False]}, {'producing': [True]}, {'producing': [False]}]},
        'source_text_sha256': __import__('hashlib').sha256(b'guide').hexdigest()}
    monkeypatch.setattr(memetic, 'request_json', lambda *a, **k: good)
    assert memetic.write_state('http://localhost', 'guide', actor) == good
    bad = {**good, 'native_state': {**actor, 'checkpoint_sha256': '0' * 64}}
    monkeypatch.setattr(memetic, 'request_json', lambda *a, **k: bad)
    with pytest.raises(ValueError, match='identity'):
        memetic.write_state('http://localhost', 'guide', actor)


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_guidance_graph_backpropagates_through_local_producer(family):
    import torch
    from natlang_neuralese.data.native_graph import graph_loss, parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel, RoleTokens, native_step, concatenate_captures
    from natlang_neuralese.model.state_block import NativeStateBlock
    from natlang_neuralese.serve.store import native_state_id
    from test_shared_state import tiny
    actor = {'checkpoint_sha256': '1'*64, 'tokenizer_fingerprint': '2'*64,
             'embedding_weight_sha256': '3'*64, 'architecture': tiny(family).config.model_type}
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    with torch.no_grad():
        opened = native_step(model, torch.tensor([[58]]))
        capture = native_step(model, torch.tensor([[5, 6]]), opened['cache'], producing=True)
    block = NativeStateBlock(text='guide', **actor, **concatenate_captures([capture])).validate()
    ident = native_state_id(block)
    written = {'id': ident,
        'source_capture': {'policy': actor, 'segments': [
            {'kind': 'tokens', 'token_ids': [58], 'producing': [False]},
            {'kind': 'tokens', 'token_ids': [5, 6], 'producing': [True, True]},
            {'kind': 'tokens', 'token_ids': [59], 'producing': [False]}], 'payload_start': 1, 'payload_stop': 3}}
    capture = {'schema': 'natlang.native-rl-sampling/1', 'policy': actor,
        'prompt': [{'kind': 'tokens', 'token_ids': [3, 60], 'producing': [False, False]},
                   {'kind': 'native_state', 'id': ident, 'native_identity': actor},
                   {'kind': 'tokens', 'token_ids': [61], 'producing': [False]}],
        'samples': [{'index': 0, 'token_id': 11, 'logp': -1., 'input_role': 'ordinary', 'terminal': False},
                    {'index': 1, 'token_id': 2, 'logp': -1., 'input_role': 'ordinary', 'terminal': True}],
        'writes': [], 'sample_range_convention': 'half-open', 'finish_reason': 'stop',
        'sampling_law': {'temperature': .8, 'seed': 3, 'guidance': False,
                         'probabilities': 'post-processing-normalized'}}
    rollout = {'task_id': 'case-1', 'sample': 0, 'split': 'train', 'reward': {'quality': 1.},
        'turns': [{'invocation_id': 'call', 'messages': [], 'target': {}, 'sampling': capture}],
        'execution_graph': [{'node': 'call#turn1', 'kind': 'model_turn', 'call_id': 'call'},
            {'node': 'read-guidance', 'kind': 'block_read', 'turn': 'call#turn1', 'block': ident,
             'inputs': [{'node': 'external', 'block': ident}]}]}
    row = memetic.guidance_training_graph(rollout, program_id='p', written=written, actor=actor,
        roles=RoleTokens(58, 59, 60, 61), fetch_block=lambda _: block)
    assert row['nodes'][1]['imports'][0]['at'] == 2
    assert row['nodes'][1]['imports'][0]['source'] == 'memetic-guidance-producer'
    example = parse_graph(row, tokenizer_fingerprint=actor['tokenizer_fingerprint'])
    loss, _ = graph_loss(model, example, checkpoint_layers=True, chunk_tokens=1)
    loss.backward()
    assert model.adapter.up.weight.grad is not None
    assert model.adapter.up.weight.grad.norm().item() > 0


def test_recipe_continuation_uses_native_earned_parent(tmp_path):
    base = Path(__file__).resolve().parents[2] / 'training/neuralese/recipes/native-state-production-base-20261015-v1.json'
    output = tmp_path / 'recipe.json'
    binding = {'path': '/work/memetic/train.jsonl', 'sha256': 'a' * 64}
    receipt = memetic.emit_recipe(base, output, binding, foundation_id='native_full_sft')
    recipe = json.loads(output.read_text())
    stage = recipe['stages'][0]
    assert receipt['stage_id'] == stage['id']
    assert recipe['stages'][1]['requires'] == [stage['id']]
    assert stage['inputs']['native_graphs'] == 'memetic_native_train_graphs'
    assert stage['inputs']['held_native_graphs'] == 'native_held_graphs'
    assert recipe['input_bindings']['memetic_native_train_graphs'] == binding


def test_recipe_keeps_downstream_stages_and_foundation_dependencies(tmp_path):
    from natlang_neuralese.rl.native_raw_rounds import _supervised_stage_inventory
    base = Path(__file__).resolve().parents[2] / 'training/neuralese/recipes/native-state-production-base-20261015-v1.json'
    recipe = json.loads(base.read_text())
    recipe['stages'].insert(0, {**recipe['stages'][0], 'id': 'prelude', 'requires': []})
    recipe['stages'][1]['requires'] = ['prelude']
    recipe['stages'].append({**recipe['stages'][1], 'id': 'later', 'requires': ['native_full_sft']})
    source = tmp_path / 'source.json'; source.write_text(json.dumps(recipe))
    destination = tmp_path / 'continued.json'
    memetic.emit_recipe(source, destination, {'path': '/work/memetic.jsonl', 'sha256': 'a' * 64},
                        foundation_id='native_full_sft')
    after = json.loads(destination.read_text())
    assert [stage['id'] for stage in after['stages']] == ['prelude', 'memetic_native_guidance', 'native_full_sft', 'later']
    assert after['stages'][1]['requires'] == ['prelude']
    assert after['stages'][2]['requires'] == ['memetic_native_guidance']
    assert after['stages'][3]['requires'] == ['native_full_sft']
    supervised = [stage['id'] for stage in _supervised_stage_inventory(after, 'native_full_sft')]
    assert supervised.index('memetic_native_guidance') < supervised.index('native_full_sft')
    assert set(supervised) == {'prelude', 'memetic_native_guidance', 'native_full_sft', 'later'}


def test_real_collector_imports_written_state_from_tiny_shared_http_server(tmp_path, monkeypatch):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    from test_shared_state import tiny
    from natlang_neuralese.model.shared_state import SharedStateModel, RoleTokens, ROLE_MARKERS
    from natlang_neuralese.model.native_model import native_identity, model_state
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.serve.store import TensorStore
    from natlang_neuralese.serve.http import serve

    repo = Path(__file__).resolve().parents[2]
    assert (repo / 'ts-host/dist/teacher/collector.js').is_file(), 'build:node before collector test'
    port = tiny('lfm')
    model = SharedStateModel(port, marker_ids=(58, 59, 60, 61), rank=4).eval()
    vocab = {f't{index}': index for index in range(port.embedding_weight.shape[0])}
    for marker, token in zip(ROLE_MARKERS, model.marker_ids):
        del vocab[f't{token}']; vocab[marker] = token
    del vocab['t5']; vocab['guide'] = 5
    raw = Tokenizer(WordLevel(vocab, unk_token='t0'))
    raw.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw,
        unk_token='t0', bos_token='t1', eos_token='t2', pad_token='t3',
        additional_special_tokens=list(ROLE_MARKERS))
    tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] }}:"
        "{% if message['role'] == 'assistant' and message.get('reasoning_content') and loop.last %}"
        "{{ message['reasoning_content'] }}{% endif %}{{ message['content'] }}{% endfor %}assistant:")
    weights = tmp_path / 'weights.pt'; torch.save(model_state(model), weights)
    config = {'architecture': port.config.model_type, 'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer)}
    identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex(weights))
    engine = Engine(port, None, tokenizer, TensorStore(), 'native-state/' + port.config.model_type, shared_model=model,
        role_tokens=RoleTokens(*model.marker_ids), model_config=config,
        checkpoint_sha256=identity['checkpoint_sha256'],
        embedding_weight_sha256=identity['embedding_weight_sha256'], device='cpu')
    engine.native_identity = identity
    engine.rl_policy_binding = {'schema': 'natlang.rl-policy-binding/2',
        'checkpoint_sha256': identity['checkpoint_sha256'], 'communication': 'native-state',
        'native_identity': dict(identity)}
    engine.context = 8192
    monkeypatch.setattr(engine, '_native_token', lambda sequence: (2, -.1))
    engine.start()
    server = serve(engine, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        url = f'http://127.0.0.1:{server.server_address[1]}'
        assert memetic.actor_identity(url) == identity
        info = memetic.request_json(url, '/v1/neuralese/info')
        stale_info = json.loads(json.dumps(info))
        stale_info['native_state']['checkpoint_sha256'] = '9' * 64
        with pytest.raises(ValueError, match='consistent native RL policy'):
            memetic.actor_identity(url, stale_info)
        unbound_info = json.loads(json.dumps(info))
        unbound_info['native_identity'] = identity
        del unbound_info['rl_policy_binding']
        with pytest.raises(ValueError, match='consistent native RL policy'):
            memetic.actor_identity(url, unbound_info)
        written = memetic.write_state(url, 'guide', identity)
        task = {**case('one'), 'version': 'natlang.program/2', 'kind': 'lambda_source',
            'semantics': {'root': 'task.nl', 'files': {'task.nl': '---\nargs: {}\nreturns: "number"\n---\nReturn 1.\n'},
                          'inputs': {}, 'expected': 1}}
        # The authored root calls the student in a child; a direct return would
        # only replay reference code and never exercise the server.
        task['curriculum']['reference']['root'][0][1]['code'] = (
            'const f = nl<() => Promise<number>>`Return 1.`; return await f();')
        packet = tmp_path / 'packet.jsonl'; packet.write_text(json.dumps(task) + '\n')
        args = Namespace(out=tmp_path, rows=1, samples=2, shards=1, temperature=1.,
            case_seconds=40, max_model_requests=2, max_turns=2, seed_base=7,
            server=url, model_id='lfm', policy_checkpoint_sha256=identity['checkpoint_sha256'])
        fit = memetic_task_runner.evaluate(args, {'id': 'actual',
            'guidance': memetic.state_guidance('guide', written['id'])}, packet,
            memetic_task_runner.default_system_prompt())
        assert fit['rollouts'] == 2
        assert memetic.native_import_reached(tmp_path / 'evals/actual', written['id'])
        assert memetic.student_capture_reached(tmp_path / 'evals/actual', identity)
        development, confirmation = tmp_path / 'dev.jsonl', tmp_path / 'confirmation.jsonl'
        for path, ids in ((development, ('a', 'b')), (confirmation, ('c', 'd'))):
            path.write_text(''.join(json.dumps({**task, 'id': name}) + '\n' for name in ids))
        baseline = memetic.main(['--program-id', 'p', '--development', str(development),
            '--confirmation', str(confirmation), '--out', str(tmp_path / 'memetic-baseline'),
            '--server', url, '--model-id', 'lfm', '--baseline-only', '--samples', '2',
            '--shards', '1', '--case-seconds', '40', '--max-model-requests', '2', '--max-turns', '2'])
        assert baseline['status'] == 'baseline_complete'
        assert baseline['student'] == identity
        assert baseline['development']['scored'] == baseline['confirmation']['scored'] == 4
    finally:
        server.shutdown(); server.server_close(); engine.stop(); thread.join(timeout=5)


def test_baseline_only_binds_student_and_continues_with_author(tmp_path, monkeypatch):
    actor = {'checkpoint_sha256': 'a' * 64, 'tokenizer_fingerprint': 'b' * 64,
             'embedding_weight_sha256': 'c' * 64, 'architecture': 'lfm2'}
    student = {'native_state': {'enabled': True, **actor},
        'rl_policy_binding': {'schema': 'natlang.rl-policy-binding/2', 'communication': 'native-state',
                              'checkpoint_sha256': actor['checkpoint_sha256'], 'native_identity': dict(actor)}}
    monkeypatch.setattr(memetic, 'request_json', lambda *args, **kwargs: student)
    monkeypatch.setattr(memetic, 'local_runtime_binding', lambda: {'sha256': 'd' * 64})
    monkeypatch.setattr(memetic.runner, 'default_system_prompt', lambda: 'exact-system')
    calls = []
    def fake_evaluate(args, out, name, guidance, rows, **kwargs):
        calls.append((name, guidance, kwargs['actor']))
        return {'scored': len(rows) * args.samples, 'successes': 2 if guidance else 0,
                'dir': str(out / 'evals' / name)}
    monkeypatch.setattr(memetic, 'evaluate', fake_evaluate)
    written = {'id': 'nzs1_' + 'e' * 52, 'source_capture': {'policy': actor},
               'source_text_sha256': hashlib.sha256(b'guide').hexdigest()}
    def fake_write(*args):
        calls.append(('write', '', actor))
        return written
    monkeypatch.setattr(memetic, 'write_state', fake_write)
    monkeypatch.setattr(memetic, 'archive_state', lambda *args: {'path': '/tmp/state', 'sha256': 'f'*64, 'bytes': 1})
    development, confirmation = tmp_path / 'dev.jsonl', tmp_path / 'confirmation.jsonl'
    development.write_text(''.join(json.dumps(case(index)) + '\n' for index in ('a', 'b')))
    confirmation.write_text(''.join(json.dumps(case(index)) + '\n' for index in ('c', 'd')))
    common = ['--program-id', 'p', '--development', str(development), '--confirmation', str(confirmation),
              '--out', str(tmp_path / 'run'), '--server', 'http://student', '--model-id', 'lfm',
              '--samples', '2', '--generations', '0']
    baseline = memetic.main([*common, '--baseline-only'])
    assert baseline['status'] == 'baseline_complete'
    assert [name for name, _, _ in calls] == ['unguided-dev', 'unguided-confirm']
    assert not (tmp_path / 'run/author.json').exists()
    assert not (tmp_path / 'run/binding.json').exists()
    ferry = memetic.teacher_ferry
    receipt = {'schema': 'natlang.memetic-author-provenance/1', 'model_id': ferry.TEACHER_MODEL_ID,
        'serve_alias': ferry.TEACHER_SERVE_ALIAS, 'revision': ferry.TEACHER_REVISION,
        'hosting': 'self-hosted-coreweave', 'endpoint': ferry.TEACHER_URL,
        'sandbox_id': ferry.TEACHER_SANDBOX_ID, 'source_bindings': list(ferry.TEACHER_SOURCE_BINDINGS),
        'checkpoint_sha256': '1'*64, 'tokenizer_fingerprint': '2'*64,
        'runtime_sha256': '3'*64, 'source_receipt_sha256': '4'*64}
    path = tmp_path / 'teacher.json'; path.write_text(json.dumps(receipt))
    author = ['--author', ferry.TEACHER_URL, '--author-model', ferry.TEACHER_SERVE_ALIAS,
              '--author-ferry', 'http://127.0.0.1:8766', '--author-provenance', str(path)]
    result = memetic.main([*common, *author])
    assert result['status'] == 'qualified'
    assert [name for name, _, _ in calls].count('write') == 1
    assert json.loads((tmp_path / 'run/binding.json').read_text())['unguided_baseline']['development']['successes'] == 0
    new_actor = {**actor, 'checkpoint_sha256': '9'*64}
    student['rl_policy_binding']['native_identity'] = new_actor
    student['rl_policy_binding']['checkpoint_sha256'] = new_actor['checkpoint_sha256']
    student['native_state']['checkpoint_sha256'] = new_actor['checkpoint_sha256']
    with pytest.raises(ValueError, match='resume changed'):
        memetic.main([*common, '--baseline-only'])
    student['rl_policy_binding']['native_identity'] = actor
    student['rl_policy_binding']['checkpoint_sha256'] = actor['checkpoint_sha256']
    student['native_state']['checkpoint_sha256'] = actor['checkpoint_sha256']
    with pytest.raises(ValueError, match='resume changed'):
        memetic.main([*common, '--baseline-only', '--max-turns', '21'])
    monkeypatch.setattr(memetic.runner, 'default_system_prompt', lambda: 'changed-system')
    with pytest.raises(ValueError, match='resume changed'):
        memetic.main([*common, '--baseline-only'])


def test_cached_baseline_fitness_requires_exact_evaluation_receipt(tmp_path):
    from natlang_neuralese.common.jsonio import write_json_durable
    write_json_durable(tmp_path / 'inputs.json', {'student': 'current'})
    write_json_durable(tmp_path / 'unguided-dev-fit.json', {'scored': 4, 'successes': 2,
        'evaluation_inputs_sha256': 'stale'})
    args = Namespace(samples=2)
    with pytest.raises(ValueError, match='cached guidance fitness'):
        memetic.evaluate(args, tmp_path, 'unguided-dev', '', [case('a'), case('b')],
            base_system='exact-system', actor={}, server_info_sha256='f'*64)


def test_memetic_budget_capture_uses_exact_shared_prefix_verifier(tmp_path,monkeypatch):
    from natlang_neuralese.rl import task_rollouts
    packet=tmp_path/'packet.jsonl';packet.write_text(json.dumps(case('one'))+'\n')
    entry={'output':str(tmp_path/'empty.jsonl'),'key':'one','index':0,'jobs':str(tmp_path/'jobs')}
    (tmp_path/'queue.jsonl').write_text(json.dumps(entry)+'\n')
    (tmp_path/'preparation.json').write_text(json.dumps({'packet_sha256':hashlib.sha256(packet.read_bytes()).hexdigest()}))
    finish={'event':'finish','key':'one','resource_limit_reason':'model_request_budget'}
    (tmp_path/'journal.jsonl').write_text(json.dumps(finish)+'\n')
    actor={key:'actor-'+key for key in memetic.IDENTITY}
    prefix={'trajectory':[{'model_response':{'rl_sampling':{'policy':actor},
        'native_state_imports':[{'id':'nzs1_'+'a'*52}]}}]}
    calls=[]
    def verified(e,t,f,**kwargs):
        calls.append((e,t,f,kwargs));return prefix
    monkeypatch.setattr(task_rollouts,'_budget_terminated_prefix',verified)
    assert memetic.student_capture_reached(tmp_path,actor)
    assert calls[0][0]==entry and calls[0][1]['id']=='one' and calls[0][2]==finish
    assert calls[0][3]['packet_source_sha256']==hashlib.sha256(packet.read_bytes()).hexdigest()
    assert memetic.native_import_reached(tmp_path,'nzs1_'+'a'*52)
    prefix['trajectory'][0]['model_response']['rl_sampling']['policy']={**actor,'checkpoint_sha256':'wrong'}
    assert not memetic.student_capture_reached(tmp_path,actor)
    monkeypatch.setattr(task_rollouts,'_budget_terminated_prefix',lambda *a,**k:None)
    assert not memetic.student_capture_reached(tmp_path,actor)
    assert not memetic.native_import_reached(tmp_path,'nzs1_'+'a'*52)
