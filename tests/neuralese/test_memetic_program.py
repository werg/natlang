import copy
import json
import threading
from argparse import Namespace
import hashlib
from pathlib import Path

import pytest

from natlang_neuralese.rl import memetic_program as memetic
from natlang_neuralese.rl import memetic_batch
from natlang_neuralese.common import custody as world_custody


def case(ident, program='p', split='train'):
    return {'id': ident, 'program_id': program, 'split': split, 'semantics': {'inputs': {'invocation': ident}},
        'curriculum': {'reference': {'root': [['eval', {'code': 'return 1'}]]}}}


PROTECTED = world_custody.ProtectedWorlds.from_cases({'held': ('0' * 64, [
    {'id': 'held-0', 'split': 'test', 'source_groups': ['held-group'], 'semantics': {'inputs': {'held': 0}}}])})


def write_custody(dev, confirmation, out):
    rows = {'development': [json.loads(line) for line in dev.read_text().splitlines()],
            'confirmation': [json.loads(line) for line in confirmation.read_text().splitlines()]}
    out.write_text(json.dumps(world_custody.verify_slices(rows, protected=PROTECTED)))
    return out


def test_repeated_program_slices_need_their_custody_receipt(tmp_path):
    dev, confirmation = tmp_path / 'dev.jsonl', tmp_path / 'confirm.jsonl'
    dev.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('a', 'b')))
    confirmation.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('c', 'd')))
    receipt = write_custody(dev, confirmation, tmp_path / 'custody.json')
    left, right, bound = memetic.load_slices(dev, confirmation, 'p', receipt)
    assert [item['id'] for item in left + right] == ['a', 'b', 'c', 'd'] and bound['protected_sha256']
    confirmation.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('c', 'e')))
    with pytest.raises(ValueError, match='custody receipt'):
        memetic.load_slices(dev, confirmation, 'p', receipt)
    confirmation.write_text(''.join(json.dumps(case(i)) + '\n' for i in ('a', 'd')))
    with pytest.raises(ValueError, match='repeats|share related'):
        write_custody(dev, confirmation, tmp_path / 'other.json')
    confirmation.write_text(''.join(json.dumps(case(i, split='test')) + '\n' for i in ('c', 'd')))
    with pytest.raises(ValueError, match='split'):
        write_custody(dev, confirmation, tmp_path / 'other.json')


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
    manifest = memetic_batch.select(source, tmp_path / 'batch', protected=PROTECTED)
    assert manifest['world_components'] == 3
    assert len(manifest['jobs']) == 1 and manifest['jobs'][0]['ready']
    job = manifest['jobs'][0]
    dev, confirmation, _ = memetic.load_slices(Path(job['development']['path']),
        Path(job['confirmation']['path']), job['program_id'], Path(job['custody']['path']))
    assert len(dev) >= 2 and len(confirmation) >= 2
    assert {item['memetic_source']['world_bindings'][0] for item in dev}.isdisjoint(
        {item['memetic_source']['world_bindings'][0] for item in confirmation})
    assert all(item['memetic_source']['source_sha256'] == manifest['source_sha256'] for item in dev + confirmation)
    with pytest.raises(ValueError, match='resume changed'):
        extra = {**rows[0], 'id': 'new-case'}
        source.write_text(source.read_text() + json.dumps(extra) + '\n')
        memetic_batch.select(source, tmp_path / 'batch', protected=PROTECTED)


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
    monkeypatch.setattr(memetic_batch.custody.ProtectedWorlds, 'load', classmethod(lambda cls, *a, **k: PROTECTED))
    memetic_batch.main(['--source', str(source), '--out', str(tmp_path / 'batch'),
        '--server', 'http://student', '--model-id', 'lfm', '--baseline-only',
        '--runtime-continuation-from', str(tmp_path / 'frozen-runtime')])
    assert len(seen) == 1 and '--baseline-only' in seen[0] and '--author' not in seen[0] and '--custody' in seen[0]
    assert seen[0][seen[0].index('--runtime-continuation-from') + 1] == str(tmp_path / 'frozen-runtime')
    progress = json.loads((tmp_path / 'batch/progress.json').read_text())
    assert progress['results'][0]['status'] == 'baseline_complete'


def test_binding_requires_actual_native_id_and_oracle_gain(tmp_path):
    ident = 'nzs1_' + 'a' * 52
    assert memetic.state_guidance('Check the contract.', ident) == '\ue002' + ident + '\ue003'
    with pytest.raises(ValueError):
        memetic.state_guidance('Check.', 'nz1_' + 'a' * 52)
    actor = {key: 'actor-' + key for key in memetic.IDENTITY}
    turn = {'sampling': {'policy': actor}, 'native_state_imports': [{'id': ident}]}
    (tmp_path / 'rollouts.jsonl').write_text(json.dumps({'task_id': 'one', 'turns': [turn]}) + '\n')
    assert memetic.rollouts_reached(tmp_path, actor, ident)
    assert not memetic.rollouts_reached(tmp_path, actor, 'nzs1_' + 'b' * 52)
    assert not memetic.rollouts_reached(tmp_path, {**actor, 'checkpoint_sha256': 'other'})


def test_author_requires_exact_self_hosted_source_receipt(tmp_path):
    from natlang_neuralese.common import memetic_teacher_ferry as ferry
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


def test_author_attempt_context_is_retry_stable_and_generation_distinct(monkeypatch):
    ferry = memetic.teacher_ferry
    prompts = []
    monkeypatch.setattr(memetic.runner, 'author_call', lambda url, model, prompt: (
        prompts.append(prompt) or {'text': '<guidance>Use the typed contract.</guidance>', 'custody': {}}))
    args = Namespace(author='http://127.0.0.1:8766', author_ferry='http://127.0.0.1:8766',
                     author_model=ferry.TEACHER_SERVE_ALIAS, program_id='exact-program')
    context0 = {'schema': 'natlang.memetic-guidance-author-attempt/1', 'program_id': 'exact-program',
                'phase': 'candidate', 'generation': 0, 'author_provenance_sha256': 'a' * 64}
    parent = {'guidance': 'Check the typed result.'}
    memetic.runner.propose_guidance(args, parent, '- development failure', attempt_context=context0)
    memetic.runner.propose_guidance(args, parent, '- development failure', attempt_context=context0)
    context1 = {**context0, 'generation': 1}
    memetic.runner.propose_guidance(args, parent, '- development failure', attempt_context=context1)
    bodies = [memetic.runner.author_request_body(ferry.TEACHER_SERVE_ALIAS, prompt) for prompt in prompts]
    request_ids = [ferry.request_id(ferry.TEACHER_SERVE_ALIAS, body) for body in bodies]
    assert request_ids[0] == request_ids[1], 'same attempt body must retain the ferry idempotency key across retries'
    assert request_ids[2] != request_ids[0], 'a distinct generation must produce a distinct exact-body request ID'
    assert 'exact-program' in prompts[0] and 'generation":0' in prompts[0]
    assert 'generation":1' in prompts[2]


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
    engine = Engine(port, tokenizer, TensorStore(), 'native-state/' + port.config.model_type, shared_model=model,
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
        args = Namespace(samples=2, shards=1, case_seconds=40, max_model_requests=2, max_turns=2, seed_base=7,
            server=url)
        fit = memetic.evaluate(args, tmp_path, 'actual', memetic.state_guidance('guide', written['id']), [task],
            base_system=memetic.runner.default_system_prompt(), actor=identity,
            server_info_sha256=memetic.canonical_json_sha256_hex(info), evaluation_inputs_sha256='e' * 64)
        assert fit['rollouts'] == 2 and fit['native_import_reached_model'] is True
        assert memetic.rollouts_reached(tmp_path / 'evals/actual', identity, written['id'])
        # The production collector must retain sampled decision captures as policy turns, including their
        # native input state and request accounting, rather than falling back to a generated tool turn.
        decision_task = copy.deepcopy(task)
        decision_task['semantics']['expected'] = 'yes'
        decision_task['semantics']['files']['task/verdict.nl'] = (
            '---\nargs: {}\nreturns: "yes" | "no"\nreadout: decision\n---\nChoose yes or no.\n')
        decision_task['curriculum']['reference']['root'][0][1]['code'] = (
            'return (await decide(verdict)).value;')
        decision_args = Namespace(**{**vars(args), 'samples': 2})
        decision_fit = memetic.evaluate(decision_args, tmp_path, 'decision', memetic.state_guidance('guide', written['id']),
            [decision_task], base_system=memetic.runner.default_system_prompt(), actor=identity,
            server_info_sha256=memetic.canonical_json_sha256_hex(info), evaluation_inputs_sha256='e' * 64)
        assert decision_fit['rollouts'] == 2
        from natlang_neuralese.common.jsonio import iter_jsonl_rows
        decision_rows = [row for _, row in iter_jsonl_rows(tmp_path / 'evals/decision/rollouts.jsonl')]
        decisions = [turn['sampling'] for row in decision_rows for turn in row['turns']
                     if (turn.get('sampling') or {}).get('schema') == 'natlang.native-decision-capture/1']
        assert len(decisions) == 2 and all(value['decision_sample']['temperature'] > 0 for value in decisions)
        development, confirmation = tmp_path / 'dev.jsonl', tmp_path / 'confirmation.jsonl'
        for path, ids in ((development, ('a', 'b')), (confirmation, ('c', 'd'))):
            path.write_text(''.join(json.dumps({**task, 'id': name, 'curriculum': {**task['curriculum'],
                'evidence': {'world': [name]}}}) + '\n' for name in ids))
        receipt = write_custody(development, confirmation, tmp_path / 'custody.json')
        baseline = memetic.main(['--program-id', 'p', '--development', str(development),
            '--confirmation', str(confirmation), '--custody', str(receipt), '--out', str(tmp_path / 'memetic-baseline'),
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
    receipt_id = 'mtf1_' + '7' * 64
    receipt = {'schema': memetic.teacher_ferry.RECEIPT_SCHEMA, 'request_id': receipt_id,
        'request_sha256': '8' * 64, 'response_sha256': '9' * 64, 'response_bytes': 1,
        'status': 'complete', 'hmac_sha256': 'a' * 64}
    custody = {'schema': 'natlang.memetic-author-call-custody/1', 'request_id': receipt_id,
        'request_sha256': '8' * 64, 'response_sha256': '9' * 64,
        'ferry_receipt_sha256': 'b' * 64, 'ferry_receipt': receipt}
    authored_calls = []
    def fake_propose(args, parent, digest, *, attempt_context, prior_rejection=None):
        authored_calls.append({'prompt_input': digest, 'attempt': attempt_context, 'parent': parent,
            'prior_rejection': prior_rejection})
        return {'text': 'Teacher-authored: check the contract and verify the result.', 'author_call': custody}
    monkeypatch.setattr(memetic.runner, 'propose_guidance', fake_propose)
    monkeypatch.setattr(memetic.runner, 'failure_digest', lambda fit: 'DEV failure evidence')
    fail_candidate_write = {'enabled': True}
    def fake_write(*args):
        calls.append(('write', '', actor))
        if len([call for call in calls if call[0] == 'write']) == 2 and fail_candidate_write['enabled']:
            raise RuntimeError('simulated candidate write interruption')
        return written
    monkeypatch.setattr(memetic, 'write_state', fake_write)
    monkeypatch.setattr(memetic, 'archive_state', lambda *args: {'path': '/tmp/state', 'sha256': 'f'*64, 'bytes': 1})
    development, confirmation = tmp_path / 'dev.jsonl', tmp_path / 'confirmation.jsonl'
    development.write_text(''.join(json.dumps(case(index)) + '\n' for index in ('a', 'b')))
    confirmation.write_text(''.join(json.dumps(case(index)) + '\n' for index in ('c', 'd')))
    receipt_path = write_custody(development, confirmation, tmp_path / 'custody.json')
    common = ['--program-id', 'p', '--development', str(development), '--confirmation', str(confirmation),
              '--custody', str(receipt_path),
              '--out', str(tmp_path / 'run'), '--server', 'http://student', '--model-id', 'lfm',
              '--samples', '2', '--generations', '1']
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
    with pytest.raises(RuntimeError, match='candidate write interruption'):
        memetic.main([*common, *author])
    assert [item['attempt']['phase'] for item in authored_calls] == ['initial-seed', 'candidate']
    assert all(item['prompt_input'] == 'DEV failure evidence' for item in authored_calls)
    assert 'confirmation' not in json.dumps(authored_calls)
    candidate_pending = json.loads((tmp_path / 'run/candidate-000-pending.json').read_text())
    assert candidate_pending['author_call']['request_id'] == receipt_id
    fail_candidate_write['enabled'] = False
    result = memetic.main([*common, *author])
    assert result['status'] == 'qualified'
    # Selection used development only; confirmation was measured once, for the selected state.
    names = [name for name, _, _ in calls]
    assert names.count('selected-confirm-' + written['id'][5:21]) == 1
    assert not any(name.endswith('-confirm') and name.startswith(('seed', 'candidate')) for name in names)
    assert result['acceptance']['selected']['confirmation'] == {'successes': 2, 'scored': 4}
    ledger = json.loads((tmp_path / 'confirmation-ledger.json').read_text())
    assert [item['artifact'] for item in ledger['spent'].values()] == [written['id']]
    assert [name for name, _, _ in calls].count('write') == 3
    assert len(authored_calls) == 2, 'resuming the candidate must retain its pending teacher answer and request ID'
    state = json.loads((tmp_path / 'run/state.json').read_text())
    assert state['best']['author_call']['request_id'] == receipt_id
    assert state['history'][0]['candidate']['author_call']['request_id'] == receipt_id
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


def test_runtime_continuation_preserves_origin_and_requires_exact_nonruntime_inputs(tmp_path, monkeypatch):
    out = tmp_path / 'run'; old_root = tmp_path / 'old-runtime'; old_root.mkdir()
    actor = {'checkpoint_sha256': 'a' * 64}
    old_runtime = {'schema': 'runtime', 'typescript_sha256': 'b' * 64, 'typescript_files': 8,
                   'python': {'sha256': 'c' * 64}, 'queue_worker_sha256': 'd' * 64,
                   'node_version': 'v22', 'python_version': '3.12'}
    new_runtime = {**old_runtime, 'python': {'sha256': 'e' * 64}}
    current_root = Path(memetic.__file__).resolve().parents[4]
    monkeypatch.setattr(memetic, 'local_runtime_binding',
        lambda root=None: new_runtime if root is None or Path(root) == current_root else old_runtime)
    monkeypatch.setattr(memetic, '_definition_bindings', lambda root: {'collector_core': 'same'})
    original = {'program_id': 'p', 'student': actor, 'sampling': {'samples': 2},
                'evaluator_runtime': old_runtime}
    old_sha, _ = memetic.evaluator_inputs(out, original, actor, None)
    old_bytes = (out / 'inputs.json').read_bytes()
    updated = {**original, 'evaluator_runtime': new_runtime}
    new_sha, receipt = memetic.evaluator_inputs(out, updated, actor, old_root)
    assert new_sha != old_sha
    assert (out / 'inputs.json').read_bytes() == old_bytes
    assert receipt['origin_inputs_sha256'] == old_sha
    assert receipt['current_inputs'] == updated
    assert receipt['history'][0]['from_inputs'] == original
    assert receipt['history'][0]['to_inputs'] == updated

    changed_actor = {**updated, 'student': {**actor, 'checkpoint_sha256': 'e' * 64}}
    with pytest.raises(ValueError, match='resume changed actor'):
        memetic.evaluator_inputs(out, changed_actor, changed_actor['student'], old_root)


def test_runtime_continuation_rejects_changed_collector_semantics(tmp_path, monkeypatch):
    out = tmp_path / 'run'; old_root = tmp_path / 'old-runtime'; old_root.mkdir()
    actor = {'checkpoint_sha256': 'a' * 64}
    old_runtime = {'schema': 'runtime', 'typescript_sha256': 'b' * 64, 'typescript_files': 8,
                   'python': {'sha256': 'c' * 64}, 'queue_worker_sha256': 'd' * 64,
                   'node_version': 'v22', 'python_version': '3.12'}
    new_runtime = {**old_runtime, 'python': {'sha256': 'e' * 64}}
    current_root = Path(memetic.__file__).resolve().parents[4]
    monkeypatch.setattr(memetic, 'local_runtime_binding',
        lambda root=None: new_runtime if root is None or Path(root) == current_root else old_runtime)
    monkeypatch.setattr(memetic, '_definition_bindings',
        lambda root: {'collector_core': 'changed' if Path(root) == current_root else 'old'})
    original = {'program_id': 'p', 'student': actor, 'sampling': {'samples': 2},
                'evaluator_runtime': old_runtime}
    memetic.evaluator_inputs(out, original, actor, None)
    updated = {**original, 'evaluator_runtime': new_runtime}
    with pytest.raises(ValueError, match='collector or policy-scoring mechanism'):
        memetic.evaluator_inputs(out, updated, actor, old_root)

    monkeypatch.setattr(memetic, '_definition_bindings', lambda root: {'collector_core': 'exact'})
    changed_ts_runtime = {**new_runtime, 'typescript_sha256': 'f' * 64}
    monkeypatch.setattr(memetic, 'local_runtime_binding',
        lambda root=None: changed_ts_runtime if root is None or Path(root) == current_root else old_runtime)
    changed_ts = {**original, 'evaluator_runtime': changed_ts_runtime}
    with pytest.raises(ValueError, match='compiled TypeScript runtime'):
        memetic.evaluator_inputs(out, changed_ts, actor, old_root)


@pytest.mark.parametrize('budget_empty', [False, True])
def test_runtime_scorer_replay_uses_exact_saved_rows_without_rewriting_them(tmp_path, budget_empty):
    from natlang_neuralese.rl.task_rollouts import authored_root_row, build_queue
    evaluation = tmp_path / 'eval'; evaluation.mkdir()
    packet = evaluation / 'packet.jsonl'
    packet.write_text(json.dumps(authored_root_row(case('one'))) + '\n')
    packet_sha = hashlib.sha256(packet.read_bytes()).hexdigest()
    (evaluation / 'preparation.json').write_text(json.dumps({
        'schema': 'natlang.task-rollout-preparation/1', 'root_mode': 'authored', 'packet_sha256': packet_sha}) + '\n')
    entries = build_queue(packet, 1, 2, evaluation, temperature=0.7, case_seconds=60,
                          max_model_requests=8, max_turns=8, seed_base=5)
    finish_rows = []
    for entry in entries:
        output = Path(entry['output']); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('' if budget_empty else json.dumps({'outcome': {'status': 'done', 'accepted': True}}) + '\n')
        finish_rows.append({'event': 'finish', 'key': entry['key'], 'status': 'failed' if budget_empty else 'complete',
                            **({'resource_limit_reason': 'model_request_budget'} if budget_empty else {})})
    (evaluation / 'queue.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in entries))
    (evaluation / 'journal.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in finish_rows))
    before = {path: path.read_bytes() for path in [packet, evaluation / 'queue.jsonl', *[Path(row['output']) for row in entries]]}
    root = Path(memetic.__file__).resolve().parents[4]
    report = memetic._score_on_runtime(root, evaluation)
    assert report['scored'] == 2 and report['successes'] == (0 if budget_empty else 2)
    assert all(path.read_bytes() == payload for path, payload in before.items())
    assert not (evaluation / 'score.json').exists() and not (evaluation / 'rewards.jsonl').exists()


def test_runtime_continuation_reuses_only_score_equivalent_exact_actor_fits(tmp_path, monkeypatch):
    import shutil
    from natlang_neuralese.rl.task_rollouts import authored_root_row, build_queue
    source_root = Path(memetic.__file__).resolve().parents[4]
    old_root = tmp_path / 'old-runtime'; old_root.mkdir()
    shutil.copytree(source_root / 'training/neuralese', old_root / 'training/neuralese')
    actor = {key: f'actor-{key}' for key in memetic.IDENTITY}
    out = tmp_path / 'program'; target = out / 'evals/unguided-dev'; target.mkdir(parents=True)
    packet = target / 'packet.jsonl'; packet.write_text(json.dumps(authored_root_row(case('one'))) + '\n')
    packet_sha = hashlib.sha256(packet.read_bytes()).hexdigest()
    (target / 'preparation.json').write_text(json.dumps({
        'schema': 'natlang.task-rollout-preparation/1', 'root_mode': 'authored', 'packet_sha256': packet_sha}) + '\n')
    entries = build_queue(packet, 1, 2, target, temperature=0.7, case_seconds=60,
                          max_model_requests=8, max_turns=8, seed_base=7)
    finishes = []
    for entry in entries:
        output = Path(entry['output']); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({'trajectory': [{'model_response': {'rl_sampling': {'policy': actor}}}],
                                      'outcome': {'status': 'done', 'accepted': True}}) + '\n')
        finishes.append({'event': 'finish', 'key': entry['key'], 'status': 'complete'})
    (target / 'queue.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in entries))
    (target / 'journal.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in finishes))
    (target / 'rollouts.jsonl').write_text(''.join(json.dumps({'task_id': 'one', 'turns': [
        {'sampling': {'policy': actor}}]}) + '\n' for _ in entries))
    old_sha = '1' * 64
    fit_path = out / 'unguided-dev-fit.json'
    fit_path.write_text(json.dumps({'evaluation_inputs_sha256': old_sha, 'actor_capture_reached_model': True,
                                    'scored': 2, 'successes': 2}) + '\n')
    runtime_old = {'typescript_sha256': 'a' * 64, 'typescript_files': 8, 'python': {'sha256': 'b' * 64},
                   'queue_worker_sha256': 'c' * 64, 'node_version': 'v22', 'python_version': '3.12'}
    runtime_new = {**runtime_old, 'python': {'sha256': 'd' * 64}}
    monkeypatch.setattr(memetic, 'local_runtime_binding', lambda root=None: runtime_old if Path(root) == old_root else runtime_new)
    monkeypatch.setattr(memetic, '_definition_bindings', lambda root: {'collector_core': 'exact'})
    rows = memetic._verify_existing_fit_outputs(out, actor, old_root, source_root, {old_sha})
    assert len(rows) == 1 and rows[0]['name'] == 'unguided-dev'

    old_score = old_root / 'training/neuralese/natlang_neuralese/rl/task_rollouts.py'
    text = old_score.read_text()
    needle = '"successes": sum(r["reward"] == 1.0 for r in rollouts),'
    assert needle in text
    old_score.write_text(text.replace(needle, '"successes": 0,'))
    with pytest.raises(ValueError, match='changes under the current scorer'):
        memetic._verify_existing_fit_outputs(out, actor, old_root, source_root, {old_sha})
