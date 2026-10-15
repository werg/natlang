import copy
import json
from pathlib import Path

import pytest

from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as converter


def _row():
    return {'id': 'execution-a', 'version': 'natlang.teacher_trajectory.native/1',
        'task': {'program_ir': {'version': 'natlang.program/2', 'split': 'train',
                              'id': 'program-a', 'source_groups': ['train-g']}},
        'outcome': {'status': 'done', 'accepted': True, 'oracle': {'accepted': True},
                    'checks': {'exact_result': True}},
        'provenance': {'model': 'pennyroyal'},
        'execution_identity': {'execution_run_id': 'run-a'},
        'trajectory': [{'phase': 'action', 'assistant': {'content': 'correct'}}]}


def test_successful_parent_does_not_approve_rejected_or_seeded_turns():
    row = _row()
    turn = {'invocation_id': 'child-a', 'request_sha256': 'c' * 64, 'raw_response_sha256': 'a' * 64,
            'assistant': {'content': ''}, 'model_response': {'raw_calls': []}}
    evidence = {'approved': False, 'reason': 'decision contains a failed proposal',
                'invocation_id': 'child-a', 'raw_response_sha256': 'a' * 64}
    assert converter.admitted_teacher_turn(row, 0, turn, {(row['id'], 0): evidence}) == (False, evidence)
    # Even a broken external approval cannot turn absent provider output into EOS supervision.
    good = {**evidence, 'approved': True}
    allowed, rejected = converter.admitted_teacher_turn(row, 0, turn, {(row['id'], 0): good})
    assert not allowed and 'no original provider action' in rejected['reason']
    turn['model_response']['raw_calls'] = [{'id': 'actual-call'}]
    assert converter.admitted_teacher_turn(row, 0, turn, {(row['id'], 0): good}) == (True, good)
    turn['raw_response_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='original invocation/response'):
        converter.admitted_teacher_turn(row, 0, turn, {(row['id'], 0): good})
    assert converter.admitted_teacher_turn(row, 1, turn, {})[0] is False


def test_shared_materializer_cache_is_bound_to_input_runtime_and_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    repo = tmp_path / 'repo'
    script = repo / 'scripts/neuralese_data/converter.py'
    script.parent.mkdir(parents=True)
    script.write_text('fixture')
    monkeypatch.setattr(converter, '__file__', str(script))
    for relative in ('ts-host/scripts/materialize-native-teacher.mjs',
                     'ts-host/scripts/jsonl-stream.mjs', 'ts-host/dist/teacher/native-materializer.js'):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture materializer asset')
    source = tmp_path / 'source.jsonl'
    source.write_text('{}\n')
    calls = []
    materialized = {'id': 'execution-a:decision:0000',
        'source_ref': {'trajectory_id': 'execution-a', 'invocation_id': 'child-a'},
        'decision': {'index': 0, 'training_approved': True, 'failed_action': False,
                     'source_raw_response_sha256': 'a' * 64},
        'training_admission': {'approved': True}}
    def execute(argv, **kwargs):
        calls.append(argv)
        Path(argv[3]).write_text(json.dumps(materialized) + '\n')
        return SimpleNamespace(returncode=0, stdout=json.dumps({'output': argv[3], 'eligible_decisions': 1}), stderr='')
    monkeypatch.setattr(converter.subprocess, 'run', execute)
    digest = converter.sha256_file(source)
    first, proof = converter.teacher_decision_admissions(source, digest)
    assert first[('execution-a', 0)]['approved'] is True
    assert converter.teacher_decision_admissions(source, digest) == (first, proof)
    assert len(calls) == 1
    source.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='immutable SHA-256'):
        converter.teacher_decision_admissions(source, digest)
    source.write_text('{}\n')
    output = Path(proof['path']).parent / 'decisions.jsonl'
    original = output.read_text()
    output.write_text(original + '{}\n')
    with pytest.raises(ValueError, match='exact source/runtime binding'):
        converter.teacher_decision_admissions(source, digest)
    output.write_text(original)
    (repo / 'ts-host/dist/teacher/native-materializer.js').write_text('new actual policy')
    _, updated = converter.teacher_decision_admissions(source, digest)
    assert len(calls) == 2 and updated['path'] != proof['path']


@pytest.mark.parametrize('scope', ['held', 'quality', 'pending'])
def test_current_protection_excludes_even_previously_selected_success(monkeypatch, scope):
    monkeypatch.setattr(converter.source_review, 'pending_family_hold_reason', lambda _: None)
    binding = {'selected_ids': {'program-a'}, 'train_groups': {'train-g'}, 'held_groups': set(),
        'source_files': {'execution-a': {}},
        'current': {key: {'train-g'} if key == scope else set() for key in ('held', 'quality', 'pending')}}
    with pytest.raises(ValueError, match='current held/quality/source-review'):
        converter.validate_teacher_row(_row(), binding)


def test_original_response_binding_rejects_changed_combined_action(tmp_path):
    row = _row()
    path = tmp_path / 'response.jsonl'
    path.write_text(json.dumps(row) + '\n')
    binding = {'path': str(path), 'sha256': converter.sha256_file(path)}
    assert converter.verify_original_row(row, binding) == path
    changed = copy.deepcopy(row)
    changed['trajectory'][0]['assistant']['content'] = 'different but claimed success'
    with pytest.raises(ValueError, match='differs from its original response'):
        converter.verify_original_row(changed, binding)
    path.write_text(json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='artifact changed'):
        converter.verify_original_row(row, binding)


def test_current_projection_verifies_content_and_live_source_review(tmp_path, monkeypatch):
    live = tmp_path / 'live'
    live.mkdir()
    (live / 'holds.jsonl').write_text('')
    monkeypatch.setattr(converter.source_review, 'REGISTRY', live)
    monkeypatch.setattr(converter.source_review, 'pending_unpinned_source_groups', lambda: {'pending-g'})
    files = []
    for name, group in [('held-union/full/groups.jsonl', 'held-g'), ('quality-groups.jsonl', 'quality-g')]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'source_groups': [group]}) + '\n')
        files.append({'path': name, 'bytes': path.stat().st_size, 'sha256': converter.sha256_file(path)})
    manifest = {'schema': 'natlang.training-protection-projection/1', 'files': files,
        'held_groups': 1, 'quality_groups': 1,
        'source_review_sha256': converter.sha256_file(live / 'holds.jsonl')}
    path = tmp_path / 'projection.json'
    path.write_text(json.dumps(manifest))
    result = converter.current_protection(path)
    assert result['held'] == {'held-g'} and result['quality'] == {'quality-g'}
    assert result['pending'] == {'pending-g'}
    (live / 'holds.jsonl').write_text('changed review\n')
    with pytest.raises(ValueError, match='current source reviews'):
        converter.current_protection(path)
    (live / 'holds.jsonl').write_text('')
    (tmp_path / 'quality-groups.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='protection artifact differs'):
        converter.current_protection(path)


def test_historical_batches_are_bound_to_their_own_packet_not_future_queue(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(converter, 'current_protection', lambda _: {'proof': {}})
    def put(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')
        return path
    source = put('native-trajectories.jsonl', {'id': 'execution-a'})
    cases = put('selected-cases.jsonl', {'id': 'program-a'})
    selection = {'schema': 'natlang.native-teacher-task-selection/1', 'status': 'passed',
        'output_sha256': converter.sha256_file(cases), 'selected_ids': ['program-a'],
        'train_groups': ['train-g'], 'held_groups': []}
    put('selection-proof.json', selection)
    teacher = {'teacher_role': 'independent stronger open teacher; not either student/onpolicy'}
    put('penny-server-binding.json', teacher)
    put('evidence-source-v2.json', {'cases_sha256': 'f' * 64, 'scope': 'new future queue'})
    put('global-protection-join.json', {'schema': 'natlang.native-teacher-global-protection-join/1',
        'status': 'passed', 'selected_packet_sha256': selection['output_sha256'],
        'held_group_intersections': [], 'quality_group_intersections': []})
    collection = {'schema': 'natlang.sandbox-strong-teacher-collection/1',
        'source_cases_sha256': selection['output_sha256'], 'protected_selection': selection,
        'teacher': teacher}
    path = put('batch-bindings/batch-0000/teacher-source.json', collection)
    item = {'path': str(path), 'source': '/work/generation/batch-0000/teacher-source.json',
            'bytes': path.stat().st_size, 'sha256': converter.sha256_file(path)}
    transfer = {'schema': 'natlang.verified-teacher-trajectory-transfer/1', 'training_admission': False,
        'combined': {'path': str(source), 'bytes': source.stat().st_size, 'sha256': converter.sha256_file(source)},
        'files': [item]}
    put('transfer-receipt.json', transfer)
    args = SimpleNamespace(source=source, protection_manifest=tmp_path / 'current.json')
    assert converter.source_binding(args)['collections'][item['source']]['sha256'] == item['sha256']
    duplicate = put('duplicate-copy/teacher-source.json', collection)
    transfer['files'].append({**item, 'path': str(duplicate)})
    put('transfer-receipt.json', transfer)
    assert converter.source_binding(args)['collections'][item['source']]['sha256'] == item['sha256']
    collection['source_cases_sha256'] = 'e' * 64
    put('batch-bindings/batch-0000/teacher-source.json', collection)
    item.update(bytes=path.stat().st_size, sha256=converter.sha256_file(path))
    put('transfer-receipt.json', transfer)
    with pytest.raises(ValueError, match='collection receipt differs'):
        converter.source_binding(args)


def test_content_dedup_uses_actual_tokens_and_weights_not_execution_identity(tmp_path):
    graph = {'schema': 'natlang.native-graph/1', 'id': 'old', 'split': 'train',
        'source_groups': ['train-g'], 'tokenizer_fingerprint': 'tokenizer-a',
        'nodes': [{'name': 'text', 'token_ids': [1, 2], 'target_weights': [0., 1.]}]}
    # Use the shared graph schema instead of guessing a parallel schema.
    from natlang_neuralese.data.native_graph import SCHEMA
    graph['schema'] = SCHEMA
    file = tmp_path / 'prior.jsonl'
    file.write_text(json.dumps(graph) + '\n')
    seen, bindings = converter.seed_content_fingerprints([file], 'tokenizer-a')
    repeated = {**graph, 'id': 'new-execution', 'conversion': {'run_id': 'different'}}
    assert converter.graph_content_sha256(repeated) in seen
    changed = copy.deepcopy(repeated)
    changed['nodes'][0]['target_weights'][1] = .05
    assert converter.graph_content_sha256(changed) not in seen
    assert bindings[0]['sha256'] == converter.sha256_file(file)
    graph['split'] = 'test'
    file.write_text(json.dumps(graph) + '\n')
    with pytest.raises(ValueError, match='train graphs'):
        converter.seed_content_fingerprints([file], 'tokenizer-a')


def test_unresolved_state_turn_is_explicitly_omitted_without_placeholder(monkeypatch):
    block = {'type': 'neuralese', 'id': 'nz1_exact-source-state'}
    def render(_tokenizer, _context, target, _tools, **_kwargs):
        error = ('ordinary gold text target contains unresolved neuralese blocks'
                 if list(converter.unresolved_neuralese_references(target)) else
                 'ordinary gold text prompt contains unresolved neuralese blocks')
        raise ValueError(error)
    monkeypatch.setattr(converter, 'native_gold_packet', render)
    packet, omission = converter.render_teacher_packet(None,
        [{'role': 'user', 'content': [{'type': 'text', 'text': 'Read this: '}, block]}],
        {'role': 'assistant', 'content': 'answer'}, [])
    assert packet is None
    assert omission == {
        'reason': 'full ordinary-text conversion requires unresolved typed Neuralese payloads; '
                  'retain this action for source-bound native graph conversion',
        'unresolved_scope': 'prompt', 'unresolved_block_ids': ['nz1_exact-source-state']}
    target_packet, target_omission = converter.render_teacher_packet(None, [],
        {'role': 'assistant', 'content': [block]}, [])
    assert target_packet is None
    assert target_omission['unresolved_scope'] == 'target'
    assert target_omission['unresolved_block_ids'] == ['nz1_exact-source-state']
    monkeypatch.setattr(converter, 'native_gold_packet', lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError('assistant target does not continue the serving generation prompt')))
    with pytest.raises(ValueError, match='does not continue'):
        converter.render_teacher_packet(None, [], {'role': 'assistant', 'content': 'answer'}, [])


def test_authenticated_text_emulation_restores_exact_provider_input_and_read_tool():
    import hashlib
    body = 'change_ai_name'
    block_id = 'nz1_exact-source-state'
    block = {'id': block_id, 'type': 'Neuralese<string>', 'body': body,
        'body_sha256': hashlib.sha256(body.encode()).hexdigest(), 'learned_vectors': False}
    label = (f'[[Neuralese text block id={block_id} type=Neuralese<string>; '
             f'exact JSON string body={json.dumps(body, ensure_ascii=False, separators=(",", ":"))}]]')
    context = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Read: '},
        {'type': 'neuralese', 'id': block_id}]}]
    rendered = [{'role': 'system', 'content': 'captured read instruction'},
        {'role': 'user', 'content': 'Read: ' + label}]
    tools = [{'type': 'function', 'function': {'name': 'return_result'}},
        {'type': 'function', 'function': {'name': 'eval'}}]
    transport = {'version': 'text-marker-standin/2', 'learned_vectors': False,
        'qualification_certificate': False, 'training_admission': False,
        'raw_request_sha256': 'a' * 64, 'rendered_request_sha256': 'b' * 64,
        'rendered_messages': rendered, 'expanded_input_blocks': [block]}
    turn = {'model_response': {'transport_provenance': transport}}
    messages, actual_tools, proof, omission = converter.captured_text_emulation_context(turn, context, tools)
    assert omission is None and messages == rendered and actual_tools == tools
    assert proof['schema'] == 'natlang.captured-text-neuralese-input/1'
    assert proof['expanded_blocks'] == [{'id': block_id, 'type': block['type'],
        'body_sha256': block['body_sha256'], 'occurrences': 1}]
    read_body = {'id': 'nz1_read-source', 'type': 'Neuralese<(v: Neuralese<unknown>) => unknown>',
        'body': 'read implementation', 'body_sha256': hashlib.sha256(b'read implementation').hexdigest(),
        'learned_vectors': False}
    read_template = {'schema': 'natlang.text-template-readout/1', 'call': 'return_result',
        'value': 'decode', 'read_body_id': read_body['id'], 'read_source_sha256': read_body['body_sha256'],
        'learned_vectors': False, 'qualification_certificate': False, 'training_admission': False}
    template_transport = {**transport, 'expanded_input_blocks': [block, read_body],
        'text_template_readout': read_template}
    read_turn = {'model_response': {'transport_provenance': template_transport}}
    messages, actual_tools, proof, omission = converter.captured_text_emulation_context(
        read_turn, context, tools)
    assert omission is None and messages == rendered
    assert actual_tools == tools[:1]
    damaged = {**template_transport, 'expanded_input_blocks': [{**block, 'body_sha256': '0' * 64}, read_body]}
    _, _, _, omission = converter.captured_text_emulation_context(
        {'model_response': {'transport_provenance': damaged}}, context, tools)
    assert omission is not None and 'digest mismatch' in omission['reason']


@pytest.mark.parametrize('code_selection', [False, 'list', 'count'])
def test_each_teacher_batch_uses_its_own_exact_program_packet(tmp_path, monkeypatch, code_selection):
    from types import SimpleNamespace
    monkeypatch.setattr(converter, 'current_protection', lambda _: {
        'held': set(), 'quality': set(), 'pending': set(), 'proof': {}})
    monkeypatch.setattr(converter.source_review, 'pending_family_hold_reason', lambda _: None)
    def put(name, value):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value)+'\n')
        return path
    teacher = {'teacher_role': 'independent stronger open teacher; not either student/onpolicy'}
    put('penny-server-binding.json', teacher)
    put('evidence-source-v2.json', {'scope': 'mutable future queue is irrelevant'})
    rows, files, batches = [], [], []
    for index in range(2):
        row = copy.deepcopy(_row())
        row['id'] = f'execution-{index}'
        program = row['task']['program_ir']
        program.update(id=f'program-{index}', source_groups=[f'train-{index}'])
        rows.append(row)
        base = f'batches/{index}'
        cases = put(base+'/selected-cases.jsonl', program)
        selection = {'schema': 'natlang.native-teacher-task-selection/1', 'status': 'passed',
            'output_sha256': converter.sha256_file(cases), 'selected_ids': [program['id']],
            'train_groups': program['source_groups'], 'held_groups': []}
        packet_sha = converter.sha256_file(cases)
        if code_selection:
            selection = {'schema': 'natlang.strong-teacher-code-source-selection/1',
                'cases_sha256': packet_sha, 'cases': 1, 'held_and_pending_intersections': 0,
                'training_admission': False, 'selected_source_groups': (1 if code_selection == 'count'
                                                                    else program['source_groups'])}
        selected = put(base+'/selection-proof.json', selection)
        protection = put(base+'/global-protection-join.json', {
            'schema': 'natlang.native-teacher-global-protection-join/1', 'status': 'passed',
            'selected_packet_sha256': packet_sha,
            'held_group_intersections': [], 'quality_group_intersections': []})
        collection_source = f'/work/generation/batch-{index:04d}/teacher-source.json'
        collection = put(base+'/teacher-source.json', {
            'schema': 'natlang.sandbox-strong-teacher-collection/1',
            'source_cases_sha256': packet_sha, 'protected_selection': selection,
            'teacher': teacher})
        files.append({'path': str(collection), 'source': collection_source,
            'bytes': collection.stat().st_size, 'sha256': converter.sha256_file(collection)})
        response = put(base+'/response.jsonl', row)
        files.append({'row_id': row['id'], 'path': str(response),
            'source': f'/work/generation/batch-{index:04d}/results/program/response.jsonl',
            'bytes': response.stat().st_size, 'sha256': converter.sha256_file(response)})
        batches.append({'collection_source': collection_source, **{name: {
            'path': str(path.relative_to(tmp_path)), 'sha256': converter.sha256_file(path)}
            for name, path in [('selection', selected), ('selected_cases', cases),
                               ('global_protection_join', protection)]}})
    source = tmp_path/'native-trajectories.jsonl'
    source.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    put('transfer-receipt.json', {'schema': 'natlang.verified-teacher-trajectory-transfer/1',
        'training_admission': False, 'files': files,
        'combined': {'path': str(source), 'bytes': source.stat().st_size,
                     'sha256': converter.sha256_file(source)}})
    mapping = {'schema': 'natlang.teacher-batch-source-bindings/1', 'batches': batches}
    put('batch-source-bindings.json', mapping)
    args = SimpleNamespace(source=source, protection_manifest=tmp_path/'current.json')
    # New batch snapshots need no irrelevant mutable future-queue boilerplate.
    (tmp_path/'evidence-source-v2.json').unlink()
    (tmp_path/'penny-server-binding.json').unlink()
    binding = converter.source_binding(args)
    assert binding['teacher'] == teacher
    for row in rows:
        assert converter.validate_teacher_row(row, binding)[0] == row['task']['program_ir']
    # Global membership in another admitted packet cannot retarget a response.
    swapped = copy.deepcopy(rows[0])
    swapped['task']['program_ir'] = rows[1]['task']['program_ir']
    with pytest.raises(ValueError, match='actual batch selected packet'):
        converter.validate_teacher_row(swapped, binding)
    changed = copy.deepcopy(rows[0])
    changed['task']['program_ir']['semantics'] = {'rewritten': True}
    with pytest.raises(ValueError, match='actual batch selected packet'):
        converter.validate_teacher_row(changed, binding)
    batches[0]['selected_cases'] = batches[1]['selected_cases']
    put('batch-source-bindings.json', mapping)
    with pytest.raises(ValueError, match='selected task packet differs'):
        converter.source_binding(args)


@pytest.mark.parametrize('field,value', [
    ('invocation_id', None), ('invocation_id', ''), ('invocation_id', '  '),
    ('request_sha256', None), ('request_sha256', 'c' * 63),
    ('raw_response_sha256', None), ('raw_response_sha256', 'A' * 64),
    ('raw_response_sha256', 'g' * 64)])
def test_original_provider_capture_identity_cannot_join_missing_or_malformed_fields(field, value):
    row = _row()
    turn = {'invocation_id': 'child-a', 'request_sha256': 'c' * 64,
        'raw_response_sha256': 'a' * 64, 'assistant': {'content': 'actual reply'},
        'model_response': {'raw_calls': []}}
    turn[field] = value
    decision = {'approved': True, 'invocation_id': turn.get('invocation_id'),
        'raw_response_sha256': turn.get('raw_response_sha256')}
    allowed, omission = converter.admitted_teacher_turn(row, 0, turn, {(row['id'], 0): decision})
    assert allowed is False and 'request/response capture' in omission['reason']
    valid = {**turn, 'invocation_id': 'child-a', 'request_sha256': 'c' * 64,
        'raw_response_sha256': 'a' * 64}
    bound = {**decision, 'invocation_id': 'child-a', 'raw_response_sha256': 'a' * 64}
    assert converter.admitted_teacher_turn(row, 0, valid, {(row['id'], 0): bound}) == (True, bound)
