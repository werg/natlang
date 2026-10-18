import copy
import json
from pathlib import Path

import pytest

from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as converter


def test_teacher_receipt_binds_only_explicit_history_profile():
    from types import SimpleNamespace

    # Mellum's declared default is KEEP, but that is not evidence that an
    # explicit conversion override was selected. Preserve the absence of the
    # field for legacy/default receipts; record an explicit override only when
    # the converter requested and actually bound that profile.
    tokenizer = SimpleNamespace(natlang_history_reasoning={'policy': 'keep'})
    assert converter.explicit_history_profile_receipt(SimpleNamespace(history_reasoning_policy=None), tokenizer) == {}
    assert converter.explicit_history_profile_receipt(SimpleNamespace(history_reasoning_policy='keep'), tokenizer) == {
        'explicit_history_reasoning_policy': 'keep'}
    with pytest.raises(ValueError, match='differs from the bound tokenizer profile'):
        converter.explicit_history_profile_receipt(
            SimpleNamespace(history_reasoning_policy='last_turn_only'), tokenizer)


def _row():
    return {'id': 'execution-a', 'version': 'natlang.teacher_trajectory.native/1',
        'task': {'program_ir': {'version': 'natlang.program/2', 'split': 'train',
                              'id': 'program-a', 'source_groups': ['train-g']}},
        'outcome': {'status': 'done', 'accepted': True, 'oracle': {'accepted': True},
                    'checks': {'exact_result': True}},
        'provenance': {'model': 'pennyroyal'},
        'execution_identity': {'execution_run_id': 'run-a'},
        'trajectory': [{'phase': 'action', 'assistant': {'content': 'correct'}}]}


def test_row_id_recovery_preserves_existing_receipt_and_exact_file_identity():
    item={'path':'/local/result','source':'/remote/result','sha256':'a'*64,'bytes':7}
    receipt={'files':[item]}
    result=converter.restore_transfer_row_ids(receipt,[{**item,'row_id':'actual-row'}])
    assert result['files']==[{**item,'row_id':'actual-row'}]
    assert receipt=={'files':[item]} and 'row_id' not in item
    for field,value in [('path','/other'),('source','/other'),('sha256','b'*64),('bytes',8)]:
        with pytest.raises(ValueError,match='unbound source file'):
            converter.restore_transfer_row_ids(receipt,[{**item,field:value,'row_id':'actual-row'}])


def test_row_id_recovery_cannot_replace_a_recorded_row_or_alias_another_file():
    first={'path':'/local/first','source':'/remote/first','sha256':'a'*64,'bytes':7,'row_id':'first'}
    second={'path':'/local/second','source':'/remote/second','sha256':'b'*64,'bytes':8}
    with pytest.raises(ValueError,match='existing identity'):
        converter.restore_transfer_row_ids({'files':[first]},[{**first,'row_id':'replacement'}])
    with pytest.raises(ValueError,match='repeats a row identity'):
        converter.restore_transfer_row_ids({'files':[first,second]},[{**second,'row_id':'first'}])
    with pytest.raises(ValueError,match='duplicate or invalid'):
        converter.restore_transfer_row_ids({'files':[second]},[{**second,'row_id':'second'}]*2)


def prepared_packet_fixture(tmp_path, mode='student'):
    program={'version':'natlang.program/2','id':'private','split':'train',
        'source_groups':['private-train'],'curriculum':{
            'reference':{'root':[['eval',{'code':'return 7;'}]]},
            'trajectory_contract':{'version':'natlang.trajectory_contract/1',
                'authored_root_eval':True,'authored_root_sha256':'original',
                'min_child_invocations':0,'min_state_edges_from_children':0,'min_iteration_steps':0}}}
    originals=[{**program,'id':'before'},program,{**program,'id':'after'}]
    actual=copy.deepcopy(program)
    if mode=='student':
        actual['curriculum']['trajectory_contract'].pop('authored_root_eval')
        actual['curriculum']['trajectory_contract'].pop('authored_root_sha256')
    else:
        import hashlib
        actual['curriculum']['trajectory_contract']['authored_root_sha256']=hashlib.sha256(b'return 7;').hexdigest()
    packet=tmp_path/'packet.jsonl';packet.write_text(json.dumps(actual)+'\n')
    preparation=tmp_path/'preparation.json'
    declaration={'schema':'natlang.task-rollout-preparation/1','root_mode':mode,
        'source_cases':'/remote/original-cases.jsonl','source_sha256':'a'*64,
        'packet_sha256':converter.sha256_file(packet)}
    preparation.write_text(json.dumps(declaration))
    return originals,actual,{'execution_packet':packet,'execution_preparation':preparation},declaration


@pytest.mark.parametrize('mode',['student','authored'])
def test_prepared_packet_preserves_original_program_and_authenticates_actual_root_view(tmp_path,mode):
    originals,actual,paths,preparation=prepared_packet_fixture(tmp_path,mode)
    before=copy.deepcopy(originals)
    result=converter.prepared_execution_packet(paths,'a'*64,originals)
    assert result=={'programs':[actual],'preparation':preparation}
    assert originals==before
    assert converter.prepared_execution_packet({},'a'*64,originals) is None
    with pytest.raises(ValueError,match='both packet and preparation'):
        converter.prepared_execution_packet({'execution_packet':paths['execution_packet']},'a'*64,originals)


@pytest.mark.parametrize('mutation',['source_sha','packet_sha','mode','relative_source','modified_program','duplicate','empty'])
def test_prepared_packet_cannot_relabel_or_change_the_execution_inputs(tmp_path,mutation):
    originals,actual,paths,preparation=prepared_packet_fixture(tmp_path)
    if mutation=='source_sha':preparation['source_sha256']='b'*64
    elif mutation=='packet_sha':preparation['packet_sha256']='b'*64
    elif mutation=='mode':preparation['root_mode']='guessed'
    elif mutation=='relative_source':preparation['source_cases']='unbound-cases.jsonl'
    else:
        if mutation=='modified_program':
            actual['curriculum']['reference']['root'][0][1]['code']='return 999;'
            raw=json.dumps(actual)+'\n'
        elif mutation=='duplicate':raw=(json.dumps(actual)+'\n')*2
        else:raw=''
        paths['execution_packet'].write_text(raw)
        preparation['packet_sha256']=converter.sha256_file(paths['execution_packet'])
    paths['execution_preparation'].write_text(json.dumps(preparation))
    with pytest.raises(ValueError):
        converter.prepared_execution_packet(paths,'a'*64,originals)


def test_teacher_action_identity_is_bound_to_exact_source_trajectory():
    # The same upstream program/action may appear in separate accepted teacher
    # trajectories. Preserve both while keeping each tokenizer rendering tied
    # to the same source identity.
    source_a = 'a' * 64
    source_b = 'b' * 64
    first = converter.teacher_action_graph_id('teacher-program:shared', source_a, 0)
    assert first == converter.teacher_action_graph_id('teacher-program:shared', source_a, 0)
    assert first != converter.teacher_action_graph_id('teacher-program:shared', source_b, 0)
    assert first != converter.teacher_action_graph_id('teacher-program:shared', source_a, 1)
    assert converter.teacher_action_graph_id('teacher-program:shared', source_a, 0, 'keep') == first + ':history-reasoning:keep'
    assert converter.teacher_action_graph_id('teacher-program:shared', source_a, 0, 'last_turn_only') == first + ':history-reasoning:last_turn_only'
    with pytest.raises(ValueError, match='source record ID'):
        converter.teacher_action_graph_id('teacher-program:shared', 'A' * 64, 0)


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
    review = tmp_path / 'decision-review.json'
    review.write_text('{"schema":"natlang.native-decision-review/1","approvals":[]}')
    _, reviewed = converter.teacher_decision_admissions(source, digest, decision_review=review)
    assert len(calls) == 3 and reviewed['path'] != updated['path']
    assert calls[-1][-2:] == ['--decision-review', str(review)]
    assert converter.teacher_decision_admissions(source, digest, decision_review=review)[1] == reviewed
    assert len(calls) == 3
    review.write_text('{"schema":"natlang.native-decision-review/1","approvals":[],"revision":2}')
    _, changed_review = converter.teacher_decision_admissions(source, digest, decision_review=review)
    assert len(calls) == 4 and changed_review['path'] != reviewed['path']
    def mutate_review(argv, **kwargs):
        result = execute(argv, **kwargs)
        review.write_text('{"changed_during_execution":true}')
        return result
    monkeypatch.setattr(converter.subprocess, 'run', mutate_review)
    review.write_text('{"revision":3}')
    with pytest.raises(ValueError, match='review changed during admission'):
        converter.teacher_decision_admissions(source, digest, decision_review=review)


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


def test_authenticated_text_emulation_preserves_renderer_omitted_null_type():
    import hashlib
    import json
    block_id, body = 'nz1_untyped-source-state', 'oos'
    block = {'id': block_id, 'type': None, 'body': body,
        'body_sha256': hashlib.sha256(body.encode()).hexdigest(), 'learned_vectors': False}
    label = (f'[[Neuralese text block id={block_id}; exact JSON string body='
             f'{json.dumps(body, ensure_ascii=False, separators=(",", ":"))}]]')
    context = [{'role': 'user', 'content': [{'type': 'neuralese', 'id': block_id}]}]
    rendered = [{'role': 'user', 'content': label}]
    transport = {'version': 'text-marker-standin/2', 'learned_vectors': False,
        'qualification_certificate': False, 'training_admission': False,
        'raw_request_sha256': 'a' * 64, 'rendered_request_sha256': 'b' * 64,
        'rendered_messages': rendered, 'expanded_input_blocks': [block]}
    messages, _, proof, omission = converter.captured_text_emulation_context(
        {'model_response': {'transport_provenance': transport}}, context, [])
    assert omission is None and messages == rendered
    assert proof['expanded_blocks'] == [{'id': block_id, 'type': None,
        'body_sha256': block['body_sha256'], 'occurrences': 1}]


@pytest.mark.parametrize('code_selection', [False, 'list', 'count'])
@pytest.mark.parametrize('explicit_collection', [False, True])
@pytest.mark.parametrize('prepared_execution', [False, True])
def test_each_teacher_batch_uses_its_own_exact_program_packet(tmp_path, monkeypatch, code_selection, explicit_collection, prepared_execution):
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
        original=copy.deepcopy(program)
        if prepared_execution:
            original['curriculum']={'trajectory_contract':{
                'version':'natlang.trajectory_contract/1','authored_root_eval':True,
                'authored_root_sha256':'original','min_child_invocations':0}}
            from natlang_neuralese.rl.task_rollouts import student_root_row
            program.clear();program.update(student_root_row(original))
        cases = put(base+'/selected-cases.jsonl', original)
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
        original_source=f'/work/generation/batch-{index:04d}/source-cases.jsonl'
        collection = put(base+'/teacher-source.json', {
            'schema': 'natlang.sandbox-strong-teacher-collection/1',
            'source_cases_sha256': packet_sha, 'protected_selection': selection,
            **({'root_mode':'student','source_cases':original_source} if prepared_execution else {}),
            'teacher': teacher})
        files.append({'path': str(collection), 'source': collection_source,
            'bytes': collection.stat().st_size, 'sha256': converter.sha256_file(collection)})
        response = put(base+'/response.jsonl', row)
        files.append({'row_id': row['id'], 'path': str(response),
            'source': f'/work/generation/batch-{index:04d}/results/program/response.jsonl',
            'bytes': response.stat().st_size, 'sha256': converter.sha256_file(response)})
        if explicit_collection:
            # A continuation has a different physical layout. Its immutable
            # transfer receipt names the actual collection, never an alias.
            files[-1]['source'] = f'/work/generation/continuation-{index}/outputs/response.jsonl'
            files[-1]['collection_source'] = collection_source
        batches.append({'collection_source': collection_source, **{name: {
            'path': str(path.relative_to(tmp_path)), 'sha256': converter.sha256_file(path)}
            for name, path in [('selection', selected), ('selected_cases', cases),
                               ('global_protection_join', protection)]}})
        if prepared_execution:
            actual=put(base+'/execution-packet.jsonl',program)
            prep=put(base+'/preparation.json',{'schema':'natlang.task-rollout-preparation/1',
                'root_mode':'student','source_cases':original_source,'source_sha256':packet_sha,
                'packet_sha256':converter.sha256_file(actual)})
            batches[-1].update({name:{'path':str(path.relative_to(tmp_path)),
                'sha256':converter.sha256_file(path)} for name,path in
                [('execution_packet',actual),('execution_preparation',prep)]})
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
    # Historical transfers may omit row_id. Restore that fact in a separate
    # continuation instead of rewriting the immutable transfer receipt.
    transfer_path = tmp_path/'transfer-receipt.json'
    original_transfer = json.loads(transfer_path.read_text())
    row_files = [item for item in original_transfer['files'] if 'row_id' in item]
    row_bindings = [{'row_id': item['row_id'], **{key: item[key] for key in
        ('path', 'source', 'sha256', 'bytes')}} for item in row_files]
    original_transfer['files'] = [{key: value for key, value in item.items() if key != 'row_id'}
                                  for item in original_transfer['files']]
    put('transfer-receipt.json', original_transfer)
    parent_sha = converter.sha256_file(transfer_path)
    continuation = {'schema': 'natlang.teacher-transfer-row-bindings/1',
        'parent_transfer_receipt_sha256': parent_sha,
        'combined_source_sha256': original_transfer['combined']['sha256'],
        'files': row_bindings}
    put('source-row-bindings.json', continuation)
    continued = converter.source_binding(args)
    assert {row['id'] for row in rows} <= set(continued['source_files'])
    for row in rows:
        assert converter.validate_teacher_row(row, continued)[0] == row['task']['program_ir']
    continuation['parent_transfer_receipt_sha256'] = '0' * 64
    put('source-row-bindings.json', continuation)
    with pytest.raises(ValueError, match='row-binding continuation'):
        converter.source_binding(args)
    (tmp_path/'source-row-bindings.json').unlink()
    put('transfer-receipt.json', {**original_transfer, 'files': files})
    binding = converter.source_binding(args)
    # Global membership in another admitted packet cannot retarget a response.
    swapped = copy.deepcopy(rows[0])
    swapped['task']['program_ir'] = rows[1]['task']['program_ir']
    with pytest.raises(ValueError, match='actual batch selected packet'):
        converter.validate_teacher_row(swapped, binding)
    changed = copy.deepcopy(rows[0])
    changed['task']['program_ir']['semantics'] = {'rewritten': True}
    with pytest.raises(ValueError, match='actual batch selected packet'):
        converter.validate_teacher_row(changed, binding)
    first = binding['source_files'][rows[0]['id']]
    original_collection = dict(first)
    for invalid in (None, '', 7, '/work/unbound/teacher-source.json'):
        first['collection_source'] = invalid
        with pytest.raises(ValueError, match='collection_source|immutable collection'):
            converter.validate_teacher_row(rows[0], binding)
    first.clear(); first.update(original_collection)
    first['collection_source'] = batches[1]['collection_source']
    with pytest.raises(ValueError, match='actual batch selected packet'):
        converter.validate_teacher_row(rows[0], binding)
    first.clear(); first.update(original_collection)
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


def authored_packet(tmp_path, monkeypatch):
    monkeypatch.setattr(converter.source_review, 'pending_unpinned_source_groups', lambda: set())
    def put(name,value):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value)+'\n');return path
    held=put('protection/held-union/full/groups.jsonl',{'source_groups':['held-world']})
    quality=put('protection/quality-groups.jsonl',{'source_groups':['quality-world']})
    protection=put('protection/manifest.json',{'schema':'natlang.training-protection-projection/1',
        'held_groups':1,'quality_groups':1,'source_review_sha256':'a'*64,
        'files':[{'path':str(path.relative_to(tmp_path/'protection')),
            'sha256':converter.sha256_file(path),'bytes':path.stat().st_size} for path in (held,quality)]})
    programs=[{'version':'natlang.program/2','id':'fresh-world-a','split':'train',
        'source_groups':['fresh-source-a'],'inputs':{'query':'fresh task'}}]
    cases=put('cases.jsonl',programs[0]);digest=converter.sha256_file(cases)
    selection=put('selection-proof.json',{'schema':'natlang.fresh-authored-teacher-packet/1',
        'cases':1,'cases_sha256':digest,'worlds':1,'held_and_pending_intersections':0,
        'training_admission':False,'canonical_programs_sha256':converter.canonical_json_sha256_hex(programs),
        'protection':{'path':'/original/machine/manifest.json','sha256':converter.sha256_file(protection),
            'held_groups':1,'quality_groups':1,'source_review_sha256':'a'*64}})
    summary_value={'cases':1,'admitted':1,'decisions':4,'approved':3,
        'held':{'seed failed proposal is not positive':1},'omitted':{},'omissions':[]}
    summary=put('reference-isolated.summary.json',summary_value)
    qualification=put('reference-qualification.json',{'schema':'natlang.authored-reference-qualification/1',
        'passed':True,'training_admission':False,'reference_traces_are_teacher_generations':False,
        'cases':1,'cases_sha256':digest,'counts':summary_value,
        'runtime_inventory':{'runtime.js':'a'*64}, 'scripts_inventory':{'replay.mjs':'b'*64},
        'runtime_sha256':converter.canonical_json_sha256_hex({'runtime.js':'a'*64}),
        'scripts_sha256':converter.canonical_json_sha256_hex({'replay.mjs':'b'*64}),
        'selection_proof_sha256':converter.sha256_file(selection),
        'reference_summary_sha256':converter.sha256_file(summary)})
    join=put('global-protection-join.json',{'schema':'natlang.native-teacher-global-protection-join/1',
        'status':'passed','selected_packet_sha256':digest,'held_group_intersections':[],
        'quality_group_intersections':[]})
    return dict(selection=selection,selected_cases=cases,global_protection_join=join,
        reference_qualification=qualification,reference_summary=summary,selection_protection=protection)


def test_fresh_authored_packet_preserves_schema_and_full_exact_witness_closure(tmp_path,monkeypatch):
    paths=authored_packet(tmp_path,monkeypatch)
    refs={name:{'path':str(path),'sha256':converter.sha256_file(path)} for name,path in paths.items()}
    result=converter.packet_binding(tmp_path,refs)
    assert result['selection']['schema']=='natlang.fresh-authored-teacher-packet/1'
    assert result['selected_ids']=={'fresh-world-a'} and result['train_groups']=={'fresh-source-a'}
    assert set(result['proofs'])==set(paths)
    # Reference qualification has no authority to turn reference traces into teacher data.
    assert result['selection']['training_admission'] is False


@pytest.mark.parametrize('tamper', ['missing-proof','failed-reference','wrong-selection','wrong-summary',
    'partial-reference','changed-programs','protected-group','nested-protection','held-overlap'])
def test_fresh_authored_packet_rejects_incomplete_or_forged_witnesses(tmp_path,monkeypatch,tamper):
    paths=authored_packet(tmp_path,monkeypatch)
    def alter(role,change):
        value=json.loads(paths[role].read_text());change(value);paths[role].write_text(json.dumps(value))
    if tamper=='missing-proof':paths.pop('reference_qualification')
    elif tamper=='failed-reference':alter('reference_qualification',lambda value:value.update(passed=False))
    elif tamper=='wrong-selection':alter('reference_qualification',lambda value:value.update(selection_proof_sha256='0'*64))
    elif tamper=='wrong-summary':alter('reference_summary',lambda value:value.update(approved=0))
    elif tamper=='partial-reference':alter('reference_qualification',lambda value:value.update(cases=0))
    elif tamper=='changed-programs':alter('selection',lambda value:value.update(canonical_programs_sha256='0'*64))
    elif tamper=='protected-group':alter('selection_protection',lambda value:value.update(source_review_sha256='b'*64))
    elif tamper=='nested-protection':alter('selection',lambda value:value['protection'].update(sha256='0'*64))
    elif tamper=='held-overlap':
        child=paths['selection_protection'].parent/'held-union/full/groups.jsonl'
        child.write_text(json.dumps({'source_groups':['fresh-source-a']}))
        alter('selection_protection',lambda value:value['files'][0].update(
            sha256=converter.sha256_file(child),bytes=child.stat().st_size))
        alter('selection',lambda value:value['protection'].update(sha256=converter.sha256_file(paths['selection_protection'])))
        alter('reference_qualification',lambda value:value.update(selection_proof_sha256=converter.sha256_file(paths['selection'])))
    refs={name:{'path':str(path),'sha256':converter.sha256_file(path)} for name,path in paths.items()}
    with pytest.raises(ValueError,match='fresh authored'):
        converter.packet_binding(tmp_path,refs)


def continuation_fixture(tmp_path, schema='natlang.sandbox-strong-teacher-continuation/1'):
    parent={'schema':'natlang.sandbox-strong-teacher-collection/1',
            'source_cases_sha256':'c'*64,'protected_selection':{'selected_ids':['task']},'seed_base':31,
            'teacher':{'checkpoint':'original'}}
    collection={**parent,'schema':schema,'teacher':{'checkpoint':'current'},
                'source_indices_seeds_preserved':True,'entries':2}
    transferred={}
    for name in ('original_source','original_queue','drain_frontier','actor','actual_teacher_execution_identity'):
        path=tmp_path/(name+'.json')
        path.write_text(json.dumps(parent if name=='original_source' else {'fixture':name}))
        reference={'path':'/remote/'+name+'.json','sha256':converter.sha256_file(path),'bytes':path.stat().st_size}
        collection[name]=reference
        transferred[reference['path']]=[{**reference,'source':reference['path'],'path':str(path)}]
    execution=collection['actual_teacher_execution_identity']
    path=Path(transferred[execution['path']][0]['path'])
    path.write_text(json.dumps({'actor':collection['actor']}))
    execution.update(sha256=converter.sha256_file(path),bytes=path.stat().st_size)
    transferred[execution['path']][0].update(sha256=execution['sha256'],bytes=execution['bytes'])
    return collection,transferred


@pytest.mark.parametrize('schema',['natlang.sandbox-strong-teacher-continuation/1',
                                   'natlang.sandbox-strong-teacher-collection/1'])
def test_continuation_keeps_its_actual_actor_and_all_exact_parent_inputs(tmp_path,schema):
    collection,transferred=continuation_fixture(tmp_path,schema)
    before=copy.deepcopy(collection)
    proofs=converter.collection_provenance(collection,transferred)
    assert set(proofs)=={'original_source','original_queue','drain_frontier','actor','actual_teacher_execution_identity'}
    assert collection==before and collection['teacher']['checkpoint']=='current'


@pytest.mark.parametrize('name',['original_source','original_queue','drain_frontier','actor','actual_teacher_execution_identity'])
def test_continuation_rejects_missing_or_changed_provenance(tmp_path,name):
    collection,transferred=continuation_fixture(tmp_path)
    missing={key:value for key,value in transferred.items() if key!=collection[name]['path']}
    with pytest.raises(ValueError,match='not transferred'):
        converter.collection_provenance(collection,missing)
    Path(transferred[collection[name]['path']][0]['path']).write_text('changed')
    with pytest.raises(ValueError,match='provenance differs'):
        converter.collection_provenance(collection,transferred)


@pytest.mark.parametrize('mutation',[{'source_indices_seeds_preserved':False},
    {'source_cases_sha256':'f'*64},{'seed_base':32},{'entries':0}])
def test_continuation_cannot_reinterpret_source_or_seed_preservation(tmp_path,mutation):
    collection,transferred=continuation_fixture(tmp_path)
    collection.update(mutation)
    with pytest.raises(ValueError,match='source or seed contract'):
        converter.collection_provenance(collection,transferred)


def test_continuation_cannot_pair_an_unrelated_valid_actor_descriptor(tmp_path):
    collection,transferred=continuation_fixture(tmp_path)
    reference=collection['actual_teacher_execution_identity']
    item=transferred[reference['path']][0];path=Path(item['path'])
    path.write_text(json.dumps({'actor':{**collection['actor'],'sha256':'0'*64}}))
    reference.update(sha256=converter.sha256_file(path),bytes=path.stat().st_size)
    item.update(sha256=reference['sha256'],bytes=reference['bytes'])
    with pytest.raises(ValueError,match='different actor'):
        converter.collection_provenance(collection,transferred)


def test_continuation_parent_is_evidence_not_an_unmapped_response_collection(tmp_path,monkeypatch):
    from types import SimpleNamespace
    collection,transferred=continuation_fixture(tmp_path)
    old=collection['original_source']['path']; new='/remote/original/teacher-source.json'
    collection['original_source']['path']=new
    transferred[new]=transferred.pop(old)
    for item in transferred[new]:item['source']=new
    collection['teacher']={'teacher_role':'independent stronger open teacher; not either student/onpolicy',
                           'checkpoint':'current'}
    path=tmp_path/'teacher-source.json';path.write_text(json.dumps(collection))
    source=tmp_path/'rows.jsonl';source.write_text('{}\n')
    remote='/remote/continued/teacher-source.json'
    files=[item for copies in transferred.values() for item in copies]
    files.append({'path':str(path),'source':remote,'sha256':converter.sha256_file(path),'bytes':path.stat().st_size})
    transfer={'schema':'natlang.verified-teacher-trajectory-transfer/1','training_admission':False,
              'combined':{'path':str(source),'sha256':converter.sha256_file(source),'bytes':source.stat().st_size},
              'files':files}
    (tmp_path/'transfer-receipt.json').write_text(json.dumps(transfer))
    (tmp_path/'batch-source-bindings.json').write_text(json.dumps({
        'schema':'natlang.teacher-batch-source-bindings/1','batches':[{'collection_source':remote}]}))
    packet={'packet_sha256':collection['source_cases_sha256'],'selection':collection['protected_selection'],
            'selected_ids':{'task'},'train_groups':{'train'},'held_groups':set(),'proofs':{}}
    monkeypatch.setattr(converter,'packet_binding',lambda *args:packet)
    monkeypatch.setattr(converter,'current_protection',lambda _: {})
    result=converter.source_binding(SimpleNamespace(source=source,protection_manifest=tmp_path/'protection'))
    assert set(result['collections'])=={remote}
    assert len(result['collections'][remote]['continuation_proofs'])==5
    assert result['teacher']==collection['teacher']


def test_nested_continuation_authenticates_history_without_mutation(tmp_path):
    flat, transferred = continuation_fixture(tmp_path)
    inner_keys = ('original_source', 'original_queue', 'drain_frontier', 'actor',
                  'actual_teacher_execution_identity', 'entries', 'source_indices_seeds_preserved')
    outer = {key: value for key, value in flat.items() if key not in inner_keys}
    outer['schema'] = 'natlang.sandbox-strong-teacher-collection/1'
    outer['at'] = 20
    outer['continuation'] = {'schema': flat['schema'], 'at': 10, **{key: flat[key] for key in inner_keys}}
    path = tmp_path / 'before.json'; path.write_text('{}')
    ref = {'path': '/remote/before.json', 'sha256': converter.sha256_file(path), 'bytes': path.stat().st_size}
    outer['metadata_history'] = [{'binding': ref, 'reason': 'packet restored'}]
    transferred[ref['path']] = [{**ref, 'path': str(path), 'source': ref['path']}]
    before = copy.deepcopy(outer)
    proofs = converter.collection_provenance(outer, transferred)
    assert len(proofs) == 6 and proofs['metadata_history:0']['sha256'] == ref['sha256']
    assert outer == before
    transferred.pop(ref['path'])
    with pytest.raises(ValueError, match='not transferred: metadata_history:0'):
        converter.collection_provenance(outer, transferred)


def test_nested_continuation_rejects_conflicting_actor(tmp_path):
    flat, transferred = continuation_fixture(tmp_path)
    flat['schema'] = 'natlang.sandbox-strong-teacher-collection/1'
    flat['continuation'] = {'schema': 'natlang.sandbox-strong-teacher-continuation/1',
                            'actor': {**flat['actor'], 'sha256': 'f'*64}}
    with pytest.raises(ValueError, match='conflicts with outer actor'):
        converter.collection_provenance(flat, transferred)


def test_scaffold_source_stream_is_exact_and_immutable(tmp_path):
    root = tmp_path / 'harvest'
    accepted = ['scaffold-00-repeat-0', 'scaffold-01-repeat-0']
    (root / 'accepted').mkdir(parents=True)
    (root / 'harvest.json').write_text(json.dumps({
        'schema': 'natlang.teacher-scaffold-qualification-harvest/1',
        'accepted_keys': accepted,
    }))
    raw = []
    for key, payload in zip(accepted, ['完整 trajectory', 'second trajectory']):
        result = root / 'accepted' / key / 'results.jsonl'
        result.parent.mkdir()
        line = json.dumps({'id': key, 'trajectory': payload}, ensure_ascii=False).encode()
        result.write_bytes(line + b'\n')
        raw.append(line + b'\n')

    source = tmp_path / 'accepted.jsonl'
    converter.prepare_scaffold_source(source, root)
    assert source.read_bytes() == b''.join(raw)
    before = source.read_bytes()
    with pytest.raises(ValueError, match='refusing to overwrite'):
        converter.prepare_scaffold_source(source, root)
    assert source.read_bytes() == before


def _content_graph(identifier='first'):
    from natlang_neuralese.data.native_graph import SCHEMA
    return {'schema': SCHEMA, 'id': identifier, 'split': 'train',
        'source_groups': ['train-g'], 'tokenizer_fingerprint': 'tokenizer-a',
        'nodes': [{'name': 'text', 'token_ids': [1, 2], 'target_weights': [0., 1.]}]}


def test_content_index_reuses_validated_result_preserving_first_row_and_input_order(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    first, second = tmp_path/'first.jsonl', tmp_path/'second.jsonl'
    first.write_text(json.dumps(_content_graph('first'))+'\n'+json.dumps(_content_graph('later-same-file'))+'\n')
    second.write_text(json.dumps(_content_graph('second-file'))+'\n')
    before = converter.seed_content_fingerprints([first, second], 'tokenizer-a')
    monkeypatch.setattr(converter, 'parse_graph', lambda *_a, **_k: pytest.fail('valid prior-file cache must avoid reparse'))
    after = converter.seed_content_fingerprints([first, second], 'tokenizer-a')
    assert after == before
    assert next(iter(after[0].values())) == {'path':str(first), 'id':'first'}
    reversed_ = converter.seed_content_fingerprints([second, first], 'tokenizer-a')
    assert next(iter(reversed_[0].values())) == {'path':str(second), 'id':'second-file'}
    assert [row['rows'] for row in after[1]] == [2, 1]


@pytest.mark.parametrize('corruption', ['key','metadata','count','fingerprints'])
def test_invalid_content_cache_revalidates_source(tmp_path, monkeypatch, corruption):
    from natlang_neuralese.common import row_index
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    path=tmp_path/'graph.jsonl';path.write_text(json.dumps(_content_graph())+'\n')
    expected=converter.seed_content_fingerprints([path], 'tokenizer-a')
    cache=next((tmp_path/'cache').glob('teacher-content-*.pkl'))
    value=row_index.load(cache)
    if corruption == 'key':value['key']['tokenizer_fingerprint']='different'
    elif corruption == 'metadata':value['metadata_sha256']='0'*64
    elif corruption == 'count':value['rows']=True
    else:value['fingerprints']={'not-a-hash':'first'}
    row_index.save(cache,value)
    calls=[];original=converter.parse_graph
    monkeypatch.setattr(converter,'parse_graph',lambda *a,**kw:calls.append(True)or original(*a,**kw))
    assert converter.seed_content_fingerprints([path], 'tokenizer-a') == expected
    assert calls == [True]


def test_content_cache_cannot_hide_changed_source_tokenizer_policy_or_test_split(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    path=tmp_path/'graph.jsonl';path.write_text(json.dumps(_content_graph())+'\n')
    original=converter.seed_content_fingerprints([path], 'tokenizer-a')
    calls=[];parse=converter.parse_graph
    monkeypatch.setattr(converter,'parse_graph',lambda *a,**kw:calls.append(True)or parse(*a,**kw))
    monkeypatch.setattr(converter,'_content_index_policy_sha256',lambda:'b'*64)
    assert converter.seed_content_fingerprints([path], 'tokenizer-a') == original
    assert len(calls)==1
    with pytest.raises(ValueError,match='tokenizer differs'):
        converter.seed_content_fingerprints([path], 'tokenizer-b')
    graph=_content_graph();graph['nodes'][0]['token_ids'][1]=3
    path.write_text(json.dumps(graph)+'\n')
    changed=converter.seed_content_fingerprints([path], 'tokenizer-a')
    assert set(changed[0]) != set(original[0])
    graph['split']='test';path.write_text(json.dumps(graph)+'\n')
    with pytest.raises(ValueError,match='train graphs'):
        converter.seed_content_fingerprints([path], 'tokenizer-a')


def test_content_cache_source_mutation_during_lookup_is_rejected(tmp_path, monkeypatch):
    from natlang_neuralese.common import row_index
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    path=tmp_path/'graph.jsonl';path.write_text(json.dumps(_content_graph())+'\n')
    converter.seed_content_fingerprints([path], 'tokenizer-a')
    load=row_index.load
    def mutate(cache_path):
        value=load(cache_path)
        changed=_content_graph();changed['nodes'][0]['token_ids'][1]=3
        path.write_text(json.dumps(changed)+'\n')
        return value
    monkeypatch.setattr(row_index,'load',mutate)
    with pytest.raises(ValueError,match='source changed during validation'):
        converter.seed_content_fingerprints([path], 'tokenizer-a')


def test_content_cache_write_failure_keeps_original_validation_result(tmp_path, monkeypatch):
    unavailable=tmp_path/'cache-is-a-file';unavailable.write_text('not a directory')
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(unavailable))
    path=tmp_path/'graph.jsonl';path.write_text(json.dumps(_content_graph())+'\n')
    seen,inputs=converter.seed_content_fingerprints([path], 'tokenizer-a')
    assert next(iter(seen.values())) == {'path':str(path),'id':'first'}
    assert inputs[0]['rows'] == 1


def test_reviewed_guided_children_require_exact_review_and_structured_input_guard():
    import copy
    import pytest
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import reviewed_guided_decision_authority, canonical_json_sha256_hex
    target = {'role':'assistant', 'content':'correct child'}
    review = {'schema':'natlang.native-decision-approval/1','trajectory_id':'row',
        'source_row_sha256':'a'*64,'decision_index':1,'target_sha256':canonical_json_sha256_hex(target),
        'assistant_history_input_sha256':'b'*64}
    row = {'target':target,'source_ref':{'trajectory_id':'row','source_row_sha256':'a'*64},
        'decision':{'index':1},'collection_guidance':{'training_admission':False},
        'training_admission':{'approved':True,'semantic_review':review}}
    assert reviewed_guided_decision_authority(row,review_bound=True) == {
        'guided_parent_review':True,'parent_trace_qualified':False,'assistant_history_input_sha256':'b'*64}
    with pytest.raises(ValueError):reviewed_guided_decision_authority(row,review_bound=False)
    for key,value in [('source_row_sha256','c'*64),('trajectory_id','other'),('decision_index',2),
                      ('target_sha256','c'*64),('assistant_history_input_sha256',None),
                      ('assistant_history_input_sha256','B'*64)]:
        changed=copy.deepcopy(row);changed['training_admission']['semantic_review'][key]=value
        with pytest.raises(ValueError):reviewed_guided_decision_authority(changed,review_bound=True)
    changed=copy.deepcopy(row);changed['source_ref']['action_provenance']={'kind':'authored_reference_root_eval'}
    with pytest.raises(ValueError):reviewed_guided_decision_authority(changed,review_bound=True)
    changed=copy.deepcopy(row);changed['training_admission']['approved']=False
    assert reviewed_guided_decision_authority(changed,review_bound=False)=={}
    changed=copy.deepcopy(row);changed['collection_guidance']={}
    assert reviewed_guided_decision_authority(changed,review_bound=False)=={}


def _legacy_content_cache(tmp_path, monkeypatch, *, corruption=None):
    from natlang_neuralese.common import row_index
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    path=tmp_path/'graph.jsonl';path.write_text(json.dumps(_content_graph())+'\n')
    expected=converter.seed_content_fingerprints([path], 'tokenizer-a')
    current=next((tmp_path/'cache').glob('teacher-content-*.pkl'))
    value=row_index.load(current);current.unlink()
    value['key']={**value['key'], 'policy_sha256':converter._REVIEWED_LEGACY_CONTENT_CACHE['legacy_policy_sha256']}
    if corruption == 'metadata':value['metadata_sha256']='0'*64
    elif corruption == 'tokenizer':value['key']['tokenizer_fingerprint']='wrong'
    elif corruption == 'source':value['key']['source_sha256']='0'*64
    elif corruption == 'rows':value['rows']=True
    # Name by the expected key so altered payloads cannot evade validation by
    # simply being absent from the lookup location.
    key={**value['key'],'source_sha256':converter.sha256_file(path),'tokenizer_fingerprint':'tokenizer-a'}
    old=row_index.entry_path('teacher-content', converter.canonical_json_sha256_hex(key))
    row_index.save(old,value)
    return path,expected,old,old.read_bytes()


def test_reviewed_content_cache_migrates_exact_legacy_without_reparse(tmp_path, monkeypatch):
    from natlang_neuralese.common import row_index
    path,expected,old,original=_legacy_content_cache(tmp_path,monkeypatch)
    monkeypatch.setattr(converter,'parse_graph',lambda *_a,**_kw:pytest.fail('reviewed exact legacy must not reparse'))
    assert converter.seed_content_fingerprints([path], 'tokenizer-a')==expected
    assert old.read_bytes()==original
    caches=list((tmp_path/'cache').glob('teacher-content-*.pkl'))
    assert len(caches)==2
    new=row_index.load(next(p for p in caches if p!=old))
    assert new['key']['policy_sha256']==converter._content_index_policy_sha256()
    assert new['metadata_sha256']==row_index.load(old)['metadata_sha256']


@pytest.mark.parametrize('corruption',['metadata','tokenizer','source','rows'])
def test_reviewed_content_cache_rejects_bad_legacy(tmp_path,monkeypatch,corruption):
    path,expected,old,original=_legacy_content_cache(tmp_path,monkeypatch,corruption=corruption)
    calls=[];parse=converter.parse_graph
    monkeypatch.setattr(converter,'parse_graph',lambda *a,**kw:calls.append(True)or parse(*a,**kw))
    assert converter.seed_content_fingerprints([path], 'tokenizer-a')==expected
    assert calls==[True] and old.read_bytes()==original


def test_content_cache_policy_tracks_law_not_unrelated_admission(tmp_path,monkeypatch):
    original=converter._content_index_policy_sha256()
    read=converter.Path.read_text;module=converter.Path(converter.__file__).resolve()
    def unrelated(path,*a,**kw):
        text=read(path,*a,**kw)
        return text+'\ndef unrelated_admission_only():\n    return False\n' if path.resolve()==module else text
    monkeypatch.setattr(converter.Path,'read_text',unrelated)
    assert converter._content_index_policy_sha256()==original
    def changed_law(path,*a,**kw):
        text=read(path,*a,**kw)
        return text.replace("'nodes': graph['nodes']", "'nodes': graph.get('nodes')") if path.resolve()==module else text
    monkeypatch.setattr(converter.Path,'read_text',changed_law)
    assert converter._content_index_policy_sha256()!=original


def test_changed_parser_law_cannot_migrate_legacy(tmp_path,monkeypatch):
    path,expected,old,original=_legacy_content_cache(tmp_path,monkeypatch)
    sha=converter.sha256_file
    monkeypatch.setattr(converter,'sha256_file',lambda p:'0'*64 if str(p).endswith('/data/native_graph.py')else sha(p))
    assert converter._content_index_policy_sha256()!=converter._REVIEWED_LEGACY_CONTENT_CACHE['policy_sha256']
    calls=[];parse=converter.parse_graph
    monkeypatch.setattr(converter,'parse_graph',lambda *a,**kw:calls.append(True)or parse(*a,**kw))
    assert converter.seed_content_fingerprints([path], 'tokenizer-a')==expected
    assert calls==[True] and old.read_bytes()==original
