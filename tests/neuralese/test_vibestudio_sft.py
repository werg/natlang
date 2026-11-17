"""Vibestudio corpus admission and the existing per-call learner boundary."""
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import vibestudio_sft as v  # noqa: E402


def episode(case_id='one', split='train', group='world-one'):
    turn = {'invocation_id': 'root', 'messages': [{'role': 'user', 'content': 'Inspect the fixture.'}],
            'tools': [], 'target': {'role': 'assistant', 'content': 'checked'}}
    suite = {'id': 'diagnosis', 'hash': 'f' * 64}
    source_ids = [f'vibestudio:system:diagnosis:{case_id}']
    source_groups = [f'vibestudio:system:diagnosis:{group}']
    return {'schema': v.EPISODE_SCHEMA, 'id': f'episode:{case_id}', 'suite': suite,
            'case': {'id': case_id, 'split': split, 'group': group, 'input_sha256': v.digest(case_id)},
            'source': {'template': 'system', 'template_commit': 'a' * 40, 'natlang_commit': 'b' * 40,
                       'vibestudio_commit': 'c' * 40, 'program_sha256': 'd' * 64, 'dirty': False},
            'teacher': {'provider': 'openai-codex', 'model_id': 'gpt-6-luna', 'identity': 'luna'},
            'verification': {'verdict': 'accepted', 'reason': 'The recorded evidence supports the answer.',
                             'method': 'independent-model-review',
                             'teacher': {'provider': 'openai-codex', 'model_id': 'gpt-6-luna', 'identity': 'luna'}},
            'rejection': None,
            'outcome': {'kind': 'returned', 'quality': 1, 'gates': {'answer': True}, 'tool_error_count': 0},
            'replay': {'verified': True, 'request_hashes': [v.request_hash(turn)]}, 'turns': [turn],
            'source_ids': source_ids, 'source_groups': source_groups, 'source_revisions': ['a' * 40]}


def manifest(*episodes):
    return {'schema': v.MANIFEST_SCHEMA, 'suite': episodes[0]['suite'],
            'source': episodes[0]['source'], 'teacher': episodes[0]['teacher'], 'cases': [
        {**e['case'], 'suite': e['suite'], **{key: e[key] for key in ('source_ids', 'source_groups', 'source_revisions')}}
        for e in episodes]}


EMPTY_HELD = (set(), set(), set())


def test_only_full_verified_training_episodes_are_admitted():
    good, protected = episode(), episode('test', 'test', 'another-world')
    bad = episode('bad', group='bad-world')
    bad['outcome']['quality'] = 0.9
    accepted, census = v.admit([good, protected, bad], manifest(good, protected, bad), EMPTY_HELD)
    assert accepted == [good]
    assert census == {'seen': 3, 'admitted': 1, 'protected_split': 1, 'not_full_success': 1}


@pytest.mark.parametrize('change,reason', [
    (lambda e: e['replay'].update(verified=False), 'replay_not_verified'),
    (lambda e: e['replay'].update(request_hashes=['x' * 64]), 'replay_not_verified'),
    (lambda e: e['turns'][0]['messages'][0].update(content='mutated observation'), 'replay_not_verified'),
    (lambda e: e['outcome'].update(tool_error_count=1), 'tool_errors_or_missing_evidence'),
    (lambda e: e['outcome']['gates'].update(answer=False), 'gate_failed'),
    (lambda e: e['verification'].update(verdict='rejected'), 'semantic_verification_missing_or_rejected'),
    (lambda e: e.update(rejection='semantic mismatch'), 'semantic_verification_missing_or_rejected'),
    (lambda e: e.update(turns=[]), 'no_model_calls'),
])
def test_replay_and_outcome_evidence_are_checked(change, reason):
    row = episode()
    inventory = manifest(row)
    change(row)
    assert v.admit([row], inventory, EMPTY_HELD)[1][reason] == 1


def test_source_group_sibling_and_held_source_are_excluded():
    train, protected = episode(), episode('protected', 'validation')
    assert v.admit([train], manifest(train, protected), EMPTY_HELD)[1]['source_group_crosses_splits'] == 1
    held = (set(), set(train['source_groups']), set())
    assert v.admit([train], manifest(train), held)[1]['held_excluded'] == 1
    held = (set(), set(), set(train['source_ids']))
    assert v.admit([train], manifest(train), held)[1]['held_excluded'] == 1


def test_identical_inputs_cannot_cross_splits_under_other_group_names():
    train, protected = episode(), episode('protected', 'validation', 'other-group')
    protected['case']['input_sha256'] = train['case']['input_sha256']
    assert v.admit([train], manifest(train, protected), EMPTY_HELD)[1]['input_crosses_splits'] == 1


def test_identical_inputs_remain_protected_after_suite_rebuild_and_group_rename():
    train, protected = episode(), episode('protected', 'test', 'old-group')
    protected['suite']['hash'] = '9' * 64
    protected['case']['input_sha256'] = train['case']['input_sha256']
    assert v.admit([train], manifest(train, protected), EMPTY_HELD)[1]['input_crosses_splits'] == 1


def test_case_ids_with_colons_still_share_the_stable_suite_namespace():
    train, protected = episode('family:new'), episode('other:old', 'test', 'old-group')
    protected['case']['input_sha256'] = train['case']['input_sha256']
    assert v.admit([train], manifest(train, protected), EMPTY_HELD)[1]['input_crosses_splits'] == 1


def test_split_and_provenance_cannot_be_relabelled():
    row = episode()
    inventory = manifest(copy.deepcopy(row))
    row['case']['split'] = 'test'
    with pytest.raises(ValueError, match='split does not match'):
        v.admit([row], inventory, EMPTY_HELD)
    row = episode()
    row['source']['dirty'] = True
    with pytest.raises(ValueError, match='clean pinned tree'):
        v.admit([row], manifest(row), EMPTY_HELD)


def test_duplicate_episode_id_is_a_source_error():
    row = episode()
    with pytest.raises(ValueError, match='duplicate episode id'):
        v.admit([row, row], manifest(row), EMPTY_HELD)


def test_request_canonical_preserves_javascript_number_spelling_and_binds_content():
    row = episode()
    turn = row['turns'][0]
    turn['messages'][0]['probability'] = 1e-7
    canonical = json.dumps({'messages': turn['messages'], 'tools': turn['tools']},
                           separators=(',', ':'), sort_keys=True).replace('1e-07', '1e-7')
    turn['request_canonical'] = canonical
    row['replay']['request_hashes'] = [hashlib.sha256(canonical.encode()).hexdigest()]
    assert v.admit([row], manifest(row), EMPTY_HELD)[0] == [row]
    turn['messages'][0]['probability'] = 0.5
    with pytest.raises(ValueError, match='does not describe this exact request'):
        v.admit([row], manifest(row), EMPTY_HELD)


def test_repeated_identical_demonstrations_do_not_add_loss_mass():
    row = episode()
    twin = {**copy.deepcopy(row), 'id': 'another-attempt'}
    twin['turns'][0]['invocation_id'] = 'another-root-identity'
    accepted, census = v.admit([row, twin], manifest(row), EMPTY_HELD)
    assert accepted == [row]
    assert census['duplicate_trajectory'] == 1


class Renderer:
    identity = {'tokenizer_sha256': 'e' * 64}

    def render(self, messages, tools, target):
        return [1, 2, 3], [4, 5, 6]

    def think_context_end(self, target_ids):
        return 1


def build_inputs(tmp_path):
    row = episode()
    episodes_path = tmp_path / 'episodes.jsonl'
    episodes_path.write_text(json.dumps(row) + '\n')
    info = {**manifest(row), 'episodes': {'path': episodes_path.name, 'sha256': v.sha256_file(episodes_path)}}
    manifest_path = tmp_path / 'collection.json'
    manifest_path.write_text(json.dumps(info))
    held_cases = tmp_path / 'held-cases.jsonl'
    held_cases.write_text(json.dumps({'id': 'unrelated-held', 'source_groups': ['unrelated']}) + '\n')
    held = tmp_path / 'held.json'
    held.write_text(json.dumps({'tasks': [{'id': 'unrelated-held'}], 'excluded_split_groups': [],
                                'cases': {'path': held_cases.name, 'sha256': v.sha256_file(held_cases)}}))
    return SimpleNamespace(episodes=episodes_path, manifest=manifest_path, out=tmp_path / 'built',
                           held_manifest=held, tokenizer='not-needed', max_per_case=4)


def test_build_is_consumable_by_ordinary_sft_and_masks_context(tmp_path):
    args = build_inputs(tmp_path)
    result = v.build(args, Renderer())
    data = pickle.loads((args.out / 'chains.pkl').read_bytes())
    assert data['schema'] == 'natlang.sft-chains/1'
    chain = data['chains'][0]
    assert chain['ids'].tolist() == [1, 2, 3, 4, 5, 6]
    start, length, row = chain['segs'][0]
    assert (start, length) == (3, 3)
    assert row['weight'].tolist() == [0, 1, 1]
    assert result['weighted_target_tokens'] == 2
    assert result['inputs']['episodes_sha256'] == v.sha256_file(args.episodes)
    assert result['outputs']['chains.pkl'] == v.sha256_file(args.out / 'chains.pkl')
    semantic = json.loads(next(v.zstd_lines(args.out / 'rows.jsonl.zst')))
    assert semantic['provenance']['program_sha256'] == 'd' * 64
    assert semantic['function'] == 'unknown'
    assert result['target_mass']['by_role']['unknown']['weighted_target_tokens'] == 2
    with pytest.raises(ValueError, match='already exists'):
        v.build(args, Renderer())


def test_mutated_collection_cannot_be_built(tmp_path):
    args = build_inputs(tmp_path)
    args.episodes.write_text(args.episodes.read_text() + '\n')
    with pytest.raises(ValueError, match='does not match collection manifest'):
        v.build(args, Renderer())


def collection_directory(directory, row, extra_cases=()):
    directory.mkdir()
    episodes_path = directory / 'episodes.jsonl'
    episodes_path.write_text(json.dumps(row) + '\n')
    info = {**manifest(row, *extra_cases),
            'episodes': {'path': episodes_path.name, 'sha256': v.sha256_file(episodes_path)}}
    (directory / 'manifest.json').write_text(json.dumps(info))
    return directory


def test_multiple_collections_build_one_training_file(tmp_path):
    args = build_inputs(tmp_path)
    one, two = episode(), episode('second', group='second-world')
    directories = [collection_directory(tmp_path / 'first-suite', one),
                   collection_directory(tmp_path / 'second-suite', two)]
    args.manifest = args.episodes = None
    args.collection = directories
    result = v.build(args, Renderer())
    assert (result['episodes'], result['calls'], result['chains']) == (2, 2, 2)
    assert len(result['inputs']['collections']) == 2
    assert result['weighted_target_tokens'] == 4


def test_split_siblings_in_another_collection_are_still_protected(tmp_path):
    train = episode()
    protected = episode('protected', 'test')
    directories = [collection_directory(tmp_path / 'training-collection', train),
                   collection_directory(tmp_path / 'other-collection', protected)]
    episodes, info, _ = v.load_collections(SimpleNamespace(collection=directories))
    accepted, census = v.admit(episodes, info, EMPTY_HELD)
    assert not accepted
    assert census['source_group_crosses_splits'] == census['protected_split'] == 1


def test_registration_describes_admitted_data_and_preserves_raw_inputs(tmp_path):
    args = build_inputs(tmp_path)
    report = v.build(args, Renderer())
    source = report['inputs']['collections'][0]
    assert (args.out / source['episodes_snapshot']).read_bytes() == args.episodes.read_bytes()
    entry, links = v.registration_entry(args.out, 'vibestudio-real-pilot-v1', 'dgx')
    assert entry['admission'] == 'sft-experiment-input'
    assert entry['source_collections'][0]['episodes_sha256'] == v.sha256_file(args.episodes)
    assert entry['stages']['admitted_episodes'] == 1
    assert 'not performed' in entry['stages']['learner_exposure']
    assert links == {'runs/vibestudio-real-pilot-v1': str(args.out)}


def test_registration_refuses_mutated_training_payload(tmp_path):
    args = build_inputs(tmp_path)
    v.build(args, Renderer())
    (args.out / 'chains.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='built output changed'):
        v.registration_entry(args.out, 'corpus-v1', 'dgx')


def test_registration_refuses_mutated_raw_collection(tmp_path):
    args = build_inputs(tmp_path)
    report = v.build(args, Renderer())
    raw = args.out / report['inputs']['collections'][0]['episodes_snapshot']
    raw.write_text('changed\n')
    with pytest.raises(ValueError, match='snapshot changed'):
        v.registration_entry(args.out, 'corpus-v1', 'dgx')


def test_registration_uses_shared_publisher_without_exposing_test_data(tmp_path, monkeypatch):
    import register_corpora
    args = build_inputs(tmp_path)
    v.build(args, Renderer())
    captured = []

    def fake_register(entries, message, **kwargs):
        captured.append((entries, message, kwargs))
        return [entries[0]['id']]

    monkeypatch.setattr(register_corpora, 'register', fake_register)
    assert v.register_build(args.out, 'pilot-v1', 'pop') == ['pilot-v1']
    entries, message, options = captured[0]
    assert entries[0]['owner'] == 'pop'
    assert options['links'] == {'runs/pilot-v1': str(args.out)}
    with pytest.raises(ValueError, match='actual data owner'):
        v.registration_entry(args.out, 'pilot-v1', 'tick')


@pytest.mark.parametrize('field,change', [
    ('source', lambda value: value.update(program_sha256='9' * 64)),
    ('source', lambda value: value.update(template_commit='9' * 40)),
    ('teacher', lambda value: value.update(model_id='different-teacher')),
    ('teacher', lambda value: value.update(identity='different-authenticated-model')),
    ('suite', lambda value: value.update(hash='9' * 64)),
])
def test_collection_binds_each_episode_source_teacher_and_suite(tmp_path, field, change):
    args = build_inputs(tmp_path)
    row = json.loads(args.episodes.read_text())
    change(row[field])
    args.episodes.write_text(json.dumps(row) + '\n')
    # A file hash is necessary but not sufficient: a resume can accidentally
    # rehash a foreign episode. It must still belong to this collection.
    collection = json.loads(args.manifest.read_text())
    collection['episodes']['sha256'] = v.sha256_file(args.episodes)
    args.manifest.write_text(json.dumps(collection))
    with pytest.raises(ValueError, match=f'episode {field} identity does not match'):
        v.load_collections(args)


def test_actual_supervised_mass_distinguishes_root_and_child_functions(tmp_path):
    args = build_inputs(tmp_path)
    row = json.loads(args.episodes.read_text())
    child = copy.deepcopy(row['turns'][0])
    child['invocation_id'] = 'child'
    child['parent_call_id'] = 'root'
    row['turns'].append(child)
    row['replay']['request_hashes'].append(v.request_hash(child))
    row['evaluation'] = {'traces': [
        {'callId': 'root', 'name': 'briefLog', 'definitionId': 'root-definition', 'parentCallId': None},
        {'callId': 'child', 'name': 'classify', 'definitionId': 'child-definition', 'parentCallId': 'root'}]}
    args.episodes.write_text(json.dumps(row) + '\n')
    collection = json.loads(args.manifest.read_text())
    collection['episodes']['sha256'] = v.sha256_file(args.episodes)
    args.manifest.write_text(json.dumps(collection))
    result = v.build(args, Renderer())
    roles = result['target_mass']['by_role']
    assert roles['root'] == roles['child'] == {'calls': 1, 'target_tokens': 3, 'weighted_target_tokens': 2}
    assert roles['unknown']['calls'] == 0
    functions = result['target_mass']['by_suite_function']
    assert {(item['function'], item['role']) for item in functions} == {('briefLog', 'root'), ('classify', 'child')}
    rows = [json.loads(line) for line in v.zstd_lines(args.out / 'rows.jsonl.zst')]
    assert rows[1]['function'] == 'classify'
    assert rows[1]['definition_id'] == 'child-definition'
    assert rows[1]['parent_invocation'] == 'root'
