import copy
import json
from pathlib import Path

import pytest

from natlang_neuralese.train import recipe
from natlang_neuralese.train import native_source_classification as migration
from test_native_source_classification import fixture


def prepared(tmp_path, monkeypatch):
    source, old_index, old_package, package, current = fixture(tmp_path, monkeypatch)
    part = migration.migrate_index(source, old_index, old_package=old_package, new_package=package)
    ref = migration._binding(source)
    old = {'native_graphs': {**ref, 'binding': 'train'},
           'held_native_graphs': {**ref, 'binding': 'held'}}
    proof = {'schema': migration.SCHEMA, 'train_sha256': ref['sha256'], 'held': ref,
        'indexes': [part], 'held_indexes': [part], 'model_allocations': 0,
        'optimizer_updates': 0, 'student_pointer_writes': 0, 'graph_position_expansions': 0}
    path = tmp_path / 'classification.json'
    def save():
        path.write_text(json.dumps(proof))
        return {**old, 'source_classification_migration': {**migration._binding(path), 'binding': 'classification'}}
    return old, save, proof, part


def test_active_handoff_records_metadata_only_input_and_preserves_recipe_identity(tmp_path, monkeypatch):
    old, save, proof, part = prepared(tmp_path, monkeypatch)
    new = save()
    def plan(roles):
        return {'recipe': {'stages': [{'id': 'full', 'kind': 'native_state_sft',
            'inputs': {k: v['binding'] for k, v in roles.items()}}],
            'input_bindings': {v['binding']: {'path':v['path'], 'sha256':v['sha256']} for v in roles.values()}},
            'stage_inputs': {'full': roles}}
    before, after = plan(old), plan(new)
    changes = recipe.unstarted_stage_input_changes(before, after, tmp_path, started_stage_ids=('full',))
    assert changes['full']['kind'] == 'metadata_classification_continuation'
    assert recipe._without_unstarted_bindings(before['recipe'], after['recipe'], changes)[0] == \
           recipe._without_unstarted_bindings(before['recipe'], after['recipe'], changes)[1]
    for role in old:
        changed = copy.deepcopy(new); changed[role]['sha256'] = 'a'*64
        assert not recipe._classification_input_continuation(old, changed)
    assert not recipe._classification_input_continuation(new, old)
    out = tmp_path/'run'; frozen = out/'runtime'/'natlang_neuralese'; frozen.mkdir(parents=True)
    (frozen/'old.py').write_text('old = True')
    package = tmp_path/'new'; package.mkdir(); (package/'new.py').write_text('new = True')
    for value in (before, after):
        value.update(device='cuda', recipe_sha256='original-lineage',
            inputs={r['path']:r['sha256'] for r in value['stage_inputs']['full'].values()})
    before['code'] = {'old.py': recipe.sha(frozen/'old.py')}
    updated = recipe.lineage_handoff(before, after, frozen, package, out/'recipe-plan.json', out,
        dry_run=True, started_stage_ids=('full',))
    assert updated['recipe_sha256'] == 'original-lineage'
    assert updated['stage_inputs']['full'] == new
    assert updated['handoffs'][-1]['stage_input_changes']['full']['kind'] == 'metadata_classification_continuation'
    (out/'full-report.json').write_text('{}')
    with pytest.raises(ValueError, match='inputs of started stage'):
        recipe.unstarted_stage_input_changes(before, after, out, started_stage_ids=('full',))
    other = copy.deepcopy(after); other['recipe']['stages'][0]['kind'] = 'native_state_rl'
    with pytest.raises(ValueError, match='inputs of started stage'):
        recipe.unstarted_stage_input_changes(before, other, tmp_path, started_stage_ids=('full',))


@pytest.mark.parametrize('fault', ['train', 'held', 'producer', 'cache', 'updates', 'policy', 'empty'])
def test_classification_input_rejects_changed_authority_before_handoff(tmp_path, monkeypatch, fault):
    old, save, proof, part = prepared(tmp_path, monkeypatch)
    if fault == 'train': proof['train_sha256'] = 'a'*64
    elif fault == 'held': proof['held']['sha256'] = 'a'*64
    elif fault == 'producer': part['producer']['path'] += '.other'
    elif fault == 'cache': part['new_index']['sha256'] = 'a'*64
    elif fault == 'updates': proof['optimizer_updates'] = 1
    elif fault == 'policy': part['old_policy'].pop(next(iter(part['old_policy'])))
    elif fault == 'empty': proof['indexes'] = []
    with pytest.raises(ValueError, match='classification'):
        recipe._classification_input_continuation(old, save())


def test_ce_continuation_composes_real_same_source_classification_proof(tmp_path, monkeypatch):
    from natlang_neuralese.train.native_supervision import SCHEMA
    old, save, proof, part = prepared(tmp_path, monkeypatch)
    policy = tmp_path / 'ce.json'
    policy.write_text(json.dumps({'schema': SCHEMA, 'require_eligible_sources': True,
        'sources': [{'sha256': old['native_graphs']['sha256'],
                     'encoding': 'native-row-objectives-v1', 'context_weight': .05}]}))
    new = {**save(), 'ce_supervision_policy':
        {**migration._binding(policy), 'binding': 'ce'}}
    assert recipe._ce_supervision_input_continuation(old, new)
    # Existing source and proof verifiers remain authoritative when composed.
    for role in ('native_graphs', 'held_native_graphs'):
        changed = copy.deepcopy(new)
        changed[role]['sha256'] = 'a' * 64
        assert not recipe._ce_supervision_input_continuation(old, changed)
    changed = copy.deepcopy(new)
    changed['source_classification_migration']['sha256'] = 'a' * 64
    with pytest.raises(ValueError):
        recipe._ce_supervision_input_continuation(old, changed)
    proof['optimizer_updates'] = 1
    changed = {**save(), 'ce_supervision_policy': new['ce_supervision_policy']}
    with pytest.raises(ValueError, match='classification'):
        recipe._ce_supervision_input_continuation(old, changed)


def test_historical_classification_readers_may_relocate_without_changing_sources(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_classification_continuation as continuation
    from scripts.neuralese_data import continue_native_source_indexes as adapter
    _, save, _, _ = prepared(tmp_path, monkeypatch)
    old = save()
    role = 'source_classification_migration'
    readers = []
    for module in (continuation, adapter):
        source = tmp_path / 'historical' / Path(module.__file__).name
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(Path(module.__file__).read_bytes())
        readers.append(migration._binding(source))
    data = {k:old['native_graphs'][k] for k in ('path', 'sha256')}
    old[role]['dependencies'] = [data, *readers]
    new = copy.deepcopy(old)
    new[role]['dependencies'] = [data, *[migration._binding(m.__file__) for m in (continuation, adapter)]]
    assert recipe._classification_reader_relocation(old, new)
    # A reader already present through another proof can be deduplicated.
    old[role]['dependencies'] = [data, readers[0]]
    assert recipe._classification_reader_relocation(old, new)
    for fault in ('data-dependency', 'held', 'proof', 'reader-bytes'):
        changed = copy.deepcopy(new)
        if fault == 'data-dependency':
            changed[role]['dependencies'] = changed[role]['dependencies'][1:]
        elif fault == 'held':
            changed['held_native_graphs']['sha256'] = 'a' * 64
        elif fault == 'proof':
            changed[role]['sha256'] = 'a' * 64
        else:
            wrong = tmp_path / Path(continuation.__file__).name
            wrong.write_text('# different implementation\n')
            changed[role]['dependencies'][1] = migration._binding(wrong)
        assert not recipe._classification_reader_relocation(old, changed)


def test_recipe_proof_checks_shared_policy_bindings_once_per_call(tmp_path, monkeypatch):
    from collections import Counter
    from natlang_neuralese.train import native_source_classification as source
    _, save, proof, part = prepared(tmp_path, monkeypatch)
    proof['indexes'] = [copy.deepcopy(part) for _ in range(100)]
    proof['held_indexes'] = [copy.deepcopy(part) for _ in range(100)]
    roles = save()
    calls = Counter()
    original = source._checked
    def checked(ref):
        calls[ref['path']] += 1
        return original(ref)
    monkeypatch.setattr(source, '_checked', checked)
    assert recipe._classification_proof(roles) == proof
    unique = {part[key]['path'] for key in ('source', 'old_index', 'new_index', 'producer')}
    unique.update(item['path'] for item in part['old_policy'].values())
    assert {path: calls[path] for path in unique} == {path: 1 for path in unique}
    calls.clear()
    assert recipe._classification_proof(roles) == proof
    assert {path: calls[path] for path in unique} == {path: 1 for path in unique}


@pytest.mark.parametrize('field,value', [('sha256', '0' * 64), ('bytes', 1)])
def test_recipe_proof_rejects_contradictory_shared_file(tmp_path, monkeypatch, field, value):
    _, save, proof, part = prepared(tmp_path, monkeypatch)
    second = copy.deepcopy(part)
    second['new_index'][field] = value
    proof['indexes'].append(second)
    with pytest.raises(ValueError):
        recipe._classification_proof(save())


def test_recipe_proof_rejects_rewritten_shared_policy_during_call(tmp_path, monkeypatch):
    import os
    from natlang_neuralese.train import native_source_classification as source
    _, save, _, part = prepared(tmp_path, monkeypatch)
    roles = save()
    policy = Path(next(iter(part['old_policy'].values()))['path'])
    original = source._checked
    def checked(ref):
        result = original(ref)
        if ref['path'] == policy.as_posix():
            state = policy.stat()
            raw = policy.read_bytes()
            policy.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
            os.utime(policy, ns=(state.st_atime_ns, state.st_mtime_ns))
        return result
    monkeypatch.setattr(source, '_checked', checked)
    with pytest.raises(ValueError):
        recipe._classification_proof(roles)
