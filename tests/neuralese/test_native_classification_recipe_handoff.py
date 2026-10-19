import copy
import json

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
