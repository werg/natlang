import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.recipe import (deep_merge, load_recipe, recipe_identity_sha256,
                                            resolve_recipe_data)

ROOT = Path(__file__).parents[2]
RECIPES = ROOT / 'training/neuralese/recipes'
CONVERTED = ['c12-ctx512', 'c12-ctx1k', 'c18', 'c18-ctx512', 'c18-ctx1k', 'c21-ctx512', 'c23-ctx512',
             'c23-ctx1k', 'c23-calibrated-v1']
# Verbatim pre-conversion copies of the original files (taken at commit 8a96d758).
ORIGINALS = Path(__file__).parent / 'recipe_originals'


def base_recipe():
    return {
        'schema': 'natlang.neuralese-training-recipe/1', 'id': 'base-r', 'description': 'base',
        'stages': [
            {'id': 'token_identity', 'kind': 'token_identity', 'requires': [],
             'parameters': {'limit': 32, 'max_tokens': 512}},
            {'id': 'embedding_distillation', 'kind': 'causal_embedding_distillation',
             'requires': ['token_identity'], 'parameters': {'cutoff': 12, 'steps': 8, 'batch': 2}},
        ]}


def write(directory, name, value):
    path = Path(directory) / (name + '.json')
    path.write_text(json.dumps(value))
    return path


def child(**overrides):
    return {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'child-r', 'extends': 'base-r',
            'description': 'child', 'overrides': overrides}


def test_deep_merge_objects_merge_lists_and_scalars_replace_and_delete_marker_removes():
    base = {'a': {'x': 1, 'y': [1, 2]}, 'b': 1, 'c': 2}
    merged = deep_merge(base, {'a': {'y': [3], 'z': None}, 'b': {'$delete': True}})
    assert merged == {'a': {'x': 1, 'y': [3], 'z': None}, 'c': 2}
    assert base == {'a': {'x': 1, 'y': [1, 2]}, 'b': 1, 'c': 2}  # inputs untouched
    with pytest.raises(ValueError):
        deep_merge({}, {'missing': {'$delete': True}})


def test_stage_overrides_merge_by_stage_id_and_lists_replace(tmp_path):
    write(tmp_path, 'base', base_recipe())
    path = write(tmp_path, 'child', child(stages={
        'embedding_distillation': {'parameters': {'cutoff': 18, 'seed': 3}, 'requires': ['token_identity', 'token_identity']},
        'token_identity': {'parameters': {'limit': {'$delete': True}}}}))
    resolved = resolve_recipe_data(path)
    stages = {s['id']: s for s in resolved['stages']}
    assert stages['embedding_distillation']['parameters'] == {'cutoff': 18, 'steps': 8, 'batch': 2, 'seed': 3}
    assert stages['embedding_distillation']['requires'] == ['token_identity', 'token_identity']  # replaced
    assert stages['token_identity']['parameters'] == {'max_tokens': 512}
    assert [s['id'] for s in resolved['stages']] == ['token_identity', 'embedding_distillation']
    assert resolved['id'] == 'child-r' and resolved['description'] == 'child'
    assert 'extends' not in resolved and 'overrides' not in resolved


def test_top_level_override_and_unknown_stage_and_forbidden_keys(tmp_path):
    write(tmp_path, 'base', base_recipe())
    resolved = resolve_recipe_data(write(tmp_path, 'c1', child(scope={'a': 1})))
    assert resolved['scope'] == {'a': 1}
    for bad in (child(stages={'nope': {'parameters': {}}}), child(id='x'), child(schema='y'),
                child(stages={'token_identity': {'id': 'renamed'}})):
        with pytest.raises(ValueError):
            resolve_recipe_data(write(tmp_path, 'bad', bad))
    with pytest.raises(ValueError):
        resolve_recipe_data(write(tmp_path, 'orphan', {**base_recipe(), 'id': 'o', 'overrides': {}}))


def test_unknown_parent_and_cycles_fail(tmp_path):
    write(tmp_path, 'child', child())
    with pytest.raises(ValueError, match='exactly one'):
        resolve_recipe_data(tmp_path / 'child.json')
    a = {**base_recipe(), 'id': 'a', 'extends': 'b'}
    b = {**base_recipe(), 'id': 'b', 'extends': 'c'}
    c = {**base_recipe(), 'id': 'c', 'extends': 'a'}
    for name, value in (('a', a), ('b', b), ('c', c)):
        write(tmp_path, name, value)
    with pytest.raises(ValueError, match='cycle'):
        resolve_recipe_data(tmp_path / 'a.json')
    write(tmp_path, 'self', {**base_recipe(), 'id': 'self', 'extends': 'self'})
    with pytest.raises(ValueError, match='cycle'):
        resolve_recipe_data(tmp_path / 'self.json')


def test_handler_whitelist_applies_after_resolution(tmp_path):
    write(tmp_path, 'base', base_recipe())
    good = write(tmp_path, 'good', child(stages={'embedding_distillation': {'parameters': {'seed': 1}}}))
    assert load_recipe(good)['stages'][1]['parameters']['seed'] == 1
    bad = write(tmp_path, 'bad', child(stages={'embedding_distillation': {'parameters': {'not_a_parameter': 1}}}))
    with pytest.raises(ValueError, match='unknown stage parameters'):
        load_recipe(bad)


def test_hash_is_canonical_over_resolved_content_and_stable(tmp_path):
    write(tmp_path, 'base', base_recipe())
    path = write(tmp_path, 'child', child(stages={'embedding_distillation': {'parameters': {'seed': 1}}}))
    first = recipe_identity_sha256(path)
    assert first == canonical_json_sha256_hex(resolve_recipe_data(path))
    # key order and whitespace in the file do not matter; resolved content does
    data = json.loads(path.read_text())
    path.write_text(json.dumps(data, indent=4, sort_keys=True))
    assert recipe_identity_sha256(path) == first
    data['overrides']['stages']['embedding_distillation']['parameters']['seed'] = 2
    path.write_text(json.dumps(data))
    assert recipe_identity_sha256(path) != first
    # a plain recipe keeps its file-bytes hash so earlier receipts stay comparable
    plain = write(tmp_path, 'plain', {**base_recipe(), 'id': 'plain'})
    import hashlib
    assert recipe_identity_sha256(plain) == hashlib.sha256(plain.read_bytes()).hexdigest()


def test_resolve_cli_prints_recipe_and_hash(tmp_path):
    out = subprocess.run([sys.executable, '-m', 'natlang_neuralese.train.recipe', 'resolve',
                          'token-preserving-foundation-maple-c18-ctx512'],
                         capture_output=True, text=True, check=True, cwd=ROOT,
                         env={**__import__('os').environ, 'PYTHONPATH': str(ROOT / 'training/neuralese')}).stdout
    lines = out.strip().splitlines()
    summary = json.loads(lines[-1])
    recipe = json.loads('\n'.join(lines[:-1]))
    assert summary['recipe_sha256'] == canonical_json_sha256_hex(recipe) and summary['inherits'] is True
    assert recipe['stages'][1]['parameters']['cutoff'] == 18


@pytest.mark.parametrize('name', CONVERTED)
def test_converted_recipe_resolves_to_the_original_content(name):
    original = json.loads((ORIGINALS / f'foundation-maple-{name}.json').read_text())
    path = RECIPES / f'foundation-maple-{name}.json'
    assert 'extends' in json.loads(path.read_text())
    resolved = load_recipe(path)
    ignore = {'extends', 'overrides', 'description'}
    assert {k: v for k, v in resolved.items() if k not in ignore} == {k: v for k, v in original.items() if k not in ignore}
    assert resolved['id'] == original['id']


def test_history_keeps_every_original_description():
    history = (RECIPES / 'HISTORY.md').read_text()
    for name in CONVERTED:
        original = json.loads((ORIGINALS / f'foundation-maple-{name}.json').read_text())
        assert original['description'] in history and original['id'] in history


def test_harness_bench_cohort_is_declared_held_and_shared_by_both_lines():
    """raw-recurrence-v3 / raw-recurrence-mellum-v2 add only the harness_bench cohort to v2 / mellum-v1, plus the text
    warm-up's cohort/context defaults made explicit (the text_warmup CLI defaults; declared in v3, not v1, so earlier
    recipes keep their identity)."""
    ignore = {'id', 'description', 'cohorts', 'stage_parameter_defaults'}
    explicit = {'context_weight': 1.0, 'feedback_weight': 0.25, 'qualification_cohort': 'native'}
    for new, old in (('raw-recurrence-v3', 'raw-recurrence-v2'), ('raw-recurrence-mellum-v2', 'raw-recurrence-mellum-v1')):
        resolved, previous = load_recipe(RECIPES / f'{new}.json'), load_recipe(RECIPES / f'{old}.json')
        assert {k: v for k, v in resolved.items() if k not in ignore} == \
            {k: v for k, v in previous.items() if k not in ignore}
        assert 'cohorts' not in previous
        defaults = previous['stage_parameter_defaults']
        assert resolved['stage_parameter_defaults'] == {
            **defaults, 'core_text_warmup': {**defaults['core_text_warmup'], **explicit}}
        assert not set(explicit) & set(defaults['core_text_warmup'])
    lfm, mellum = load_recipe(RECIPES / 'raw-recurrence-v3.json'), load_recipe(RECIPES / 'raw-recurrence-mellum-v2.json')
    cohort = lfm['cohorts']['harness_bench']
    assert mellum['cohorts'] == lfm['cohorts']
    # Owner decision 2026-10-10: admitted, effective per twin and stage as the trainer supports it.
    assert cohort['admitted'] is True and 'DECISIONS.md 2026-10-10' in cohort['admission']
    assert cohort['recurrence']['admitted'] is False
    assert cohort['recurrence']['trajectory_trainer']['view'] == 'written'
    registry = {c['id']: c for c in json.loads((ROOT / 'training/neuralese_corpora.json').read_text())['corpora']}
    assert registry[cohort['source']['corpus']]['training_admission'] is False
    for twin in cohort['twins'].values():
        entry = registry[twin['corpus']]
        assert entry['training_admission'] is twin['admitted']
        assert 'DECISIONS.md 2026-10-10' in entry['admission'] and 'DECISIONS.md 2026-10-10' in twin['admission']
        assert entry['build']['text_jsonl_sha256'] == twin['text_jsonl_sha256']
        assert cohort['source']['corpus'] in entry['derived_from']


def test_view_operator_stage_is_declared_held_and_shared_by_both_lines():
    """raw-recurrence-v4 / raw-recurrence-mellum-v3 add only the held view operator stage and its gate to v3 /
    mellum-v2 (stages_added after adapted_runtime; recurrence_warmup additionally requires the gate)."""
    added = {'view_operator', 'view_gate'}
    ignore = {'id', 'description', 'input_bindings', 'view_operator', 'stages', 'cohorts'}
    for new, old in (('raw-recurrence-v4', 'raw-recurrence-v3'), ('raw-recurrence-mellum-v3', 'raw-recurrence-mellum-v2')):
        resolved, previous = load_recipe(RECIPES / f'{new}.json'), load_recipe(RECIPES / f'{old}.json')
        assert {k: v for k, v in resolved.items() if k not in ignore} == \
            {k: v for k, v in previous.items() if k not in ignore}
        # The harness cohort's recurrence gains only the declared view length (an addition).
        recurrence = copy.deepcopy(previous['cohorts']['harness_bench']['recurrence'])
        recurrence['trajectory_trainer']['view_tokens_per_vector'] = 4
        recurrence['view_length'] = resolved['cohorts']['harness_bench']['recurrence']['view_length']
        expected = copy.deepcopy(previous['cohorts'])
        expected['harness_bench']['recurrence'] = recurrence
        assert resolved['cohorts'] == expected
        assert 'view_operator' not in previous and 'input_bindings' not in previous
        old_stages = {s['id']: s for s in previous['stages']}
        new_stages = [s for s in resolved['stages'] if s['id'] not in added]
        assert [s['id'] for s in new_stages] == [s['id'] for s in previous['stages']]
        for stage in new_stages:
            expected = dict(old_stages[stage['id']])
            if stage['id'] == 'recurrence_warmup':
                expected['requires'] = expected['requires'] + ['view_gate']
            assert stage == expected
        ids = [s['id'] for s in resolved['stages']]
        assert ids.index('adapted_runtime') + 1 == ids.index('view_operator') == ids.index('view_gate') - 1
        view = next(s for s in resolved['stages'] if s['id'] == 'view_operator')
        gate = next(s for s in resolved['stages'] if s['id'] == 'view_gate')
        assert view['admitted'] is False and gate['admitted'] is False and resolved['view_operator']['admitted'] is False
        assert view['kind'] == 'raw_recurrence_training' and gate['kind'] == 'view_operator_gate'
        assert view['parameters']['view'] == 'written' and view['parameters']['purpose_contrast'] > 0
        assert view['parameters']['distill'] == 1.0 and view['parameters']['stop_pg'] > 0
        assert gate['inputs']['records'] == view['inputs']['records']
    lfm, mellum = load_recipe(RECIPES / 'raw-recurrence-v4.json'), load_recipe(RECIPES / 'raw-recurrence-mellum-v3.json')
    assert lfm['view_operator'] == mellum['view_operator'] and lfm['input_bindings'] == mellum['input_bindings']
    lfm_view = next(s for s in lfm['stages'] if s['id'] == 'view_operator')['parameters']
    mellum_view = next(s for s in mellum['stages'] if s['id'] == 'view_operator')['parameters']
    mellum_rec = next(s for s in mellum['stages'] if s['id'] == 'recurrence_warmup')['parameters']
    lfm_rec = next(s for s in lfm['stages'] if s['id'] == 'recurrence_warmup')['parameters']
    # The only differences between the lines' view stages are the Mellum line's declared recurrence overrides.
    assert {k: v for k, v in mellum_view.items() if lfm_view.get(k) != v} == \
        {k: v for k, v in mellum_rec.items() if lfm_rec.get(k) != v}


def test_harness_bench_cohort_repinned_to_s1_placed_records_v4():
    """raw-recurrence-v5 / raw-recurrence-mellum-v4 change only the harness_bench pins of v4 / mellum-v3: records v4
    (splits follow S1's cross-corpus placement) and its v2 twins, with the Mellum twin admitted for the text stages."""
    v4_records = 'harness-bench-swe-rebench-openhands-pi-records-20261010-v4'
    registry = {c['id']: c for c in json.loads((ROOT / 'training/neuralese_corpora.json').read_text())['corpora']}
    changed = {'id', 'description', 'input_bindings', 'view_operator', 'stages', 'cohorts'}
    pairs = (('raw-recurrence-v5', 'raw-recurrence-v4'), ('raw-recurrence-mellum-v4', 'raw-recurrence-mellum-v3'))
    for new, old in pairs:
        resolved, previous = load_recipe(RECIPES / f'{new}.json'), load_recipe(RECIPES / f'{old}.json')
        assert {k: v for k, v in resolved.items() if k not in changed} == \
            {k: v for k, v in previous.items() if k not in changed}
        bindings = dict(previous['input_bindings'])
        del bindings['harness-bench-records-v3'], bindings['harness-bench-pieces-v3']
        assert {k: v for k, v in resolved['input_bindings'].items() if not k.startswith('harness-bench-')} == bindings
        assert resolved['input_bindings']['harness-bench-records-v4']['path'].endswith(f'{v4_records}/records.jsonl')
        manifest = {f['path']: f['sha256'] for f in
                    json.loads((ROOT / f'training/corpus-manifests/{v4_records}.json').read_text())['files']}
        assert resolved['input_bindings']['harness-bench-records-v4']['sha256'] == manifest['records.jsonl']
        assert resolved['input_bindings']['harness-bench-pieces-v4']['sha256'] == manifest['pieces.jsonl']
        gate = copy.deepcopy(previous['view_operator'])
        gate['gate']['metrics'] = [m.replace('harness-bench v3', 'harness-bench v4') for m in gate['gate']['metrics']]
        assert resolved['view_operator'] == gate
        for stage, before in zip(resolved['stages'], previous['stages']):
            if stage['id'] == 'view_gate':
                before = {**before, 'inputs': {**before['inputs'], 'harness_records': 'harness-bench-records-v4',
                                               'harness_pieces': 'harness-bench-pieces-v4'}}
            assert stage == before
        cohort, before = resolved['cohorts']['harness_bench'], previous['cohorts']['harness_bench']
        assert cohort['source'] == {**before['source'], 'corpus': v4_records}
        assert cohort['recurrence'] == {**before['recurrence'], 'inputs': {**before['recurrence']['inputs'],
                                                                           'records': v4_records, 'pieces': v4_records}}
        assert cohort['recurrence']['admitted'] is False and cohort['admitted'] is True
        assert cohort['text_stages'] == before['text_stages'] and cohort['backbone_inherent_differences'] == before['backbone_inherent_differences']
        for name, twin in cohort['twins'].items():
            entry = registry[twin['corpus']]
            assert twin['admitted'] is True and entry['training_admission'] is True
            assert entry['derived_from'] == [v4_records] and entry['build']['text_jsonl_sha256'] == twin['text_jsonl_sha256']
            assert entry['build']['tokenizer_sha256'] == twin['tokenizer_sha256'] == before['twins'][name]['tokenizer_sha256']
            assert entry['build']['history_reasoning']['policy'] == twin['history_reasoning'] == before['twins'][name]['history_reasoning']
            assert twin['tokens'] == {split: entry['tokens'][split]['tokens'] for split in ('train', 'test')}
            # Same documents, other splits: the token total is the v1 twin's.
            assert sum(twin['tokens'].values()) == sum(before['twins'][name]['tokens'].values())
            assert registry[before['twins'][name]['corpus']]['superseded_by'] == twin['corpus']
        assert registry[before['source']['corpus']]['superseded_by'] == v4_records
        assert registry[v4_records]['training_admission'] is False
    lfm, mellum = load_recipe(RECIPES / 'raw-recurrence-v5.json'), load_recipe(RECIPES / 'raw-recurrence-mellum-v4.json')
    assert lfm['cohorts'] == mellum['cohorts'] and lfm['input_bindings'] == mellum['input_bindings']
    # The Mellum line's differences from the shared recipe are mellum-v3's, unchanged.
    v4, mellum_v3 = load_recipe(RECIPES / 'raw-recurrence-v4.json'), load_recipe(RECIPES / 'raw-recurrence-mellum-v3.json')
    assert json.loads((RECIPES / 'raw-recurrence-mellum-v4.json').read_text())['overrides'] == \
        json.loads((RECIPES / 'raw-recurrence-mellum-v3.json').read_text())['overrides']
    assert [s for s in mellum['stages'] if s not in lfm['stages']] != [] and \
        len([s for s in mellum['stages'] if s not in lfm['stages']]) == len([s for s in mellum_v3['stages'] if s not in v4['stages']])


def test_stages_added_insert_after_a_named_stage(tmp_path):
    (tmp_path / 'base.json').write_text(json.dumps(base_recipe()))
    extra = {'id': 'token_identity_again', 'kind': 'token_identity', 'requires': ['token_identity'],
             'parameters': {'limit': 8, 'max_tokens': 64}}
    child = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'child-r', 'extends': 'base-r',
             'overrides': {'stages_added': [{'after': 'token_identity', 'stage': extra}]}}
    (tmp_path / 'child.json').write_text(json.dumps(child))
    resolved = resolve_recipe_data(tmp_path / 'child.json')
    assert [s['id'] for s in resolved['stages']] == ['token_identity', 'token_identity_again', 'embedding_distillation']
    for bad in ({'after': 'nowhere', 'stage': extra}, {'after': 'token_identity', 'stage': {**extra, 'id': 'token_identity'}},
                {'stage': extra}):
        child['overrides']['stages_added'] = [bad]
        (tmp_path / 'child.json').write_text(json.dumps(child))
        with pytest.raises(ValueError):
            resolve_recipe_data(tmp_path / 'child.json')


def test_v6_line_declares_its_evaluation_and_checkpoint_points_in_steps():
    """Owner 2026-10-10: every write point is an evaluation point and the stage end is the last one (its gate)."""
    from natlang_neuralese.train.loop import check_declared_points
    from natlang_neuralese.train.recipe import resolve_recipe_data
    for name in ('raw-recurrence-v6', 'raw-recurrence-mellum-v5', 'raw-recurrence-mellum-v6', 'raw-recurrence-v7', 'raw-recurrence-mellum-v7', 'raw-recurrence-v8', 'raw-recurrence-mellum-v8', 'raw-recurrence-mellum-v9', 'raw-recurrence-mellum-v10'):
        path = RECIPES / f'{name}.json'
        if not path.exists():
            continue
        recipe = resolve_recipe_data(path)
        defaults = recipe.get('stage_parameter_defaults', {})
        for stage in recipe['stages']:
            effective = {**defaults.get(stage['kind'], {}), **stage.get('parameters', {})}
            if 'eval_every' in effective:
                check_declared_points(effective['steps'], effective['eval_every'], effective.get('checkpoint_every'))
                assert 'checkpoint_minutes' not in effective and 'eval_minutes' not in effective
