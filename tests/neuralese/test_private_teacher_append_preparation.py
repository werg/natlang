"""Small real NativeGraphSource fixtures for the shared append preparer."""
from __future__ import annotations

import hashlib
import json
import copy
import shutil
from pathlib import Path

import pytest
import zstandard

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.train.native_supervision import load_policy
from natlang_neuralese.train import native_resident
from natlang_neuralese.train.recipe import load_recipe
from scripts.neuralese_data.build_native_ce_supervision_policy import build as build_ce
from scripts.neuralese_data.prepare_private_teacher_append import prepare, _seed_source_bindings
from test_native_state_sft import graph


REPO = Path(__file__).resolve().parents[2]


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n')
    return path


def _binding(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha256_file_hex(path)}


def _zstd_jsonl(path, rows):
    payload = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in rows).encode()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(zstandard.ZstdCompressor().compress(payload))
    return Path(path)


def _row(identifier, *, split='train', group=None, token_ids=None, weights=None):
    value = graph(identifier, weights or [0, 1, 1])
    value['split'] = split
    value['source_groups'] = [group or f'group-{identifier}']
    value['token_ids'] = None
    value['nodes'][0]['token_ids'] = token_ids or [1, 2, 3]
    value['tokenizer_fingerprint'] = 'tokenizer-sha'
    return value


def _private_receipt(identity, *, valid=True):
    seen = dict(identity)
    if not valid:
        seen['sha256'] = 'f' * 64
    return {'private': True, 'identity': seen, 'independent_verification': dict(seen),
        'run': 'owner/private-run', 'name': 'teacher/' + identity['sha256'] + '/archive.tar.zst',
        'source': '/immutable/teacher-archive.tar.zst'}


def _make_package(artifact_root, corpus_id, role, rows, history_policy, *, private_valid=True,
                  role_key=None, tokenizer='tokenizer-sha', failure_policy='mask-runtime-failures',
                  effective_history=None, history_probe=None):
    package = artifact_root / 'packages' / corpus_id
    leaf = _zstd_jsonl(package / role / 'train' / 'native-graphs.jsonl.zst', rows)
    identity = {'sha256': hashlib.sha256((corpus_id + '-archive').encode()).hexdigest(), 'bytes': 123}
    if effective_history is None:
        effective_history = history_policy or ('keep' if role == 'mellum21' else None)
    if history_probe is None and effective_history is not None:
        history_probe = {'keep': 'keep', 'last_turn_only': 'drops-history'}.get(effective_history)
    history_record = ({'policy': effective_history,
        'backbone': 'mellum' if role == 'mellum21' else 'lfm2',
        'template_kwargs': {}, 'probe': history_probe} if effective_history is not None else None)
    receipt = {
        'model_key': role_key or role,
        'failed_assistant_history_policy': failure_policy,
        'tokenizer': {'expanded_fingerprint': tokenizer, **({'history_reasoning': history_record} if history_record else {})},
    }
    if role == 'lfm25':
        receipt['explicit_history_reasoning_policy'] = history_policy
    receipt_path = _write_json(package / role / 'receipt.json', receipt)
    manifest_path = artifact_root / 'training' / 'corpus-manifests' / f'{corpus_id}.json'
    _write_json(manifest_path, {'schema': 'test-manifest/1', 'files': [
        {'path': f'{role}/train/native-graphs.jsonl.zst', 'sha256': sha256_file_hex(leaf), 'bytes': leaf.stat().st_size},
        {'path': f'{role}/receipt.json', 'sha256': sha256_file_hex(receipt_path), 'bytes': receipt_path.stat().st_size},
    ]})
    private_path = _write_json(artifact_root / 'exports' / corpus_id / 'private-receipt.json',
                               _private_receipt(identity, valid=private_valid))
    ready_path = _write_json(artifact_root / 'exports' / corpus_id / 'ready.json', {
        'passed': True, 'training_exposure': False, 'corpus_id': corpus_id,
        'entry_sha256': 'pending', 'manifest_sha256': sha256_file_hex(manifest_path),
        'identity': identity, 'private_receipt': str(private_path.resolve()),
    })
    entry = {'id': corpus_id, 'path': str(package.relative_to(artifact_root)), 'training_admission': True}
    ready = json.loads(ready_path.read_text())
    ready['entry_sha256'] = canonical_json_sha256_hex(entry)
    _write_json(ready_path, ready)
    return {'entry': entry, 'ready_path': ready_path, 'leaf': leaf}


def _refresh_manifest_and_ready(descriptor, corpus_id, package_root):
    artifact = Path(descriptor['artifact_repo']['root'])
    role = descriptor['model_role']
    manifest_path = Path(descriptor['artifact_repo']['manifests_root']) / f'{corpus_id}.json'
    manifest = json.loads(manifest_path.read_text())
    for item in manifest['files']:
        path = package_root / item['path']
        item['sha256'] = sha256_file_hex(path)
        item['bytes'] = path.stat().st_size
    _write_json(manifest_path, manifest)
    ready_path = Path(descriptor['ready_export_root']) / corpus_id / 'ready.json'
    ready = json.loads(ready_path.read_text())
    ready['manifest_sha256'] = sha256_file_hex(manifest_path)
    _write_json(ready_path, ready)


def _fixture(tmp_path, role, *, geometry_limit=16, stage_history=None,
             capacity_limits=None, model_history='keep', optional_sources=False):
    artifact = tmp_path / 'artifact-repo'
    supervision = tmp_path / 'supervision'
    output = tmp_path / 'prepared-output'
    export_root = artifact / 'exports'
    tokenizer_history = 'keep' if role == 'lfm25' else None

    old_leaf = _zstd_jsonl(artifact / 'retained' / 'old-train.jsonl.zst', [_row('old-a'), _row('old-b')])
    held_leaf = _zstd_jsonl(artifact / 'retained' / 'held.jsonl.zst', [_row('held-a', split='test', group='protected')])
    old_tree_path = _write_json(artifact / 'retained' / 'native-training-sources.json', {
        'schema': 'natlang.native-graph-sources/1',
        'sources': [{'path': str(old_leaf.resolve()), 'sha256': sha256_file_hex(old_leaf),
                     'bytes': old_leaf.stat().st_size, 'kind': 'native-graphs', 'epochs': 1, 'weight': 1.0}],
    })
    _write_json(old_tree_path.parent / '.natlang-source-map.json', {'schema': 'natlang.staged-source-map/1', 'paths': {}})
    policy = build_ce([old_tree_path], legacy_pair_context_weight=.05, held_sources=[held_leaf])
    policy['source_contract']['inventory'] = [_binding(old_tree_path), _binding(held_leaf)]
    ce_path = _write_json(artifact / 'retained' / 'ce-policy.json', policy)
    from natlang_neuralese.data.native_sources import NativeGraphSource
    old_source = NativeGraphSource(old_tree_path, 'tokenizer-sha', required_split='train')
    old_indexes = native_resident._source_witness(old_source, REPO / 'training' / 'neuralese' / 'natlang_neuralese')
    memory = {'schema': 'test-memory-request/1', 'source': _binding(old_tree_path),
        'identity': {'model_config': {'tokenizer_fingerprint': 'tokenizer-sha',
                                      'marker_ids': [], 'backbone': 'mellum' if role == 'mellum21' else 'lfm2',
                                      **({'history_reasoning_policy': model_history} if model_history is not None else {})}},
        'runtime_code_sha256': 'a' * 64,
        'cases': [{'id': 'real-envelope-case', 'forward_tokens': geometry_limit, 'kv_slots': geometry_limit}],
        'reference': {'id': 'actual-reference', 'binding': 'b' * 64, 'index': 0}, 'indexes': old_indexes,
        'scope': 'Existing measured-envelope test fixture; real entry checks remain required.'}
    memory_refs = []
    for index, (token_limit, kv_limit) in enumerate(capacity_limits or [(geometry_limit, geometry_limit)]):
        request = copy.deepcopy(memory)
        request['cases'] = [{'id': f'real-envelope-case-{index}', 'forward_tokens': token_limit,
                             'kv_slots': kv_limit}]
        memory_path = _write_json(supervision / f'memory-request-{index:02d}.json', request)
        memory_refs.append(_binding(memory_path))
    parameters = {'epochs': 1, 'checkpoint_every': 25, 'eval_every': 50,
                  'memory_qualification': memory_refs if len(memory_refs) > 1 else memory_refs[0]}
    if stage_history is not None:
        parameters['history_reasoning_policy'] = stage_history
    bindings = {'train': _binding(old_tree_path), 'held': _binding(held_leaf), 'ce': _binding(ce_path)}
    stage_inputs = {'native_graphs': 'train', 'held_native_graphs': 'held', 'ce_supervision_policy': 'ce'}
    if optional_sources:
        bindings.update({'foundation': _binding(old_tree_path), 'continued': _binding(old_tree_path),
                         'continued_held': _binding(held_leaf)})
        stage_inputs.update({'foundation_native_graphs': 'foundation', 'continued_native_graphs': 'continued',
                             'continued_held_native_graphs': 'continued_held'})
    recipe = {'schema': 'natlang.neuralese-training-recipe/1', 'id': f'{role}-fixture',
        'identity_qualification': 'native_state_runtime_qualified', 'input_bindings': bindings,
        'stages': [{'id': 'native_full_sft', 'kind': 'native_state_sft', 'requires': [],
                    'parameters': parameters, 'inputs': stage_inputs}],
        'remaining_production_recipe': {'private_teacher_append_20261025': {
            'active_rows': 2, 'new_rows': 2, 'existing_rows_unchanged': True,
            'held_unchanged': True, 'training_exposure': False,
        }}}
    recipe_path = _write_json(supervision / 'recipe.json', recipe)
    config = {'production_recipe': str(recipe_path.resolve()), 'recipe_lineage_sha256': 'c' * 64,
        'checkpoint': str(supervision / 'checkpoint.pt'),
        'seed_command': ['natlang_neuralese.train.native_state_sft', '{trainer_args}'],
        'trainer_parameters': {'native_attention_precision': 'ieee'},
        'seed_trainer_parameters': {'stage_id': 'native_full_sft', 'checkpoint_every': 50,
                                    'eval_every': 50, 'native_attention_precision': 'ieee'},
    }
    config_path = _write_json(supervision / 'config.json', config)
    plan_path = _write_json(supervision / 'plan.json', {'schema': 'test-plan/1', 'content': 'same'})
    packages = []
    packages.append(_make_package(artifact, 'append-partial', role,
        [_row('old-a'), _row('new-c')], tokenizer_history))
    packages.append(_make_package(artifact, 'append-duplicates-only', role,
        [_row('old-b')], tokenizer_history))
    packages.append(_make_package(artifact, 'append-whole-source-equal', role,
        [_row('old-a'), _row('old-b')], tokenizer_history))
    packages.append(_make_package(artifact, 'append-fully-new', role,
        [_row('new-d')], tokenizer_history))
    _write_json(artifact / 'training' / 'neuralese_corpora.json', {'corpora': [x['entry'] for x in packages]})
    descriptor = {'schema': 'natlang.private-teacher-append-descriptor/1',
        'runtime_root': str(REPO),
        'artifact_repo': {'root': str(artifact), 'registry_path': str(artifact / 'training' / 'neuralese_corpora.json'),
                          'manifests_root': str(artifact / 'training' / 'corpus-manifests')},
        'supervision': {'root': str(supervision), 'config_path': str(config_path), 'plan_path': str(plan_path)},
        'ready_export_root': str(export_root), 'model_role': role, 'output_root': str(output),
        'stage_id': 'native_full_sft', 'append_key': 'private_teacher_append_20261026'}
    return descriptor, config_path, plan_path, packages, old_tree_path, ce_path


@pytest.mark.parametrize('role,stage_history', [('lfm25', 'keep'), ('mellum21', None)])
def test_real_native_graph_append_deduplicates_without_changing_old_history_or_cas(
        tmp_path, monkeypatch, role, stage_history):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, config_path, plan_path, packages, old_tree_path, old_ce_path = _fixture(
        tmp_path, role, stage_history=stage_history)
    before_config, before_plan = config_path.read_bytes(), plan_path.read_bytes()
    old_recipe = load_recipe(Path(json.loads(config_path.read_text())['production_recipe']))
    old_policy = json.loads(old_ce_path.read_text())

    result = prepare(descriptor)

    assert result['passed'] is True and result['deployed'] is False
    assert result['ready_packages'] == 4
    assert result['selected_sources'] == 2
    assert result['duplicate_only_sources'] == 1
    assert result['already_active_sources'] == 1
    assert result['selection_proofs'] == 3
    assert result['new_rows'] == 2 and result['active_rows'] == 4
    assert config_path.read_bytes() == before_config and plan_path.read_bytes() == before_plan

    out = Path(descriptor['output_root'])
    prior_tree = json.loads(old_tree_path.read_text())
    appended_tree = json.loads((out / 'native-training-sources.json').read_text())
    assert appended_tree['sources'][:len(prior_tree['sources'])] == prior_tree['sources']
    assert all(item['training_priority'] == 1 for item in appended_tree['sources'][len(prior_tree['sources']):])
    next_recipe = load_recipe(out / 'recipe.json')
    assert next_recipe['stages'][0]['parameters']['urgent_cohort_fairness'] is True
    assert next_recipe['input_bindings']['train']['path'] == str((out / 'native-training-sources.json').resolve())
    assert next_recipe['input_bindings']['held'] == old_recipe['input_bindings']['held']
    assert next_recipe['remaining_production_recipe']['private_teacher_append_history'][0]['append'] == \
        old_recipe.get('remaining_production_recipe', {}).get('private_teacher_append_20261025')
    assert next_recipe['remaining_production_recipe']['private_teacher_append_20261026']['training_exposure'] is False
    old_memory = json.loads(Path(old_recipe['stages'][0]['parameters']['memory_qualification']['path']).read_text())
    new_memory_ref = next_recipe['stages'][0]['parameters']['memory_qualification']
    new_memory = json.loads(Path(new_memory_ref['path']).read_text())
    assert new_memory['cases'] == old_memory['cases']
    assert new_memory['indexes'][:len(old_memory['indexes'])] == old_memory['indexes']
    assert len(new_memory['indexes']) == len(old_memory['indexes']) + 2
    new_policy = load_policy(_binding(out / 'ce-supervision-policy.json'))
    new_sources = {item['sha256']: item for item in new_policy['sources']}
    assert all(new_sources[item['sha256']] == item for item in old_policy['sources'])
    prior_inventory = {(x['path'], x['sha256']) for x in old_policy['source_contract']['inventory']}
    new_inventory = {(x['path'], x['sha256']) for x in new_policy['source_contract']['inventory']}
    assert prior_inventory <= new_inventory
    assert (str(old_ce_path.resolve()), sha256_file_hex(old_ce_path)) in new_inventory
    assert (str((out / 'ready-source-census.json').resolve()), sha256_file_hex(out / 'ready-source-census.json')) in new_inventory
    assert len(json.loads((out / 'ready-source-census.json').read_text())['records']) == 4

    next_config = json.loads((out / 'next-config.json').read_text())
    assert next_config['trainer_parameters']['urgent_cohort_fairness'] is True
    assert next_config['seed_trainer_parameters']['urgent_cohort_fairness'] is True
    assert next_config['seed_trainer_parameters']['checkpoint_every'] == 50
    assert next_config['seed_trainer_parameters']['native_attention_precision'] == 'ieee'
    assert ('history_reasoning_policy' in next_config['seed_trainer_parameters']) == (role == 'lfm25')
    assert next_config['seed_trainer_parameters'].get('history_reasoning_policy') == (('keep' if role == 'lfm25' else None))
    assert next_config['production_recipe'] == str((out / 'recipe.json').resolve())
    assert result['prior_config']['sha256'] == hashlib.sha256(before_config).hexdigest()
    assert result['prior_plan']['sha256'] == hashlib.sha256(before_plan).hexdigest()
    assert next_config['checkpoint'] == json.loads(before_config)['checkpoint']


def test_append_rebind_preserves_every_resolved_optional_seed_source(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'mellum21', stage_history=None, optional_sources=True)
    result = prepare(descriptor)
    next_recipe = load_recipe(Path(result['recipe']['path']))
    next_config = json.loads(Path(result['next_config']['path']).read_text())
    input_bindings = next_recipe['input_bindings']
    for role in ('foundation_native_graphs', 'continued_native_graphs', 'continued_held_native_graphs'):
        spec = next_recipe['stages'][0]['inputs'][role]
        expected = input_bindings[spec]
        assert next_config[role] == expected['path']
        assert '--' + role.replace('_', '-') in next_config['seed_command']
    # The source merger must also preserve supported optional roles which this
    # small recipe does not need to fully decode (pending/classification).
    refs = {role: {'path': f'/tmp/{role}.json', 'sha256': role[0] * 64}
            for role in ('pending_native_sources', 'source_classification_migration')}
    merged = _seed_source_bindings(refs,
        train={'path': '/tmp/new-train.json', 'sha256': 'a' * 64},
        held={'path': '/tmp/held.json', 'sha256': 'b' * 64},
        ce={'path': '/tmp/ce.json', 'sha256': 'c' * 64})
    assert merged['pending_native_sources'] == refs['pending_native_sources']
    assert merged['source_classification_migration'] == refs['source_classification_migration']


@pytest.mark.parametrize('model_role,history', [('lfm25', 'keep'), ('mellum21', None)])
def test_foundation_sharing_train_binding_follows_append_in_recipe_and_seed(tmp_path, monkeypatch, model_role, history):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, config_path, *_ = _fixture(tmp_path, model_role,
        stage_history=history, optional_sources=True)
    recipe_path = Path(json.loads(config_path.read_text())['production_recipe'])
    recipe = json.loads(recipe_path.read_text())
    recipe['stages'][0]['inputs']['foundation_native_graphs'] = 'train'
    del recipe['input_bindings']['foundation']
    _write_json(recipe_path, recipe)
    result = prepare(descriptor)
    updated = load_recipe(Path(result['recipe']['path']))
    config = json.loads(Path(result['next_config']['path']).read_text())
    assert result['new_rows'] == 2 and result['existing_prefix_unchanged']
    assert config['foundation_native_graphs'] == config['seed_native_graphs']
    assert config['foundation_native_graphs'] == result['native_training_sources']['path']
    assert config['continued_native_graphs'] == recipe['input_bindings']['continued']['path']
    assert updated['input_bindings']['continued'] == recipe['input_bindings']['continued']


def test_conflicting_identity_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'lfm25', stage_history='keep')
    artifact = Path(descriptor['artifact_repo']['root'])
    conflict = _make_package(artifact, 'append-conflict', 'lfm25',
        [_row('old-a', weights=[0, .5, 1])], 'keep')
    registry_path = Path(descriptor['artifact_repo']['registry_path'])
    registry = json.loads(registry_path.read_text())
    registry['corpora'].append(conflict['entry'])
    _write_json(registry_path, registry)
    with pytest.raises(ValueError, match='conflicting admitted graph identity'):
        prepare(descriptor)


def test_registered_package_symlink_to_authenticated_external_storage_is_supported(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'mellum21', stage_history=None)
    artifact = Path(descriptor['artifact_repo']['root'])
    link = artifact / 'packages' / 'append-fully-new'
    external = tmp_path / 'mnt' / 'external' / 'append-fully-new'
    external.parent.mkdir(parents=True)
    shutil.move(str(link), str(external))
    link.symlink_to(external, target_is_directory=True)
    result = prepare(descriptor)
    assert result['passed'] is True
    assert result['new_rows'] == 2


def test_registered_package_path_traversal_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'mellum21', stage_history=None)
    registry_path = Path(descriptor['artifact_repo']['registry_path'])
    registry = json.loads(registry_path.read_text())
    entry = next(item for item in registry['corpora'] if item['id'] == 'append-fully-new')
    entry['path'] = '../../outside-package'
    _write_json(registry_path, registry)
    # Keep ready's canonical entry digest in sync so the namespace guard is the
    # failing condition under test.
    ready_path = Path(descriptor['ready_export_root']) / 'append-fully-new' / 'ready.json'
    ready = json.loads(ready_path.read_text())
    ready['entry_sha256'] = canonical_json_sha256_hex(entry)
    _write_json(ready_path, ready)
    with pytest.raises(ValueError, match='nontraversing artifact-relative'):
        prepare(descriptor)


def test_protected_held_overlap_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'lfm25', stage_history='keep')
    artifact = Path(descriptor['artifact_repo']['root'])
    overlap = _make_package(artifact, 'append-held-overlap', 'lfm25',
        [_row('held-overlap', group='protected')], 'keep')
    registry_path = Path(descriptor['artifact_repo']['registry_path'])
    registry = json.loads(registry_path.read_text()); registry['corpora'].append(overlap['entry']); _write_json(registry_path, registry)
    with pytest.raises(ValueError, match='overlap protected held source groups'):
        prepare(descriptor)


@pytest.mark.parametrize('role,stage_history,receipt_history', [
    ('lfm25', 'keep', None),
    ('mellum21', None, 'last_turn_only'),
])
def test_role_history_authority_rejects_wrong_profile(tmp_path, monkeypatch, role, stage_history, receipt_history):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, role, stage_history=stage_history)
    artifact = Path(descriptor['artifact_repo']['root'])
    bad = _make_package(artifact, 'append-invalid-history', role, [_row('history-new')],
        receipt_history if role == 'lfm25' else None, effective_history=receipt_history)
    registry_path = Path(descriptor['artifact_repo']['registry_path'])
    registry = json.loads(registry_path.read_text()); registry['corpora'].append(bad['entry']); _write_json(registry_path, registry)
    with pytest.raises(ValueError, match='history'):
        prepare(descriptor)


def test_mellum_missing_explicit_history_uses_recorded_policy_and_model_default(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'mellum21', stage_history=None, model_history=None)
    prepare(descriptor)


def test_mellum_contradictory_recorded_history_probe_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'mellum21', stage_history=None)
    artifact = Path(descriptor['artifact_repo']['root'])
    package_root = artifact / 'packages' / 'append-fully-new'
    receipt_path = package_root / 'mellum21' / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert 'explicit_history_reasoning_policy' not in receipt
    receipt['tokenizer']['history_reasoning']['probe'] = 'drops-history'
    _write_json(receipt_path, receipt)
    _refresh_manifest_and_ready(descriptor, 'append-fully-new', package_root)
    with pytest.raises(ValueError, match='history probe contradicts'):
        prepare(descriptor)


@pytest.mark.parametrize('role,stage_history', [('lfm25', 'keep'), ('mellum21', None)])
def test_invalid_private_receipt_is_rejected(tmp_path, monkeypatch, role, stage_history):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, role, stage_history=stage_history)
    artifact = Path(descriptor['artifact_repo']['root'])
    bad = _make_package(artifact, 'append-invalid-private', role, [_row('private-new')],
        'keep' if role == 'lfm25' else None, private_valid=False)
    registry_path = Path(descriptor['artifact_repo']['registry_path'])
    registry = json.loads(registry_path.read_text()); registry['corpora'].append(bad['entry']); _write_json(registry_path, registry)
    with pytest.raises(ValueError, match='independent verification'):
        prepare(descriptor)


@pytest.mark.parametrize('role,stage_history', [('lfm25', 'keep'), ('mellum21', None)])
def test_wrong_role_receipt_is_rejected(tmp_path, monkeypatch, role, stage_history):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, role, stage_history=stage_history)
    artifact = Path(descriptor['artifact_repo']['root'])
    bad = _make_package(artifact, 'append-invalid-role', role, [_row('role-new')],
        'keep' if role == 'lfm25' else None, role_key='other-model')
    registry_path = Path(descriptor['artifact_repo']['registry_path'])
    registry = json.loads(registry_path.read_text()); registry['corpora'].append(bad['entry']); _write_json(registry_path, registry)
    with pytest.raises(ValueError, match='role receipt model_key'):
        prepare(descriptor)


def test_existing_geometry_envelope_rejects_larger_complete_graph_without_truncation(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'lfm25', geometry_limit=3, stage_history='keep')
    artifact = Path(descriptor['artifact_repo']['root'])
    large = _make_package(artifact, 'append-larger-graph', 'lfm25',
        [_row('larger', token_ids=[1, 2, 3, 4], weights=[0, 1, 1, 1])], 'keep')
    registry_path = Path(descriptor['artifact_repo']['registry_path'])
    registry = json.loads(registry_path.read_text()); registry['corpora'].append(large['entry']); _write_json(registry_path, registry)
    with pytest.raises(ValueError, match=r'complete retained graph geometry tokens=4.*at least one existing capacity request'):
        prepare(descriptor)


def test_graph_must_fit_every_bound_memory_qualification_request(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'lfm25', stage_history='keep',
        capacity_limits=[(16, 16), (3, 3)])
    artifact = Path(descriptor['artifact_repo']['root'])
    large = _make_package(artifact, 'append-larger-than-one-request', 'lfm25',
        [_row('larger-one-request', token_ids=[1, 2, 3, 4], weights=[0, 1, 1, 1])], 'keep')
    registry_path = Path(descriptor['artifact_repo']['registry_path'])
    registry = json.loads(registry_path.read_text()); registry['corpora'].append(large['entry']); _write_json(registry_path, registry)
    with pytest.raises(ValueError, match=r'at least one existing capacity request'):
        prepare(descriptor)


def test_staged_source_map_and_transfer_proof_preserve_private_receipt_custody(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'row-index-cache'))
    descriptor, *_ = _fixture(tmp_path, 'mellum21', stage_history=None)
    artifact = Path(descriptor['artifact_repo']['root'])
    ready_path = artifact / 'exports' / 'append-duplicates-only' / 'ready.json'
    ready = json.loads(ready_path.read_text())
    original_path = Path(ready['private_receipt'])
    content = original_path.read_bytes()
    staged_path = ready_path.parent / 'relocated-private-receipt.json'
    staged_path.write_bytes(content)
    original_path.unlink()
    declared_path = '/dgx/immutable/exports/append-duplicates-only/private-receipt.json'
    ready['private_receipt'] = declared_path
    _write_json(ready_path, ready)
    source_map_path = artifact / '.natlang-source-map.json'
    transfer_path = artifact / 'transfer.json'
    staged_rel = 'exports/append-duplicates-only/relocated-private-receipt.json'
    paths, transfer_files = {}, []
    for ready_file in sorted((artifact / 'exports').glob('*/ready.json')):
        package_ready = json.loads(ready_file.read_text())
        original = package_ready['private_receipt']
        if package_ready['corpus_id'] == 'append-duplicates-only':
            original = declared_path
            staged = staged_path.resolve()
            relative = staged_rel
        else:
            staged = Path(original).resolve()
            relative = staged.relative_to(artifact).as_posix()
        paths[original] = str(staged)
        transfer_files.append({'original': original, 'path': relative,
            'sha256': sha256_file_hex(staged), 'bytes': staged.stat().st_size})
    _write_json(source_map_path, {'schema': 'natlang.staged-source-map/1', 'paths': paths})
    _write_json(transfer_path, {'model_role': 'mellum21', 'files': transfer_files})
    descriptor['private_receipt_custody'] = {
        'source_map': _binding(source_map_path),
        'transfer_manifest': _binding(transfer_path),
    }
    result = prepare(descriptor)
    out = Path(descriptor['output_root'])
    staged_map = out / 'receipt-custody' / '.natlang-source-map.json'
    staged_transfer = out / 'receipt-custody' / 'transfer.json'
    assert staged_map.read_bytes() == source_map_path.read_bytes()
    assert staged_transfer.read_bytes() == transfer_path.read_bytes()
    census = json.loads((out / 'ready-source-census.json').read_text())
    row = next(item for item in census['records'] if item['corpus_id'] == 'append-duplicates-only')
    custody = row['private_receipt_custody']
    assert custody['declared_path'] == declared_path
    assert custody['actual_path'] == str(staged_path.resolve())
    assert custody['actual_binding']['sha256'] == hashlib.sha256(content).hexdigest()
    assert custody['relocation_map']['source_map']['source']['sha256'] == sha256_file_hex(source_map_path)
    assert result['private_receipt_custody']['source_map']['staged']['sha256'] == sha256_file_hex(staged_map)
    assert result['private_receipt_custody']['transfer_manifest']['staged']['sha256'] == sha256_file_hex(staged_transfer)
