"""Topology classes and explicit historic-credit migration share one graph law."""
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.common import row_index
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.data.native_graph_schema import (
    parse_graph, indexed_graph_metadata, communication_classification)
from natlang_neuralese.data.native_sources import NativeGraphSource, _row_bindings
from natlang_neuralese.train import native_source_classification as migration
from natlang_neuralese.train.native_continuation import raw_communication_rows, communication_foundation
from test_native_graph import row


def variant(kind):
    value = row()
    if kind == 'ordinary':
        value['nodes'] = value['nodes'][:1]
    elif kind == 'marker-only':
        value['nodes'][1]['imports'] = []
    elif kind == 'empty':
        value['nodes'][1]['imports'][0]['stop'] = 1
    elif kind == 'no-supervision':
        value['nodes'][1]['target_weights'] = [0, 0, 0]
    elif kind == 'external':
        value['nodes'] = value['nodes'][1:]
        value['nodes'][0]['imports'][0].update(source='outside', start=0, stop=2)
        value['external_states'] = {'outside': {'id': 'nzs1_' + 'a' * 52, 'token_count': 2}}
    return value


@pytest.mark.parametrize('kind,expected', [('internal', (True, True, False)),
    ('external', (True, False, True)), ('ordinary', (False, False, False)),
    ('marker-only', (False, False, False)), ('empty', (False, False, False)),
    ('no-supervision', (False, False, False))])
def test_normalized_metadata_matches_compiled_identity_and_classes(kind, expected, monkeypatch):
    value = variant(kind)
    example = parse_graph(value, tokenizer_fingerprint='f' * 64)
    compiled_binding = _row_bindings([example])
    monkeypatch.setattr('natlang_neuralese.data.native_graph_schema.compile_graph',
                        lambda *a, **k: pytest.fail('metadata path expanded a graph'))
    normalized = indexed_graph_metadata(value, tokenizer_fingerprint='f' * 64)
    assert _row_bindings([normalized]) == compiled_binding
    assert tuple(communication_classification(normalized).values()) == expected
    assert communication_classification(normalized) == communication_classification(example)


def fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    source = tmp_path / 'source.jsonl'
    source.write_text(json.dumps(row()) + '\n')
    current = NativeGraphSource(source, 'f' * 64, required_split='train')
    package = Path(migration.__file__).resolve().parents[1]
    old_package = tmp_path / 'old-package'
    for name in migration.POLICY_FILES:
        target = old_package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    with (old_package / 'data/native_sources.py').open('a') as handle:
        handle.write('\n# prior classification policy\n')
    new_path = next((tmp_path / 'cache').glob('*.pkl'))
    new = row_index.load(new_path)
    old_key = {**new['key'], 'policy_sha256': migration.source_policy(old_package)}
    old_entry = {k: v for k, v in new['entries'][0].items() if k not in migration.CLASS_FIELDS}
    old_entry['communication'] = False
    old_path = tmp_path / 'prior.pkl'
    row_index.save(old_path, {'key': old_key, 'entries': [old_entry], 'metadata_sha256': digest([old_entry])})
    return source, old_path, old_package, package, current


def test_migration_reuses_bindings_writes_new_cache_and_never_compiles(tmp_path, monkeypatch):
    source, old, old_package, package, current = fixture(tmp_path, monkeypatch)
    prior_bytes = old.read_bytes()
    monkeypatch.setattr('natlang_neuralese.data.native_graph_schema.compile_graph',
                        lambda *a, **k: pytest.fail('migration compiled graph'))
    part = migration.migrate_index(source, old, old_package=old_package, new_package=package)
    assert old.read_bytes() == prior_bytes
    assert part['graph_position_expansions'] == part['optimizer_updates'] == 0
    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps({'schema': migration.SCHEMA,
        'train_sha256': current.receipt()['sha256'], 'indexes': [part]}))
    classes = migration.migration_classes(proof, current)
    assert classes[current.entries[0]['id']]['producer_consumer']
    from natlang_neuralese.train.native_operation_report import SourceIdentityView
    monkeypatch.setattr(current,'receipt',lambda:pytest.fail('classification built full group histograms'))
    assert migration.migration_classes(proof,SourceIdentityView(current))==classes
    # A second equivalent invocation is idempotent and leaves identical bytes.
    before = Path(part['new_index']['path']).read_bytes()
    again = migration.migrate_index(source, old, old_package=old_package, new_package=package)
    assert again == part and Path(part['new_index']['path']).read_bytes() == before


@pytest.mark.parametrize('change', ['mask', 'id', 'old-key', 'old-binding'])
def test_migration_rejects_changed_identity_before_writing(tmp_path, monkeypatch, change):
    source, old, old_package, package, current = fixture(tmp_path, monkeypatch)
    if change in ('mask', 'id'):
        value = row()
        if change == 'mask': value['nodes'][1]['target_weights'][2] = .5
        else: value['id'] = 'changed'
        source.write_text(json.dumps(value) + '\n')
    else:
        value = row_index.load(old)
        if change == 'old-key': value['key']['tokenizer'] = 'different'
        else: value['entries'][0]['binding'] = 'different'
        value['metadata_sha256'] = digest(value['entries'])
        row_index.save(old, value)
    with pytest.raises(ValueError):
        migration.migrate_index(source, old, old_package=old_package, new_package=package)


def test_history_uses_actual_counts_and_preserves_optimizer_rng_quotas_and_old_frame():
    optimizer = {'moment': torch.tensor([1.25, -2.5])}
    rng = torch.random.get_rng_state()
    frame = {'options': {'_communication_ids': ['legacy']},
        'source_row_bindings': {'internal': 'i', 'external': 'e', 'unvisited': 'u', 'legacy': 'l'},
        'exposure_counts': {'internal': 3, 'external': 7, 'unvisited': 0, 'legacy': 1},
        'required_visits': {'internal': 2, 'external': 1, 'unvisited': 1, 'legacy': 1},
        'optimizer': optimizer, 'rng': rng}
    classes = {'internal': {'binding': 'i', 'communication': True, 'producer_consumer': True, 'external_state_consumption': False},
        'external': {'binding': 'e', 'communication': True, 'producer_consumer': False, 'external_state_consumption': True},
        'unvisited': {'binding': 'u', 'communication': True, 'producer_consumer': True, 'external_state_consumption': False}}
    with pytest.raises(ValueError, match='explicit'):
        migration.require_explicit_reclassification({'s': frame}, [dict(id=k, **v) for k, v in classes.items()], {})
    changed = migration.reclassify_progress({'s': frame}, classes)['s']
    for name in ('optimizer', 'rng', 'exposure_counts', 'required_visits', 'source_row_bindings'):
        assert changed[name] is frame[name]
    assert frame['options'] == {'_communication_ids': ['legacy']}
    assert torch.equal(torch.random.get_rng_state(), rng)
    rows = raw_communication_rows({'s': changed})
    assert set(rows) == {'internal', 'unvisited', 'legacy'}
    assert rows['internal']['graph_visits'] == 3 and rows['internal']['qualified']
    assert rows['unvisited']['graph_visits'] == 0 and not rows['unvisited']['qualified']
    bad = copy.deepcopy(classes); bad['internal']['binding'] = 'changed'
    with pytest.raises(ValueError, match='historical row binding'):
        migration.reclassify_progress({'s': frame}, bad)


def test_external_only_does_not_establish_participating_model_foundation(tmp_path):
    source = SimpleNamespace(entries=[{'id': 'external', 'binding': 'e', 'communication': True,
        'producer_consumer': False, 'external_state_consumption': True,
        'groups':['external'],'supervised_tokens':1}],
        path=tmp_path/'external',source_identity_path=tmp_path/'external',
        file_sha256='external',kind='native-graphs',
        ordered_ids=lambda: ['external'])
    result = communication_foundation(source, {'external': 99}, epochs=1, trained_bindings={'external': 'e'})
    assert result['graphs'] == 0 and not result['complete']


@pytest.mark.parametrize('filtered', [False, True])
def test_registered_ordinary_topology_law_avoids_any_graph_payload_pass(tmp_path, monkeypatch, filtered):
    from test_native_capacity_activation import fixture as ordinary_fixture
    source, part, package = ordinary_fixture(tmp_path, monkeypatch, filtered=filtered)
    leaf = source.children[0] if getattr(source, 'children', None) else source
    cache = next(path for path in (tmp_path / 'cache').glob('*.pkl')
                 if row_index.load(path)['key']['source_sha256'] == leaf.file_sha256)
    indexed = row_index.load(cache)
    monkeypatch.setattr(migration, '_jsonl', lambda *a, **k: pytest.fail('ordinary proof parsed graph payload'))
    migration._ordinary_partition_classes(part, migration._binding(leaf.path), indexed)
    assert all(not entry['communication'] for entry in indexed['entries'])
    forged = copy.deepcopy(indexed); forged['entries'][0]['forward_tokens'] += 1
    with pytest.raises(ValueError, match='producer/source proof|retained execution'):
        migration._ordinary_partition_classes(part, migration._binding(leaf.path), forged)


def test_actual_resume_with_classification_proof_matches_same_parent_training(tmp_path, monkeypatch):
    from test_native_model import tiny_source
    from test_native_optimizer_checkpointing import assert_exact
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    from test_native_state_sft import graph
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    base = tmp_path / 'base'; tiny_source(base)
    _, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    fingerprint = config['tokenizer_fingerprint']
    source = tmp_path / 'train.jsonl'; held = tmp_path / 'held.jsonl'
    source.write_text(json.dumps({**row(), 'tokenizer_fingerprint': fingerprint}) + '\n')
    held.write_text(json.dumps({**graph('held'), 'split': 'test', 'tokenizer_fingerprint': fingerprint}) + '\n')
    out = tmp_path / 'first'
    args = ['--base', str(base), '--dtype', 'float32', '--rank', '4', '--device', 'cpu',
        '--native-graphs', str(source), '--held-native-graphs', str(held), '--epochs', '2',
        '--stage-id', 'full', '--eval-every', '1', '--out', str(out),
        '--refresh-at-checkpoint', '--checkpoint-every', '1', '--forward-atol', '1e-4', '--gradient-atol', '1e-5']
    assert trainer.main(args) == 0
    parent = torch.load(out / 'checkpoint.pt', weights_only=False)
    assert sum(parent['training']['exposure_counts'].values()) == 1
    legacy = copy.deepcopy(parent)
    legacy['training']['options']['_communication_ids'] = []
    legacy['training']['options']['_producer_consumer_ids'] = []
    for frame in legacy['training'].get('sft_progress', {}).values():
        frame['options']['_communication_ids'] = []
        frame['options']['_producer_consumer_ids'] = []
    legacy_path = tmp_path / 'legacy.pt'; torch.save(legacy, legacy_path)
    current = NativeGraphSource(source, fingerprint, required_split='train', role_ids=config['marker_ids'])
    package = Path(migration.__file__).resolve().parents[1]
    old_package = tmp_path / 'old-package'
    for name in migration.POLICY_FILES:
        target = old_package / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    with (old_package / 'data/native_sources.py').open('a') as handle: handle.write('\n# historical classification\n')
    cache = next(path for path in (tmp_path / 'cache').glob('*.pkl')
        if row_index.load(path)['key']['source_sha256'] == migration._binding(source)['sha256']
        and row_index.load(path)['key']['role_ids'] == config['marker_ids'])
    prior = row_index.load(cache)
    for entry in prior['entries']:
        for name in migration.CLASS_FIELDS: entry.pop(name, None)
        entry['communication'] = False
    prior['key']['policy_sha256'] = migration.source_policy(old_package)
    prior['metadata_sha256'] = digest(prior['entries'])
    prior_path = tmp_path / 'prior.pkl'; row_index.save(prior_path, prior)
    part = migration.migrate_index(source, prior_path, old_package=old_package, new_package=package)
    proof = tmp_path / 'migration.json'; proof.write_text(json.dumps({'schema': migration.SCHEMA,
        'train_sha256': current.receipt()['sha256'], 'indexes': [part]}))
    baseline = tmp_path / 'baseline'; changed = tmp_path / 'changed'
    continuation = [*args, '--continue-from', str(out / 'checkpoint.pt'), '--objective-continuation']
    continuation[continuation.index('--out') + 1] = str(baseline)
    assert trainer.main(continuation) == 0
    continuation[continuation.index('--out') + 1] = str(changed)
    continuation[continuation.index('--continue-from') + 1] = str(legacy_path)
    assert trainer.main([*continuation, '--source-classification-migration', str(proof)]) == 0
    expected = torch.load(baseline / 'checkpoint.pt', weights_only=False)
    actual = torch.load(changed / 'checkpoint.pt', weights_only=False)
    for field in ('backbone_state', 'interface_state'):
        assert_exact(actual[field], expected[field], field)
    assert_exact(actual['training']['optimizer'], expected['training']['optimizer'], 'optimizer')
    assert_exact(actual['training']['rng'], expected['training']['rng'], 'rng')
    assert actual['training']['global_step'] == expected['training']['global_step'] == 2
    assert actual['training']['exposure_counts'] == expected['training']['exposure_counts']
    assert actual['training']['options']['source_classification_history'][0]['graph_visits_created'] == 0


def test_inventory_builder_preserves_declared_kind_model_and_rejects_duplicate_leaves(tmp_path):
    from scripts.neuralese_data.migrate_native_source_classification import manifest_leaves
    source = tmp_path / 'pair.jsonl'; source.write_text('{}\n')
    manifest = tmp_path / 'sources.json'
    reference = migration._binding(source)
    manifest.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1',
        'sources': [{**reference, 'kind': 'semantic-pairs', 'model': 'mellum'}]}))
    assert manifest_leaves(migration._binding(manifest)) == [
        (str(source.resolve()), reference['sha256'], 'semantic-pairs', 'mellum')]
    # The leaf helper preserves duplicates; the builder rejects them rather than
    # silently turning a malformed training union into a supposedly valid one.
    value = json.loads(manifest.read_text()); value['sources'] *= 2
    manifest.write_text(json.dumps(value))
    actual = manifest_leaves(migration._binding(manifest))
    assert len(actual) == 2 and len(set(actual)) == 1
    value['sources'][0].pop('model'); manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='selected semantic-pair model'):
        manifest_leaves(migration._binding(manifest))


def test_empty_semantic_pair_migration_keeps_zero_communication_credit(tmp_path, monkeypatch):
    from test_native_state_sft import authenticated_pair
    from natlang_neuralese.common.hashing import sha256_hex
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    value = authenticated_pair('empty'); value['payload_sha256'] = sha256_hex(b'')
    for view in value['models'].values():
        view['graph']['nodes'][0].update(token_ids=[1,58,59], target_weights=[0,1,1], producing=[[2,2]])
        view['graph']['nodes'][1]['imports'][0]['stop'] = 2
        view['export'].update(stop=2, text='', offsets=[])
        view['witness']['payload_sha256'] = value['payload_sha256']
        view['graph']['source_pairing']['consumed_occurrences'][0].update(wrapped_token_stop=2, payload_sha256=value['payload_sha256'])
    source = tmp_path / 'pairs.jsonl'; source.write_text(json.dumps(value) + '\n')
    current = NativeGraphSource(source, '1' * 64, required_split='train', pair_model='lfm', role_ids=(58,59,60,61))
    package = Path(migration.__file__).resolve().parents[1]; old_package = tmp_path / 'old'
    for name in migration.POLICY_FILES:
        target = old_package / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    with (old_package / 'data/native_sources.py').open('a') as handle: handle.write('\n# prior parser\n')
    prior = row_index.load(next((tmp_path / 'cache').glob('*.pkl')))
    prior['key']['policy_sha256'] = migration.source_policy(old_package)
    prior['entries'][0].pop('producer_consumer'); prior['entries'][0].pop('external_state_consumption')
    prior['metadata_sha256'] = digest(prior['entries'])
    old = tmp_path / 'old.pkl'; row_index.save(old, prior)
    monkeypatch.setattr(migration, '_jsonl', lambda *a, **k: pytest.fail('semantic pair was reparsed'))
    result = migration.migrate_index(source, old, old_package=old_package, new_package=package)
    entry = migration._index(result['new_index'])['entries'][0]
    assert not any(entry[name] for name in migration.CLASS_FIELDS)


@pytest.mark.parametrize('damage',[None,'changed-code','missing-origin','wrong-bytes'])
def test_recipe_historical_classifier_retains_origin_with_identical_code(tmp_path,monkeypatch,damage):
    from natlang_neuralese.train.recipe import _classification_proof
    source,old,old_package,package,current=fixture(tmp_path,monkeypatch)
    part=migration.migrate_index(source,old,old_package=old_package,new_package=package)
    origin=tmp_path/'historical-runtime/train/native_source_classification.py';origin.parent.mkdir(parents=True)
    origin.write_bytes(Path(migration.__file__).read_bytes())
    part['producer']=migration._binding(origin)
    if damage=='changed-code':
        origin.write_bytes(origin.read_bytes()+b'\n# other producer\n');part['producer']=migration._binding(origin)
    elif damage=='missing-origin':origin.unlink()
    elif damage=='wrong-bytes':part['producer']['bytes']+=1
    proof={'schema':migration.SCHEMA,'train_sha256':current.receipt()['sha256'],
           'held':current.receipt(),'indexes':[part],'held_indexes':[part],
           'optimizer_updates':0,'student_pointer_writes':0,'model_allocations':0,'graph_position_expansions':0}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof));before=path.read_bytes()
    inputs={'native_graphs':current.receipt(),'held_native_graphs':current.receipt(),
            'source_classification_migration':{**migration._binding(path),'binding':'historical'}}
    if damage:
        with pytest.raises((ValueError,FileNotFoundError)):_classification_proof(inputs)
    else:
        assert _classification_proof(inputs)==proof
        assert Path(proof['indexes'][0]['producer']['path'])==origin
    assert path.read_bytes()==before
