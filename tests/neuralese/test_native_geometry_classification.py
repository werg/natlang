"""Historical geometry stays strict across an authenticated class-only migration."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from natlang_neuralese.common import row_index
from natlang_neuralese.train import native_source_classification as migration
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.native_geometry_classification import continue_geometry
from test_native_source_classification import fixture


def setup(tmp_path, monkeypatch):
    source, old_path, old_package, package, current = fixture(tmp_path, monkeypatch)
    part = migration.migrate_index(source, old_path, old_package=old_package, new_package=package)
    proof = tmp_path / 'proof.json'
    proof.write_text(json.dumps({'schema': migration.SCHEMA,
        'train_sha256': current.receipt()['sha256'], 'indexes': [part]}))
    old = row_index.load(old_path)
    example = next(iter(current))
    maximum = {'index': 0, 'id': example.id, 'binding': old['entries'][0]['binding'],
               'forward_tokens': example.plan.token_count, 'kv_slots': len(example.plan.slot_sources)}
    report = {'schema': 'natlang.native-whole-delta-geometry/1', 'passed': True,
        'source': resident.binding(source), 'prior_train': None,
        'old_train_rows': 0, 'new_train_rows': 1,
        'tokenizer_fingerprint': current.tokenizer_fingerprint,
        'role_ids': list(current.role_ids), 'entries_sha256': resident.entries_digest(old['entries']),
        'indexes': [{'cache': resident.binding(old_path), 'key': old['key'],
                     'metadata_sha256': old['metadata_sha256']}],
        'maxima': {name: maximum for name in ('forward_tokens', 'kv_slots')},
        'provenance': [migration._binding(proof)]}
    path = tmp_path / 'geometry.json'; path.write_text(json.dumps(report))
    return current, package, path, proof, part


def test_real_geometry_verifier_accepts_only_authenticated_class_projection(tmp_path, monkeypatch):
    source, package, path, proof, part = setup(tmp_path, monkeypatch)
    before = path.read_bytes(); old_index = open(part['old_index']['path'], 'rb').read()
    with pytest.raises(ValueError, match='metadata differs'):
        resident.geometry_from_evidence(source, 0, resident.binding(path), package=package)
    ref = continue_geometry(migration._binding(path), migration._binding(proof), source,
                            tmp_path / 'continued.json')
    result = resident.geometry_from_evidence(source, 0, ref, package=package)
    original = json.loads(before); continued = json.loads(resident.bound(ref).read_text())
    assert result == {name: original['maxima'][name][name] for name in original['maxima']}
    assert continued['maxima'] == original['maxima']
    assert continued['source'] == original['source']
    assert path.read_bytes() == before
    assert open(part['old_index']['path'], 'rb').read() == old_index
    assert continued['classification_continuation']['training_credit'] == 0


@pytest.mark.parametrize('change', ['old-digest', 'omitted-index', 'old-witness', 'new-binding',
    'old-binding', 'producer', 'maximum'])
def test_changed_original_facts_or_migration_cannot_gain_geometry(tmp_path, monkeypatch, change):
    source, package, path, proof, part = setup(tmp_path, monkeypatch)
    report = json.loads(path.read_text()); p = json.loads(proof.read_text())
    if change == 'old-digest': report['entries_sha256'] = '0' * 64
    elif change == 'omitted-index': report['indexes'] = []
    elif change == 'old-witness': report['indexes'][0]['key']['tokenizer'] = 'changed'
    elif change in ('new-binding', 'old-binding'):
        role = 'new_index' if change == 'new-binding' else 'old_index'
        index = row_index.load(p['indexes'][0][role]['path'])
        index['entries'][0]['binding'] = 'changed'
        index['metadata_sha256'] = resident.entries_digest(index['entries'])
        row_index.save(Path(p['indexes'][0][role]['path']), index)
        p['indexes'][0][role] = migration._binding(p['indexes'][0][role]['path'])
    elif change == 'producer': p['indexes'][0]['producer']['sha256'] = '0' * 64
    elif change == 'maximum': report['maxima']['forward_tokens']['forward_tokens'] += 1
    path.write_text(json.dumps(report)); proof.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        ref = continue_geometry(migration._binding(path), migration._binding(proof), source,
                                tmp_path / 'continued.json')
        resident.geometry_from_evidence(source, 0, ref, package=package)


def test_original_geometry_cache_can_be_identical_bytes_at_another_path(tmp_path, monkeypatch):
    source, package, path, proof, part = setup(tmp_path, monkeypatch)
    report = json.loads(path.read_text())
    alias = tmp_path / 'original-helper-cache.pkl'
    shutil.copyfile(part['old_index']['path'], alias)
    report['indexes'][0]['cache'] = resident.binding(alias)
    path.write_text(json.dumps(report))
    ref = continue_geometry(migration._binding(path), migration._binding(proof), source,
                            tmp_path / 'continued.json')
    assert resident.geometry_from_evidence(source, 0, ref, package=package)
    alias.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        continue_geometry(migration._binding(path), migration._binding(proof), source,
                          tmp_path / 'rejected.json')
