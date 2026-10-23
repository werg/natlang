import json
import shutil
from pathlib import Path

import pytest

from natlang_neuralese.common import row_index
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.train import native_source_classification as classifier
from scripts.neuralese_data.native_source_routing_continuation import (
    continue_classified_index, prove_source_routing_extension,
)


@pytest.fixture
def packages(tmp_path, monkeypatch):
    current = Path(classifier.__file__).parents[1]
    prior = Path(__file__).parent / 'fixtures/native-source-routing-predecessor'
    old, new = tmp_path / 'old', tmp_path / 'new'
    for name in (*classifier.POLICY_FILES, 'data/native_source_tree.py', 'train/native_source_classification.py'):
        for target, source in ((new, current / name), (old, prior / name if (prior / name).exists() else current / name)):
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, path)
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'indexes'))
    return old, new


def test_actual_production_predecessor_has_only_reviewed_routing_extension(packages):
    old, new = packages
    proof = prove_source_routing_extension(old, new)
    assert proof['row_metadata_unchanged'] is True
    assert proof['old_policy_sha256'] == classifier.source_policy(old)
    assert proof['new_policy_sha256'] == classifier.source_policy(new)
    assert proof['old_policy_sha256'] != proof['new_policy_sha256']


@pytest.mark.parametrize('name,append', [
    ('data/native_sources.py', '\nUNREVIEWED_SOURCE_POLICY = True\n'),
    ('data/native_source_tree.py', '\nUNREVIEWED_MANIFEST_POLICY = True\n'),
    ('common/hashing.py', '\nUNREVIEWED_HASH_POLICY = True\n'),
    ('data/native_graph_schema.py', '\nUNREVIEWED_ROW_POLICY = True\n'),
])
def test_unrelated_reader_changes_are_rejected(packages, name, append):
    old, new = packages
    with (new / name).open('a') as output:
        output.write(append)
    with pytest.raises(ValueError):
        prove_source_routing_extension(old, new)


def test_routing_helper_behavior_cannot_change(packages):
    old, new = packages
    path = new / 'data/native_source_tree.py'
    path.write_text(path.read_text().replace('return target', 'return Path(reference)', 1))
    with pytest.raises(ValueError, match='reviewed helper'):
        prove_source_routing_extension(old, new)


def test_index_reuse_preserves_every_row_and_existing_destination(packages, tmp_path):
    old, new = packages
    source = tmp_path / 'source.jsonl'
    source.write_text('{"bound":"source bytes"}\n')
    source_ref = classifier._binding(source)
    entries = [{'id': 'a', 'binding': 'a' * 64, 'communication': False,
                'producer_consumer': False, 'external_state_consumption': False},
               {'id': 'b', 'binding': 'b' * 64, 'communication': True,
                'producer_consumer': True, 'external_state_consumption': False}]
    key = {'source_sha256': source_ref['sha256'], 'tokenizer': 'c' * 64,
           'split': 'train', 'policy_sha256': classifier.source_policy(old),
           'kind': 'native-graphs', 'pair_model': None, 'role_ids': [1, 2, 3, 4]}
    previous = row_index.entry_path('native-graphs', digest(key))
    row_index.save(previous, {'schema': 'natlang.row-index/1', 'key': key,
                             'entries': entries, 'metadata_sha256': digest(entries)})
    prior = tmp_path / 'prior-classification.json'
    prior.write_text(json.dumps({'schema': classifier.SCHEMA, 'indexes': [{
        'source': source_ref, 'new_index': classifier._binding(previous),
        'rows': len(entries), 'producer': classifier._binding(old / 'train/native_source_classification.py')}]}))
    prior_ref = classifier._binding(prior)
    result = continue_classified_index(source_ref, prior_ref, old_package=old, new_package=new)
    target = Path(result['new_index']['path'])
    loaded = row_index.load(target)
    assert loaded['entries'] == entries
    assert loaded['metadata_sha256'] == digest(entries)
    assert loaded['key'] == {**key, 'policy_sha256': classifier.source_policy(new)}
    before = target.stat().st_mtime_ns
    assert continue_classified_index(source_ref, prior_ref, old_package=old, new_package=new) == result
    assert target.stat().st_mtime_ns == before
    source.write_text('{"different":"bytes"}\n')
    with pytest.raises(ValueError, match='artifact changed'):
        continue_classified_index(source_ref, prior_ref, old_package=old, new_package=new)


def test_forged_index_or_rewritten_cache_is_not_prior_proof_authority(packages, tmp_path):
    old, new = packages
    source = tmp_path / 'source.jsonl'
    source.write_text('{"bound":"source"}\n')
    source_ref = classifier._binding(source)
    key = {'source_sha256': source_ref['sha256'], 'tokenizer': 'c' * 64,
           'split': 'train', 'policy_sha256': classifier.source_policy(old),
           'kind': 'native-graphs', 'pair_model': None, 'role_ids': [1, 2, 3, 4]}
    entry = {'id': 'genuine', 'binding': 'a' * 64, 'communication': False,
             'producer_consumer': False, 'external_state_consumption': False}
    index = row_index.entry_path('native-graphs', digest(key))
    value = {'schema': 'natlang.row-index/1', 'key': key,
             'entries': [entry], 'metadata_sha256': digest([entry])}
    row_index.save(index, value)
    prior = tmp_path / 'prior.json'
    proof = {'schema': classifier.SCHEMA, 'indexes': [{
        'source': source_ref, 'new_index': classifier._binding(index), 'rows': 1,
        'producer': classifier._binding(old / 'train/native_source_classification.py')}]}
    prior.write_text(json.dumps(proof))
    reference = classifier._binding(prior)
    forged = {**entry, 'id': 'fabricated'}
    forged_value = {**value, 'entries': [forged], 'metadata_sha256': digest([forged])}
    arbitrary = tmp_path / 'arbitrary.pkl'
    row_index.save(arbitrary, forged_value)
    # Replacing a canonical cache still fails: the saved recipe proof pins bytes.
    row_index.save(index, forged_value)
    with pytest.raises(ValueError, match='artifact changed'):
        continue_classified_index(source_ref, reference, old_package=old, new_package=new)
    # Substituting a new self-reported proof also fails against the original pin.
    proof['indexes'][0]['new_index'] = classifier._binding(arbitrary)
    prior.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='artifact changed'):
        continue_classified_index(source_ref, reference, old_package=old, new_package=new)
