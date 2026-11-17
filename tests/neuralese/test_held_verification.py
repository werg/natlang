"""Held verification uses an attested exact pool relocation, never edits manifests."""
import json

import pytest

from natlang_neuralese.rl import held_eval as h


def artifacts(tmp_path):
    pool = tmp_path / 'relocated-pool.jsonl'
    pool.write_text(''.join(json.dumps(row) + '\n' for row in [
        {'id': 'held', 'source_groups': ['same-world']},
        {'id': 'sibling', 'source_groups': ['same-world']}]))
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'held', 'source_groups': ['same-world']}) + '\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'pool': {'path': '/original-machine/pool.jsonl', 'sha256': h._sha256(pool)},
                                    'cases': {'path': cases.name, 'sha256': h._sha256(cases)},
                                    'tasks': [{'id': 'held'}]}))
    return pool, cases, manifest


def test_exact_pool_relocation_preserves_immutable_manifest_and_group_detection(tmp_path):
    pool, _, manifest = artifacts(tmp_path)
    before = manifest.read_bytes()
    result = h.verify(manifest, {'sibling'}, pool_path=pool)
    assert result['overlap_ids'] == []
    assert result['overlap_groups'] == ['held']
    assert result['source_pool']['relocated'] is True
    assert result['source_pool']['sha256'] == h._sha256(pool)
    assert result['manifest_sha256'] == h._sha256(manifest)
    assert manifest.read_bytes() == before


def test_changed_relocated_pool_is_rejected(tmp_path):
    pool, _, manifest = artifacts(tmp_path)
    pool.write_text(pool.read_text() + '\n')
    with pytest.raises(ValueError, match='source pool differs'):
        h.verify(manifest, set(), pool_path=pool)


def test_changed_held_cases_are_rejected(tmp_path):
    pool, cases, manifest = artifacts(tmp_path)
    cases.write_text(cases.read_text() + '\n')
    with pytest.raises(ValueError, match='held cases changed'):
        h.verify(manifest, set(), pool_path=pool)
