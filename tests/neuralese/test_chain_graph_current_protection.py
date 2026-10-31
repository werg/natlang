from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def module_under_test():
    path = Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/build_chain_graphs.py'
    spec = importlib.util.spec_from_file_location('build_chain_graphs_current_protection_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_protection_receipt_binds_complete_registry_and_source_receipt(tmp_path, monkeypatch):
    module = module_under_test()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
    from natlang_neuralese.common.custody import REGISTRY, ProtectedWorlds

    class FakeProtected:
        digest = 'a' * 64
        sources = ('{"id":"held","kind":"source-groups"}',)

    monkeypatch.setattr(ProtectedWorlds, 'load', classmethod(lambda cls: FakeProtected()))
    registry_sha = module.sha256_file_hex(module.ROOT / REGISTRY)
    source_receipt = tmp_path / 'receipt.json'
    source_receipt.write_text(json.dumps({
        'schema': 'natlang.neuralese-e2e-cohort/2',
        'current_protected_worlds': {'registry': REGISTRY, 'registry_sha256': registry_sha,
                                     'protected_sha256': 'a' * 64},
    }) + '\n')
    binding = module.bind_current_protection_receipt(source_receipt)
    assert binding['registry'] == REGISTRY
    assert binding['registry_sha256'] == registry_sha
    assert binding['protected_sha256'] == 'a' * 64
    assert binding['source_receipt'] == {
        'path': str(source_receipt.resolve()), 'sha256': module.sha256_file_hex(source_receipt)}
    assert binding['source_categories'] == [{'id': 'held', 'kind': 'source-groups'}]


def test_current_protection_receipt_rejects_stale_union_or_wrong_schema(tmp_path, monkeypatch):
    module = module_under_test()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
    from natlang_neuralese.common.custody import REGISTRY, ProtectedWorlds

    class FakeProtected:
        digest = 'b' * 64
        sources = ()

    monkeypatch.setattr(ProtectedWorlds, 'load', classmethod(lambda cls: FakeProtected()))
    registry_sha = module.sha256_file_hex(module.ROOT / REGISTRY)
    path = tmp_path / 'receipt.json'
    path.write_text(json.dumps({'schema': 'natlang.neuralese-e2e-cohort/2',
        'current_protected_worlds': {'registry': REGISTRY, 'registry_sha256': registry_sha,
                                     'protected_sha256': 'a' * 64}}))
    with pytest.raises(ValueError, match='exact current ProtectedWorlds union'):
        module.bind_current_protection_receipt(path)
    path.write_text(json.dumps({'schema': 'wrong',
        'current_protected_worlds': {'registry': REGISTRY, 'registry_sha256': registry_sha,
                                     'protected_sha256': 'b' * 64}}))
    with pytest.raises(ValueError, match='exact current ProtectedWorlds union'):
        module.bind_current_protection_receipt(path)
