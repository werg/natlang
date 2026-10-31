from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_render_continuation_retries_only_unattempted_overcap_ids_and_preserves_other_omissions():
    import json
    from scripts.neuralese_data.build_chain_graphs import split_render_retry_rows

    rows = [
        {'trajectory': 'large-a', 'reason': 'graph-over-token-cap'},
        {'trajectory': 'large-b', 'reason': 'graph-over-token-cap'},
        {'trajectory': 'recoverable-unknown',
         'reason': 'render-error: producer occurrence is not the exact authenticated payload value'},
        {'trajectory': 'bad-source', 'reason': 'render-error: source writer is unauthenticated'},
        {'trajectory': 'protected', 'reason': 'current_protected_worlds_custody'},
    ]
    committed = b''.join(json.dumps(row, separators=(',', ':')).encode() + b'\n' for row in rows)
    retained, retry_ids, reasons = split_render_retry_rows(committed, {'large-b'}, {'recoverable-unknown'})
    assert retry_ids == ['large-a', 'recoverable-unknown']
    assert reasons == {'graph-over-token-cap': 1,
                       'render-error: producer occurrence is not the exact authenticated payload value': 1}
    assert retained == [rows[1], rows[3], rows[4]]


def test_render_continuation_policy_does_not_change_immutable_dedupe_plan_arguments():
    from argparse import Namespace
    from scripts.neuralese_data.build_chain_graphs import render_plan_arguments

    original = Namespace(max_tokens=64000, reader_cap=64, target_cap=64,
                         continuation_max_tokens=None, continuation_old_code_fingerprint=None,
                         workers=4, batch_chains=512)
    continued = Namespace(max_tokens=64000, reader_cap=64, target_cap=64,
                          continuation_max_tokens=335000, continuation_old_code_fingerprint='a' * 64,
                          workers=2, batch_chains=128)
    assert render_plan_arguments(original) == render_plan_arguments(continued)


def module_under_test():
    path = Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/build_chain_graphs.py'
    spec = importlib.util.spec_from_file_location('build_chain_graphs_current_protection_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_protection_snapshot_is_loaded_for_default_graph_build(monkeypatch):
    module = module_under_test()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
    from natlang_neuralese.common.custody import REGISTRY, ProtectedWorlds
    import scripts.neuralese_data.e2e_mine as e2e

    class FakeProtected:
        digest = 'a' * 64
        sources = ('{"id":"held","kind":"source-groups"}',)

    monkeypatch.setattr(ProtectedWorlds, 'load', classmethod(lambda cls: FakeProtected()))
    registry_sha = module.sha256_file_hex(module.ROOT / REGISTRY)
    monkeypatch.setattr(e2e, 'current_protected_source_groups', lambda: (
        {'quality-source-groups': {'quality:one'}, 'v4-reserved': {'2wiki-q:held'}},
        [{'id': 'quality-source-groups', 'sha256': 'b' * 64}, {'id': 'v4-reserved', 'sha256': 'c' * 64}],
        registry_sha))
    binding, worlds, groups = module.current_protection_snapshot()
    assert worlds.digest == 'a' * 64
    assert binding['protected_sha256'] == 'a' * 64
    assert binding['registry_sha256'] == registry_sha
    assert groups == {'quality-source-groups': {'quality:one'}, 'v4-reserved': {'2wiki-q:held'}}
    assert len(binding['source_group_bindings']) == 2


def test_current_graph_filter_keeps_quality_only_distinct_from_mixed_reserved_overlap():
    module = module_under_test()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
    from natlang_neuralese.common.custody import ProtectedWorlds
    quality = 'quality:only'
    reserved = '2wiki-q:00310dc8094611ebbdaeac1f6bf848b6'
    protected = ProtectedWorlds.from_cases({'current': ('a' * 64, [
        {'id': 'quality-held', 'split': 'test', 'family': 'held', 'source_groups': [quality],
         'semantics': {'root': 'q.nl', 'files': {'q.nl': 'q'}, 'inputs': {}}},
        {'id': 'reserved-held', 'split': 'test', 'family': 'held', 'source_groups': [reserved],
         'semantics': {'root': 'r.nl', 'files': {'r.nl': 'r'}, 'inputs': {}}},
    ])})
    base = {'id': 'program', 'family': 'fixture', 'semantics': {
        'root': 'main.nl', 'files': {'main.nl': 'body'}, 'inputs': {'request': 'x'}}}
    quality_only = {'q:0': {'id': 'q:0', 'split': 'train',
        'source_ref': {'trajectory_id': 'quality-only'}, 'source_groups': [quality],
        'task': {'program_ir': base}}}
    result = module.current_protection_omission(quality_only, protected, {quality}, {reserved})
    assert result['reason'] == 'source_quality_quarantine_chain'
    assert result['applicable_exclusions'] == ['source_quality_quarantine_chain']

    mixed = {'both:0': {'id': 'both:0', 'split': 'train',
        'source_ref': {'trajectory_id': 'quality-and-reserved'}, 'source_groups': [quality, reserved],
        'task': {'program_ir': base}}}
    result = module.current_protection_omission(mixed, protected, {quality}, {reserved})
    assert result['reason'] == 'current_protected_worlds_custody'
    assert result['applicable_exclusions'] == [
        'current_protected_worlds_custody', 'source_quality_quarantine_chain']
    assert result['protected_identity_kinds'] == ['binding', 'group']

    clean = {'clean:0': {'id': 'clean:0', 'split': 'train',
        'source_ref': {'trajectory_id': 'clean'}, 'source_groups': ['train:clean'],
        'task': {'program_ir': {**base, 'id': 'clean-program'}}}}
    assert module.current_protection_omission(clean, protected, {quality}, {reserved}) is None


def test_default_chain_conversion_omits_a_real_reserved_source_group(monkeypatch):
    module = module_under_test()
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
    from natlang_neuralese.common.custody import ProtectedWorlds
    reserved = '2wiki-q:00310dc8094611ebbdaeac1f6bf848b6'
    program = {'id': 'train-program', 'family': 'fixture', 'semantics': {
        'root': 'main.nl', 'files': {'main.nl': 'body'}, 'inputs': {'request': 'x'}}}
    protected = ProtectedWorlds.from_cases({'current-v4': ('b' * 64, [
        {'id': 'held', 'split': 'test', 'family': 'held', 'source_groups': [reserved],
         'semantics': {'root': 'held.nl', 'files': {'held.nl': 'other'}, 'inputs': {}}}])})
    rows = {'record:0': {'id': 'record:0', 'split': 'train', 'source_ref': {'trajectory_id': 'trajectory'},
                         'source_groups': [reserved], 'task': {'program_ir': program}}}
    monkeypatch.setattr(module, 'read_records', lambda _locations: rows)
    module.STATE.update({'current_worlds': protected, 'current_quality_groups': set(),
                         'current_nonquality_groups': {reserved}, 'holds': set(),
                         'acquired_protection': {'held': set(), 'quality': set(), 'pending': set()}})
    result = module.convert_unit(({'trajectory': 'trajectory', 'split': 'train', 'records': ['record:0']},
                                  {'record:0': ('fixture', 0, 1)}, [], 'pieces.jsonl',
                                  {'supervised': set(), 'edges': set()}))
    assert result[0:2] == ('omission', 'train')
    assert result[2]['reason'] == 'current_protected_worlds_custody'
    assert result[2]['protected_identity_kinds'] == ['binding', 'group']
