"""Shared world custody (common.custody) and the acceptance gate (common.acceptance)."""
import json

import pytest

from natlang_neuralese.common import acceptance, custody
from natlang_neuralese.common.hashing import sha256_file_hex


def world(ident, split='train', **fields):
    return {'id': ident, 'split': split, 'family': 'fixture',
            'semantics': {'root': 'r.nl', 'files': {'r.nl': 'Count.'}, 'inputs': {'n': ident}}, **fields}


def registry(tmp_path):
    (tmp_path / 'runs/dev').mkdir(parents=True)
    (tmp_path / 'runs/dev/cases.jsonl').write_text(json.dumps(world('dev-0', 'test', source_groups=['dev-g'])) + '\n')
    (tmp_path / 'training/decision-holds').mkdir(parents=True)
    (tmp_path / 'training/decision-holds/h.jsonl').write_text(json.dumps({'id': 'held-row'}) + '\n')
    (tmp_path / 'groups.jsonl').write_text(json.dumps({'source_groups': ['held-g']}) + '\n')
    (tmp_path / 'reservations.json').write_text(json.dumps({'schema': 'natlang.source-split-reservations/1',
        'records': [{'id': 'r0', 'source_ids': ['reserved/q']}]}))
    (tmp_path / 'corpora.json').write_text(json.dumps({'corpora': [
        {'id': 'dev', 'path': 'runs/dev', 'admission': custody.PROTECTED_ADMISSION, 'include': ['cases.jsonl']},
        {'id': 'train', 'path': 'runs/train', 'admission': 'decision-filtered', 'include': ['**/*.jsonl']}]}))
    sources = [{'kind': 'evaluation-packets', 'id': 'eval', 'registry': 'corpora.json'},
               {'kind': 'source-groups', 'id': 'held', 'path': 'groups.jsonl',
                'sha256': sha256_file_hex(tmp_path / 'groups.jsonl')},
               {'kind': 'decision-holds', 'id': 'holds', 'path': 'training/decision-holds'},
               {'kind': 'split-reservations', 'id': 'res', 'path': 'reservations.json'}]
    path = tmp_path / 'protected.json'
    path.write_text(json.dumps({'schema': custody.REGISTRY_SCHEMA, 'sources': sources}))
    return path


def test_protected_worlds_union_every_declared_source_and_fail_closed(tmp_path):
    path = registry(tmp_path)
    protected = custody.ProtectedWorlds.load(tmp_path, path)
    receipt = protected.check(world('t0'))
    assert receipt['schema'] == custody.CUSTODY_SCHEMA and receipt['protected_sha256'] == protected.digest
    refused = [world('t1', source_groups=['dev-g']), world('t2', source_groups=['held-g']),
               world('held-row'), world('t3', source_ids=['reserved/q']), world('dev-0'),
               {**world('fresh'), 'semantics': world('dev-0')['semantics']},  # same content, fresh labels
               world('t4', curriculum={'evidence': {'world': ['x']}}, split='development')]
    for case in refused:
        with pytest.raises(custody.CustodyError):
            protected.check(case)
    (tmp_path / 'groups.jsonl').write_text(json.dumps({'source_groups': ['other']}) + '\n')
    with pytest.raises(custody.CustodyError, match='changed'):
        custody.ProtectedWorlds.load(tmp_path, path)
    with pytest.raises(custody.CustodyError, match='not readable'):
        custody.ProtectedWorlds.load(tmp_path, tmp_path / 'missing.json')
    (tmp_path / 'runs/dev/cases.jsonl').unlink()
    with pytest.raises(custody.CustodyError, match='fails closed'):
        custody.ProtectedWorlds.load(tmp_path, path)


def test_related_worlds_stay_on_one_side(tmp_path):
    evidence = {'evidence': {'world': ['shared fact']}}
    # A single shared evidence item relates invocations being split, not a train world to a protected one.
    protected = custody.ProtectedWorlds.from_cases({'dev': ('0' * 64, [world('dev-0', 'test', curriculum={
        'evidence': {'world': ['shared fact', 'other fact']}})])})
    rows = [world('a'), world('b', curriculum=evidence), world('c', curriculum=evidence),
            world('d', source_groups=['g']), world('e', source_groups=['g'])]
    assert custody.related_components(rows) == [[0], [1, 2], [3, 4]]
    receipt = custody.verify_slices({'development': rows[:3], 'confirmation': rows[3:]}, protected=protected)
    assert custody.check_slice_receipt(receipt, {'development': rows[:3], 'confirmation': rows[3:]}) == receipt
    with pytest.raises(custody.CustodyError, match='share related'):
        custody.verify_slices({'development': rows[:2], 'confirmation': rows[2:]}, protected=protected)
    with pytest.raises(custody.CustodyError, match='receipt'):
        custody.check_slice_receipt(receipt, {'development': rows[:3], 'confirmation': rows[3:4]})
    with pytest.raises(custody.CustodyError, match='protected-world custody'):
        custody.verify_slices({'development': rows}, protected=None)


def test_gate_selects_on_development_and_spends_confirmation_once(tmp_path):
    m = lambda successes: {'successes': successes, 'scored': 4}
    assert acceptance.select(m(1), m(2)) and not acceptance.select(m(2), m(2))
    assert not acceptance.select(m(1), {'successes': 3, 'scored': 3})
    ledger, slice_ = tmp_path / 'ledger.json', {'rows': 2, 'rows_sha256': 'a' * 64}
    baseline = {'development': m(1), 'confirmation': m(1)}
    decision = acceptance.confirm(artifact='state-a', baseline=baseline,
                                  selected={'development': m(3), 'confirmation': m(2)},
                                  min_gain_fraction=.25, confirmation=slice_, ledger=ledger)
    assert decision['status'] == 'qualified' and decision['selected']['confirmation'] == m(2)
    assert acceptance.confirm(artifact='state-a', baseline=baseline,  # an exact resume is idempotent
                              selected={'development': m(3), 'confirmation': m(2)},
                              min_gain_fraction=.25, confirmation=slice_, ledger=ledger) == decision
    acceptance.require_untouched(slice_, 'state-a', ledger)
    with pytest.raises(acceptance.AcceptanceError, match='already spent'):
        acceptance.require_untouched(slice_, 'state-b', ledger)
    with pytest.raises(acceptance.AcceptanceError, match='already spent'):
        acceptance.confirm(artifact='state-b', baseline=baseline, selected={'development': m(4), 'confirmation': m(4)},
                           min_gain_fraction=.25, confirmation=slice_, ledger=ledger)
    other = acceptance.confirm(artifact='state-b', baseline=baseline,
                               selected={'development': m(4), 'confirmation': m(1)}, min_gain_fraction=.25,
                               confirmation={**slice_, 'rows_sha256': 'b' * 64}, ledger=ledger)
    assert other['status'] == 'no_lift'
