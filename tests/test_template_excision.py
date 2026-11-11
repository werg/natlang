import json

import pytest
from scripts.neuralese_data import prepare_template_excision as tx


class Source:
    file_sha256 = 'a' * 64
    entries = [{'id': 'w-1'}, {'id': 'w-2'}, {'id': 't-1'}, {'id': 'x-1'}]


FAMILY = {'w-1': tx.WRAPPER, 'w-2': tx.WRAPPER, 't-1': tx.TEMPLATE, 'x-1': 'natlang'}
FRAME = {'exposure_counts': {'w-1': 1}, 'required_visits': {'w-1': 1, 'w-2': 1, 't-1': 2}}


def plan(*ids):
    return {'schema': 'natlang.template-excision-plan/1', 'families': list(tx.FAMILIES),
            'source': {'sha256': Source.file_sha256},
            'withdraw': {i: {'family': FAMILY[i], 'criterion': 'test'} for i in ids}}


def test_plan_withdraws_unearned_rows_and_reports_earned_rows_kept():
    chosen, earned = tx.planned_withdrawal(plan('w-2', 't-1'), FRAME, FAMILY, Source)
    assert chosen == {'w-2': tx.WRAPPER, 't-1': tx.TEMPLATE} and earned == {tx.WRAPPER: 1}


@pytest.mark.parametrize('ids', [('w-1',), ('x-1',), ('nope',)])
def test_plan_refuses_an_earned_foreign_or_unknown_row(ids):
    family = {**FAMILY, 'nope': tx.WRAPPER}
    p = plan('w-2')
    p['withdraw'] = {ids[0]: {'family': family[ids[0]], 'criterion': 'test'}}
    with pytest.raises(ValueError):
        tx.planned_withdrawal(p, FRAME, family, Source)


def test_manifest_rewrite_drops_whole_leaves_and_swaps_filtered_ones(tmp_path):
    (tmp_path / 'manifests').mkdir()
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': '/x/a.jsonl', 'sha256': 'a'}, {'kind': 'native-graphs', 'path': '/x/b.jsonl', 'sha256': 'b'},
        {'kind': 'native-graphs', 'path': '/x/c.jsonl', 'sha256': 'c'}]}))
    doc, changed = tx.rewrite_manifest(manifest, {'a': None, 'b': {'path': '/y/b2.jsonl', 'sha256': 'b2', 'bytes': 3}}, tmp_path)
    assert changed and [s['sha256'] for s in doc['sources']] == ['b2', 'c'] and doc['sources'][0]['bytes'] == 3
    assert tx.rewrite_manifest(manifest, {'a': None, 'b': None, 'c': None}, tmp_path)[0] is None


def test_commit_admission_records_the_window_once(tmp_path):
    state = tmp_path / 'state.json'
    state.write_text(json.dumps({'interleave': {'sft_windows': 2}}))
    patch = tmp_path / 'patch.json'
    patch.write_text(json.dumps({'ledger_entry': {'recipe': {'path': 'r', 'sha256': 'r'}, 'record': {'path': 'a', 'sha256': 'a'}},
                                 'settled_recipe': {'path': 's', 'sha256': 's'}}))
    with pytest.raises(ValueError):
        tx.commit_admission(state, patch, 3, write=True)  # not at that boundary
    tx.commit_admission(state, patch, 2, write=False)
    assert 'memetic_distillation' not in json.loads(state.read_text())['interleave']
    tx.commit_admission(state, patch, 2, write=True)
    ledger = json.loads(state.read_text())['interleave']['memetic_distillation']
    assert ledger['admissions'][-1]['window'] == 2 and ledger['recipe']['path'] == 's'
    with pytest.raises(ValueError):
        tx.commit_admission(state, patch, 2, write=True)
