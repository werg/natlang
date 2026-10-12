"""Program-header training efficiency (plans/neuralese/HEADER_EFFICIENCY_20261012.md): linear decision chains merged
into one forward (text_warmup.merge_prefix_rows), whole-window train supervision (--train-target-tokens) with one
balanced segment per decision (text_supervision.segmented_position_weights), and the windowing continuation that moves
a running stage onto the new geometry."""
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train import direct_sft as trainer
from natlang_neuralese.train.text_supervision import (ROLE_CODES, balanced_position_weights,
                                                      segmented_position_weights)
from natlang_neuralese.train.text_warmup import document_windows, merge_prefix_rows, prepare_text_windows


def test_one_suffix_segment_is_exactly_the_balanced_weights():
    torch.manual_seed(0)
    for size, start in ((12, 5), (30, 29), (9, 0), (17, 17)):
        span = torch.randint(3, 50, (1, size))
        roles = torch.randint(0, len(ROLE_CODES), (1, size))
        exclude = torch.rand(1, size) < .2
        for kwargs in ({}, {'roles': roles}, {'roles': roles, 'exclude': exclude}):
            expected = balanced_position_weights(span, [start if start < size else None], **kwargs)
            spans = [(start, size)] if start < size else []
            assert torch.allclose(segmented_position_weights(span, spans, **kwargs), expected, atol=1e-6)


def test_each_decision_keeps_half_its_segment_on_its_suffix():
    span = torch.zeros(1, 20, dtype=torch.long)
    weights = segmented_position_weights(span, [(4, 6), (15, 20)])[0]
    assert float(weights.sum()) == pytest.approx(20)
    first, second = weights[:6], weights[6:]
    assert float(weights[4:6].sum()) == pytest.approx(.5 * 6 + .5 * 6 * 2 / 6)  # suffix half plus its context share
    assert float(first.sum()) == pytest.approx(6) and float(second.sum()) == pytest.approx(14)
    # A context-only tail after the last suffix is a context segment.
    tail = segmented_position_weights(span, [(2, 4)])[0]
    assert torch.allclose(tail[4:], torch.ones(16)) and float(tail.sum()) == pytest.approx(20)
    with pytest.raises(ValueError):
        segmented_position_weights(span, [(5, 9), (8, 12)])


def rows_with_chains():
    head = list(range(100, 170))  # a 70-token program header shared by every decision
    a = head + [1, 2, 3]
    b = a + [4, 5, 6, 7]
    c = b + [8, 9]
    branch = a + [40, 41, 42]  # diverges from b after a: its own forward
    other = list(range(300, 380))
    row = lambda ids, start, name, split='train': {'text': name, 'id': name, 'token_ids': ids, 'split': split,
                                                   'source_groups': ['g'], 'supervised_suffix_start': start}
    return [row(c, 77, 'c'), row(a, 70, 'a'), row(branch, 73, 'branch'), row(b, 73, 'b'), row(other, 75, 'other'),
            row(b, 73, 'held', split='test')]


def test_linear_chains_merge_into_their_longest_row_and_branches_stay_apart():
    rows = rows_with_chains()
    merged, receipt = merge_prefix_rows(rows)
    by_id = {r['id']: r for r in merged}
    assert sorted(by_id) == ['branch', 'c', 'held', 'other']
    # a is a prefix of b (then c) and of branch: byte order puts b right after a, so a goes to b's container c.
    assert [m[1] for m in by_id['c']['merged_rows']] == ['c', 'a', 'b']
    assert by_id['c']['supervised_spans'] == [(70, 73), (73, 77), (77, 79)]
    assert 'supervised_spans' not in by_id['branch']
    assert 'supervised_spans' not in by_id['other'] and 'supervised_spans' not in by_id['held']
    assert receipt['contained_rows'] == 2 and receipt['merged_rows'] == 4
    again, _ = merge_prefix_rows(list(reversed(rows)))
    assert {r['id']: r.get('supervised_spans') for r in again} == {r['id']: r.get('supervised_spans') for r in merged}


def test_windows_clip_the_suffix_spans_to_their_span():
    windows = document_windows(range(30), open_id=1, close_id=2, tokens=16, prefix_tokens=4, target_tokens=10,
                               supervised_spans=[(3, 5), (12, 14), (25, 30)])
    # Span position i of a window predicts token_ids position start + prefix + i - 1 (the frame's open token).
    covered = sorted(window['start'] + window['prefix'] + i - 1 for window in windows
                     for b, e in window['supervised_spans'] for i in range(b, e))
    assert covered == [3, 4, 12, 13, 25, 26, 27, 28, 29]


def engine():
    tokenizer = SimpleNamespace(unk_token_id=-1, convert_tokens_to_ids=lambda value: -1)
    return SimpleNamespace(tokenizer=tokenizer, backbone=SimpleNamespace(
        controls=SimpleNamespace(open_id=1, close_id=2)), _tokens=lambda text: [])


def test_train_windows_take_the_train_span_and_held_windows_keep_theirs():
    rows, _ = merge_prefix_rows(rows_with_chains())
    windows, _ = prepare_text_windows(engine(), rows, tokens=64, prefix_tokens=4, target_tokens=8,
                                      train_target_tokens=60)
    train = {w['document_id']: w for w in windows['train'] if trainer.target_bearing(w)}
    assert all(len(w['ids']) - w['prefix'] <= 60 for w in windows['train'])
    assert all(len(w['ids']) - w['prefix'] <= 8 for w in windows['test'])
    merged = train['c']
    assert merged['merged_rows'][0][1] == 'c' and merged['supervised_spans']
    old, _ = prepare_text_windows(engine(), rows_with_chains(), tokens=64, prefix_tokens=4, target_tokens=8)
    forward = lambda ws: sum(len(w['ids']) for w in ws if trainer.target_bearing(w))
    assert forward(windows['train']) < forward(old['train'])


def test_a_merged_window_is_unseen_while_any_constituent_is():
    window = {'document': 'h-c', 'document_id': 'c', 'merged_rows': [['h-c', 'c'], ['h-b', 'b']]}
    seen = lambda shas: {'seen_document_sha256': shas, 'seen_document_ids': []}
    assert trainer.continued_unseen([window], seen(['h-c'])) == [window]
    assert trainer.continued_unseen([window], seen(['h-c', 'h-b'])) == []


def plan_args(**overrides):
    return SimpleNamespace(**{'seed': 0, 'batch_tokens': 400, 'batch': 4, 'epochs': 1.0, 'steps': None, 'tokens': 64,
                              'prefix_tokens': 4, 'target_tokens': 8, 'train_target_tokens': None,
                              'merge_prefix_rows': False, **overrides})


def corpus(programs=30):
    rows = []
    for p in range(programs):
        head = [100 + p] * 3 + list(range(200, 266))
        ids = list(head)
        for d in range(1 + p % 4):
            start = len(ids)
            ids = ids + [p % 7 + 3] * (2 + d)
            rows.append({'text': f'p{p}d{d}', 'id': f'p{p}:decision:{d}', 'token_ids': list(ids), 'split': 'train',
                         'source_groups': [f'g{p}'], 'supervised_suffix_start': start})
            ids = ids + [9, 9]  # the tool response the next decision sees
    rows.append({'text': 'held', 'id': 'held', 'token_ids': list(range(150, 230)), 'split': 'test',
                 'source_groups': ['held'], 'supervised_suffix_start': 70})
    return rows


def targets_of(rows, a):
    if a.merge_prefix_rows:
        rows, _ = merge_prefix_rows(rows)
    windows, _ = prepare_text_windows(engine(), rows, tokens=a.tokens, prefix_tokens=a.prefix_tokens,
                                      target_tokens=a.target_tokens, train_target_tokens=a.train_target_tokens)
    return [w for w in windows['train'] if trainer.target_bearing(w)], windows['test']


def test_a_windowing_continuation_rebuilds_the_trained_plan_and_trains_the_unseen_rest_once():
    rows = corpus()
    old_a = plan_args()
    new_a = plan_args(train_target_tokens=56, merge_prefix_rows=True)
    old_targets, held = targets_of(rows, old_a)
    new_targets, _ = targets_of(rows, new_a)
    _, old_plan = trainer.direct_schedule(old_targets, old_a)
    consumed = 4
    stage = {'id': 'direct_sft', 'progress': [0.0, 0.35], 'start': 0, 'end': 'derived'}
    record, plan = trainer.data_continuation(old_targets, held, new_targets, held, consumed=consumed, previous=None,
                                             quantization_stage=stage, a=new_a, previous_a=old_a)
    assert record['windowing'] == {'previous': {'train_target_tokens': None, 'merge_prefix_rows': False},
                                   'current': {'train_target_tokens': 56, 'merge_prefix_rows': True}}
    assert record['held']['rebased'] is False
    seen = {w['document'] for batch, _ in old_plan[:consumed] for w in batch}
    trained = [w for batch, _ in plan[consumed:] for w in batch]
    assert all(any(sha not in seen for sha, _ in w.get('merged_rows', [[w['document'], None]])) for w in trained)
    expected = trainer.continued_unseen(new_targets, record)
    assert sorted(map(id, trained)) == sorted(map(id, expected))
    # Fewer forward tokens for the same decisions: the chains are forwarded once.
    forward = lambda ws: sum(len(w['ids']) for w in ws)
    assert forward(new_targets) < forward(old_targets)
    # A chained continuation verifies that its previous plan rebuilds: from the old geometry's windows it does not.
    trainer.data_continuation(new_targets, held, new_targets, held, consumed=consumed + 1, previous=record,
                              quantization_stage=stage, a=new_a)
    with pytest.raises(ValueError, match='does not rebuild'):
        trainer.data_continuation(old_targets, held, new_targets, held, consumed=consumed + 1, previous=record,
                                  quantization_stage=stage, a=new_a)


def test_a_new_train_window_geometry_on_the_same_data_is_a_continuation(tmp_path):
    data, heads = tmp_path / 'r3.jsonl', tmp_path / 'heads.pt'
    data.write_text('r3')
    heads.write_text('h')
    a = SimpleNamespace(heads=heads, records=None, pieces=None, text_data=data, continue_from=None,
                        continue_identity=None, continue_monitor_reference=None, continued_text_data=None,
                        train_target_tokens=None, merge_prefix_rows=False)
    inputs = trainer.identity_inputs(a)
    record = {'consumed_updates': 7}
    state = {'step': 9, 'identity': {'inputs': inputs, 'options': {}}, 'data_continuation': record}
    assert trainer.continuation_mode(a, state, inputs) == ('resume', record)
    a.train_target_tokens, a.merge_prefix_rows = 31744, True
    assert trainer.continuation_mode(a, state, inputs) == ('continue', record)
    state['identity']['options'] = {'train_target_tokens': 31744, 'merge_prefix_rows': True}
    assert trainer.continuation_mode(a, state, inputs) == ('resume', record)


def test_the_recipe_handoff_admits_a_window_geometry_change_of_an_unfinished_direct_stage(tmp_path):
    from natlang_neuralese.train import recipe as runner
    stage = {'id': 'direct_sft', 'kind': 'neuralese_direct', 'parameters': {'epochs': 1.0}}
    old = {'stage_parameter_defaults': {'neuralese_direct': {'tokens': 32768}}, 'stages': [stage]}
    new = {'stage_parameter_defaults': {'neuralese_direct': {'tokens': 32768, 'train_target_tokens': 31744,
                                                             'merge_prefix_rows': True}}, 'stages': [stage]}
    changes, old_stripped, new_stripped = runner.direct_windowing_handoffs(old, new, tmp_path)
    assert changes == {'direct_sft': {'train_target_tokens': [None, 31744], 'merge_prefix_rows': [None, True]}}
    assert old_stripped == new_stripped


tiny = pytest.importorskip('transformers')


def test_a_merged_chain_gives_every_contained_decision_its_own_token_losses(tmp_path):
    """Causal exactness on a tiny Mellum (sliding window 8, YaRN past 16 tokens): the token losses of a contained
    row's positions inside its container's forward equal those of the row's own forward (all but the framing close)."""
    if not hasattr(tiny, 'MellumConfig'):
        pytest.skip('installed transformers has no Mellum')
    from test_mellum_port import tiny_mellum
    from natlang_neuralese.maple.model import load_maple
    tiny_mellum(tmp_path)
    model = load_maple(tmp_path, device='cpu', dtype=torch.float32, ternary_attention=False)
    torch.manual_seed(1)
    container = torch.randint(3, 128, (1, 31))
    contained = container[:, :19]  # longer than the window and YaRN's original context
    losses = lambda ids: torch.nn.functional.cross_entropy(model(ids[:, :-1]).logits[0], ids[0, 1:], reduction='none')
    with torch.no_grad():
        alone, merged = losses(contained), losses(container)
    assert torch.allclose(alone, merged[:alone.shape[0]], atol=1e-5, rtol=1e-5)
