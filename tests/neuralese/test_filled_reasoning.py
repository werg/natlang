"""Filled-in reasoning for teacher trajectories (scripts/neuralese_data/filled_reasoning.py): leakage, the hint turn's
place, length-banded selection, rows with the filled history, per-row sampled flags, and the build's ``replaces``."""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import exp0_sft  # noqa: E402
import filled_reasoning as fr  # noqa: E402
import sft_sources  # noqa: E402

CLOSE, IM_START = 9, 1


def test_leak_catches_quotes_and_names_of_hinted_text_not_seen_in_context():
    action = '<tool_call><function=eval><parameter=code>const total = rows.reduce((a, r) => a + r.amount, 0);' \
             '</parameter></function></tool_call>'
    assert fr.leak('I will compute const total = rows.reduce((a, r) => a + r.amount, 0); and return it', [action], '')
    assert fr.leak('Let me sum the amounts with a reduce over the rows.', [action], '') is None
    # a quote of what the student has already seen is not a leak
    seen = 'const total = rows.reduce((a, r) => a + r.amount, 0);'
    assert fr.leak('as before: const total = rows.reduce((a, r) => a + r.amount, 0); again', [action], seen) is None
    # a short action named outright, and a reference to the hint itself
    assert fr.leak('so the answer is readFile("notes.md") now', ['readFile("notes.md")'], '') == 'readFile("notes.md")'
    assert fr.leak('The verified reference says to read the file.', [action], '') == 'hint_phrase'


def test_hint_turn_sits_before_the_assistant_opening_so_the_history_is_a_shared_prefix():
    context = [IM_START, 5, 6, 2, IM_START, 7, 2, IM_START, 8, 3]   # system, tool, assistant opening
    hinted = fr.hint_context(context, [IM_START, 40, 41, 2], IM_START)
    assert hinted[:7] == context[:7] and hinted[-3:] == context[-3:] and hinted[7:11] == [IM_START, 40, 41, 2]


def test_improvement_is_per_action_token_over_the_common_spans():
    assert abs(fr.improvement([-0.1, -0.5], [-0.3, -0.5, -0.9], [10, 30]) - 0.05) < 1e-9
    assert fr.improvement([], [-1.], [3]) is None


def test_selection_compares_only_within_the_students_own_length_band():
    def c(origin, length, imp, admitted=True, closed=True):
        return {'origin': origin, 'length': length, 'improvement': imp, 'admitted': admitted, 'closed': closed}
    short_best = c('hinted', 100, 0.9)          # closes early and scores best: outside every band
    in_band = c('hinted', 900, 0.2)
    own = c('unhinted', 1000, 0.1)
    chosen, anchor = fr.select([short_best, in_band, own], (0.67, 1.5), random.Random(0))
    assert chosen is in_band and anchor == 1000
    # nothing admitted inside the band: no fill (the call keeps an empty block)
    assert fr.select([short_best, c('unhinted', 1000, -0.1, admitted=False)], (0.67, 1.5), random.Random(0)) == (None, None)
    # no closed unhinted sample: no anchor, no fill
    assert fr.select([in_band, c('unhinted', 4096, None, admitted=False, closed=False)], (0.67, 1.5),
                     random.Random(0)) == (None, None)


def chain_of(calls):
    """A teacher chain as exp0_sft.chains_of builds it: prompts extend (prompt, sample, suffix) of the call before."""
    rows, prompt = [], [IM_START, 50, 51, 2, IM_START, 3]
    for index, (think, action, suffix) in enumerate(calls):
        sample = think + [CLOSE] + action + [2]
        weight = np.array([0.] * (len(think) + 1) + [1.] * (len(action) + 1), np.float32)
        rows.append({'id': f'r{index}', 'invocation': 'inv', 'index': index, 'prompt': np.array(prompt, np.int32),
                     'sample': np.array(sample, np.int32), 'weight': weight})
        prompt = prompt + sample + suffix
    return exp0_sft.chains_of([{'task_id': 't', 'round': 'teacher:k', 'rows': rows}])


def test_filled_rows_carry_the_filled_history_and_train_block_and_action():
    [chain] = chain_of([([70, 71], [80], [2, IM_START, 60, 2, IM_START, 3]), ([72], [81, 82], [])])
    calls = fr.calls_of(chain, CLOSE)
    assert [list(c['action']) for c in calls] == [[80, 2], [81, 82, 2]]
    fills = [{'kind': 'filled', 'reasoning': [100, 101, 102, CLOSE]}, {'kind': 'fallback', 'reasoning': None}]
    rows = fr.filled_rows(calls, fills, CLOSE, 'filled:teacher:k')
    # call 1's prompt holds call 0's filled block, never the teacher's think tokens
    assert list(rows[1]['prompt']) == list(rows[0]['prompt']) + [100, 101, 102, CLOSE, 80, 2] + [2, IM_START, 60, 2, IM_START, 3]
    assert list(rows[0]['weight']) == [1.] * 4 + [1., 1.] and list(rows[0]['sampled']) == [True] * 4 + [False] * 2
    assert list(rows[1]['sample']) == [CLOSE, 81, 82, 2] and list(rows[1]['weight']) == [0., 1., 1., 1.]
    assert not rows[1]['sampled'].any()
    built = exp0_sft.chains_of([{'task_id': 't', 'round': 'filled:teacher:k', 'rows': rows}])
    assert len(built) == 1   # the rebuilt prompts still extend each other: one sequence
    assert exp0_sft.sampled_targets(built[0]).tolist() == [True] * 4 + [False] * 2 + [False] * 4
    assert exp0_sft.sampled_mask(built, 'cpu').tolist() == exp0_sft.sampled_targets(built[0]).tolist()


def test_a_context_only_action_keeps_its_filled_block_untrained():
    [chain] = chain_of([([70], [80], [])])
    calls = fr.calls_of(chain, CLOSE)
    calls[0]['action_weight'][:] = 0   # a backend detour: all context
    rows = fr.filled_rows(calls, [{'reasoning': [100, CLOSE]}], CLOSE, 'n')
    assert not rows[0]['weight'].any()


def test_chain_level_sampled_flags_are_unchanged_without_row_flags():
    def chain(round_name):
        return {'ids': np.zeros(4, np.int32), 'segs': [(1, 2, {'sample': np.zeros(2, np.int32)})], 'task': 't',
                'round': round_name}
    assert exp0_sft.sampled_targets(chain('rl:x')).tolist() == [True, True]
    assert exp0_sft.sampled_targets(chain('teacher:x')).tolist() == [False, False]


def test_aux_replaces_names_the_episode_a_filled_unit_stands_in_for():
    part = sft_sources.parse_aux('filled=/x/chains.pkl,share=0.2,replaces=teacher')
    assert part['replaces'] == 'teacher' and part['share'] == 0.2
    assert sft_sources.replaced_round(part, {'round': 'filled:teacher:pool:t1:0'}) == 'teacher:pool:t1:0'
    assert sft_sources.replaced_round(part, {'round': 'filled:rl:run:g1'}) is None
    assert sft_sources.replaced_round(sft_sources.parse_aux('ops=/x.pkl'), {'round': 'filled:teacher:a'}) is None
