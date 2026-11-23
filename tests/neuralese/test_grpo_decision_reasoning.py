"""GRPO trains a reasoned decision's action reasoning (DECISIONS 2026-11-23 "GRPO trains the action reasoning"): the
sample whose choice the episode took keeps its generation's exact capture through ``rollout_turns``, and
``lower_rollout_turns.decision_reasoning_turns`` lowers it as an ordinary sampled call (exact prompt and sampled ids,
the actor's log-probabilities, through its ``</think>``); the other samples give no row; ``--no-decision-reasoning``
opts out."""
import collections
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese')]
import exp0_sft  # noqa: E402
import grpo  # noqa: E402
from lower_rollout_turns import decision_reasoning_turns  # noqa: E402
from natlang_neuralese.rl import jolt  # noqa: E402
from natlang_neuralese.rl.rollout_math import truncated_rollout  # noqa: E402
from natlang_neuralese.rl.task_rollouts import rollout_turns  # noqa: E402

CLOSE = 99
PROMPT = [1, 2, 3]
# per sample: reasoning ids, then </think>, then the free reply the readout never used
SAMPLED = [[5, 6, CLOSE, 40, 41], [7, CLOSE, 42], [8, 9, 10, CLOSE, 43]]


def _capture(prompt, ids, finish='stop'):
    return {'schema': 'natlang.native-rl-sampling/2', 'policy': {'checkpoint_sha256': 'p'},
            'actor_export': {'sha256': 'e'}, 'finish_reason': finish,
            'prompt': [{'kind': 'tokens', 'token_ids': list(prompt), 'producing': [False] * len(prompt)}],
            'sampling_law': {'temperature': 1, 'top_p': 0.95, 'probabilities': 'post-temperature-untruncated'},
            'samples': [{'index': i, 'token_id': t, 'logp': -0.1 * (i + 1), 'input_role': 'ordinary', 'terminal': False,
                         'query_own_position': len(prompt) + i} for i, t in enumerate(ids)],
            'generated_inputs': [{'sample_index': i, 'token_id': t, 'input_role': 'ordinary'} for i, t in enumerate(ids)]}


def _trajectory(action=2, finish='stop', scored_exact=True):
    """A reasoned readout as the collector writes it: the reasoning generations, then the scored entries."""
    steps = []
    for j, ids in enumerate(SAMPLED):
        steps.append({'phase': 'decision_reasoning', 'invocation_id': 'c1', 'context': [],
                      'decision_reasoning': {'index': j, 'samples': len(SAMPLED)},
                      'assistant': {'content': 'free', 'reasoning': f'r{j}', 'calls': []},
                      'model_response': {'calls': [], 'raw_calls': [],
                                         'rl_sampling': _capture(PROMPT, ids, finish if j == action else 'stop')}})
    for j, ids in enumerate(SAMPLED):
        kept = ids[:ids.index(CLOSE) + 1]
        scored = PROMPT + (kept if scored_exact else [kept[0] + 1] + kept[1:])
        capture = {'schema': 'natlang.native-decision-capture/1', 'policy': {'checkpoint_sha256': 'p'},
                   'prompt': [{'kind': 'tokens', 'token_ids': scored, 'producing': [False] * len(scored)}],
                   **({'decision_sample': {'chosen': 0}} if j == action else {})}
        steps.append({'phase': 'action', 'invocation_id': 'c1', 'context': [], 'tools_offered': [],
                      'decision_reasoning': {'index': j, 'samples': len(SAMPLED), 'action': j == action},
                      'assistant': {'content': '"a"', 'reasoning': f'r{j}', 'calls': []},
                      'model_response': {'calls': [], 'raw_calls': [], 'text': '"a"', 'rl_sampling': capture}})
    return {'trajectory': steps}


def test_only_the_action_entry_keeps_its_reasonings_exact_capture():
    turns = rollout_turns(_trajectory(action=2))
    marks = [t['decision_reasoning'] for t in turns]
    assert [m['action'] for m in marks] == [False, False, True]
    assert ['reasoning_sampling' in m for m in marks] == [False, False, True]
    kept = marks[2]['reasoning_sampling']
    assert [s['token_id'] for s in kept['samples']] == SAMPLED[2] and kept['finish_reason'] == 'stop'
    assert marks[2]['reasoning_sample_ids'] == SAMPLED[2]


def test_the_action_reasoning_lowers_as_an_ordinary_call_through_its_close():
    row = {'turns': rollout_turns(_trajectory(action=2))}
    stats = collections.Counter()
    rows = decision_reasoning_turns(row, stats, think_close=CLOSE)
    assert len(rows) == 1
    (call,) = rows
    assert call['invocation'] == 'c1' and call['index'] == 2 and call['decision_reasoning'] is True
    assert call['prompt'].tolist() == PROMPT
    assert call['sample'].tolist() == [8, 9, 10, CLOSE]                     # the free reply after </think> is not kept
    assert np.allclose(call['logp'], [-0.1, -0.2, -0.3, -0.4])               # the actor's own log-probabilities
    assert call['temperature'] == 1 and call['top_p'] == 0.95 and call['served'] == 'e'
    assert stats['decision_reasoning_rows'] == 1 and stats['decision_reasoning_tokens'] == 4
    assert stats['decision_reasoning_ids_exact'] == 1 and not stats['decision_reasoning_unclosed']
    # the scorer re-rendered the reasoning to other ids: the sampled ids still train, the mismatch is counted
    stats = collections.Counter()
    rows = decision_reasoning_turns({'turns': rollout_turns(_trajectory(action=0, scored_exact=False))}, stats,
                                    think_close=CLOSE)
    assert rows[0]['sample'].tolist() == [5, 6, CLOSE] and stats['decision_reasoning_ids_rendered'] == 1
    # without the close id nothing is guessed
    stats = collections.Counter()
    assert decision_reasoning_turns(row, stats, think_close=None) == []
    assert stats['skip_decision_reasoning_no_close_id'] == 1


def test_a_reasoning_cut_by_the_cap_keeps_every_sample_and_its_episode_counts_as_truncated():
    row = {'turns': rollout_turns(_trajectory(action=1, finish='length'))}
    sampling = row['turns'][1]['decision_reasoning']['reasoning_sampling']
    sampling['samples'] = sampling['samples'][:1]                            # cut before its </think>
    stats = collections.Counter()
    (call,) = decision_reasoning_turns(row, stats, think_close=CLOSE)
    assert call['sample'].tolist() == [7] and stats['decision_reasoning_unclosed'] == 1
    assert truncated_rollout(row)
    assert not truncated_rollout({'turns': rollout_turns(_trajectory(action=1))})
    # a non-action sample cut by the cap only estimated the confidence: no truncation of the episode
    assert not truncated_rollout({'turns': rollout_turns(_trajectory(action=0, finish='stop'))})


def test_student_form_strips_guidance_from_the_action_reasoning_too():
    row = {'turns': rollout_turns(_trajectory(action=2))}
    sampling = row['turns'][2]['decision_reasoning']['reasoning_sampling']
    sampling['prompt'] = [{'kind': 'tokens', 'token_ids': [1, 2]},
                          {'kind': 'tokens', 'token_ids': [90, 91], 'privileged': True},
                          {'kind': 'tokens', 'token_ids': [3]}]
    sampling['privileged'] = {'mode': 'text'}
    stats = collections.Counter()
    converted = jolt.student_form(row, stats)
    kept = converted['turns'][2]['decision_reasoning']['reasoning_sampling']
    assert [s['token_ids'] for s in kept['prompt']] == [[1, 2], [3]] and 'privileged' not in kept
    assert stats['teacher_privileged_tokens_removed'] == 2


def test_load_group_trains_the_action_reasoning_by_default_and_opts_out(tmp_path):
    turns = rollout_turns(_trajectory(action=2))
    for turn in turns:   # the choice's own lowering (decision_turns) is not under test here
        turn['sampling'].pop('decision_sample', None)
    rollout = {'task_id': 't', 'reward': {'quality': 1.0}, 'outcome': 'task_success', 'turns': turns}
    (tmp_path / 'rollouts.jsonl').write_text(json.dumps(rollout) + '\n')
    for on, expected in ((True, [[8, 9, 10, CLOSE]]), (False, [])):
        collector = SimpleNamespace(groups={0: {'dir': tmp_path, 'task_id': 't', 'versions': [0]}}, generated=0,
                                    episodes=0, states=None, think_close=CLOSE,
                                    args=SimpleNamespace(top_p=0.95, reward='binary', decision_reasoning=on))
        stats = collections.Counter()
        _, (episode,), _ = grpo.load_group(collector, 0, stats)
        assert [r['sample'].tolist() for r in episode['rows']] == expected
        chains = exp0_sft.chains_of([episode])
        assert [c['ids'].tolist() for c in chains] == [PROMPT + e for e in expected]
    assert grpo.build_parser().parse_args(['train', '--work', 'w', '--checkpoint', 'c']).decision_reasoning is True
    assert grpo.build_parser().parse_args(['train', '--work', 'w', '--checkpoint', 'c', '--no-decision-reasoning']).decision_reasoning is False
