"""Top-p as one sampling law (owner 2026-11-13): the shared controls, the native draw, the recorded law, the learner's
refusal of another law and the held comparison's refusal to pair laws."""
import collections
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import grpo  # noqa: E402
from lower_rollout_turns import episode_turns  # noqa: E402
from natlang_neuralese.common import sampling  # noqa: E402
from natlang_neuralese.rl import held_eval  # noqa: E402


def test_shared_controls_and_law():
    assert sampling.RL_TOP_P == 0.95
    assert sampling.rl_sampling_controls() == {'guidance': False, 'top_p': 0.95}
    assert sampling.rl_sampling_controls(capture=True) == {'guidance': False, 'x_natlang_rl_capture': True,
                                                           'x_natlang_rl_top_k': 0, 'top_p': 0.95}
    assert sampling.rl_sampling_controls(1.0, capture=True) == {'guidance': False, 'x_natlang_rl_capture': True,
                                                                'x_natlang_rl_top_k': 0}   # off: the old controls
    assert sampling.sampling_law_of(.9) == {'temperature': 1., 'top_p': .9} and sampling.check_top_p(None) == 1.
    for bad in (0, 1.5, -1, 'x', float('nan'), True):
        with pytest.raises(ValueError):
            sampling.check_top_p(bad)


def test_nucleus_draw_keeps_a_token_while_the_mass_before_it_is_within_top_p():
    probs = torch.tensor([.05, .5, .15, .3])
    ordered = probs.sort(descending=True).values                       # .5 .3 .15 .05, mass before: 0 .5 .8 .95
    assert sampling.nucleus_keep(ordered, .75).tolist() == [True, True, False, False]
    assert sampling.nucleus_keep(ordered, .85).tolist() == [True, True, True, False]
    assert sampling.nucleus_keep(ordered, 1e-9).tolist() == [True, False, False, False]
    generator = torch.Generator().manual_seed(0)
    drawn = collections.Counter(sampling.nucleus_sample(probs.log(), .75, generator) for _ in range(400))
    assert set(drawn) == {1, 3} and 200 < drawn[1] < 300               # renormalized: 5/8 and 3/8


def _turn(samples, law):
    return {'invocation_id': 'i', 'sampling': {
        'prompt': [{'kind': 'tokens', 'token_ids': [1, 2, 3]}], 'policy': {'checkpoint_sha256': 'p'},
        'sampling_law': {'temperature': 1, 'guidance': False, **law},
        'samples': [{'token_id': t, 'logp': lp, 'input_role': 'ordinary'} for t, lp in samples]}}


def test_the_learner_refuses_calls_sampled_under_another_top_p():
    row = {'turns': [_turn([(5, -.1)], {'probabilities': 'post-processing-normalized'}),
                     _turn([(5, -.1)], {'probabilities': 'post-temperature-untruncated', 'top_p': .95}),
                     _turn([(5, -.1)], {'probabilities': 'post-temperature-untruncated', 'top_p': .9})]}
    rows = episode_turns(row, collections.Counter())
    assert [r['top_p'] for r in rows] == [1., .95, .9]
    for top_p, kept in ((.95, [1]), (1., [0]), (.9, [2]), (.8, [])):
        stats = collections.Counter()
        assert [r['index'] for r in grpo.trainable_rows(rows, stats, top_p)] == kept
        assert stats['skip_other_sampling_law'] == 3 - len(kept)


def _result(policy, law=None):
    outcomes = {(f't{i}', s): float(i < 4) for i in range(8) for s in (0, 1)}
    return {'schema': held_eval.RESULT_SCHEMA, 'policy_id': policy, 'started_at': 1, 'series': {'key': 's'}, 'complete': True,
            'ran': len(outcomes), 'scored': len(outcomes), 'successes': 8, 'loop_rate': 0., 'repeated_action_rate': 0.,
            'outcome_counts': {}, **({'sampling': law} if law else {}),
            'samples': [{'task_id': t, 'sample': s, 'reward': r, 'ran': True} for (t, s), r in outcomes.items()]}


def test_held_compare_refuses_results_of_different_sampling_laws():
    nucleus, plain = sampling.sampling_law_of(.95), sampling.sampling_law_of(1.)
    assert held_eval.compare(_result('a', nucleus), _result('b', nucleus))['primary']['delta'] == 0
    assert held_eval.compare(_result('a'), _result('b', plain))['primary']['delta'] == 0   # unrecorded: untruncated
    for first, second in ((nucleus, plain), (None, nucleus), (nucleus, sampling.sampling_law_of(.9))):
        with pytest.raises(ValueError, match='different laws'):
            held_eval.compare(_result('a', first), _result('b', second))
