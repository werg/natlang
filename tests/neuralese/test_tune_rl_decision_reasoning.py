"""Tune RL (whole-graph native replay) trains a reasoned decision's action reasoning with its choice, as GRPO does
(DECISIONS 2026-11-23 "GRPO trains the action reasoning", tune follow-up): the tune rollout records every scored entry
as a turn (only the action entry carries the sampling receipt) and keeps the action reasoning's capture;
``episode_graph.whole_episode_turns`` adds it as a sampled turn through its ``</think>``; ``compile_replay`` counts its
node among the decision turn's; the other samples are context."""
import copy
from types import SimpleNamespace

import pytest

from natlang_neuralese.rl.episode_graph import REASONING_SUFFIX, action_reasoning_capture, whole_episode_turns
from natlang_neuralese.rl.native_replay import compile_replay

CLOSE = 99
IDENTITY = {'checkpoint_sha256': '1' * 64, 'tokenizer_fingerprint': '2' * 64, 'embedding_weight_sha256': '3' * 64,
            'architecture': 'lfm2'}
ROLES = SimpleNamespace(produce_open=58, produce_close=59, consume_open=60, consume_close=61)
PROMPT = [1, 2, 3]
SAMPLED = [[5, 6, CLOSE, 40, 2], [7, CLOSE, 41, 2]]


def _reasoning_capture(ids):
    return {'schema': 'natlang.native-rl-sampling/2', 'policy': dict(IDENTITY), 'finish_reason': 'stop',
            'prompt': [{'kind': 'tokens', 'token_ids': PROMPT, 'producing': [False] * 3}], 'writes': [],
            'sample_range_convention': 'half-open',
            'sampling_law': {'temperature': 1, 'seed': 4, 'guidance': False, 'probabilities': 'post-processing-normalized'},
            'samples': [{'index': i, 'token_id': t, 'logp': -0.1 * (i + 1), 'input_role': 'ordinary',
                         'terminal': i == len(ids) - 1, 'query_own_position': len(PROMPT) - 1 + i}
                        for i, t in enumerate(ids)],
            'generated_inputs': [{'sample_index': i, 'token_id': t, 'input_role': 'ordinary'}
                                 for i, t in enumerate(ids[:-1])]}


def _episode(action=1):
    """A reasoned readout of two samples as tune rollouts record it: one turn per scored entry, in graph order."""
    turns = []
    for j, ids in enumerate(SAMPLED):
        scored = PROMPT + ids[:ids.index(CLOSE) + 1]
        capture = {'schema': 'natlang.native-decision-capture/1', 'policy': dict(IDENTITY),
                   'prompt': [{'kind': 'tokens', 'token_ids': scored, 'producing': [False] * len(scored)}],
                   'continuations': [[11, 2], [13, 2]], 'reasoning': f'r{j}'}
        marker = {'index': j, 'samples': len(SAMPLED), 'action': j == action}
        if j == action:
            capture['decision_sample'] = {'schema': 'natlang.decision-sample/1', 'log_probs': [-0.2, -1.7],
                                          'chosen': 0, 'temperature': 1, 'seed': 7}
            marker['reasoning_sampling'] = _reasoning_capture(ids)
        turns.append({'messages': [{'role': 'user', 'content': 'x'}], 'invocation_id': 'c1', 'decision_reasoning': marker,
                      'target': {'role': 'assistant', 'content': '"a"', 'reasoning_content': f'r{j}'},
                      'sampling': capture})
    return {'task_id': 't', 'turns': turns, 'execution_graph': []}


def _compile(episode, think_close):
    turns = whole_episode_turns(episode, think_close)
    return turns, compile_replay(turns, identity=IDENTITY, roles=ROLES, fetch_block=None)


def test_the_action_reasoning_trains_with_its_decision_and_the_other_sample_is_context():
    turns, replay = _compile(_episode(action=1), CLOSE)
    assert [t['graph_turn'] for t in turns] == ['c1#turn1', 'c1#turn2' + REASONING_SUFFIX, 'c1#turn2']
    reasoning = [a for a in replay.actions if a['turn'] == 'c1#turn2' + REASONING_SUFFIX]
    assert [a['token_id'] for a in reasoning] == [7, CLOSE]                   # through </think>; the free reply is not
    assert [a['behavior_logp'] for a in reasoning] == pytest.approx([-0.1, -0.2])
    assert not [a for a in replay.actions if a['turn'] == 'c1#turn1']         # the other sample: context only
    assert [d['turn'] for d in replay.decision_actions] == ['c1#turn2']       # the choice still trains
    assert replay.turn_nodes['c1#turn2'][0] == 'c1#turn2' + REASONING_SUFFIX  # one turn: reasoning, then its options
    assert set(replay.turn_nodes) == {'c1#turn1', 'c1#turn2'}
    assert {n for nodes in replay.turn_nodes.values() for n in nodes} == {node.name for node in replay.plan.invocations}


def test_the_opt_out_and_uncut_captures_train_the_choice_only():
    _, replay = _compile(_episode(), None)
    assert not replay.actions and len(replay.decision_actions) == 1
    turn = _episode()['turns'][1]
    writing = copy.deepcopy(turn)
    writing['decision_reasoning']['reasoning_sampling']['writes'] = [{'state_id': 'x'}]
    assert action_reasoning_capture(writing, CLOSE) is None
    unsampled = copy.deepcopy(turn)
    del unsampled['sampling']['decision_sample']
    assert action_reasoning_capture(unsampled, CLOSE) is None
    cut = action_reasoning_capture(turn, CLOSE)
    assert [s['token_id'] for s in cut['samples']] == [7, CLOSE] and len(cut['generated_inputs']) == 2
    unclosed = copy.deepcopy(turn)
    unclosed['decision_reasoning']['reasoning_sampling'] = _reasoning_capture([7, 8])
    assert [s['token_id'] for s in action_reasoning_capture(unclosed, CLOSE)['samples']] == [7, 8]
