"""CPU checks of grpo.py's reasoning RL against fixed teacher actions (``ReasoningRL``, ``rrl_episodes``): a search along
a teacher trajectory segment samples the student's think blocks before each teacher action, rewards them by the teacher
actions' likelihood after them (return-to-go), carries the best forward and trains the rest as groups."""
import base64
import collections
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402

CLOSE, END = 7, 9


def record(logp):
    """One fork natlang_top_k record with one entry: float32 sampled logp, int32 id, float16 logp."""
    return base64.b64encode(np.float32(logp).tobytes() + np.int32(0).tobytes() + np.float16(logp).tobytes()).decode()


def test_split_reply():
    assert grpo.split_reply([1, 2, CLOSE, 3, 4], CLOSE) == 3
    assert grpo.split_reply([1, 2, 3], CLOSE) is None
    assert grpo.split_reply([1, CLOSE], CLOSE) is None          # nothing after the close: no action
    assert grpo.split_reply([1, CLOSE, 5, CLOSE, 6], CLOSE) == 2


def test_teacher_call_keeps_action_and_its_weights():
    row = {'prompt': np.array([1, 2, 3], np.int32), 'sample': np.array([4, 5, CLOSE, 0, 6, END], np.int32),
           'weight': np.array([0, 0, 0, 0, 1, 1], np.float32), 'index': 2}
    call = grpo.teacher_call(row, CLOSE, {100})
    assert call['action'].tolist() == [0, 6, END]
    assert call['action_weight'].tolist() == [0, 1, 1]        # template whitespace stays context
    assert call['prompt'].tolist() == [1, 2, 3] and call['index'] == 2
    assert grpo.teacher_call({**row, 'imports': [(1, 'b')]}, CLOSE, set()) == 'imports'
    assert grpo.teacher_call({**row, 'sample': np.array([4, 5])}, CLOSE, set()) == 'no_action'
    assert grpo.teacher_call({**row, 'sample': np.array([CLOSE, 100, END])}, CLOSE, {100}) == 'role_markers'


def trajectory():
    """Three calls of one invocation, each prompt extending the last call's full text, and one of another."""
    p0 = [1, 2, 3]
    r0 = [20, CLOSE, 30, END]
    p1 = p0 + r0 + [40, 41, 3]
    r1 = [21, 21, CLOSE, 31, END]
    p2 = p1 + r1 + [42, 3]
    r2 = [CLOSE, 32, END]
    rows = [{'prompt': np.array(p, np.int32), 'sample': np.array(r, np.int32), 'invocation': 'a', 'index': i}
            for i, (p, r) in enumerate([(p0, r0), (p1, r1), (p2, r2)])]
    rows.append({'prompt': np.array([5, 3], np.int32), 'sample': np.array([CLOSE, 33, END], np.int32),
                 'invocation': 'b', 'index': 0})
    return rows


def test_teacher_segments_chain_calls_with_suffixes():
    runs = grpo.teacher_segments(trajectory(), CLOSE, set())
    runs.sort(key=len)
    assert [len(r) for r in runs] == [1, 3]
    a = runs[1]
    assert a[0]['suffix'].tolist() == [40, 41, 3] and a[1]['suffix'].tolist() == [42, 3] and 'suffix' not in a[2]
    assert [c['action'].tolist() for c in a] == [[30, END], [31, END], [32, END]]
    rows = trajectory()
    rows[1] = {**rows[1], 'sample': np.array([21, 21], np.int32)}   # no action: the run breaks there
    assert sorted(len(r) for r in grpo.teacher_segments(rows, CLOSE, set())) == [1, 1, 1]


def test_teacher_history_keeps_the_trace_or_renders_empty_think_blocks():
    """By default later prompts keep the teacher's recorded reasoning; 'empty' renders empty think blocks."""
    rows = trajectory()
    default = sorted(grpo.teacher_segments(rows, CLOSE, set()), key=len)[1]
    assert [c['prompt'].tolist() for c in default] == [r['prompt'].tolist() for r in rows[:3]]
    calls = sorted(grpo.teacher_segments(rows, CLOSE, set(), history='empty'), key=len)[1]
    assert calls[0]['prompt'].tolist() == [1, 2, 3]
    assert calls[1]['prompt'].tolist() == [1, 2, 3, CLOSE, 30, END, 40, 41, 3]            # teacher think [20] gone
    assert calls[2]['prompt'].tolist() == [1, 2, 3, CLOSE, 30, END, 40, 41, 3, CLOSE, 31, END, 42, 3]
    assert all(20 not in c['prompt'] and 21 not in c['prompt'] for c in calls)
    # an unusable call in the middle (a Neuralese marker in its action) still renders empty in the history after it
    rows[1] = {**rows[1], 'sample': np.array([21, 21, CLOSE, 100, END], np.int32)}
    rows[2] = {**rows[2], 'prompt': np.concatenate([rows[1]['prompt'], rows[1]['sample'], [42, 3]]).astype(np.int32)}
    runs = sorted(grpo.teacher_segments(rows, CLOSE, {100}, history='empty'), key=lambda r: r[0]['index'])
    last = [r for r in runs if r[0]['index'] == 2][0][0]
    assert last['prompt'].tolist() == [1, 2, 3, CLOSE, 30, END, 40, 41, 3, CLOSE, 100, END, 42, 3]


def test_trajectories_from_the_first_call_and_modes():
    rrl = sampler()
    segments = rrl.segments({'task_id': 't', 'round': 1, 'rows': trajectory()}, 'demos')
    assert sorted(len(seg['calls']) for seg in segments) == [1, 3]          # both invocations, whole
    assert {seg['mode'] for seg in segments} == {'whole'}
    whole = max(segments, key=lambda seg: len(seg['calls']))
    assert whole['calls'][0]['prompt'].tolist() == [1, 2, 3]                # from the first call
    rrl.args.reasoning_rl_calls = 2                                         # the cap is opt-in
    capped = rrl.segments({'task_id': 't', 'round': 1, 'rows': trajectory()}, 'demos')
    assert sorted(len(seg['calls']) for seg in capped) == [1, 2]
    rrl.args.reasoning_rl_calls, rrl.args.reasoning_rl_rewrite_share = 0, 1.0
    assert {seg['mode'] for seg in rrl.segments({'task_id': 't', 'round': 1, 'rows': trajectory()}, 'x')} \
        == {'rewrite'}
    rrl.args.reasoning_rl_rewrite_share = 0.0
    rrl.args.max_sequence_tokens = 6 + grpo.MIN_REASONING_ROOM              # a first call with no room to reason
    assert [len(seg['calls']) for seg in rrl.segments({'task_id': 't', 'round': 1, 'rows': trajectory()}, 'x')] \
        == [1]


def test_rewrite_keeps_teacher_tokens_out_of_the_target(monkeypatch):
    """Every position's teacher think block is rewritten from a random cut; an improved block trains only the
    student's tokens (the surviving teacher prefix is prompt), with the log-probabilities they were drawn with."""
    rrl = sampler()
    calls = sorted(grpo.teacher_segments(trajectory(), CLOSE, set()), key=len)[1]
    assert [c['think'].tolist() for c in calls] == [[20, CLOSE], [21, 21, CLOSE], [CLOSE]]
    monkeypatch.setattr(rrl, '_post', FakeSGLang([[50, CLOSE], [50, CLOSE], [50, CLOSE]]))
    episode = rrl.rewrite({'calls': calls, 'source': 'demos', 'task_id': 't', 'key': 'k', 'mode': 'rewrite'})
    assert episode['sd_source'] == 'rewrite' and len(episode['rows']) == 3
    for call, row in zip(calls, episode['rows']):
        assert row['sample'].tolist() == [50, CLOSE] and np.allclose(row['logp'], -0.25)
        prompt = row['prompt'].tolist()
        assert prompt[:len(call['prompt'])] == call['prompt'].tolist()
        teacher = prompt[len(call['prompt']):]                         # the surviving teacher prefix, as context
        assert teacher == call['think'].tolist()[:len(teacher)] and CLOSE not in teacher
    assert rrl.counts['rewrite_improved'] == 3 and rrl.counts['generated_tokens_rewrite'] == 6
    import exp0_sft
    assert len(exp0_sft.chains_of([episode])) == 3                       # positions are independent sequences


def test_rewrite_rejects_worse_and_unclosed_candidates(monkeypatch):
    rrl = sampler()
    calls = sorted(grpo.teacher_segments(trajectory(), CLOSE, set()), key=len)[1]
    monkeypatch.setattr(rrl, '_post', FakeSGLang([[60, END], [60, END], [60, END]]))
    assert rrl.rewrite({'calls': calls, 'source': 'demos', 'task_id': 't', 'key': 'k'}) is None
    assert rrl.counts['rewrite_unclosed'] == 3
    monkeypatch.setattr(rrl, '_post', FakeSGLang([[60, CLOSE]] * 3))     # closed but no better than the teacher's
    assert rrl.rewrite({'calls': calls[:2], 'source': 'demos', 'task_id': 't', 'key': 'k'}) is None


def test_future_ids_horizon_and_length():
    calls = sorted(grpo.teacher_segments(trajectory(), CLOSE, set()), key=len)[1]
    ids, spans = grpo.future_ids([1, 2, 3, 77, CLOSE], calls, 0, CLOSE, horizon=1)
    assert ids == [1, 2, 3, 77, CLOSE, 30, END, 40, 41, 3, CLOSE, 31, END] and len(spans) == 2
    ids, spans = grpo.future_ids([1, 2, 3, 77, CLOSE], calls, 0, CLOSE, max_len=12)
    assert len(spans) == 1 and ids == [1, 2, 3, 77, CLOSE, 30, END, 40, 41, 3]   # the next action did not fit


def test_groups_are_emitted_per_step_with_their_version(monkeypatch):
    rrl = sampler()
    calls = sorted(grpo.teacher_segments(trajectory(), CLOSE, set()), key=len)[1][:2]
    fake = FakeSGLang([[50, CLOSE], [61, CLOSE], [50, 51, CLOSE], [63, CLOSE]])
    monkeypatch.setattr(rrl, '_post', fake)
    emitted = []

    def emit(group):
        emitted.append((group['step'], group['version']))
        rrl.collector.version += 1                 # a learner step lands while the trajectory continues
    groups, _ = rrl.run({'calls': calls, 'source': 'demos', 'task_id': 't', 'key': 'k'}, emit=emit)
    assert emitted == [(0, 3), (1, 4)] and [g['version'] for g in groups] == [3, 4]


def test_future_ids_and_discount():
    calls = sorted(grpo.teacher_segments(trajectory(), CLOSE, set()), key=len)[1]
    ids, spans = grpo.future_ids([1, 2, 3, 77, CLOSE], calls, 0, CLOSE)
    assert ids == [1, 2, 3, 77, CLOSE, 30, END, 40, 41, 3, CLOSE, 31, END, 42, 3, CLOSE, 32, END]
    assert spans == [(5, 2), (11, 2), (16, 2)]
    assert [ids[p:p + n] for p, n in spans] == [[30, END], [31, END], [32, END]]
    assert grpo.discounted([-1., -2., None], 0.5) == -2.0


def test_rrl_advantages():
    a, reason = grpo.rrl_advantages([-1.0, -0.5, -9.0, -0.2], [True, True, False, True], 0.01, 'floor')
    assert reason is None
    assert abs(sum(a)) < 1e-9 and abs(sum(x * x for x in a) / 4 - 1) < 1e-9
    assert a[2] == a[0] == min(a)                              # floor: an unclosed sample ties with the worst closed one
    assert a[3] == max(a)
    assert grpo.rrl_advantages([-1., -2.], [False, False], 0.01) == (None, 'none_closed')
    assert grpo.rrl_advantages([-0.300, -0.301, -0.302], [True] * 3, 0.01) == (None, 'low_spread')
    assert grpo.rrl_advantages([-0.3, -5.], [True, False], 0.01, 'floor') == (None, 'low_spread')   # one closed


def test_rrl_advantages_drop_unclosed_samples_by_default():
    # cut by the token cap: no advantage, and the group is normalized over its closed samples alone
    a, reason = grpo.rrl_advantages([-1.0, -0.5, -9.0, -0.2], [True, True, False, True], 0.01)
    closed, _ = grpo.rrl_advantages([-1.0, -0.5, -0.2], [True] * 3, 0.01)
    assert reason is None and a[2] is None and [a[0], a[1], a[3]] == closed
    assert grpo.rrl_advantages([-0.3, -5.], [True, False], 0.01) == (None, 'low_spread')


def group(returns, closed, action_weight=None):
    samples = [{'ids': [10 + k] * (3 + k) + ([CLOSE] if c else [END]), 'logp': [-0.5] * (4 + k), 'closed': c,
                'reward': r, 'return': r, 'high': c and r >= -0.1} for k, (r, c) in enumerate(zip(returns, closed))]
    return {'id': 1, 'task_id': 't', 'key': 'k', 'source': 'demos', 'version': 0, 'min_std': 0.01, 'step': 0,
            'hard': False, 'r0': -0.5, 'context': np.array([1, 2, 3], np.int32),
            'action': np.array([0, 6, END], np.int32), 'action_weight': action_weight, 'samples': samples}


def test_rrl_episodes_rows_and_advantages():
    g = group([-1.0, -2.0, -0.05], [True, False, True], np.array([0, 1, 1], np.float32))
    items, stats = grpo.rrl_episodes([g], 2.0, 0.05, 'floor')
    assert len(items) == 3 and all(role == grpo.RRL_ROLE for _, role in items)
    expected, _ = grpo.rrl_advantages([-1.0, -2.0, -0.05], [True, False, True], 0.01, 'floor')
    rows = [episode['rows'] for episode, _ in items]
    assert [r[0]['advantage'] for r in rows] == [2.0 * x for x in expected]
    assert [len(r) for r in rows] == [2, 1, 2]                 # no action after the unclosed sample
    reasoning, action = rows[0]
    assert reasoning['prompt'].tolist() == [1, 2, 3] and reasoning['sample'].tolist() == [10, 10, 10, CLOSE]
    assert reasoning['logp'].dtype == np.float32 and len(reasoning['logp']) == len(reasoning['sample'])
    assert action['prompt'].tolist() == [1, 2, 3, 10, 10, 10, CLOSE] and 'logp' not in action
    assert action['sample'].tolist() == [0, 6, END] and action['weight'].tolist() == [0, 1, 1]
    assert action['advantage'] == 0.05
    assert stats['groups_trained'] == 1 and stats['unclosed'] == 1 and stats['high'] == 1
    import exp0_sft
    chains = exp0_sft.chains_of([items[0][0]])
    assert len(chains) == 1 and len(chains[0]['segs']) == 2   # think block and action: one sequence
    record = grpo.rrl_record(stats, collections.Counter(segments=1), 0)
    assert record['groups_trained'] == 1 and record['reward_mean'] == -0.525 and record['r0_mean'] == -0.5


def test_rrl_episodes_leave_unclosed_samples_untrained_by_default():
    g = group([-1.0, -2.0, -0.05], [True, False, True])
    items, stats = grpo.rrl_episodes([g], 2.0, 0.05)
    expected, _ = grpo.rrl_advantages([-1.0, -0.05], [True, True], 0.01)
    assert [episode['rows'][0]['advantage'] for episode, _ in items] == [2.0 * x for x in expected]
    assert all(episode['rows'][0]['sample'][-1] == CLOSE for episode, _ in items)
    assert stats['unclosed'] == 1 and stats['unclosed_dropped'] == 1
    assert grpo.rrl_record(stats, collections.Counter(), 0)['unclosed_dropped'] == 1


def test_rrl_episodes_without_spread_train_actions_only():
    items, stats = grpo.rrl_episodes([group([-0.3, -0.3], [True, True])], 1.0, 0.05)
    assert stats['groups_low_spread'] == 1 and len(items) == 2
    assert all(episode['rows'][0]['advantage'] == 0 and episode['rows'][1]['advantage'] == 0.05
               for episode, _ in items)
    items, _ = grpo.rrl_episodes([group([-0.3, -0.3], [True, True])], 1.0, 0.0)
    assert items == []


def test_learner_mixed_rows():
    import torch
    reasoning = {'logp': np.array([-0.2, -1.0], np.float32), 'advantage': 1.5}
    action = {'weight': np.array([0, 1], np.float32), 'advantage': 0.05}
    chain = {'role': grpo.RRL_ROLE, 'advantage': 1., 'segs': [(3, 2, reasoning), (5, 2, action)]}
    logp = torch.tensor([-0.1, -0.9, -3.0, -8.0])
    behaviour, weights = grpo.behaviour_logp(chain, logp)
    assert np.allclose(behaviour.numpy(), [-0.2, -1.0, -3.0, -8.0])   # actor law, then ratio-free teacher tokens
    assert weights.tolist() == [1, 1, 0, 1]
    advantage = grpo.token_advantage(chain, 'cpu')
    assert np.allclose(advantage.numpy(), [1.5, 1.5, 0.05, 0.05])
    chain['advantage'] = 0.                                   # --advantage-control zero reaches row advantages
    assert grpo.token_advantage(chain, 'cpu').abs().sum() == 0
    rho = torch.ones(4)
    weight, tail = grpo.policy_weight(rho, behaviour, torch.tensor([1.5, -1., 0.05, 0.05]), 2., -7.)
    assert tail.tolist() == [False, False, False, True] and weight.tolist() == [1, 1, 1, 0]
    plain = {'role': 'student', 'advantage': -0.5, 'segs': [(0, 2, {'logp': np.array([-0.1, -0.2], np.float32)})]}
    assert grpo.token_advantage(plain, 'cpu') == -0.5
    behaviour, weights = grpo.behaviour_logp(plain, torch.zeros(2))
    assert weights is None and np.allclose(behaviour.numpy(), [-0.1, -0.2])


def test_tail_floor_exempts_targets_the_student_never_sampled():
    """Review of ddd97a765: its exemption named a role ``rrl_action`` that no chain has. The teacher action rows of the
    reasoning RL term (``rrl == 'action'``) and the carried chains' ``teacher`` rows read the learner's own logp, not a
    sampling law: the floor no longer masks them (``--tail-floor-unsampled floor`` restores it); sampled rows keep it."""
    import torch
    reasoning = {'logp': np.array([-0.2, -8.0], np.float32), 'advantage': 1.5, 'rrl': 'reasoning'}
    action = {'advantage': 0.05, 'rrl': 'action'}
    chain = {'role': grpo.RRL_ROLE, 'advantage': 1., 'segs': [(3, 2, reasoning), (5, 2, action)]}
    behaviour, _ = grpo.behaviour_logp(chain, torch.tensor([-0.1, -0.9, -3.0, -8.0]))
    advantage = grpo.token_advantage(chain, 'cpu')
    args = SimpleNamespace(tail_floor_unsampled='exempt')
    weight, tail = grpo.policy_weight(torch.ones(4), behaviour, advantage, 2., -7., grpo.unsampled_tail_exempt(chain, args))
    assert tail.tolist() == [False, True, False, False] and weight.tolist() == [1, 0, 1, 1]
    args.tail_floor_unsampled = 'floor'
    assert grpo.unsampled_tail_exempt(chain, args) is None
    weight, tail = grpo.policy_weight(torch.ones(4), behaviour, advantage, 2., -7., grpo.unsampled_tail_exempt(chain, args))
    assert tail.tolist() == [False, True, False, True]
    carried = {'role': 'sd', 'advantage': 1., 'segs': [(0, 2, {'logp': np.array([-0.1, -0.2], np.float32)}),
                                                      (2, 1, {'teacher': True})]}
    assert grpo.unsampled_tail_exempt(carried, SimpleNamespace()).tolist() == [False, False, True]
    plain = {'role': 'student', 'advantage': 1., 'segs': [(0, 2, {'logp': np.array([-0.1, -9.], np.float32)})]}
    assert grpo.unsampled_tail_exempt(plain, SimpleNamespace()) is None     # every target sampled: the floor as before
    assert grpo.unsampled_tail_exempt({'role': 'sd_teacher', 'segs': plain['segs']}, SimpleNamespace()) is None


def sampler(**overrides):
    args = SimpleNamespace(reasoning_rl_sources='jolt,demos', seed=1, reasoning_rl_calls=0, reasoning_rl_horizon=4,
                           reasoning_rl_rewrite_share=0.0, reasoning_rl_rewrite_rounds=1,
                           reasoning_rl_rewrite_tokens=64, reasoning_rl_rewrite_accept='best',
                           reasoning_rl_rewrite_temperature=0.05, max_sequence_tokens=65536,
                           reasoning_rl_max_tokens=64, reasoning_rl_samples=4, reasoning_rl_max_samples=8,
                           reasoning_rl_wave=2, reasoning_rl_high_logp=-0.1, reasoning_rl_hard_logp=-1.0,
                           reasoning_rl_gamma=1.0, reasoning_rl_beam=1, top_p=0.95, reasoning_rl_groups=2,
                           reasoning_rl_workers=1, reasoning_rl_min_std=0.01, reasoning_rl_teacher_history='keep')
    for key, value in overrides.items():
        setattr(args, key, value)
    collector = SimpleNamespace(version=3, paused=threading.Event(), active_slots=1, stop=threading.Event(),
                                args=SimpleNamespace(pass_band=(0.1, 0.9)), base_rates={}, own={})
    return grpo.ReasoningRL(args, 'http://sglang', CLOSE, [END], [], [100], collector, None, log=lambda r: None)


class FakeSGLang:
    """Draws think blocks from a script per context length; an action's logprob is high (-0.01) when the think block
    before it holds the token 50, else -2.0; without reasoning (an empty think block) -1.5 at step 0, -0.05 later."""

    def __init__(self, draws):
        self.draws, self.bodies, self.lock = draws, [], threading.Lock()

    def __call__(self, body):
        with self.lock:
            self.bodies.append(body)
        params = body['sampling_params']
        if 'stop_token_ids' in params:
            with self.lock:
                ids = self.draws.pop(0)
            return {'output_ids': ids, 'meta_info': {'natlang_top_k': [record(-0.25) for _ in ids]}}
        ids, start = body['input_ids'], body.get('logprob_start_len')
        if not body.get('return_logprob'):
            return {'output_ids': [1], 'meta_info': {}}
        out = [[None, 0, None]]
        for position in range(start + 1, len(ids)):
            # the think block before this token: back to the last '3' (a generation prompt's end)
            back = position - 1
            while back > 0 and ids[back] != 3:
                back -= 1
            think = ids[back + 1:position]
            if CLOSE in think:
                think = think[:think.index(CLOSE)]
            value = -0.01 if 50 in think else (-1.5 if not think and back == 2 else -0.05 if not think else -2.0)
            out.append([value, ids[position], None])
        return {'meta_info': {'input_token_logprobs': out}}


def test_search_waves_carry_and_sd_chain(monkeypatch):
    rrl = sampler()
    calls = sorted(grpo.teacher_segments(trajectory(), CLOSE, set()), key=len)[1][:2]
    # step 0 (hard: no-reasoning logp -1.5): wave 1 misses, wave 2 hits; step 1 (ordinary): wave 1 hits at once
    fake = FakeSGLang([[60, CLOSE], [61, CLOSE], [50, CLOSE], [62, END], [50, 51, CLOSE], [63, CLOSE]])
    monkeypatch.setattr(rrl, '_post', fake)
    groups, carried = rrl.run({'calls': calls, 'source': 'demos', 'task_id': 't', 'key': 'k'})
    assert all(g['mode'] == 'whole' for g in groups)
    assert [g['step'] for g in groups] == [0, 1] and groups[0]['hard'] and not groups[1]['hard']
    assert groups[0]['r0'] == -1.5 and groups[1]['r0'] == -0.05
    assert len(groups[0]['samples']) == 4 and len(groups[1]['samples']) == 2
    first = {tuple(s['ids']): s for s in groups[0]['samples']}
    assert first[(50, CLOSE)]['high'] and first[(50, CLOSE)]['reward'] == -0.01
    assert first[(62, END)]['closed'] is False and not first[(62, END)]['high']
    # the return adds the later action after an empty think block: -0.05
    assert abs(first[(50, CLOSE)]['return'] - (-0.01 - 0.05)) < 1e-9
    # step 1 reasons on the carried history: prompt, the best think block, action 0 and its tool output
    assert groups[1]['context'].tolist() == [1, 2, 3, 50, CLOSE, 30, END, 40, 41, 3]
    draws = [b for b in fake.bodies if 'stop_token_ids' in b['sampling_params']]
    assert len({b['sampling_params']['sampling_seed'] for b in draws}) == len(draws) == 6
    assert draws[0]['sampling_params']['custom_params'] == {'natlang_top_k': 1}
    assert draws[0]['sampling_params']['top_p'] == 0.95 and CLOSE in draws[0]['sampling_params']['stop_token_ids']
    # the carried chain: one sequence of think blocks and teacher actions, high think blocks trained
    import exp0_sft
    assert carried['sd_source'] == 'search' and carried['role'] == 'sd'
    chains = exp0_sft.chains_of([carried])
    assert len(chains) == 1 and len(chains[0]['segs']) == 4
    assert chains[0]['ids'].tolist() == [1, 2, 3, 50, CLOSE, 30, END, 40, 41, 3, 50, 51, CLOSE, 31, END]
    assert all('weight' not in row for row in carried['rows'][::2])   # both carried think blocks scored high


def test_carried_chain_needs_a_high_step_and_masks_low_ones(monkeypatch):
    rrl = sampler(reasoning_rl_samples=2, reasoning_rl_max_samples=2)
    calls = sorted(grpo.teacher_segments(trajectory(), CLOSE, set()), key=len)[1][:2]
    monkeypatch.setattr(rrl, '_post', FakeSGLang([[60, CLOSE], [61, CLOSE], [62, CLOSE], [63, CLOSE]]))
    _, carried = rrl.run({'calls': calls, 'source': 'jolt', 'task_id': 't', 'key': 'k'})
    assert carried is None
    monkeypatch.setattr(rrl, '_post', FakeSGLang([[60, CLOSE], [61, CLOSE], [50, CLOSE], [63, CLOSE]]))
    _, carried = rrl.run({'calls': calls, 'source': 'jolt', 'task_id': 't', 'key': 'k'})
    assert carried['rows'][0]['weight'].sum() == 0 and 'weight' not in carried['rows'][2]


def test_sampler_queue_offer_take_and_idle():
    rrl = sampler()
    rrl.offer([{'role': 'teacher', 'quality': 1.0, 'task_id': 't', 'round': 1, 'rows': trajectory()},
               {'role': 'teacher', 'quality': 0.0, 'task_id': 't', 'round': 2, 'rows': trajectory()},
               {'role': 'student', 'quality': 1.0, 'task_id': 't', 'round': 3, 'rows': trajectory()}])
    assert len(rrl.jolt) == 2 and rrl.counts['jolt_offered'] == 2     # both invocations of the one success
    assert rrl._next()['source'] == 'jolt'
    rrl.ready = [{'version': v} for v in (0, 1, 2, 3)]
    rrl.sd_ready = ['chain']
    taken, dropped, chains = rrl.take(4, 2, 2)
    assert dropped == 2 and [g['version'] for g in taken] == [2, 3] and rrl.ready == [] and chains == ['chain']
    assert not rrl._idle()
    rrl.collector.paused.set()
    assert rrl._idle()                                        # a held measurement on the training actor
    rrl.collector.paused.clear()
    rrl.collector.active_slots = 0
    assert rrl._idle()                                        # the learner phase (actor released)


def test_rrl_task_ids_skip_solved_tasks():
    collector = SimpleNamespace(base_rates={'a': {'successes': 10, 'scored': 10}, 'b': {'successes': 1, 'scored': 10}},
                                own={'b': (10, 10), 'c': (0, 8)})
    assert grpo.rrl_task_ids(collector, (0.1, 0.9), ['a', 'b', 'c', 'd']) == ['c', 'd']


def test_teacher_demos_take_keeps_separate_counters(monkeypatch):
    import sft_sources
    demos = grpo.TeacherDemos.__new__(grpo.TeacherDemos)
    demos.lock, demos.rng, demos.renderer = threading.Lock(), __import__('random').Random(0), None
    demos.by_task = {'t': [{'task_id': 't', 'key': 'k', 'work': 'w', 'name': 'n', 'teacher': 'm', 'uses': 0}]}
    monkeypatch.setattr(sft_sources, 'replayed_turns', lambda work, name: [{}])
    monkeypatch.setattr(sft_sources, 'call_row', lambda *a, **k: ({'prompt': np.array([1]), 'sample': np.array([2])}, 0))
    stats = collections.Counter()
    out = demos.take(['t'], 1, stats, use='rrl_uses', prefix='rrl_demo')
    assert len(out) == 1 and stats['rrl_demo_candidates'] == 1
    item = demos.by_task['t'][0]
    assert item['rrl_uses'] == 1 and item['uses'] == 0       # the SD replay's counter is untouched
    assert math.isclose(out[0]['reward'], 1.0)


def test_a_request_failing_at_stop_is_abandoned_quietly(monkeypatch):
    """The actor goes with a stopping run and takes in-flight requests with it (grpo-v4-run2: 114 'request failed'
    events, all at stops): counted apart, not logged as actor failures, and the worker exits."""
    logged = []
    rrl = sampler()
    rrl.log = logged.append
    stop = threading.Event()
    segment = {'calls': trajectory(), 'source': 'jolt', 'task_id': 't', 'key': 1, 'mode': 'whole'}
    monkeypatch.setattr(rrl, '_next', lambda: segment)

    def run(seg, emit):
        rrl.collector.stop.set()
        raise ConnectionError('Remote end closed connection without response')
    monkeypatch.setattr(rrl, 'run', run)
    monkeypatch.setattr(rrl, '_idle', lambda: False)
    rrl._loop(stop)                                           # returns instead of retrying
    assert rrl.counts['abandoned_at_stop'] == 1 and rrl.counts['failures'] == 0 and logged == []


def test_reasoning_rl_seed_differs_per_resume():
    a, b = sampler(), sampler()
    assert a.rng.random() == b.rng.random()
    args = a.args
    c = grpo.ReasoningRL(args, 'http://sglang', CLOSE, [END], [], [100], a.collector, None, log=lambda r: None,
                         first_group=57)
    assert c.rng.random() != sampler().rng.random()


def test_ready_searches_survive_a_stop_without_unserved_policies(tmp_path):
    rrl = sampler()
    rrl.ready = [{'version': v, 'samples': [np.array([v], np.int32)]} for v in (3, 4, 5)]
    rrl.sd_ready = [{'task_id': 't', 'rows': []}]
    path = tmp_path / 'reasoning-rl-ready.pkl'
    rrl.save(path)
    resumed = sampler()
    assert resumed.restore(path, 4) == 3                      # versions 3 and 4 were served by the resumed state
    assert [g['version'] for g in resumed.ready] == [3, 4] and resumed.sd_ready == [{'task_id': 't', 'rows': []}]
    assert resumed.ready[0]['samples'][0].tolist() == [3]
    rrl.save(None)                                            # persistence off: nothing written
    assert sampler().restore(tmp_path / 'missing.pkl', 4) == 0


# ---------------------------------------------------------------- the twin objective (--reasoning-rl-objective twin-fkl)
# Strauss, Fowl & Razin (arXiv 2611.23397): a sequence-level reward on the think block has a future agreement bias;
# the default trains each position on the forward KL from the policy's privileged twin instead (myopic, token level).

def fkl_inputs(seed=0, n=5, vocab=7, dim=4):
    import torch
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, dim, generator=g, requires_grad=True)
    weight = torch.randn(vocab, dim, generator=g, requires_grad=True)
    targets = torch.randint(0, vocab, (n,), generator=g)
    teacher = torch.log_softmax(torch.randn(n, vocab, generator=g) * 2, -1)
    return h, weight, targets, teacher


def test_twin_forward_kl_is_the_exact_forward_kl_with_the_full_vocabulary():
    import torch
    h, weight, targets, teacher = fkl_inputs()
    logits = h @ weight.T
    target_logp = torch.log_softmax(logits, -1).gather(1, targets[:, None])[:, 0]
    ids = torch.arange(7).repeat(5, 1)
    got = grpo.twin_forward_kl(h, torch.arange(5), weight, targets, target_logp, ids, teacher)
    exact = (teacher.exp() * (teacher - torch.log_softmax(logits, -1))).sum(-1)
    assert torch.allclose(got, exact, atol=1e-5)
    # its logit gradient is p_policy - p_twin (the paper's eq. 4 for FKL), through h and the logits' normalizer
    z = logits.detach().requires_grad_(True)
    zl = torch.log_softmax(z, -1)
    assert grpo.twin_forward_kl(h, torch.arange(0), weight, targets[:0], target_logp[:0], ids[:0],
                                teacher[:0]).numel() == 0
    loss = (teacher.exp() * (teacher - zl)).sum()
    loss.backward()
    assert torch.allclose(z.grad, torch.softmax(logits.detach(), -1) - teacher.exp(), atol=1e-5)


def test_twin_forward_kl_top_k_matches_the_rest_as_one_bucket_and_vanishes_at_agreement():
    import torch
    h, weight, targets, _ = fkl_inputs(1)
    logits = (h @ weight.T).detach()
    policy = torch.log_softmax(logits, -1)
    target_logp = policy.gather(1, targets[:, None])[:, 0]
    top = policy.topk(3, -1)
    # the twin equals the policy: zero loss and zero gradient, though only 3 of 7 ids are listed
    hh = h.detach().clone().requires_grad_(True)
    tl = torch.log_softmax(hh @ weight.detach().T, -1).gather(1, targets[:, None])[:, 0]
    got = grpo.twin_forward_kl(hh, torch.arange(5), weight.detach(), targets, tl, top.indices, top.values)
    assert got.abs().max() < 1e-5
    got.sum().backward()
    assert hh.grad.abs().max() < 1e-4
    # a different twin: the KL of the coarse partition {3 ids, rest}, never more than the full KL
    teacher = torch.log_softmax(torch.randn(5, 7, generator=torch.Generator().manual_seed(9)) * 2, -1)
    t_top = teacher.topk(3, -1)
    coarse = grpo.twin_forward_kl(h.detach(), torch.arange(5), weight.detach(), targets, target_logp, t_top.indices,
                                  t_top.values)
    p_s = policy.exp().gather(1, t_top.indices)
    p_t = t_top.values.exp()
    rest_t, rest_s = 1 - p_t.sum(-1), 1 - p_s.sum(-1)
    expected = (p_t * (t_top.values - p_s.log())).sum(-1) + rest_t * (rest_t.log() - rest_s.log())
    assert torch.allclose(coarse, expected, atol=1e-5)
    full = (teacher.exp() * (teacher - policy)).sum(-1)
    assert (coarse <= full + 1e-5).all()
    # -inf pads count for nothing
    ids = torch.cat([t_top.indices, torch.zeros(5, 1, dtype=torch.long)], 1)
    lp = torch.cat([t_top.values, torch.full((5, 1), float('-inf'))], 1)
    padded = grpo.twin_forward_kl(h.detach(), torch.arange(5), weight.detach(), targets, target_logp, ids, lp)
    assert torch.allclose(padded, coarse, atol=1e-6)


def test_twin_distributions_always_list_the_close():
    meta = {'input_top_logprobs': [None, [[-0.1, 4, None], [-2.5, CLOSE, None]], [[-0.2, 5, None], [-3.0, 6, None]]],
            'input_token_ids_logprobs': [None, [[-2.5, CLOSE, None]], [[-4.0, CLOSE, None]]]}
    ids, logp, close = grpo.twin_distributions(meta, CLOSE, 2)
    assert ids.shape == (2, 3)
    assert ids[0].tolist() == [4, CLOSE, 0] and np.isneginf(logp[0, 2])           # listed once, then a pad
    assert ids[1].tolist() == [5, 6, CLOSE] and logp[1, 2] == np.float32(-4.0)     # appended from token_ids_logprob
    assert np.allclose(close, np.exp([-2.5, -4.0]))


def test_twin_context_inserts_a_turn_after_the_leading_system_turn():
    IM_END, NL = 8, 10
    ids, found = grpo.twin_context([1, 2, IM_END, NL, 5, 3], [90, 91, 30, 31], [IM_END, NL], IM_END, NL)
    assert found and ids == [1, 2, IM_END, NL, 90, 91, 30, 31, IM_END, NL, 5, 3]
    ids, found = grpo.twin_context([5, 3], [90], [IM_END, NL], IM_END, NL)
    assert not found and ids == [90, IM_END, NL, 5, 3]


class TwinSGLang(FakeSGLang):
    """FakeSGLang that also answers the twin's top-logprob requests: the twin puts p 0.5 on </think> everywhere."""

    def __call__(self, body):
        if body.get('top_logprobs_num'):
            with self.lock:
                self.bodies.append(body)
            n = len(body['input_ids']) - body['logprob_start_len'] - 1
            top = [[math.log(0.5), CLOSE, None], [math.log(0.3), 50, None]]
            return {'meta_info': {'input_top_logprobs': [None] + [top] * n,
                                  'input_token_ids_logprobs': [None] + [[[math.log(0.5), CLOSE, None]]] * n}}
        return super().__call__(body)


def twin_sampler(**overrides):
    rrl = sampler(**overrides)
    rrl.args.reasoning_rl_objective = 'twin-fkl'
    rrl.args.reasoning_rl_twin_top_k = 2
    rrl.args.reasoning_rl_rewrite_share = None
    return grpo.ReasoningRL(rrl.args, 'http://sglang', CLOSE, [END], [], [100], rrl.collector, None,
                            log=lambda r: None, twin={'header': [90, 91], 'footer': [END, 10], 'im_end': END, 'newline': 10,
                                                      'close': CLOSE})


def test_twin_walk_trains_every_sample_on_its_twin_and_carries_no_sd_chain(monkeypatch):
    rrl = twin_sampler()
    calls = sorted(grpo.teacher_segments(trajectory(), CLOSE, set()), key=len)[1][:2]
    # one wave of 2 per step, high or not (no search); the second sample of step 0 is cut by the cap
    fake = TwinSGLang([[60, CLOSE], [61, 62], [50, CLOSE], [63, CLOSE]])
    monkeypatch.setattr(rrl, '_post', fake)
    groups, carried = rrl.run({'calls': calls, 'source': 'demos', 'task_id': 't', 'key': 'k'})
    assert carried is None                                          # the best-return chain is not distilled
    assert [len(g['samples']) for g in groups] == [2, 2] and all(g['objective'] == 'twin-fkl' for g in groups)
    twin_bodies = [b for b in fake.bodies if b.get('top_logprobs_num')]
    assert len(twin_bodies) == 4 and all(b['token_ids_logprob'] == [CLOSE] for b in twin_bodies)
    # the twin's prompt: the guidance turn with the teacher action (its end of turn replaced by the footer) first
    first = twin_bodies[0]['input_ids']
    assert first[:6] == [90, 91, 30, END, 10, 1] and first[-2:] in ([60, CLOSE], [61, 62])
    for g in groups:
        for s in g['samples']:
            assert s['twin_ids'].shape == (len(s['ids']), 3) and np.isclose(s['twin_close_mean'], 0.5)
    for g in groups:
        g.update(id=1, min_std=0.01)
    items, stats = grpo.rrl_episodes(groups, 0.5, 0.05)
    assert stats['groups_twin'] == 2 and len(items) == 4               # the unclosed sample trains too
    for episode, role in items:
        reasoning = episode['rows'][0]
        assert reasoning['advantage'] == 0. and reasoning['twin_weight'] == 0.5
        assert reasoning['twin_logp'].shape[0] == len(reasoning['sample'])
    unclosed = [e for e, _ in items if e['rows'][0]['sample'].tolist() == [61, 62]][0]
    assert len(unclosed['rows']) == 1                                # no teacher action after a cut block
    record = grpo.rrl_record(stats, {}, 0)
    assert record['groups_twin'] == 2 and np.isclose(record['twin_close_mean'], 0.5)


def test_objective_defaults():
    seq = SimpleNamespace(reasoning_rl_objective='sequence', reasoning_rl_rewrite_share=None, reasoning_rl_carry_sd=None)
    twin = SimpleNamespace(reasoning_rl_objective='twin-fkl', reasoning_rl_rewrite_share=None, reasoning_rl_carry_sd=None)
    assert grpo.rrl_rewrite_share(seq) == 0.3 and grpo.rrl_carry_sd(seq)
    assert grpo.rrl_rewrite_share(twin) == 0. and not grpo.rrl_carry_sd(twin)
    twin.reasoning_rl_rewrite_share, twin.reasoning_rl_carry_sd = 0.2, 'on'
    assert grpo.rrl_rewrite_share(twin) == 0.2 and grpo.rrl_carry_sd(twin)
    assert grpo.rrl_objective(SimpleNamespace()) == 'sequence'      # an args object from before the flag


def test_twin_rows_align_with_chain_targets():
    reasoning = {'twin_ids': np.array([[1, CLOSE], [2, 3]], np.int32), 'twin_logp': np.array([[-0.1, -2.], [-0.5, -1.]],
                 np.float32), 'twin_weight': 0.5}
    chain = {'segs': [(3, 2, reasoning), (5, 3, {'sample': np.array([1, 2, 3])})]}
    ids, logp, scale = grpo.twin_rows(chain, 'cpu')
    assert ids.shape == (5, 2) and scale.tolist() == [0.5, 0.5, 0, 0, 0]
    assert np.isneginf(logp[2:].numpy()).all()
    assert grpo.twin_rows({'segs': [(0, 2, {})]}, 'cpu') is None
