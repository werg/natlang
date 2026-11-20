"""CPU checks of grpo.py's dynamic sampling (first waves, posterior early stop), the construct-weighted exploration walk and
the teacher demonstrations of the self-distillation replay."""
import collections
import json
import math
import random
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402
import sft_sources  # noqa: E402
from test_grpo_task_supply import parsed  # noqa: E402


# ---- dynamic sampling ------------------------------------------------------------------------------------------------
def test_all_equal_probability_is_the_beta_binomial_end_mass():
    # uniform prior: E[p^m] = 1 / (m + 1)
    for m in range(5):
        assert grpo.all_equal_probability(1., 1., m, True) == pytest.approx(1. / (m + 1))
        assert grpo.all_equal_probability(1., 1., m, False) == pytest.approx(1. / (m + 1))
    # Beta(2, 1): E[p^m] = 2 / (m + 2)
    assert grpo.all_equal_probability(2., 1., 3, True) == pytest.approx(2. / 5)
    # uniform prior, 3 agreeing: the other 5 agree with probability 4/9
    assert grpo.spread_after(1., 1., 8, 3, True) == pytest.approx(1. - (1. / 9) / (1. / 4))


def test_wave_plan_splits_only_tasks_likely_to_agree():
    k, saved, lost = grpo.wave_plan(30., 0.3, 8, 0.1)        # nearly always passes
    assert k is not None and 2 <= k <= 6 and saved >= 1. and lost <= 0.1 * saved
    assert grpo.spread_after(30., 0.3, 8, k, True) <= 0.1
    assert grpo.wave_plan(2., 2., 8, 0.1)[0] is None          # mixed task: all eight run at once
    assert grpo.wave_plan(0.2, 30., 8, 0.1)[0] is not None    # nearly always fails
    assert grpo.wave_plan(30., 0.3, 8, 0.)[0] is None          # 0: off


def test_pass_rate_prior_fits_bimodal_pools_tight_and_mixed_pools_loose():
    rng = random.Random(0)
    bimodal = {f't{i}': ((8 if rng.random() < .5 else 0), 8) for i in range(200)}
    mixed = {f't{i}': (sum(rng.random() < .5 for _ in range(8)), 8) for i in range(200)}
    tight = grpo.PassRatePrior(bimodal, lambda t: 'f')
    loose = grpo.PassRatePrior(mixed, lambda t: 'f')
    assert tight.concentration < 0.5 < 10 < loose.concentration
    # a task's own outcomes on top of its family's leave-one-out mean
    families = {'easy': [(8, 8)] * 30, 'hard': [(0, 8)] * 30}
    rates = {f'{f}{i}': r for f, rs in families.items() for i, r in enumerate(rs)}
    rates['mixed'] = (4, 8)
    prior = grpo.PassRatePrior(rates, lambda t: 'easy' if t.startswith('easy') else 'hard')
    a, b = prior.beta('easy-new')
    assert a / (a + b) > 0.9
    a, b = prior.beta('hard3')
    assert a / (a + b) < 0.1 and b >= 8           # its own 0/8 included
    assert grpo.PassRatePrior({}, lambda t: None).beta('x') == pytest.approx((1., 1.))   # default c = 2, mean 1/2


def wave_collector(tmp_path, outcomes, size=8, wave=3, rates=None):
    """A collector with one split group (first ``wave`` entries pending, the rest held back); the first-wave episodes'
    (binary, graded) outcomes are ``outcomes``."""
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = SimpleNamespace(actor='http://actor', dynamic_sampling_max_spread=0.1, reward='graded')
    c.work, c.runtime, c.env = tmp_path, tmp_path / 'ts-host', {}
    c.lock, c.stop, c.paused = threading.Lock(), threading.Event(), threading.Event()
    c.active_slots, c.version, c.outstanding, c.running = 1, 0, 1, 0
    c.dynamic = collections.Counter()
    c.by_id, c.own = {'t': {'id': 't', 'family': 'easy'}}, {}
    c.base_rates = rates or {f'e{i}': {'successes': 8, 'scored': 8, 'family': 'easy'} for i in range(30)}
    c.prior, c.prior_built, c.prior_stale = None, 0., True
    d = tmp_path / 'g0'
    (d / 'workers').mkdir(parents=True)
    entries = [{'key': f'k{i}', 'index': 0, 'output': str(d / f'out{i}.jsonl')} for i in range(size)]
    (d / 'queue.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    (d / 'packet.jsonl').write_text(json.dumps({'id': 't', 'semantics': {}}) + '\n')
    assert c._wave('t', size) is not None          # the prior built from the rates splits this task
    c.groups = {0: {'dir': d, 'size': size, 'done': 0, 'versions': [], 'task_id': 't', 'created': 0, 'wave': wave,
                    'held_back': entries[wave:]}}
    c.pending = collections.deque((0, e) for e in entries[:wave])
    c.scored = []
    c._score = c.scored.append
    by_key = {f'k{i}': o for i, o in enumerate(outcomes)}
    return c, by_key


def run_wave(c, monkeypatch, by_key, extra=0):
    from natlang_neuralese.rl import task_rollouts
    monkeypatch.setattr(task_rollouts, 'entry_outcome', lambda entry, task, finish: ('x', *by_key[entry['key']]))
    started = []

    def popen(*_args, **_kwargs):
        started.append(1)
        return SimpleNamespace(returncode=0, pid=0, poll=lambda: 0, wait=lambda: 0)

    def take():
        if c.pending and len(started) < len(by_key) + extra:
            return c.pending.popleft()
        c.stop.set()
        return None
    monkeypatch.setattr(grpo.subprocess, 'Popen', popen)
    monkeypatch.setattr(c, '_take', take)
    c.slot()
    return len(started)


def test_an_agreeing_first_wave_cancels_the_rest_and_scores_what_ran(tmp_path, monkeypatch):
    c, by_key = wave_collector(tmp_path, [(1., 1.)] * 3)
    assert run_wave(c, monkeypatch, by_key) == 3
    info = c.groups[0]
    assert c.scored == [0] and info['size'] == 3 and info['held_back'] == []
    kept = [json.loads(l)['key'] for l in (info['dir'] / 'queue.jsonl').read_text().splitlines()]
    cancelled = [json.loads(l)['key'] for l in (info['dir'] / 'cancelled.jsonl').read_text().splitlines()]
    assert kept == ['k0', 'k1', 'k2'] and cancelled == [f'k{i}' for i in range(3, 8)]
    assert c.dynamic == {'groups_split': 1, 'groups_stopped': 1, 'samples_saved': 5, 'stopped_all_pass': 1}


def test_a_split_first_wave_or_graded_spread_runs_the_rest_first(tmp_path, monkeypatch):
    for outcomes in ([(1., 1.), (0., 0.), (1., 1.)],          # binary spread
                     [(0., 0.), (0., 0.25), (0., 0.)],        # graded spread among failures (the advantage reward)
                     [(1., 1.), (None, None), (1., 1.)]):     # an unscored sample: no verdict, run the rest
        c, by_key = wave_collector(tmp_path / str(len(list(tmp_path.iterdir()))), outcomes)
        c.pending.append((1, {'key': 'other'}))               # another group's entry waits behind the wave
        run_wave(c, monkeypatch, by_key)
        assert c.scored == [] and c.groups[0]['size'] == 8 and c.groups[0]['held_back'] == []
        assert [e['key'] for _, e in list(c.pending)[:5]] == [f'k{i}' for i in range(3, 8)]   # ahead of other groups
        assert c.dynamic['groups_released'] == 1
    # the released samples complete the group as usual
    c, by_key = wave_collector(tmp_path / 'full', [(1., 1.), (0., 0.), (1., 1.)] + [(1., 1.)] * 5)
    assert run_wave(c, monkeypatch, by_key) == 8 and c.scored == [0]


def test_a_mixed_task_is_never_split():
    rates = {f'm{i}': {'successes': 4, 'scored': 8, 'family': 'easy'} for i in range(30)}
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = SimpleNamespace(dynamic_sampling_max_spread=0.1)
    c.dynamic, c.by_id, c.own, c.base_rates = collections.Counter(), {'t': {'family': 'easy'}}, {}, rates
    c.prior, c.prior_built, c.prior_stale = None, 0., True
    assert c._wave('t', 8) is None and c.dynamic['groups_split'] == 0
    c.args.dynamic_sampling_max_spread = 0.
    assert c._wave('t', 8) is None


def test_dynamic_sampling_is_on_by_default_with_an_opt_out(monkeypatch):
    args = parsed(monkeypatch)
    assert args.dynamic_sampling_max_spread == 0.1
    assert parsed(monkeypatch, '--dynamic-sampling-max-spread', '0').dynamic_sampling_max_spread == 0.


# ---- construct-weighted exploration ----------------------------------------------------------------------------------
def walk_rows():
    rows = [{'id': f'w{i}', 'family': f'wf{i % 5}'} for i in range(60)] + \
           [{'id': f'h{i}', 'family': f'hf{i % 3}'} for i in range(20)] + \
           [{'id': f'p{i}', 'family': 'pf'} for i in range(20)]
    random.Random(0).shuffle(rows)
    return rows


def features(row):
    kind = row['id'][0]
    return (('neuralese_result', 'multi_function') if kind == 'h' else ('multi_function',) if kind == 'p' else (),
            kind == 'w')


def test_explore_walk_caps_wrappers_and_prefers_held_constructs(monkeypatch):
    rows = walk_rows()
    monkeypatch.setattr(grpo, 'task_features', lambda tasks: {r['id']: features(r) for r in tasks})
    walk = grpo.ExploreWalk(rows, wrapper_share=0.15, held_weight=3.)
    rng, cursor, taken = random.Random(1), 0, []
    for _ in range(40):
        row, cursor = walk.next(cursor, set(), rng)
        taken.append(row['id'][0])
    assert taken.count('w') <= 0.15 * len(taken) + 1e-9
    assert taken.count('h') > taken.count('p')            # multi_function alone is not a preferred construct
    report = walk.pool_report()
    assert report['wrappers'] == 60 and report['declaring_held'] == 20
    # off: the plain walk in pool order, measured rows skipped
    plain = grpo.ExploreWalk(rows)
    assert not plain.weighted and plain.features == {}
    row, cursor = plain.next(0, {rows[0]['id']}, rng)
    assert row is rows[1] and cursor == 2


def test_explore_walk_takes_anything_once_the_pool_is_measured(monkeypatch):
    rows = walk_rows()
    monkeypatch.setattr(grpo, 'task_features', lambda tasks: {r['id']: features(r) for r in tasks})
    walk = grpo.ExploreWalk(rows, wrapper_share=0., held_weight=1e9)
    measured = {r['id'] for r in rows}
    row, cursor = walk.next(0, measured, random.Random(0))
    assert cursor == 2 * len(rows) + 1                       # the original walk's fallback after two passes
    only_wrappers = [{'id': f'w{i}'} for i in range(5)]
    walk = grpo.ExploreWalk(only_wrappers, wrapper_share=0., held_weight=2.)
    assert walk.next(0, set(), random.Random(0))[0]['id'].startswith('w')   # never stuck


def test_construct_weighting_defaults(monkeypatch):
    args = parsed(monkeypatch)
    assert args.explore_wrapper_share == 0.15 and args.explore_held_weight == 3.
    off = parsed(monkeypatch, '--explore-wrapper-share', '1', '--explore-held-weight', '1')
    assert not grpo.ExploreWalk([{'id': 'x'}], off.explore_wrapper_share, off.explore_held_weight).weighted


# ---- teacher demonstrations ------------------------------------------------------------------------------------------
class FakeRenderer:
    identity = {'fake': True}

    def __init__(self, tokenizer):
        pass

    class tokenizer:   # the target's first token closes an empty think block
        @staticmethod
        def convert_tokens_to_ids(token):
            return {'</think>': 7}.get(token, -1)

        @staticmethod
        def decode(ids):
            return {7: '</think>'}.get(ids[0], 'x')

    def render(self, messages, tools, target):
        return [1, 2, 3], [7, 8, 9]


def held_manifest(tmp_path, row):
    cases = tmp_path / 'held-cases.jsonl'
    cases.write_text(json.dumps(row) + '\n')
    manifest = {'tasks': [{'id': row['id']}], 'excluded_split_groups': [],
                'cases': {'path': cases.name, 'sha256': sft_sources.sha256_file(cases)}}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    return str(tmp_path / 'manifest.json')


def teacher_work(tmp_path, episodes):
    """An sft_sources WORK: extracted teacher successes, their replay verdicts and replayed turns."""
    work = tmp_path / 'work'
    (work / 'replay').mkdir(parents=True)
    writer = sft_sources.ZstdWriter(work / 'episodes-teacher.jsonl.zst')
    with (work / 'replay.jsonl').open('w') as replays:
        for key, task_id, model, exact in episodes:
            task = {'id': task_id, 'source_groups': [f'sg-{task_id}']}
            writer.write({'key': key, 'source': 'teacher', 'task_id': task_id, 'task': task, 'sample': 0,
                          'teacher': {'model_id': model}, 'turns': []})
            name = sft_sources.safe_name(key)
            replays.write(json.dumps({'key': key, 'name': name, 'source': 'teacher', 'task_id': task_id,
                                      'outcome': 'task_success', 'exact': exact}) + '\n')
            (work / 'replay' / name).mkdir()
            turn = {'messages': [{'role': 'user', 'content': 'q'}], 'tools': [], 'invocation_id': 'root',
                    'target': {'role': 'assistant', 'content': 'a'}}
            (work / 'replay' / name / 'rollouts.jsonl').write_text(json.dumps({'turns': [turn, turn]}) + '\n')
    writer.close()
    return work


def test_teacher_demonstrations_are_verified_held_closed_capped_and_least_replayed(tmp_path, monkeypatch):
    monkeypatch.setattr(sft_sources, 'TemplateRenderer', FakeRenderer)
    episodes = [(f'teacher:p:a{i}:0', 'a', f'm{i % 2}', True) for i in range(6)] + \
               [('teacher:p:b:0', 'b', 'm0', False),                  # replay not exact: never admitted
                ('teacher:p:h:0', 'h', 'm0', True),                   # a held task
                ('teacher:p:x:0', 'x', 'm0', True)]                   # not an eligible pool task
    work = teacher_work(tmp_path, episodes)
    manifest = held_manifest(tmp_path, {'id': 'h'})
    by_id = {t: {'id': t} for t in ('a', 'b', 'h')}
    demos = grpo.TeacherDemos([str(work), str(tmp_path / 'missing')], by_id, manifest, None, max_per_task=4, seed=0,
                              log=lambda r: None)
    assert set(demos.by_task) == {'a'} and len(demos.by_task['a']) == 4
    assert collections.Counter(item['teacher'] for item in demos.by_task['a']) == {'m0': 2, 'm1': 2}   # turns
    assert demos.funnel['teacher']['replay_inexact'] == 1 and demos.funnel['teacher']['held_excluded'] == 1
    stats = collections.Counter()
    first = demos.take(['a', 'b'], 8, stats)
    assert len(first) == 1 and stats['sd_teacher_candidates'] == 4          # one per task a step
    episode = first[0]
    assert episode['role'] == 'sd_teacher' and len(episode['rows']) == 2
    row = episode['rows'][0]
    assert row['prompt'].tolist() == [1, 2, 3] and row['sample'].tolist() == [7, 8, 9] and row['teacher']
    assert row['weight'].tolist() == [0, 1, 1]                             # the empty think block is context
    seen = {first[0]['round']}
    for _ in range(3):
        seen.add(demos.take(['a'], 8, stats)[0]['round'])
    assert len(seen) == 4                                                   # least replayed first
    assert demos.take([], 8, stats) == []


def test_teacher_demonstration_uses_persist_across_a_resume(tmp_path, monkeypatch):
    """A resume continues least-used-first from the saved counters instead of replaying the same demonstrations."""
    monkeypatch.setattr(sft_sources, 'TemplateRenderer', FakeRenderer)
    work = teacher_work(tmp_path, [(f'teacher:p:a{i}:0', 'a', 'm0', True) for i in range(4)])
    manifest = held_manifest(tmp_path, {'id': 'h'})
    by_id = {'a': {'id': 'a'}, 'h': {'id': 'h'}}
    uses = tmp_path / 'run' / 'teacher-demo-uses.json'
    uses.parent.mkdir()

    def build(take_seed):
        return grpo.TeacherDemos([str(work)], by_id, manifest, None, max_per_task=4, seed=0, log=lambda r: None,
                                 uses_path=uses, take_seed=take_seed)
    stats = collections.Counter()
    demos = build('0:0')
    first = {demos.take(['a'], 1, stats)[0]['round'] for _ in range(2)}
    demos.take(['a'], 1, stats, use='rrl_uses', prefix='rrl_demo')
    demos.save()
    resumed = build('0:9')
    assert sum(item['uses'] for item in resumed.by_task['a']) == 2
    assert sum(item.get('rrl_uses', 0) for item in resumed.by_task['a']) == 1
    after = {resumed.take(['a'], 1, stats)[0]['round'] for _ in range(2)}
    assert first.isdisjoint(after) and len(first | after) == 4              # the unused two come next


def test_a_teacher_demonstration_trains_ratio_free_with_its_row_weights():
    import torch
    import exp0_sft
    rows = [{'prompt': np.array([1, 2], np.int32), 'sample': np.array([3, 4, 5], np.int32), 'invocation': 'r',
             'index': 0, 'weight': np.array([0., 1., 1.], np.float32), 'teacher': True}]
    chain = exp0_sft.chains_of([{'task_id': 't', 'round': 'k', 'rows': rows}])[0]
    chain['role'] = 'sd_teacher'
    logp = torch.tensor([-9., -0.5, -2.])
    actor, weights = grpo.behaviour_logp(chain, logp)
    assert torch.equal(actor, logp) and weights.tolist() == [0., 1., 1.]
    rho = (logp - actor).exp()
    weight, tail = grpo.policy_weight(rho, actor, 0.05, 2., -7.)
    assert (weight * weights).tolist() == [0., 1., 1.] and tail.tolist() == [True, False, False]
    own = dict(chain, role='sd')
    for _, _, row in own['segs']:
        row['logp'] = np.array([-1., -1., -1.], np.float32)
    actor, weights = grpo.behaviour_logp(own, logp)
    assert actor.tolist() == [-1., -1., -1.] and weights is None


def test_teacher_tasks_are_those_in_band_or_all_fail():
    c = SimpleNamespace(base_rates={'a': {'successes': 0, 'scored': 3}, 'b': {'successes': 3, 'scored': 3},
                                    'c': {'successes': 1, 'scored': 2}, 'd': {'successes': 0, 'scored': 0}},
                        own={'b': (4, 8), 'e': (8, 8), 'f': (0, 8)})
    assert grpo.sd_teacher_tasks(c, (0.1, 0.9)) == ['a', 'b', 'c', 'f']


def test_teacher_demonstrations_default_on_with_an_opt_out(monkeypatch):
    args = parsed(monkeypatch)
    assert args.sd_teacher_weight == 0.05 and args.sd_teacher_episodes == 8 and args.sd_teacher_source is None
    assert parsed(monkeypatch, '--sd-teacher-weight', '0').sd_teacher_weight == 0.
    stats = collections.Counter(sd_teacher_episodes=2, sd_own_episodes=1)
    record = grpo.restore_telemetry(stats, SimpleNamespace(sd_weight=0., sd_teacher_weight=0.05))
    assert record['sd']['teacher_episodes'] == 2 and record['sd']['own_episodes'] == 1
    assert 'sd' not in grpo.restore_telemetry(stats, SimpleNamespace(sd_weight=0., sd_teacher_weight=0.))
    assert math.isfinite(args.sd_teacher_weight)


def test_prefix_retained_memory_is_bounded_by_default(monkeypatch):
    assert parsed(monkeypatch).prefix_retained_gb == 8.
    assert parsed(monkeypatch, '--prefix-retained-gb', '0').prefix_retained_gb == 0.


def test_sd_buffer_survives_a_resume_without_rolled_back_steps(tmp_path):
    path = tmp_path / 'sd-buffer.pkl'
    rows = [{'prompt': np.array([1, 2], np.int32), 'sample': np.array([3], np.int32)}]
    buffer = [{'episode': {'task_id': 't', 'rows': rows, 'decisions': []}, 'version': v, 'uses': v % 2} for v in (3, 4, 5)]
    grpo.sd_buffer_save(path, buffer)
    restored = grpo.sd_buffer_load(path, 5)                  # the state saved after step 5 (lowered at 0..4)
    assert [item['version'] for item in restored] == [3, 4] and restored[1]['uses'] == 0
    assert restored[0]['episode']['rows'][0]['prompt'].tolist() == [1, 2]
    assert grpo.sd_buffer_load(None, 5) == [] and grpo.sd_buffer_load(tmp_path / 'missing.pkl', 5) == []
