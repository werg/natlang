"""Expert iteration (scripts/neuralese_data/expert_iteration.py) and its pool rules in sft_sources build."""
import argparse
import collections
import os
import json
import random
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO)]

import expert_iteration as ei  # noqa: E402
import sft_sources  # noqa: E402


def write_group(root, name, task_id, rewards, role='student'):
    directory = root / 'groups' / name
    directory.mkdir(parents=True)
    (directory / 'rewards.jsonl').write_text(''.join(
        json.dumps({'task_id': task_id, 'reward': r, 'jolt_role': role}) + '\n' for r in rewards))
    return directory


def test_group_outcomes_count_student_samples_only(tmp_path):
    a = write_group(tmp_path, 'g000000', 't1', [1.0, 0.0, None, 1.0])
    b = write_group(tmp_path, 'g000001', 't1', [1.0], role='teacher')
    c = write_group(tmp_path, 'g000002', 't2', [0.5, 0.0])
    assert ei.group_outcomes([a, b, c, tmp_path / 'missing']) == {'t1': [2, 3], 't2': [0, 2]}


@pytest.mark.parametrize('s,n,expected', [(4, 4, 'solved'), (3, 4, 'solved'), (1, 4, 'sometimes'), (0, 4, 'never'),
                                          (1, 1, 'unmeasured'), (0, 0, 'unmeasured')])
def test_tier_by_pass_rate(s, n, expected):
    assert ei.tier(s, n) == expected


def test_selection_order_puts_band_then_failures_then_unmeasured_and_drops_solved():
    rows = [{'id': f'{kind}{i}', 'family': f'f{i % 2}'} for kind in 'bfus' for i in range(4)]
    rates = {**{f'b{i}': (1, 4) for i in range(4)}, **{f'f{i}': (0, 4) for i in range(4)},
             **{f's{i}': (4, 4) for i in range(4)}}
    order, counts = ei.selection_order(rows, rates, random.Random(0))
    kinds = [r['id'][0] for r in order]
    assert kinds == ['b'] * 4 + ['f'] * 4 + ['u'] * 4
    assert counts == {'band': 4, 'failed': 4, 'unmeasured': 4}
    families = [r['family'] for r in order[:4]]
    assert families[0] != families[1]                         # families take turns within a tier
    with_solved, _ = ei.selection_order(rows, rates, random.Random(0), include_solved=True)
    assert [r['id'][0] for r in with_solved][-4:] == ['s'] * 4


def test_frontier_drop_keeps_the_cheapest_verified_source_per_task():
    tiers = {'solved': 'solved', 'some': 'sometimes', 'never': 'never'}
    own = {'some'}
    assert sft_sources.frontier_drop('solved', tiers, own) == 'frontier_task_solved'
    assert sft_sources.frontier_drop('some', tiers, own) == 'frontier_own_success'
    assert sft_sources.frontier_drop('never', tiers, own) is None          # teacher (filled first) trains here
    assert sft_sources.frontier_drop('unmeasured-task', tiers, own) is None


def test_think_treatment_takes_the_most_specific_entry_and_filled_renders_as_context(tmp_path):
    spec = {'schema': 'natlang.teacher-think-treatment/1', 'default': 'context',
            'models': {'haiku': 'train', 'luna': 'filled'}, 'sources': {'teacher:x/pool': 'context'},
            'episodes': {'teacher:x/pool:g1:0': 'train'}}
    path = tmp_path / 'treatment.json'
    path.write_text(json.dumps(spec))
    t = sft_sources.ThinkTreatment.load(str(path), 'context')
    assert t.of({'source': 'teacher', 'teacher': 'haiku', 'run': 'teacher:y/pool', 'key': 'k'}) == 'train'
    assert t.of({'source': 'teacher', 'teacher': 'haiku', 'run': 'teacher:x/pool', 'key': 'k'}) == 'context'
    assert t.of({'source': 'teacher', 'teacher': 'luna', 'run': 'teacher:x/pool', 'key': 'teacher:x/pool:g1:0'}) == 'train'
    assert t.render({'source': 'teacher', 'teacher': 'luna', 'run': 'r', 'key': 'k'}) == 'context'
    assert t.of({'source': 'rl-own', 'teacher': 'haiku'}) is None
    assert t.render({'source': 'rl-own'}) == 'context'
    assert sft_sources.ThinkTreatment.load('none', 'train').of({'source': 'teacher', 'teacher': 'haiku'}) == 'train'
    with pytest.raises(ValueError):
        sft_sources.ThinkTreatment({'models': {'m': 'sometimes'}})


def test_load_frontier_refuses_other_files(tmp_path):
    path = tmp_path / 'f.json'
    path.write_text(json.dumps({'schema': 'x', 'tasks': {}}))
    with pytest.raises(ValueError):
        sft_sources.load_frontier(path)
    path.write_text(json.dumps({'schema': ei.FRONTIER_SCHEMA, 'policy': 'abc', 'tasks': {'t': {'tier': 'never'}}}))
    assert sft_sources.load_frontier(path) == ('abc', {'t': {'tier': 'never'}})
    assert sft_sources.load_frontier(None) == (None, {})


def test_episode_sizes_cache_per_candidate_treatment(tmp_path):
    class Renderer:
        identity = {'id': 1}
    sizes = sft_sources.EpisodeSizes(tmp_path / 'sizes.jsonl', Renderer(), lambda c: c['think'], 'context')
    assert sizes.policy('train') != sizes.policy('context')
    fixed = sft_sources.EpisodeSizes(tmp_path / 'sizes.jsonl', Renderer(), 'context', 'context')
    assert fixed.think_of({'think': 'train'}) == 'context' and fixed.policy('context') == sizes.policy('context')


class _Base:
    """The parts of grpo.Collector the frontier subclass relies on."""

    def __init__(self, tasks, own):
        self.by_id = {t['id']: t for t in tasks}
        self.own, self.pending, self.lock, self.outstanding = own, collections.deque(), threading.Lock(), 0
        self.paused = threading.Event()
        self.args = type('A', (), {'max_outstanding': 2})()
        self.created = []

    def _learnable(self):
        pass

    def _new_group(self):
        row = self._pick()
        self.created.append(row['id'])
        self.outstanding += 1
        self.pending.extend((row['id'], i) for i in range(2))


def test_frontier_collector_walks_the_order_once_skips_scored_tasks_and_stops_at_the_deadline():
    tasks = [{'id': t} for t in ('a', 'b', 'c', 'd')]
    collector = ei.frontier_collector(_Base)(tasks, own={'b': (1, 4)})
    collector.setup(['c', 'b', 'x', 'a', 'd'], time.time() + 3600)
    assert list(collector.order) == ['c', 'a', 'd']            # scored and unknown tasks out, order kept
    taken = [collector._take() for _ in range(4)]
    assert [t[0] for t in taken] == ['c', 'c', 'a', 'a']
    assert collector._take() is None                           # max_outstanding 2: waits for the learner-free drain
    collector.outstanding = 0
    collector.deadline = time.time() - 1
    assert collector._take() is None and collector.finished()  # past the deadline: no new group, done when drained
    assert collector.created == ['c', 'a']


def test_cycle_fill_command_parses_as_filled_reasoning_generate(tmp_path, monkeypatch):
    """The fill stage's argv is a valid ``filled_reasoning.py generate`` command: seed WORKs then the role's, the
    select stage's rates first, a fixed share of the collect actor."""
    import types
    import filled_reasoning
    from natlang_neuralese.common import data_registry
    monkeypatch.setattr(data_registry, 'role_paths', lambda name: ['/data/demos', '/data/seed'])
    args = types.SimpleNamespace(dir=str(tmp_path / 'c'), pin=str(tmp_path / 'pin'), commit='a' * 40, start_sha='b' * 64,
                                 start='x', sft_work=str(tmp_path / 'w'), fill=str(tmp_path / 'fill'),
                                 seed_work=['/data/seed'], slots=128, fill_inflight=16)
    cycle = ei.Cycle(args)
    (cycle.dir / 'tasks.rates.json').write_text('{}')
    command = cycle.fill_command({'sglang': 'http://127.0.0.1:1', 'url': 'http://127.0.0.1:2', 'export': '/e'})
    assert command[1].endswith('scripts/neuralese_data/filled_reasoning.py')
    parsed = filled_reasoning.build_parser().parse_args(command[2:])
    assert parsed.command == 'generate' and parsed.work == ['/data/seed', '/data/demos']
    assert parsed.prefer_rates == [str(cycle.dir / 'tasks.rates.json')] and parsed.polite
    # beside the collection: the actor runs fill_inflight more requests than collection slots; 2x workers reach the cap
    assert (parsed.min_inflight, parsed.max_inflight, parsed.max_running, parsed.workers) == (16, 32, 144, 32)
    assert cycle.max_running() == 144
    args.fill_window_inflight, args.fill_window_workers = 128, 96
    window = filled_reasoning.build_parser().parse_args(cycle.fill_command(
        {'sglang': 'http://127.0.0.1:1', 'url': 'http://127.0.0.1:2', 'export': '/e'}, window=True)[2:])
    assert not window.polite and (window.min_inflight, window.max_inflight, window.max_running, window.workers) == \
        (128, 128, 128, 96)
    args.fill_repo = str(tmp_path / 'newer')        # fills may run newer data tooling than the pin
    assert cycle.fill_command({'sglang': 'a', 'url': 'b', 'export': 'c'})[1] == \
        str(tmp_path / 'newer' / 'scripts/neuralese_data/filled_reasoning.py')


def test_fill_takes_failed_then_unmeasured_tasks_first():
    import filled_reasoning
    episodes = [{'task_id': t, 'key': t} for t in ('solved', 'unmeasured', 'failed', 'band')]
    rates = {'solved': (4, 4), 'failed': (0, 4), 'band': (1, 4)}
    assert [e['task_id'] for e in filled_reasoning.prefer(episodes, rates)] == ['failed', 'unmeasured', 'band', 'solved']


def test_cycle_builds_the_works_own_neuralese_graph_by_default(tmp_path):
    """The own Neuralese successes (consumer calls included) enter each cycle's mix as a graph part: the data stage
    runs ``sft_sources neuralese-graph`` on the WORK with the build's run exclusions; ``--no-own-neuralese-graph``
    opts out (DECISIONS 2026-11-21 ran-out)."""
    import types
    args = types.SimpleNamespace(dir=str(tmp_path / 'c'), pin=str(tmp_path / 'pin'), commit='a' * 40, start_sha='b' * 64,
                                 start='x', sft_work=str(tmp_path / 'w'), no_own_neuralese_graph=False,
                                 own_neuralese_graph_share=0.05)
    cycle = ei.Cycle(args)
    command = cycle.own_neuralese_graph_command()
    assert command[1].endswith('scripts/neuralese_data/sft_sources.py') and command[2] == 'neuralese-graph'
    assert command[command.index('--work') + 1] == str(tmp_path / 'w')
    assert command[command.index('--out') + 1] == str(cycle.dir / 'neuralese-graph-own')
    assert [command[i + 1] for i, a in enumerate(command) if a == '--exclude-run'] == ['grpo-v2/', 'grpo-v4/run1']
    current = command[command.index('--current-surface-run') + 1]
    import re
    label = 'rl:' + '/'.join(cycle.collect_dir.parts[-2:])
    assert re.search(current, label) and not re.search(current, 'rl:grpo-v4/run2')
    assert sft_sources.current_surface_run(types.SimpleNamespace(current_surface_run=[current]), {'run': label})
    assert not sft_sources.current_surface_run(types.SimpleNamespace(current_surface_run=[]), {'run': label})
    import sys
    argv = sys.argv
    try:
        sys.argv = ['expert_iteration.py', 'cycle', '--dir', 'd', '--name', 'n', '--start', 's', '--commit', 'c']
        parser_args = {}

        def capture(a):
            parser_args['a'] = a
        original = ei.cycle
        ei.cycle = capture
        try:
            ei.main()
        finally:
            ei.cycle = original
    finally:
        sys.argv = argv
    assert parser_args['a'].no_own_neuralese_graph is False and parser_args['a'].own_neuralese_graph_share == 0.05


def test_merge_frontier_lets_the_trained_policys_own_rates_win_over_the_siblings():
    exact = {'schema': ei.FRONTIER_SCHEMA, 'policy': 'start', 'collections': ['/own'],
             'tasks': {'a': {'successes': 0, 'scored': 4, 'tier': 'never'}}}
    sibling = {'schema': ei.FRONTIER_SCHEMA, 'policy': 'cand', 'collections': ['/spec'],
               'tasks': {'a': {'successes': 4, 'scored': 4, 'tier': 'solved'},
                         'b': {'successes': 1, 'scored': 4, 'tier': 'sometimes'}}}
    merged = ei.merge_frontier(exact, sibling)
    assert merged['policy'] == 'start' and merged['sibling_policy'] == 'cand'
    assert merged['tasks']['a'] == {'successes': 0, 'scored': 4, 'tier': 'never', 'measured_by': 'own'}
    assert merged['tasks']['b']['measured_by'] == 'sibling' and merged['tiers'] == {'never': 1, 'sometimes': 1}
    assert merged['collections'] == ['/own', '/spec']


@pytest.mark.parametrize('verdict,setting,expected', [('failure', 'auto', 'recollect'), ('neutral', 'auto', 'keep'),
                                                      ('neutral', 'recollect', 'recollect'), ('failure', 'keep', 'keep')])
def test_rejected_collection_mode(verdict, setting, expected):
    assert ei.rejected_collection_mode(verdict, setting) == expected


def test_next_cycle_argv_replaces_this_cycles_own_flags_and_adds_seed_works():
    assert ei.next_name('sft-v7') == 'sft-v8' and ei.next_name('sft-v9') == 'sft-v10' and ei.next_name('x') is None
    argv = ['--dir', '/ops/ei/sft-v7', '--name', 'sft-v7', '--start', '/v6.pt', '--commit', 'c7', '--k', '4',
            '--aux', 'live=/l/chains.pkl,share=all', '--seed-work', '/s1', '--start-sha=aaa', '--prefer-unseen', '/m6',
            '--fill', '/fill', '--next-seed-work', '/qwen', '--fill-repo', '/pin-new', '--no-teacher-campaign']
    out = ei.next_cycle_argv(argv, directory='/ops/ei/sft-v8', name='sft-v8', start='/v7/sft/checkpoint.pt',
                             start_sha='bbb', commit='c8', previous='/ops/ei/sft-v7', prefer_unseen=['/ops/ei/sft-v7/mix'],
                             fallback_prefer_unseen=['/m6'], extra_seed_work=['/qwen', '/s1'])
    parsed = _cycle_args(out)
    assert (parsed.dir, parsed.name, parsed.start, parsed.start_sha, parsed.commit) == \
        ('/ops/ei/sft-v8', 'sft-v8', '/v7/sft/checkpoint.pt', 'bbb', 'c8')
    assert parsed.seed_work == ['/s1', '/qwen'] and parsed.next_seed_work == [] and parsed.fill_repo is None
    assert parsed.speculative_after == '/ops/ei/sft-v7' and parsed.prefer_unseen == ['/ops/ei/sft-v7/mix']
    assert parsed.fallback_prefer_unseen == ['/m6'] and parsed.aux == ['live=/l/chains.pkl,share=all']
    assert parsed.no_teacher_campaign and parsed.fill == '/fill' and parsed.k == 4 and not parsed.no_pipeline


def _cycle_args(argv):
    import sys
    captured = []
    old_argv, old_cycle = sys.argv, ei.cycle
    sys.argv, ei.cycle = ['expert_iteration.py', 'cycle', *argv], captured.append
    try:
        ei.main()
    finally:
        sys.argv, ei.cycle = old_argv, old_cycle
    return captured[0]


def test_reusable_held_takes_only_a_complete_result_of_the_policy_in_the_series(tmp_path):
    def result(name, **kw):
        path = tmp_path / name / 'result.json'
        path.parent.mkdir()
        path.write_text(json.dumps({'complete': True, 'policy_id': 'p', 'samples_per_task': 2,
                                    'series': {'key': 's1'}, **kw}))
        return path
    paths = [result('a', complete=False), result('b', series={'key': 's0'}), result('c', samples_per_task=4),
             result('d', policy_id='q'), result('e')]
    assert ei.reusable_held(paths, 'p', 2, 's1') == paths[-1]
    assert ei.reusable_held(paths[:-1], 'p', 2, 's1') is None


def _speculative_cycle(tmp_path, verdict):
    import types
    prev = tmp_path / 'sft-v7'
    (prev / 'stages').mkdir(parents=True)
    (prev / 'drive.log').write_text(json.dumps({'stage': 'cycle', 'start': '/v6.pt', 'start_sha': 'six'}) + '\n')
    if verdict:
        (prev / 'verdict.json').write_text(json.dumps(verdict))
    args = types.SimpleNamespace(dir=str(tmp_path / 'sft-v8'), pin=str(tmp_path / 'pin'), commit='a' * 40,
                                 start_sha='cand', start='/v7.pt', sft_work=str(tmp_path / 'w'), fill=None,
                                 speculative_after=str(prev), prefer_unseen=['/v7/mix'], fallback_prefer_unseen=['/m6'],
                                 rejected_collection='auto', slots=128, fill_inflight=32)
    return ei.Cycle(args)


def test_gate_trains_the_adopted_candidate_as_collected(tmp_path):
    cycle = _speculative_cycle(tmp_path, {'verdict': 'success', 'adopt': True, 'vs_start': {}, 'next_start': '/v7.pt'})
    assert cycle.speculative and cycle.train_policy() == ('/v7.pt', 'cand', ['/v7/mix'])
    cycle.gate_stage()
    assert cycle.train_policy() == ('/v7.pt', 'cand', ['/v7/mix']) and cycle.collect_policy() == ('/v7.pt', 'cand')
    assert cycle.gate()['collection'] == 'adopted'


def test_gate_keeps_a_neutral_rejections_collection_and_trains_the_previous_start(tmp_path):
    cycle = _speculative_cycle(tmp_path, {'verdict': 'neutral', 'adopt': False, 'vs_start': {}, 'next_start': '/v6.pt'})
    assert not cycle.recollect_due()
    cycle.gate_stage()
    assert cycle.train_policy() == ('/v6.pt', 'six', ['/m6'])
    assert cycle.collect_policy() == ('/v7.pt', 'cand')          # the sibling's collection stays


def test_gate_recollects_from_the_previous_start_after_a_failure(tmp_path, monkeypatch):
    cycle = _speculative_cycle(tmp_path, {'verdict': 'failure', 'adopt': False, 'vs_start': {}, 'next_start': '/v6.pt'})
    assert cycle.recollect_due()
    (cycle.collect_dir).mkdir()
    (cycle.collect_dir / 'runs.jsonl').write_text(json.dumps({'checkpoint_sha256': 'cand'}) + '\n')
    served = []

    def fake_collect():
        served.append(cycle.collect_policy())
        cycle.set_collect_aside(cycle.collect_policy()[1])
        cycle.mark('collect')
    monkeypatch.setattr(cycle, 'collect_stage', fake_collect)
    cycle.gate_stage()
    assert served == [('/v6.pt', 'six')] and cycle.train_policy()[:2] == ('/v6.pt', 'six')
    assert (cycle.dir / 'collect-spec-cand').exists() and not cycle.collect_dir.exists()


def test_cycle_pipeline_window_and_reuse_flags_default_on():
    parsed = _cycle_args(['--dir', 'd', '--name', 'n', '--start', 's', '--commit', 'c'])
    assert not parsed.no_pipeline and not parsed.no_fill_window and not parsed.no_held_reuse
    assert parsed.fill_inflight == 32 and parsed.next_commit == 'origin/main' and parsed.rejected_collection == 'auto'


def test_wait_ends_the_cycle_when_the_stage_unit_failed(monkeypatch):
    """sft-v7's data unit failed (unregistered archive path) and the stage was still marked done."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        failed = argv[:4] == ['systemctl', '--user', 'is-failed', '-q']
        return type('R', (), {'returncode': 0 if failed else 3})()
    monkeypatch.setattr(ei.subprocess, 'run', run)
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    with pytest.raises(SystemExit, match='x-data failed'):
        ei.Cycle.wait(object.__new__(ei.Cycle), 'x-data')
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: type('R', (), {'returncode': 3})())
    ei.Cycle.wait(object.__new__(ei.Cycle), 'x-data')     # ended cleanly: returns


def test_data_repo_builds_the_mix_with_its_sft_sources_and_is_not_carried_to_the_next_cycle(tmp_path):
    """``--data-repo``: the mix build runs that checkout's sft_sources (on its NATLANG_REPO/PYTHONPATH) while held
    scoring stays on the pin; the default is the pin, and the next pipelined cycle does not inherit it."""
    import types
    base = dict(dir=str(tmp_path / 'c'), pin=str(tmp_path / 'pin'), commit='a' * 40, start_sha='b' * 64, start='x',
                sft_work=str(tmp_path / 'w'), no_own_neuralese_graph=False, own_neuralese_graph_share=0.05, aux=[],
                teacher_budget=1, own_share=0.35, prefer_unseen=[])
    pinned = ei.Cycle(types.SimpleNamespace(**base))
    repo, command = pinned.build_command()
    assert repo == Path(tmp_path / 'pin') and command[1] == str(tmp_path / 'pin/scripts/neuralese_data/sft_sources.py')
    newer = ei.Cycle(types.SimpleNamespace(**base, data_repo=str(tmp_path / 'main')))
    repo, command = newer.build_command()
    assert repo == Path(tmp_path / 'main') and command[1] == str(tmp_path / 'main/scripts/neuralese_data/sft_sources.py')
    assert command[2] == 'build' and command[command.index('--work') + 1] == str(tmp_path / 'w')
    out = ei.next_cycle_argv(['--k', '4', '--data-repo', '/main', '--fill', '/f'], directory='/d', name='n', start='s',
                             start_sha='c' * 64, commit='d' * 40, previous='/p', prefer_unseen=[],
                             fallback_prefer_unseen=[])
    assert '--data-repo' not in out and '/main' not in out and '/f' in out


def test_pop_held_measures_the_start_and_the_base_only_on_request(tmp_path, monkeypatch):
    measured, reused = [], []
    monkeypatch.setattr(ei.Cycle, 'pop_held', lambda self, name, checkpoint: measured.append(name))
    monkeypatch.setattr(ei.Cycle, 'reuse_held', lambda self, out, checkpoint: reused.append(out.name))
    args = dict(dir=str(tmp_path), pin=str(tmp_path / 'pin'), held_samples=2, after=None, no_held_reuse=False)
    ei.pop_held(argparse.Namespace(start='/v7.pt', held_base=False, **args))
    assert (measured, reused) == (['start'], ['base'])
    measured.clear()
    ei.pop_held(argparse.Namespace(start='/v7.pt', held_base=True, **args))
    ei.pop_held(argparse.Namespace(start=ei.BASE_CKPT, held_base=False, **args))
    assert measured == ['start', 'base', 'base']


def test_latest_base_takes_the_newest_complete_base_result_of_any_series(tmp_path):
    paths = []
    for i, r in enumerate([{'complete': True, 'policy_id': ei.BASE_SHA, 'samples_per_task': 2, 'series': {'key': 'a'}},
                           {'complete': True, 'policy_id': ei.BASE_SHA, 'samples_per_task': 2, 'series': {'key': 'b'}},
                           {'complete': False, 'policy_id': ei.BASE_SHA, 'samples_per_task': 2},
                           {'complete': True, 'policy_id': 'other', 'samples_per_task': 2}]):
        path = tmp_path / f'c{i}' / 'held' / 'base' / 'result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(r))
        os.utime(path, (i, i))
        paths.append(path)
    assert ei.latest_base(paths, 2) == paths[1].parent
    assert ei.latest_base(paths, 4) is None


def test_held_pin_measures_on_its_harness_and_is_carried_to_the_next_cycle(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: calls.append((argv, kw.get('env'))))
    argv = ['--dir', str(tmp_path / 'sft-v8'), '--name', 'sft-v8', '--start', '/v7.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', '/pin-new', '--held-pin', '/pin-held']
    cycle = ei.Cycle(_cycle_args(argv))
    monkeypatch.setattr(ei.Cycle, 'reuse_held', lambda self, out, checkpoint: False)
    cycle.pop_held('candidate', tmp_path / 'c.pt')
    assert calls[-1][0][2] == '/pin-held' and calls[-1][1] is None
    assert cycle.held_env['NATLANG_REPO'] == '/pin-held' and cycle.env['NATLANG_REPO'] == '/pin-new'
    out = ei.next_cycle_argv(argv, directory='/sft-v9', name='sft-v9', start='/v8.pt', start_sha='ccc', commit='d' * 40,
                             previous='/sft-v8', prefer_unseen=[], fallback_prefer_unseen=[])
    assert _cycle_args(out).held_pin == '/pin-held'


def test_popbase_unit_runs_the_drivers_pop_held_on_the_held_pin(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: calls.append(argv))
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: False))
    argv = ['--dir', str(tmp_path / 'sft-v8'), '--name', 'sft-v8', '--start', '/v7.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', '/pin-new', '--held-pin', '/pin-held']
    ei.Cycle(_cycle_args(argv)).popbase_stage()
    unit = calls[-1]
    assert all(isinstance(x, str) for x in unit) and '--setenv=NATLANG_REPO=/pin-held' in unit
    assert unit[unit.index('pop-held') - 1] == str(Path(ei.__file__).resolve()) and '/pin-held' in unit


def test_held_every_skips_held_on_unmeasured_cycles_and_adopts_their_candidate(tmp_path, monkeypatch):
    cycle = _speculative_cycle(tmp_path, {'verdict': 'success', 'adopt': True, 'vs_start': {}, 'next_start': '/v7.pt'})
    cycle.a.held_every, cycle.a.since_held = 2, 0
    assert not cycle.measured()
    monkeypatch.setattr(ei.Cycle, 'pop_held', lambda *a: pytest.fail('an unmeasured cycle must not use Pop'))
    cycle.held_stage()
    cycle.verdict_stage()
    v = json.loads((cycle.dir / 'verdict.json').read_text())
    assert v['verdict'] == 'unmeasured' and v['adopt'] and v['next_start'] == str(cycle.dir / 'sft' / 'checkpoint.pt')
    assert v['reference'] == {'checkpoint': '/v7.pt'}           # the last measured policy (the previous next_start)
    cycle.a.since_held = 1
    assert cycle.measured()


def test_a_measured_cycle_after_an_unmeasured_one_is_judged_against_and_rolls_back_to_the_reference(tmp_path, monkeypatch):
    cycle = _speculative_cycle(tmp_path, {'verdict': 'unmeasured', 'adopt': True, 'vs_start': {},
                                          'next_start': '/v7.pt', 'reference': {'checkpoint': '/v6.pt'}})
    cycle.a.held_every, cycle.a.since_held, cycle.a.fail_pts, cycle.a.success_pts = 2, 1, 3., 2.
    cycle.a.held_samples = 2
    monkeypatch.setattr(ei, 'latest_base', lambda results, samples: None)
    compared = []

    def compare(self, a, b, out):
        compared.append((a.name, b.name))
        return {'primary': {'delta': -0.06, 'permutation_p': 0.01}, 'ran_out_rate_delta': 0}
    monkeypatch.setattr(ei.Cycle, 'compare', compare)
    cycle.held.mkdir(parents=True)
    cycle.verdict_stage()
    v = json.loads((cycle.dir / 'verdict.json').read_text())
    assert compared[0] == ('start', 'candidate')
    assert v['verdict'] == 'failure' and v['next_start'] == '/v6.pt' and v['reference'] == {'checkpoint': '/v6.pt'}


def test_next_cycle_counts_unmeasured_cycles():
    argv = ['--dir', '/d', '--name', 'sft-v8', '--start', '/v7.pt', '--commit', 'c', '--since-held', '1']
    out = ei.next_cycle_argv(argv, directory='/d9', name='sft-v9', start='/v8.pt', start_sha='s', commit='c9',
                             previous='/d', prefer_unseen=[], fallback_prefer_unseen=[], since_held=0)
    assert _cycle_args(out).since_held == 0 and _cycle_args(out).held_every == 2
    out = ei.next_cycle_argv(argv, directory='/d9', name='sft-v9', start='/v8.pt', start_sha='s', commit='c9',
                             previous='/d', prefer_unseen=[], fallback_prefer_unseen=[], since_held=2)
    assert _cycle_args(out).since_held == 2


def test_pop_fill_draws_at_pops_running_cap_and_opts_out(tmp_path, monkeypatch):
    argv = ['--dir', str(tmp_path / 'sft-v8'), '--name', 'sft-v8', '--start', '/v7.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', str(tmp_path / 'pin'), '--fill', str(tmp_path / 'fill')]
    cycle = ei.Cycle(_cycle_args(argv))
    monkeypatch.setattr(ei.data_registry if hasattr(ei, 'data_registry') else
                        __import__('natlang_neuralese.common.data_registry', fromlist=['x']), 'role_paths', lambda r: [])
    command = cycle.fill_command({'url': 'http://pop:18531', 'sglang': 'http://127.0.0.1:19631', 'export': '/e'}, pop=True)
    i = command.index('--max-running')
    assert command[i + 1] == '32' and command[command.index('--workers') + 1] == '48' and '--polite' not in command
    started = []
    monkeypatch.setattr(ei.threading, 'Thread', lambda **kw: started.append(kw) or type('T', (), {'start': lambda s: 0})())
    cycle.pop_fill_open()
    assert started
    started.clear()
    cycle = ei.Cycle(_cycle_args([*argv, '--no-pop-fill']))
    cycle.pop_fill_open()
    assert not started
