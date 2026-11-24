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


@pytest.fixture(autouse=True)
def _no_live_ledger(monkeypatch):
    """Cycle.present asks the memory ledger whether a unit is requeued: tests never read the live DGX ledger."""
    monkeypatch.setattr(ei, 'ledger_queued', lambda unit: False)


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
    assert ei.group_outcomes([a, b, c, tmp_path / 'missing']) == {'t1': [2, 4], 't2': [0, 2]}   # unscored: a failure


def test_group_outcomes_leave_infrastructure_losses_out(tmp_path):
    directory = tmp_path / 'groups' / 'g000000'
    directory.mkdir(parents=True)
    rows = [{'task_id': 't', 'reward': 1.0, 'outcome': 'task_success'},
            {'task_id': 't', 'reward': None, 'outcome': 'infrastructure_error'},   # the actor died: no measurement
            {'task_id': 't', 'reward': None, 'outcome': 'unresolved'},            # the policy ran out of wall clock
            {'task_id': 'u', 'reward': None, 'outcome': 'infrastructure_error'}]
    (directory / 'rewards.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert ei.group_outcomes([directory]) == {'t': [1, 2]}


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


def test_lineage_builds_carry_what_the_start_trained_into_the_next_cycles_prefer_unseen():
    assert ei.lineage_builds('/v7/mix', ['/v6/mix']) == ['/v7/mix', '/v6/mix']
    assert ei.lineage_builds('/v7/mix', ['/v6/mix', '/v7/mix', '/v6/mix']) == ['/v7/mix', '/v6/mix']
    assert ei.lineage_builds('/v7/mix', []) == ['/v7/mix']


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


def test_a_failure_ends_the_pipeline_at_the_gate_and_collects_the_reference(tmp_path, monkeypatch):
    """AGENTS.md: a significant unexplained held regression ends the experiment at that eval. The speculative cycle
    after a failed candidate trains nothing and launches no next cycle: it writes bug-hunt.json, starts a detached
    collection of the reference (the machines keep working) and exits; a neutral rejection proceeds."""
    cycle = _speculative_cycle(tmp_path, {'verdict': 'failure', 'adopt': False, 'vs_start': {'delta': -0.05},
                                          'next_start': '/v6.pt', 'reference': {'checkpoint': '/v6.pt'}})
    cycle.a.kv_gb, cycle.a.k, cycle.a.collect_hours, cycle.a.no_neuralese_variants = 12, 4, 6., True
    cycle.a.name, cycle.a.no_pipeline = 'sft-v8', False
    launched = []
    monkeypatch.setattr(cycle, 'ledger', lambda unit, budget, prio, cmd, log, **kw: launched.append((unit, cmd, kw)))
    monkeypatch.setattr(cycle, 'active', lambda unit: False)
    with pytest.raises(SystemExit, match='experiment ends'):
        cycle.gate_stage()
    assert not cycle.gate() and cycle.done('halted')
    assert json.loads((cycle.dir / 'bug-hunt.json').read_text())['reference'] == '/v6.pt'
    (unit, cmd, kw), = launched
    assert unit == 'sft-v8-recollect' and kw.get('detach') and cmd[cmd.index('--checkpoint') + 1] == '/v6.pt'
    assert cmd[cmd.index('--checkpoint-sha256') + 1] == 'six'
    with pytest.raises(SystemExit, match='experiment ends'):        # nor does it launch a next cycle
        cycle.next_stage()
    neutral = _speculative_cycle(tmp_path / 'n', {'verdict': 'neutral', 'adopt': False, 'vs_start': {},
                                                  'next_start': '/v6.pt'})
    neutral.gate_stage()
    assert neutral.train_policy()[:2] == ('/v6.pt', 'six')


def test_a_late_failure_stops_the_cycle_and_its_next_one(tmp_path, monkeypatch):
    cycle = _speculative_cycle(tmp_path, None)
    cycle.a.name = 'sft-v8'
    (cycle.dir / 'stages' / 'gate.done').write_text(json.dumps({'start': '/v7.pt', 'start_sha': 'cand',
                                                                'prefer_unseen': [], 'verdict_pending': True}))
    cycle.mark('next', name='sft-v9', unit='sft-v9-drive-1')
    (Path(cycle.a.speculative_after) / 'verdict.json').write_text(json.dumps(
        {'verdict': 'failure', 'adopt': False, 'vs_start': {}, 'next_start': '/v6.pt'}))
    stopped = []
    monkeypatch.setattr(cycle, 'active', lambda unit: unit in ('sft-v9-drive-1', 'sft-v9-collect'))
    monkeypatch.setattr(cycle, 'stop_unit', stopped.append)
    with pytest.raises(SystemExit, match='experiment ends'):
        cycle.held_stage()
    assert stopped == ['sft-v9-drive-1', 'sft-v9-collect'] and not cycle.done('held')


def test_after_the_hunt_a_failure_is_a_rejection_that_recollects_from_the_previous_start(tmp_path, monkeypatch):
    cycle = _speculative_cycle(tmp_path, {'verdict': 'failure', 'adopt': False, 'vs_start': {}, 'next_start': '/v6.pt'})
    cycle.a.resume_after_hunt = True
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
    args = dict(dir=str(tmp_path), pin=str(tmp_path / 'pin'), held_samples=2, after=None, no_held_reuse=False,
                held_profile='pop', held_slots=64, name=None)     # the subcommand's defaults (sft-v9 spawns it so)
    ei.pop_held(argparse.Namespace(start='/v7.pt', held_base=False, **args))
    assert (measured, reused) == (['start'], ['base'])
    measured.clear()
    ei.pop_held(argparse.Namespace(start='/v7.pt', held_base=True, **args))
    ei.pop_held(argparse.Namespace(start=ei.BASE_CKPT, held_base=False, **args))
    assert measured == ['start', 'base', 'base']


def test_latest_base_takes_the_newest_complete_base_result_of_any_series_of_the_host(tmp_path):
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
            '--commit', 'c' * 40, '--pin', '/pin-new', '--held-pin', '/pin-held', '--held-profile', 'pop']
    cycle = ei.Cycle(_cycle_args(argv))
    monkeypatch.setattr(ei.Cycle, 'reuse_held', lambda self, out, checkpoint: False)
    import pop_rl_actor
    stopped = []
    monkeypatch.setattr(pop_rl_actor, 'stop_pop_fills', lambda: stopped.append(1))
    monkeypatch.setattr(pop_rl_actor, 'pop_held_units', lambda pattern='*popheld*': [])   # no unit left to join
    cycle.pop_held('candidate', tmp_path / 'c.pt')
    assert stopped                     # a measurement first stops any Pop fill
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
            '--commit', 'c' * 40, '--pin', '/pin-new', '--held-pin', '/pin-held', '--held-profile', 'pop']
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
    monkeypatch.setattr(ei, 'latest_base', lambda results, samples, host: None)
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


@pytest.mark.parametrize('previous,expected', [
    ({'verdict': 'unmeasured', 'since_held': 0, 'held_every': 4}, (1, 4)),
    ({'verdict': 'success', 'since_held': 1, 'held_every': 2}, (0, 4)),
    ({'verdict': 'success', 'since_held': 7, 'held_every': 8}, (0, 8)),
    ({'verdict': 'neutral', 'since_held': 3, 'held_every': 4}, (0, 2)),
    ({'verdict': 'failure', 'since_held': 1, 'held_every': 2}, (0, 1)),
    ({'verdict': 'success'}, (0, 2))])                # an older verdict: the flags
def test_held_interval_doubles_on_success_and_resets_otherwise(tmp_path, previous, expected):
    cycle = _speculative_cycle(tmp_path, {'adopt': True, 'vs_start': {}, 'next_start': '/v7.pt', **previous})
    cycle.a.held_every, cycle.a.held_every_max, cycle.a.since_held = 2, 8, 0
    assert cycle.held_schedule() == expected
    assert cycle.measured() == (expected[0] + 1 >= expected[1])


def test_pop_serves_stop_earlier_pop_fills(monkeypatch):
    import pop_rl_actor
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        out = 'sft-v8-popfill.service loaded active running x\nsft-v8-popfill-tunnel.service loaded active running y\n'
        return type('R', (), {'stdout': out if argv[:2] == ['systemctl', '--user'] and argv[2] == 'list-units' else '',
                              'returncode': 0})()
    monkeypatch.setattr(pop_rl_actor.subprocess, 'run', run)
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    pop_rl_actor.stop_pop_fills()
    stopped = [a[-1] for a in calls if a[:3] == ['systemctl', '--user', 'stop']]
    assert stopped == ['sft-v8-popfill-waiter.service', 'sft-v8-popfill.service',
                       'sft-v8-popfill-tunnel-waiter.service', 'sft-v8-popfill-tunnel.service']
    monkeypatch.setattr(pop_rl_actor, 'ssh', lambda *a, **k: '')
    calls.clear()
    pop_rl_actor.stop_pop_actors()
    assert any(a[:3] == ['systemctl', '--user', 'stop'] for a in calls)


def test_unreachable_pop_defers_the_held_eval_and_the_next_cycle_is_measured(tmp_path, monkeypatch):
    previous = tmp_path / 'sft-v8'
    previous.mkdir()
    (previous / 'verdict.json').write_text(json.dumps({'verdict': 'unmeasured', 'adopt': True, 'since_held': 0,
                                                       'held_every': 2, 'next_start': '/v8.pt'}))
    argv = ['--dir', str(tmp_path / 'sft-v9'), '--name', 'sft-v9', '--start', '/v8.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', str(tmp_path / 'pin'), '--speculative-after', str(previous),
            '--pop-wait-hours', '0', '--held-profile', 'pop']
    cycle = ei.Cycle(_cycle_args(argv))
    (cycle.dir / 'stages').mkdir(parents=True, exist_ok=True)
    assert cycle.measured()
    monkeypatch.setattr(ei, 'pop_reachable', lambda: False)
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: False))
    monkeypatch.setattr(ei.Cycle, 'pop_held', lambda *a: pytest.fail('measured with Pop down'))
    cycle.held_stage()
    assert cycle.held_deferred()
    cycle.verdict_stage()
    v = json.loads((cycle.dir / 'verdict.json').read_text())
    assert v['verdict'] == 'unmeasured' and v['adopt'] and v['deferred'] == 'pop unreachable'
    assert v['since_held'] == 1 and v['held_every'] == 2      # the next cycle: since 1 + 1 >= 2, measured


def test_sft_gets_the_activation_memory_flag(tmp_path, monkeypatch):
    argv = ['--dir', str(tmp_path / 'sft-v9'), '--name', 'sft-v9', '--start', '/v8.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', str(tmp_path / 'pin')]
    args = _cycle_args(argv)
    assert args.sft_activation_gb == 56 and args.sft_budget == 100
    cycle = ei.Cycle(_cycle_args([*argv, '--sft-activation-gb', '24']))
    (cycle.mix).mkdir(parents=True)
    (cycle.mix / 'chains.sha256').write_text('abc\n')
    seen = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: seen.append(kw['env']))
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: False))
    monkeypatch.setattr(ei.Cycle, 'wait', lambda self, unit, poll=60: None)
    with pytest.raises(SystemExit):           # no checkpoint written by the stubbed launch
        cycle.sft_stage()
    assert seen[0]['ACT'] == '24' and seen[0]['BUDGET'] == '100'


def test_collection_runs_neuralese_variants_by_default_and_opts_out(tmp_path):
    argv = ['--dir', str(tmp_path / 'sft-v10'), '--name', 'sft-v10', '--start', '/v9.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', str(tmp_path / 'pin')]
    cycle = ei.Cycle(_cycle_args(argv))
    flags = cycle.variant_args()
    assert flags[0] == '--neuralese-variants' and flags[1].endswith('natlang-rl-neuralese-variants-20261113/cases.jsonl')
    assert float(flags[3]) == 0.5
    assert ei.Cycle(_cycle_args([*argv, '--no-neuralese-variants'])).variant_args() == []



# ---- the DGX held series (async-train §1) -----------------------------------------------------------------------------
def _held_result(path, host=None, fmt='sglang-hf-bfloat16-vendor-projection-kv-fp8_e4m3', policy='p', samples=4, key='k'):
    path.mkdir(parents=True, exist_ok=True)
    series = {'key': key, **({'engine_host': host} if host else {})}
    (path / 'result.json').write_text(json.dumps({'complete': True, 'policy_id': policy, 'samples_per_task': samples,
                                                  'series': series, 'engine': {'export_format': fmt}}))
    return path / 'result.json'


def test_a_dgx_cycle_measures_on_the_dgx_through_the_ledger_and_re_measures_a_pop_start(tmp_path, monkeypatch):
    """--held-profile dgx was accepted but measured on Pop; popbase wrote held/start from Pop beside the candidate."""
    argv = ['--dir', str(tmp_path / 'sft-v10'), '--name', 'sft-v10', '--start', '/v9.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', str(tmp_path / 'pin'), '--held-every', '1']
    cycle = ei.Cycle(_cycle_args(argv))
    assert cycle.held_host == 'dgx' and cycle.a.held_samples == 4 and cycle.base_unit() == 'sft-v10-heldbase'
    pop_start = _held_result(cycle.held / 'start', fmt='sglang-pop-rtx4060-bf16-vendor-projection-kv-fp8_e4m3')
    (cycle.dir / 'sft').mkdir()
    (cycle.dir / 'sft' / 'checkpoint.pt').write_bytes(b'candidate')
    monkeypatch.setattr(ei.Cycle, 'reference', lambda self: {'checkpoint': '/v9.pt'})
    monkeypatch.setattr(ei.Cycle, 'reuse_held', lambda self, out, checkpoint: False)
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: False))
    monkeypatch.setattr(ei, 'ledger_queued', lambda unit: False)
    monkeypatch.setattr(ei, 'pop_reachable', lambda: pytest.fail('a DGX cycle never waits for Pop'))
    monkeypatch.setattr(ei.Cycle, 'pop_held', lambda *a: pytest.fail('a DGX cycle never measures on Pop'))
    import natlang_neuralese.common.hashing as hashing
    monkeypatch.setattr(hashing, 'sha256_file_hex_cached', lambda path: 'h' * 64)
    units = []

    def ledger(self, unit, budget, priority, command, log, detach=False, workdir=None, repo=None):
        units.append((unit, budget, priority, command, repo))
        _held_result(Path(command[command.index('--out') + 1]), host='dgx', policy='h' * 64)
    monkeypatch.setattr(ei.Cycle, 'ledger', ledger)
    cycle.held_stage()
    names = [u[0] for u in units]
    assert names == ['sft-v10-held-candidate', 'sft-v10-held-start']       # the Pop start is re-measured here
    unit, budget, priority, command, repo = units[0]
    assert command[1] == str(Path(ei.__file__).resolve()) and command[2] == 'dgx-held' and repo == cycle.held_pin
    assert command[command.index('--samples') + 1] == '4' and budget == ei.held_budget_gb() == 32   # fits beside a collection
    assert [p.name for p in cycle.held.iterdir() if p.name.startswith('start-pop-')]   # kept aside, not paired
    assert json.loads(pop_start.read_text())['series']['engine_host'] == 'dgx'      # held/start is the DGX one now
    assert json.loads((cycle.dir / 'stages' / 'held.done').read_text())['profile'] == 'dgx'


def test_held_reuse_and_the_latest_base_never_cross_hosts(tmp_path):
    pop = _held_result(tmp_path / 'a', fmt='sglang-pop-rtx4060-bf16-vendor-projection-kv-fp8_e4m3')
    dgx_old = _held_result(tmp_path / 'b')                     # predates engine_host: placed by its export format
    dgx = _held_result(tmp_path / 'c', host='dgx')
    assert ei.reusable_held([pop], 'p', 4, 'k', host='dgx') is None      # one key (an old pin), another host
    assert ei.reusable_held([pop, dgx_old], 'p', 4, 'k', host='dgx') == dgx_old
    assert ei.reusable_held([dgx, pop], 'p', 4, 'k', host='pop') == pop
    # any_series: without an exact match, any complete result of the policy at >= 2 samples, exact preferred
    assert ei.reusable_held([pop], 'p', 4, 'k', host='dgx', any_series=True) == pop
    assert ei.reusable_held([pop, dgx], 'p', 4, 'k', host='dgx', any_series=True) == dgx
    bases = [_held_result(tmp_path / f'base{i}', policy=ei.BASE_SHA, host=h, fmt=f)
             for i, (h, f) in enumerate([(None, 'sglang-pop-rtx4060-x'), ('dgx', 'sglang-hf-x')])]
    os.utime(bases[0], (5, 5))
    os.utime(bases[1], (1, 1))
    assert ei.latest_base(bases, 4, 'pop') == bases[0].parent and ei.latest_base(bases, 4, 'dgx') == bases[1].parent


def test_compare_refuses_a_pop_result_against_a_dgx_one():
    from natlang_neuralese.rl import held_eval
    sample = {'task_id': 't', 'sample': 0, 'ran': True, 'scored': True, 'success': True, 'outcome': 'pass', 'reward': 1.0}

    def result(**kw):
        return {'complete': True, 'policy_id': 'p', 'series': {'key': 'k'}, 'samples': [sample], 'ran': 1, 'scored': 1,
                'successes': 1, 'loop_rate': 0, **kw}
    pop = result(engine={'engine': 'sglang', 'export_format': 'sglang-pop-rtx4060-bf16-vendor-projection-kv-fp8_e4m3'})
    dgx = result(engine={'engine': 'sglang', 'export_format': 'sglang-hf-bfloat16-vendor-projection-kv-fp8_e4m3'})
    assert held_eval.engine_host_of(pop) == 'pop' and held_eval.engine_host_of(dgx) == 'dgx'
    assert held_eval.engine_host_of({**dgx, 'series': {'key': 'k', 'engine_host': 'colab_g4'}}) == 'colab_g4'
    with pytest.raises(ValueError, match='different engine'):
        held_eval.compare(pop, dgx, strict_series=True)   # refused (engine settings or hosts)
    # by default (owner 2026-11-22) two hosts pair, with the difference recorded as a warning
    assert any('different engine hosts' in w for w in held_eval.compare(pop, dgx, resamples=10)['warnings'])


def test_the_dgx_series_key_has_its_host_and_pops_stays(monkeypatch):
    from natlang_neuralese.rl.held_eval import series
    monkeypatch.delenv('NATLANG_HELD_SERIES_HOST', raising=False)
    legacy = series()
    assert series(engine_host='pop') == legacy and 'engine_host' not in legacy
    monkeypatch.setenv('NATLANG_HELD_SERIES_HOST', 'dgx')
    assert series()['engine_host'] == 'dgx' and series()['key'] != legacy['key']


def test_a_carried_pop_held_pin_starts_the_dgx_series_on_the_cycles_own_pin_and_the_next_cycle_carries_it(tmp_path):
    """sft-v9's driver launches sft-v10 with --held-pin pin-5b58 (its Pop series) and no host: that harness predates
    host keys and is another machine's series."""
    argv = ['--dir', str(tmp_path / 'sft-v10'), '--name', 'sft-v10', '--start', '/v9.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', '/pin-own', '--held-pin', '/pin-5b58']
    cycle = ei.Cycle(_cycle_args(argv))
    assert cycle.held_pin == Path('/pin-own') and cycle.held_env['NATLANG_REPO'] == '/pin-own'
    out = ei.next_cycle_argv(argv, directory=tmp_path / 'sft-v11', name='sft-v11', start='/v10.pt', start_sha='ccc',
                             commit='d' * 40, previous='/sft-v10', prefer_unseen=[], fallback_prefer_unseen=[],
                             held_pin=cycle.held_pin, held_series_host=cycle.held_host)
    nxt = _cycle_args(out)
    assert (nxt.held_pin, nxt.held_series_host) == ('/pin-own', 'dgx') and out.count('--held-pin') == 1
    assert ei.Cycle(nxt).held_pin == Path('/pin-own')                       # the series stays fixed from here on
    pop = ei.Cycle(_cycle_args([*argv, '--held-profile', 'pop']))
    assert pop.held_pin == Path('/pin-5b58') and pop.held_host == 'pop' and pop.a.held_samples == 2


def test_dgx_held_serves_the_pinned_engine_claims_priority_and_measures_on_the_training_law(tmp_path, monkeypatch):
    out, pin = tmp_path / 'held' / 'candidate', tmp_path / 'pin'
    manifest = tmp_path / 'held-v3' / 'manifest.json'
    manifest.parent.mkdir()
    manifest.write_text('{}')
    (manifest.parent / 'engine.json').write_text(json.dumps({'serve_args': ['--engine', 'sglang', '--sglang-projection',
                                                                            'vendor', '--sglang-kv-cache-dtype', 'fp8_e4m3']}))
    monkeypatch.setattr(ei, 'HELD_PRIORITY_DIR', tmp_path / 'prio')
    launched, claims = [], []

    class Proc:
        pid = 4321

        def __init__(self, argv, **kw):
            launched.append((argv, kw['env']))

        def poll(self):
            return 0
    monkeypatch.setattr(ei.subprocess, 'Popen', Proc)
    monkeypatch.setattr(ei, 'wait_actor', lambda url, sha, process, log, timeout=3600: {})

    def run(argv, **kw):
        claims.extend(ei.held_priority_claims(tmp_path / 'prio'))
        launched.append((argv, kw['env']))
        return subprocess_result(0)
    import subprocess as sp
    subprocess_result = lambda rc: sp.CompletedProcess([], rc)
    monkeypatch.setattr(ei.subprocess, 'run', run)
    monkeypatch.setattr(ei.signal, 'signal', lambda *a: None)
    args = argparse.Namespace(out=str(out), checkpoint='/c.pt', policy_id='h' * 64, samples=4, slots=16, kv_gb=3,
                              max_running=16,
                              held_pin=str(pin), manifest=str(manifest), label='sft-v10-candidate',
                              unit='sft-v10-held-candidate')
    assert ei.dgx_held(args) == 0
    serve, env = launched[0]
    assert serve[1] == str(pin / 'scripts/neuralese_data/serve_native_checkpoint.py') and env['NATLANG_REPO'] == str(pin)
    assert serve[serve.index('--sglang-projection') + 1] == 'vendor' and '--sglang-overlap-schedule' not in serve
    # the pinned args first, the beside-collection size after them (argparse: the last value wins)
    assert serve[len(serve) - 1 - serve[::-1].index('--sglang-kv-gb') + 1] == '3'
    assert serve[len(serve) - 1 - serve[::-1].index('--sglang-max-running') + 1] == '16'
    measure, env = launched[1]
    assert measure[1:4] == ['-m', 'natlang_neuralese.rl.held_eval', 'measure'] and '--allow-other-engine' not in measure
    assert measure[measure.index('--samples') + 1] == '4' and measure[measure.index('--top-p') + 1] == '0.95'
    assert measure[measure.index('--max-model-requests') + 1] == '40'
    assert env['NATLANG_HELD_SERIES_HOST'] == 'dgx' and claims and claims[0]['unit'] == 'sft-v10-held-candidate'
    assert not list((tmp_path / 'prio').iterdir())             # the claim ends with the measurement


def test_pop_held_subcommand_measures_on_the_profile_it_is_given(tmp_path, monkeypatch):
    measured = []
    monkeypatch.setattr(ei.Cycle, 'dgx_held', lambda self, name, checkpoint: measured.append(('dgx', name)))
    monkeypatch.setattr(ei.Cycle, 'pop_held', lambda self, name, checkpoint: measured.append(('pop', name)))
    monkeypatch.setattr(ei.Cycle, 'reuse_held', lambda self, out, checkpoint: None)
    args = dict(dir=str(tmp_path), pin=str(tmp_path / 'pin'), held_samples=4, after=None, no_held_reuse=False,
                held_base=False, held_slots=64, name='sft-v10', start='/v9.pt')
    ei.pop_held(argparse.Namespace(held_profile='dgx', **args))
    ei.pop_held(argparse.Namespace(held_profile='pop', **args))
    assert measured == [('dgx', 'start'), ('pop', 'start')]


def test_the_gate_never_waits_for_a_held_eval_and_a_late_rejection_is_judged_at_the_held_stage(tmp_path, monkeypatch):
    """Training never waits: the DGX held of the previous candidate may still run when this cycle's collection ends."""
    cycle = _speculative_cycle(tmp_path, None)
    cycle.a.held_every, cycle.a.held_every_max, cycle.a.since_held = 2, 8, 0
    monkeypatch.setattr(ei.time, 'sleep', lambda s: pytest.fail('the gate waited'))
    cycle.gate_stage()
    assert cycle.gate()['verdict_pending'] and cycle.train_policy() == ('/v7.pt', 'cand', ['/v7/mix'])
    assert cycle.collect_policy() == ('/v7.pt', 'cand')
    prev = Path(cycle.a.speculative_after)
    (prev / 'verdict.json').write_text(json.dumps({'verdict': 'neutral', 'adopt': False, 'vs_start': {},
                                                   'next_start': '/v6.pt', 'reference': {'checkpoint': '/v6.pt'},
                                                   'since_held': 0, 'held_every': 2}))
    assert cycle.held_schedule() == (0, 1) and cycle.measured()           # the rejected lineage is judged now
    assert cycle.reference() == {'checkpoint': '/v6.pt'}                    # against, and back to, the measured start


def test_a_preempted_held_unit_is_joined_until_it_resumes_and_finishes(tmp_path, monkeypatch):
    argv = ['--dir', str(tmp_path / 'sft-v10'), '--name', 'sft-v10', '--start', '/v9.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', str(tmp_path / 'pin')]
    cycle = ei.Cycle(_cycle_args(argv))
    monkeypatch.setattr(ei.Cycle, 'reuse_held', lambda self, out, checkpoint: False)
    import natlang_neuralese.common.hashing as hashing
    monkeypatch.setattr(hashing, 'sha256_file_hex_cached', lambda path: 'h' * 64)
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    launches, queue = [], iter([True, True, False])          # preempted: requeued twice, then relaunched by the ledger

    def ledger(self, unit, budget, priority, command, log, **kw):
        launches.append((unit, priority))
    monkeypatch.setattr(ei.Cycle, 'ledger', ledger)
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: False))

    def queued(unit):
        state = next(queue, False)
        if state is False and not (cycle.held / 'candidate' / 'result.json').exists():
            _held_result(cycle.held / 'candidate', host='dgx', policy='h' * 64)
        return state
    monkeypatch.setattr(ei, 'ledger_queued', queued)
    cycle.dgx_held('candidate', tmp_path / 'c.pt')
    assert launches == [('sft-v10-held-candidate', 'medium')]       # joined, not launched twice; below training's high


def test_select_never_takes_a_held_v4_task_or_sibling_and_the_gate_stays_held_v3(tmp_path, monkeypatch):
    """B1: expert-iteration selection (pool_args -> grpo.load_tasks) excludes every held-eval* role, not only held-v3."""
    import grpo
    from natlang_neuralese.common import data_registry
    from natlang_neuralese.rl import held_eval

    def held_set(name, task_id, group):
        directory = tmp_path / name
        directory.mkdir()
        case = {'id': task_id, 'curriculum': {'split_group': group}}
        (directory / 'cases.jsonl').write_text(json.dumps(case) + '\n')
        sha = held_eval._sha256(directory / 'cases.jsonl')
        (directory / 'manifest.json').write_text(json.dumps({
            'schema': held_eval.MANIFEST_SCHEMA, 'tasks': [{'id': task_id}], 'excluded_split_groups': [group],
            'pool': {'path': str(directory / 'cases.jsonl'), 'sha256': sha}, 'cases': {'path': 'cases.jsonl', 'sha256': sha}}))
        return str(directory / 'manifest.json')

    v3, v4 = held_set('v3', 'a', 'ga'), held_set('v4', 'b', 'gb')
    roles = [('held-eval', v3), ('held-eval-semantic-fp', v4)]
    monkeypatch.setattr(data_registry, 'role_prefix_paths', lambda prefix, repo=None: roles)
    monkeypatch.setattr(data_registry, 'role_path', lambda name, repo=None: dict(roles)[name])
    rows = [{'id': i, 'split': 'train', 'family': 'f', 'curriculum': {'split_group': g}}
            for i, g in (('a', 'ga'), ('b', 'gb'), ('b-sibling', 'gb'), ('c', 'gc'))]
    pool = tmp_path / 'pool.jsonl'
    pool.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    args = argparse.Namespace(pool=[str(pool)], held_manifest=None, exclude_family_regex='(?!)', seed=0)
    assert grpo.default_held() == v3
    eligible, excluded = grpo.load_tasks(ei.pool_args(args), None)
    assert [r['id'] for r in eligible] == ['c'] and excluded == 3


def test_cycle_stages_outrank_its_collection_and_a_requeued_unit_is_joined(tmp_path, monkeypatch):
    """The cycle's frontier, data and build run at the stage priority (above every collection's), and a preempted,
    requeued unit is not over: ``wait`` and the collect loop join it, ``ledger`` never launches it a second time."""
    argv = ['--dir', str(tmp_path / 'sft-v10'), '--name', 'sft-v10', '--start', '/v9.pt', '--start-sha', 'bbb',
            '--commit', 'c' * 40, '--pin', str(tmp_path / 'pin')]
    cycle = ei.Cycle(_cycle_args(argv))
    (cycle.dir / 'stages').mkdir(parents=True, exist_ok=True)
    seen = []

    def ledger(self, unit, budget, priority, command, log, **kw):
        seen.append((unit, priority))
        cycle.mix.mkdir(parents=True, exist_ok=True)
        (cycle.mix / 'chains.pkl').write_bytes(b'x')
        (cycle.mix / 'build.json').write_text(json.dumps({'episodes': 1, 'tasks': 1, 'forward_tokens': 1,
                                                          'mix': {'parts': {}}}))
    monkeypatch.setattr(ei.Cycle, 'ledger', ledger)
    monkeypatch.setattr(ei.Cycle, 'build_command', lambda self: (cycle.pin, ['build']))
    cycle.build_stage()
    assert seen == [('sft-v10-build', ei.STAGE_PRIORITY)] and ei.STAGE_PRIORITY == 'high'
    assert ei.COLLECTION_PRIORITY == 'medium'
    # a requeued unit: joined until the ledger relaunched it and it ended
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: False))
    states = iter([True, True, False])
    monkeypatch.setattr(ei, 'ledger_queued', lambda unit: next(states, False))
    calls = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: calls.append(argv) or
                        argparse.Namespace(returncode=1))
    cycle.wait('sft-v10-collect')
    assert next(states, 'drained') == 'drained'
    monkeypatch.undo()
    monkeypatch.setattr(ei, 'ledger_queued', lambda unit: True)
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: False))
    monkeypatch.setattr(ei.Cycle, 'wait', lambda self, unit, poll=60: None)
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: pytest.fail(f'launched again: {argv}'))
    cycle.ledger('sft-v10-collect', 74, ei.COLLECTION_PRIORITY, ['collect'], tmp_path / 'c.log', detach=True)

def test_teacher_specs_come_from_the_ops_file_at_each_launch(tmp_path, monkeypatch):
    import json
    import pytest
    path = tmp_path / 'teachers.json'
    monkeypatch.setattr(ei, 'TEACHERS_FILE', path)
    assert ei.teachers() == ei.TEACHERS
    specs = [{'model_id': 'm', 'endpoint': 'https://integrate.api.nvidia.com/v1', 'api_key_env': 'NVIDIA_API_KEY', 'slots': 1}]
    path.write_text(json.dumps(specs))
    assert ei.teachers() == specs
    command = ei.teacher_command(tmp_path, tmp_path / 'w', [], [], 5)
    assert command[command.index('--teacher') + 1] == json.dumps(specs[0])
    path.write_text('[]')
    with pytest.raises(SystemExit):
        ei.teachers()
    assert 'openrouter' in ei.TEACHER_ENV_FILES


def test_the_standing_teacher_queue_takes_every_task_with_every_collections_rates(tmp_path, monkeypatch):
    """2026-11-24: a campaign took only --limit = the newest frontier's rare tasks (41-46 with 1 h snapshots) and its 24
    slots sat idle ~1.5 h of every 2. The standing launch passes --limit 0 (the whole ordered queue), the newest
    collection's tasks.rates.json and the frontier (newest measurement), and clears a drain request of its WORK."""
    import json
    monkeypatch.setattr(ei, 'TEACHERS_FILE', tmp_path / 'none.json')
    frontier = tmp_path / 'r0005' / 'frontier.json'
    frontier.parent.mkdir()
    frontier.write_text(json.dumps({'tasks': {'a': {'successes': 4, 'scored': 4, 'tier': 'solved'}}}))
    work = tmp_path / 'teacher-r0005'
    work.mkdir()
    (work / 'DRAIN').write_text('{}')
    launches = []
    runner = argparse.Namespace(pin=tmp_path / 'pin', ledger=lambda unit, gb, prio, argv, log, detach: launches.append(argv))
    assert ei.launch_teacher(runner, work, frontier, [], 'aei-teacher') == 0 and not launches   # no rare task
    rates = tmp_path / 'tasks.rates.json'
    assert ei.launch_teacher(runner, work, frontier, ['g/*'], 'aei-teacher', standing=True, student_rates=[rates]) == 0
    assert not (work / 'DRAIN').exists()
    script = (work / 'launch.sh').read_text()
    assert "'--limit' '0'" in script or '--limit 0' in script
    assert f'--student-rates {rates}' in script and f'--frontier-rates {frontier}' in script
    command = ei.teacher_command(tmp_path, work, [], [], 7)
    assert command[command.index('--limit') + 1] == '7' and '--frontier-rates' not in command
