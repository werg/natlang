"""Expert iteration (scripts/neuralese_data/expert_iteration.py) and its pool rules in sft_sources build."""
import collections
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
    assert (parsed.min_inflight, parsed.max_inflight, parsed.max_running) == (16, 32, 128)


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
