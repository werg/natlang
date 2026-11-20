#!/usr/bin/env python3
"""Simulate the RL task selector over a pool and measure the mixture it actually samples.

  simulate_selection.py --cases POOL/cases.jsonl [--rounds 400] [--rows 16] [--group 8] [--compare-without-quota] -o report.json
  simulate_selection.py --grpo [--cases POOL ...] [--groups 2000] [--explore-order family] -o report.json

``--grpo``: the GRPO collector's own selection (``grpo.Collector._pick``: ``--explore`` of groups from the unmeasured-task
walk ``grpo.ExploreWalk``, the rest cycling tasks in the pass band, each task's latest group replacing its rate) over the
GRPO pool minus the held closure (``grpo.load_tasks``; ``--cases`` replaces the registry role grpo-pool), with the latent outcome model,
once with the construct weighting off (wrapper share 1, held weight 1) and once with grpo's defaults; it reports the
held constructs' shares among explored and among all groups.

Runs ``curriculum.select_tasks`` (policy ``natlang.rl-task-curriculum/2`` with its
default ``class_quota``) for many rounds against synthetic outcomes: a task's pass probability falls with its level and each
group of samples is drawn from it, so family frontiers move and outcome-spread groups appear as in a live run. Synthetic
outcomes only drive the selector's adaptation; nothing here is a model result. The report measures, per round and overall,
the share of rows whose oracle verifies a world (recomputed from the rows with ``measure_repetition.program_features``,
independent of the pool's ``mix_class`` tags), and the distinct rows, programs and families actually sampled.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from measure_repetition import program_features  # noqa: E402

PASS_BY_LEVEL = {0: .85, 1: .65, 2: .45, 3: .25}
LATENT_CONCENTRATION = 1.5  # Beta(m*c, (1-m)*c): task pass rates spread to both ends, as measured outcome groups do
FAMILY_LEARNING = .06       # logit gained by a family per group of its tasks sampled (training effect), capped below
FAMILY_LEARNING_CAP = 2.5


class OutcomeModel:
    """Synthetic outcomes for the selector. ``level``: one pass probability per level (the original model; a stable
    mixed-outcome row keeps being re-picked). ``latent``: every task has its own pass rate drawn once from a Beta around
    its level's mean (most tasks near 0 or 1, some mixed, as real attempt groups are), and every group sampled from a
    family raises that family's skill (the policy learns), so tasks solve and drop out of the spread band over the run."""

    def __init__(self, kind, rng):
        self.kind, self.rng, self.latent, self.skill = kind, rng, {}, collections.Counter()

    def probability(self, row_id, level, family):
        mean = PASS_BY_LEVEL.get(level, .25)
        if self.kind == 'level':
            return mean
        if row_id not in self.latent:
            self.latent[row_id] = self.rng.betavariate(mean * LATENT_CONCENTRATION, (1 - mean) * LATENT_CONCENTRATION)
        base = min(max(self.latent[row_id], 1e-4), 1 - 1e-4)
        import math
        logit = math.log(base / (1 - base)) + min(self.skill[family], FAMILY_LEARNING_CAP)
        return 1 / (1 + math.exp(-logit))

    def learned(self, family):
        if self.kind == 'latent':
            self.skill[family] += FAMILY_LEARNING


def _top_share(picked, keys, total, top=8):
    """The ``top`` largest groups (family or program) by share of all sampled rows."""
    groups = collections.Counter()
    for index, count in picked.items():
        groups[keys[index]] += count
    return {str(key)[:80]: round(count / total, 4) for key, count in groups.most_common(top)}


def simulate(cases, *, rounds, rows, group, config, seed=1103, outcomes='level'):
    from natlang_neuralese.rl import curriculum
    from natlang_neuralese.rl.curriculum import metadata, record_visit
    rng = random.Random(seed)
    model = OutcomeModel(outcomes, random.Random(seed + 1))
    groups_total = groups_spread = 0
    features = [program_features(row) for row in cases]
    history, offset = {}, 0
    picked = collections.Counter()
    per_round = []
    for number in range(rounds):
        stats = list(history.values())
        chosen = curriculum.select_tasks(cases, stats, {'rows': rows, 'group': group, 'offset': offset}, curriculum=config)
        effect = sum(1 for item in chosen if features[item['row']]['effect_state_oracle'])
        per_round.append(effect / len(chosen))
        for item in chosen:
            row = cases[item['row']]
            picked[item['row']] += 1
            level = metadata(row)['level']
            family = metadata(row)['family']
            p = model.probability(row['id'], level, family)
            wins = sum(rng.random() < p for _ in range(item['samples']))
            groups_total += 1
            groups_spread += 0 < wins < item['samples']
            model.learned(family)
            history[row['id']] = {'task_id': row['id'], 'scored': item['samples'], 'mean': wins / item['samples'],
                                  'spread': 0 < wins < item['samples'], 'observed_round': number,
                                  **record_visit(history.get(row['id']), number)}
        offset = (chosen[-1]['row'] + 1) % len(cases)
    tagged = {i for i, row in enumerate(cases) if metadata(row).get('mix_class') == 'effect_state'}
    strict = {i for i, f in enumerate(features) if f['effect_state_oracle']}
    selected = set(picked)
    programs_pool = {f['functional'] for f in features}
    families_pool = {metadata(row)['family'] for row in cases}
    total = sum(picked.values())
    per_round_sorted = sorted(per_round)
    visits = sorted(picked.values())
    return {'rounds': rounds, 'rows_per_round': rows, 'group': group, 'pool_rows': len(cases), 'outcomes': outcomes,
            'rows_with_20plus_visits': sum(1 for v in visits if v >= 20),
            'groups_sampled': groups_total, 'groups_with_outcome_spread': round(groups_spread / groups_total, 4),
            'visit_histogram': {'1': sum(1 for v in visits if v == 1), '2-4': sum(1 for v in visits if 2 <= v <= 4),
                                '5-19': sum(1 for v in visits if 5 <= v <= 19), '20+': sum(1 for v in visits if v >= 20)},
            'pool_effect_rows_strict': len(strict), 'pool_effect_share': round(len(strict) / len(cases), 4),
            'tagged_rows': len(tagged), 'tags_equal_strict_world_rows': tagged == strict if tagged else None,
            'sampled_rows': total,
            'sampled_effect_share': round(sum(n for i, n in picked.items() if i in strict) / total, 4),
            'round_effect_share': {'min': round(per_round_sorted[0], 4), 'p05': round(per_round_sorted[len(per_round) // 20], 4),
                                   'mean': round(sum(per_round) / rounds, 4), 'max': round(per_round_sorted[-1], 4),
                                   'rounds_below_0.30': sum(1 for v in per_round if v < .30)},
            'distinct_rows_sampled': len(selected), 'distinct_rows_share': round(len(selected) / len(cases), 4),
            'distinct_effect_rows_sampled': len(selected & strict),
            'distinct_functional_programs_sampled': len({features[i]['functional'] for i in selected}),
            'functional_programs_in_pool': len(programs_pool),
            'distinct_families_sampled': len({metadata(cases[i])['family'] for i in selected}),
            'families_in_pool': len(families_pool),
            'sampled_rows_by_generator': dict(collections.Counter(
                features[i]['generator'] for i, n in picked.items() for _ in range(n))),
            'sampled_effect_rows_by_generator': dict(collections.Counter(
                features[i]['generator'] for i, n in picked.items() if i in strict for _ in range(n))),
            'sampled_share_by_generator': {g: round(n / total, 4) for g, n in sorted(collections.Counter(
                features[i]['generator'] for i, n in picked.items() for _ in range(n)).items())},
            'sampled_share_by_family_top': _top_share(picked, [metadata(row)['family'] for row in cases], total),
            'sampled_share_by_program_top': _top_share(picked, [f['program'] for f in features], total),
            'sampled_share_by_external_source': {k: round(v / total, 4) for k, v in sorted(collections.Counter(
                s for i, n in picked.items()
                for s in [str(((cases[i].get('curriculum') or {}).get('external') or {}).get('source') or 'other')]
                for _ in range(n)).items())},
            'most_visited_rows': [{'id': cases[i]['id'][:90], 'visits': n, 'family': metadata(cases[i])['family'],
                                   'level': metadata(cases[i])['level'], 'generator': features[i]['generator']}
                                  for i, n in picked.most_common(8)],
            'visits_p99_one_row': sorted(picked.values())[int(.99 * (len(picked) - 1))],
            'max_visits_one_row': max(picked.values())}


def simulate_grpo(tasks, *, groups, group, wrapper_share, held_weight, explore, band=(0.1, 0.9), seed=1103):
    """``grpo.Collector._pick`` on a bare collector (no actor): synthetic outcomes, the real selection code."""
    import threading
    from types import SimpleNamespace
    import grpo
    from natlang_neuralese.rl.curriculum import metadata
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = SimpleNamespace(pass_rates=None, pass_band=band, pass_min_scored=2, explore=explore)
    c.tasks, c.by_id, c.lock = tasks, {row['id']: row for row in tasks}, threading.Lock()
    c.learnable_ids, c.learnable_order, c.measured, c.rates_read = [], [], set(), 0.
    c.base_rates, c.own, c.rates_changed, c.prior_stale = {}, {}, True, True
    c.rng, c.cursor = random.Random(seed), 0
    c.walk = grpo.ExploreWalk(tasks, wrapper_share, held_weight)
    features = c.walk.features or grpo.task_features(tasks)
    model, rng = OutcomeModel('latent', random.Random(seed + 1)), random.Random(seed + 2)
    shares = {'explored': collections.Counter(), 'all': collections.Counter()}
    spread = 0
    for _ in range(groups):
        taken = c.walk.counts['taken']
        row = c._pick()
        explored = c.walk.counts['taken'] > taken
        held, wrapper = features[row['id']]
        meta = metadata(row)
        p = model.probability(row['id'], meta['level'], meta['family'])
        wins = sum(rng.random() < p for _ in range(group))
        spread += 0 < wins < group
        model.learned(meta['family'])
        c.own[row['id']] = (wins, group)
        c.rates_changed = True
        for kind in ('explored', 'all') if explored else ('all',):
            counter = shares[kind]
            counter['groups'] += 1
            counter['dataset_wrapper'] += wrapper
            counter['declaring_held_construct'] += bool(set(held) - grpo.ExploreWalk.COMMON)
            for name in held:
                counter[name] += 1
    out = {kind: {'groups': counter['groups'], **{name: round(v / max(1, counter['groups']), 4)
                                                   for name, v in sorted(counter.items()) if name != 'groups'}}
           for kind, counter in shares.items()}
    out['groups_with_spread'] = round(spread / groups, 4)
    out['learnable_tasks_at_end'] = len(c.learnable_ids)
    return out


def grpo_report(args):
    from types import SimpleNamespace
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'neuralese_data'))
    import grpo
    tasks, excluded = grpo.load_tasks(SimpleNamespace(
        pool=args.cases or grpo.default_pools(), held_manifest=grpo.default_held(), exclude_manifest=[], task_ids=None,
        exclude_family_regex=args.exclude_family_regex, exclude_ids=[], no_default_errata=False, seed=args.seed,
        explore_order=args.explore_order))
    pool = grpo.ExploreWalk(tasks, args.wrapper_share, args.held_weight)
    report = {'schema': 'natlang.grpo-selection-simulation/1', 'pools': args.cases or grpo.default_pools(), 'eligible': len(tasks),
              'held_excluded': excluded, 'explore_order': args.explore_order, 'explore': args.explore, 'groups': args.groups,
              'pool': pool.pool_report(), 'outcomes': 'latent (OutcomeModel)'}
    for name, share, weight in (('before', 1., 1.), ('after', args.wrapper_share, args.held_weight)):
        report[name] = {'wrapper_share': share, 'held_weight': weight,
                        **simulate_grpo(tasks, groups=args.groups, group=args.group, wrapper_share=share,
                                        held_weight=weight, explore=args.explore)}
    Path(args.output).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', action='append', help='pool cases.jsonl (repeatable with --grpo)')
    parser.add_argument('--grpo', action='store_true', help="simulate grpo.py's collector selection instead")
    parser.add_argument('--groups', type=int, default=2000, help='--grpo: groups to select')
    parser.add_argument('--explore', type=float, default=0.4)
    parser.add_argument('--explore-order', choices=('row', 'family'), default='family')
    parser.add_argument('--exclude-family-regex', default='(?!)')
    parser.add_argument('--wrapper-share', type=float, default=0.15)
    parser.add_argument('--held-weight', type=float, default=3.)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--rounds', type=int, default=400)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--group', type=int, default=8)
    parser.add_argument('--compare-without-quota', action='store_true')
    parser.add_argument('--no-visit-decay', action='store_true',
                        help='disable the per-row visit decay (visit_penalty=0, visit_cap=0): the pre-2026-11-04 selector')
    parser.add_argument('--visit-penalty', type=float, help='override visit_penalty (levels per doubling of visits)')
    parser.add_argument('--visit-cap', type=int, help='override visit_cap')
    parser.add_argument('--visit-cap-window', type=int, help='override visit_cap_window (rounds)')
    parser.add_argument('--outcomes', choices=('level', 'latent'), default='level',
                        help='synthetic outcome model: per-level pass rate, or per-task latent rates with family learning')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args(argv)
    if args.grpo:
        return grpo_report(args)
    if not args.cases or len(args.cases) != 1:
        parser.error('--cases POOL/cases.jsonl (once) without --grpo')
    args.cases = args.cases[0]
    from natlang_neuralese.rl.curriculum import DEFAULT
    cases = [json.loads(line) for line in open(args.cases) if line.strip()]
    base = {**DEFAULT, 'visit_penalty': 0, 'visit_cap': 0} if args.no_visit_decay else DEFAULT
    base = {**base, **{key: value for key, value in (('visit_penalty', args.visit_penalty), ('visit_cap', args.visit_cap),
                                                      ('visit_cap_window', args.visit_cap_window)) if value is not None}}
    report = {'schema': 'natlang.rl-selection-simulation/1', 'cases': args.cases,
              'config': base, 'with_quota': simulate(cases, rounds=args.rounds, rows=args.rows, group=args.group, config=base, outcomes=args.outcomes)}
    if args.compare_without_quota:
        report['without_quota'] = simulate(cases, rounds=args.rounds, rows=args.rows, group=args.group,
                                           config={**base, 'class_quota': {}}, outcomes=args.outcomes)
    Path(args.output).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
