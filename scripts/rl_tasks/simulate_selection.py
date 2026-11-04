#!/usr/bin/env python3
"""Simulate the RL task selector over a pool and measure the mixture it actually samples.

  simulate_selection.py --cases POOL/cases.jsonl [--rounds 400] [--rows 16] [--group 8] [--compare-without-quota] -o report.json

Runs ``production.select_tasks`` (the interleaved controller's own entry, policy ``natlang.rl-task-curriculum/2`` with its
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


def _top_share(picked, keys, total, top=8):
    """The ``top`` largest groups (family or program) by share of all sampled rows."""
    groups = collections.Counter()
    for index, count in picked.items():
        groups[keys[index]] += count
    return {str(key)[:80]: round(count / total, 4) for key, count in groups.most_common(top)}


def simulate(cases, *, rounds, rows, group, config, seed=1103):
    from natlang_neuralese.rl import production
    from natlang_neuralese.rl.curriculum import metadata
    rng = random.Random(seed)
    features = [program_features(row) for row in cases]
    history, offset = {}, 0
    picked = collections.Counter()
    per_round = []
    tagged_disagree = 0
    for number in range(rounds):
        stats = list(history.values())
        chosen = production.select_tasks(cases, stats, {'rows': rows, 'group': group, 'offset': offset}, curriculum=config)
        effect = sum(1 for item in chosen if features[item['row']]['effect_state_oracle'])
        per_round.append(effect / len(chosen))
        for item in chosen:
            row = cases[item['row']]
            picked[item['row']] += 1
            level = metadata(row)['level']
            p = PASS_BY_LEVEL.get(level, .25)
            wins = sum(rng.random() < p for _ in range(item['samples']))
            history[row['id']] = {'task_id': row['id'], 'scored': item['samples'], 'mean': wins / item['samples'],
                                  'spread': 0 < wins < item['samples'], 'observed_round': number}
        offset = (chosen[-1]['row'] + 1) % len(cases)
    tagged = {i for i, row in enumerate(cases) if metadata(row).get('mix_class') == 'effect_state'}
    strict = {i for i, f in enumerate(features) if f['effect_state_oracle']}
    selected = set(picked)
    programs_pool = {f['functional'] for f in features}
    families_pool = {metadata(row)['family'] for row in cases}
    total = sum(picked.values())
    per_round_sorted = sorted(per_round)
    return {'rounds': rounds, 'rows_per_round': rows, 'group': group, 'pool_rows': len(cases),
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
            'visits_p99_one_row': sorted(picked.values())[int(.99 * (len(picked) - 1))],
            'max_visits_one_row': max(picked.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--rounds', type=int, default=400)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--group', type=int, default=8)
    parser.add_argument('--compare-without-quota', action='store_true')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args(argv)
    from natlang_neuralese.rl.curriculum import DEFAULT
    cases = [json.loads(line) for line in open(args.cases) if line.strip()]
    report = {'schema': 'natlang.rl-selection-simulation/1', 'cases': args.cases,
              'config': DEFAULT, 'with_quota': simulate(cases, rounds=args.rounds, rows=args.rows, group=args.group, config=DEFAULT)}
    if args.compare_without_quota:
        report['without_quota'] = simulate(cases, rounds=args.rounds, rows=args.rows, group=args.group,
                                           config={**DEFAULT, 'class_quota': {}})
    Path(args.output).write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report))
    return 0


if __name__ == '__main__':
    sys.exit(main())
