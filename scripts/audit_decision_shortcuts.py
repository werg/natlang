#!/usr/bin/env python3
"""Trivial-feature baselines for decision families (scripts/build_decision_cases.py output).

A family is only a useful test of judgement if what a model must read cannot be replaced by something trivial. For
each family this fits, on the `train` role, and scores, on `heldout`:

- prior: the train label distribution, the same for every case (what a constant answer achieves);
- length: the label distribution within the case's state-length quintile (what reading the length achieves).

Scores are Brier (choice, yes/no) or ranked probability score (ordered levels), plus top-1 agreement with the gold
argmax. Flags mark families whose held-out labels a trivial feature already predicts, and families whose length
baseline beats the prior by a margin that means length carries label information. Read the flags with the family:
a skewed label prior is a property of the task (spam is rare), not a defect, but scores must be compared with it.

  python3 scripts/audit_decision_shortcuts.py --cases decision-cases.jsonl --out shortcuts.json
"""
import argparse
import json
from collections import defaultdict


def gold(case):
    if case['kind'] == 'choice':
        return [float(o == case['answer']) if isinstance(case['answer'], str) else float(case['answer'].get(o, 0))
                for o in case['options']]
    if case['kind'] == 'noul':
        p = float(case['answer'])
        return [p, 1 - p]
    levels = case['levels']
    x = levels.index(case['answer']) if isinstance(case['answer'], str) and case['answer'] in levels else float(case['answer'])
    out = [0.0] * len(levels)
    low = int(x)
    out[low] += 1 - (x - low)
    if x > low:
        out[low + 1] += x - low
    return out


def score(kind, predicted, target):
    top = float(max(range(len(predicted)), key=predicted.__getitem__) == max(range(len(target)), key=target.__getitem__))
    if kind == 'score':
        cp = cg = total = 0.0
        for p, g in list(zip(predicted, target))[:-1]:
            cp, cg = cp + p, cg + g
            total += (cp - cg) ** 2
        return {'rps': total / max(1, len(target) - 1), 'top1': top}
    return {'brier': sum((p - g) ** 2 for p, g in zip(predicted, target)), 'top1': top}


def mean_distribution(rows):
    total = [0.0] * len(rows[0])
    for row in rows:
        total = [a + b for a, b in zip(total, row)]
    return [v / len(rows) for v in total]


def audit(cases):
    train, held = [c for c in cases if c['role'] == 'train'], [c for c in cases if c['role'] == 'heldout']
    if not train or not held:
        return None
    kind = train[0]['kind']
    prior = mean_distribution([gold(c) for c in train])
    lengths = sorted(len(c['state']) for c in train)
    edges = [lengths[int(len(lengths) * q / 5)] for q in range(1, 5)]
    bucket = lambda c: sum(len(c['state']) > e for e in edges)
    by_bucket = defaultdict(list)
    for c in train:
        by_bucket[bucket(c)].append(gold(c))
    # A bucket's distribution is smoothed toward the prior so small buckets do not overfit.
    smoothed = {b: [(sum(r[i] for r in rows) + 5 * prior[i]) / (len(rows) + 5) for i in range(len(prior))]
                for b, rows in by_bucket.items()}
    results = {'prior': defaultdict(float), 'length': defaultdict(float)}
    for c in held:
        target = gold(c)
        for name, predicted in (('prior', prior), ('length', smoothed.get(bucket(c), prior))):
            for key, value in score(kind, predicted, target).items():
                results[name][key] += value / len(held)
    metric = 'rps' if kind == 'score' else 'brier'
    flags = []
    if results['prior']['top1'] >= 0.9:
        flags.append(f"a constant answer matches {results['prior']['top1']:.0%} of held-out labels")
    if results['prior'][metric] - results['length'][metric] >= 0.05:
        flags.append(f"state length alone improves {metric} by {results['prior'][metric] - results['length'][metric]:.3f} over the prior")
    return {'kind': kind, 'train': len(train), 'heldout': len(held), 'prior_distribution': [round(p, 4) for p in prior],
            'prior': {k: round(v, 4) for k, v in results['prior'].items()},
            'length': {k: round(v, 4) for k, v in results['length'].items()}, 'flags': flags}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    families = defaultdict(list)
    with open(args.cases) as stream:
        for line in stream:
            case = json.loads(line)
            families[case['family']].append(case)
    report = {family: audit(cases) for family, cases in sorted(families.items())}
    with open(args.out, 'x') as stream:
        json.dump({'schema': 'natlang.decision-shortcuts/1', 'cases': args.cases, 'families': report}, stream, indent=2)
        stream.write('\n')
    for family, row in report.items():
        if row:
            metric = 'rps' if row['kind'] == 'score' else 'brier'
            print(f"{family:32} prior {metric} {row['prior'][metric]:.3f} top1 {row['prior']['top1']:.2f} | "
                  f"length {metric} {row['length'][metric]:.3f}", '; '.join(row['flags']))


if __name__ == '__main__':
    main()
