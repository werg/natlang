#!/usr/bin/env python3
"""Graded skill episodes from real data with dense host-only scores. No model calls.

Sources and their scores (ts-host/src/skills/graded.ts):
  hotpot-answer   HotpotQA distractor questions with their ten paragraphs; answer token F1.
  hotpot-support  The same questions; rank the paragraph titles by usefulness, NDCG against the supporting titles.
  knights         Knights-and-knaves puzzles; fraction of inhabitants classified correctly.
  xlam            xLAM function-calling queries with their tool specs; call/argument F1.

Each episode has four support and four query items from one family, and four transfer items from the next family.
The last fifth (at least two) of each family's episodes is held out, alternating validation and test."""
import argparse
import glob
import hashlib
import json
import os

RAW = '/mnt/external/sdkb-archive/raw'


def digest(value):
    return hashlib.sha256((value if isinstance(value, str) else json.dumps(value, sort_keys=True)).encode()).hexdigest()


def target(name, args, returns, prompt):
    return {'kind': 'improvement-case', 'entry': 'solve.nl', 'exportName': 'default',
            'source': {'schema': 'natlang.skill-graded-target/1', 'id': name},
            'files': {'solve.nl': f'---\nargs: {args}\nreturns: {returns}\n---\n{prompt}\n'}}


def paragraphs(row):
    context = row['context']
    return [{'title': title, 'text': ' '.join(sentences)} for title, sentences in zip(context['title'], context['sentences'])]


def hotpot(kind):
    import pyarrow.parquet as pq
    rows = []
    for path in sorted(glob.glob(os.path.join(RAW, 'hotpotqa-distractor-*', 'train-*.parquet'))):
        rows += pq.read_table(path, columns=['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']).to_pylist()
    families = {}
    for row in rows:
        families.setdefault(f"{row['type']}-{row['level']}", []).append(row)
    if kind == 'hotpot-answer':
        spec = target('hotpot-answer-v1', '{ question: string, paragraphs: { title: string, text: string }[] }', 'string',
                      'Answer the question from the paragraphs, several of which are distractors. '
                      'Return only the answer: a short span, a name, a number, or yes or no.')
        item = lambda row: ([row['question'], paragraphs(row)], {'kind': 'gold-answer', 'value': row['answer']})
        metric = 'answer-token-f1'
    else:
        spec = target('hotpot-support-v1', '{ question: string, paragraphs: { title: string, text: string }[] }', 'string[]',
                      'Rank the paragraph titles from most to least useful for answering the question. '
                      'Return every title exactly once.')
        item = lambda row: ([row['question'], paragraphs(row)],
                            {'kind': 'relevant-set', 'items': sorted(set(row['supporting_facts']['title']))})
        metric = 'ranking-ndcg'
    return {f'qa:hotpot-{name}': (members, lambda row: row['id']) for name, members in sorted(families.items())}, spec, item, metric, \
        'CC-BY-SA-4.0 (HotpotQA)', 'hotpotqa-distractor'


def knights():
    families = {}
    for path in sorted(glob.glob(os.path.join(RAW, 'worlds-20260927', 'knights-and-knaves', 'train', 'people*_num*.jsonl'))):
        with open(path) as stream:
            for row in map(json.loads, stream):
                families.setdefault(f"logic:kk-people{len(row['names'])}", []).append(row)
    spec = target('knights-v1', '{ quiz: string, names: string[] }', 'Record<string, "knight" | "knave">',
                  'Solve the puzzle. Return an object that maps every inhabitant named in `names` to "knight" or "knave".')
    item = lambda row: ([row['quiz'], row['names']],
                        {'kind': 'assignment', 'value': {name: 'knight' if knight else 'knave' for name, knight in zip(row['names'], row['solution'])}})
    return {name: (members, lambda row: row['quiz']) for name, members in sorted(families.items())}, spec, item, \
        'assignment-accuracy', 'CC-BY-NC-SA-4.0 (Knights and Knaves)', 'knights-and-knaves'


def xlam():
    with open(os.path.join(RAW, 'agentic-20260927', 'xlam-function-calling-60k', 'xlam_function_calling_60k.json')) as stream:
        rows = json.load(stream)
    families = {}
    for row in rows:
        calls, tools = json.loads(row['answers']), json.loads(row['tools'])
        size = 'single' if len(calls) == 1 else 'parallel-same' if len({call['name'] for call in calls}) == 1 else 'multiple'
        families.setdefault(f"tools:xlam-{size}-{'one-tool' if len(tools) == 1 else 'choice'}", []).append(row)
    spec = target('xlam-v1', '{ query: string, tools: string }', '{ name: string, arguments: Record<string, unknown> }[]',
                  'Choose the function calls that answer the query, using only the tools described in `tools` (JSON). '
                  'Return the calls in order, each with the function name and its arguments.')
    item = lambda row: ([row['query'], row['tools']], {'kind': 'function-calls', 'calls': json.loads(row['answers'])})
    return {name: (members, lambda row: str(row['id'])) for name, members in sorted(families.items())}, spec, item, \
        'call-f1', 'CC-BY-4.0 (xLAM function calling 60k)', 'xlam-function-calling-60k'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=['hotpot-answer', 'hotpot-support', 'knights', 'xlam'])
    parser.add_argument('--out', required=True)
    parser.add_argument('--episodes-per-family', type=int, default=10)
    args = parser.parse_args()
    if args.source.startswith('hotpot'):
        families, spec, item, metric, license_, source = hotpot(args.source)
    else:
        families, spec, item, metric, license_, source = knights() if args.source == 'knights' else xlam()
    n_per = args.episodes_per_family
    names = [name for name, (members, _) in families.items() if len(members) >= 12 * n_per]
    if len(names) < 2:
        raise SystemExit('need at least two families with enough items: ' + json.dumps({k: len(v[0]) for k, v in families.items()}))
    pools = {}
    for name in names:
        members, key = families[name]
        pools[name] = (sorted(members, key=lambda row: digest(source + ':' + key(row))), key)

    def case(row, key):
        values, expected = item(row)
        ident = digest(f'{source}:{key(row)}')
        return {'id': 'case-' + ident[:20], 'group': 'g-' + digest(f'{source}-group:{key(row)}')[:24], 'args': values, 'expected': expected}

    episodes = []
    for index, name in enumerate(names):
        other = names[(index + 1) % len(names)]
        (own, key), (theirs, other_key) = pools[name], pools[other]
        for n in range(n_per):
            chosen = own[8 * n: 8 * n + 8]
            # Transfer items come from the other family's tail, which its own episodes never use.
            transfer = theirs[len(theirs) - 4 * (n + 1): len(theirs) - 4 * n]
            support, query = [case(r, key) for r in chosen[:4]], [case(r, key) for r in chosen[4:]]
            moved = [case(r, other_key) for r in transfer]
            groups = sorted(c['group'] for c in support + query + moved)
            held = max(2, n_per // 5)
            rank = n - (n_per - held)
            split = ('validation' if rank % 2 == 0 else 'test') if rank >= 0 else 'train'
            slug = name.split(':', 1)[1]
            graded = {'schema': 'natlang.skill-graded/1', 'kind': metric}
            episodes.append({'version': 'natlang.skill-episode/1', 'id': f'skill-{args.source}-{slug}-{n}', 'family': name, 'split': split,
                             'source_groups': ['group-commitment:sha256:' + digest(groups)], 'license': license_,
                             'target': spec, 'library': {'kind': 'empty', 'skills': {}},
                             'support': {'cases': support}, 'query': {'cases': query},
                             'transfer': {'family': other, 'target': spec, 'cases': moved},
                             'operations': ['create', 'revise', 'select', 'test'], 'limits': {'maxSteps': 6},
                             'provenance': {'generator': 'natlang.skill-graded-episodes/1', 'source': source, 'family_source': slug,
                                            'transfer_source': other.split(':', 1)[1], 'metric': graded, 'transfer_metric': graded}})
    # Cases must not repeat across episodes: transfer tails and episode heads are disjoint by construction.
    seen = [c['group'] for e in episodes for part in ('support', 'query') for c in e[part]['cases']]
    assert len(seen) == len(set(seen)), 'support/query groups repeat across episodes'
    os.makedirs(args.out, exist_ok=True)
    body = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in episodes)
    with open(os.path.join(args.out, f'{args.source}-episodes.jsonl'), 'x') as stream:
        stream.write(body)
    manifest = {'schema': 'natlang.skill-graded-episodes/1', 'source': args.source, 'metric': metric, 'episodes': len(episodes),
                'splits': {s: sum(1 for e in episodes if e['split'] == s) for s in ['train', 'validation', 'test']},
                'families': names, 'license': license_, 'model_calls': 0, 'sha256': digest(body)}
    with open(os.path.join(args.out, f'{args.source}-episodes.manifest.json'), 'x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
