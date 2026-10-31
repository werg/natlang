#!/usr/bin/env python3
"""Diversity report over chain-graph outputs (build_chain_graphs.py receipts).

    chain_graph_diversity.py --out report.json DIR [DIR ...]

Streams every graphs-*.jsonl.zst named by each DIR/receipt.json and reports, per output and per family:
graphs, nodes, producer spans, imports, fan-in (imports per consuming node), distinct producer payloads
(exact token spans) and distinct supervised node targets, raw and prefix-shared token volume."""
import argparse
import collections
import hashlib
import io
import json
from pathlib import Path

import zstandard


def rows(path):
    with open(path, 'rb') as raw:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding='utf-8'):
            if line.strip():
                yield json.loads(line)


def digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()[:32]


def summarize(directory):
    receipt = json.loads((directory / 'receipt.json').read_text())
    per = collections.defaultdict(lambda: {'graphs': 0, 'nodes': 0, 'producer_spans': 0, 'imports': 0,
                                           'tokens': 0, 'supervised_tokens': 0, 'fan_in': collections.Counter(),
                                           'payloads': set(), 'targets': set()})
    for name, binding in receipt['files'].items():
        if not name.startswith('graphs-'):
            continue
        for graph in rows(binding['path']):
            conv = graph.get('chain_conversion') or {}
            key = (name.split('-', 1)[1], conv.get('family') or conv.get('source') or '?')
            stats = per[key]
            stats['graphs'] += 1
            for node in graph['nodes']:
                ids, weights = node['token_ids'], node['target_weights']
                stats['nodes'] += 1
                stats['tokens'] += len(ids)
                supervised = [i for i, w in zip(ids, weights) if w]
                stats['supervised_tokens'] += len(supervised)
                if supervised:
                    stats['targets'].add(digest(supervised))
                for start, stop in node['producing']:
                    stats['producer_spans'] += 1
                    stats['payloads'].add(digest(ids[start:stop]))
                if node['imports']:
                    stats['imports'] += len(node['imports'])
                    stats['fan_in'][len(node['imports'])] += 1
    out = {}
    for (split, family), stats in sorted(per.items()):
        fan = stats.pop('fan_in')
        stats['distinct_producer_payloads'] = len(stats.pop('payloads'))
        stats['distinct_supervised_targets'] = len(stats.pop('targets'))
        stats['consumer_nodes'] = sum(fan.values())
        stats['fan_in'] = {str(k): v for k, v in sorted(fan.items())}
        out.setdefault(split, {})[family] = stats
    totals = receipt['totals']
    return {'corpus_id': receipt['corpus_id'], 'chains': receipt['chains'], 'by_split_family': out,
            'prefix_shared_tokens': {k.split(':')[0]: v for k, v in totals.items() if k.endswith(':prefix_shared_tokens')},
            'omitted': {k: v for k, v in totals.items() if k.startswith('omitted:')},
            'prompts': {k: v for k, v in totals.items() if k.startswith('prompt:')},
            'distinct_targets_phase1': totals.get('distinct_targets'),
            'distinct_payloads_phase1': totals.get('distinct_payloads'),
            'distinct_payloads_kept': totals.get('distinct_payloads_kept')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dirs', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema': 'natlang.chain-graph-diversity/1', 'outputs': {str(d): summarize(d) for d in args.dirs}}
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({d: {s: sum(f['graphs'] for f in fam.values()) for s, fam in o['by_split_family'].items()}
                      for d, o in report['outputs'].items()}))


if __name__ == '__main__':
    main()
