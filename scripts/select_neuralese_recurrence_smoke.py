#!/usr/bin/env python3
"""Select recurrence stress readers while retaining every indexed producer.

This is a stress cohort, not a representative quality evaluation. Splits and
admission remain unchanged; selecting a reader never removes its producer data.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer
from natlang_neuralese.model.lfm2_port import resolve_base
from natlang_neuralese.serve.chat import render_messages
from natlang_neuralese.serve.engine import Engine
from natlang_neuralese.serve.grad import GradSession
from natlang_neuralese.train.trajectories import crisp_messages, handover_notes, reads, target_write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--pieces', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train', type=int, default=64)
    p.add_argument('--eval', type=int, default=24)
    p.add_argument('--depth', type=int, default=3)
    a = p.parse_args()
    if min(a.train, a.eval, a.depth) < 1:
        raise ValueError('positive cohort sizes and depth required')
    if a.out.exists():
        raise ValueError('use a fresh cohort output directory')
    rows = [json.loads(line) for line in a.records.open()]
    texts = {r['name']: r['text'] for r in map(json.loads, a.pieces.open())}
    tok = AutoTokenizer.from_pretrained(resolve_base(None), local_files_only=True)
    surface = SimpleNamespace(tokenizer=tok, backbone=None, heads=None)
    surface._template = lambda messages, tools: Engine._template(surface, messages, tools)
    surface._template_tokens = lambda text: Engine._template_tokens(surface, text)
    surface.specials = tuple(sorted(set(tok.all_special_tokens) | {'<|neuralese|>', '<|/neuralese|>'}))
    session = GradSession(surface)
    producers = {}
    for row in rows:
        name = target_write(row)
        if name:
            producers.setdefault(name, row)
    lengths = {}
    for row in rows:
        messages = crisp_messages(row['messages'], texts, handover_notes(row))
        rendered = render_messages(messages, row.get('tools'), surface._template, surface.specials)
        lengths[row['id']] = len(session._items(rendered.segments, rendered.blocks))
    def children(row, visiting=(), producer=False):
        names = (reads(row) | set(handover_notes(row))) & producers.keys()
        if producer:
            names.discard(target_write(row))
        return sorted(names - set(visiting))
    def measure(row):
        sites = []
        def visit(name, depth, visiting):
            producer = producers[name]
            below = children(producer, visiting + (name,), producer=True)
            sites.append((name, depth, lengths[producer['id']], len(below)))
            if depth < a.depth:
                for child in below:
                    visit(child, depth + 1, visiting + (name,))
        roots = children(row)
        for name in roots:
            visit(name, 1, ())
        return {'id': row['id'], 'split': row.get('split'), 'reader_tokens': lengths[row['id']],
                'largest_producer_tokens': max((s[2] for s in sites), default=0),
                'producer_tokens_sum': sum(s[2] for s in sites), 'write_sites': len(sites),
                'depth': max((s[1] for s in sites), default=0),
                'branching': max([len(roots)] + [s[3] for s in sites])}
    stats = [measure(row) for row in rows]
    def select(split, count):
        eligible = [s for s in stats if (s['split'] == 'test') == (split == 'test') and s['write_sites']]
        rankings = [[s['id'] for s in sorted(eligible, key=lambda x: (x[key], x['producer_tokens_sum'], x['id']), reverse=True)]
                    for key in ['largest_producer_tokens', 'write_sites', 'depth', 'branching']]
        chosen = []
        for i in range(len(eligible)):
            for ranking in rankings:
                if ranking[i] not in chosen:
                    chosen.append(ranking[i])
                    if len(chosen) == count:
                        return chosen
        return chosen
    selected_train, selected_held = select('train', a.train), select('test', a.eval)
    selected = selected_train + selected_held
    index = {r['id']: r for r in rows}
    ordered = [index[k] for k in selected] + [r for r in rows if r['id'] not in set(selected)]
    a.out.mkdir(parents=True)
    output = a.out / 'records.jsonl'
    with output.open('w') as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    report = {'purpose': 'recurrence-memory-stress-not-quality-evaluation', 'train': len(selected_train),
              'heldout': len(selected_held), 'depth_budget': a.depth, 'indexed_producers_retained': len(producers),
              'source_sha256': hashlib.sha256(a.records.read_bytes()).hexdigest(),
              'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'selected': [next(s for s in stats if s['id'] == k) for k in selected]}
    (a.out / 'selection.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'selected'}))
    print(json.dumps({'extremes': {key: max(s[key] for s in report['selected']) for key in
                                ['reader_tokens', 'largest_producer_tokens', 'write_sites', 'depth', 'branching']}}))


if __name__ == '__main__':
    main()
