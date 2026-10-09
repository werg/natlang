"""Admitted causal-feature curriculum; cached contexts are not new trajectories."""
import hashlib
import json
from pathlib import Path

from ..data.text_corpus import (_canonical, _soft_writer_sources,
                                authenticated_crisp_context_messages)
from ..serve.chat import render_messages


def context_ids(engine, records, pieces, limit, tokens):
    texts = {row['name']: row['text'] for row in map(json.loads, Path(pieces).open())}
    rows = list(map(json.loads, Path(records).open()))
    source_hashes = {row.get('id', ''): hashlib.sha256(
        _canonical(dict(row)).encode('utf-8')).hexdigest() for row in rows}
    writer_sources = _soft_writer_sources(rows, source_hashes)
    values = {'train': set(), 'test': set()}
    omitted = []
    for row in rows:
        split = row.get('split')
        if split not in values or row.get('training_admission', {}).get('approved') is not True:
            continue
        try:
            messages, _, _ = authenticated_crisp_context_messages(row, texts, writer_sources)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            omitted.append({'id': row.get('id'), 'source_record_sha256': source_hashes.get(row.get('id', '')),
                            'split': split,
                            'source_groups': sorted(set(g for g in (row.get('source_groups') or [])
                                                        if isinstance(g, str) and g)),
                            'reason': str(exc)[:240]})
            continue
        prompt = render_messages(messages, row.get('tools'), engine._template, engine.specials)
        if prompt.blocks:
            raise ValueError('crisp bootstrap context unexpectedly contains blocks')
        ids = tuple(token for segment in prompt.segments
                    for token in engine._template_tokens(segment, prompt.escape_nonce))[-tokens:]
        if ids:
            values[split].add(ids)
    overlap = values['train'] & values['test']
    values['test'] -= overlap
    key = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    receipt = {'policy': ('admitted crisp contexts with exact hash-bound Neuralese bodies, provider-read receipts, '
                          'tool-argument hydration and capture augmentation; unresolved rows are explicitly listed '
                          'and excluded from the context curriculum; tail windows are only for feedback distillation, '
                          'not a runtime context limit'),
               'excluded_shared_held_windows': len(overlap), 'unique_available': {k: len(v) for k, v in values.items()}}
    receipt['omitted_unrenderable_admitted_contexts'] = sorted(omitted, key=lambda row: row.get('id') or '')
    return {split: sorted(ids, key=key)[:limit] for split, ids in values.items()}, receipt
