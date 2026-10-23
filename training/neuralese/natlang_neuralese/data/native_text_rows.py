"""Authenticated ordinary-text rows prepared for native model training and diagnostics."""
from __future__ import annotations

import hashlib
import json
from array import array
from pathlib import Path

from ..common.jsonio import open_jsonl_text


def load_native_text_rows(records, pieces=None, text_data=None, *, tokenizer=None):
    """Load source-disjoint text rows, validate token custody, and remove exact held duplicates."""
    if text_data:
        rows = []
        with open_jsonl_text(text_data) as source:
            for line in source:
                row = json.loads(line)
                if 'token_ids' in row:
                    ids = row['token_ids']
                    if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < 2**32 for i in ids):
                        raise ValueError('text token IDs must be unsigned integer coordinates')
                    row['token_ids'] = array('I', ids)
                rows.append(row)
        if any(r.get('split') not in ('train', 'test') or not isinstance(r.get('text'), str)
               or not r['text'].strip() or not r.get('source_groups') for r in rows):
            raise ValueError('text JSONL needs nonempty text, train/test split and source_groups')
        if any('sampling_group' in r and (not isinstance(r['sampling_group'], str) or not r['sampling_group'])
               for r in rows):
            raise ValueError('sampling_group must be a nonempty string when present')
    else:
        from .text_corpus import gold_text_rows
        records_path = Path(records)
        record_lines = [line for line in records_path.read_bytes().splitlines() if line]
        record_rows = [json.loads(line) for line in record_lines]
        for row, line in zip(record_rows, record_lines):
            row['_source_record_sha256'] = hashlib.sha256(line).hexdigest()
        pieces_path = Path(pieces) if pieces else records_path.parent / 'pieces.jsonl'
        piece_rows = list(map(json.loads, pieces_path.open())) if pieces_path.is_file() else []
        rows, _, _, _ = gold_text_rows(record_rows, piece_rows, tokenizer=tokenizer)
    if any(not isinstance(r.get('text_cohort', 'native'), str) or not r.get('text_cohort', 'native') for r in rows):
        raise ValueError('text_cohort must be a nonempty string')
    if any('supervised_suffix_start' in r and 'token_ids' not in r for r in rows):
        raise ValueError('supervised suffix requires native token IDs')
    encoded = [r for r in rows if 'token_ids' in r]
    if encoded:
        from .text_corpus import tokenizer_fingerprint
        if tokenizer is None:
            raise ValueError('native gold token IDs require the student tokenizer')
        fingerprint = tokenizer_fingerprint(tokenizer)
        vocab_size = len(tokenizer)
        for row in encoded:
            ids = row['token_ids']
            if row.get('tokenizer_sha256') != fingerprint or not isinstance(ids, (list, array)) or not ids or any(
                    type(i) is not int or i < 0 or i >= vocab_size for i in ids):
                raise ValueError('gold token IDs or tokenizer fingerprint mismatch')
            if 'supervised_suffix_start' in row and (type(row['supervised_suffix_start']) is not int or
                    not 0 <= row['supervised_suffix_start'] < len(ids)):
                raise ValueError('gold supervised suffix coordinate is invalid')
    groups = {split: set(group for row in rows if row['split'] == split for group in row['source_groups'])
              for split in ('train', 'test')}
    if groups['train'] & groups['test']:
        raise ValueError('text factual source groups cross train/test')
    held = {row['text'] for row in rows if row['split'] == 'test'}
    dedup, excluded = {}, 0
    for row in rows:
        if row['split'] == 'train' and row['text'] in held:
            excluded += 1
            continue
        ids_digest = hashlib.sha256(array('I', row.get('token_ids', [])).tobytes()).digest()
        key = (row['split'], row['text'], ids_digest, row.get('supervised_suffix_start'), row.get('sampling_group'))
        if key in dedup and dedup[key].get('text_cohort', 'native') != row.get('text_cohort', 'native'):
            raise ValueError('an identical document belongs to multiple cohorts; resolve ownership before training')
        dedup.setdefault(key, row)
    rows = list(dedup.values())
    if not all(any(row['split'] == split for row in rows) for split in ('train', 'test')):
        raise ValueError('nonempty independent train and held text required')
    return rows, {'policy': 'explicit source-disjoint ordinary text; gold targets, no teacher model',
                  'excluded_train_exact_held_duplicates': excluded, 'documents': len(rows)}
