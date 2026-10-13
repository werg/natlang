"""Apply a pinned mined-recovery selection to an existing rendered training row.

This changes sampling metadata only. It never changes text, token ids, masks,
source groups, or the train/held split.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached


def load_selection(config):
    if config is None:
        return None
    if (not isinstance(config, dict) or set(config) != {'path', 'sha256', 'rules', 'rendering'} or
            not isinstance(config['path'], str) or not isinstance(config['sha256'], str) or
            not isinstance(config['rules'], list) or not config['rules'] or
            any(rule not in ('identical', 'empty', 'failed') for rule in config['rules']) or
            config['rendering'] not in ('program-header', 'whole-trajectory', 'full')):
        raise ValueError('recovery_selection needs path, sha256, rendering and nonempty exact recovery rules')
    path = Path(config['path'])
    if sha256_file_hex_cached(path) != config['sha256']:
        raise ValueError('recovery selection manifest changed')
    rendering = config['rendering']
    selected = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('rendering') != rendering or not set(row.get('rules', ())) & set(config['rules']):
                continue
            key = row.get('id')
            split = row.get('split')
            if not isinstance(key, str) or not key or split not in ('train', 'test'):
                raise ValueError('invalid recovery selection id or split')
            previous = selected.setdefault(key, split)
            if previous != split:
                raise ValueError(f'recovery selection crosses split for {key}')
    if not selected:
        raise ValueError(f'no selected recovery rows for rendering {rendering}')
    return selected


def mark_row(row, selected):
    if selected is None or row['id'] not in selected:
        return row
    if row['split'] != selected[row['id']]:
        raise ValueError(f'recovery selection split mismatch: {row["id"]}')
    if row.get('text_cohort', 'native') not in ('native', 'recovery'):
        raise ValueError(f'recovery selection conflicts with existing cohort: {row["id"]}')
    return {**row, 'text_cohort': 'recovery'}
