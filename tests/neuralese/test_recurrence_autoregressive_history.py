import hashlib
import json

import pytest

from natlang_neuralese.eval.recurrence_autoregressive_history import (
    _load_selected_text_rows,
    token_ids_sha256,
    validate_window_selection,
)
from natlang_neuralese.train.text_warmup import ROLE_CODES


def _row(row_id='held-1', *, split='test', groups=None, text='a held document'):
    return {
        'id': row_id,
        'split': split,
        'text': text,
        'source_groups': groups or [row_id],
        'token_ids': list(range(10)),
    }


def _selection(row):
    prefix = [99, *row['token_ids'][3:8]]
    target = row['token_ids'][8:10]
    return {
        'schema': 'natlang.matched-ar-window-selection/1',
        'source_recipe_sha256': 'recipe-sha',
        'text_data_sha256': 'text-sha',
        'split': 'test',
        'max_context_tokens': 8,
        'expected_cells': [
            {'kind': 'prose', 'target_tokens': 2},
            {'kind': 'code', 'target_tokens': 3},
        ],
        'windows': [{
            'window_id': 'window-1', 'context_id': 'context-1',
            'stratum': {'kind': 'prose', 'target_tokens': 2},
            'row_id': row['id'], 'split': 'test',
            'document_sha256': hashlib.sha256(row['text'].encode()).hexdigest(),
            'source_groups': row['source_groups'],
            'prefix_start': 3, 'target_start': 8, 'target_end': 10,
            'prefix_token_ids_sha256': token_ids_sha256(prefix),
            'target_token_ids_sha256': token_ids_sha256(target),
        }],
        'unavailable_cells': [{'stratum': {'kind': 'code', 'target_tokens': 3},
                               'reason': 'No independently grouped held code window of this length.'}],
    }


def test_selection_validates_exact_window_hashes_roles_and_unavailable_cells():
    row = _row()
    roles = {row['id']: [ROLE_CODES.index('assistant_reply')] * len(row['token_ids'])}
    actual = validate_window_selection(_selection(row), [row], roles,
        source_recipe_sha256='recipe-sha', text_data_sha256='text-sha',
        context_limit=8, open_id=99)
    assert len(actual) == 1
    assert actual[0]['prefix_token_ids'] == [99, 3, 4, 5, 6, 7]
    assert actual[0]['target_token_ids'] == [8, 9]


@pytest.mark.parametrize('mutation', ['wrong_target_hash', 'user_target', 'wrong_prefix', 'missing_unavailable'])
def test_selection_rejects_mismatched_or_unaccounted_window(mutation):
    row = _row()
    manifest = _selection(row)
    roles = {row['id']: [ROLE_CODES.index('assistant_reply')] * len(row['token_ids'])}
    if mutation == 'wrong_target_hash':
        manifest['windows'][0]['target_token_ids_sha256'] = '0' * 64
    elif mutation == 'user_target':
        roles[row['id']][8] = ROLE_CODES.index('user')
    elif mutation == 'wrong_prefix':
        manifest['windows'][0]['prefix_start'] = 2
    else:
        manifest['unavailable_cells'] = []
    with pytest.raises(ValueError):
        validate_window_selection(manifest, [row], roles,
            source_recipe_sha256='recipe-sha', text_data_sha256='text-sha',
            context_limit=8, open_id=99)


def test_streaming_reader_retains_only_selected_rows_and_checks_split_groups(tmp_path):
    rows = [_row('train-1', split='train'), _row('held-1')]
    path = tmp_path / 'corpus.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    selected = _load_selected_text_rows(path, ['held-1'])
    assert list(selected) == ['held-1']
    assert selected['held-1']['token_ids'] == list(range(10))

    rows[0]['source_groups'] = rows[1]['source_groups']
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='cross train/test'):
        _load_selected_text_rows(path, ['held-1'])


def test_streaming_reader_rejects_missing_or_duplicate_selected_source(tmp_path):
    path = tmp_path / 'corpus.jsonl'
    row = _row()
    path.write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='duplicate'):
        _load_selected_text_rows(path, ['held-1'])

    path.write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='missing'):
        _load_selected_text_rows(path, ['not-present'])
