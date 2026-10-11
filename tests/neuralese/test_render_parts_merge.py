"""render_parts.merge_parts keeps global split decontamination across parts (Pop 2026-10-12)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))
import render_parts  # noqa: E402


def write(root, part, tok, rows):
    path = root / 'parts' / part / 'program-header' / tok
    path.mkdir(parents=True, exist_ok=True)
    (path / 'text-data.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_cross_part_held_group_duplicate_and_twin_alignment(tmp_path):
    for tok in ('lfm25', 'mellum21'):
        write(tmp_path, 'p00', tok, [
            {'id': 'a', 'split': 'train', 'source_groups': ['g1'], 'text': 'A' + tok},
            {'id': 'b', 'split': 'train', 'source_groups': ['g2'], 'text': 'B' + tok}])
        write(tmp_path, 'p01', tok, [
            {'id': 'c', 'split': 'test', 'source_groups': ['g1'], 'text': 'C' + tok},
            {'id': 'd', 'split': 'train', 'source_groups': ['g3'], 'text': 'B' + tok},
            {'id': 'e', 'split': 'train', 'source_groups': ['g4'], 'text': ('E' + tok) if tok == 'lfm25' else ('B' + tok)}])
    merged = render_parts.merge_parts(tmp_path)
    ids = {tok: [json.loads(line)['id'] for line in open(tmp_path / 'program-header' / tok / 'text-data.jsonl')]
           for tok in ('lfm25', 'mellum21')}
    # a: shares held group g1 with test doc c of another part; d: duplicate of b; e: duplicate only for mellum21,
    # so it is dropped for both twins.
    assert ids == {'lfm25': ['b', 'c'], 'mellum21': ['b', 'c']}
    assert merged['merged']['program-header/lfm25']['merge_omissions'] == {
        'exact_duplicate_document_across_parts': 2, 'train_document_shares_held_source_group_across_parts': 1}
    assert not list((tmp_path / 'parts').glob('*/*/*/text-data.jsonl'))
