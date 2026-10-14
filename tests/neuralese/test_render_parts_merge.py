"""render_parts.merge_parts keeps global split decontamination across parts (Pop 2026-10-12)."""
import json
import hashlib
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))
import render_parts  # noqa: E402


def write(root, part, tok, rows):
    path = root / 'parts' / part / 'program-header' / tok
    path.mkdir(parents=True, exist_ok=True)
    (path / 'text-data.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_completed_part_compression_preserves_exact_bytes_and_resumes(tmp_path):
    rows = [{'id': 'a', 'split': 'train', 'source_groups': ['g'], 'text': 'native context'}]
    write(tmp_path, 'p00', 'lfm25', rows)
    plain = tmp_path / 'parts/p00/program-header/lfm25/text-data.jsonl'
    before = plain.read_bytes()
    render_parts.compress_part_views(tmp_path / 'parts/p00')
    render_parts.compress_part_views(tmp_path / 'parts/p00')
    paths = render_parts.part_files(tmp_path, 'program-header', 'lfm25')
    assert len(paths) == 1 and paths[0].suffix == '.zst' and not plain.exists()
    assert b''.join(line for line, _ in render_parts.read_rows(paths[0])) == before


@pytest.mark.parametrize('compress_all', [False, True])
def test_cross_part_held_group_duplicate_and_twin_alignment(tmp_path, compress_all):
    for tok in ('lfm25', 'mellum21'):
        write(tmp_path, 'p00', tok, [
            {'id': 'a', 'split': 'train', 'source_groups': ['g1'], 'text': 'A' + tok},
            {'id': 'b', 'split': 'train', 'source_groups': ['g2'], 'text': 'B' + tok}])
        write(tmp_path, 'p01', tok, [
            {'id': 'c', 'split': 'test', 'source_groups': ['g1'], 'text': 'C' + tok},
            {'id': 'd', 'split': 'train', 'source_groups': ['g3'], 'text': 'B' + tok},
            {'id': 'e', 'split': 'train', 'source_groups': ['g4'], 'text': ('E' + tok) if tok == 'lfm25' else ('B' + tok)}])
    merged = render_parts.merge_parts(tmp_path, compress_all=compress_all)
    filename = 'text-data.jsonl.zst' if compress_all else 'text-data.jsonl'
    ids = {tok: [row['id'] for _, row in render_parts.read_rows(tmp_path / 'program-header' / tok / filename)]
           for tok in ('lfm25', 'mellum21')}
    # a: shares held group g1 with test doc c of another part; d: duplicate of b; e: duplicate only for mellum21,
    # so it is dropped for both twins.
    assert ids == {'lfm25': ['b', 'c'], 'mellum21': ['b', 'c']}
    assert merged['merged']['program-header/lfm25']['merge_omissions'] == {
        'exact_duplicate_document_across_parts': 2, 'train_document_shares_held_source_group_across_parts': 1}
    assert not list((tmp_path / 'parts').glob('*/*/*/text-data.jsonl'))
    for tok in ('lfm25', 'mellum21'):
        path = tmp_path / 'program-header' / tok / filename
        binding = merged['merged'][f'program-header/{tok}']
        assert binding['text_data_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        if compress_all:
            decoded = b''.join(line for line, _ in render_parts.read_rows(path))
            assert binding['decoded_text_data_sha256'] == hashlib.sha256(decoded).hexdigest()
            assert binding['decoded_text_data_bytes'] == len(decoded)
