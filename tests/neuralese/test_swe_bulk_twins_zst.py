"""SWE bulk twins: each shard's text is kept as one verified zstd copy; a twin is the concatenation of the frames."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))
pytestmark = pytest.mark.skipif(not shutil.which('zstd'), reason='needs zstd')


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_shard(root: Path, name: str, tokenizers) -> dict:
    out = root / 'render' / name
    receipt = {'renders': {}, 'selection': {'selected': 1}, 'renderer_code': {}, 'tokenizers': {t: t for t in tokenizers},
               'whole_trajectory': 'w'}
    for rendering in ('program-header', 'whole-trajectory'):
        for tok in tokenizers:
            text = f'{{"id": "{name}-{rendering}-{tok}"}}\n'.encode() * 3
            d = out / rendering / tok
            d.mkdir(parents=True)
            (d / 'text-data.jsonl').write_bytes(text)
            (d / 'omissions.jsonl').write_text('')
            receipt['renders'][f'{rendering}/{tok}'] = {
                'text_data_sha256': sha(text), 'by_split': {'train': {'documents': 3, 'tokens': 9}},
                'max_document_tokens': 3, 'merge_omissions': {}}
    for tok in tokenizers:
        (out / 'render-omissions' / tok).mkdir(parents=True)
        (out / 'render-omissions' / tok / 'omissions.jsonl').write_text('')
    (out / 'receipt.json').write_text(json.dumps(receipt))
    return receipt


def test_compact_then_assemble_keeps_one_copy(tmp_path):
    import swe_bulk_twins as twins
    tokenizers = {'lfm25': 'x', 'mellum21': 'y'}
    shards = [tmp_path / 'bulk' / n for n in ('00', '01')]
    receipts = {s.name: fake_shard(tmp_path, s.name, tokenizers) for s in shards}
    for s in shards:
        twins.compact_shard(tmp_path / 'render' / s.name, receipts[s.name], tokenizers)
        twins.compact_shard(tmp_path / 'render' / s.name, receipts[s.name], tokenizers)  # idempotent
        assert not list((tmp_path / 'render' / s.name).glob('*/*/text-data.jsonl'))
    twins.assemble(tmp_path, 'lfm25', 'program-header', shards, receipts)
    target = tmp_path / 'program-header' / 'lfm25'
    plain = subprocess.run(['zstd', '-dcq', str(target / 'text.jsonl.zst')], check=True, capture_output=True).stdout
    assert plain == b''.join(f'{{"id": "{n}-program-header-lfm25"}}\n'.encode() * 3 for n in ('00', '01'))
    summary = json.loads((target / 'receipt.json').read_text())
    assert summary['text_jsonl_sha256'] == sha(plain) and summary['text_file'] == 'text.jsonl.zst'
    assert summary['text_zst_sha256'] == sha((target / 'text.jsonl.zst').read_bytes())
    assert summary['tokens']['train']['documents'] == 6
    assert not (tmp_path / 'render' / '00' / 'program-header' / 'lfm25' / 'text-data.jsonl.zst').exists()
    assert (tmp_path / 'render' / '00' / 'whole-trajectory' / 'lfm25' / 'text-data.jsonl.zst').exists()
    twins.assemble(tmp_path, 'lfm25', 'program-header', shards, receipts)  # resumed: kept by its receipt


def test_compact_rejects_a_changed_text(tmp_path):
    import swe_bulk_twins as twins
    tokenizers = {'lfm25': 'x'}
    receipt = fake_shard(tmp_path, '00', tokenizers)
    (tmp_path / 'render' / '00' / 'program-header' / 'lfm25' / 'text-data.jsonl').write_text('changed\n')
    with pytest.raises(ValueError):
        twins.compact_shard(tmp_path / 'render' / '00', receipt, tokenizers)
