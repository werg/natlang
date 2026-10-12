"""Union: document ids are unique per rendering (first member wins), .zst members stream, member files are bound."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))
LFM = ROOT / 'models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f'

pytestmark = pytest.mark.skipif(not LFM.exists() or not shutil.which('zstd'), reason='needs the LFM2.5 tokenizer and zstd')


def doc(rid, text, split='train', group=None):
    return {'id': rid, 'split': split, 'source_groups': [group or f'g-{rid}'], 'text': text, 'token_ids': [1, 2, 3],
            'supervised_suffix_start': 1}


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def test_duplicate_ids_dropped_from_later_member_and_zst_streamed(tmp_path):
    import build_native_turn_union as union
    a_rows = [doc('a', 'A'), doc('b', 'B from first member')]
    b_rows = [doc('b', 'B rendered differently'), doc('c', 'C')]
    members = []
    for tok in ('lfm25', 'mellum21'):
        a = write(tmp_path / 'a' / tok / 'text-data.jsonl', a_rows)
        b = write(tmp_path / 'b' / tok / 'text-data.jsonl', b_rows)
        subprocess.run(['zstd', '-q', '--long=27', '--rm', str(b), '-o', str(b) + '.zst'], check=True)
        members += ['--member', f'first:program-header:{tok}={a}', '--member', f'second:program-header:{tok}={b}.zst']
    out = tmp_path / 'union'
    union.main(['--out-dir', str(out), '--tokenizer', f'lfm25={LFM}', '--tokenizer', f'mellum21={LFM}', *members])
    for tok in ('lfm25', 'mellum21'):
        got = [json.loads(line) for line in (out / 'program-header' / tok / 'text-data.jsonl').read_text().splitlines()]
        assert [(r['id'], r['text']) for r in got] == [('a', 'A'), ('b', 'B from first member'), ('c', 'C')]
    omissions = [json.loads(line) for line in (out / 'program-header' / 'omissions.jsonl').read_text().splitlines()]
    assert omissions == [{'id': 'b', 'member': 'second', 'reason': 'duplicate_document_id (first in first)'}]
    receipt = json.loads((out / 'receipt.json').read_text())
    bound = receipt['member_files']['program-header']['second']['lfm25']
    packed = Path(bound['path'])
    assert packed.suffix == '.zst' and bound['sha256'] == hashlib.sha256(packed.read_bytes()).hexdigest()
