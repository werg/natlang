"""LFM3 twin of a real registered verified-teacher package: same graph ids/splits/groups, exact target parse."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'qwen-verified-positive-native-graphs-20261017-recovery7853-batch0090-tranche0006'


def _available():
    from scripts.neuralese_data import teacher_graph_student_twin as twin
    entries = {e['id']: e for e in json.loads((ROOT / 'training/neuralese_corpora.json').read_text())['corpora']}
    if PACKAGE not in entries:
        return None
    root = twin.package_root(entries[PACKAGE])
    if not (root / 'lfm25' / 'receipt.json').exists():
        return None
    try:
        from scripts.neuralese_data.student_tokenizers import resolve
        resolve('lfm3')
    except SystemExit:
        return None
    return root


def test_lfm3_twin_matches_reference_rows(tmp_path):
    root = _available()
    if root is None:
        pytest.skip('registered package or pinned LFM3 tokenizer not on this machine')
    from scripts.neuralese_data import teacher_graph_student_twin as twin
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    receipt = twin.convert(PACKAGE, 'lfm3', tmp_path)
    assert receipt['verification']['passed'], receipt['verification']
    assert receipt['model_id'].startswith('lfm3:baf2f497')
    assert receipt['tokenizer']['role_token_ids'] == [125033, 125034, 125035, 125036]
    reference = [(g['id'], g['split'], g['source_groups'])
                 for _, g in iter_jsonl_rows(root / 'lfm25' / 'train' / 'native-graphs.jsonl.zst')]
    omitted = {o['id'] for o in receipt['omissions'] if 'reason' in o}
    produced = [(g['id'], g['split'], g['source_groups'])
                for _, g in iter_jsonl_rows(Path(receipt['outputs']['train']['path']))]
    assert produced == [r for r in reference if r[0] not in omitted]
    # Resumable: a finished package is returned from its receipt, not rebuilt.
    assert twin.convert(PACKAGE, 'lfm3', tmp_path) == receipt
