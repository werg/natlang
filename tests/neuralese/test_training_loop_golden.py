"""Golden parity for the shared training-loop skeleton (train/loop.py).

The tiny text warm-up below ran on the trainer BEFORE it moved onto train.loop; the committed numbers
(golden/text_warmup_tiny.json) are what that code produced. Regenerate only for an intended numerics change:
NATLANG_UPDATE_GOLDEN=1 pytest tests/neuralese/test_training_loop_golden.py. NATLANG_GOLDEN_EXACT=1 also compares the
bit-exact tensor digests, which hold on the machine that produced them.
"""
import hashlib
import json
import os
import signal
from pathlib import Path

import pytest
import torch

from test_text_warmup import tiny_student

GOLDEN = Path(__file__).parent / 'golden' / 'text_warmup_tiny.json'
LEGACY_RUN = Path(__file__).parent / 'golden' / 'legacy_text_warmup_run'


def _digest(tree):
    sha = hashlib.sha256()
    total, absolute, count = 0., 0., 0

    def visit(value):
        nonlocal total, absolute, count
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            sha.update(str(tensor.dtype).encode() + str(tuple(tensor.shape)).encode())
            sha.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b'')
            if tensor.is_floating_point() and tensor.numel():
                total += float(tensor.double().sum())
                absolute += float(tensor.double().abs().sum())
            count += tensor.numel()
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                sha.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            sha.update(repr(value).encode())

    visit(tree)
    return {'sha256': sha.hexdigest(), 'sum': total, 'abs_sum': absolute, 'elements': count}


def run_tiny_warmup(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.train import text_warmup

    def load(*_args):
        backbone, heads = tiny_student()
        return SimpleNamespace(backbone=backbone, heads=heads, tokenizer=None,
                               _tokens=lambda _text: [9, 3, 5, 8]), None

    monkeypatch.setattr(text_warmup, 'load_initial', load)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        heads_path = tmp_path / 'heads.pt'
        torch.save({}, heads_path)
        records = tmp_path / 'records.jsonl'
        records.write_text('')
        text = tmp_path / 'text.jsonl'
        text.write_text('\n'.join(json.dumps({'text': s, 'split': split, 'source_groups': [s]})
                                  for s, split in [('train', 'train'), ('held', 'test')]) + '\n')
        args = ['--heads', str(heads_path), '--records', str(records), '--text-data', str(text),
                '--out', str(tmp_path / 'run'), '--device', 'cpu', '--steps', '4', '--tokens', '8',
                '--prefix-tokens', '2', '--batch', '1', '--eval-batch', '1', '--held-documents', '1',
                '--eval-every', '1', '--checkpoint-every', '1', '--optimizer', 'adamw',
                '--backbone-training', 'full', '--projection-patience', '1', '--projection-min-evals', '2',
                '--projection-min-improvement', '1', '--backbone-ramp-evals', '1', '--pass-ramp-evals', '1']
        summary = {}
        for label, extra in (('first', []), ('resumed', ['--steps', '6'])):
            text_warmup.main(args + extra)
            run = tmp_path / 'run'
            rows = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
            saved = torch.load(run / 'checkpoint.pt', weights_only=False)
            if label == 'first' and os.environ.get('NATLANG_UPDATE_GOLDEN'):
                import shutil
                shutil.rmtree(LEGACY_RUN, ignore_errors=True)
                shutil.copytree(run, LEGACY_RUN, ignore=shutil.ignore_patterns('*.pending', '.checkpoint-space.reserve'))
            summary[label] = {
                'steps_logged': [row['step'] for row in rows],
                'phases': [row['phase'] for row in rows],
                'sequence_passes': [row['schedule']['sequence_passes'] for row in rows],
                'losses': [row['loss'] for row in rows],
                'checkpoint_step': saved['step'],
                'schedule': saved['schedule'],
                'updates': saved['updates'],
                'student_parameters': _digest(saved['student_parameters']),
                'heads': _digest(saved['heads']),
                'optimizer': _digest(saved['optimizer']),
                'rng': _digest([saved['torch_rng'], saved['python_rng']]),
                'export': _digest(torch.load(run / 'heads.pt', weights_only=False)['heads']),
            }
        return summary
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def test_text_warmup_matches_the_golden_run_through_a_resume(tmp_path, monkeypatch):
    summary = run_tiny_warmup(tmp_path, monkeypatch)
    if os.environ.get('NATLANG_UPDATE_GOLDEN'):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    golden = json.loads(GOLDEN.read_text())
    exact = bool(os.environ.get('NATLANG_GOLDEN_EXACT'))
    assert summary.keys() == golden.keys()
    for label in golden:
        got, want = summary[label], golden[label]
        for key in ('steps_logged', 'phases', 'sequence_passes', 'checkpoint_step', 'schedule', 'updates'):
            assert got[key] == want[key], (label, key)
        assert got['losses'] == pytest.approx(want['losses'], rel=1e-5)
        for key in ('student_parameters', 'heads', 'optimizer', 'rng', 'export'):
            assert got[key]['elements'] == want[key]['elements'], (label, key)
            assert got[key]['sum'] == pytest.approx(want[key]['sum'], rel=1e-4, abs=1e-5), (label, key)
            assert got[key]['abs_sum'] == pytest.approx(want[key]['abs_sum'], rel=1e-4, abs=1e-5), (label, key)
            if exact:
                assert got[key]['sha256'] == want[key]['sha256'], (label, key)
