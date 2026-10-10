"""A checkpoint written by the pre-skeleton trainer resumes on the shared training loop (frozen runs keep their artifacts)."""
import json
import shutil
import signal

import pytest
import torch

from test_text_warmup import tiny_student
from test_training_loop_golden import GOLDEN, LEGACY_RUN


def test_a_checkpoint_written_by_the_pre_loop_trainer_still_resumes(tmp_path, monkeypatch):
    """golden/legacy_text_warmup_run is the first four steps exactly as the pre-skeleton code wrote them."""
    from types import SimpleNamespace
    from natlang_neuralese.train import text_warmup

    def load(*_args):
        backbone, heads = tiny_student()
        return SimpleNamespace(backbone=backbone, heads=heads, tokenizer=None,
                               _tokens=lambda _text: [9, 3, 5, 8]), None

    monkeypatch.setattr(text_warmup, 'load_initial', load)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        run = tmp_path / 'run'
        shutil.copytree(LEGACY_RUN, run)
        heads_path = tmp_path / 'heads.pt'
        torch.save({}, heads_path)
        records = tmp_path / 'records.jsonl'
        records.write_text('')
        text = tmp_path / 'text.jsonl'
        text.write_text('\n'.join(json.dumps({'text': s, 'split': split, 'source_groups': [s]})
                                  for s, split in [('train', 'train'), ('held', 'test')]) + '\n')
        saved = torch.load(run / 'checkpoint.pt', weights_only=False)
        assert saved['step'] == 4
        # The fixture pins the absolute input paths of the run that wrote it: move it to this directory's inputs
        # (same bytes, so the same digests) and leave every other identity field exactly as the old code wrote it.
        moved = {'heads': heads_path, 'records': records, 'text_data': text}
        for key, path in moved.items():
            saved['identity']['options'][key] = str(path)
        by_name = {path.name: str(path.resolve()) for path in moved.values()}
        saved['identity']['inputs'] = {by_name[key.rsplit('/', 1)[1]]: digest
                                       for key, digest in saved['identity']['inputs'].items()}
        # Later recipe fields (the cohort sampler's declaration) are an identity change of newer code, not of the
        # skeleton: declare them at their defaults the way a continuation on that code would.
        saved['identity']['supervision_policy'] = {
            **text_warmup.text_supervision_policy(), 'context_weight': 1.0, 'feedback_weight': .25,
            'sampler': 'cohort-then-document-then-window/1', 'cohort_weights': None}
        torch.save(saved, run / 'checkpoint.pt')
        text_warmup.main(['--heads', str(heads_path), '--records', str(records), '--text-data', str(text),
            '--out', str(run), '--device', 'cpu', '--steps', '6', '--tokens', '8',
            '--prefix-tokens', '2', '--batch', '1', '--eval-batch', '1', '--held-documents', '1',
            '--eval-every', '1', '--checkpoint-every', '1', '--optimizer', 'adamw',
            '--backbone-training', 'full', '--projection-patience', '1', '--projection-min-evals', '2',
            '--projection-min-improvement', '1', '--backbone-ramp-evals', '1', '--pass-ramp-evals', '1'])
        golden = json.loads(GOLDEN.read_text())['resumed']
        after = torch.load(run / 'checkpoint.pt', weights_only=False)
        assert after['step'] == golden['checkpoint_step']
        rows = [json.loads(line) for line in (run / 'train.jsonl').read_text().splitlines()]
        assert [row['step'] for row in rows] == golden['steps_logged']
        assert [row['loss'] for row in rows[4:]] == pytest.approx(golden['losses'][4:], rel=1e-5)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
