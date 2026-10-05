"""One-forward corrective-prefix SFT and a checkpoint-gated repair outbox.

Boundaries are complete native assistant actions, never arbitrary token cutoffs.
The outbox contains references to immutable training rows, not admitted repair data.
"""
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def gated_loss(model, encoded, action_ends, *, mean_nll, token_nll, full_gold=None):
    import torch
    labels = encoded['labels'][:, 1:]
    logits = model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'],
                   logits_to_keep=encoded['labels'].shape[-1]).logits[:, :-1].float()
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1), ignore_index=-100, reduction='none').reshape_as(labels)
    valid = labels != -100
    if not torch.isfinite(losses[valid]).all():
        raise ValueError('non-finite online-repair token loss')
    keep = valid.clone()
    receipts = []
    full_gold = full_gold or [False] * len(action_ends)
    if len(full_gold) != len(action_ends):
        raise ValueError("full-gold flags must match batch")
    # One detached transfer per microbatch. No extra model forward / scoring graph.
    observed = losses.detach().cpu().tolist()
    valid_cpu = valid.cpu().tolist()
    for i, ends in enumerate(action_ends):
        positions = [j for j, yes in enumerate(valid_cpu[i]) if yes]
        if not positions or not ends or ends[-1] != positions[-1] + 1:
            raise ValueError('action boundaries must cover all supervised targets')
        if any(not isinstance(e, int) for e in ends) or any(a >= b for a,b in zip([0]+ends[:-1],ends)):
            raise ValueError('invalid action boundaries')
        total = math.fsum(observed[i][j] for j in positions) / len(positions)
        hard = next((j for j in positions if observed[i][j] > token_nll), None)
        # If only aggregate NLL is high, choose first action over the mean gate.
        previous = 0
        if hard is None and total > mean_nll:
            for end in ends:
                span = [j for j in positions if previous <= j < end]
                if span and math.fsum(observed[i][j] for j in span)/len(span) > mean_nll:
                    hard = span[0]; break
                previous = end
        cutoff = next(e for e in ends if e > hard) if hard is not None else None
        if cutoff is not None and not full_gold[i]:
            keep[i, cutoff:] = False
        receipts.append({'mean_nll': total, 'hard_token': hard,
                         'cutoff': cutoff, 'cutoff_action':ends.index(cutoff) if cutoff is not None else None, 'full_gold':bool(full_gold[i]), 'supervised_tokens':len(positions),
                         'retained_tokens':sum(yes and (full_gold[i] or cutoff is None or j < cutoff)
                                               for j,yes in enumerate(valid_cpu[i]))})
    # Detached thresholds; gradient only through retained gold tokens.
    return ((losses * keep).sum(1) / keep.sum(1)).mean(), receipts


class RepairOutbox:
    """Idempotent observations; consumers may only export checkpoint-committed steps."""
    def __init__(self, path, identity):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS identity (value TEXT NOT NULL)')
        value = json.dumps(identity, sort_keys=True)
        old = self.db.execute('SELECT value FROM identity').fetchone()
        if old and old[0] != value:
            raise ValueError('repair outbox identity mismatch')
        if not old:self.db.execute('INSERT INTO identity VALUES (?)', (value,))
        self.db.execute('CREATE TABLE IF NOT EXISTS observations (key TEXT PRIMARY KEY, step INTEGER, row_offset INTEGER, receipt TEXT)')
        self.db.commit()

    def record_step(self, step, observations):
        with self.db:
            for offset, receipt in observations:
                key = hashlib.sha256(f'{step}:{offset}'.encode()).hexdigest()
                self.db.execute('INSERT OR REPLACE INTO observations VALUES (?,?,?,?)',
                                (key,step,offset,json.dumps(receipt,sort_keys=True)))

    def export(self, checkpoint_step):
        return [dict(key=key, step=step, row_offset=offset, **json.loads(receipt))
                for key,step,offset,receipt in self.db.execute(
                    'SELECT key,step,row_offset,receipt FROM observations WHERE step<=? ORDER BY step,row_offset',
                    (checkpoint_step,))]
