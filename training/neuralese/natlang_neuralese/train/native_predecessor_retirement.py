"""Authenticated retirement of an already-applied source append argument."""
import json
from pathlib import Path

SCHEMA = 'natlang.native-source-predecessor-retirement/1'


def load_retirement(path):
    value = json.loads(Path(path).read_text())
    if value.get('schema') != SCHEMA or any(not isinstance(value.get(k), str) or len(value[k]) != 64 for k in ('checkpoint_sha256','train_sha256','held_sha256','predecessor_sha256')):
        raise ValueError('invalid source predecessor retirement proof')
    return value


def verify_retirement(training, proof, *, checkpoint_sha256, train_sha256, held_sha256):
    if proof['checkpoint_sha256'] != checkpoint_sha256 or proof['train_sha256'] != train_sha256 or proof['held_sha256'] != held_sha256:
        raise ValueError('source predecessor retirement checkpoint or source differs')
    receipts=training.get('source_receipts') or []
    if len(receipts)<2 or receipts[0].get('sha256')!=train_sha256 or receipts[1].get('sha256')!=held_sha256:
        raise ValueError('source predecessor retirement saved source differs')
    predecessors=(training.get('options') or {}).get('source_predecessors') or {}
    if predecessors.get('continued_native_graphs')!=proof['predecessor_sha256']:
        raise ValueError('source predecessor retirement lacks exact saved historical predecessor')
    return proof
