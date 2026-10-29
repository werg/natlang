import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('controller_supply', ROOT / 'training/neuralese/learned-optimizer/controller-supply.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_controller_references_only_training_worlds_and_reject_conflicting_labels(tmp_path, monkeypatch):
    # Exercise real custody against a portable fixture, without reading the
    # workstation's protected production packets from a linked checkout.
    protection_root = tmp_path / 'protection'
    (protection_root / 'training').mkdir(parents=True)
    groups = protection_root / 'held.jsonl'
    groups.write_text(json.dumps({'source_groups': ['protected-world']}) + '\n')
    (protection_root / 'training/protected-worlds.json').write_text(json.dumps({
        'schema': 'natlang.protected-worlds-registry/1', 'sources': [{
            'kind': 'source-groups', 'id': 'fixture-held', 'path': str(groups),
            'sha256': hashlib.sha256(groups.read_bytes()).hexdigest()}]}))
    program = tmp_path / 'controller'
    split = program / '.natlang/tune/splits.json'
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps({'schema': 'natlang.tune-split/v1',
                                 'components': {'train-world': 'train', 'held-world': 'development'}}))
    train = {'group': 'train-world', 'function': 'adapt/shouldStop.nl', 'input': {'state': 17}, 'expected': True}
    held = {**train, 'group': 'held-world'}
    examples = tmp_path / 'examples.jsonl'
    examples.write_text('\n'.join(map(json.dumps, [train, train, held])) + '\n')
    monkeypatch.setattr(sys, 'argv', ['controller-supply', '--controller', str(program), '--examples', str(examples),
                                      '--root', str(protection_root)])
    MODULE.main()
    rows = [json.loads(line) for line in (program / 'tune.privileged.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['group'] == 'train-world'
    assert rows[0]['result'] is True and rows[0]['custody']['split'] == 'train'
    examples.write_text('\n'.join(map(json.dumps, [train, {**train, 'expected': False}])) + '\n')
    with pytest.raises(ValueError, match='conflicting exact controller labels'):
        MODULE.main()
    split.write_text(json.dumps({'schema': 'natlang.tune-split/v1',
                                 'components': {'protected-world': 'train'}}))
    examples.write_text(json.dumps({**train, 'group': 'protected-world'}) + '\n')
    with pytest.raises(ValueError, match='protected'):
        MODULE.main()
