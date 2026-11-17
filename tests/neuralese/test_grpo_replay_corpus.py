"""Fixed corpora trained in every GRPO step (`--replay-corpus`: operator rows, whole-graph Neuralese episodes)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402


def _chain(task, n=3, weight=None):
    row = {'kind': 'consumer', 'sample': np.arange(n, dtype=np.int32)}
    if weight is not None:
        row['weight'] = np.full(n, weight, np.float32)
    return {'ids': np.arange(5 + n, dtype=np.int32), 'segs': [(5, n, row)], 'imports': [], 'sources': {},
            'task': task, 'round': 0}


def test_units_are_deterministic_per_step_cover_each_epoch_and_drop_held(tmp_path):
    path = tmp_path / 'chains.pkl'
    chains = [_chain(f'case{i}') for i in range(5)] + [_chain('case0'), _chain('held-task')]
    path.write_bytes(pickle.dumps({'chains': chains}))
    corpus = grpo.ReplayCorpus(f'ops={path},weight=0.1,units=2', {'held-task'}, 7, log=lambda r: None)
    assert corpus.weight == 0.1 and corpus.units == 2 and len(corpus.keys) == 5
    first = [key for step in range(3) for key, _ in corpus.take(step)][:5]
    assert sorted(first) == [f'case{i}' for i in range(5)]           # one epoch visits every unit once
    assert [k for k, _ in corpus.take(1)] == [k for k, _ in grpo.ReplayCorpus(
        f'ops={path},weight=0.1,units=2', {'held-task'}, 7, log=lambda r: None).take(1)]   # a resume continues
    unit = dict(corpus.take(0))
    if 'case0' in unit:
        assert len(unit['case0']) == 2                                # a unit is every chain of its task key
    with pytest.raises(SystemExit):
        grpo.ReplayCorpus(f'ops={path},rate=1', set(), 7, log=lambda r: None)


def test_replay_chains_train_ratio_free_with_row_weights_and_no_tail_floor():
    import torch
    chain = dict(_chain('c', weight=0.5), role='replay')
    logp = torch.tensor([-9., -0.5, -2.])
    actor, weights = grpo.behaviour_logp(chain, logp)
    assert torch.equal(actor, logp) and weights.tolist() == [0.5, 0.5, 0.5]
    weight, tail = grpo.policy_weight((logp - actor).exp(), actor, 0.05, 2., None)
    assert weight.tolist() == [1., 1., 1.] and not tail.any()
