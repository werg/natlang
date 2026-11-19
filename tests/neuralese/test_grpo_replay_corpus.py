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


class _Tokens:
    """A tokenizer stand-in: ids index a fixed vocabulary."""
    vocab = ['</think>', '\n\n', 'answer', '<|im_end|>', 'plan']

    def convert_tokens_to_ids(self, token):
        return self.vocab.index(token)

    def decode(self, ids):
        return ''.join(self.vocab[i] for i in ids)


def _rows_chain(task, samples, weight=None):
    segs, start = [], 4
    for sample in samples:
        row = {'kind': 'consumer', 'sample': np.asarray(sample, np.int32)}
        if weight is not None:
            row['weight'] = np.full(len(sample), weight, np.float32)
        segs.append((start, len(sample), row))
        start += 4 + len(sample)
    return {'ids': np.zeros(start, np.int32), 'segs': segs, 'imports': [], 'sources': {}, 'task': task, 'round': 0}


def test_a_leading_empty_think_block_is_context_and_reasoned_replies_keep_every_token(tmp_path):
    import exp0_sft
    blank = lambda token: token == 1                                       # noqa: E731
    assert exp0_sft.empty_think_lead(np.array([0, 1, 2, 3]), 0, blank) == 2   # </think> \n\n | answer <|im_end|>
    assert exp0_sft.empty_think_lead(np.array([4, 0, 2, 3]), 0, blank) == 0   # reasoning first: nothing masked
    assert exp0_sft.empty_think_lead(np.array([], np.int32), 0, blank) == 0
    path = tmp_path / 'chains.pkl'
    chains = [_rows_chain('a', [[0, 1, 2, 3], [4, 0, 2, 3]]), _rows_chain('b', [[0, 2, 3]], weight=0.5)]
    path.write_bytes(pickle.dumps({'chains': chains}))
    corpus = grpo.ReplayCorpus(f'ops={path}', set(), 7, log=lambda r: None, tokenizer=_Tokens())
    assert corpus.think == 'context' and corpus.think_masked == 2
    (_, _, first), (_, _, reasoned) = corpus.groups['a'][0]['segs']
    assert first['weight'].tolist() == [0, 0, 1, 1] and 'weight' not in reasoned
    (_, _, seeded), = corpus.groups['b'][0]['segs']
    assert seeded['weight'].tolist() == [0, .5, .5]                       # the row's own weight stays on the action
    # the weights reach the loss as for every chain
    assert exp0_sft.chain_weights(corpus.groups['a'][0], 'cpu').tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    kept = grpo.ReplayCorpus(f'ops={path},think=train', set(), 7, log=lambda r: None, tokenizer=_Tokens())
    assert kept.think_masked == 0 and 'weight' not in kept.groups['a'][0]['segs'][0][2]
    with pytest.raises(SystemExit):
        grpo.ReplayCorpus(f'ops={path},think=maybe', set(), 7, log=lambda r: None)


def test_student_sampled_targets_split_teacher_and_corpus_data_from_the_students_own_samples():
    def chain(role, round_name='rl:x', rrl=None):
        rows = [{'sample': np.zeros(2, np.int32), **({'teacher': True} if r == 'teacher-row' else {'rrl': r} if r else {})}
                for r in (rrl or [None])]
        return {'ids': np.zeros(8, np.int32), 'segs': [(1 + 2 * i, 2, row) for i, row in enumerate(rows)],
                'task': 't', 'round': round_name, 'role': role}
    sampled = grpo.student_sampled_targets
    assert sampled(chain('student')).tolist() == [True, True]
    assert sampled(chain('sd')).tolist() == [True, True]
    assert sampled(chain('teacher')).tolist() == [True, True]              # JOLT+ attempts: the actor sampled them
    assert sampled(chain('sd_teacher', 'teacher:pool')).tolist() == [False, False]
    assert sampled(chain('replay', 'rl-neuralese-graph:e')).tolist() == [True, True]   # own graph rows
    assert sampled(chain('replay', '0')).tolist() == [False, False]                   # operator seeds and answers
    rrl = chain(grpo.RRL_ROLE, rrl=['reasoning', 'action'])
    assert sampled(rrl).tolist() == [True, True, False, False]           # the teacher action after the student's reasoning
    carried = chain('sd', 'rrl-carried-x', rrl=[None, 'teacher-row'])     # carried search chain: reasoning, teacher action
    assert sampled(carried).tolist() == [True, True, False, False]


def test_save_step_chains_keeps_the_exact_batch_with_its_loss_factors_and_only_the_newest_steps(tmp_path):
    import pickle
    import zstandard
    chain = {'ids': np.arange(6, dtype=np.int32), 'segs': [(2, 3, {'sample': np.arange(3, dtype=np.int32),
                                                                   'logp': np.full(3, -0.5, np.float32)})],
             'advantage': -0.7, 'episode': 0, 'role': 'student'}
    for step in (1, 2, 3):
        grpo.save_step_chains(tmp_path, step, [chain], {0: (1 / 300, 0.)}, keep=2)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['step-00002.pkl.zst', 'step-00003.pkl.zst']
    record = pickle.loads(zstandard.ZstdDecompressor().decompress((tmp_path / 'step-00003.pkl.zst').read_bytes()))
    saved = record['chains'][0]
    assert record['step'] == 3 and saved['token_scale'] == 1 / 300 and saved['advantage'] == -0.7
    assert saved['segs'][0][2]['logp'].tolist() == [-0.5] * 3 and 'token_scale' not in chain
