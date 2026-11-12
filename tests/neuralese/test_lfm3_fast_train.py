"""`scripts/neuralese_data/lfm3_fast_train` (explicit fused layers, windowed attention, document-aware conv, fused linear
NLL) against the stock HF forward and backward on a tiny LFM3 twin whose window is shorter than the sequence (FP32, CPU:
the same math, not the same kernels)."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
import lfm3_fast_train  # noqa: E402
from test_lfm3_port import tiny_lfm3_hf  # noqa: E402


def _pair():
    stock = tiny_lfm3_hf(window=4)
    fast = tiny_lfm3_hf(window=4)
    fast.load_state_dict(stock.state_dict())
    for model in (stock, fast):
        model.train()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    lfm3_fast_train.prepare(fast, compile_layers=False)
    return stock, fast


def _grads_close(stock, fast, label):
    for (name, a), (_, b) in zip(stock.named_parameters(), fast.named_parameters()):
        assert (b.grad - a.grad).norm() <= 1e-4 * a.grad.norm() + 1e-12, (label, name)


def test_fast_forward_and_gradients_match_stock():
    stock, fast = _pair()
    for length in (3, 8, 11):                       # inside one window, a whole number of windows, a partial last window
        ids = torch.randint(0, 60, (length,))
        expected = stock.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
        actual = lfm3_fast_train.hidden(fast, ids)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        weights = torch.randn_like(expected)
        for model, value in ((stock, expected), (fast, actual)):
            model.zero_grad()
            (value * weights).sum().backward()
        _grads_close(stock, fast, length)


def test_packed_documents_equal_separate_sequences():
    stock, fast = _pair()
    lengths = [5, 2, 9, 1, 6]                       # documents shorter and longer than the conv width and the window
    docs = [torch.randint(0, 60, (n,)) for n in lengths]
    expected = torch.cat([stock.model(input_ids=d[None], use_cache=False).last_hidden_state[0] for d in docs])
    actual = lfm3_fast_train.hidden(fast, torch.cat(docs), lengths=lengths)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    weights = torch.randn_like(expected)
    for model, value in ((stock, expected), (fast, actual)):
        model.zero_grad()
        (value * weights).sum().backward()
    _grads_close(stock, fast, 'packed')


def test_linear_nll_matches_cross_entropy():
    torch.manual_seed(0)
    h = torch.randn(13, 16, requires_grad=True)
    weight = torch.randn(40, 16, requires_grad=True)
    target = torch.randint(0, 40, (13,))
    upstream = torch.randn(13)
    expected = F.cross_entropy(F.linear(h, weight), target, reduction='none')
    (expected * upstream).sum().backward()
    gh, gw = h.grad.clone(), weight.grad.clone()
    h.grad = weight.grad = None
    actual = lfm3_fast_train.linear_nll(h, weight, target, chunk=5)
    torch.testing.assert_close(actual, expected)
    (actual * upstream).sum().backward()
    torch.testing.assert_close(h.grad, gh)
    torch.testing.assert_close(weight.grad, gw)


def test_exp0_packed_rows_give_per_chain_nll():
    import numpy as np
    import exp0_sft
    _, fast = _pair()
    rng = np.random.default_rng(0)
    chains = []
    for n in (6, 3, 9, 4):
        ids = rng.integers(0, 60, n)
        chains.append({'ids': ids, 'segs': [(n - 2, 2, {'sample': ids[n - 2:]})]})
    rows = exp0_sft.packed_rows(chains, 13)
    assert [len(r) for r in rows] == [2, 2]
    assert [len(r) for r in exp0_sft.packed_rows(chains, 0)] == [1, 1, 1, 1]
    expected = torch.cat([lfm3_fast_train.linear_nll(lfm3_fast_train.hidden(fast, ids)[pos], fast.lm_head.weight, tgt)
                          for ids, pos, tgt in (exp0_sft.chain_tensors(c, 'cpu') for c in chains)])
    actual = []
    for row in rows:
        ids, pos, tgt, weights, lengths = exp0_sft.row_tensors(row, 'cpu')
        actual.append(lfm3_fast_train.linear_nll(lfm3_fast_train.hidden(fast, ids, lengths)[pos], fast.lm_head.weight, tgt))
        assert torch.equal(weights, torch.ones(len(tgt)))
    torch.testing.assert_close(torch.cat(actual), expected, rtol=1e-5, atol=1e-6)
