"""`scripts/neuralese_data/lfm3_fast_train` (explicit fused layers, windowed attention, document-aware conv, fused linear
NLL) against the stock HF forward and backward on a tiny LFM3 twin whose window is shorter than the sequence (FP32, CPU:
the same math, not the same kernels)."""
import sys
from pathlib import Path

import pytest
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


def _layer_states(model, ids):
    """Each layer's normed pre-RoPE keys and values (attention) or B*x (conv) of a stock forward, per row."""
    states, handles = [None] * len(model.model.layers), []
    for index, layer in enumerate(model.model.layers):
        if layer.is_attention_layer:
            attn = layer.self_attn
            states[index] = {'kind': 'attention'}
            handles.append(attn.k_norm.register_forward_hook(
                lambda m, a, out, s=states[index]: s.__setitem__('raw_k', out[0].detach().clone())))
            handles.append(attn.v_proj.register_forward_hook(
                lambda m, a, out, s=states[index], d=attn.head_dim: s.__setitem__('v', out[0].detach().reshape(out.shape[1], -1, d).clone())))
        else:
            states[index] = {'kind': 'finite_convolution'}
            def bx(m, a, out, s=states[index]):
                b, _, x = out[0].detach().chunk(3, dim=-1)
                s['gated_inputs'] = (b * x).clone()
            handles.append(layer.conv.in_proj.register_forward_hook(bx))
    with torch.no_grad():
        model.model(input_ids=ids[None], use_cache=False)
    for handle in handles:
        handle.remove()
    return states


def test_imported_native_state_reproduces_the_producing_context():
    """A consumer whose rows [s, s + n) import the state the same model wrote there (the SGLang fork's splice: normed
    pre-RoPE K/V and conv B*x at those rows, the rows' own tokens irrelevant) computes exactly what the original sequence
    computes after the span, on the fast path and through the stock-path hooks, with matching gradients; without the
    import the placeholder tokens change it."""
    from types import SimpleNamespace
    import exp0_sft
    stock, fast = _pair()
    torch.manual_seed(5)
    ids = torch.randint(0, 60, (14,))
    start, count = 3, 6                               # longer than the window (4): later rows see the span only via conv/full attention
    states = _layer_states(stock, ids)
    block = SimpleNamespace(token_ids=ids[start:start + count],
                            layers=tuple({k: (v[start:start + count] if k != 'kind' else v) for k, v in s.items()} for s in states))
    consumer = ids.clone()
    consumer[start:start + count] = torch.randint(0, 60, (count,))
    after = slice(start + count, None)
    with torch.no_grad():
        expected = stock.model(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    imports = lfm3_fast_train.import_tensors(fast, [(start, block)], 'cpu')
    assert imports['rows'].tolist() == list(range(start, start + count))
    actual = lfm3_fast_train.hidden(fast, consumer, imports=imports)
    torch.testing.assert_close(actual[after], expected[after], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual[:start], expected[:start], rtol=1e-4, atol=1e-5)
    hooked = exp0_sft.hidden(stock, consumer, imports)
    torch.testing.assert_close(hooked[after], expected[after], rtol=1e-4, atol=1e-5)
    plain = lfm3_fast_train.hidden(fast, consumer)
    assert (plain[after] - expected[after]).abs().max() > 1e-2
    # gradients: the fast path and the stock hooks agree on the consumer's own rows (the state is a constant)
    weights = torch.randn_like(actual)
    weights[start:start + count] = 0
    for model, value in ((stock, hooked), (fast, actual)):
        model.zero_grad()
        (value * weights).sum().backward()
    _grads_close(stock, fast, 'imports')
    # packed: the consumer as the second document, its import rows offset by the first document's length
    first = torch.randint(0, 60, (5,))
    packed = lfm3_fast_train.hidden(fast, torch.cat([first, consumer]), lengths=[5, len(consumer)],
                                    imports=lfm3_fast_train.import_tensors(fast, [(5 + start, block)], 'cpu'))
    torch.testing.assert_close(packed[5:][after], expected[after], rtol=1e-4, atol=1e-5)


def test_chains_never_merge_across_different_imported_state():
    import numpy as np
    import exp0_sft
    a = np.arange(10, dtype=np.int32)
    row = lambda index, prompt, imports: {'invocation': 'i', 'index': index, 'prompt': prompt,
                                          'sample': np.array([50, 51], np.int32), 'imports': imports}
    full = np.concatenate([a, [50, 51]]).astype(np.int32)
    same = [row(0, a, [(2, 'x')]), row(1, np.concatenate([full, [7]]).astype(np.int32), [(2, 'x'), (12, 'y')])]
    chains = exp0_sft.chains_of([{'rows': same, 'task_id': 't', 'round': 0}])
    assert len(chains) == 1 and chains[0]['imports'] == [(2, 'x'), (12, 'y')]
    other = [row(0, a, [(2, 'x')]), row(1, np.concatenate([full, [7]]).astype(np.int32), [(2, 'z')])]
    chains = exp0_sft.chains_of([{'rows': other, 'task_id': 't', 'round': 0}])
    assert [c['imports'] for c in chains] == [[(2, 'x')], [(2, 'z')]]


def _dense_nucleus(h, weight, target, top_p):
    """Brute force: the full sort, SGLang's rule (a token is kept while the mass before it is <= top_p), ties with the
    last kept token and the target added; (nll, nucleus nll, outside) through plain autograd."""
    logits = F.linear(h, weight).float()
    logp = torch.log_softmax(logits, -1)
    values, order = logp.sort(dim=-1, descending=True)
    p = values.exp()
    count = ((p.cumsum(-1) - p) <= top_p).sum(-1)
    cut = values.gather(1, (count - 1)[:, None])
    onehot = F.one_hot(target, logits.shape[-1]).bool()
    kept = (logp >= cut) | onehot
    nucleus = torch.logsumexp(logits.masked_fill(~kept, float('-inf')), -1) - logits.gather(1, target[:, None])[:, 0]
    return -logp.gather(1, target[:, None])[:, 0], nucleus, ~(logp >= cut).gather(1, target[:, None])[:, 0]


def test_nucleus_nll_matches_brute_force_and_dense_autograd():
    torch.manual_seed(0)
    rows, width, vocab = 37, 12, 90
    weight = (torch.randn(vocab, width) * .9).requires_grad_()
    h = torch.randn(rows, width, requires_grad=True)
    target = torch.randint(0, vocab, (rows,))
    target[:6] = F.linear(h, weight).argmin(-1)[:6]            # sampled tokens outside the nucleus
    scale, scale_nucleus = torch.randn(rows), torch.randn(rows)
    for top_p in (.5, .95):
        want = _dense_nucleus(h, weight, target, top_p)
        want_grads = torch.autograd.grad((want[0] * scale).sum() + (want[1] * scale_nucleus).sum(), (h, weight))
        assert want[2][:6].all() and not want[2].all()
        for top_k in (4, 1024):                                # 4: most rows take the full-sort path
            got = lfm3_fast_train.linear_nll_nucleus(h, weight, target, top_p, chunk=16, top_k=top_k)
            assert torch.allclose(got[0], want[0], atol=1e-5) and torch.allclose(got[1], want[1], atol=1e-5)
            assert torch.equal(got[2], want[2]) and not got[2].requires_grad
            grads = torch.autograd.grad((got[0] * scale).sum() + (got[1] * scale_nucleus).sum(), (h, weight))
            assert torch.allclose(grads[0], want_grads[0], atol=1e-5) and torch.allclose(grads[1], want_grads[1], atol=1e-5)
        # the nucleus holds at least top_p of the mass, and the truncated log-probability is never below the full one
        mass = (got[1] - got[0]).exp()
        assert (mass[~got[2]] >= top_p - 1e-6).all() and (mass <= 1 + 1e-6).all()
    only = lfm3_fast_train.linear_nll_nucleus(h, weight, target, .95, chunk=16)[1]   # one output used: the other grad is None
    assert torch.allclose(torch.autograd.grad(only.sum(), h)[0],
                          torch.autograd.grad(_dense_nucleus(h, weight, target, .95)[1].sum(), h)[0], atol=1e-5)


def test_nucleus_ties_and_top_p_one_identity():
    weight = torch.eye(6)
    h = torch.tensor([[2., 1., 1., 1., -9., -9.]])            # three tied tokens straddle the cut: all are in
    nll, nucleus, outside = lfm3_fast_train.linear_nll_nucleus(h, weight, torch.tensor([3]), .6)
    assert not outside.item()
    assert nucleus.item() == pytest.approx(float(torch.logsumexp(h[0, :4], 0) - 1.), abs=1e-6)
    torch.manual_seed(1)
    h, weight, target = torch.randn(9, 5), torch.randn(20, 5), torch.randint(0, 20, (9,))
    plain = lfm3_fast_train.linear_nll(h, weight, target)
    assert torch.equal(plain, F.cross_entropy(F.linear(h, weight), target, reduction='none')) or \
        torch.allclose(plain, F.cross_entropy(F.linear(h, weight), target, reduction='none'), atol=1e-6)
    almost = lfm3_fast_train.linear_nll_nucleus(h, weight, target, 1 - 1e-7)   # the whole vocabulary: the same law
    assert torch.allclose(almost[0], plain, atol=1e-6) and torch.allclose(almost[1], plain, atol=1e-5)
    for bad in (1., 0., 1.5):
        with pytest.raises(ValueError):
            lfm3_fast_train.linear_nll_nucleus(h, weight, target, bad)
