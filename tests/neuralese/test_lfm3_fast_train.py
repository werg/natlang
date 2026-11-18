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
        for top_k, wide_k in ((4, 8), (4, 16384), (1024, 16384)):   # 4: most rows take the wide paths (8: the full sort)
            lfm3_fast_train.NUCLEUS_WIDE_K, saved = wide_k, lfm3_fast_train.NUCLEUS_WIDE_K
            try:
                got = lfm3_fast_train.linear_nll_nucleus(h, weight, target, top_p, chunk=16, top_k=top_k)
            finally:
                lfm3_fast_train.NUCLEUS_WIDE_K = saved
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel check')
def test_nucleus_gradient_kernel_casts_inside_bitwise():
    """The compiled gradient kernel takes the BF16 logits and returns the BF16 gradient: bitwise what the fp32 kernel
    followed by an eager cast gave (the casts moved inside only to keep the chunk's fp32 tensors off the peak)."""
    def fp32_kernel(lf, target, lse, set_lse, cut, grad, grad_nucleus):
        onehot = torch.arange(lf.shape[-1], device=lf.device)[None, :] == target[:, None]
        kept = (lf >= cut[:, None]) | onehot
        d = torch.exp(lf - lse[:, None]) * grad[:, None] + torch.where(kept, torch.exp(lf - set_lse[:, None]), 0.) * grad_nucleus[:, None]
        return d - onehot * (grad + grad_nucleus)[:, None]
    torch.manual_seed(0)
    rows, vocab = 300, 5000
    logits = (torch.randn(rows, vocab, device='cuda') * 4).to(torch.bfloat16)
    lf = logits.float()
    lse = torch.logsumexp(lf, -1)
    cut, set_lse = lfm3_fast_train._nucleus_cut_lse(lf, lse, .95, 1024)
    target = torch.randint(0, vocab, (rows,), device='cuda')
    grad, grad_nucleus = torch.randn(rows, device='cuda'), torch.randn(rows, device='cuda')
    old = lfm3_fast_train._compile(fp32_kernel)(lf, target, lse, set_lse, cut, grad, grad_nucleus).to(torch.bfloat16)
    new = lfm3_fast_train._nucleus_grad_kernel('cuda')(logits, target, lse, set_lse, cut, grad, grad_nucleus)
    assert new.dtype == torch.bfloat16 and torch.equal(new, old)


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


def test_continuation_after_a_past_equals_the_whole_document():
    _, fast = _pair()
    ids = torch.randint(0, 60, (17,))
    whole = lfm3_fast_train.hidden(fast, ids)
    for cuts in ((5,), (1, 9), (2, 3, 4), (12, 13)):  # pieces shorter and longer than the conv width and the window
        past, pieces, start = lfm3_fast_train.Past(), [], 0
        for end in (*cuts, len(ids)):
            pieces.append(lfm3_fast_train.hidden(fast, ids[start:end], pasts=[past]) if past
                          else lfm3_fast_train.hidden(fast, ids[start:end]))
            _, (segment,) = lfm3_fast_train._hidden_past(fast, ids[start:end], [end - start], [past], False, keep=True)
            past = lfm3_fast_train.Past((*past, segment))
            start = end
        torch.testing.assert_close(torch.cat(pieces), whole, rtol=1e-4, atol=1e-5)


def test_packed_continuations_of_different_pasts_equal_whole_documents():
    """One varlen forward of documents continuing different histories (none, short, longer than the window, two
    segments deep) next to a fresh document gives each whole document's hidden states and gradients."""
    stock, fast = _pair()
    docs = [torch.randint(0, 60, (n,)) for n in (9, 14, 6, 13)]
    cuts = [0, 3, 2, 11]
    expected = [stock.model(input_ids=d[None], use_cache=False).last_hidden_state[0][c:] for d, c in zip(docs, cuts)]
    pasts = []
    for doc, cut in zip(docs, cuts):
        past = lfm3_fast_train.Past()
        for a, b in ((0, cut // 2), (cut // 2, cut)):
            if b > a:
                _, (segment,) = lfm3_fast_train._hidden_past(fast, doc[a:b], [b - a], [past], False, keep=True)
                past = lfm3_fast_train.Past((*past, segment))
        pasts.append(past)
    lengths = [len(d) - c for d, c in zip(docs, cuts)]
    actual = lfm3_fast_train.hidden(fast, torch.cat([d[c:] for d, c in zip(docs, cuts)]), lengths, pasts=pasts)
    torch.testing.assert_close(actual, torch.cat(expected), rtol=1e-4, atol=1e-5)


def _shared_sequences():
    import numpy as np
    rng = np.random.default_rng(0)
    system, task = rng.integers(0, 60, 7), rng.integers(0, 60, 5)
    seqs = [np.concatenate([system, task, rng.integers(0, 60, n)]) for n in (4, 6, 3)]   # one group's prompt
    seqs.append(np.concatenate([seqs[0][:14], rng.integers(0, 60, 5)]))                  # a later chain of episode 0
    seqs.append(np.concatenate([system, rng.integers(0, 60, 8)]))                         # another task, same system
    seqs.append(rng.integers(0, 60, 9))                                                   # nothing shared
    caps = [12, 12, 12, 15, 9, 3]       # the first row each sequence's targets read
    return seqs, caps


def test_prefix_tree_covers_every_sequence_once_within_caps():
    seqs, caps = _shared_sequences()
    nodes = lfm3_fast_train.prefix_tree(seqs, caps, 2)
    seen = []

    def walk(node, start):
        assert node['start'] == start
        if 'index' in node:
            seen.append(node['index'])
            assert start <= caps[node['index']]
            return
        assert node['end'] - node['start'] >= 2 and len(node['children']) >= 2
        for child in node['children']:
            walk(child, node['end'])
    for node in nodes:
        walk(node, 0)
    assert sorted(seen) == list(range(len(seqs)))
    lengths = [len(s) for s in seqs]
    assert lfm3_fast_train.tree_tokens(nodes, lengths) < sum(lengths)
    assert lfm3_fast_train.tree_tokens(lfm3_fast_train.prefix_tree(seqs, caps, 0), lengths) == sum(lengths)


def _tree_run(model, seqs, caps, weights, pack_tokens, backward=True, budget=None, mode='levels', retained_gb=None):
    """run_prefix_tree over ``seqs`` with the loss sum(h[cap:] * weight): {index: h[cap:]} and the batches seen."""
    actual, batches = {}, []

    def leaf(items, forward):
        batches.append(list(items))
        ids = torch.cat([torch.from_numpy(seqs[i][s:]).long() for i, s in items])
        h = forward(ids, [len(seqs[i]) - s for i, s in items])
        loss, offset = 0., 0
        for i, s in items:
            rows = h[offset + caps[i] - s:offset + len(seqs[i]) - s]
            actual[i] = rows.detach()
            loss = loss + (rows * weights[i]).sum()
            offset += len(seqs[i]) - s
        if backward:
            loss.backward()
    nodes = lfm3_fast_train.prefix_tree(seqs, caps, 2)
    stats = lfm3_fast_train.run_prefix_tree(model, nodes, lambda i, a, b: torch.from_numpy(seqs[i][a:b]).long(), leaf,
                                            lambda i, s: len(seqs[i]) - s, pack_tokens, budget, backward, mode=mode,
                                            batch_tokens=pack_tokens, retained_gb=retained_gb)
    return actual, batches, stats


def test_prefix_tree_gradients_equal_separate_sequences():
    seqs, caps = _shared_sequences()
    _, flat = _pair()
    weights = [torch.randn(len(s) - c, flat.config.hidden_size) for s, c in zip(seqs, caps)]
    expected = {}
    for index, seq in enumerate(seqs):
        h = lfm3_fast_train.hidden(flat, torch.from_numpy(seq).long())
        expected[index] = h[caps[index]:].detach()
        (h[caps[index]:] * weights[index]).sum().backward()
    nodes = lfm3_fast_train.prefix_tree(seqs, caps, 2)
    assert any('children' in n and any('children' in c for c in n['children']) for n in nodes)   # two levels shared
    # one leaf per batch, a few per batch, every leaf of the tree in one batch; with and without an activation budget
    for pack_tokens, budget in ((1, None), (12, 1e-4), (1000, None), (1000, 1.)):
        _, tree = _pair()
        tree.load_state_dict(flat.state_dict())
        actual, batches, stats = _tree_run(tree, seqs, caps, weights, pack_tokens, budget=budget)
        trained = sorted(i for batch in batches for i, _ in batch)
        assert trained == list(range(len(seqs)))                                    # every chain once
        if pack_tokens == 1000:
            assert len(batches) == 2 and stats['prefix_batches'] == 2               # wholes, continuations; 2 levels
        for index in expected:
            torch.testing.assert_close(actual[index], expected[index], rtol=1e-4, atol=1e-5)
        _grads_close(flat, tree, f'prefix tree pack {pack_tokens}')
    without, _, _ = _tree_run(tree, seqs, caps, weights, 12, backward=False)   # the KL reference pass's walk
    for index in expected:
        torch.testing.assert_close(without[index], expected[index], rtol=1e-4, atol=1e-5)
    # tree-mask: one flattened forward per batch; small batches split subtrees (each with its own copy of the path)
    for batch_tokens in (1, 9, 20, 1000):
        _, tree = _pair()
        tree.load_state_dict(flat.state_dict())
        actual, batches, stats = _tree_run(tree, seqs, caps, weights, batch_tokens, mode='tree-mask')
        assert sorted(i for batch in batches for i, _ in batch) == list(range(len(seqs)))   # every chain once
        if batch_tokens == 1000:
            assert stats['batches'] == 1
        for index in expected:
            torch.testing.assert_close(actual[index], expected[index], rtol=1e-4, atol=1e-5)
        _grads_close(flat, tree, f'tree-mask batch {batch_tokens}')


def test_tree_block_mask_lists_every_visible_tile_and_only_true_full_tiles():
    """The flex-attention BlockMask built from the tree's structure (CPU tensors, 4-row tiles): a tile with any visible
    (query, key) pair is listed (partial or full), and a full tile has every pair visible, for full and sliding windows."""
    lengths, parents = [9, 6, 11, 5, 7, 3, 10], [-1, 0, 0, 1, -1, 4, 2]
    pos, doc, _, ancestors = lfm3_fast_train.tree_layout(lengths, parents, 4)
    pos, doc, ancestors = torch.from_numpy(pos), torch.from_numpy(doc), torch.from_numpy(ancestors)
    fulls = 0
    for window in (-1, 3, 8):
        dense = lfm3_fast_train.tree_masks(pos, doc, ancestors, [window], 'cpu')[window]

        def mask_mod(b, h, q, k):
            seen = ancestors[doc[q], doc[k]] & (pos[k] <= pos[q])
            return seen & (pos[q] - pos[k] <= window) if window >= 0 else seen
        mask = lfm3_fast_train._tree_block_mask(pos, doc, ancestors, window, mask_mod, block=4)
        blocks = -(-len(pos) // 4)
        listed = {kind: set() for kind in ('partial', 'full')}
        for kind, num, idx in (('partial', mask.kv_num_blocks, mask.kv_indices),
                               ('full', mask.full_kv_num_blocks, mask.full_kv_indices)):
            for q in range(blocks):
                listed[kind] |= {(q, int(k)) for k in idx[0, 0, q, :num[0, 0, q]]}
        assert not listed['partial'] & listed['full']
        for q in range(blocks):
            for k in range(blocks):
                tile = dense[4 * q:4 * q + 4, 4 * k:4 * k + 4]
                if tile.any():
                    assert (q, k) in listed['partial'] | listed['full'], (window, q, k)
                if (q, k) in listed['full']:
                    assert tile.all() and tile.shape == (4, 4), (window, q, k)
        fulls += len(listed['full'])
    assert fulls                                                     # the structure finds full tiles at all


@pytest.mark.skipif(not torch.cuda.is_available(), reason='flex-attention BlockMask path (CUDA)')
def test_tree_mask_on_cuda_equals_the_dense_path_mask():
    """On CUDA the tree mask is a flex-attention BlockMask built from the tree's structure: in FP32 its states and
    gradients equal the CPU's dense-mask run of the same flattened tree (which equals the whole sequences)."""
    seqs, caps = _shared_sequences()
    _, cpu = _pair()
    _, gpu = _pair()
    gpu.load_state_dict(cpu.state_dict())
    gpu.cuda()
    weights = [torch.randn(len(s) - c, cpu.config.hidden_size) for s, c in zip(seqs, caps)]
    states = {}
    for model, device in ((cpu, 'cpu'), (gpu, 'cuda')):
        actual = states[device] = {}

        def leaf(items, forward, device=device, actual=actual):
            ids = torch.cat([torch.from_numpy(seqs[i][s:]).long().to(device) for i, s in items])
            h, loss, offset = forward(ids, [len(seqs[i]) - s for i, s in items]), 0., 0
            for i, s in items:
                rows = h[offset + caps[i] - s:offset + len(seqs[i]) - s]
                actual[i] = rows.detach().cpu()
                loss = loss + (rows * weights[i].to(device)).sum()
                offset += len(seqs[i]) - s
            loss.backward()
        lfm3_fast_train.run_prefix_tree(model, lfm3_fast_train.prefix_tree(seqs, caps, 2),
                                        lambda i, a, b, device=device: torch.from_numpy(seqs[i][a:b]).long().to(device),
                                        leaf, lambda i, s: len(seqs[i]) - s, 1000, mode='tree-mask', batch_tokens=1000)
    for index in states['cpu']:
        torch.testing.assert_close(states['cuda'][index], states['cpu'][index], rtol=1e-4, atol=1e-5)
    for (name, a), (_, b) in zip(cpu.named_parameters(), gpu.named_parameters()):
        assert (b.grad.cpu() - a.grad).norm() <= 1e-4 * a.grad.norm() + 1e-9, name


def test_tree_batch_plan_covers_every_chain_once():
    """The planned tree-mask batches hold every chain once, within the largest batch the budget allows; with room for
    the whole tree unrecomputed the plan is one batch."""
    seqs, caps = _shared_sequences()
    _, model = _pair()
    nodes = lfm3_fast_train.prefix_tree(seqs, caps, 2)
    tokens = lambda i, s: len(seqs[i]) - s
    total = lfm3_fast_train.tree_tokens(nodes, [len(s) for s in seqs])
    layers = len(model.model.layers)
    for budget, batches_expected in ((1.0, 1), (None, None)):
        if budget is None:                  # resident: 6 rows; the largest batch (every layer recomputed): ~4x that
            budget = 6 * layers * lfm3_fast_train.ACTIVATION_BYTES_PER_TOKEN_LAYER / 2**30
        cap, batches = lfm3_fast_train.plan_tree_batches(model, nodes, tokens, budget)
        seen = []

        def leaves(node):
            if 'index' in node:
                seen.append(node['index'])
            for child in node.get('children', ()):
                leaves(child)
        for batch in batches:
            for node in batch:
                leaves(node)
        assert sorted(seen) == list(range(len(seqs)))
        assert cap <= max(1, lfm3_fast_train.tree_batch_tokens(model, budget))
        if batches_expected:
            assert len(batches) == batches_expected and cap >= total


def test_tree_layout_paths_positions_and_conv_predecessors():
    """Path positions continue the parent's; a document's first taps read its parent's last tokens (several levels up
    when the parent is shorter than the conv), never a sibling's; roots read zeros."""
    import numpy as np
    lengths, parents = [4, 2, 3, 5, 1], [-1, 0, 0, 1, -1]      # 3 continues 1 (2 tokens) continues 0
    pos, doc, pred, ancestors = lfm3_fast_train.tree_layout(lengths, parents, 4)
    assert pos.tolist() == [0, 1, 2, 3, 4, 5, 4, 5, 6, 6, 7, 8, 9, 10, 0]
    assert ancestors[3].tolist() == [True, True, False, True, False] and not ancestors[1, 2]
    rows = {d: np.flatnonzero(doc == d) for d in range(5)}
    zero = len(doc)
    assert pred[:, rows[2][0]].tolist() == [3, 2, 1]                           # doc 2's first token: doc 0's last three
    assert pred[:, rows[3][0]].tolist() == [5, 4, 3]                           # doc 3: doc 1's two, then doc 0's last
    assert pred[:, rows[3][1]].tolist() == [rows[3][0], 5, 4]
    assert pred[:, rows[0][1]].tolist() == [0, zero, zero] and pred[:, rows[4][0]].tolist() == [zero] * 3




def test_prefix_tree_continuation_cotangents_accumulate_in_fp32():
    """The segments the continuations read are FP32 leaves: several continuations' gradients sum there in FP32 (not
    in the prefix's BF16), and the prefix graph receives that sum once."""
    seqs, caps = _shared_sequences()
    _, tree = _pair()
    seen = []
    original = lfm3_fast_train._hidden_past

    def spy(model, ids, lengths, pasts, compiled, keep):
        for past in pasts:
            for segment in past:
                seen.extend(t.dtype for t in segment.tensors())
        return original(model, ids, lengths, pasts, compiled, keep)
    lfm3_fast_train._hidden_past = spy
    try:
        tree.to(torch.bfloat16)
        weights = [torch.randn(len(s) - c, tree.config.hidden_size, dtype=torch.bfloat16) for s, c in zip(seqs, caps)]
        _tree_run(tree, seqs, caps, weights, 1000)
    finally:
        lfm3_fast_train._hidden_past = original
    assert seen and set(seen) == {torch.float32}
    assert all(p.grad is not None and torch.isfinite(p.grad.float()).all() for p in tree.parameters())


def _sft_chains():
    """exp0_sft chains over `_shared_sequences` (one to three supervised segments, partial loss weights)."""
    import numpy as np
    seqs, caps = _shared_sequences()
    rng = np.random.default_rng(1)
    chains = []
    for index, (seq, cap) in enumerate(zip(seqs, caps)):
        segs, start = [], cap + 1
        while start < len(seq):
            length = min(len(seq) - start, 2 + index % 2)
            row = {'sample': seq[start:start + length]}
            if index % 2:
                row['weight'] = rng.uniform(0, 1, length).astype(np.float32)
            segs.append((start, length, row))
            start += length + 1
        chains.append({'ids': seq, 'segs': segs, 'imports': [], 'task': f't{index % 3}', 'round': f'r{index}'})
    return chains


def test_exp0_step_with_shared_prefixes_equals_whole_chains():
    import exp0_sft
    chains = _sft_chains()
    denominator = sum(exp0_sft.chain_mass(c) for c in chains)
    _, whole = _pair()
    _, tree = _pair()
    tree.load_state_dict(whole.state_dict())
    expected, plain_tokens = exp0_sft.step_backward(whole, lfm3_fast_train.hidden, chains, denominator, 13, 1., 0)
    for mode in ('levels', 'tree-mask'):
        tree.zero_grad()
        actual, tree_tokens = exp0_sft.step_backward(tree, lfm3_fast_train.hidden, chains, denominator, 13, 1., 2,
                                                     mode=mode, tree_batch_tokens=30)
        assert plain_tokens == sum(len(c['ids']) for c in chains) and tree_tokens < plain_tokens
        assert abs(actual - expected) <= 1e-5 * abs(expected)
        _grads_close(whole, tree, f'exp0 prefix tree {mode}')


def test_exp0_step_groups_cover_every_chain_once_per_epoch():
    import exp0_sft
    chains = _sft_chains() * 7
    for i, chain in enumerate(chains):
        chains[i] = {**chain, 'round': f'r{i % 11}'}
    for grouped in (True, False):
        steps = exp0_sft.step_groups(chains, 3, 9, 5, grouped)
        assert steps == exp0_sft.step_groups(chains, 3, 9, 5, grouped)          # reproducible from the seed
        assert steps != exp0_sft.step_groups(chains, 3, 9, 6, grouped)
        if grouped:
            flat = [i for step in steps for i in step]
            assert sorted(flat) == sorted(list(range(len(chains))) * 3)
            seen = 0
            for step in steps:                      # epochs do not mix: each prefix of whole epochs is a permutation
                seen += len(step)
                if seen % len(chains) == 0:
                    assert sorted(flat[:seen]) == sorted(list(range(len(chains))) * (seen // len(chains)))
            assert all(sum(exp0_sft.chain_loss_tokens(chains[i]) for i in step) >= 9 // 2 for step in steps)

def test_exp0_step_groups_fill_a_batch_with_weighted_loss_tokens():
    import exp0_sft
    import numpy as np

    def chain(n, masked):
        row = {'sample': np.zeros(n, np.int32), 'weight': (np.arange(n) >= masked).astype(np.float32)}
        return {'ids': np.zeros(n + 1, np.int32), 'segs': [(1, n, row)], 'task': 't', 'round': 'r'}
    chains = [chain(10, 8) for _ in range(10)]                  # 10 target tokens, 2 of them weighted
    steps = exp0_sft.step_groups(chains, 1, 4, 0, False)
    assert [len(step) for step in steps] == [2] * 5              # 4 weighted tokens a step, not 4 raw ones


def test_exp0_stratified_steps_carry_the_whole_sets_source_mix():
    import exp0_sft
    import numpy as np

    def chain(source, i, n):
        row = {'sample': np.zeros(n, np.int32)}
        return {'ids': np.zeros(n + 1, np.int32), 'segs': [(1, n, row)], 'task': f'{source}{i}', 'round': f'{source}:{i}'}
    # a few large own chains among many small teacher chains (sft-v4-run1: 92 own of 16k chains, 19% of the tokens)
    chains = [chain('rl', i, 20) for i in range(24)] + [chain('teacher', i, 4) for i in range(500)]
    own_share = 24 * 20 / (24 * 20 + 500 * 4)

    def shares(steps):
        return [sum(exp0_sft.chain_mass(chains[i]) for i in step if chains[i]['round'].startswith('rl:'))
                / sum(exp0_sft.chain_mass(chains[i]) for i in step) for step in steps]
    steps = exp0_sft.step_groups(chains, 1, 124, 3, False, stratified=True)
    assert sorted(i for step in steps for i in step) == list(range(len(chains)))
    # every step carries own tokens near the overall share (one own chain is 16% of a step)
    assert all(abs(share - own_share) < 0.17 for share in shares(steps)) and min(shares(steps)) > 0
    assert steps == exp0_sft.step_groups(chains, 1, 124, 3, False, stratified=True)
    # a plain shuffle of the same chains leaves some steps without own tokens and piles them up in others
    random_steps = exp0_sft.step_groups(chains, 1, 124, 3, False)
    assert max(abs(share - own_share) for share in shares(random_steps)) > 0.17


def test_exp0_student_sampled_chains_take_the_nucleus_nll_and_others_the_full_softmax(monkeypatch):
    import exp0_sft
    import numpy as np

    def chain(round_name, **extra):
        return {'ids': np.zeros(4, np.int32), 'segs': [(1, 2, {'sample': np.zeros(2, np.int32)})], 'task': 't',
                'round': round_name, **extra}
    own, graph, teacher, ops = chain('rl:x'), chain('rl-neuralese-graph:y'), chain('teacher:z'), chain('0')
    assert [exp0_sft.student_sampled(c) for c in (own, graph, teacher, ops)] == [True, True, False, False]
    assert not exp0_sft.student_sampled(chain('rl:x', sampled=False))
    mask = exp0_sft.sampled_mask([teacher, own], 'cpu')
    assert mask.tolist() == [False, False, True, True]
    monkeypatch.setattr(exp0_sft, 'token_nll', lambda m, h, p, t: torch.full((4,), 1.))
    monkeypatch.setattr(exp0_sft, 'token_nll_nucleus', lambda m, h, p, t, top_p: (torch.full((4,), 1.), torch.full((4,), .5), None))
    assert exp0_sft.targets_nll(None, None, None, None, mask, 0.95).tolist() == [1., 1., .5, .5]
    assert exp0_sft.targets_nll(None, None, None, None, mask, 1.).tolist() == [1.] * 4
    assert exp0_sft.targets_nll(None, None, None, None, exp0_sft.sampled_mask([teacher, ops], 'cpu'), 0.95).tolist() == [1.] * 4

def test_exp0_context_masks_zero_an_empty_think_lead_and_targets_after_the_end_token():
    import exp0_sft
    import numpy as np

    class Tok:
        vocab = ['</think>', '\n', '"A"', '<|im_end|>', 'plan', '<|endoftext|>']

        def convert_tokens_to_ids(self, tokens):
            if isinstance(tokens, list):
                return [self.vocab.index(t) for t in tokens]
            return self.vocab.index(tokens)

        def decode(self, ids):
            return ''.join(self.vocab[i] for i in ids)
    readout = {'sample': np.array([0, 2, 3], np.int32)}                       # </think> "A" <|im_end|> (a scored option)
    operator = {'sample': np.array([0, 2, 3, 1], np.int32)}                   # ... <|im_end|> \n (template separator)
    sampled = {'sample': np.array([4, 0, 2, 3], np.int32)}                    # plan </think> "A" <|im_end|>
    chains = [{'ids': np.zeros(5, np.int32), 'segs': [(1, len(r['sample']), r)], 'task': 't', 'round': 'r'}
              for r in (readout, operator, sampled)]
    assert exp0_sft.context_masks(chains, Tok()) == {'empty_think_rows': 2, 'after_end_rows': 1}
    assert readout['weight'].tolist() == [0, 1, 1]
    assert operator['weight'].tolist() == [0, 1, 1, 0]
    assert 'weight' not in sampled


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA kernel check')
def test_nucleus_nll_on_bf16_logits_matches_the_reference_pass_bitwise():
    """The nucleus NLL's log-probability is linear_nll's own kernel (what the KL reference pass computes): at equal weights
    the two agree bit for bit; the nucleus read from the bf16 logits (top-k and the wide paths) equals the fp32 one's."""
    torch.manual_seed(0)
    rows, vocab, width = 300, 5000, 64
    h = torch.randn(rows, width, device='cuda').to(torch.bfloat16)
    weight = (torch.randn(vocab, width, device='cuda') * .4).to(torch.bfloat16)
    target = torch.randint(0, vocab, (rows,), device='cuda')
    plain = lfm3_fast_train.linear_nll(h, weight, target)
    lf = F.linear(h, weight).float()
    lse = torch.logsumexp(lf, -1)
    for top_k, wide_k in ((4, 8), (4, 16384), (1024, 16384)):
        lfm3_fast_train.NUCLEUS_WIDE_K, saved = wide_k, lfm3_fast_train.NUCLEUS_WIDE_K
        try:
            nll, nucleus, outside = lfm3_fast_train.linear_nll_nucleus(h, weight, target, .95, chunk=128, top_k=top_k)
            want_cut, want_inside = lfm3_fast_train._nucleus_cut_lse(lf, lse, .95, top_k)
            cut, inside = lfm3_fast_train._nucleus_cut_lse(F.linear(h, weight), lse, .95, top_k)
        finally:
            lfm3_fast_train.NUCLEUS_WIDE_K = saved
        assert torch.equal(nll, plain)
        assert torch.equal(cut, want_cut) and torch.equal(inside, want_inside)
        own = lf.gather(-1, target[:, None])[:, 0]
        assert torch.equal(outside, own < want_cut)
        assert torch.allclose(nucleus, torch.where(outside, torch.logaddexp(want_inside, own), want_inside) - own, atol=1e-5)


def _passthrough_case():
    """A producer document (a text write: [open] + text ids from position 0, span rows 1..n) and a consumer that
    imports its state at rows [start, start + n) (placeholders = the span's ids), the two token ranges disjoint."""
    torch.manual_seed(11)
    document = torch.cat([torch.tensor([63]), torch.randint(40, 60, (6,))])
    consumer = torch.randint(0, 40, (15,))
    start, count = 4, len(document) - 1
    consumer[start:start + count] = document[1:]
    sources = {'b': {'ids': document.numpy(), 'start': 1, 'stop': len(document)}}
    return document, consumer, start, count, sources


def _passthrough_forward(model, consumer, start, sources, frozen=False):
    plan = lfm3_fast_train.passthrough_plan([(0, [(start, 'b')], sources)], len(consumer), frozen_all=frozen)
    ids = torch.cat([consumer] + [torch.as_tensor(d, dtype=torch.long) for d in plan['documents']])
    lengths = [len(consumer)] + [len(d) for d in plan['documents']]
    return lfm3_fast_train.hidden(model, ids, lengths, lfm3_fast_train.passthrough_tensors(plan, 'cpu'))[:len(consumer)]


def test_passthrough_state_equals_the_captured_block_and_trains_the_producer():
    """The in-graph state of the producer document equals the state a capture of it stores (the stock HF forward's
    normed pre-RoPE K/V and conv B*x at the span rows, spliced as the actor's fork does): the consumer's hidden states
    agree. The consumer's loss then reaches the embedding rows of tokens that occur only in the producer document; with
    the state frozen it does not, and the consumer's own gradients equal the stored-state path's."""
    from types import SimpleNamespace
    stock, fast = _pair()
    document, consumer, start, count, sources = _passthrough_case()
    states = _layer_states(stock, document)
    block = SimpleNamespace(token_ids=document[1:], layers=tuple(
        {k: (v[1:] if k != 'kind' else v) for k, v in s.items()} for s in states))
    stored = lfm3_fast_train.hidden(fast, consumer, imports=lfm3_fast_train.import_tensors(fast, [(start, block)], 'cpu'))
    passed = _passthrough_forward(fast, consumer, start, sources)
    torch.testing.assert_close(passed, stored, rtol=1e-4, atol=1e-5)
    weights = torch.randn_like(passed)
    weights[:start + count] = 0                      # the loss on the consumer's rows after the import
    embed = fast.model.embed_tokens.weight
    producer_only = sorted(set(document.tolist()) - set(consumer[:start].tolist() + consumer[start + count:].tolist()))
    fast.zero_grad()
    (passed * weights).sum().backward()
    assert embed.grad[producer_only].abs().sum(-1).min() > 0
    fast.zero_grad()
    (_passthrough_forward(fast, consumer, start, sources, frozen=True) * weights).sum().backward()
    assert embed.grad[producer_only].abs().max() == 0
    frozen_grads = {name: p.grad.clone() for name, p in fast.named_parameters()}
    fast.zero_grad()
    (stored * weights).sum().backward()
    for name, p in fast.named_parameters():
        assert (frozen_grads[name] - p.grad).norm() <= 1e-4 * p.grad.norm() + 1e-12, name


def test_passthrough_gradient_matches_finite_differences():
    """Float64: d(consumer loss)/d(a weight) through the pass-through equals the central difference of the loss with
    the producer's state recomputed at the perturbed weights (the real dependency), and differs from the frozen-state
    gradient on weights that act on the producer document. (The conv accumulates in FP32 even in a float64 model, as the
    inference kernel does: a step of 1e-3 keeps that rounding far below the tolerance.)"""
    _, fast = _pair()
    fast.double()
    document, consumer, start, count, sources = _passthrough_case()
    weights = torch.randn(len(consumer), fast.config.hidden_size, dtype=torch.float64)
    weights[:start + count] = 0
    loss = lambda frozen=False: (_passthrough_forward(fast, consumer, start, sources, frozen) * weights).sum()
    checked = 0
    for name in ('model.layers.0.conv.in_proj.weight', 'model.layers.1.self_attn.k_proj.weight'):
        parameter = dict(fast.named_parameters())[name]
        fast.zero_grad()
        loss().backward()
        analytic = parameter.grad.clone()
        fast.zero_grad()
        loss(frozen=True).backward()
        frozen = parameter.grad.clone()
        for index in [(0, 0), (3, 5), (7, 2)]:
            eps = 1e-3
            with torch.no_grad():
                original = parameter[index].item()
                parameter[index] = original + eps
                up = loss().item()
                parameter[index] = original - eps
                down = loss().item()
                parameter[index] = original
            numeric = (up - down) / (2 * eps)
            assert abs(numeric - analytic[index].item()) <= 2e-3 * max(1., abs(numeric)), (name, index, numeric)
            assert abs(numeric - frozen[index].item()) > 2e-2 * max(1., abs(numeric)) or index != (0, 0), (name, index)
            checked += 1
        assert (analytic - frozen).abs().max() > 1e-6, name
    assert checked == 6


def test_exp0_rows_and_steps_carry_producer_documents():
    """exp0_sft: a chain that imports state packs with others (its producer document follows the row), skips the prefix
    tree, and its step equals the direct pass-through forward."""
    import numpy as np
    import exp0_sft
    _, fast = _pair()
    document, consumer, start, count, sources = _passthrough_case()
    sample = consumer[12:].numpy().astype(np.int32)
    importing = {'ids': consumer.numpy().astype(np.int32), 'segs': [(12, len(sample), {'sample': sample})],
                 'imports': [(start, 'b')], 'sources': sources, 'task': 't', 'round': 0}
    other = {'ids': np.arange(9, dtype=np.int32), 'segs': [(5, 4, {'sample': np.arange(5, 9, dtype=np.int32)})],
             'imports': [], 'sources': {}, 'task': 'u', 'round': 0}
    assert exp0_sft.source_tokens(importing) == len(document) and exp0_sft.source_tokens(other) == 0
    ids, positions, targets, weights, lengths, imports = exp0_sft.row_inputs([other, importing], 'cpu')
    assert lengths == [9, len(consumer), len(document)] and ids[-len(document):].tolist() == document.tolist()
    assert imports['pass_rows'].tolist() == list(range(9 + start, 9 + start + count))
    assert imports['pass_source'].tolist() == list(range(9 + len(consumer) + 1, 9 + len(consumer) + len(document)))
    assert 'pass_frozen' not in imports
    assert exp0_sft.row_inputs([importing], 'cpu', passthrough=False)[-1]['pass_frozen'].all()
    with pytest.raises(ValueError, match='without a producer document'):
        exp0_sft.row_inputs([{**importing, 'sources': {}}], 'cpu')
    chains = [importing, other, {**other, 'task': 'v'}]
    denominator = sum(exp0_sft.chain_mass(c) for c in chains)
    # one sequence per forward, packed batches, and the tree-mask's flattened batch (prefix rows shift the imports)
    for pack_tokens, mode in ((0, 'levels'), (64, 'levels'), (64, 'tree-mask')):
        fast.zero_grad()
        total, computed = exp0_sft.step_backward(fast, lfm3_fast_train.hidden, chains, denominator, pack_tokens, 1., 4,
                                                 mode=mode, tree_batch_tokens=200)
        if pack_tokens == 0:
            grads = {name: p.grad.clone() for name, p in fast.named_parameters()}
        for name, p in fast.named_parameters():
            assert (grads[name] - p.grad).norm() <= 1e-4 * p.grad.norm() + 1e-12, (mode, name)
    h = _passthrough_forward(fast, consumer, start, sources)
    direct = exp0_sft.token_nll(fast, h, torch.arange(11, 11 + len(sample)), torch.as_tensor(sample, dtype=torch.long))
    direct = direct.sum().item() + sum(exp0_sft.token_nll(fast, lfm3_fast_train.hidden(fast, torch.as_tensor(c['ids'], dtype=torch.long)),
                                                         torch.arange(4, 8), torch.arange(5, 9)).sum().item() for c in chains[1:])
    assert abs(total - direct) <= 1e-5 * abs(direct)
    assert len(consumer) + len(document) <= computed <= len(consumer) + len(document) + 18


def test_rollout_lowering_resolves_imports_to_their_write_sources():
    import collections
    from lower_rollout_turns import episode_turns
    write = {'kind': 'block_write', 'block': 'nzs1_x', 'node': 'p#1', 'source_capture': {
        'segments': [{'kind': 'tokens', 'token_ids': [63], 'producing': [False]},
                     {'kind': 'tokens', 'token_ids': [41, 42, 43], 'producing': [True] * 3},
                     {'kind': 'tokens', 'token_ids': [62], 'producing': [False]}], 'payload_start': 1, 'payload_stop': 4}}
    sample = [{'token_id': 7, 'logp': -0.5, 'input_role': 'ordinary'}]
    turn = lambda prompt: {'invocation_id': 'c', 'sampling': {
        'prompt': prompt, 'samples': sample, 'generated_inputs': [{'sample_index': 0, 'token_id': 7}],
        'policy': {'checkpoint_sha256': 's'}, 'sampling_law': {'temperature': 1.0}}}
    prompt = [{'kind': 'tokens', 'token_ids': [1, 61]}, {'kind': 'native_state', 'id': 'nzs1_x', 'source_occurrence': 'p#1'},
              {'kind': 'tokens', 'token_ids': [60, 5]}]
    frozen = [{**prompt[1], 'stop_gradient': True}]
    stats = collections.Counter()
    rows = episode_turns({'turns': [turn(prompt), turn(frozen)], 'execution_graph': [write]}, stats)
    assert rows[0]['prompt'].tolist() == [1, 61, 41, 42, 43, 60, 5] and rows[0]['imports'] == [(2, 'nzs1_x')]
    source = rows[0]['sources']['nzs1_x']
    assert source['ids'].tolist() == [63, 41, 42, 43] and (source['start'], source['stop']) == (1, 4)
    assert rows[1]['imports'] == [(0, 'nzs1_x#frozen')] and rows[1]['sources']['nzs1_x#frozen']['frozen']
    assert stats['passthrough_imports'] == 2
    stats = collections.Counter()
    assert episode_turns({'turns': [turn(prompt)]}, stats) == [] and stats['skip_native_state_prompt'] == 1


def test_prefix_parts_bound_prefix_tokens_and_keep_every_leaf_once():
    leaf = lambda i: {'start': 4, 'index': i}
    nodes = [{'start': 0, 'end': 4, 'first': 0, 'children': [
        {'start': 4, 'end': 10, 'first': 0, 'children': [leaf(0), leaf(1)]},
        {'start': 4, 'end': 9, 'first': 2, 'children': [leaf(2), leaf(3)]},
        {'start': 4, 'end': 12, 'first': 4, 'children': [leaf(4)]}]}, {'start': 0, 'index': 5}]
    parts = lfm3_fast_train.prefix_parts(nodes, 12)
    leaves = []

    def walk(node):
        if 'index' in node:
            leaves.append(node['index'])
        else:
            for child in node['children']:
                walk(child)
    for part in parts:
        assert sum(lfm3_fast_train._prefix_tokens(n) for n in part) <= 12
        for node in part:
            walk(node)
    assert sorted(leaves) == list(range(6)) and len(parts) == 3
    assert lfm3_fast_train.prefix_parts(nodes, 1000) == [nodes]


def test_prefix_tree_in_parts_gives_the_same_gradients():
    """``retained_gb`` splits the forest (shared ancestors recomputed per part): outputs and gradients unchanged."""
    seqs, caps = _shared_sequences()
    _, whole = _pair()
    weights = [torch.randn(len(s) - c, whole.config.hidden_size) for s, c in zip(seqs, caps)]
    expected, _, plan = _tree_run(whole, seqs, caps, weights, 12)
    assert 'parts' not in plan
    _, split = _pair()
    split.load_state_dict(whole.state_dict())
    per_token = lfm3_fast_train.retained_bytes_per_token(split)
    actual, batches, plan = _tree_run(split, seqs, caps, weights, 12, retained_gb=3 * per_token / 2**30)
    assert plan['parts'] > 1 and sorted(i for b in batches for i, _ in b) == list(range(len(seqs)))
    for index in expected:
        torch.testing.assert_close(actual[index], expected[index], rtol=1e-4, atol=1e-5)
    _grads_close(whole, split, 'prefix tree in parts')
