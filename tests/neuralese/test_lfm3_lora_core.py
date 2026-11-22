"""Automated correctness suite for the shared LFM3 `lora+rows` learner core (`lfm3opti.md` §5, §6, v2 §10.1):
- Independent eager FP32 mathematical oracle (`allow_tf32=False`) with exact LoRA + trainable sparse rows + nucleus law.
- Adapter bypass across all 9 target families, B=0 exact base identity, step-0 grad_B != 0 & step-1 grad_A != 0.
- Frozen base/head/norm regression: 0 frozen tensors in AdamW, 0 drift after step, nonzero grad on layer-0 & special rows.
- Packing & multi-child prefix tree (`levels` & `tree-mask`) parity with scale != 1 and separate child backwards.
- Differentiable Neuralese pass-through gradients into producer adapters (zero when detached).
- Interleaved reference-view switching with zero aliasing or freed-graph errors.
- Merge/scaling round-trip parity (factored learner vs merged BF16 checkpoint).
- Atomic save/resume replay parity (exact next step, RNG, optimizer state, ref factors, cursors).
- Deterministic OOM retry state machine and Blackwell FP8 frozen-base GEMM logprob/gradient agreement.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'tests/neuralese')]

from natlang_neuralese.train import lfm3_fast  # noqa: E402
from natlang_neuralese.train.learner_state import (  # noqa: E402
    export_merged_checkpoint,
    load_learner_state,
    save_learner_state,
)
from natlang_neuralese.train.lfm3_learner import (  # noqa: E402
    AdapterSpec,
    Lfm3Learner,
    OOMRetryPlan,
    run_step_with_oom_retry,
)
from natlang_neuralese.tune.targets import LFM3_LORA_FAMILIES, packed_adapter_ranks  # noqa: E402
from test_lfm3_port import tiny_lfm3_hf  # noqa: E402


def _eager_fp32_oracle_forward(learner: Lfm3Learner, ids: torch.Tensor, *, use_adapters: bool = True,
                               use_rows: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent eager FP32 mathematical oracle: materializes W0 + (alpha/r) B A in FP32 and runs stock HF forward."""
    base = copy.deepcopy(learner.model).float()
    with torch.no_grad():
        if use_adapters:
            for param_name, (layer_idx, _kind, short_name) in learner.target_map.items():
                leaf = getattr(learner.bundles[layer_idx], short_name)
                mod_name = param_name.removesuffix('.weight')
                mod = base.get_submodule(mod_name)
                mod.weight.copy_(mod.weight.float() + float(leaf.scale) * (leaf.b.detach().float() @ leaf.a.detach().float()))
        if use_rows and learner.trainable_rows is not None:
            idx = learner.trainable_rows.token_ids.cpu()
            base.model.embed_tokens.weight[idx] = learner.trainable_rows.rows.detach().cpu().float()
    h = base.model(input_ids=ids.cpu()[None], use_cache=False).last_hidden_state[0]
    logits = F.linear(h, base.model.embed_tokens.weight)
    return h, logits


def _make_learner(rank=4, alpha=8.0, seed=0, trainable_rows=True, special_ids=(58, 59), dtype=torch.float32,
                  device='cpu', fp8_base=False, override_ids=()):
    model = tiny_lfm3_hf(window=4).to(device=device, dtype=dtype)
    spec = AdapterSpec(target_families=LFM3_LORA_FAMILIES, rank=rank, alpha=alpha, trainable_rows=trainable_rows)
    return Lfm3Learner.from_model(
        model, spec, special_ids=special_ids, marker_ids=tuple(i for i in special_ids if i not in override_ids),
        override_ids=override_ids,
        compile_layers=False, fp8_base=fp8_base, seed=seed,
    )


def _randomize_b(learner, std=0.05, seed=0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for b in learner.bundles:
            for _, leaf in b.iter_leaves():
                leaf.b.copy_(torch.randn(leaf.b.shape, generator=gen) * std)


def _logp(model, ids, **kw):
    return lfm3_fast.token_logprobs(model, ids, torch.arange(len(ids) - 1), ids[1:], **kw)


def test_adapter_bypass_all_9_families_and_b0_identity_and_step0_step1_grads():
    learner = _make_learner(rank=4, alpha=8.0, seed=1)
    ids = torch.tensor([1, 5, 12, 58, 7, 59, 9, 14, 2, 3], dtype=torch.long)

    # B=0 reproduces the bare base EXACTLY (live factors vs the bare-base reference path), and matches the oracle.
    h_fast = lfm3_fast.hidden(learner.model, ids)
    assert torch.equal(h_fast, lfm3_fast.hidden(learner.model, ids, adapters=()))
    h_base, _ = _eager_fp32_oracle_forward(learner, ids, use_adapters=False, use_rows=True)
    torch.testing.assert_close(h_fast, h_base, rtol=1e-4, atol=1e-5)

    # Nonzero (A, B) changes output for EACH of the 9 target families independently.
    for family in LFM3_LORA_FAMILIES:
        probe = _make_learner(rank=4, alpha=8.0, seed=2)
        with torch.no_grad():
            for param_name, (layer_idx, _kind, short_name) in probe.target_map.items():
                if f'.{family}.weight' in param_name:
                    leaf = getattr(probe.bundles[layer_idx], short_name)
                    leaf.b.fill_(0.25)
        h_mod = lfm3_fast.hidden(probe.model, ids)
        h_ref, _ = _eager_fp32_oracle_forward(probe, ids, use_adapters=True, use_rows=True)
        torch.testing.assert_close(h_mod, h_ref, rtol=1e-4, atol=1e-5)
        assert (h_mod - lfm3_fast.hidden(probe.model, ids, adapters=())).abs().max().item() > 1e-4, family

    # Step 0: grad_B != 0, grad_A == 0 exactly; Step 1 (after B moves): grad_A != 0.
    opt = learner.build_optimizer(lr=1e-2)
    h = lfm3_fast.hidden(learner.model, ids)
    loss = lfm3_fast.linear_nll(h[:-1], learner.model.lm_head.weight, ids[1:]).mean()
    loss.backward()
    assert all(leaf.b.grad is not None and leaf.b.grad.abs().sum().item() > 0
               for b in learner.bundles for _, leaf in b.iter_leaves())
    assert all(leaf.a.grad is not None and leaf.a.grad.abs().sum().item() == 0
               for b in learner.bundles for _, leaf in b.iter_leaves())
    assert learner.trainable_rows.rows.grad is not None and learner.trainable_rows.rows.grad.abs().sum().item() > 0
    opt.step()
    opt.zero_grad(set_to_none=True)

    h2 = lfm3_fast.hidden(learner.model, ids)
    lfm3_fast.linear_nll(h2[:-1], learner.model.lm_head.weight, ids[1:]).mean().backward()
    assert all(leaf.a.grad is not None and leaf.a.grad.abs().sum().item() > 0
               for b in learner.bundles for _, leaf in b.iter_leaves())


def test_bf16_factored_linear_keeps_the_fp32_accumulator_and_rounds_once():
    """The BF16 tensor-core branch (CPU BF16 here): W0 x keeps its FP32 accumulator, the BF16 low-rank delta is added in
    FP32 and the sum is rounded once. A double-rounding mutant (round W0 x to BF16 first) or a scale applied twice fails
    the exact comparison."""
    from natlang_neuralese.model.native_adapters import factored_linear
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(37, 48, generator=gen).to(torch.bfloat16)
    w = (torch.randn(80, 48, generator=gen) * 0.2).to(torch.bfloat16)
    a = torch.randn(8, 48, generator=gen) / 7
    b = torch.randn(80, 8, generator=gen) * 0.1
    s = 2.0
    base32 = x.float() @ w.float().t()
    delta = torch.nn.functional.linear(torch.nn.functional.linear(x, a.to(torch.bfloat16)), b.to(torch.bfloat16))
    expected = (base32 + s * delta.float()).to(torch.bfloat16)
    got = factored_linear(x, w, a, b, s)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got, expected)
    double_rounded = (base32.to(torch.bfloat16).float() + s * delta.float()).to(torch.bfloat16)
    assert not torch.equal(double_rounded, expected)          # the test can tell the two apart
    # B = 0 with the lora+rows base mode equals the bare base in the same mode exactly
    zero = factored_linear(x, w, a, torch.zeros_like(b), s)
    assert torch.equal(zero, factored_linear(x, w, fp8_base='f32acc'))
    assert torch.equal(zero, base32.to(torch.bfloat16))
    # gradients reach A, B and x through the BF16 branch
    xg, ag, bg = x.clone().requires_grad_(True), a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    factored_linear(xg, w, ag, bg, s).float().pow(2).sum().backward()
    assert xg.grad.dtype == torch.bfloat16 and ag.grad.abs().sum() > 0 and bg.grad.abs().sum() > 0


def test_bf16_learner_b0_identity_reference_and_oracle():
    """A BF16 learner (the production dtype) on the CPU: B = 0 is the bare base exactly; trained factors stay within BF16
    tolerance of the FP32 merged oracle."""
    learner = _make_learner(rank=4, alpha=8.0, seed=11, dtype=torch.bfloat16)
    ids = torch.tensor([1, 5, 12, 58, 7, 59, 9, 14, 2, 3, 6, 6, 8], dtype=torch.long)
    assert torch.equal(_logp(learner.model, ids), _logp(learner.model, ids, adapters=(), trainable_rows=False))
    _randomize_b(learner, 0.05, seed=3)
    _h, logits = _eager_fp32_oracle_forward(learner, ids)
    oracle = torch.log_softmax(logits[:-1].float(), -1).gather(1, ids[1:, None])[:, 0]
    got = _logp(learner.model, ids)
    assert (got - oracle).abs().max().item() < 0.1
    assert (got - _logp(learner.model, ids, adapters=())).abs().max().item() > 1e-3


def test_compiled_layers_match_eager_forward_and_gradients():
    """The compiled path (torch.compile of the four layer kernels, fullgraph) on the CPU in FP32: same hidden states and
    trainable-leaf gradients as eager."""
    results = {}
    for compiled in (False, True):
        learner = _make_learner(rank=4, alpha=8.0, seed=12)
        _randomize_b(learner, 0.05, seed=4)
        lfm3_fast.prepare(learner.model, compile_layers=compiled, fp8_base=False, parameterization='lora')
        ids = torch.tensor([1, 5, 12, 58, 7, 59, 9, 14, 2, 3, 6, 6, 8, 9, 10], dtype=torch.long)
        h = lfm3_fast.hidden(learner.model, ids, lengths=[9, 6])
        lfm3_fast.linear_nll(h[:-1], learner.model.lm_head.weight, ids[1:]).sum().backward()
        results[compiled] = (h.detach(), {n: t.grad.clone() for n, t in learner.named_trainable_tensors().items()})
    # FP32 with Inductor's fused reductions: summation order differs (~1e-5), a defect is orders of magnitude larger
    torch.testing.assert_close(results[True][0], results[False][0], rtol=1e-4, atol=5e-5)
    for name, grad in results[False][1].items():
        torch.testing.assert_close(results[True][1][name], grad, rtol=1e-3, atol=5e-5)


def test_frozen_base_zero_optimizer_entries_zero_drift_and_layer0_grad():
    learner = _make_learner(rank=4, alpha=12.0, seed=3)
    base_before = {name: p.detach().clone() for name, p in learner.model.named_parameters()}
    opt = learner.build_optimizer(lr=5e-3)
    opt_param_ids = {id(p) for g in opt.param_groups for p in g['params']}
    assert all(id(p) not in opt_param_ids for p in learner.model.parameters())

    ids = torch.tensor([3, 58, 11, 4, 59, 6, 8, 9], dtype=torch.long)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        h = lfm3_fast.hidden(learner.model, ids)
        lfm3_fast.linear_nll(h[:-1], learner.model.lm_head.weight, ids[1:],
                             trainable_rows=learner.trainable_rows).mean().backward()
        assert any(leaf.b.grad is not None and leaf.b.grad.abs().sum().item() > 0
                   for _, leaf in learner.bundles[0].iter_leaves())
        opt.step()

    for name, p in learner.model.named_parameters():
        assert p.grad is None
        assert torch.equal(p, base_before[name]), f'frozen param drifted: {name}'


def test_prefix_tree_multi_child_and_packing_parity_with_nonunit_scale():
    """Whole vs packed vs multi-child `levels` and `tree-mask` with scale = alpha/r = 3.0 != 1."""
    torch.manual_seed(7)
    base_learner = _make_learner(rank=4, alpha=12.0, seed=4)
    with torch.no_grad():
        for b in base_learner.bundles:
            for _, leaf in b.iter_leaves():
                leaf.b.normal_(0, 0.05)

    prefix = np.array([1, 2, 3, 4, 5, 58], dtype=np.int64)
    seqs = [
        np.concatenate([prefix, [10, 11, 12, 13]]),
        np.concatenate([prefix, [20, 21, 22, 59, 23]]),
        np.concatenate([prefix, [30, 31, 32]]),
        np.array([40, 41, 42, 43, 44, 45], dtype=np.int64),
    ]
    caps = [len(prefix), len(prefix), len(prefix), 0]

    def run_mode(mode_name):
        lr = _make_learner(rank=4, alpha=12.0, seed=4)
        lr.model.load_state_dict(base_learner.model.state_dict())
        with torch.no_grad():
            for name, t in lr.named_trainable_tensors().items():
                t.copy_(base_learner.named_trainable_tensors()[name])
        lfm3_fast.set_checkpointing(lr.model, 1000, 0.0)   # exercise checkpoint(use_reentrant=False) too!
        if mode_name == 'whole':
            loss = 0.0
            for seq, cap in zip(seqs, caps):
                t_ids = torch.from_numpy(seq)
                h = lfm3_fast.hidden(lr.model, t_ids)
                pos = torch.arange(cap, len(seq) - 1)
                loss = loss + lfm3_fast.linear_nll(h[pos], lr.model.lm_head.weight, t_ids[pos + 1],
                                                   trainable_rows=lr.trainable_rows).sum()
            loss.backward()
        elif mode_name == 'packed':
            flat = torch.cat([torch.from_numpy(s) for s in seqs])
            lengths = [len(s) for s in seqs]
            h = lfm3_fast.hidden(lr.model, flat, lengths=lengths)
            offsets = np.cumsum([0] + lengths[:-1])
            pos = torch.cat([torch.arange(int(o) + c, int(o) + len(s) - 1) for o, s, c in zip(offsets, seqs, caps)])
            tgt = torch.cat([torch.from_numpy(s[c + 1:]) for s, c in zip(seqs, caps)])
            loss = lfm3_fast.linear_nll(h[pos], lr.model.lm_head.weight, tgt,
                                        trainable_rows=lr.trainable_rows).sum()
            loss.backward()
        else:
            nodes = lfm3_fast.prefix_tree(seqs, caps, min_share=3)
            total_loss = [0.0]

            def leaf(items, run):
                lengths = [len(seqs[i]) - st for i, st in items]
                ids = torch.cat([torch.from_numpy(seqs[i][st:]) for i, st in items])
                h = run(ids, lengths, None)
                offsets = np.cumsum([0] + lengths[:-1])
                pos = torch.cat([torch.arange(int(o) + caps[i] - st, int(o) + len(seqs[i]) - st - 1)
                                 for o, (i, st) in zip(offsets, items)])
                tgt = torch.cat([torch.from_numpy(seqs[i][caps[i] + 1:]) for i, _ in items])
                l_val = lfm3_fast.linear_nll(h[pos], lr.model.lm_head.weight, tgt,
                                             trainable_rows=lr.trainable_rows).sum()
                l_val.backward()
                total_loss[0] += l_val.item()
            lfm3_fast.run_prefix_tree(
                lr.model, nodes, lambda i, a, b: torch.from_numpy(seqs[i][a:b]),
                leaf, lambda i, st: len(seqs[i]) - st, pack_tokens=16, mode=mode_name,
            )
            loss = torch.tensor(total_loss[0])
        grads = {name: t.grad.detach().clone() for name, t in lr.named_trainable_tensors().items()}
        return float(loss), grads

    ref_loss, ref_grads = run_mode('whole')
    for mode in ('packed', 'levels', 'tree-mask'):
        got_loss, got_grads = run_mode(mode)
        assert got_loss == pytest.approx(ref_loss, rel=1e-4, abs=1e-4), mode
        for name in ref_grads:
            torch.testing.assert_close(got_grads[name], ref_grads[name], rtol=1e-3, atol=1e-4,
                                       msg=lambda m, n=name, md=mode: f'{md} {n}: {m}')


def test_neuralese_passthrough_gradient_flows_to_producer_adapters_and_stops_when_detached():
    """Token 60 occurs only in the producer document, so its trainable row's gradient is the producer path's gradient:
    nonzero with pass-through, exactly zero when the imported state is detached (fixed context)."""
    learner = _make_learner(rank=4, alpha=8.0, seed=5, special_ids=(58, 59, 60))
    _randomize_b(learner, 0.05, seed=5)
    row = learner.trainable_rows.token_ids.tolist().index(60)
    consumer = np.array([1, 2, 58, 50, 51, 52, 59, 6, 7, 8], dtype=np.int64)
    producer = np.array([58, 60, 11, 12], dtype=np.int64)
    sources = {'blk': {'ids': producer, 'start': 1, 'stop': 4}}
    grads = {}
    for passthrough in (True, False):
        learner.zero_grad(set_to_none=True)
        plan = lfm3_fast.passthrough_plan([(0, [(3, 'blk')], sources)], len(consumer), frozen_all=not passthrough)
        ids = torch.cat([torch.from_numpy(consumer), torch.from_numpy(plan['documents'][0])])
        imports = lfm3_fast.passthrough_tensors(plan, 'cpu')
        h = lfm3_fast.hidden(learner.model, ids, lengths=[len(consumer), len(producer)], imports=imports)
        h[7:10].pow(2).sum().backward()                     # consumer rows after the import only
        grads[passthrough] = {n: t.grad.clone() for n, t in learner.named_trainable_tensors().items()}
    assert grads[True]['trainable_rows.rows'][row].abs().max().item() > 1e-6
    assert torch.equal(grads[False]['trainable_rows.rows'][row], torch.zeros_like(grads[False]['trainable_rows.rows'][row]))
    diffs = [float((grads[True][n] - grads[False][n]).abs().max()) for n in grads[True]]
    assert max(diffs) > 1e-4, 'pass-through must contribute nonzero producer-path gradient to adapters'


def _trained_learner(seed=6, steps=2, dtype=torch.float32, special_ids=(58, 59), override_ids=()):
    learner = _make_learner(rank=4, alpha=8.0, seed=seed, dtype=dtype, special_ids=special_ids, override_ids=override_ids)
    opt = learner.build_optimizer(lr=3e-2)
    ids = torch.tensor([2, 3, 58, 4, 5, 59, 6, 7, 8], dtype=torch.long)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        h = lfm3_fast.hidden(learner.model, ids)
        lfm3_fast.linear_nll(h[:-1], learner.model.lm_head.weight, ids[1:]).mean().backward()
        opt.step()
    return learner, opt, ids


def test_reference_view_switches_the_model_and_none_is_the_bare_base():
    fresh = _make_learner(rank=4, alpha=8.0, seed=6)
    ids = torch.tensor([2, 3, 58, 4, 5, 59, 6, 7, 8], dtype=torch.long)
    base_logp = _logp(fresh.model, ids)
    learner = _make_learner(rank=4, alpha=8.0, seed=6)
    ref_bundles, ref_rows = learner.snapshot_reference()
    assert ref_bundles == ()                                  # B = 0 at start: no adapter copy
    learner, opt, ids = _trained_learner(seed=6)
    live = _logp(learner.model, ids)
    assert (live - base_logp).abs().max().item() > 1e-3
    for args in ((None, None), ((), None), (ref_bundles, None)):
        with learner.reference_view(*args) as view:
            adapters, rows = view                             # tuple unpacking
            assert adapters == () and rows is False and view.kwargs == {'adapters': (), 'trainable_rows': False}
            assert torch.equal(_logp(learner.model, ids), base_logp)          # implicit: the bare base, not live
            assert torch.equal(_logp(learner.model, ids, **view.kwargs), base_logp)
            assert torch.equal(lfm3_fast.linear_nll(lfm3_fast.hidden(learner.model, ids)[:-1],
                                                    learner.model.lm_head.weight, ids[1:]), -base_logp)
    assert torch.equal(_logp(learner.model, ids), live)                       # restored on exit
    # a snapshot of the trained policy is the trained policy, and restores on error too
    snap_bundles, snap_rows = learner.snapshot_reference()
    assert snap_bundles != ()
    with pytest.raises(KeyError):
        with learner.reference_view(snap_bundles, snap_rows):
            assert torch.equal(_logp(learner.model, ids), live)
            raise KeyError('boom')
    assert learner.model._lfm3_adapters is learner.bundles and learner.model._lfm3_trainable_rows is learner.trainable_rows


def test_reference_pass_between_forward_and_backward_and_a_stale_graph_is_refused():
    learner, opt, ids = _trained_learner(seed=7)
    # clean gradient
    opt.zero_grad(set_to_none=True)
    lfm3_fast.linear_nll(lfm3_fast.hidden(learner.model, ids)[:-1], learner.model.lm_head.weight, ids[1:]).sum().backward()
    clean = {n: t.grad.clone() for n, t in learner.named_trainable_tensors().items()}
    # interleaved: forward, a reference pass, then the pending backward: the same gradient exactly
    opt.zero_grad(set_to_none=True)
    loss = lfm3_fast.linear_nll(lfm3_fast.hidden(learner.model, ids)[:-1], learner.model.lm_head.weight, ids[1:]).sum()
    with learner.reference_view(None, None):
        _logp(learner.model, ids)
    loss.backward()
    for n, t in learner.named_trainable_tensors().items():
        assert torch.equal(t.grad, clean[n]), n
    # an optimizer step between a forward and its backward changes the saved leaves: autograd must refuse the graph
    opt.zero_grad(set_to_none=True)
    loss = lfm3_fast.linear_nll(lfm3_fast.hidden(learner.model, ids)[:-1], learner.model.lm_head.weight, ids[1:]).sum()
    for t in learner.trainable_parameters():
        t.grad = torch.ones_like(t)
    opt.step()
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        loss.backward()


def test_linear_nll_defaults_to_the_owning_models_trainable_rows():
    learner, _opt, ids = _trained_learner(seed=8)
    start = learner.model.model.embed_tokens.weight[learner.trainable_rows.token_ids]
    assert (learner.trainable_rows.rows - start).abs().max().item() > 1e-4      # the rows moved
    h = lfm3_fast.hidden(learner.model, ids)
    w = learner.model.lm_head.weight
    implicit = lfm3_fast.linear_nll(h[:-1], w, ids[1:])
    explicit = lfm3_fast.linear_nll(h[:-1], w, ids[1:], trainable_rows=learner.trainable_rows)
    bare = lfm3_fast.linear_nll(h[:-1], w, ids[1:], trainable_rows=False)
    assert torch.equal(implicit, explicit) and not torch.equal(implicit, bare)
    nuc_implicit = lfm3_fast.linear_nll_nucleus(h[:-1], w, ids[1:], 0.9)[1]
    nuc_explicit = lfm3_fast.linear_nll_nucleus(h[:-1], w, ids[1:], 0.9, trainable_rows=learner.trainable_rows)[1]
    assert torch.equal(nuc_implicit, nuc_explicit)


def _native_state(learner, *, tied_layout=True, override_ids=()):
    """A source checkpoint in the real tied layout: output_rows = None, extra_output_rows = None (sft-v8, sft-v6)."""
    embed = learner.model.model.embed_tokens.weight.detach().to(torch.bfloat16)
    marker_ids = [i for i in learner.special_ids if i not in override_ids]
    interface = {'format': 'neuralese.native-channel/2', 'marker_ids': tuple(marker_ids),
                 'input_rows': embed[torch.tensor(marker_ids)].clone(), 'output_rows': None if tied_layout else
                 embed[torch.tensor(marker_ids)].clone(), 'sparse_row_dtype': 'torch.bfloat16'}
    if override_ids:
        interface.update(token_override_ids=tuple(override_ids),
                         extra_input_rows=embed[torch.tensor(list(override_ids))].clone(),
                         extra_output_rows=None if tied_layout else embed[torch.tensor(list(override_ids))].clone())
    backbone = {k: v.detach().to(torch.bfloat16).clone() for k, v in learner.model.state_dict().items()}
    backbone['lm_head.weight'] = backbone['model.embed_tokens.weight']
    return {'schema': 'natlang.native-checkpoint/1',
            'model_config': {'marker_ids': marker_ids, 'token_override_ids': list(override_ids)},
            'interface_state': interface, 'backbone_state': backbone, 'step': 10}


def test_merged_export_is_exact_and_handles_the_real_tied_interface_layout(tmp_path):
    learner, _opt, ids = _trained_learner(seed=9, special_ids=(58, 59, 60), override_ids=(60,))
    _randomize_b(learner, 0.1, seed=9)                        # a large delta so a scale error is visible
    learner.state = _native_state(learner, override_ids=(60,))
    # merged weights are bf16(fp32(W0) + s * B @ A) exactly (a mutant applying s twice fails here)
    backbone32, _ = learner.merged_backbone_and_interface(dtype=torch.float32)
    for name, (layer_idx, _kind, short) in learner.target_map.items():
        leaf = getattr(learner.bundles[layer_idx], short)
        expected = learner.model.state_dict()[name].float() + leaf.scale * (leaf.b.detach() @ leaf.a.detach())
        assert torch.equal(backbone32[name], expected), name
    # export with output_rows / extra_output_rows = None (the default tied layout) succeeds and keeps them None
    info = export_merged_checkpoint(learner, learner.state, tmp_path / 'merged.pt', source_sha256='a' * 64,
                                    learner_step=2, special_ids=learner.special_ids)
    assert info['learner_step'] == 2 and info['source_step'] == 10
    ckpt = torch.load(tmp_path / 'merged.pt', map_location='cpu', weights_only=False)
    iface = ckpt['interface_state']
    assert iface['output_rows'] is None and iface['extra_output_rows'] is None
    rows = learner.trainable_rows.rows.detach().to(torch.bfloat16)
    assert torch.equal(iface['input_rows'], rows[:2]) and torch.equal(iface['extra_input_rows'], rows[2:])
    special = torch.tensor(learner.special_ids)
    assert torch.equal(ckpt['backbone_state']['model.embed_tokens.weight'][special],
                       learner.state['backbone_state']['model.embed_tokens.weight'][special])
    # the merged model (interface rows spliced in, as the native engine applies them) reproduces the learner in FP32
    merged = tiny_lfm3_hf(window=4)
    mb = {k: v for k, v in backbone32.items()}
    mb['model.embed_tokens.weight'] = mb['model.embed_tokens.weight'].clone()
    mb['model.embed_tokens.weight'][special] = learner.trainable_rows.rows.detach()
    merged.load_state_dict(mb, strict=False)
    lfm3_fast.prepare(merged, compile_layers=False, fp8_base=False)
    torch.testing.assert_close(_logp(merged, ids), _logp(learner.model, ids), rtol=0, atol=2e-5)
    # untied layout: the output rows are updated too
    learner.state = _native_state(learner, tied_layout=False, override_ids=(60,))
    _b, iface2 = learner.merged_backbone_and_interface()
    assert torch.equal(iface2['output_rows'], rows[:2]) and torch.equal(iface2['extra_output_rows'], rows[2:])


def test_learner_api_used_by_the_trainers():
    learner = _make_learner(rank=4, alpha=8.0, seed=13, special_ids=(58, 59, 60), override_ids=(60,))
    learner.state = _native_state(learner, override_ids=(60,))
    assert learner.checkpoint_state is learner.state
    assert learner.special_ids == [58, 59, 60]
    assert learner.fp8_base is False
    native = learner.native_rows
    assert native['tied_head'] is True and 'lm_head.weight' not in native['names']
    assert torch.equal(native['embedding_rows'], learner.state['backbone_state']['model.embed_tokens.weight'][[58, 59, 60]])
    model = learner.model
    for name in ('lora', 'lora+rows'):
        assert lfm3_fast.resident_tokens(model, 1.0, parameterization=name) == lfm3_fast.resident_tokens(model, 1.0)
        assert lfm3_fast.set_checkpointing(model, 4096, 0.01, parameterization=name) == lfm3_fast.set_checkpointing(model, 4096, 0.01)
    assert lfm3_fast.resident_tokens(model, 1.0, parameterization='full') != lfm3_fast.resident_tokens(model, 1.0)
    with pytest.raises(ValueError):
        lfm3_fast.resident_tokens(model, 1.0, parameterization='lora-typo')
    seqs = [np.array([1, 2, 3, 4], dtype=np.int64), np.array([1, 2, 5], dtype=np.int64)]
    seen = []
    with torch.no_grad():
        lfm3_fast.run_prefix_tree(model, lfm3_fast.prefix_tree(seqs, [0, 0], 0), lambda i, a, b: torch.from_numpy(seqs[i][a:b]),
                                  lambda items, run: seen.append(run(torch.cat([torch.from_numpy(seqs[i][st:]) for i, st in items]),
                                                                    [len(seqs[i]) - st for i, st in items], None).shape[0]),
                                  lambda i, st: len(seqs[i]) - st, 16, parameterization='lora', backward=False)
    assert sum(seen) == 7
    with pytest.raises(ValueError):
        lfm3_fast.run_prefix_tree(model, [], None, None, None, 16, parameterization='full')
    assert lfm3_fast.choose_retained_gib(model, [{'start': 0, 'index': 0}], lambda i, s: 10, 16, 0.0) == lfm3_fast.PREFIX_RETAINED_MIN_GIB
    assert lfm3_fast.choose_retained_gib(model, [{'start': 0, 'index': 0}], lambda i, s: 10, 16, None) == lfm3_fast.PREFIX_RETAINED_GIB


def test_state_resume_replays_the_next_step_and_carries_trainer_state(tmp_path):
    learner, opt, ids = _trained_learner(seed=10)
    ref_bundles, ref_rows = learner.snapshot_reference()
    state_dir = tmp_path / 'learner-state'
    trainer_state = {'oom_plan': OOMRetryPlan(pack_tokens=512, activation_gib=0.0, retries=3).to_dict(),
                     'best': {'loss': 1.25, 'step': 4, 'leaves': {k: v.detach().clone() for k, v in
                                                                 learner.named_trainable_tensors().items()}},
                     'schedule': {'total': 7, 'steps': 7}}
    save_learner_state(
        state_dir, learner, opt, learner_step=2, source_checkpoint_id='base.pt', source_step=10,
        base_weights_sha256='a' * 64, objective_version='sft_v1', optimizer_config={'lr': 1e-2},
        ambient_state={'seed': 42, 'step': 2, 'p_clean': 0.75, 'temperature': 2.5},
        batch_cursor={'next_step': 2}, rl_rollout_ledger={'consumed_groups': [0, 1]},
        ref_bundles=ref_bundles, ref_rows=ref_rows, trainer_state=trainer_state,
    )
    restored = _make_learner(rank=4, alpha=8.0, seed=99)
    restored_opt = restored.build_optimizer(lr=3e-2)
    loaded = load_learner_state(state_dir, restored, restored_opt, verify_base_sha='a' * 64)
    assert loaded['manifest']['learner_step'] == 2
    assert loaded['rl_rollout_ledger'] == {'consumed_groups': [0, 1]}
    assert OOMRetryPlan.from_dict(loaded['trainer_state']['oom_plan']) == OOMRetryPlan(512, 0.0, 1, 3)
    assert loaded['trainer_state']['schedule'] == {'total': 7, 'steps': 7}
    assert loaded['trainer_state']['best']['step'] == 4
    for name, t in loaded['trainer_state']['best']['leaves'].items():
        assert torch.equal(t, trainer_state['best']['leaves'][name])
    assert restored.effective_policy_sha256('a' * 64) == learner.effective_policy_sha256('a' * 64)
    for lr_obj, opt_obj in ((learner, opt), (restored, restored_opt)):
        opt_obj.zero_grad(set_to_none=True)
        h_next = lfm3_fast.hidden(lr_obj.model, ids)
        lfm3_fast.linear_nll(h_next[:-1], lr_obj.model.lm_head.weight, ids[1:]).mean().backward()
        opt_obj.step()
    assert restored.effective_policy_sha256('a' * 64) == learner.effective_policy_sha256('a' * 64)


def test_oom_retry_restores_rng_retries_the_same_batch_and_drops_nothing():
    assert packed_adapter_ranks(32)['qkv_proj'] == 96
    assert packed_adapter_ranks(32)['gate_up_proj'] == 64
    learner = _make_learner(rank=4, alpha=4.0, seed=0)
    ids = torch.tensor([2, 3, 58, 4, 5, 59, 6, 7, 8], dtype=torch.long)
    batch = [ids, ids.flip(0)]

    def body(seen, fail_first):
        def step(pack_tokens, activation_gib):
            noise = torch.rand(3)                              # RNG drawn inside the step (e.g. a dropout or a coin)
            seen.append((pack_tokens, activation_gib, noise, [b.clone() for b in batch]))
            for index, chain in enumerate(batch):
                h = lfm3_fast.hidden(learner.model, chain)
                lfm3_fast.linear_nll(h[:-1], learner.model.lm_head.weight, chain[1:]).sum().backward()
                if len(seen) <= fail_first and index == 0:     # OOM after a partial backward
                    raise torch.OutOfMemoryError('synthetic OOM')
            return noise
        return step

    torch.manual_seed(123)
    clean_seen = []
    clean_noise = run_step_with_oom_retry(body(clean_seen, 0), learner, OOMRetryPlan(16384, 8.0, 1024))
    clean = {n: t.grad.clone() for n, t in learner.named_trainable_tensors().items()}
    learner.zero_grad()

    torch.manual_seed(123)
    seen, retried = [], []
    plan = OOMRetryPlan(pack_tokens=16384, activation_gib=8.0, min_pack_tokens=1024)
    noise = run_step_with_oom_retry(body(seen, 3), learner, plan, on_retry=lambda p, a, e: retried.append(a))
    assert plan.retries == 3 and retried == [1, 2, 3]
    assert [(p, a) for p, a, _n, _b in seen] == [(16384, 8.0), (16384, 4.0), (16384, 0.0), (8192, 0.0)]
    assert all(torch.equal(n, clean_noise) for _p, _a, n, _b in seen) and torch.equal(noise, clean_noise)
    assert all(all(torch.equal(x, y) for x, y in zip(b, batch)) for _p, _a, _n, b in seen)
    for n, t in learner.named_trainable_tensors().items():     # the failed attempts' partial gradients are gone
        assert torch.equal(t.grad, clean[n]), n
    # the shrunk plan persists: the next step starts from it
    later = []
    run_step_with_oom_retry(lambda p, a: later.append((p, a)), learner, plan)
    assert later == [(8192, 0.0)]


def test_oom_retry_releases_the_failed_graph_before_retrying():
    import gc
    import weakref
    learner = _make_learner(rank=4, alpha=4.0, seed=0)
    refs, calls = [], []

    def step(pack_tokens, activation_gib):
        calls.append(1)
        if len(calls) == 1:
            big = torch.ones(1000, requires_grad=True) * 2.0   # a graph tensor held only by the failing frame
            refs.append(weakref.ref(big))
            raise torch.OutOfMemoryError('synthetic OOM')
        gc.collect()
        return refs[0]() is None

    assert run_step_with_oom_retry(step, learner, OOMRetryPlan(64, 0.0, 1)) is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Blackwell CUDA FP8 test')
def test_fp8_frozen_base_is_opt_in_rowwise_packing_invariant_and_compiles():
    from natlang_neuralese.model.native_adapters import fp8_base_default, fp8_base_supported
    if not fp8_base_supported('cuda'):
        pytest.skip('FP8 frozen-base GEMMs require Blackwell (sm >= 12.0)')
    import os
    if os.environ.get('NATLANG_FP8_BASE') != '1':
        assert fp8_base_default('cuda') is False               # inaccurate on the real model: off by default
    model = tiny_lfm3_hf(window=4, seed=3)
    # FP8 GEMMs need multiples of 16: widen the tiny model
    spec = AdapterSpec(target_families=LFM3_LORA_FAMILIES, rank=4, alpha=4.0)
    learner = Lfm3Learner.from_model(model.to('cuda', torch.bfloat16), spec, special_ids=(58, 59), marker_ids=(58, 59),
                                     compile_layers=True, fp8_base=True)
    assert learner.fp8_base is True
    _randomize_b(learner, 0.05)
    ids = torch.randint(0, 60, (48,), device='cuda')
    other1, other2 = torch.randint(0, 60, (40,), device='cuda'), torch.randint(0, 60, (40,), device='cuda')
    pos = torch.arange(47, device='cuda')
    p1 = lfm3_fast.token_logprobs(learner.model, torch.cat([ids, other1]), pos, ids[1:], lengths=[48, 40])
    p2 = lfm3_fast.token_logprobs(learner.model, torch.cat([ids, other2]), pos, ids[1:], lengths=[48, 40])
    assert torch.equal(p1, p2)                                 # per-token scales: packed documents never couple
    h = lfm3_fast.hidden(learner.model, ids)                   # grad mode, compiled
    lfm3_fast.linear_nll(h[:-1], learner.model.lm_head.weight, ids[1:]).mean().backward()
    assert any(leaf.a.grad is not None and leaf.a.grad.abs().sum().item() > 0
               for _, leaf in learner.bundles[0].iter_leaves())


def test_weight_sync_7step_atomic_publication_and_rollback(tmp_path, monkeypatch):
    """v2 §7.1: 7-step atomic merged-export publication state machine attests all 4 routes and rolls back to k-1."""
    import json
    import sys
    from safetensors.torch import save_file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
    import grpo

    learner = _make_learner(rank=4, alpha=8.0, trainable_rows=True)
    export_dir = tmp_path / 'init-export'
    export_dir.mkdir()
    init_tensors = {k: v.detach().to(torch.bfloat16).cpu() for k, v in learner.model.named_parameters() if k != 'lm_head.weight'}
    save_file(init_tensors, str(export_dir / 'model.safetensors'), metadata={'format': 'pt'})
    (export_dir / 'config.json').write_text('{}')

    calls = []
    fail_rebind = {'on': False}
    info_state = {'policy_id': None, 'weights_sha256': None}

    def fake_post(url, body=None, timeout=600):
        endpoint = url.rsplit('/', 1)[1]
        calls.append((endpoint, body))
        if endpoint == 'rebind-policy':
            if fail_rebind['on'] and body.get('version') == 2:
                raise RuntimeError('simulated rebind failure')
            info_state['policy_id'] = body['policy_id']
            info_state['weights_sha256'] = body['weights_sha256']
            return json.dumps({'ok': True})
        return json.dumps({'success': True})

    class FakeResponse:
        def __init__(self, payload):
            self._data = json.dumps(payload).encode()
        def read(self):
            return self._data
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    def fake_urlopen(url, timeout=30):
        pid, wsha = info_state['policy_id'], info_state['weights_sha256']
        return FakeResponse({
            'rl_policy_binding': {'checkpoint_sha256': pid, 'served_weights_sha256': wsha},
            'routes': {
                'generation': {'policy_sha256': pid, 'weights_sha256': wsha},
                'decision': {'policy_sha256': pid, 'weights_sha256': wsha},
                'capture': {'policy_sha256': pid, 'weights_sha256': wsha},
                'native_fallback': {'policy_sha256': pid, 'weights_sha256': wsha, 'synchronized': True},
            },
        })

    monkeypatch.setattr(grpo, 'post', fake_post)
    monkeypatch.setattr(grpo.urllib.request, 'urlopen', fake_urlopen)

    sync = grpo.WeightSync.for_learner(
        learner, export_dir, 'http://sglang:9000', tmp_path / 'weights', front_url='http://front:8000',
    )
    done = []
    sync.snapshot()
    sync.publish(1, lambda v, ok: done.append((v, ok)))
    sync.wait()
    assert done == [(1, True)]
    assert sync.last['ok'] is True
    assert sync.last['routes_attested'] is True
    assert sync.last_good['version'] == 1
    assert (tmp_path / 'weights' / 'v-00001' / 'model.safetensors').exists()
    assert [c[0] for c in calls] == ['pause_generation', 'update_weights_from_disk', 'rebind-policy', 'continue_generation']

    calls.clear()
    fail_rebind['on'] = True
    done.clear()
    sync.snapshot()
    sync.publish(2, lambda v, ok: done.append((v, ok)))
    sync.wait()
    assert done == [(2, False)]
    assert sync.last['ok'] is False
    assert sync.last['rolled_back_to'] == 1
    assert sync.last_good['version'] == 1
    rollback_updates = [body for ep, body in calls if ep == 'update_weights_from_disk' and body['weight_version'] == '1']
    assert len(rollback_updates) == 1
    assert rollback_updates[0]['model_path'].endswith('v-00001')
    assert calls[-1][0] == 'continue_generation'



def test_lora_activation_table_is_the_measured_gb10_peak():
    """2026-11-22 GB10 measurement (r = 128, compiled, no recompute): 127 KB/token/layer of peak; eager 150 KB. A table
    near the old 34-38 KB sized packs ~3x too long for the memory they need."""
    learner = _make_learner(rank=4, alpha=8.0, seed=14)
    for arch in ('sm_121', 'sm_120', 'default'):
        for kind in ('conv', 'attention'):
            assert lfm3_fast.activation_bytes_per_token_layer(parameterization='lora', layer_kind=kind,
                                                             device_arch=arch) >= 127_000
    assert lfm3_fast.activation_bytes_per_token_layer(learner.model) == pytest.approx(
        lfm3_fast.LORA_PEAK_BYTES_PER_TOKEN_LAYER * lfm3_fast.EAGER_ACTIVATION_FACTOR)     # compile_layers=False here
    learner.model._lfm3_fast['compile'] = True
    assert lfm3_fast.activation_bytes_per_token_layer(learner.model) == lfm3_fast.LORA_PEAK_BYTES_PER_TOKEN_LAYER
    # 56 GiB on the 24-layer 1.7B model holds ~19.6k tokens unrecomputed, not ~70k
    assert 56 * 2**30 // (lfm3_fast.LORA_PEAK_BYTES_PER_TOKEN_LAYER * 24) < 20_000
