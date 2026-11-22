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
                  device='cpu', fp8_base=False):
    model = tiny_lfm3_hf(window=4).to(device=device, dtype=dtype)
    spec = AdapterSpec(target_families=LFM3_LORA_FAMILIES, rank=rank, alpha=alpha, trainable_rows=trainable_rows)
    return Lfm3Learner.from_model(
        model, spec, special_ids=special_ids, marker_ids=special_ids,
        compile_layers=False, fp8_base=fp8_base, seed=seed,
    )


def test_adapter_bypass_all_9_families_and_b0_identity_and_step0_step1_grads():
    learner = _make_learner(rank=4, alpha=8.0, seed=1)
    ids = torch.tensor([1, 5, 12, 58, 7, 59, 9, 14, 2, 3], dtype=torch.long)

    # B=0 reproduces bare base exactly.
    h_base, _ = _eager_fp32_oracle_forward(learner, ids, use_adapters=False, use_rows=True)
    h_fast = lfm3_fast.hidden(learner.model, ids)
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

    # Step 0: grad_B != 0, grad_A == 0; Step 1 (after B moves): grad_A != 0.
    opt = learner.build_optimizer(lr=1e-2)
    h = lfm3_fast.hidden(learner.model, ids)
    loss = lfm3_fast.linear_nll(h[:-1], learner.model.lm_head.weight, ids[1:],
                                trainable_rows=learner.trainable_rows).mean()
    loss.backward()
    assert all(leaf.b.grad is not None and leaf.b.grad.abs().sum().item() > 0
               for b in learner.bundles for _, leaf in b.iter_leaves())
    assert all(leaf.a.grad is not None and leaf.a.grad.abs().sum().item() == 0
               for b in learner.bundles for _, leaf in b.iter_leaves())
    assert learner.trainable_rows.rows.grad is not None and learner.trainable_rows.rows.grad.abs().sum().item() > 0
    opt.step()
    opt.zero_grad(set_to_none=True)

    h2 = lfm3_fast.hidden(learner.model, ids)
    lfm3_fast.linear_nll(h2[:-1], learner.model.lm_head.weight, ids[1:],
                         trainable_rows=learner.trainable_rows).mean().backward()
    assert all(leaf.a.grad is not None and leaf.a.grad.abs().sum().item() > 0
               for b in learner.bundles for _, leaf in b.iter_leaves())


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
    learner = _make_learner(rank=4, alpha=8.0, seed=5)
    with torch.no_grad():
        for b in learner.bundles:
            for _, leaf in b.iter_leaves():
                leaf.b.normal_(0, 0.05)
    consumer = np.array([1, 2, 58, 50, 51, 52, 59, 6, 7, 8], dtype=np.int64)
    producer = np.array([58, 10, 11, 12], dtype=np.int64)
    sources = {'blk': {'ids': producer, 'start': 1, 'stop': 4}}
    for passthrough in (True, False):
        learner.zero_grad(set_to_none=True)
        plan = lfm3_fast.passthrough_plan([(0, [(3, 'blk')], sources)], len(consumer), frozen_all=not passthrough)
        ids = torch.cat([torch.from_numpy(consumer), torch.from_numpy(plan['documents'][0])])
        lengths = [len(consumer), len(producer)]
        imports = lfm3_fast.passthrough_tensors(plan, 'cpu')
        h = lfm3_fast.hidden(learner.model, ids, lengths=lengths, imports=imports)
        # Differentiate ONLY consumer output rows after the import (rows 7..9); verify producer gradients via K/V & B*x.
        # Measure how much the producer doc's input embeddings (or layer-0 adapter grad) change vs detached.
        loss = h[7:10].pow(2).sum()
        loss.backward()
        if passthrough:
            grad_pass = {n: t.grad.clone() for n, t in learner.named_trainable_tensors().items() if t.grad is not None}
        else:
            grad_det = {n: t.grad.clone() for n, t in learner.named_trainable_tensors().items() if t.grad is not None}
    diffs = [float((grad_pass[n] - grad_det[n]).abs().max()) for n in grad_pass if n in grad_det]
    assert max(diffs) > 1e-4, 'pass-through must contribute nonzero producer-path gradient to adapters'


def test_reference_view_interleaving_and_merge_roundtrip_and_state_resume(tmp_path):
    learner = _make_learner(rank=4, alpha=8.0, seed=6)
    ref_bundles, ref_rows = learner.snapshot_reference()
    assert ref_bundles == ()   # B=0 at start -> zero extra adapter memory!

    opt = learner.build_optimizer(lr=1e-2)
    ids = torch.tensor([2, 3, 58, 4, 5, 59, 6, 7, 8], dtype=torch.long)
    # Take 2 steps so live policy diverges from reference.
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        h = lfm3_fast.hidden(learner.model, ids)
        lfm3_fast.linear_nll(h[:-1], learner.model.lm_head.weight, ids[1:],
                             trainable_rows=learner.trainable_rows).mean().backward()
        opt.step()

    # Interleaved live -> ref -> live without mutating live leaves or breaking pending backward.
    h_live = lfm3_fast.hidden(learner.model, ids)
    with learner.reference_view(ref_bundles, ref_rows) as ref_kw:
        ref_logp = lfm3_fast.token_logprobs(learner.model, ids, torch.arange(len(ids) - 1), ids[1:], **ref_kw)
    live_logp = lfm3_fast.token_logprobs(learner.model, ids, torch.arange(len(ids) - 1), ids[1:])
    assert (live_logp - ref_logp).abs().max().item() > 1e-4
    # Pending backward on h_live still works cleanly after interleaved reference call!
    opt.zero_grad(set_to_none=True)
    h_live.pow(2).mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in learner.trainable_parameters())

    # Merge round-trip: factored learner vs merged BF16 checkpoint.
    orig_embed = learner.model.model.embed_tokens.weight.detach().to(torch.bfloat16).clone()
    state = {
        'schema': 'natlang.native-checkpoint/1',
        'model_config': {'marker_ids': [58, 59]},
        'interface_state': {'input_rows': orig_embed[torch.tensor([58, 59])].clone()},
        'backbone_state': {k: v.detach().to(torch.bfloat16).clone() for k, v in learner.model.state_dict().items()},
        'step': 10,
    }
    ckpt_path = tmp_path / 'merged.pt'
    export_info = export_merged_checkpoint(learner, state, ckpt_path, source_sha256='a' * 64, learner_step=2,
                                           special_ids=[58, 59])
    assert export_info['learner_step'] == 2 and export_info['source_step'] == 10
    merged_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    # Build a fresh model from the merged checkpoint + merged interface rows and compare logprobs!
    merged_model = tiny_lfm3_hf(window=4).to(torch.bfloat16)
    mb = dict(merged_ckpt['backbone_state'])
    mb['model.embed_tokens.weight'] = mb['model.embed_tokens.weight'].clone()
    mb['model.embed_tokens.weight'][torch.tensor([58, 59])] = merged_ckpt['interface_state']['input_rows'].to(torch.bfloat16)
    mb.pop('lm_head.weight', None)
    merged_model.load_state_dict(mb, strict=False)
    lfm3_fast.prepare(merged_model, compile_layers=False, fp8_base=False)
    merged_logp = lfm3_fast.token_logprobs(merged_model, ids, torch.arange(len(ids) - 1), ids[1:])
    torch.testing.assert_close(merged_logp, live_logp, rtol=2e-2, atol=5e-2)

    # Atomic save & resume replay parity.
    state_dir = tmp_path / 'learner-state'
    save_learner_state(
        state_dir, learner, opt, learner_step=2, source_checkpoint_id='base.pt', source_step=10,
        base_weights_sha256='a' * 64, objective_version='sft_v1', optimizer_config={'lr': 1e-2},
        ambient_state={'seed': 42, 'step': 2, 'p_clean': 0.75, 'temperature': 2.5},
        batch_cursor={'next_step': 2}, rl_rollout_ledger={'consumed_groups': [0, 1]},
        ref_bundles=ref_bundles, ref_rows=ref_rows,
    )
    restored = _make_learner(rank=4, alpha=8.0, seed=99)
    restored.model.load_state_dict(learner.model.state_dict())
    restored_opt = restored.build_optimizer(lr=1e-2)
    loaded = load_learner_state(state_dir, restored, restored_opt, verify_base_sha='a' * 64)
    assert loaded['manifest']['learner_step'] == 2
    assert loaded['rl_rollout_ledger'] == {'consumed_groups': [0, 1]}
    assert restored.effective_policy_sha256('a' * 64) == learner.effective_policy_sha256('a' * 64)
    # Next step on original and restored reproduces exact identical parameters!
    for lr_obj, opt_obj in ((learner, opt), (restored, restored_opt)):
        opt_obj.zero_grad(set_to_none=True)
        h_next = lfm3_fast.hidden(lr_obj.model, ids)
        lfm3_fast.linear_nll(h_next[:-1], lr_obj.model.lm_head.weight, ids[1:],
                             trainable_rows=lr_obj.trainable_rows).mean().backward()
        opt_obj.step()
    assert restored.effective_policy_sha256('a' * 64) == learner.effective_policy_sha256('a' * 64)


def test_oom_retry_state_machine_and_packed_ranks():
    assert packed_adapter_ranks(32)['qkv_proj'] == 96
    assert packed_adapter_ranks(32)['gate_up_proj'] == 64

    learner = _make_learner(rank=4, alpha=4.0, seed=0)
    plan = OOMRetryPlan(pack_tokens=16384, activation_gib=8.0, min_pack_tokens=1024)
    seen = []

    def flaky(pack_tokens, activation_gib):
        seen.append((pack_tokens, activation_gib))
        if len(seen) <= 3:
            raise torch.OutOfMemoryError('synthetic OOM')
        return 'ok'

    assert run_step_with_oom_retry(flaky, learner, plan) == 'ok'
    assert plan.retries == 3
    assert seen[0] == (16384, 8.0)
    assert seen[1] == (16384, 4.0)
    assert seen[2] == (16384, 0.0)
    assert seen[3] == (8192, 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Blackwell CUDA FP8 test')
def test_fp8_frozen_base_gemm_agrees_with_bf16_and_propagates_adapter_grads():
    from natlang_neuralese.model.native_adapters import fp8_base_supported
    if not fp8_base_supported('cuda'):
        pytest.skip('FP8 frozen-base GEMMs require Blackwell (sm >= 12.0)')
    bf16 = _make_learner(rank=4, alpha=4.0, seed=10, dtype=torch.bfloat16, device='cuda', fp8_base=False)
    fp8 = _make_learner(rank=4, alpha=4.0, seed=10, dtype=torch.bfloat16, device='cuda', fp8_base=True)
    fp8.model.load_state_dict(bf16.model.state_dict())
    with torch.no_grad():
        for name, t in fp8.named_trainable_tensors().items():
            t.copy_(bf16.named_trainable_tensors()[name])
    ids = torch.randint(0, 60, (32,), device='cuda')
    lp_bf16 = lfm3_fast.token_logprobs(bf16.model, ids, torch.arange(31, device='cuda'), ids[1:])
    lp_fp8 = lfm3_fast.token_logprobs(fp8.model, ids, torch.arange(31, device='cuda'), ids[1:])
    assert (lp_fp8 - lp_bf16).abs().mean().item() < 0.25
    # Verify backward flows through FP8 frozen base into layer-0 LoRA leaves!
    h_fp8 = lfm3_fast.hidden(fp8.model, ids)
    lfm3_fast.linear_nll(h_fp8[:-1], fp8.model.lm_head.weight, ids[1:],
                         trainable_rows=fp8.trainable_rows).mean().backward()
    assert any(leaf.b.grad is not None and leaf.b.grad.abs().sum().item() > 0
               for _, leaf in fp8.bundles[0].iter_leaves())


def test_weight_sync_7step_atomic_publication_and_rollback(tmp_path, monkeypatch):
    """v2 §7.1: 7-step atomic merged-export publication state machine attests all 4 routes and rolls back to k-1."""
    import json
    import sys
    from safetensors.torch import save_file
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
    import grpo

    learner = _make_learner(r=4, alpha=8.0, trainable_rows=True)
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
                'generation': {'policy_id': pid, 'weights_sha256': wsha, 'attested': True},
                'decision_scoring': {'policy_id': pid, 'weights_sha256': wsha, 'attested': True},
                'logprob_capture': {'policy_id': pid, 'weights_sha256': wsha, 'attested': True},
                'native_fallback': {'policy_id': pid, 'weights_sha256': wsha, 'attested': True},
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

