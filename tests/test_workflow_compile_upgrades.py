"""CPU unit tests for workflow compilation upgrades:
- grpo.py bounded-graded reward + pass-only conciseness tiebreaker
- tune.low_dof shared subspace basis extraction, collapse, and execution-grounded held-query adaptation
- compiler.hkb_bridge Stage-B Neuralese payloads, counterfactual outcome credit assignment, and pruning
- program_tune.py upgraded grpo/extract-basis CLI plumbing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'neuralese_data'))
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))

import grpo
import program_tune
from natlang_neuralese.tune.low_dof import (
    LowDofExecConfig,
    compose_subspace_factors,
    export_subspace_factors,
    extract_factor_basis,
    fit_subspace_execution,
)


def test_bounded_graded_reward_strict_pass_dominance():
    assert grpo.episode_training_reward(1.0, 0.2, mode='bounded-graded') == 1.0
    assert grpo.episode_training_reward(0.0, 1.0, mode='bounded-graded') == pytest.approx(0.49)
    assert grpo.episode_training_reward(0.0, 0.5, mode='bounded-graded') == pytest.approx(0.245)
    assert grpo.episode_training_reward(0.0, None, mode='bounded-graded') == 0.0
    assert grpo.episode_training_reward(0.0, -0.3, mode='bounded-graded') == 0.0
    assert grpo.episode_training_reward(0.0, 0.8, mode='binary') == 0.0
    assert grpo.episode_training_reward(0.0, 0.8, mode='graded') == pytest.approx(0.8)


def test_concise_pass_bonus_mixed_only_and_never_all_pass():
    # Mixed group with >= 2 passes and >= 1 fail: shorter pass gets full +0.05, longer pass gets +0.0, fail untouched.
    mixed = [
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 200},
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 600},
        {'quality': 0.0, 'reward': 0.3, 'sampled_tokens': 100},
    ]
    grpo.apply_concise_pass_bonus(mixed, 0.05)
    assert mixed[0]['reward'] == pytest.approx(1.05)
    assert mixed[1]['reward'] == pytest.approx(1.00)
    assert mixed[2]['reward'] == pytest.approx(0.30)

    # All-pass group: must be a no-op (prevents length-compression / </think> collapse).
    all_pass = [
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 200},
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 600},
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 900},
    ]
    grpo.apply_concise_pass_bonus(all_pass, 0.05)
    assert [e['reward'] for e in all_pass] == [1.0, 1.0, 1.0]

    # Memetic anchor group: must be a no-op.
    memetic = [
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 200, 'anchor': 0.5},
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 600, 'anchor': 0.5},
        {'quality': 0.0, 'reward': 0.0, 'sampled_tokens': 100, 'anchor': 0.5},
    ]
    grpo.apply_concise_pass_bonus(memetic, 0.05)
    assert [e['reward'] for e in memetic] == [1.0, 1.0, 0.0]


def _write_fake_factors(path: Path, seed: int, rank: int = 4, scale: float = 0.5):
    path.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    scales = {}
    for name, (out_d, in_d) in (
        ('model.layers.0.self_attn.q_proj', (16, 12)),
        ('model.layers.0.feed_forward.w1', (20, 12)),
    ):
        tensors[f'lora_B.{name}'] = torch.randn((out_d, rank), generator=g) * 0.1
        tensors[f'lora_A.{name}'] = torch.randn((rank, in_d), generator=g) * 0.1
        scales[name] = scale
    save_file(tensors, str(path / 'adapter_factors_fp32.safetensors'))
    (path / 'policy_manifest.json').write_text(json.dumps({'scales': scales}), encoding='utf-8')
    return {
        name: scale * (tensors[f'lora_B.{name}'] @ tensors[f'lora_A.{name}'])
        for name in scales
    }


def test_low_dof_basis_extraction_roundtrip_and_export(tmp_path):
    d1 = _write_fake_factors(tmp_path / 'donor1', seed=1)
    d2 = _write_fake_factors(tmp_path / 'donor2', seed=2)

    basis_plain = extract_factor_basis(
        [tmp_path / 'donor1', tmp_path / 'donor2'], max_rank=8, max_components=4, include_family_slices=False
    )
    assert basis_plain.num_components == 2

    # With default family slices enabled, even 2 donors expand into >= 3 orthonormal structured directions.
    basis = extract_factor_basis([tmp_path / 'donor1', tmp_path / 'donor2'], max_rank=8, max_components=4)
    assert basis.num_components >= 3

    saved = basis.save(tmp_path / 'basis')
    reloaded = basis.load(saved)
    assert reloaded.num_components == basis.num_components

    # Each donor lies exactly in the 2-component span: projecting d1 onto components recovers d1 exactly.
    flat_d1 = torch.cat([
        (reloaded.left_u[n].T @ d1[n] @ reloaded.right_v[n]).reshape(-1)
        for n in reloaded.target_names
    ])
    comp_mat = torch.stack([
        torch.cat([reloaded.components[n][k].reshape(-1) for n in reloaded.target_names])
        for k in range(reloaded.num_components)
    ], dim=0)
    coeffs = comp_mat @ flat_d1
    composed = compose_subspace_factors(reloaded, coeffs)
    for name, target_delta in d1.items():
        b, a = composed[name]
        assert torch.allclose(b @ a, target_delta, atol=1e-4)

    out_dir = export_subspace_factors(reloaded, coeffs, tmp_path / 'exported')
    assert (out_dir / 'adapter_factors_fp32.safetensors').exists()
    manifest = json.loads((out_dir / 'policy_manifest.json').read_text())
    assert len(manifest['subspace_coeffs']) == reloaded.num_components


def test_low_dof_execution_accum_matches_full_batch_and_rejects_overfit():
    from natlang_neuralese.tune.low_dof import fit_subspace_execution_accum

    true_dir = torch.tensor([1.0, -0.5, 0.25, 0.0])
    examples = []
    g = torch.Generator().manual_seed(7)
    for task_idx in range(8):
        pos = true_dir + 0.25 * torch.randn((4,), generator=g)
        neg = -true_dir + 0.25 * torch.randn((4,), generator=g)
        examples.append({'group': f'task-{task_idx}', 'ref_lp': 0.0, 'turns': [{'feat': pos}], 'reward': 1.0})
        examples.append({'group': f'task-{task_idx}', 'ref_lp': 0.0, 'turns': [{'feat': neg}], 'reward': 0.0})

    cfg = LowDofExecConfig(steps=16, lr=0.1, kl_beta=0.02, prox_lambda=0.01, eval_every=4)
    res_full = fit_subspace_execution(
        lambda c, b: (torch.stack([e['turns'][0]['feat'] for e in b]) @ c, torch.zeros(len(b))),
        examples, num_components=4, cfg=cfg,
    )
    res_acc = fit_subspace_execution_accum(
        lambda c, t: t['feat'] @ c,
        examples, num_components=4, cfg=cfg,
    )
    assert res_acc.accepted is True
    assert torch.allclose(res_acc.coeffs_best, res_full.coeffs_best, atol=1e-4)


def test_low_dof_execution_accepts_signal_and_rejects_overfit():
    true_dir = torch.tensor([1.0, -0.5, 0.25, 0.0])

    # Clean signal: every task group has both a positive-alignment (pass) and negative-alignment (fail) rollout.
    examples = []
    g = torch.Generator().manual_seed(7)
    for task_idx in range(8):
        pos = true_dir + 0.25 * torch.randn((4,), generator=g)
        neg = -true_dir + 0.25 * torch.randn((4,), generator=g)
        examples.append({'group': f'task-{task_idx}', 'feat': pos, 'reward': 1.0})
        examples.append({'group': f'task-{task_idx}', 'feat': neg, 'reward': 0.0})

    def score_fn(coeffs, batch):
        feats = torch.stack([ex['feat'] for ex in batch], dim=0)
        ref = torch.zeros((len(batch),), dtype=torch.float32)
        pol = feats @ coeffs
        return pol, ref

    res = fit_subspace_execution(
        score_fn,
        examples,
        num_components=4,
        cfg=LowDofExecConfig(steps=20, lr=0.1, kl_beta=0.02, prox_lambda=0.01, eval_every=4),
    )
    assert res.accepted is True
    assert res.best_query_score > res.base_query_score
    assert float(torch.nn.functional.cosine_similarity(res.coeffs_best[None], true_dir[None]).item()) > 0.7

    # Adversarial overfit case: query labels are inverted relative to support -> mandatory gate MUST reject.
    from natlang_neuralese.tune.latent_ttt import split_support_and_query

    sup, qry = split_support_and_query(examples, query_fraction=0.25)
    sup_groups = {e['group'] for e in sup}
    flipped = [
        {**e, 'reward': (e['reward'] if e['group'] in sup_groups else (1.0 - e['reward']))}
        for e in examples
    ]
    res_bad = fit_subspace_execution(
        score_fn,
        flipped,
        num_components=4,
        cfg=LowDofExecConfig(steps=20, lr=0.1, kl_beta=0.02, prox_lambda=0.01, eval_every=4),
    )
    assert res_bad.accepted is False
    assert torch.allclose(res_bad.coeffs_best, torch.zeros(4))


def test_hkb_bridge_neuralese_payloads_credit_assignment_and_pruning(tmp_path):
    try:
        from natlang_neuralese.compiler.hkb_bridge import ExperienceEntry, HeldAdmission, ScopedExperienceKB
    except Exception as exc:
        pytest.skip(f'hindsight-kb unavailable: {exc}')

    admission = HeldAdmission(known=frozenset({'t1', 't2'}), excluded=frozenset())
    kb_path = tmp_path / 'exp.sqlite'
    q = torch.zeros(128)
    q[0] = 1.0
    v = torch.ones(256) * 0.1

    with ScopedExperienceKB(kb_path, admission=admission, key_dim=128, value_dim=256) as kb:
        kb.add(ExperienceEntry(
            record_id='good-rec', scope='fn:route', key=q, value=v,
            text='good idiom', task_ids=('t1',), strength=0.5, compacted_state_id='nzs1_good',
        ))
        kb.add(ExperienceEntry(
            record_id='bad-rec', scope='fn:route', key=q, value=v,
            text='bad hint', task_ids=('t2',), strength=0.5, compacted_state_id=None,
        ))
        payload = kb.retrieve_neuralese_payloads(q, scopes=('fn:route',), top_k=4, threshold=0.1)
        assert len(payload['blocks']) == 1 and payload['blocks'][0]['compacted_state_id'] == 'nzs1_good'
        assert len(payload['text_fallbacks']) == 1 and payload['text_fallbacks'][0]['record_id'] == 'bad-rec'

        # Positive lift strengthens good-rec; negative lift weakens bad-rec until pruned.
        for _ in range(5):
            g_out = kb.record_outcomes([('fn:route', 'good-rec')], reward=1.0, baseline_reward=0.2)
            b_out = kb.record_outcomes([('fn:route', 'bad-rec')], reward=0.0, baseline_reward=0.6)
        assert g_out['fn:route:good-rec']['new_strength'] > 0.5
        assert b_out['fn:route:bad-rec']['new_strength'] < 0.5

        pruned = kb.prune_weak(min_injections=5, min_ema_lift=0.0)
        assert pruned == [('fn:route', 'bad-rec')]
        remaining = kb.outcome_stats('fn:route')
        assert [r['record_id'] for r in remaining] == ['good-rec']


def test_program_tune_extract_basis_and_grpo_dry_run(tmp_path):
    _write_fake_factors(tmp_path / 'f1', seed=10)
    _write_fake_factors(tmp_path / 'f2', seed=11)
    rc = program_tune.main([
        'extract-basis',
        '--factors', str(tmp_path / 'f1'), str(tmp_path / 'f2'),
        '--out', str(tmp_path / 'basis_out'),
        '--max-rank', '8', '--max-components', '4',
    ])
    assert rc == 0
    assert (tmp_path / 'basis_out' / 'subspace_basis.json').exists()

    prog = tmp_path / 'prog'
    (prog / 'dev').mkdir(parents=True)
    (prog / 'dev' / 'manifest.json').write_text('{}', encoding='utf-8')
    (prog / 'split.json').write_text(json.dumps({'family': 'test_fam', 'ids': {'support': ['a', 'b']}}))
    ckpt = tmp_path / 'ckpt.pt'
    ckpt.write_bytes(b'fake')
    rc2 = program_tune.main([
        'grpo', '--dir', str(prog), '--checkpoint', str(ckpt), '--checkpoint-sha256', '0' * 64,
        '--reward', 'bounded-graded', '--lora-rank', '8', '--concise-pass-bonus', '0.05', '--dry-run',
    ])
    assert rc2 == 0
    launch = json.loads((prog / 'grpo' / 'launch.json').read_text())
    cmd = launch['command']
    assert '--reward' in cmd and 'bounded-graded' in cmd
    assert '--lora-rank' in cmd and '8' in cmd
    assert '--concise-pass-bonus' in cmd and '0.05' in cmd


class _TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [1 + (ord(ch) % 30) for ch in text[:32]] or [1]}


class _TinyEngine:
    def __init__(self, embed_dim=32):
        self.tokenizer = _TinyTokenizer()
        g = torch.Generator().manual_seed(42)
        self.embed = torch.nn.Embedding(32, embed_dim)
        with torch.no_grad():
            self.embed.weight.copy_(torch.randn((32, embed_dim), generator=g) * 0.2)
        port = type('Port', (), {'input_embedding_weight': self.embed.weight})()
        self.shared_model = type('Model', (), {'port': port})()


def test_hkb_two_mode_encoding_and_alternating_em_reproduction(tmp_path):
    try:
        from natlang_neuralese.compiler.hkb_bridge import ExperienceEntry, HeldAdmission, ScopedExperienceKB
    except Exception as exc:
        pytest.skip(f'hindsight-kb unavailable: {exc}')

    from natlang_neuralese.compiler.hkb_em import (
        EntryOptConfig,
        HindsightEncodeConfig,
        ReproductionConfig,
        hindsight_encode_step,
        optimize_kb_entry,
        reproduction_step,
    )
    from natlang_neuralese.compiler.live_kb_writer import LiveExperienceWriter

    torch.manual_seed(0)
    engine = _TinyEngine(embed_dim=32)
    writer = LiveExperienceWriter(embed_dim=32, value_dim=16, key_dim=8, num_heads=2, hidden_dim=32)
    opt = torch.optim.AdamW(writer.parameters(), lr=1e-2)

    # Mode 1: cross_peer strictly rejects self-leakage (source_task_id == target_task_id).
    with pytest.raises(ValueError, match='cross_peer leakage'):
        hindsight_encode_step(
            writer, engine, opt,
            [{'source_text': 'src', 'target': torch.zeros(16), 'source_task_id': 't1', 'target_task_id': 't1'}],
            lambda z, k, tgts: (z - torch.stack(tgts)).pow(2).mean(),
            cfg=HindsightEncodeConfig(mode='cross_peer'),
        )

    # Mode 1: valid cross_peer step reduces loss cleanly.
    tgt_vec = torch.ones(16) * 0.3
    m1 = hindsight_encode_step(
        writer, engine, opt,
        [{
            'source_text': 'repair lesson alpha',
            'query_text': 'call context beta',
            'target': tgt_vec,
            'source_task_id': 't1',
            'target_task_id': 't2',
        }],
        lambda z, k, tgts: (z - torch.stack(tgts)).pow(2).mean(),
        cfg=HindsightEncodeConfig(mode='cross_peer'),
    )
    assert m1['total_loss'] > 0.0

    # Mode 2: E-step (direct KB entry drift on execution signal) -> M-step (encoder reproduces drifted entry).
    admission = HeldAdmission(known=frozenset({'t1'}), excluded=frozenset())
    kb_path = tmp_path / 'em.sqlite'
    with ScopedExperienceKB(kb_path, admission=admission, key_dim=8, value_dim=16) as kb:
        with torch.no_grad():
            z0, k0 = writer.encode_experience_texts(engine, ['fix tool retry loop'])
        kb.add(ExperienceEntry(
            record_id='entry-1', scope='fn:state_games', key=k0[0], value=z0[0],
            text='fix tool retry loop', task_ids=('t1',), strength=0.8,
        ))

        target_v = torch.linspace(-0.5, 0.5, 16)
        target_q = torch.nn.functional.normalize(torch.linspace(1.0, -1.0, 8), dim=-1)
        g = torch.Generator().manual_seed(9)
        examples = []
        for idx in range(8):
            pos = target_v + 0.1 * torch.randn((16,), generator=g)
            neg = -target_v + 0.1 * torch.randn((16,), generator=g)
            examples.append({'group': f'g{idx}', 'feat': pos, 'query_key': target_q, 'reward': 1.0})
            examples.append({'group': f'g{idx}', 'feat': neg, 'query_key': -target_q, 'reward': 0.0})

        def score_fn(val, key, batch):
            feats = torch.stack([b['feat'] for b in batch], dim=0)
            return feats @ val, torch.zeros((len(batch),), dtype=torch.float32)

        e_res = optimize_kb_entry(
            z0[0], k0[0], score_fn, examples,
            cfg=EntryOptConfig(steps=20, lr=0.1, key_lr=0.1, kl_beta=0.01, prox_lambda=0.01, eval_every=4),
            kb=kb, scope='fn:state_games', record_id='entry-1',
        )
        assert e_res.accepted is True
        assert e_res.drift_norm > 0.0
        drifted = kb.iter_drifted_entries(scope='fn:state_games', only_unreproduced=True)
        assert len(drifted) == 1 and drifted[0]['drift_step'] == 1

        # M-step: encoder reproduces the drifted entry and marks it reproduced.
        m_res = reproduction_step(
            writer, engine, opt, kb,
            scope='fn:state_games',
            cfg=ReproductionConfig(steps=30, key_weight=0.5, mark_after=True),
        )
        assert m_res['entries'] == 1.0
        assert m_res['final_loss'] < m_res['initial_loss'] * 0.25
        assert m_res['marked'] == 1.0
        assert kb.iter_drifted_entries(scope='fn:state_games', only_unreproduced=True) == []


def test_run_t2_em_flags_and_program_tune_adapt_subspace_cli():
    from natlang_neuralese.compiler import run_t2
    import inspect
    assert hasattr(run_t2, '_em_entry_drift_and_repro')
    sig = inspect.signature(run_t2._em_entry_drift_and_repro)
    assert 'repro_optimizer' not in sig.parameters  # single MasterAdamW only
    assert 'support_batch' in sig.parameters and 'query_batch' in sig.parameters and 'repro_weight' in sig.parameters
    assert hasattr(program_tune, 'adapt_subspace_cmd')
    assert hasattr(program_tune, '_lower_subspace_rollout_turns')


def test_compiler_fixes_1_to_5(tmp_path):
    # Fix #1: NativeQueryProjector distills cleanly to teacher 128-d query targets & PerCall accepts projector without writer.
    from natlang_neuralese.compiler.native_query import NativeQueryProjector, distill_native_query_projector
    torch.manual_seed(0)
    proj = NativeQueryProjector(tap_dims=(16, 16, 16), key_dim=128, hidden_dim=32)
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-2)
    taps = [torch.randn(4, 5, 16) for _ in range(3)]
    teacher_q = torch.nn.functional.normalize(torch.randn(4, 128), dim=-1)
    first = distill_native_query_projector(proj, opt, taps, teacher_q)
    for _ in range(25):
        last = distill_native_query_projector(proj, opt, taps, teacher_q)
    assert last['mean_cosine'] > first['mean_cosine'] + 0.3

    # Fix #2: run_t2.native_state_refs extracts referenced block IDs cleanly.
    from natlang_neuralese.compiler import run_t2
    msgs = [{'role': 'user', 'content': [{'type': 'native_state', 'id': 'nzs1_aabbcc'}]}]
    assert run_t2.native_state_refs(msgs) == ['nzs1_aabbcc']

    # Fix #3: neuralese_variants AUTHORED covers all 4 active workflows & program_tune forwards flags.
    sys.path.insert(0, str(REPO / 'scripts' / 'rl_tasks'))
    import neuralese_variants
    for fam in (
        'ext_decisions_tally',
        'ext_decisions_folder_route',
        'auth_multi_round_state_games',
        'ext_natlang-codebase_natlang_codebase_mentions',
    ):
        assert fam in neuralese_variants.AUTHORED

    # Fix #4 & #5: project_factor_cores + adapter_distillation_loss + tap_source='hidden' train cleanly.
    from natlang_neuralese.compiler.compiled_site import (
        CompiledSiteConfig,
        CompiledSiteHead,
        adapter_distillation_loss,
        project_factor_cores,
    )
    _write_fake_factors(tmp_path / 'win_lora', seed=3, rank=4, scale=1.0)
    targets = ('model.layers.0.self_attn.q_proj', 'model.layers.0.feed_forward.w1')
    left_b = {
        'model.layers.0.self_attn.q_proj': torch.eye(16, 4),
        'model.layers.0.feed_forward.w1': torch.eye(20, 4),
    }
    right_b = {
        'model.layers.0.self_attn.q_proj': torch.eye(12, 4),
        'model.layers.0.feed_forward.w1': torch.eye(12, 4),
    }
    tgt_cores = project_factor_cores(str(tmp_path / 'win_lora'), left_b, right_b)
    assert set(tgt_cores.keys()) == set(targets)

    cfg = CompiledSiteConfig(
        layout=(('layers.0.v', (8,)),),
        targets=targets,
        rank=4,
        num_slots=4,
        value_dim=32,
        key_dim=16,
        num_heads=4,
        adapter_heads=4,
        adapter_inner_rank=0,
        tap_source='hidden',
        hidden_tap_dims=(16, 16),
    )
    head = CompiledSiteHead(cfg)
    h_opt = torch.optim.AdamW(head.parameters(), lr=2e-2)
    init_vals = {'layers.0.v': torch.zeros(3, 8)}
    hidden_taps = [torch.randn(1, 3, 16), torch.randn(1, 3, 16)]
    loss0 = float(adapter_distillation_loss(head, init_vals, tgt_cores, hidden_taps=hidden_taps).item())
    for _ in range(25):
        h_opt.zero_grad(set_to_none=True)
        loss = adapter_distillation_loss(head, init_vals, tgt_cores, hidden_taps=hidden_taps)
        loss.backward()
        h_opt.step()
    loss1 = float(adapter_distillation_loss(head, init_vals, tgt_cores, hidden_taps=hidden_taps).item())
    assert loss1 < loss0 * 0.25

    # Regression for P0 #4: project_factor_cores normalizes 'port.hf.' vs bare HF target names and fails closed on 0 matches.
    prefixed_left = {f'port.hf.{k}': v for k, v in left_b.items()}
    prefixed_right = {f'port.hf.{k}': v for k, v in right_b.items()}
    prefixed_cores = project_factor_cores(str(tmp_path / 'win_lora'), prefixed_left, prefixed_right)
    assert set(prefixed_cores.keys()) == set(prefixed_left.keys())
    with pytest.raises(ValueError, match='matched 0 targets'):
        project_factor_cores(str(tmp_path / 'win_lora'), {'bogus': torch.eye(4)}, {'bogus': torch.eye(4)})

    # Regression for P0 #3: PplxWarmstartWriter does NOT shadow encode_experience_blocks with single-block shim.
    import inspect
    from natlang_neuralese.compiler.live_kb_writer import PplxWarmstartWriter
    src = inspect.getsource(PplxWarmstartWriter.encode_experience_blocks)
    assert 'writer_encoder.encode_blocks' in src


def test_four_design_improvements_end_to_end(tmp_path):
    # Improvement #2: sealed_fanin_variant enforces information bottleneck + total_tokens pass-only conciseness.
    sys.path.insert(0, str(REPO / 'scripts' / 'rl_tasks'))
    import neuralese_variants
    fake_row = {
        'id': 'ext:test:1',
        'family': 'ext_decisions_tally',
        'curriculum': {'external': {'framing': 'plain'}},
        'semantics': {
            'root': 'main.nl',
            'inputs': {'items': ['a', 'b'], 'rubric': 'r'},
            'files': {'main.nl': '---\nargs:\n  items: "string[]"\nreturns: "number"\n---\nCount matching items.\n'},
        },
    }
    h_name, h_args, h_instr, h_note = neuralese_variants.AUTHORED['ext_decisions_tally']
    sealed = neuralese_variants.sealed_fanin_variant(
        fake_row, f'{h_name}.nl', neuralese_variants.helper_file(h_args, h_instr), h_note, max_note_tokens=96,
    )
    assert sealed['curriculum']['external']['framing'] == neuralese_variants.SEALED_FRAMING
    assert sealed['curriculum']['external']['neuralese_bottleneck'] is True
    assert '96 tokens' in sealed['semantics']['files'][f'main/{h_name}.nl']
    assert 'Information-bottleneck contract' in sealed['semantics']['files']['main.nl']

    eps = [
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 300, 'total_tokens': 600},
        {'quality': 1.0, 'reward': 1.0, 'sampled_tokens': 300, 'total_tokens': 1600},
        {'quality': 0.0, 'reward': 0.0, 'sampled_tokens': 100, 'total_tokens': 400},
    ]
    grpo.apply_concise_pass_bonus(eps, 0.05)
    assert eps[0]['reward'] == pytest.approx(1.05)
    assert eps[1]['reward'] == pytest.approx(1.00)

    # Improvement #4: extract_grpo_decision_rows + calibrate-decisions --from-grpo-work --attach-manifest.
    work = tmp_path / 'grpo_work'
    (work / 'decisions').mkdir(parents=True)
    (work / 'decisions' / 'step-00001.jsonl').write_text(
        '\n'.join([
            json.dumps({'option_scores': [3.0, -1.0], 'chosen': 0, 'quality': 1.0, 'labels': ['yes', 'no']}),
            json.dumps({'option_scores': [-1.0, 2.5], 'chosen': 1, 'quality': 1.0, 'labels': ['yes', 'no']}),
            json.dumps({'option_scores': [0.5, 0.4], 'chosen': 0, 'quality': 0.0, 'labels': ['yes', 'no']}),
        ]) + '\n',
        encoding='utf-8',
    )
    extracted = program_tune.extract_grpo_decision_rows(work)
    assert len(extracted) == 2
    manifest_path = tmp_path / 'natlang.adapters.json'
    manifest_path.write_text(json.dumps({'schema': 'natlang.adapters/v1', 'digest': '0' * 64}), encoding='utf-8')
    rc = program_tune.main([
        'calibrate-decisions', '--dir', str(tmp_path), '--from-grpo-work', str(work),
        '--attach-manifest', str(manifest_path), '--binding-key', 'intent',
    ])
    assert rc == 0
    updated_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert 'intent' in updated_manifest['decisionCalibration']
    assert updated_manifest['digest'] != '0' * 64

    # Improvement #3: SharedBasisBCADispatcher exact match to dense (U @ C @ V^T) across mixed concurrent sites.
    from natlang_neuralese.serve.shared_basis_bca import SharedBasisBCADispatcher
    torch.manual_seed(1)
    u = torch.randn(10, 4)
    v = torch.randn(8, 4)
    disp = SharedBasisBCADispatcher({'proj': u}, {'proj': v})
    c1 = torch.randn(4, 4)
    c2 = torch.randn(4, 4)
    disp.register_site('siteA', {'proj': c1})
    disp.register_site('siteB', {'proj': c2})
    x = torch.randn(3, 5, 8)
    out = disp.apply_delta('proj', x, ['siteA', None, 'siteB'], scales=[1.0, 1.0, 0.5])
    assert torch.allclose(out[0], x[0] @ (u @ c1 @ v.T).T, atol=1e-5)
    assert torch.allclose(out[1], torch.zeros(5, 10), atol=1e-6)
    assert torch.allclose(out[2], 0.5 * (x[2] @ (u @ c2 @ v.T).T), atol=1e-5)
