"""Targeted unit tests for plans/neuralese/async-train.md v2 mechanisms:
- FP32 LoRA factor export + CPU DGX FP32 merge + fail-closed SHA checks + round dedup ledger (§2, §4)
- Multi-engine policy publication (SGLang export + lfm-latest pointer) (§5)
- FP8-weight collection actor profile vs BF16 held/GRPO + Colab G4 MPS + remote rebind & 4-route attestation (§1, §3, §5, §7.1)
- JOLT+ jolt-guided tier (own > jolt-guided > teacher) + adjacent-policy own priority (§3, §8)
- GRPO rho_by_staleness telemetry, compact scored-group pack/unpack, multi-actor dispatch, JOLT+/twin defaults (§7.1, §7.2, §8)
- Async round blob-first commit, 30-min DGX takeover profile-change record, and SFT <-> GRPO PhaseController (§2, §4, §7.4)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / 'training/neuralese', REPO, REPO / 'scripts/neuralese_data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---- §2 / §4: FP32 factor export, DGX CPU FP32 merge, dedup ledger --------------------------------------------------
class _MiniLeaf(torch.nn.Module):
    def __init__(self, in_f, out_f, rank=4, alpha=4.0, seed=10):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = torch.nn.Parameter(torch.randn(rank, in_f, generator=g, dtype=torch.float32) * 0.1)
        self.b = torch.nn.Parameter(torch.randn(out_f, rank, generator=g, dtype=torch.float32) * 0.1)
        self.scale = float(alpha) / float(rank)

    def delta_fp32(self):
        return (self.b.float() @ self.a.float()) * self.scale


class _MiniLayerBundle(torch.nn.Module):
    def __init__(self, seed=10):
        super().__init__()
        self.q_proj = _MiniLeaf(16, 16, rank=4, alpha=4.0, seed=seed)
        self.k_proj = _MiniLeaf(16, 16, rank=4, alpha=4.0, seed=seed + 1)


class _MiniTrainableRows(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('token_ids', torch.tensor([3, 5, 7], dtype=torch.int64))
        self.rows = torch.nn.Parameter(torch.full((3, 16), 0.25, dtype=torch.float32))


class _FakeLearner:
    def __init__(self):
        from natlang_neuralese.train.lfm3_learner import AdapterSpec
        self.device = torch.device('cpu')
        self.spec = AdapterSpec(rank=4, alpha=4.0, target_families=('q_proj', 'k_proj'), trainable_rows=True)
        self.bundles = (_MiniLayerBundle(seed=11),)
        self.target_map = {
            'model.layers.0.self_attn.q_proj.weight': (0, 'attn', 'q_proj'),
            'model.layers.0.self_attn.k_proj.weight': (0, 'attn', 'k_proj'),
        }
        self.trainable_rows = _MiniTrainableRows()

    def named_trainable_tensors(self):
        out = {}
        for param_name, (layer_idx, _kind, short_name) in sorted(self.target_map.items()):
            leaf = getattr(self.bundles[layer_idx], short_name)
            out[f'lora_A.{param_name}'] = leaf.a
            out[f'lora_B.{param_name}'] = leaf.b
        out['trainable_rows.rows'] = self.trainable_rows.rows
        return out

    def effective_policy_sha256(self, base_sha=''):
        return hashlib.sha256(f'effective:{base_sha}'.encode()).hexdigest()


def test_factor_export_and_dgx_fp32_merge_exact_and_fail_closed(tmp_path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train.learner_state import (
        FACTOR_EXPORT_SCHEMA,
        export_adapter_factors,
        load_learner_state,
        merge_factors_to_checkpoint,
        save_learner_state,
    )

    learner = _FakeLearner()
    base_ckpt = tmp_path / 'base.pt'
    g = torch.Generator().manual_seed(0)
    embed_w = torch.randn(32, 16, generator=g, dtype=torch.bfloat16) * 0.05
    backbone = {
        'model.embed_tokens.weight': embed_w,
        'lm_head.weight': embed_w,
        'model.layers.0.self_attn.q_proj.weight': torch.randn(16, 16, generator=g, dtype=torch.bfloat16) * 0.05,
        'model.layers.0.self_attn.k_proj.weight': torch.randn(16, 16, generator=g, dtype=torch.bfloat16) * 0.05,
    }
    base_payload = {
        'schema': 'natlang.lfm3-checkpoint/1',
        'model_config': {'vocab_size': 32, 'hidden_size': 16, 'marker_ids': [3, 5], 'token_override_ids': [7]},
        'interface_state': {
            'input_rows': torch.zeros(2, 16, dtype=torch.bfloat16),
            'extra_input_rows': torch.zeros(1, 16, dtype=torch.bfloat16),
        },
        'step': 0,
        'backbone_state': backbone,
    }
    torch.save(base_payload, base_ckpt)
    base_sha = sha256_file_hex(base_ckpt)

    # Export factors in blob-first order
    factors_dir = tmp_path / 'factors'
    manifest = export_adapter_factors(
        learner, factors_dir, base_checkpoint_sha256=base_sha, learner_step=12, round_id='round-001',
    )
    assert manifest['schema'] == FACTOR_EXPORT_SCHEMA
    assert manifest['round_id'] == 'round-001'
    assert (factors_dir / 'adapter_factors_fp32.safetensors').is_file()
    assert (factors_dir / 'policy_manifest.json').is_file()

    # Merge on CPU in FP32 and verify exact bitwise match with bf16(fp32(W0) + scale * B @ A)
    merged_ckpt = tmp_path / 'merged.pt'
    result = merge_factors_to_checkpoint(base_ckpt, factors_dir, merged_ckpt)
    assert merged_ckpt.is_file()
    assert result['checkpoint_sha256'] == sha256_file_hex(merged_ckpt)
    merged = torch.load(merged_ckpt, map_location='cpu', weights_only=False)
    for short in ('q_proj', 'k_proj'):
        key = f'model.layers.0.self_attn.{short}.weight'
        w0 = backbone[key].float()
        leaf = getattr(learner.bundles[0], short)
        expected = (w0 + leaf.delta_fp32()).to(torch.bfloat16)
        assert torch.equal(merged['backbone_state'][key], expected)
    # Trainable rows applied to interface_state (input_rows + extra_input_rows) and tied lm_head preserved
    assert torch.equal(merged['interface_state']['input_rows'], torch.full((2, 16), 0.25, dtype=torch.bfloat16))
    assert torch.equal(merged['interface_state']['extra_input_rows'], torch.full((1, 16), 0.25, dtype=torch.bfloat16))
    assert torch.equal(merged['backbone_state']['lm_head.weight'], merged['backbone_state']['model.embed_tokens.weight'])

    # Fail-closed on corrupted factor safetensors
    bad_factors = tmp_path / 'bad-factors'
    bad_factors.mkdir()
    (bad_factors / 'policy_manifest.json').write_bytes((factors_dir / 'policy_manifest.json').read_bytes())
    raw = bytearray((factors_dir / 'adapter_factors_fp32.safetensors').read_bytes())
    raw[-1] ^= 0xFF
    (bad_factors / 'adapter_factors_fp32.safetensors').write_bytes(bytes(raw))
    with pytest.raises(ValueError, match='checksum mismatch'):
        merge_factors_to_checkpoint(base_ckpt, bad_factors, tmp_path / 'should_not_exist.pt')

    # Fail-closed on wrong base checkpoint
    other_base = tmp_path / 'other_base.pt'
    other_base.write_bytes(b'not-the-same-base')
    with pytest.raises(ValueError, match='base checkpoint mismatch'):
        merge_factors_to_checkpoint(other_base, factors_dir, tmp_path / 'should_not_exist2.pt')

    # Round dedup ledger persists through save_learner_state / load_learner_state
    state_dir = tmp_path / 'learner-state'
    save_learner_state(
        state_dir, learner, learner_step=12, source_checkpoint_id=str(base_ckpt), base_weights_sha256=base_sha,
        trained_rounds=['round-001', 'round-002'], trained_episode_ids=['ep-1', 'ep-2'],
    )
    loaded = load_learner_state(state_dir, learner, restore_rng=False, verify_base_sha=base_sha)
    assert loaded['trained_rounds'] == ['round-001', 'round-002']
    assert loaded['trained_episode_ids'] == ['ep-1', 'ep-2']


# ---- §5: Multi-engine policy publication -----------------------------------------------------------------------------
def test_publish_merged_policy_writes_pointer_and_invokes_exports(tmp_path, monkeypatch):
    from natlang_neuralese.export import native_hf

    ckpt = tmp_path / 'policy.pt'
    ckpt.write_bytes(b'fake-merged-checkpoint')
    calls = {}

    def fake_export(checkpoint, out_dir, *, base=None, dtype=None):
        calls['sglang'] = (str(checkpoint), str(base), str(out_dir))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return {'weights_sha256': '1' * 64, 'schema': 'natlang.sglang-hf-export/1'}

    monkeypatch.setattr(native_hf, 'export', fake_export)
    pointer = tmp_path / 'current-policy.json'
    out = native_hf.publish_merged_policy(
        ckpt, base=tmp_path / 'base', sglang_export_dir=tmp_path / 'sglang-hf',
        lfm_latest_path=pointer, label='async-r1', why='unit test',
    )
    assert out['sglang_export']['weights_sha256'] == '1' * 64
    assert pointer.is_file()
    ptr = json.loads(pointer.read_text())
    assert ptr['checkpoint']['sha256'] == hashlib.sha256(b'fake-merged-checkpoint').hexdigest()
    assert ptr['label'] == 'async-r1'


# ---- §1 / §3 / §5: FP8 collection actor profile, MPS, remote rebind + 4-route attestation ----------------------------
def test_sglang_actor_fp8_weight_dtype_binding_differs_from_bf16(monkeypatch):
    from natlang_neuralese.serve.sglang_actor import SglangEngine

    monkeypatch.setenv('NATLANG_APPEND_ONLY_PROMPTS', '0')

    def _make_engine():
        eng = SglangEngine.__new__(SglangEngine)
        eng.rl_policy_binding = {}
        eng.checkpoint_sha256 = '3' * 64
        eng.tokenizer = None
        eng.role_tokens = SimpleNamespace(produce_open=10, produce_close=11, consume_open=12, consume_close=13)
        return eng

    manifest = {
        'weights_sha256': '2' * 64, 'checkpoint_sha256': '3' * 64, 'dtype': 'bfloat16',
        'serving': {'banned_token_ids': [10, 11, 12, 13]},
    }
    fork = {'commit': 'deadbeef', 'dirty': False}
    e_bf16 = _make_engine()
    e_bf16.bind_sglang(
        'http://127.0.0.1:9460', export_manifest=manifest, fork=fork,
        projection='vendor', kv_cache_dtype='fp8_e4m3', weight_dtype='bfloat16',
    )
    e_fp8 = _make_engine()
    e_fp8.bind_sglang(
        'http://127.0.0.1:9460', export_manifest=manifest, fork=fork,
        projection='vendor', kv_cache_dtype='fp8_e4m3', weight_dtype='fp8',
    )
    assert e_bf16.rl_policy_binding['serving_engine']['dtype'] == 'bfloat16'
    assert e_fp8.rl_policy_binding['serving_engine']['dtype'] == 'fp8'
    assert e_fp8.rl_policy_binding['serving_engine']['conv_and_head_dtype'] == 'bfloat16'
    assert e_fp8.sglang_actor_export['format'].endswith('-w-fp8')
    assert not e_bf16.sglang_actor_export['format'].endswith('-w-fp8')
    with pytest.raises(ValueError, match='weight_dtype'):
        e_bf16.bind_sglang(
            'http://127.0.0.1:9460', export_manifest=manifest, fork=fork,
            projection='vendor', weight_dtype='int4',
        )


def test_pop_rl_actor_fp8_colab_command_mps_and_rebind(tmp_path, monkeypatch):
    import pop_rl_actor

    cfg = pop_rl_actor.host_config('colab_g4', host='g4-a')
    cmd_fp8, claim, sglang_fp8, fmt_fp8 = pop_rl_actor.colab_serve_command(
        cfg, checkpoint='/content/ckpt.pt', sha='a' * 64, port=18531, work='/content/work',
        beside_learner=True, kv_cache_dtype='fp8_e4m3', weight_dtype='fp8',
    )
    assert '--sglang-weight-dtype' in cmd_fp8 and 'fp8' in cmd_fp8
    assert '--sglang-kv-gb' in cmd_fp8
    assert fmt_fp8.endswith('-w-fp8')
    assert sglang_fp8['weight_dtype'] == 'fp8'
    assert claim['vram_gib'] < 96

    # BF16 keeps original format without -w-fp8
    cmd_bf16, _, sglang_bf16, fmt_bf16 = pop_rl_actor.colab_serve_command(
        cfg, checkpoint='/content/ckpt.pt', sha='a' * 64, port=18531, work='/content/work',
        beside_learner=False, kv_cache_dtype='fp8_e4m3', weight_dtype='bfloat16',
    )
    assert '--sglang-weight-dtype' not in cmd_bf16
    assert not fmt_bf16.endswith('-w-fp8')
    assert 'weight_dtype' not in sglang_bf16

    # MPS check respects opt-out and SSH call
    monkeypatch.setenv('NATLANG_COLAB_MPS', '0')
    assert pop_rl_actor.ensure_colab_mps('g4-a')['mps'] == 'disabled'
    monkeypatch.setenv('NATLANG_COLAB_MPS', '1')
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', lambda h, cmd, **kw: 'started\n')
    assert pop_rl_actor.ensure_colab_mps('g4-a')['mps'] == 'started'

    # Local rebind_actor calls pause -> update_weights_from_disk -> rebind-policy -> info/attest -> continue
    exp_dir = tmp_path / 'hf-v2'
    exp_dir.mkdir()
    (exp_dir / 'model.safetensors').write_bytes(b'fake-safetensors')
    w_sha = hashlib.sha256(b'fake-safetensors').hexdigest()
    pol_id = 'b' * 64
    calls = []

    def fake_http(url, payload=None, *, host=None, timeout=600):
        calls.append((url, payload))
        if url.endswith('/update_weights_from_disk'):
            return {'success': True}
        if url.endswith('/v1/neuralese/info'):
            route = {'measured': True, 'policy_sha256': pol_id, 'version': 2,
                     'model_path': str(exp_dir), 'weights_sha256': w_sha}
            return {
                'routes': {
                    'generation': route, 'decision': route, 'capture': route,
                    'native_fallback': {'policy_sha256': pol_id, 'version': 2, 'resident': False,
                                        'source_weights_sha256': w_sha, 'pending_tensor_identity': None},
                }
            }
        return {'ok': True}

    monkeypatch.setattr(pop_rl_actor, '_http_json', fake_http)
    res = pop_rl_actor.rebind_actor(
        front_url='http://127.0.0.1:8460', policy_id=pol_id, version=2, export_dir=exp_dir,
    )
    assert res['routes_attested'] is True
    urls = [u for u, _ in calls]
    assert urls == [
        'http://127.0.0.1:9460/pause_generation',
        'http://127.0.0.1:9460/update_weights_from_disk',
        'http://127.0.0.1:8460/v1/neuralese/rebind-policy',
        'http://127.0.0.1:8460/v1/neuralese/info',
        'http://127.0.0.1:9460/continue_generation',
    ]


# ---- §3 / §8: jolt-guided tier & adjacent-policy own priority --------------------------------------------------------
def test_sft_sources_jolt_guided_tier_and_adjacent_policy_priority():
    import sft_sources

    assert 'rl-own' in sft_sources.OWN_SOURCES
    assert 'jolt-guided' in sft_sources.OWN_SOURCES

    tiers = {'t_solved': 'solved', 't_own': 'sometimes', 't_jolt': 'never', 't_teacher': 'never'}
    own_tasks = {'t_own'}
    guided_tasks = {'t_jolt'}
    assert sft_sources.frontier_drop('t_solved', tiers, own_tasks, guided_tasks) == 'frontier_task_solved'
    assert sft_sources.frontier_drop('t_own', tiers, own_tasks, guided_tasks) == 'frontier_own_success'
    assert sft_sources.frontier_drop('t_jolt', tiers, own_tasks, guided_tasks) == 'frontier_jolt_guided'
    assert sft_sources.frontier_drop('t_teacher', tiers, own_tasks, guided_tasks) is None

    # Adjacent-policy own priority: current, previous, and episodes straddling both stay in tier 0; older -> tier 1
    p_cur, p_prev, p_old = 'c' * 64, 'd' * 64, 'e' * 64
    assert sft_sources.own_policy_tier({'policy': p_cur}, p_cur, p_prev) == 0
    assert sft_sources.own_policy_tier({'policy': p_prev}, p_cur, p_prev) == 0
    assert sft_sources.own_policy_tier({'policy': p_cur, 'policies': [p_prev, p_cur]}, p_cur, p_prev) == 0
    assert sft_sources.own_policy_tier({'policy': p_old}, p_cur, p_prev) == 1
    assert sft_sources.own_policy_tier({'policy': p_cur, 'policies': [p_old, p_cur]}, p_cur, p_prev) == 1


# ---- §7.1 / §7.2 / §8: GRPO rho_by_staleness, compact group transport, multi-actor, defaults -------------------------
def test_grpo_rho_by_staleness_compact_group_and_defaults(tmp_path):
    import grpo

    # Default JOLT+ on (2), privileged twin off (0.0)
    args = grpo.build_parser().parse_args(['train', '--work', str(tmp_path), '--checkpoint', str(tmp_path / 'c.pt')])
    assert args.teacher_attempts == 2
    assert args.reasoning_rl_weight == 0.0

    # rho_by_staleness summary
    summary = grpo.rho_by_staleness_summary({
        0: {'tokens': 100, 'rho_sum': 101.0, 'abs_log_rho_sum': 1.5, 'rho_capped': 0},
        2: {'tokens': 50, 'rho_sum': 60.0, 'abs_log_rho_sum': 5.0, 'rho_capped': 5},
    })
    assert summary['0'] == {'tokens': 100, 'rho_mean': 1.01, 'abs_log_rho_mean': 0.015, 'rho_capped_fraction': 0.0}
    assert summary['2'] == {'tokens': 50, 'rho_mean': 1.2, 'abs_log_rho_mean': 0.1, 'rho_capped_fraction': 0.1}

    # Multi-actor endpoint parsing
    specs = grpo.parse_actor_specs('http://dgx:8460', ['http://g4a:18531=32', 'http://g4b:18531=32'], default_slots=96)
    assert specs == [('http://dgx:8460', 32), ('http://g4a:18531', 32), ('http://g4b:18531', 32)]

    # Compact scored-group pack / unpack round-trip
    gdir = tmp_path / 'g000001'
    gdir.mkdir()
    (gdir / 'cases.jsonl').write_text(json.dumps({'id': 'task-1'}) + '\n')
    (gdir / 'rewards.jsonl').write_text(json.dumps({'task_id': 'task-1', 'reward': 1.0}) + '\n')
    (gdir / 'versions.json').write_text(json.dumps([3, 3]) + '\n')
    (gdir / 'rollouts.jsonl').write_text(json.dumps({
        'sample': 0, 'task_id': 'task-1', 'outcome': 'success', 'reward': {'quality': 1.0},
        'policy': {'checkpoint_sha256': 'f' * 64},
        'turns': [{'capture': {'prompt_ids': [1, 2, 3], 'sample_ids': [4, 5], 'sample_logprobs': [-0.2, -0.3]}}],
    }) + '\n')
    archive = tmp_path / 'g000001.compact.zst'
    meta = grpo.pack_scored_group(gdir, archive)
    assert archive.is_file() and meta['episodes'] == 1
    out_dir = tmp_path / 'unpacked'
    unpacked = grpo.unpack_scored_group(archive, out_dir)
    assert unpacked['versions'] == [3, 3]
    ep = unpacked['episodes'][0]
    assert np.array_equal(ep['turns'][0]['sample_ids'], np.asarray([4, 5], dtype=np.int32))
    assert np.allclose(ep['turns'][0]['sample_logprobs'], np.asarray([-0.2, -0.3], dtype=np.float32))


# ---- §2 / §4 / §7.4: Async round blob-first commit, DGX takeover record, and SFT <-> GRPO PhaseController ------------
def test_async_round_manifest_takeover_and_phase_controller(tmp_path):
    import expert_iteration as ei

    round_dir = tmp_path / 'round-0001'
    chains = tmp_path / 'chains.pkl'
    chains.write_bytes(b'chains-payload-bytes')
    manifest = ei.write_round_manifest(
        round_dir, round_id='r0001', base_policy_sha='1' * 64, previous_policy_sha='0' * 64,
        chains_path=chains, episode_ids=['ep-b', 'ep-a', 'ep-a'], prefer_unseen=['/mix/0'],
        profile='colab_g4',
    )
    assert manifest['schema'] == ei.ROUND_MANIFEST_SCHEMA
    assert manifest['episode_ids'] == ['ep-a', 'ep-b']
    assert (round_dir / 'chains.sha256').read_text().strip() == manifest['chains_sha256']
    assert (round_dir / 'round.json').is_file()

    updated = ei.record_profile_change(
        round_dir, from_profile='colab_g4', to_profile='dgx', reason='unreachable > 30 min', at=1234.5,
    )
    assert updated['profile'] == 'dgx'
    assert updated['profile_changes'][-1]['from_profile'] == 'colab_g4'
    assert updated['profile_changes'][-1]['to_profile'] == 'dgx'

    # PhaseController transitions: success -> doubles held_every; 2 consecutive neutral -> sft->grpo -> grpo->sft; failure -> rollback
    state_file = tmp_path / 'phase-state.json'
    ctrl = ei.PhaseController.initialize(
        state_file, start_checkpoint='/ckpt/v0.pt', start_sha='0' * 64, held_every=2, held_every_max=8,
    )
    assert ctrl.phase == 'sft'
    assert ctrl.should_measure_held() is False
    ctrl.record_round_published('r1', '/ckpt/v1.pt', '1' * 64, mix_dir='/mix/1')
    assert ctrl.should_measure_held() is True

    ev1 = ctrl.record_verdict({'verdict': 'success', 'adopt': True, 'checkpoint': '/ckpt/v1.pt', 'sha256': '1' * 64})
    assert ev1['action'] == 'adopt'
    assert ctrl.state['held_every'] == 4
    assert ctrl.state['reference_policy']['sha256'] == '1' * 64

    # Two consecutive neutral verdicts flip SFT -> GRPO starting from newest held-adopted policy
    ev2 = ctrl.record_verdict({'verdict': 'neutral', 'adopt': False, 'checkpoint': '/ckpt/v2.pt', 'sha256': '2' * 64})
    assert ev2['action'] == 'continue' and ctrl.phase == 'sft'
    ev3 = ctrl.record_verdict({'verdict': 'neutral', 'adopt': False, 'checkpoint': '/ckpt/v3.pt', 'sha256': '3' * 64})
    assert ev3['action'] == 'switch_phase' and ev3['from_phase'] == 'sft' and ev3['to_phase'] == 'grpo'
    assert ctrl.phase == 'grpo'
    assert ctrl.state['current_policy']['sha256'] == '1' * 64  # newest held-adopted policy

    # Two consecutive neutral GRPO verdicts flip GRPO -> SFT
    ctrl.record_verdict({'verdict': 'neutral', 'adopt': True, 'checkpoint': '/ckpt/g1.pt', 'sha256': '4' * 64})
    ev4 = ctrl.record_verdict({'verdict': 'neutral', 'adopt': False, 'checkpoint': '/ckpt/g2.pt', 'sha256': '5' * 64})
    assert ev4['action'] == 'switch_phase' and ev4['from_phase'] == 'grpo' and ev4['to_phase'] == 'sft'
    assert ctrl.state['reference_policy']['sha256'] == '4' * 64

    # Failure rolls back to reference_policy, sets held_every = 1, flags bug_hunt_required
    ctrl.record_round_published('r4', '/ckpt/bad.pt', '9' * 64)
    ev5 = ctrl.record_verdict({'verdict': 'failure', 'adopt': False, 'checkpoint': '/ckpt/bad.pt', 'sha256': '9' * 64})
    assert ev5['action'] == 'rollback_and_bug_hunt'
    assert ctrl.state['current_policy']['sha256'] == '4' * 64
    assert ctrl.state['held_every'] == 1
    assert ctrl.state['bug_hunt_required'] is True


def test_collection_defaults_bf16_weights_and_no_guided_attempts_until_flipped(tmp_path, monkeypatch):
    """Until FP8 collection weights and JOLT+ custody work in production, every collection path (the cycle's collect
    command, ``collect``'s and ``async-drive``'s parsers) serves BF16 weights and runs no guided attempts."""
    import expert_iteration as ei
    assert ei.COLLECT_WEIGHT_DTYPE == 'bfloat16' and ei.TEACHER_ATTEMPTS == 0
    c = ei.Cycle.__new__(ei.Cycle)
    c.a = SimpleNamespace(kv_gb=12, slots=8, fill=None, k=4, collect_hours=1., no_neuralese_variants=True,
                          neuralese_variants=None)
    c.pin, c.collect_dir, c.dir = tmp_path, tmp_path / 'collect', tmp_path
    argv = c.collect_command('u', 'ckpt', 'sha', 8)
    assert argv[argv.index('--collect-weight-dtype') + 1] == 'bfloat16'
    assert argv[argv.index('--teacher-attempts') + 1] == '0'
