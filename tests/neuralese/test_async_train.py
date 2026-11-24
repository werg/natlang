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
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / 'training/neuralese', REPO, REPO / 'scripts/neuralese_data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---- §2 / §4: FP32 factor export, DGX CPU FP32 merge, dedup ledger --------------------------------------------------
def _tiny_trained_learner(alpha=8.0, rank=4, tied_layout=True):
    """A tiny real LFM3 learner (alpha != rank, so the scale is not 1) with random B factors and distinct trainable rows
    per id (a row permutation is visible), over a BF16 source checkpoint state with marker and override rows."""
    sys.path.insert(0, str(REPO / 'tests/neuralese'))
    from test_lfm3_port import tiny_lfm3_hf
    from natlang_neuralese.train.lfm3_learner import AdapterSpec, Lfm3Learner
    from natlang_neuralese.tune.targets import LFM3_LORA_FAMILIES
    model = tiny_lfm3_hf(window=4).to(dtype=torch.bfloat16)
    spec = AdapterSpec(target_families=LFM3_LORA_FAMILIES, rank=rank, alpha=alpha, trainable_rows=True)
    learner = Lfm3Learner.from_model(model, spec, special_ids=(58, 59, 60), marker_ids=(58, 59), override_ids=(60,),
                                     compile_layers=False, fp8_base=False, seed=3)
    g = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for bundle in learner.bundles:
            for _, leaf in bundle.iter_leaves():
                leaf.b.copy_(torch.randn(leaf.b.shape, generator=g) * 0.1)
        learner.trainable_rows.rows.copy_(torch.randn(learner.trainable_rows.rows.shape, generator=g))
    embed = model.model.embed_tokens.weight.detach().clone()
    rows = lambda ids: embed[torch.tensor(ids)].clone()
    interface = {'marker_ids': (58, 59), 'input_rows': rows([58, 59]), 'output_rows': None if tied_layout else rows([58, 59]),
                 'token_override_ids': (60,), 'extra_input_rows': rows([60]),
                 'extra_output_rows': None if tied_layout else rows([60])}
    backbone = {k: v.detach().clone() for k, v in model.state_dict().items()}
    backbone['lm_head.weight'] = backbone['model.embed_tokens.weight']
    learner.state = {'schema': 'natlang.native-checkpoint/1', 'step': 7,
                     'model_config': {'marker_ids': [58, 59], 'token_override_ids': [60]},
                     'interface_state': interface, 'backbone_state': backbone}
    return learner


@pytest.mark.parametrize('tied_layout', [True, False])
def test_factor_merge_equals_the_learners_own_export_and_the_formula(tmp_path, tied_layout):
    """The DGX merge of shipped factors (``merge_factors_to_checkpoint``) equals the learner's own export
    (``export_merged_checkpoint``) tensor for tensor, and both equal an independent oracle: bf16(fp32(W0) + (alpha/r)
    B A) per target, every interface row from its own id (input and, untied, output), the source embedding rows kept at
    the interface ids, the head tied. Mutants (scale dropped, scale applied twice, rows reversed) fail the oracle."""
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train.learner_state import (export_adapter_factors, export_merged_checkpoint,
                                                       merge_factors_to_checkpoint)
    learner = _tiny_trained_learner(alpha=8.0, rank=4, tied_layout=tied_layout)
    base_ckpt = tmp_path / 'base.pt'
    torch.save(learner.state, base_ckpt)
    base_sha = sha256_file_hex(base_ckpt)
    own = export_merged_checkpoint(learner, learner.state, tmp_path / 'own.pt', source_sha256=base_sha, learner_step=3)
    factors = tmp_path / 'factors'
    manifest = export_adapter_factors(learner, factors, base_checkpoint_sha256=base_sha, learner_step=3, round_id='r1')
    assert set(manifest['scales'].values()) == {2.0}
    shipped = merge_factors_to_checkpoint(base_ckpt, factors, tmp_path / 'shipped.pt')
    assert shipped['effective_policy_sha256'] == own['effective_policy_sha256']
    a = torch.load(tmp_path / 'own.pt', map_location='cpu', weights_only=False)
    b = torch.load(tmp_path / 'shipped.pt', map_location='cpu', weights_only=False)
    assert a['backbone_state'].keys() == b['backbone_state'].keys()
    for name in a['backbone_state']:
        assert torch.equal(a['backbone_state'][name], b['backbone_state'][name]), name
    for key in ('input_rows', 'output_rows', 'extra_input_rows', 'extra_output_rows'):
        x, y = a['interface_state'][key], b['interface_state'][key]
        assert (x is None and y is None) or torch.equal(x, y), key
    # the independent oracle
    src = learner.state['backbone_state']
    for name, (layer, _kind, short) in learner.target_map.items():
        leaf = getattr(learner.bundles[layer], short)
        expected = (src[name].float() + (8.0 / 4.0) * (leaf.b.detach().float() @ leaf.a.detach().float())).to(torch.bfloat16)
        assert torch.equal(b['backbone_state'][name], expected), name
    rows = dict(zip(learner.trainable_rows.token_ids.tolist(), learner.trainable_rows.rows.detach().to(torch.bfloat16)))
    iface = b['interface_state']
    assert torch.equal(iface['input_rows'], torch.stack([rows[58], rows[59]]))
    assert torch.equal(iface['extra_input_rows'], torch.stack([rows[60]]))
    if tied_layout:
        assert iface['output_rows'] is None and iface['extra_output_rows'] is None
    else:
        assert torch.equal(iface['output_rows'], torch.stack([rows[58], rows[59]]))
        assert torch.equal(iface['extra_output_rows'], torch.stack([rows[60]]))
    assert torch.equal(b['backbone_state']['model.embed_tokens.weight'], src['model.embed_tokens.weight'])
    assert b['backbone_state']['lm_head.weight'] is b['backbone_state']['model.embed_tokens.weight'] or torch.equal(
        b['backbone_state']['lm_head.weight'], b['backbone_state']['model.embed_tokens.weight'])
    assert b['round_id'] == 'r1' and b['inference_source']['adapter_factors_sha256'] == manifest['adapter_factors_sha256']


def test_vm_merge_reproduces_the_dgx_publication_bytes_or_leaves_nothing(tmp_path):
    """A Colab learner's next start is the DGX's publication (``publish-round`` writes policy/checkpoint.pt); the VM
    merges the same factors over its copy of the base (``pop_rl_actor._REMOTE_MERGE``) instead of receiving 3.4 GB.
    torch.save names its archive after the file, so only a merge written as checkpoint.pt hashes to the DGX's sha
    (the old rebind merge wrote CHECKPOINTS/<sha>.pt directly: other bytes under that name); a wrong expected sha
    leaves no file."""
    import subprocess
    import pop_rl_actor
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train.learner_state import export_adapter_factors, merge_factors_to_checkpoint
    learner = _tiny_trained_learner()
    base = tmp_path / 'base.pt'
    torch.save(learner.state, base)
    factors = tmp_path / 'factors'
    export_adapter_factors(learner, factors, base_checkpoint_sha256=sha256_file_hex(base), learner_step=3, round_id='r1')
    dgx = merge_factors_to_checkpoint(base, factors, tmp_path / 'r0001' / 'policy' / 'checkpoint.pt')
    sha = dgx['checkpoint_sha256']
    direct = merge_factors_to_checkpoint(base, factors, tmp_path / f'{sha}.pt')
    assert direct['checkpoint_sha256'] != sha                      # the name is part of the bytes

    def vm(expect):
        target = tmp_path / 'vm' / f'{expect}.pt'
        target.parent.mkdir(exist_ok=True)
        out = subprocess.run([sys.executable, '-c', pop_rl_actor._REMOTE_MERGE, str(REPO), str(base), str(factors),
                              str(tmp_path / 'vm' / '.merge'), str(target), expect], capture_output=True, text=True,
                             check=True).stdout
        return json.loads(out.strip().splitlines()[-1]), target
    result, target = vm(sha)
    assert result == {'ok': True, 'checkpoint_sha256': sha} and sha256_file_hex(target) == sha
    result, target = vm('0' * 64)
    assert result['ok'] is False and not target.exists() and not (tmp_path / 'vm' / '.merge').exists()


def test_factor_merge_fails_closed_and_the_round_ledger_persists(tmp_path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train.learner_state import (export_adapter_factors, load_learner_state,
                                                       merge_factors_to_checkpoint, save_learner_state)
    learner = _tiny_trained_learner()
    base_ckpt = tmp_path / 'base.pt'
    torch.save(learner.state, base_ckpt)
    base_sha = sha256_file_hex(base_ckpt)
    factors = tmp_path / 'factors'
    export_adapter_factors(learner, factors, base_checkpoint_sha256=base_sha, learner_step=1)
    bad = tmp_path / 'bad'
    bad.mkdir()
    (bad / 'policy_manifest.json').write_bytes((factors / 'policy_manifest.json').read_bytes())
    raw = bytearray((factors / 'adapter_factors_fp32.safetensors').read_bytes())
    raw[-1] ^= 0xFF
    (bad / 'adapter_factors_fp32.safetensors').write_bytes(bytes(raw))
    with pytest.raises(ValueError, match='checksum mismatch'):
        merge_factors_to_checkpoint(base_ckpt, bad, tmp_path / 'x.pt')
    other = tmp_path / 'other.pt'
    other.write_bytes(b'not-the-base')
    with pytest.raises(ValueError, match='base checkpoint mismatch'):
        merge_factors_to_checkpoint(other, factors, tmp_path / 'y.pt')
    noscale = tmp_path / 'noscale'
    noscale.mkdir()
    m = json.loads((factors / 'policy_manifest.json').read_text())
    m['scales'] = {}
    (noscale / 'policy_manifest.json').write_text(json.dumps(m))
    (noscale / 'adapter_factors_fp32.safetensors').write_bytes((factors / 'adapter_factors_fp32.safetensors').read_bytes())
    with pytest.raises(ValueError, match='scale'):
        merge_factors_to_checkpoint(base_ckpt, noscale, tmp_path / 'z.pt')
    assert not any((tmp_path / n).exists() for n in ('x.pt', 'y.pt', 'z.pt'))
    state_dir = tmp_path / 'learner-state'
    save_learner_state(state_dir, learner, learner_step=1, source_checkpoint_id=str(base_ckpt),
                       base_weights_sha256=base_sha, trained_rounds=['r1', 'r2', 'r1'])
    loaded = load_learner_state(state_dir, learner, restore_rng=False, verify_base_sha=base_sha)
    assert loaded['trained_rounds'] == ['r1', 'r2']


# ---- §5: Multi-engine policy publication -----------------------------------------------------------------------------
def test_publish_merged_policy_writes_every_engine_and_a_failed_engine_raises(tmp_path, monkeypatch):
    from natlang_neuralese.export import native_gguf, native_hf

    ckpt = tmp_path / 'policy.pt'
    ckpt.write_bytes(b'fake-merged-checkpoint')
    sha = hashlib.sha256(b'fake-merged-checkpoint').hexdigest()
    calls = []

    def fake_export(checkpoint, out_dir, *, base=None, dtype=None):
        calls.append('sglang')
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / 'model.safetensors').write_bytes(b'w')
        return {'weights_sha256': '1' * 64, 'schema': 'natlang.sglang-hf-export/1', 'checkpoint_sha256': sha}

    def fake_gguf(checkpoint, out_dir, *, outtypes=('bf16',), base=None, **_):
        calls.append(('gguf', tuple(outtypes)))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        files = []
        for t in outtypes:
            (Path(out_dir) / f'model-{t}.gguf').write_bytes(b'g')
            files.append({'outtype': t, 'file': f'model-{t}.gguf', 'sha256': '2' * 64})
        return {'checkpoint_sha256': sha, 'files': files}

    monkeypatch.setattr(native_hf, 'export', fake_export)
    monkeypatch.setattr(native_gguf, 'export', fake_gguf)
    pointer = tmp_path / 'current-policy.json'
    out = native_hf.publish_merged_policy(ckpt, base=tmp_path / 'base', sglang_export_dir=tmp_path / 'sglang-hf',
                                          gguf_dir=tmp_path / 'gguf', lfm_latest_path=pointer, label='r1', why='t')
    assert calls == ['sglang', ('gguf', ('q8_0',))]
    assert out['native']['sha256'] == sha and out['sglang_export']['weights_sha256'] == '1' * 64
    assert out['gguf']['files']['q8_0']['file'] == 'model-q8_0.gguf'
    ptr = json.loads(pointer.read_text())
    assert ptr['checkpoint']['sha256'] == sha and ptr['label'] == 'r1'
    # without a pointer request the pointer is not touched; a GGUF export that lacks its file is an error, not a skip
    pointer.unlink()
    monkeypatch.setattr(native_gguf, 'export', lambda *a, **k: {'checkpoint_sha256': sha, 'files': []})
    with pytest.raises(RuntimeError, match='GGUF'):
        native_hf.publish_merged_policy(ckpt, sglang_export_dir=tmp_path / 'sglang-hf', gguf_dir=tmp_path / 'g2')
    assert not pointer.exists()


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

    from natlang_neuralese.common.execution_profile import G4_HARDWARE, colab_layout
    layout = colab_layout(G4_HARDWARE)
    cfg = pop_rl_actor.host_config('colab_g4', host='g4-a')
    cmd_fp8, claim, sglang_fp8, fmt_fp8 = pop_rl_actor.colab_serve_command(
        cfg, checkpoint='/content/ckpt.pt', sha='a' * 64, port=18531, work='/content/work',
        beside_learner=True, kv_cache_dtype='fp8_e4m3', weight_dtype='fp8', layout=layout,
    )
    assert '--sglang-weight-dtype' in cmd_fp8 and 'fp8' in cmd_fp8
    assert '--sglang-kv-gb' in cmd_fp8
    assert fmt_fp8.endswith('-w-fp8')
    assert sglang_fp8['weight_dtype'] == 'fp8'
    assert claim['vram_gib'] < 96

    # BF16 keeps original format without -w-fp8
    cmd_bf16, _, sglang_bf16, fmt_bf16 = pop_rl_actor.colab_serve_command(
        cfg, checkpoint='/content/ckpt.pt', sha='a' * 64, port=18531, work='/content/work',
        beside_learner=False, kv_cache_dtype='fp8_e4m3', weight_dtype='bfloat16', layout=layout,
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


def test_fp8_collection_weights_need_fp8_hardware_and_the_forks_fp8_projections():
    import serve_native_checkpoint as serve
    fork = {'fp8_weights': True}
    assert serve.serving_weight_dtype('bfloat16', fork, (12, 1)) == ('bfloat16', None)
    assert serve.serving_weight_dtype('fp8', fork, (12, 1)) == ('fp8', None)          # GB10
    assert serve.serving_weight_dtype('fp8', fork, (8, 9)) == ('fp8', None)           # Pop's RTX 4060
    dtype, why = serve.serving_weight_dtype('fp8', fork, (8, 0))                      # Colab A100: no FP8 GEMM
    assert dtype == 'bfloat16' and 'sm_89' in why
    dtype, why = serve.serving_weight_dtype('fp8', {}, (12, 1))                       # a fork without FP8 projections
    assert dtype == 'bfloat16' and 'NATLANG_FP8_WEIGHTS' in why
    assert serve.serving_weight_dtype('fp8', fork, None)[0] == 'bfloat16'

    # the compact scored-group exchange and the extra actors' publication: tests/neuralese/test_grpo_exchange.py


# ---- §2 / §4 / §7.4: Async round blob-first commit, DGX takeover record, and SFT <-> GRPO PhaseController ------------
def test_round_manifest_is_blob_first_and_records_a_takeover(tmp_path):
    import expert_iteration as ei
    round_dir = tmp_path / 'r0001'
    chains = tmp_path / 'chains.pkl'
    chains.write_bytes(b'chains-payload-bytes')
    manifest = ei.write_round_manifest(round_dir, round_id='r0001', base_policy_sha='1' * 64,
                                       previous_policy_sha='0' * 64, chains_path=chains,
                                       episode_ids=['ep-b', 'ep-a', 'ep-a'], prefer_unseen=['/mix/0'], profile='colab')
    assert manifest['episode_ids'] == ['ep-a', 'ep-b']
    assert (round_dir / 'chains.sha256').read_text().strip() == manifest['chains_sha256']
    updated = ei.record_profile_change(round_dir, from_profile='colab', to_profile='dgx', reason='away', at=1.)
    assert updated['profile'] == 'dgx' and updated['profile_changes'][-1]['from_profile'] == 'colab'
    ei.update_round_manifest(round_dir, trained_from='1' * 64)
    assert json.loads((round_dir / 'round.json').read_text())['trained_from'] == '1' * 64


def _policy(i):
    return f'/ckpt/v{i}.pt', f'{i:x}' * 64 if i < 16 else hashlib.sha256(str(i).encode()).hexdigest()


@pytest.mark.parametrize('held_every', [1, 2, 3])
def test_held_schedule_counted_in_rounds_matches_the_cycle_rule(tmp_path, held_every):
    """With --held-every N the async loop measures every Nth round, as cycles do (Cycle.measured: since + 1 >= N with
    since the unmeasured cycles before this one); the first round is measured only when N is 1."""
    import expert_iteration as ei
    ctrl = ei.PhaseController.initialize(tmp_path / 's.json', start_checkpoint='/ckpt/v0.pt', start_sha='0' * 64,
                                         held_every=held_every)
    measured, since = [], 0
    cycle_measured = []
    for i in range(1, 7):
        ckpt, sha = _policy(i)
        ctrl.record_round_published(f'r{i}', ckpt, sha, base_sha256=ctrl.current['sha256'])
        if ctrl.held_due():
            ctrl.start_measurement()
            measured.append(i)
        cycle = since + 1 >= held_every          # the cycle path's rule with its own counter
        cycle_measured.append(i) if cycle else None
        since = 0 if cycle else since + 1
    assert measured == cycle_measured == list(range(held_every, 7, held_every))


def test_rollback_returns_to_the_reference_and_previous_is_never_the_rejected_policy(tmp_path):
    import expert_iteration as ei
    ctrl = ei.PhaseController.initialize(tmp_path / 's.json', start_checkpoint='/ckpt/v0.pt', start_sha='0' * 64,
                                         held_every=1)
    for i in (1, 2):
        ctrl.record_round_published(f'r{i}', *_policy(i), base_sha256=ctrl.current['sha256'])
    ctrl.start_measurement()
    assert ctrl.record_verdict({'verdict': 'success', 'adopt': True, 'checkpoint': _policy(2)[0],
                                'sha256': _policy(2)[1]})['action'] == 'adopt'
    assert ctrl.state['held_every'] == 2
    # rounds 3 and 4 descend from 2 speculatively; 3 is measured and fails while 4 exists
    for i in (3, 4):
        ctrl.record_round_published(f'r{i}', *_policy(i), base_sha256=ctrl.current['sha256'])
    ev = ctrl.record_verdict({'verdict': 'failure', 'adopt': False, 'checkpoint': _policy(3)[0], 'sha256': _policy(3)[1]})
    assert ev['action'] == 'rollback' and 'bug_hunt' in ev
    assert ctrl.current['sha256'] == _policy(2)[1]                     # the reference
    assert ctrl.previous['sha256'] == _policy(1)[1]                    # its predecessor, not the rejected 3
    assert not ctrl.in_lineage(_policy(3)[1]) and not ctrl.in_lineage(_policy(4)[1])
    assert ctrl.state['held_every'] == 1
    # a round trained from the discarded 4 cannot publish; the next round from the reference is measured
    with pytest.raises(ValueError, match='lineage head'):
        ctrl.record_round_published('r5', *_policy(5), base_sha256=_policy(4)[1])
    ctrl.record_round_published('r6', *_policy(6), base_sha256=ctrl.current['sha256'])
    assert ctrl.held_due()
    # a late verdict of a policy already rolled back is moot
    assert ctrl.record_verdict({'verdict': 'success', 'adopt': True, 'checkpoint': _policy(4)[0],
                                'sha256': _policy(4)[1]})['action'].startswith('moot')
    assert ctrl.state['reference_policy']['sha256'] == _policy(2)[1]


def test_an_unmeasured_start_is_judged_against_the_given_reference_and_rolls_back_to_it(tmp_path):
    """--reference: a start from speculative cycles (sft-v9 on sft-v7's verdict) measures against the last adopted
    policy, and a neutral rollback returns to that policy, not to the unmeasured start."""
    import expert_iteration as ei
    ref = ('/ckpt/ref.pt', 'f' * 64)
    ctrl = ei.PhaseController.initialize(tmp_path / 's.json', start_checkpoint='/ckpt/v0.pt', start_sha='0' * 64,
                                         held_every=1, reference_checkpoint=ref[0], reference_sha=ref[1])
    ctrl.record_round_published('r1', *_policy(1), base_sha256=ctrl.current['sha256'])
    assert ctrl.held_due()
    assert ctrl.start_measurement()['reference'] == {'checkpoint': ref[0], 'sha256': ref[1]}
    ev = ctrl.record_verdict({'verdict': 'neutral', 'adopt': False, 'checkpoint': _policy(1)[0],
                              'sha256': _policy(1)[1]})
    assert ev['action'] == 'rollback'
    assert ctrl.current['sha256'] == ref[1] and not ctrl.in_lineage('0' * 64)


def test_two_neutral_measured_rounds_switch_phase_from_the_reference_and_back(tmp_path):
    import expert_iteration as ei
    ctrl = ei.PhaseController.initialize(tmp_path / 's.json', start_checkpoint='/ckpt/v0.pt', start_sha='0' * 64,
                                         held_every=1)
    def publish_and_judge(i, verdict, adopt):
        ctrl.record_round_published(f'r{i}', *_policy(i), base_sha256=ctrl.current['sha256'])
        ctrl.start_measurement()
        return ctrl.record_verdict({'verdict': verdict, 'adopt': adopt, 'checkpoint': _policy(i)[0],
                                    'sha256': _policy(i)[1]})
    publish_and_judge(1, 'neutral', True)
    assert ctrl.phase == 'sft' and ctrl.state['reference_policy']['sha256'] == _policy(1)[1]
    ev = publish_and_judge(2, 'neutral', False)
    assert ev['switch_phase'] == {'from': 'sft', 'to': 'grpo'} and ctrl.phase == 'grpo'
    assert ctrl.current['sha256'] == _policy(1)[1]                     # the newest held-adopted policy
    publish_and_judge(3, 'neutral', True)
    ev = publish_and_judge(4, 'neutral', True)
    assert ev['switch_phase'] == {'from': 'grpo', 'to': 'sft'} and ctrl.current['sha256'] == _policy(4)[1]


def _driver(tmp_path, monkeypatch, **over):
    import expert_iteration as ei
    monkeypatch.setattr(ei, 'OPS', tmp_path / 'ops')
    (tmp_path / 'ops' / 'grpo-v4').mkdir(parents=True, exist_ok=True)
    pin = tmp_path / 'pin'
    (pin / 'scripts' / 'neuralese_data').mkdir(parents=True, exist_ok=True)
    start = tmp_path / 'start.pt'
    start.write_bytes(b'start')
    argv = ['async-drive', '--root', str(tmp_path / 'loop'), '--name', 'aei', '--start', str(start),
            '--start-sha', 'a' * 64, '--commit', 'c' * 40, '--pin', str(pin), '--sft-work', str(tmp_path / 'work'),
            *over.pop('argv', [])]
    parser_args = {}
    monkeypatch.setattr(sys, 'argv', ['expert_iteration.py', *argv])
    monkeypatch.setattr(ei, 'async_drive', lambda a: parser_args.setdefault('a', a) and 0)
    ei.main()
    return ei, ei.AsyncRoundDriver(parser_args['a'])


def test_one_sft_recipe_for_colab_and_dgx_and_the_dgx_learner_runs_through_the_ledger(tmp_path, monkeypatch):
    ei, d = _driver(tmp_path, monkeypatch)
    rd = d.root / 'r0001'
    (rd / 'mix').mkdir(parents=True)
    (rd / 'chains.sha256').write_text('f' * 64 + '\n')
    cur = d.controller.current
    g4 = d.sft_env(rd, cur, activation_gb=40, preflight_only=True)
    dgx = d.sft_env(rd, cur, activation_gb=56, preflight_only=False)
    for env in (g4, dgx):
        assert (env['WARMUP'], env['BLT']) == ('10', '30000')          # launch-sft.sh adds --epochs 1
        extra = env['EXTRA'].split()
        assert extra[extra.index('--round-id') + 1] == 'r0001' and '--factors-only' in extra and '--resume' in extra
        assert extra[extra.index('--export-factors-dir') + 1] == str(rd / 'sft' / 'factors')
        assert env['CKPT'] == cur['checkpoint'] and env['SHA'] == cur['sha256'] and env['WORK'] == str(rd / 'sft')
    assert {k: v for k, v in g4.items() if k not in ('ACT', 'PREFLIGHT_ONLY')} == \
        {k: v for k, v in dgx.items() if k != 'ACT'}
    assert g4['PREFLIGHT_ONLY'] == '1' and 'PREFLIGHT_ONLY' not in dgx
    launched = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: launched.append((argv, kw['env'])) or
                        SimpleNamespace(returncode=0))
    def finished(unit, poll=60, between=None):
        (rd / 'sft' / 'factors').mkdir(parents=True, exist_ok=True)
        (rd / 'sft' / 'factors' / 'policy_manifest.json').write_text('{}')
    monkeypatch.setattr(d.c, 'active', lambda unit: False)
    monkeypatch.setattr(d.c, 'present', lambda unit: False)
    monkeypatch.setattr(d.c, 'wait', finished)
    stops = []
    monkeypatch.setattr(d, 'stop_collections', lambda **kw: stops.append((kw, len(launched))))
    assert d.train_sft(rd, cur) == rd / 'sft' / 'factors'
    # the DGX collection yields for the whole SFT before the learner launches (preempted, the ledger would relaunch it
    # beside the learner); remote actors keep collecting
    assert stops == [({'remote': False}, 0)]
    (argv, env), = launched
    assert argv[:2] == ['/bin/sh', str(ei.OPS / 'grpo-v4' / 'launch-sft.sh')] and 'PREFLIGHT_ONLY' not in env
    # launch-sft.sh itself execs memory_ledger.py run (the learner is a ledger unit)
    from natlang_neuralese.common.paths import root
    script = root('data_nvme') / 'lfm3-rl-dgx-20261029' / 'ops' / 'grpo-v4' / 'launch-sft.sh'
    if script.exists():
        text = script.read_text()
        assert 'memory_ledger.py run --unit "$UNIT"' in text and '--epochs 1' in text and 'exp0_sft.py $ARGS' in text


def test_a_round_skips_an_answering_session_slower_than_the_dgx_and_trains_on_the_dgx_at_once(tmp_path, monkeypatch):
    """An L4 session answers (its actor keeps serving fills) but trains slower than the DGX: the round does not run
    run-sft there and does not wait --fallback-wait-minutes; it records the takeover and launches the DGX learner."""
    ei, d = _driver(tmp_path, monkeypatch, argv=['--sft-profile', 'colab', '--colab-hosts', 'colab-g4'])
    rd = d.root / 'r0001'
    (rd / 'mix').mkdir(parents=True)
    (rd / 'chains.sha256').write_text('f' * 64 + '\n')
    (rd / 'round.json').write_text('{}')
    import pop_rl_actor
    monkeypatch.setattr(pop_rl_actor, 'reachable_hosts', lambda profile, hosts: ['colab-g4'])
    monkeypatch.setattr(pop_rl_actor, 'colab_sft_fit', lambda host: (False, 'NVIDIA L4: 0.97x'))
    launched = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: launched.append(argv) or SimpleNamespace(returncode=0))
    monkeypatch.setattr(ei.time, 'sleep', lambda s: (_ for _ in ()).throw(AssertionError('waited for a session')))
    def finished(unit, poll=60, between=None):
        (rd / 'sft' / 'factors').mkdir(parents=True, exist_ok=True)
        (rd / 'sft' / 'factors' / 'policy_manifest.json').write_text('{}')
    monkeypatch.setattr(d.c, 'active', lambda unit: False)
    monkeypatch.setattr(d.c, 'present', lambda unit: False)
    monkeypatch.setattr(d.c, 'wait', finished)
    monkeypatch.setattr(d, 'stop_collections', lambda **kw: None)
    assert d.train_sft(rd, d.controller.current) == rd / 'sft' / 'factors'
    assert not any('run-sft' in argv for argv in launched)
    assert launched and launched[-1][:2] == ['/bin/sh', str(ei.OPS / 'grpo-v4' / 'launch-sft.sh')]
    changes = json.loads((rd / 'round.json').read_text())['profile_changes']
    assert changes[-1]['to_profile'] == 'dgx' and 'slower than the DGX' in changes[-1]['reason']


def test_sft_handback_decision_moves_only_with_a_clear_margin(tmp_path):
    """r0001 on 2026-11-23: step ~306 of 1150 on the DGX at ~85 s/step (~20 h left); an A100 (4.09x measured, 2.5x by
    TFLOPS) finishes from the last save in a fraction of that, the move included. A small or marginal gain, or an
    unknown GPU, keeps the round where it is."""
    import expert_iteration as ei
    from natlang_neuralese.common.execution_profile import sft_speedup
    a100 = {'gpu_name': 'NVIDIA A100-SXM4-80GB'}
    assert sft_speedup(a100) == (4.09, 'measured (execution_profile)')
    assert sft_speedup(a100, {'NVIDIA A100-SXM4-80GB': 3.2}) == (3.2, 'measured by this loop')
    assert sft_speedup({'gpu_name': 'NVIDIA H100 80GB HBM3'}) == (6.05, 'BF16 TFLOPS ratio')
    assert sft_speedup({'gpu_name': 'mystery'}) == (None, 'unknown GPU')
    rule = dict(margin=0.25, min_gain_seconds=3600)
    r0001 = ei.sft_handback_decision(steps_total=1150, dgx_step=309, resume_step=304, dgx_s_per_step=85.0,
                                     speedup=2.5, move_seconds=2400, **rule)
    assert r0001['move'] and r0001['dgx_left_s'] == 841 * 85 and r0001['colab_left_s'] == round(2400 + 846 * 85 / 2.5)
    # near the end: the move costs more than it saves
    assert not ei.sft_handback_decision(steps_total=1150, dgx_step=1120, resume_step=1110, dgx_s_per_step=85.0,
                                        speedup=4.09, move_seconds=2400, **rule)['move']
    # faster, but not by the margin (a 1.3x session)
    assert not ei.sft_handback_decision(steps_total=1150, dgx_step=300, resume_step=300, dgx_s_per_step=85.0,
                                        speedup=1.3, move_seconds=600, **rule)['move']
    assert not ei.sft_handback_decision(steps_total=1150, dgx_step=300, resume_step=300, dgx_s_per_step=85.0,
                                        speedup=None, move_seconds=600, **rule)['move']
    log = tmp_path / 'train.jsonl'
    log.write_text('{"resumed_from": "x"}\n' + ''.join(
        json.dumps({'step': i, 'of': 1150, 'step_seconds': 80.0 + (i % 3) * 5}) + '\n' for i in range(270, 310)))
    assert ei.dgx_sft_progress(log) == {'step': 309, 'of': 1150, 's_per_step': 85.0, 'window': 20}
    log.write_text(json.dumps({'step': 1, 'of': 9, 'step_seconds': 3}) + '\n')
    assert ei.dgx_sft_progress(log) is None


def _dgx_round(tmp_path, monkeypatch, *, saved_age=60.0, argv=()):
    """A driver whose round r0001 trains on the DGX (a takeover): unit present, train.jsonl at step 309 of 1150 at
    85 s/step, its learner state saved ``saved_age`` s ago at step 304, a fit A100 session answering."""
    ei, d = _driver(tmp_path, monkeypatch, argv=['--sft-profile', 'colab', '--colab-hosts', 'colab-g4', *argv])
    rd = d.root / 'r0001'
    run = rd / 'sft'
    (run / 'learner-state').mkdir(parents=True)
    (rd / 'mix').mkdir()
    (rd / 'chains.sha256').write_text('f' * 64 + '\n')
    (rd / 'round.json').write_text(json.dumps({'profile': 'dgx', 'profile_changes': [
        {'from_profile': 'colab', 'to_profile': 'dgx', 'reason': 'Colab session slower than the DGX for SFT'}]}))
    (run / 'launch.json').write_text(json.dumps({'exp0_sft_args': 'train --chains /c.pkl --checkpoint /s.pt',
                                                 'chains_sha256': 'f' * 64, 'start_checkpoint_sha256': 'a' * 64}))
    (run / 'train.jsonl').write_text(''.join(json.dumps({'step': i, 'of': 1150, 'step_seconds': 85.0}) + '\n'
                                             for i in range(280, 310)))
    (run / 'train.log').write_text('dgx log\n')
    (run / 'learner-state' / 'learner_manifest.json').write_text(json.dumps(
        {'learner_step': 304, 'batch_cursor': {'next_step': 304, 'total_steps': 1150},
         'saved_at': ei.time.time() - saved_age}))
    (run / 'learner-state' / 'optimizer_state.pt').write_bytes(b'o' * 1000)
    import pop_rl_actor
    events = []
    state = {'present': True}
    monkeypatch.setattr(pop_rl_actor, 'reachable_hosts', lambda profile, hosts: ['colab-g4'])
    monkeypatch.setattr(pop_rl_actor, 'colab_sft_fit', lambda host: (True, 'A100: 2.5x'))
    monkeypatch.setattr(pop_rl_actor, 'colab_hardware', lambda host: {'gpu_name': 'NVIDIA A100-SXM4-80GB',
                                                                      'vram_total_gib': 80.0})
    monkeypatch.setattr(pop_rl_actor, 'sft_move_seconds', lambda host, cfg, **kw: (
        2400.0, {'state': 300.0, 'setup': 900.0, 'chains': 400.0, 'checkpoint': 800.0}))
    monkeypatch.setattr(pop_rl_actor, 'colab_checkout', lambda host, cfg, commit: events.append(('checkout', commit)))

    def push(host, cfg, remote_run, local_state, *, run_name, via):
        events.append(('push', remote_run, Path(local_state).name, run_name, via, state['present']))
        return {'step': 304, 'via': 'drive', 'seconds': 310.0, 'bytes': 1000, 'files': 2}
    monkeypatch.setattr(pop_rl_actor, 'push_learner_state', push)
    monkeypatch.setattr(d.c, 'present', lambda unit: state['present'])
    monkeypatch.setattr(d.c, 'active', lambda unit: state['present'])
    monkeypatch.setattr(d.c, 'stop_unit', lambda unit: events.append(('stop', unit)) or state.update(present=False))
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: events.append(('run', argv[:3])) or
                        SimpleNamespace(returncode=0))

    def wait(unit, poll=60, between=None):
        events.append(('wait', unit))
        for _ in range(3):              # three polls while the DGX learner runs
            if not state['present']:
                break
            between()
    monkeypatch.setattr(d.c, 'wait', wait)
    monkeypatch.setattr(d, 'stop_collections', lambda **kw: events.append(('stop_collections', kw)))
    monkeypatch.setattr(d, 'ensure_remote_collection', lambda policy: None)
    monkeypatch.setattr(d, 'ensure_collection', lambda policy: events.append(('ensure_collection', policy['sha256'])))

    def colab(round_dir, policy):
        events.append(('colab_sft', state['present']))
        (run / 'factors').mkdir()
        (run / 'factors' / 'policy_manifest.json').write_text('{}')
    monkeypatch.setattr(d, 'colab_sft', colab)
    return ei, d, rd, events


def test_a_round_on_the_dgx_moves_to_a_fit_session_from_its_verified_state(tmp_path, monkeypatch):
    """2026-11-23: r0001 went to the DGX because the session was an L4; the keeper then got an A100, but the round
    stayed on the DGX for ~20 h. A restarted driver joins the running DGX learner (never a second learner on the
    session), the hand-back pushes its latest save to the VM, and only after that push verified does the DGX unit
    stop; the round then continues on the session (run-sft resumes that state), the DGX collection resumes, and the
    move is recorded in round.json, handback.json and the driver log, with the DGX logs kept."""
    ei, d, rd, events = _dgx_round(tmp_path, monkeypatch)
    cur = d.controller.current
    assert d.train_sft(rd, cur) == rd / 'sft' / 'factors'
    kinds = [e[0] for e in events]
    assert kinds.index('wait') < kinds.index('push') < kinds.index('stop') < kinds.index('colab_sft')
    assert not any(e[0] == 'run' and 'launch-sft.sh' in str(e[1]) for e in events)   # the running learner joined
    (push,) = [e for e in events if e[0] == 'push']
    assert push[1:] == ('/content/natlang-work/runs/sft-aei-r0001', 'learner-state', 'aei-r0001', 'auto', True)
    assert ('checkout', 'c' * 40) in events and ('stop', 'aei-r0001-sft') in events
    assert ('colab_sft', False) in events and ('ensure_collection', cur['sha256']) in events
    manifest = json.loads((rd / 'round.json').read_text())
    change = manifest['profile_changes'][-1]
    assert manifest['profile'] == 'colab' and (change['from_profile'], change['to_profile']) == ('dgx', 'colab')
    assert 'hand-back to colab-g4' in change['reason'] and 'step 304 verified' in change['reason']
    (move,) = json.loads((rd / 'sft' / 'handback.json').read_text())
    assert move['pushed_step'] == 304 and move['dgx_step'] == 309 and move['speedup'] == 4.09
    assert move['decision']['move'] and move['dgx_s_per_step'] == 85.0
    assert (rd / 'sft' / 'train.dgx-1.jsonl').exists() and (rd / 'sft' / 'train.dgx-1.log').read_text() == 'dgx log\n'
    log = [json.loads(line) for line in (d.root / 'drive.log').read_text().splitlines()]
    assert any(r.get('handback') == 'colab' for r in log) and any('handback_check' in r for r in log)
    # the session's measured speed on the steps it trained becomes the next move's estimate
    (rd / 'sft' / 'train.jsonl').write_text(''.join(json.dumps({'step': i, 'of': 1150, 'step_seconds': 25.0}) + '\n'
                                                    for i in range(304, 340)))
    d.record_sft_speed(rd)
    assert d.measured_sft_speed() == {'NVIDIA A100-SXM4-80GB': 3.4}


def test_a_driver_killed_mid_push_and_its_successor_never_race_the_dgx_learner(tmp_path, monkeypatch):
    """2026-11-23: driver 1 decided to move r0001 to the A100 (22:36) and was stopped mid-push (22:41); driver 2,
    started with the same argv, saw no DGX unit (systemd and the ledger answered "not present" while the learner kept
    training), entered the Colab path and run-sft trained r0001 on the A100 beside the DGX learner (both to step 353).
    Now: driver 1 dies leaving the DGX learner as it was; driver 2 finds the learner by its process (--out-dir sft/),
    joins it, and the only way the round reaches the session is the verified hand-back: push, stop, the learner gone,
    then run-sft. run-sft itself refuses while a DGX learner trains the run."""
    import subprocess
    import pop_rl_actor
    ei, d, rd, events = _dgx_round(tmp_path, monkeypatch)
    learner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)', str(tmp_path / 'pin' / 'exp0_sft.py'),
                                'train', '--out-dir', str(rd / 'sft')])
    try:
        assert pop_rl_actor.local_learners(rd / 'sft') == [learner.pid]
        assert pop_rl_actor.local_learners(rd / 'mix') == []

        class Killed(BaseException):        # the driver's unit stopped: nothing after the push runs
            pass

        def killed_mid_push(*a, **k):
            events.append(('push-started',))
            raise Killed()
        monkeypatch.setattr(pop_rl_actor, 'push_learner_state', killed_mid_push)
        with pytest.raises(Killed):
            d.train_sft(rd, d.controller.current)
        kinds = [e[0] for e in events]
        assert 'push-started' in kinds and 'stop' not in kinds and 'colab_sft' not in kinds
        assert learner.poll() is None and not (rd / 'sft' / 'handback.json').exists()
        assert json.loads((rd / 'round.json').read_text())['profile'] == 'dgx'

        # driver 2: same root and argv; its systemd/ledger view says the unit is gone, the learner still trains
        ei2, d2 = _driver(tmp_path, monkeypatch, argv=['--sft-profile', 'colab', '--colab-hosts', 'colab-g4'])
        log = []
        monkeypatch.setattr(d2.c, 'present', lambda unit: False)
        monkeypatch.setattr(d2.c, 'active', lambda unit: False)
        monkeypatch.setattr(d2.c, 'wait', lambda unit, poll=60, between=None: log.append(('wait', unit)))
        monkeypatch.setattr(d2, 'stop_collections', lambda **kw: log.append(('stop_collections',)))
        monkeypatch.setattr(d2, 'ensure_remote_collection', lambda policy: None)
        monkeypatch.setattr(d2, 'ensure_collection', lambda policy: log.append(('ensure_collection',)))
        monkeypatch.setattr(d2.c, 'session', lambda host, profile, opened: None)

        def stop(unit):
            log.append(('stop', unit))
            learner.terminate()
            learner.wait(30)
        monkeypatch.setattr(d2.c, 'stop_unit', stop)

        def push(host, cfg, remote_run, local_state, *, run_name, via):
            log.append(('push', learner.poll() is None))
            return {'step': 304, 'via': 'drive', 'seconds': 300.0, 'bytes': 1000, 'files': 2}
        monkeypatch.setattr(pop_rl_actor, 'push_learner_state', push)

        def run(argv, **kw):
            argv = list(map(str, argv))
            if 'run-sft' in argv:
                log.append(('run-sft', learner.poll() is None))
                (rd / 'sft' / 'factors').mkdir(parents=True, exist_ok=True)
                (rd / 'sft' / 'factors' / 'policy_manifest.json').write_text('{}')
            elif any('launch-sft.sh' in a for a in argv):
                log.append(('launch-sft', 'PREFLIGHT_ONLY' in kw.get('env', {})))
            return SimpleNamespace(returncode=0)
        monkeypatch.setattr(ei2.subprocess, 'run', run)
        assert d2.train_sft(rd, d2.controller.current) == rd / 'sft' / 'factors'
        kinds = [e[0] for e in log]
        # joined (no DGX launch, no Colab learner) until the verified hand-back stopped it
        assert kinds.index('wait') < kinds.index('push') < kinds.index('stop') < kinds.index('run-sft')
        assert ('launch-sft', False) not in log and ('run-sft', False) in log and ('run-sft', True) not in log
        assert kinds.count('run-sft') == 1 and ('push', True) in log
        notes = (d2.root / 'drive.log').read_text()
        assert 'dgx_learner_without_unit' in notes and '"handback": "colab"' in notes
        assert json.loads((rd / 'round.json').read_text())['profile_changes'][-1]['to_profile'] == 'colab'
    finally:
        learner.kill()
        learner.wait()


def test_run_sft_and_the_colab_path_refuse_while_a_dgx_learner_trains_the_round(tmp_path, monkeypatch):
    """Whatever round.json's profile says and whatever systemd answers: colab_sft does not start run-sft while this
    round's DGX learner process lives, and run-sft refuses at its start and again before the VM learner starts."""
    import pop_rl_actor
    ei, d, rd, events = _dgx_round(tmp_path, monkeypatch)
    monkeypatch.setattr(d.c, 'present', lambda unit: False)
    monkeypatch.setattr(pop_rl_actor, 'local_learners', lambda work: [4242] if Path(work) == rd / 'sft' else [])
    launched = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: launched.append(argv) or SimpleNamespace(returncode=0))
    ei.AsyncRoundDriver.colab_sft(d, rd, d.controller.current)
    assert launched == [] and 'colab_refused' in (d.root / 'drive.log').read_text()
    args = SimpleNamespace(host='colab-g4', run_name='aei-r0001', work=str(rd / 'sft'),
                           launch_json=str(rd / 'sft' / 'launch.json'), commit=None, state_sync_minutes=15.0,
                           vram_gib=None, host_ram_gib=None)
    monkeypatch.setattr(pop_rl_actor, 'host_config', lambda *a, **k: (_ for _ in ()).throw(AssertionError('went on')))
    with pytest.raises(SystemExit, match='a round has one learner'):
        pop_rl_actor.run_remote_sft(args)


def test_no_handback_when_opted_out_unfavourable_unsaved_or_unverified(tmp_path, monkeypatch):
    """The DGX learner keeps training when: --no-sft-handback; the push to the VM fails (nothing stopped, retried
    after 30 min); the last save is too old to push before the next one lands (it waits for that save); the session is
    an L4; or the round's Colab learner failed (blocked)."""
    def run(argv=(), saved_age=60.0, setup=None):
        ei, d, rd, events = _dgx_round(tmp_path / str(len(list(tmp_path.iterdir()))), monkeypatch,
                                       saved_age=saved_age, argv=argv)
        if setup:
            setup(ei, d)
        with pytest.raises(SystemExit, match='no factors'):
            d.train_sft(rd, d.controller.current)
        return d, rd, [e[0] for e in events]
    import pop_rl_actor
    d, rd, kinds = run(argv=['--no-sft-handback'])
    assert 'push' not in kinds and 'stop' not in kinds and not (d.root / 'drive.log').exists()

    def failing(ei, d):
        monkeypatch.setattr(pop_rl_actor, 'push_learner_state', lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError('the learner state on colab-g4 does not match the DGX snapshot')))
    d, rd, kinds = run(setup=failing)
    assert 'stop' not in kinds and 'colab_sft' not in kinds
    assert json.loads((rd / 'round.json').read_text())['profile'] == 'dgx' and not (rd / 'sft' / 'handback.json').exists()
    assert d.handback_next['r0001'] > time.time() + 1700
    d, rd, kinds = run(saved_age=800.0)                     # 800 s + 300 s push > the 15 min save interval
    assert 'push' not in kinds and 'waits_for_save' in (d.root / 'drive.log').read_text()

    def l4(ei, d):
        monkeypatch.setattr(pop_rl_actor, 'colab_sft_fit', lambda host: (False, 'NVIDIA L4: 0.97x'))
    d, rd, kinds = run(setup=l4)
    assert 'push' not in kinds

    def blocked(ei, d):
        d.handback_blocked.add('r0001')
    d, rd, kinds = run(setup=blocked)
    assert 'push' not in kinds and 'stop' not in kinds


def _fake_vm(tmp_path, monkeypatch):
    """A Colab VM in a directory: ssh commands run in bash with /content/natlang-work mapped there; the tunnel copies."""
    import shutil
    import subprocess
    import pop_rl_actor
    vm = tmp_path / 'vm'
    vm.mkdir()
    mapped = lambda text: str(text).replace('/content/natlang-work', str(vm))
    sent = []

    def ssh_host(host, command, *, input=None, timeout=600, check=True):
        sent.append(command)
        out = subprocess.run(['bash', '-c', mapped(command)], input=input, text=True, capture_output=True)
        if check and out.returncode:
            raise RuntimeError(out.stderr)
        return out.stdout.replace(str(vm), '/content/natlang-work')

    def sync_to(local, remote, host, **kw):
        sent.append(('tunnel', remote))
        shutil.copytree(str(local).rstrip('/'), mapped(remote).rstrip('/'), dirs_exist_ok=True)
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh_host)
    monkeypatch.setattr(pop_rl_actor, 'sync_to_remote', sync_to)
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    return vm, sent


def _state(path, step, fill=b's'):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'learner_manifest.json').write_text(json.dumps({'learner_step': step, 'batch_cursor': {'next_step': step}}))
    (path / 'optimizer_state.pt').write_bytes(fill * 4096)
    (path / 'adapter_factors_fp32.safetensors').write_bytes(fill * 2048)


def test_the_dgx_learner_state_replaces_the_vms_only_whole_and_verified(tmp_path, monkeypatch):
    """The hand-back's push: a consistent snapshot of the DGX state reaches the VM through Drive (the tunnel when Drive
    fails), every file's sha256 is checked against the snapshot, and only then does it replace the VM's state; a copy
    that arrives altered is discarded with the VM's state untouched; a learner running there is never overwritten."""
    import pop_rl_actor
    from scripts import drive_tier
    vm, sent = _fake_vm(tmp_path, monkeypatch)
    cfg = pop_rl_actor.host_config('colab', host='colab-g4')
    run = pop_rl_actor.remote_sft_run(cfg, 'aei-r0001')
    local = tmp_path / 'sft' / 'learner-state'
    _state(local, 304)
    _state(vm / 'runs' / 'sft-aei-r0001' / 'learner-state', 120, fill=b'v')
    monkeypatch.setattr(pop_rl_actor, 'remote_learner_state', lambda host, remote_run: 'rc 1')
    drive = {}
    monkeypatch.setattr(drive_tier, 'put_dir', lambda ns: drive.update(prefix=ns.prefix, files=sorted(
        p.name for p in Path(ns.dir).iterdir())) or 0)

    def get_dir(host, cfg, command, prefix, dest, **kw):
        import shutil
        shutil.copytree(local, str(dest).replace('/content/natlang-work', str(vm)))
        return {'event': 'get-dir', 'prefix': prefix}
    monkeypatch.setattr(pop_rl_actor, 'drive_vm', get_dir)
    assert pop_rl_actor.remote_state_step('colab-g4', run) == 120
    pushed = pop_rl_actor.push_learner_state('colab-g4', cfg, run, local, run_name='aei-r0001')
    assert pushed['step'] == 304 and pushed['via'] == 'drive' and pushed['files'] == 3
    assert drive == {'prefix': 'colab/aei-r0001/handback-state',
                     'files': ['adapter_factors_fp32.safetensors', 'learner_manifest.json', 'optimizer_state.pt']}
    assert pop_rl_actor.remote_state_step('colab-g4', run) == 304
    assert (vm / 'runs/sft-aei-r0001/learner-state/optimizer_state.pt').read_bytes() == b's' * 4096
    assert sorted(p.name for p in (vm / 'runs/sft-aei-r0001').iterdir()) == ['learner-state']
    assert not list(local.parent.glob('learner-state.*'))            # the snapshot is gone
    # Drive fails: the tunnel; an altered copy: discarded, the VM's state as it was
    _state(local, 330, fill=b't')
    monkeypatch.setattr(pop_rl_actor, 'drive_vm', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('403')))
    real_sync = pop_rl_actor.sync_to_remote

    def corrupt(local_dir, remote, host, **kw):
        real_sync(local_dir, remote, host)
        (vm / 'runs/sft-aei-r0001/learner-state.incoming/optimizer_state.pt').write_bytes(b'x')
    monkeypatch.setattr(pop_rl_actor, 'sync_to_remote', corrupt)
    with pytest.raises(RuntimeError, match='does not match'):
        pop_rl_actor.push_learner_state('colab-g4', cfg, run, local, run_name='aei-r0001')
    assert pop_rl_actor.remote_state_step('colab-g4', run) == 304
    assert not (vm / 'runs/sft-aei-r0001/learner-state.incoming').exists()
    monkeypatch.setattr(pop_rl_actor, 'sync_to_remote', real_sync)
    assert pop_rl_actor.push_learner_state('colab-g4', cfg, run, local, run_name='aei-r0001')['via'] == 'tunnel'
    assert pop_rl_actor.remote_state_step('colab-g4', run) == 330
    monkeypatch.setenv('NATLANG_COLAB_DRIVE', '0')                    # the opt-out never touches Drive
    drive.clear()
    _state(local, 331)
    assert pop_rl_actor.push_learner_state('colab-g4', cfg, run, local, run_name='aei-r0001')['via'] == 'tunnel'
    assert drive == {}
    monkeypatch.setattr(pop_rl_actor, 'remote_learner_state', lambda host, remote_run: 'running')
    with pytest.raises(RuntimeError, match='is running'):
        pop_rl_actor.push_learner_state('colab-g4', cfg, run, local, run_name='aei-r0001')


def test_run_sft_pushes_a_dgx_state_ahead_of_the_vms_and_keeps_a_newer_vm_state(tmp_path, monkeypatch):
    """run-sft's resume rule: the DGX state goes to the VM when the VM holds none or an older one (a hand-back whose
    DGX saved once more before it stopped; a VM whose earlier learner failed, 'rc 1', kept a stale state the old rule
    resumed), never when the VM's own state is newer or its learner runs."""
    import pop_rl_actor
    vm, sent = _fake_vm(tmp_path, monkeypatch)
    cfg = pop_rl_actor.host_config('colab', host='colab-g4')
    run = pop_rl_actor.remote_sft_run(cfg, 'aei-r0001')
    local = tmp_path / 'sft' / 'learner-state'
    _state(local, 320)
    states = {'learner': 'rc 1'}
    monkeypatch.setattr(pop_rl_actor, 'remote_learner_state', lambda host, remote_run: states['learner'])
    pushes = []
    monkeypatch.setattr(pop_rl_actor, 'push_learner_state', lambda *a, **k: pushes.append(k['via']) or
                        {'step': 320, 'via': k['via'], 'seconds': 1.0, 'bytes': 1, 'files': 1})
    assert pop_rl_actor.sft_state_push_needed('colab-g4', run, local)             # the VM has none
    _state(vm / 'runs/sft-aei-r0001/learner-state', 304)
    assert pop_rl_actor.sft_state_push_needed('colab-g4', run, local)             # stale (rc 1)
    _state(vm / 'runs/sft-aei-r0001/learner-state', 340)
    assert not pop_rl_actor.sft_state_push_needed('colab-g4', run, local)         # the VM's own is newer
    _state(vm / 'runs/sft-aei-r0001/learner-state', 304)
    states['learner'] = 'running'
    assert not pop_rl_actor.sft_state_push_needed('colab-g4', run, local)
    assert not pop_rl_actor.sft_state_push_needed('colab-g4', run, tmp_path / 'none')


def test_the_move_estimate_counts_only_what_the_vm_lacks(tmp_path, monkeypatch):
    import pop_rl_actor
    vm, sent = _fake_vm(tmp_path, monkeypatch)
    monkeypatch.setenv('NATLANG_COLAB_DRIVE', '0')
    cfg = pop_rl_actor.host_config('colab', host='colab-g4')
    chains, ckpt = tmp_path / 'chains.pkl', tmp_path / 'start.pt'
    chains.write_bytes(b'c' * 2_400_000)
    ckpt.write_bytes(b'k' * 4_800_000)
    launch = {'exp0_sft_args': f'train --chains {chains} --checkpoint {ckpt}', 'start_checkpoint_sha256': 'a' * 64}
    seconds, parts = pop_rl_actor.sft_move_seconds('colab-g4', cfg, run_name='aei-r0001', launch=launch,
                                                   state_bytes=1_200_000)
    assert parts == {'state': 0.5, 'setup': pop_rl_actor.SFT_REMOTE_SETUP_SECONDS, 'chains': 1.0, 'checkpoint': 2.0}
    (vm / 'checkpoints').mkdir()
    (vm / 'checkpoints' / f'{"a" * 64}.pt').write_bytes(b'k')
    (vm / 'runs/sft-aei-r0001').mkdir(parents=True)
    (vm / 'runs/sft-aei-r0001/chains.pkl').write_bytes(b'c')
    seconds, parts = pop_rl_actor.sft_move_seconds('colab-g4', cfg, run_name='aei-r0001', launch=launch,
                                                   state_bytes=1_200_000)
    assert (parts['chains'], parts['checkpoint']) == (0.0, 0.0) and seconds == 0.5 + pop_rl_actor.SFT_REMOTE_SETUP_SECONDS


def test_round_stages_preempt_the_standing_collection_which_resumes_on_its_policy(tmp_path, monkeypatch):
    """2026-11-23: the standing collection (high, 74 GB, not preemptible by medium) held its round's build (medium)
    and every medium job behind it for 96 minutes. Round stages run high; collections medium and resumable: the ledger
    preempts the collection for a waiting stage and requeues it with its launch spec (same policy, pin and work dir);
    the driver joins a requeued collection (no duplicate launch) and cancels it with a stop."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
    import memory_ledger as ml
    ei, d = _driver(tmp_path, monkeypatch)
    assert ml.PRIORITIES[ei.STAGE_PRIORITY] > ml.PRIORITIES[ei.COLLECTION_PRIORITY]
    gib = ml.GIB
    collection = {'budget': 74 * gib, 'used': 70 * gib, 'priority': ei.COLLECTION_PRIORITY, 'class': 'experiment',
                  'admitted': 0, 'reserve': 0.}
    wanted = lambda priority: {'priority': priority, 'budget': 24 * gib, 'reserve': 0}
    assert ml.preemption_plan({'aei-collect-v0001': collection}, 10 * gib, wanted(ei.STAGE_PRIORITY)) == \
        ['aei-collect-v0001']
    assert ml.preemption_plan({'aei-collect-v0001': collection}, 10 * gib, wanted('medium')) == []  # others wait
    launched, stopped, queued = [], [], set()
    monkeypatch.setattr(d.c, 'active', lambda unit: False)
    monkeypatch.setattr(ei, 'ledger_queued', lambda unit: unit in queued)
    monkeypatch.setattr(d.c, 'stop_unit', stopped.append)

    def ledger(unit, budget, priority, command, log, **kw):
        launched.append((unit, priority, command))
        if unit.endswith('-select'):
            Path(command[command.index('--out') + 1]).write_text('[]')
    monkeypatch.setattr(d.c, 'ledger', ledger)
    cur = d.controller.current
    unit = d.ensure_collection(cur)
    assert [(u, p) for u, p, _ in launched] == [(f'{unit}-select', 'medium'), (unit, ei.COLLECTION_PRIORITY)]
    argv = launched[-1][2]
    assert argv[argv.index('--checkpoint-sha256') + 1] == cur['sha256'] and argv[argv.index('--work') + 1] == \
        str(d.collect_dir(cur))
    queued.add(unit)                                 # preempted for a stage: requeued until room returns
    assert d.ensure_collection(cur) == unit and len(launched) == 2      # joined, not launched twice
    d.stop_collections(remote=False)                 # a DGX SFT: the requeued collection must not relaunch beside it
    assert stopped == [unit]
    # the round's frontier, data and build run at the stage priority
    rd = d.root / 'r0001'
    (d.collect_dir(cur) / 'runs.jsonl').write_text(json.dumps({'checkpoint_sha256': cur['sha256'], 'started': 0}) + '\n')
    launched.clear()
    monkeypatch.setattr(d, 'prepare_plan', lambda rd, policies, size=None: (['frontier'], [['data']], ['build']))
    monkeypatch.setattr(ei, 'record_collection', lambda *a, **kw: None)
    monkeypatch.setattr(ei, 'write_round_manifest', lambda *a, **kw: {})
    monkeypatch.setattr(d.c, 'work_tokenizer', lambda: None)
    monkeypatch.setattr(d, 'ensure_teacher', lambda rd: None)

    def stage(unit, budget, priority, command, log, **kw):
        launched.append((unit, priority))
        (rd / 'mix').mkdir(parents=True, exist_ok=True)
        (rd / 'mix' / 'build.json').write_text('{}')
    monkeypatch.setattr(d.c, 'ledger', stage)
    d.prepare(rd, d.lineage_policies())
    assert launched == [(f'aei-r0001-{s}', ei.STAGE_PRIORITY) for s in ('frontier', 'data', 'build')]


def test_a_round_prepares_from_the_lineages_collections_with_frontier_and_previous_policy(tmp_path, monkeypatch):
    ei, d = _driver(tmp_path, monkeypatch, argv=['--aux', 'x=/a.pkl,share=0.1'])
    cur = d.controller.current
    d.controller.record_round_published('r0001', '/ckpt/v1.pt', '1' * 64, base_sha256=cur['sha256'])
    prev, cur = d.controller.previous, d.controller.current
    for p in (prev, cur):
        d.collect_dir(p).mkdir(parents=True)
        (d.collect_dir(p) / 'runs.jsonl').write_text(json.dumps({'checkpoint_sha256': p['sha256'], 'started': 0}) + '\n')
    rd = d.root / 'r0002'
    frontier, lines, build = d.prepare_plan(rd, d.lineage_policies())
    assert frontier[frontier.index('--policy') + 1] == '1' * 64
    assert frontier[frontier.index('--previous-policy') + 1] == 'a' * 64 and '--allow-adjacent-policies' in frontier
    assert [frontier[i + 1] for i, x in enumerate(frontier) if x == '--collect'] == \
        [str(d.collect_dir(prev)), str(d.collect_dir(cur))]
    extract = next(x for x in lines if 'extract' in x)
    assert [extract[i + 1] for i, x in enumerate(extract) if x == '--rl'] == \
        [f'{d.collect_dir(p)}:{d.collect_dir(p)}' for p in (prev, cur)]
    assert extract[extract.index('--out-name') + 1] == 'episodes-aei-r0002'
    assert any('replay' in x for x in lines) and any('neuralese-graph' in x for x in lines)
    assert build[build.index('--frontier') + 1] == str(rd / 'frontier.json')
    assert build[build.index('--previous-policy') + 1] == 'a' * 64
    assert 'x=/a.pkl,share=0.1' in build


def test_a_restarted_driver_resumes_prepare_after_its_last_finished_step(tmp_path, monkeypatch):
    """frontier, data and build each record stages/<step>.done: a driver restarted while the build runs joins the
    build and never reruns frontier or data beneath it."""
    ei, d = _driver(tmp_path, monkeypatch)
    cur = d.controller.current
    d.collect_dir(cur).mkdir(parents=True)
    (d.collect_dir(cur) / 'runs.jsonl').write_text(json.dumps({'checkpoint_sha256': cur['sha256'], 'started': 0}) + '\n')
    monkeypatch.setattr(ei, 'record_collection', lambda *a, **kw: None)
    monkeypatch.setattr(d, 'prepare_plan', lambda rd, pols, size=None: (['frontier'], [['data']], ['build']))
    monkeypatch.setattr(d.c, 'work_tokenizer', lambda: None)
    monkeypatch.setattr(d.c, 'present', lambda unit: False)  # simulated ledger owns no real systemd jobs
    rd = d.root / 'r0001'
    launched = []

    def crashing(unit, budget, prio, cmd, log, **kw):
        launched.append(unit.rsplit('-', 1)[1])
        if unit.endswith('-build'):
            raise SystemExit('driver stopped while the build ran')
    monkeypatch.setattr(d.c, 'ledger', crashing)
    with pytest.raises(SystemExit):
        d.prepare(rd, [cur])
    assert launched == ['frontier', 'data', 'build']
    assert (rd / 'stages' / 'data.done').exists() and not (rd / 'stages' / 'build.done').exists()

    launched.clear()

    def finishing(unit, budget, prio, cmd, log, **kw):
        launched.append(unit.rsplit('-', 1)[1])
        (rd / 'mix').mkdir(parents=True, exist_ok=True)
        (rd / 'mix' / 'build.json').write_text(json.dumps({'admitted_episode_keys': []}))
        (rd / 'mix' / 'chains.pkl').write_bytes(b'x')
    monkeypatch.setattr(d.c, 'ledger', finishing)
    try:
        d.prepare(rd, [cur])
    except Exception:
        pass                                  # what follows the build (manifest, registry) is not this test's subject
    assert launched == ['build']


def test_prepare_never_rewrites_the_data_script_beneath_a_running_data_unit(tmp_path, monkeypatch):
    """sh reads its script by offset: a driver restarted while the data unit runs must leave data.sh as it is."""
    ei, d = _driver(tmp_path, monkeypatch)
    cur = d.controller.current
    d.collect_dir(cur).mkdir(parents=True)
    (d.collect_dir(cur) / 'runs.jsonl').write_text(json.dumps({'checkpoint_sha256': cur['sha256'], 'started': 0}) + '\n')
    monkeypatch.setattr(ei, 'record_collection', lambda *a, **kw: None)
    monkeypatch.setattr(d, 'prepare_plan', lambda rd, pols, size=None: (['frontier'], [['new-data']], ['build']))
    monkeypatch.setattr(d.c, 'work_tokenizer', lambda: None)
    rd = d.root / 'r0002'
    (rd / 'stages').mkdir(parents=True)
    (rd / 'stages' / 'frontier.done').write_text('{}')
    (rd / 'data.sh').write_text('set -e\nold-data\n')
    monkeypatch.setattr(d.c, 'present', lambda unit: unit.endswith('-data'))

    def stop(unit, *a, **kw):
        raise SystemExit('stop after data')
    monkeypatch.setattr(d.c, 'ledger', stop)
    with pytest.raises(SystemExit):
        d.prepare(rd, [cur])
    assert (rd / 'data.sh').read_text() == 'set -e\nold-data\n'


def test_rounds_publish_advance_and_a_crash_after_the_merge_resumes_without_retraining(tmp_path, monkeypatch):
    """The loop publishes every round and advances the lineage (the GRPO phase too); a driver that died after the
    publication but before recording it resumes the same round, finds policy/published.json and only records it."""
    ei, d = _driver(tmp_path, monkeypatch, argv=['--held-every', '2', '--max-rounds', '2'])
    calls = []
    monkeypatch.setattr(d.c, 'pin_stage', lambda: None)
    monkeypatch.setattr(d, 'held_loop', lambda: None)
    monkeypatch.setattr(d, 'ensure_collection', lambda p: calls.append(('collect', p['version'])))
    monkeypatch.setattr(d, 'ensure_remote_collection', lambda p: None)
    monkeypatch.setattr(d, 'stop_collections', lambda **kw: calls.append(('stop',)))
    monkeypatch.setattr(d, 'prepare_ahead', lambda rd, pols: None)

    def prepare(rd, pols):
        if not (rd / 'round.json').exists():
            (rd / 'mix').mkdir(parents=True, exist_ok=True)
            (rd / 'mix' / 'chains.pkl').write_bytes(rd.name.encode())
            ei.write_round_manifest(rd, round_id=rd.name, base_policy_sha=pols[-1]['sha256'], previous_policy_sha=None,
                                    chains_path=rd / 'mix' / 'chains.pkl', episode_ids=[])
            (rd / 'stages' / 'prepared.done').write_text('{}')
        calls.append(('prepare', rd.name))
    monkeypatch.setattr(d, 'prepare', prepare)
    monkeypatch.setattr(d, 'train_sft', lambda rd, p: calls.append(('sft', rd.name, p['version'])) or rd / 'sft' / 'factors')
    monkeypatch.setattr(d, 'train_grpo', lambda rd, p: calls.append(('grpo', rd.name, p['version'])) or rd / 'gf')
    published = []

    def publish_cmd(rd, p, f):
        out = rd / 'policy'
        out.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(rd.name.encode()).hexdigest()
        (out / 'published.json').write_text(json.dumps({'checkpoint': str(out / 'checkpoint.pt'), 'checkpoint_sha256': sha,
                                                        'sglang_export': {'dir': str(out / 'sglang-hf')}}))
        published.append(rd.name)
        return ['true']
    monkeypatch.setattr(d, 'publish_command', publish_cmd)
    monkeypatch.setattr(d.c, 'ledger', lambda unit, budget, prio, cmd, log, **kw: None)
    d.publish_command_orig = None
    # publish() runs publish_command through the ledger: make the ledger call the command builder
    monkeypatch.setattr(d, 'publish', lambda rd, p, f: (publish_cmd(rd, p, f), json.loads((rd / 'policy' / 'published.json').read_text()))[1])
    assert d.run() == 0
    assert [c for c in calls if c[0] == 'sft'] == [('sft', 'r0001', 0), ('sft', 'r0002', 1)]
    assert d.controller.current['version'] == 2 and d.controller.state['trained_rounds'] == ['r0001', 'r0002']
    assert d.controller.state['measuring']['candidate']['version'] == 2            # every 2nd round
    # a crash after r0003's publication: the active round resumes, nothing trains again
    rid = d.controller.begin_round(d.controller.current['sha256'])
    assert rid == 'r0003'
    publish_cmd(d.root / rid, None, None)
    calls.clear()
    d.a.max_rounds = 1
    d2 = ei.AsyncRoundDriver(d.a)
    for name in ('ensure_collection', 'ensure_remote_collection', 'stop_collections', 'prepare_ahead', 'prepare', 'held_loop',
                 'train_grpo'):
        monkeypatch.setattr(d2, name, getattr(d, name))
    monkeypatch.setattr(d2.c, 'pin_stage', lambda: None)
    monkeypatch.setattr(d2.c, 'ledger', lambda *a, **k: None)
    monkeypatch.setattr(d2, 'train_sft', lambda rd, p: (_ for _ in ()).throw(AssertionError('retrained')) if not
                        (rd / 'policy' / 'published.json').exists() else rd / 'sft' / 'factors')
    monkeypatch.setattr(d2, 'publish', lambda rd, p, f: json.loads((rd / 'policy' / 'published.json').read_text()))
    assert d2.run() == 0
    assert d2.controller.state['trained_rounds'][-1] == 'r0003' and d2.controller.current['version'] == 3
    # the GRPO phase publishes and advances too
    d2.controller.state['phase'] = 'grpo'
    monkeypatch.setattr(d2, 'publish', lambda rd, p, f: (publish_cmd(rd, p, f), json.loads((rd / 'policy' / 'published.json').read_text()))[1])
    d2.run()
    assert d2.controller.current['version'] == 4 and any(c[0] == 'grpo' for c in calls)
    assert ('stop',) in calls


def test_a_failure_verdict_ends_the_experiment_and_the_collection_serves_the_reference(tmp_path, monkeypatch, capsys):
    """AGENTS.md: a significant unexplained held regression ends the experiment at that eval. The failure arriving while
    r0001 trains from the candidate rolls the lineage back, discards r0001, issues no new round, leaves the standing
    collection on the reference, writes the bug-hunt record and exits 3 with a message; --resume-after-hunt goes on
    from the reference. A neutral rejection rolls back and keeps going."""
    ei, d = _driver(tmp_path, monkeypatch, argv=['--held-every', '1', '--max-rounds', '2'])
    monkeypatch.setattr(d.c, 'pin_stage', lambda: None)
    monkeypatch.setattr(d, 'held_loop', lambda: None)
    collected = []
    monkeypatch.setattr(d, 'ensure_collection', lambda p: collected.append(p['sha256']))
    for name in ('ensure_remote_collection', 'stop_collections'):
        monkeypatch.setattr(d, name, lambda *a, **kw: None)
    monkeypatch.setattr(d, 'prepare_ahead', lambda rd, pols: None)
    monkeypatch.setattr(d, 'prepare', lambda rd, pols: None)
    monkeypatch.setattr(ei, 'update_round_manifest', lambda *a, **k: None)
    d.controller.record_round_published('r0000', '/ckpt/v1.pt', '1' * 64, base_sha256='a' * 64)
    d.controller.start_measurement()
    trained = []
    verdict = {'verdict': 'failure', 'adopt': False, 'checkpoint': '/ckpt/v1.pt', 'sha256': '1' * 64,
               'vs_start': {'delta': -0.06, 'p': 0.01}}

    def train(rd, p):
        trained.append(p['sha256'])
        if len(trained) == 1:       # the held verdict of v1 arrives while r0001 trains from it
            d.verdicts.append(dict(verdict))
        return rd / 'f'
    monkeypatch.setattr(d, 'train_sft', train)
    monkeypatch.setattr(d, 'publish', lambda rd, p, f: {'checkpoint': str(rd / 'c.pt'),
                                                        'checkpoint_sha256': hashlib.sha256(rd.name.encode()).hexdigest()})
    assert d.run() == 3
    assert trained == ['1' * 64] and (d.root / 'r0001' / 'discarded.json').exists()
    assert collected[-1] == 'a' * 64 and d.controller.current['sha256'] == 'a' * 64
    assert (d.root / f'bug-hunt-{"1" * 12}.json').exists() and d.controller.state['halted']['sha256'] == '1' * 64
    assert 'experiment ends at this eval' in capsys.readouterr().err
    assert d.run() == 3 and trained == ['1' * 64]            # a rerun without the flag stays halted
    d.a.resume_after_hunt, d.a.max_rounds = True, 1
    assert d.run() == 0 and trained == ['1' * 64, 'a' * 64] and not d.controller.state['halted']


def test_a_neutral_rejection_rolls_back_and_keeps_going(tmp_path):
    import expert_iteration as ei
    ctrl = ei.PhaseController.initialize(tmp_path / 's.json', start_checkpoint='/ckpt/v0.pt', start_sha='0' * 64,
                                         held_every=1)
    ctrl.record_round_published('r1', *_policy(1), base_sha256='0' * 64)
    ctrl.start_measurement()
    ev = ctrl.record_verdict({'verdict': 'neutral', 'adopt': False, 'checkpoint': _policy(1)[0], 'sha256': _policy(1)[1]})
    assert ev['action'] == 'rollback' and not ctrl.state.get('halted') and ctrl.current['sha256'] == '0' * 64


def test_dry_run_prints_one_rounds_commands(tmp_path, monkeypatch, capsys):
    ei, d = _driver(tmp_path, monkeypatch, argv=['--dry-run'])
    plan = d.plan()
    assert plan['collect'][plan['collect'].index('--collect-weight-dtype') + 1] == ei.COLLECT_WEIGHT_DTYPE == 'fp8'
    assert plan['publish'][2] == 'publish-round' and plan['grpo'][2] == 'train'
    assert '--held-every-hours' in plan['grpo'] and plan['held']['every'] == 2
    assert json.loads(capsys.readouterr().out)['phase'] == 'sft'


def test_collection_defaults_fp8_weights_and_two_guided_attempts(tmp_path, monkeypatch):
    """Every collection path (the cycle's collect command, ``collect``'s and ``async-drive``'s parsers) serves FP8
    weights (BF16 where the fork or GPU has no FP8 path) and runs two JOLT+ guided attempts (async-train §3, §8)."""
    import expert_iteration as ei
    assert ei.COLLECT_WEIGHT_DTYPE == 'fp8' and ei.TEACHER_ATTEMPTS == 2
    c = ei.Cycle.__new__(ei.Cycle)
    c.a = SimpleNamespace(kv_gb=12, slots=8, fill=None, k=4, collect_hours=1., no_neuralese_variants=True,
                          neuralese_variants=None)
    c.pin, c.collect_dir, c.dir = tmp_path, tmp_path / 'collect', tmp_path
    argv = c.collect_command('u', 'ckpt', 'sha', 8)
    assert argv[argv.index('--collect-weight-dtype') + 1] == 'fp8'
    assert argv[argv.index('--teacher-attempts') + 1] == '2'


def test_a_round_is_marked_trained_only_after_its_export_committed():
    import exp0_sft
    events = []

    def failing():
        events.append('factors')
        raise RuntimeError('preempted mid-export')
    with pytest.raises(RuntimeError):
        exp0_sft.export_round(factors=failing, checkpoint=lambda: events.append('checkpoint'), factors_only=True,
                              round_id='r1', mark_trained=lambda: events.append('marked'))
    assert events == ['factors']                    # unmarked: the resumed run exports again
    events.clear()
    exp0_sft.export_round(factors=lambda: events.append('factors'), checkpoint=lambda: events.append('checkpoint'),
                          factors_only=True, round_id='r1', mark_trained=lambda: events.append('marked'))
    assert events == ['factors', 'marked']
    events.clear()
    exp0_sft.export_round(factors=None, checkpoint=lambda: events.append('checkpoint'), factors_only=False,
                          round_id=None, mark_trained=lambda: events.append('marked'))
    assert events == ['checkpoint']


def test_publication_is_idempotent_and_goes_through_the_ledger(tmp_path, monkeypatch):
    ei, d = _driver(tmp_path, monkeypatch)
    rd = d.root / 'r0001'
    ran = []

    def ledger(unit, budget, prio, cmd, log, **kw):
        ran.append((unit, cmd))
        (rd / 'policy').mkdir(parents=True, exist_ok=True)
        (rd / 'policy' / 'published.json').write_text(json.dumps({'checkpoint_sha256': 'b' * 64}))
    monkeypatch.setattr(d.c, 'ledger', ledger)
    cur = d.controller.current
    assert d.publish(rd, cur, rd / 'sft' / 'factors')['checkpoint_sha256'] == 'b' * 64
    (unit, cmd), = ran
    assert unit == 'aei-r0001-publish' and cmd[2] == 'publish-round'
    assert cmd[cmd.index('--base-checkpoint') + 1] == cur['checkpoint'] and cmd[cmd.index('--out') + 1] == str(rd / 'policy')
    d.publish(rd, cur, rd / 'sft' / 'factors')     # a rerun after a crash: published.json exists, nothing runs again
    assert len(ran) == 1


# ---- remote (Colab) collection through a DGX SFT, rebinds and the round size (DECISIONS 2026-11-23) -------------------
def _remote_driver(tmp_path, monkeypatch, *argv):
    ei, d = _driver(tmp_path, monkeypatch, argv=['--rebind-actor', 'colab-g4@http://127.0.0.1:8570', *argv])
    import pop_rl_actor
    monkeypatch.setattr(pop_rl_actor, 'remote_endpoint', lambda host, port, bind_host=None: f'http://127.0.0.1:{port + 10161}')
    return ei, d


def test_a_dgx_sft_stops_only_the_dgx_collection_and_keeps_the_remote_one_collecting(tmp_path, monkeypatch):
    """The DGX learner needs the DGX actor's memory, not the Colab actor's: stop_collections(remote=False) leaves the
    remote collectors running; the SFT wait keeps (re)starting the remote collection of the policy it trains from
    (a session the keeper replaced and served again gets its collector back)."""
    ei, d = _remote_driver(tmp_path, monkeypatch)
    cur = d.controller.current
    d.collect_dir(cur).mkdir(parents=True)
    unit = d.collection_unit(cur)
    assert d.remote_units(unit) == [f'{unit}-remote0']
    present, stopped = {unit, f'{unit}-fill', f'{unit}-remote0'}, []
    monkeypatch.setattr(d.c, 'present', lambda u: u in present)
    monkeypatch.setattr(d.c, 'stop_unit', lambda u: (stopped.append(u), present.discard(u)))
    d.stop_collections(remote=False)
    assert stopped == [unit, f'{unit}-fill'] and f'{unit}-remote0' in present
    d.stop_collections()
    assert stopped[-1] == f'{unit}-remote0'

    rd = d.root / 'r0001'
    (rd / 'mix').mkdir(parents=True)
    (rd / 'chains.sha256').write_text('f' * 64 + '\n')
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: SimpleNamespace(returncode=0))
    monkeypatch.setattr(d.c, 'active', lambda u: False)
    monkeypatch.setattr(d, 'stop_collections', lambda **kw: None)
    ensured = []
    monkeypatch.setattr(d, 'ensure_remote_collection', lambda p: ensured.append(p['sha256']))

    def wait(unit, poll=60, between=None):
        between()
        between()
        (rd / 'sft' / 'factors').mkdir(parents=True, exist_ok=True)
        (rd / 'sft' / 'factors' / 'policy_manifest.json').write_text('{}')
    monkeypatch.setattr(d.c, 'wait', wait)
    d.train_sft(rd, cur)
    assert ensured == [cur['sha256']] * 2
    # a restarted driver joining the running learner still stops the DGX collection its round start queued again
    (rd / 'sft' / 'factors' / 'policy_manifest.json').unlink()
    stops = []
    monkeypatch.setattr(d, 'stop_collections', lambda **kw: stops.append(kw))
    monkeypatch.setattr(d.c, 'active', lambda u: u.endswith('-sft'))
    monkeypatch.setattr(d.c, 'present', lambda u: u.endswith('-sft'))
    monkeypatch.setattr(d, 'launch_sft', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('relaunched')))
    d.train_sft(rd, cur)
    assert stops == [{'remote': False}]


def test_cycle_wait_calls_between_and_survives_its_failure(tmp_path, monkeypatch):
    ei, d = _driver(tmp_path, monkeypatch)
    polls = iter([True, True, False])
    monkeypatch.setattr(d.c, 'present', lambda u: next(polls))
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: SimpleNamespace(returncode=1))
    calls = []

    def between():
        calls.append(1)
        raise RuntimeError('tunnel down')
    d.c.wait('u', between=between)
    assert len(calls) == 2


def test_the_next_rounds_snapshot_waits_for_the_remote_collection_while_the_dgx_one_is_stopped(tmp_path, monkeypatch):
    """A DGX SFT stops the DGX collection minutes after the round starts: the next round's snapshot (prepared ahead)
    then waited for nothing and froze the newest policy's frontier at those minutes. It waits for the remote
    collection (--snapshot-min-hours from the earliest start) and snapshots at once only when no collection runs."""
    ei, d = _remote_driver(tmp_path, monkeypatch, '--snapshot-min-hours', '2')
    cur = d.controller.current
    remote = d.collect_dir(cur).with_name(d.collect_dir(cur).name + '-remote0')
    for w, started in ((d.collect_dir(cur), 1000.), (remote, 2000.)):
        w.mkdir(parents=True)
        (w / 'runs.jsonl').write_text(json.dumps({'started': started}) + '\n')
    now = {'t': 1000. + 3600}
    present = {f'{d.collection_unit(cur)}-remote0'}
    monkeypatch.setattr(ei.time, 'time', lambda: now['t'])
    slept = []

    def sleep(s):
        slept.append(s)
        now['t'] += 3600
    monkeypatch.setattr(ei.time, 'sleep', sleep)
    monkeypatch.setattr(d.c, 'present', lambda u: u in present)
    monkeypatch.setattr(ei, 'record_collection', lambda *a, **kw: None)
    monkeypatch.setattr(d, 'prepare_plan', lambda rd, pols, size=None: (_ for _ in ()).throw(StopIteration('snapshot taken')))
    with pytest.raises(StopIteration):
        d.prepare(d.root / 'r0002', [cur])
    assert len(slept) == 1                      # waited one poll: 2 h after the DGX collection's start
    slept.clear()
    present.clear()
    now['t'] = 1000.
    with pytest.raises(StopIteration):          # nothing collects: what was scored is the snapshot
        d.prepare(d.root / 'r0003', [cur])
    assert not slept


def test_a_failed_or_missed_rebind_is_retried_and_an_actor_already_serving_is_skipped(tmp_path, monkeypatch):
    ei, d = _remote_driver(tmp_path, monkeypatch)
    import pop_rl_actor
    cur = d.controller.current
    d.controller.record_round_published('r0001', str(tmp_path / 'v1.pt'), '1' * 64, base_sha256=cur['sha256'])
    new = d.controller.current
    published = d.round_dir('r0001') / 'policy' / 'published.json'
    published.parent.mkdir(parents=True)
    published.write_text(json.dumps({'checkpoint_sha256': '1' * 64, 'checkpoint': str(tmp_path / 'v1.pt'),
                                     'factors_dir': str(tmp_path / 'factors'), 'sglang_export': {'dir': '/x'}}))
    serving = {'sha': cur['sha256']}
    monkeypatch.setattr(ei, 'endpoint_alive', lambda url, sha, timeout=10: serving['sha'] == sha)
    rebinds = []

    def rebind(**kw):
        rebinds.append(kw)
        if len(rebinds) == 1:
            raise RuntimeError('ssh: connection closed')
        serving['sha'] = kw['policy_id']
    monkeypatch.setattr(pop_rl_actor, 'rebind_actor', rebind)
    monkeypatch.setattr(d.c, 'present', lambda u: True)          # the collector of the new policy runs: no launch
    d.ensure_remote_collection(new)
    assert len(rebinds) == 1 and rebinds[0]['checkpoint'] == str(tmp_path / 'v1.pt')
    assert rebinds[0]['front_url'] == 'http://127.0.0.1:8570' and rebinds[0]['host'] == 'colab-g4'
    d.ensure_remote_collection(new)                              # within the retry interval: no second try
    assert len(rebinds) == 1
    d._rebind_tried.clear()
    d.ensure_remote_collection(new)
    assert len(rebinds) == 2 and serving['sha'] == '1' * 64
    d._rebind_tried.clear()
    d.ensure_remote_collection(new)                              # already serving: nothing to do
    d.rebind_remote(json.loads(published.read_text()), str(tmp_path / 'factors'), new['version'])
    assert len(rebinds) == 2
    d.ensure_remote_collection(cur)                              # the start policy has no factors: never rebound
    assert len(rebinds) == 2


def test_a_rebound_colab_actors_record_names_the_new_policy_for_the_keepers_re_serve(tmp_path, monkeypatch):
    import pop_rl_actor
    sys.path.insert(0, str(REPO / 'scripts' / 'colab'))
    import session_keeper
    record = {'policy_id': 'a' * 64, 'endpoint': 'http://127.0.0.1:18731', 'pid': 7,
              'serve_args': {'checkpoint': '/ckpt/v0.pt', 'policy_id': 'a' * 64, 'commit': 'c' * 40, 'port': 8570,
                             'kv_cache_dtype': 'fp8_e4m3', 'weight_dtype': 'bfloat16', 'host': 'colab-g4'}}
    rebound = {'front_url': 'http://127.0.0.1:8570', 'policy_id': 'b' * 64, 'version': 3,
               'export_dir': '/content/natlang-work/sglang-hf-v3-bbbb', 'weights_sha256': 'e' * 64}
    out = pop_rl_actor.rebound_record(record, rebound, '/data/aei1/r0003/policy/checkpoint.pt')
    assert out['policy_id'] == 'b' * 64 and out['serve_args']['policy_id'] == 'b' * 64
    assert out['serve_args']['checkpoint'] == '/data/aei1/r0003/policy/checkpoint.pt'
    assert out['serve_args']['commit'] == 'c' * 40 and out['rebound']['from_policy_id'] == 'a' * 64
    assert pop_rl_actor.rebound_record(record, rebound, None) is None                 # no DGX checkpoint to serve
    assert pop_rl_actor.rebound_record(record, {**rebound, 'front_url': 'http://127.0.0.1:8571'}, '/c') is None
    assert pop_rl_actor.rebound_record({}, rebound, '/c') is None
    # rebind_actor writes it for a Colab host after the attestation
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path)
    pop_rl_actor.write_json(pop_rl_actor.colab_state_path('colab-g4'), record)
    monkeypatch.setattr(pop_rl_actor, 'host_config', lambda profile, host=None: {})
    monkeypatch.setattr(pop_rl_actor, '_prepare_remote_rebind_export', lambda **kw: (rebound['export_dir'], 'e' * 64))
    monkeypatch.setattr(pop_rl_actor, '_http_json', lambda url, payload=None, **kw:
                        {'success': True} if url.endswith('update_weights_from_disk') else {})
    import natlang_neuralese.serve.sglang_actor as sa
    monkeypatch.setattr(sa, 'attest_all_routes', lambda info, expected: (True, {}))
    pop_rl_actor.rebind_actor(front_url='http://127.0.0.1:8570', policy_id='b' * 64, version=3,
                              factors_dir=tmp_path, host='colab-g4', checkpoint='/data/aei1/r0003/policy/checkpoint.pt')
    # the keeper serves a replaced session again from the record: the rebound policy
    keeper = session_keeper.Keeper(SimpleNamespace(order='A100:hm,G4,L4', state=str(tmp_path / 'k.json'),
                                                   actor_root=str(tmp_path), alias='colab-g4'))
    serve = keeper.actor_record()['serve_args']
    assert (serve['checkpoint'], serve['policy_id']) == ('/data/aei1/r0003/policy/checkpoint.pt', 'b' * 64)


def test_the_round_mix_is_capped_with_its_composition_whole_episodes_and_unseen_first(tmp_path):
    import numpy as np
    import sft_sources

    def chain(part, task, round_, n):
        return {'part': part, 'task': task, 'round': round_, 'ids': np.zeros(n + 4, dtype=np.int32),
                'segs': [(4, n, {'sample': np.zeros(n)})]}
    chains = [chain('rl-own', 't1', 'own:1', 100), chain('rl-own', 't2', 'own:2', 100),
              chain('teacher', 't3', 'teacher:3', 50), chain('teacher', 't3', 'teacher:3', 50),
              chain('teacher', 't4', 'teacher:4', 100),
              chain('filled', 't5', 'teacher:5', 100), chain('filled', 't6', 'teacher:6', 100)]
    kept, report = sft_sources.cap_target_tokens(chains, 0)
    assert kept is chains and not report['applied']
    kept, report = sft_sources.cap_target_tokens(chains, 10_000)
    assert kept is chains and not report['applied']
    kept, report = sft_sources.cap_target_tokens(chains, 300, trained_keys={'own:1', 'teacher:5'})
    assert report['applied'] and report['fraction'] == 0.5
    rounds = [c['round'] for c in kept]
    # every part keeps half; the trained episode goes last (own:2, teacher:6 kept); a two-chain episode stays whole
    assert rounds == ['own:2', 'teacher:3', 'teacher:3', 'teacher:6']
    assert {p: v['kept_weighted_target_tokens'] for p, v in report['parts'].items()} == \
        {'rl-own': 100, 'teacher': 100, 'filled': 100}
    # the next build's prefer-unseen treats a dropped episode as never trained
    out = tmp_path / 'mix'
    out.mkdir()
    import zstandard
    rows = [{'episode': k, 'task_id': t} for k, t in (('own:1', 't1'), ('own:2', 't2'), ('teacher:4', 't4'))]
    (out / 'rows.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(
        ''.join(json.dumps(r) + '\n' for r in rows).encode()))
    (out / 'build.json').write_text(json.dumps({'target_cap': {'dropped_episode_keys': ['own:1', 'teacher:4']}}))
    keys, tasks = sft_sources.trained_in([out])
    assert keys == {'own:2'} and tasks == {'t2'}


def test_async_rounds_build_capped_mixes_by_default(tmp_path, monkeypatch):
    ei, d = _remote_driver(tmp_path, monkeypatch)
    cur = d.controller.current
    d.collect_dir(cur).mkdir(parents=True)
    (d.collect_dir(cur) / 'runs.jsonl').write_text(json.dumps({'started': 0}) + '\n')
    _, _, build = d.prepare_plan(d.root / 'r0001', [cur])
    assert build[build.index('--max-target-tokens') + 1] == str(ei.ROUND_TARGET_TOKENS) == '12000000'
    assert "'--max-target-tokens'" in (REPO / 'scripts/neuralese_data/sft_sources.py').read_text()
    _, d0 = _remote_driver(tmp_path / 'off', monkeypatch, '--round-target-tokens', '0')
    d0.collect_dir(d0.controller.current).mkdir(parents=True)
    (d0.collect_dir(d0.controller.current) / 'runs.jsonl').write_text(json.dumps({'started': 0}) + '\n')
    assert '--max-target-tokens' not in d0.prepare_plan(d0.root / 'r0001', [d0.controller.current])[2]


def test_dry_run_shows_the_remote_collection_and_the_dgx_sft_sizing(tmp_path, monkeypatch, capsys):
    ei, d = _remote_driver(tmp_path, monkeypatch, '--remote-slots', '64', '--dry-run')
    monkeypatch.setattr(ei, 'endpoint_alive', lambda url, sha, timeout=10: True)
    plan = d.plan()
    (remote,) = plan['remote_collect']
    assert remote['endpoint'] == 'http://127.0.0.1:18731' and remote['serves_policy'] and remote['budget_gb'] == 23
    argv = remote['collect']
    assert argv[argv.index('--actor') + 1] == 'http://127.0.0.1:18731=64' and argv[argv.index('--slots') + 1] == '64'
    assert argv[argv.index('--work') + 1].endswith('-remote0')
    assert argv[argv.index('--checkpoint-sha256') + 1] == d.controller.current['sha256']
    assert plan['sft_dgx'] == {'activation_gb': 56, 'budget_gb': 80}


# -- placement-aware round budgets (DECISIONS 2026-11-24) -------------------------------------------------------------

def _colab_driver(tmp_path, monkeypatch, *argv, gpu='NVIDIA A100-SXM4-80GB', fit=True):
    ei, d = _remote_driver(tmp_path, monkeypatch, '--sft-profile', 'colab', '--colab-hosts', 'colab-g4', *argv)
    import pop_rl_actor
    monkeypatch.setattr(pop_rl_actor, 'reachable_hosts', lambda profile, hosts: list(hosts))
    monkeypatch.setattr(pop_rl_actor, 'colab_sft_fit', lambda host: (fit, 'test'))
    monkeypatch.setattr(pop_rl_actor, 'colab_hardware', lambda host: {'gpu_name': gpu, 'vram_total_gib': 80})
    d.colab_hosts = ['colab-g4']
    cur = d.controller.current
    d.collect_dir(cur).mkdir(parents=True)
    (d.collect_dir(cur) / 'runs.jsonl').write_text(json.dumps({'started': 0}) + '\n')
    return ei, d


def test_a_round_for_a_fit_session_is_sized_to_its_hours_and_a_dgx_round_keeps_the_dgx_size(tmp_path, monkeypatch):
    ei, d = _colab_driver(tmp_path, monkeypatch)
    size = d.round_size()
    # 2 h at the A100's 30.8k chain tokens/s: 221.8M forward tokens, 1.73x the 12M-target DGX round (x10.7 forward)
    assert size['placement'] == 'colab' and size['max_forward_tokens'] == 2 * 3600 * 30_800
    assert size['scale'] == round(2 * 3600 * 30_800 / (12_000_000 * 10.7), 3)
    assert abs(size['max_target_tokens'] - 12_000_000 * size['scale']) < 12_000 and size['teacher_budget'] > 180_000_000
    _, _, build = d.prepare_plan(d.root / 'r0001', [d.controller.current], size)
    assert build[build.index('--max-forward-tokens') + 1] == str(size['max_forward_tokens'])
    assert build[build.index('--teacher-forward-budget') + 1] == str(size['teacher_budget'])
    # measured throughput wins over the default; a GPU with neither uses the GB10's times sft_speedup
    (d.root / ei.ROUND_SPEED_FILE).write_text(json.dumps({'NVIDIA A100-SXM4-80GB': {'chain_tokens_per_s': 40_000}}))
    assert d.round_size()['max_forward_tokens'] == 2 * 3600 * 40_000
    _, unfit = _colab_driver(tmp_path / 'unfit', monkeypatch, fit=False)
    assert unfit.round_size()['placement'] == 'dgx'
    _, off = _colab_driver(tmp_path / 'off', monkeypatch, '--colab-round-hours', '0')
    assert off.round_size()['placement'] == 'dgx'
    _, _, build = off.prepare_plan(off.root / 'r0001', [off.controller.current])
    assert '--max-forward-tokens' not in build and build[build.index('--max-target-tokens') + 1] == '12000000'
    assert build[build.index('--teacher-forward-budget') + 1] == '180000000'
    # never smaller than the DGX round
    small = ei.placement_round_size(round_target_tokens=12_000_000, teacher_budget=180_000_000, hours=0.1,
                                    gpu_name='x', chain_tokens_per_s=30_800)
    assert small['scale'] == 1.0 and small['max_target_tokens'] == 12_000_000


def test_a_round_prepared_ahead_prefers_unseen_over_the_round_training_now(tmp_path, monkeypatch):
    ei, d = _colab_driver(tmp_path, monkeypatch)
    trained = d.root / 'r0001' / 'mix'
    trained.mkdir(parents=True)
    (trained / 'rows.jsonl.zst').write_bytes(b'')
    d.controller.state['active_round'] = {'round_id': 'r0001', 'base_sha256': 'a' * 64, 'phase': 'sft'}
    assert d.prefer_unseen_for(d.root / 'r0002')[-1] == str(trained)
    assert str(trained) not in d.prefer_unseen_for(d.root / 'r0001')
    _, _, build = d.prepare_plan(d.root / 'r0002', [d.controller.current])
    assert build[len(build) - 1 - build[::-1].index('--prefer-unseen') + 1] == str(trained)
    _, off = _colab_driver(tmp_path / 'off', monkeypatch, '--no-prefer-inflight')
    off.controller.state['active_round'] = {'round_id': 'r0001'}
    (off.root / 'r0001' / 'mix').mkdir(parents=True)
    (off.root / 'r0001' / 'mix' / 'rows.jsonl.zst').write_bytes(b'')
    assert off.prefer_unseen_for(off.root / 'r0002') == off.controller.state['prefer_unseen']


def test_a_round_sized_for_a_session_that_starts_on_the_dgx_is_recapped_first(tmp_path, monkeypatch):
    ei, d = _colab_driver(tmp_path, monkeypatch)
    rd = d.root / 'r0001'
    (rd / 'mix').mkdir(parents=True)
    (rd / 'mix' / 'chains.pkl').write_bytes(b'full')
    ei.write_round_manifest(rd, round_id='r0001', base_policy_sha='a' * 64, previous_policy_sha=None,
                            chains_path=rd / 'mix' / 'chains.pkl', episode_ids=['e1', 'e2'],
                            extra={'size': {'placement': 'colab'}, 'mix_dir': str(rd / 'mix')})
    ran = []

    def ledger(unit, budget, priority, argv, log, **kw):
        ran.append(list(map(str, argv)))
        out = Path(argv[argv.index('--out') + 1])
        out.mkdir(parents=True)
        (out / 'chains.pkl').write_bytes(b'capped')
        (out / 'build.json').write_text(json.dumps({'admitted_episode_keys': ['e1'], 'mix': {'forward_tokens': 1},
                                                    'target_cap': {'kept_weighted_target_tokens': 5}}))
    monkeypatch.setattr(d.c, 'ledger', ledger)
    # a learner that already trained keeps its step list
    (rd / 'sft' / 'learner-state').mkdir(parents=True)
    (rd / 'sft' / 'learner-state' / 'learner_manifest.json').write_text('{}')
    assert not d.recap_for_dgx(rd) and not ran
    (rd / 'sft' / 'learner-state' / 'learner_manifest.json').unlink()
    assert d.recap_for_dgx(rd)
    argv = ran[0]
    assert 'recap' in argv and argv[argv.index('--max-target-tokens') + 1] == '12000000'
    manifest = json.loads((rd / 'round.json').read_text())
    assert manifest['mix_dir'] == str(rd / 'mix-dgx') and manifest['episode_ids'] == ['e1']
    assert manifest['chains_sha256'] == hashlib.sha256(b'capped').hexdigest() == (rd / 'chains.sha256').read_text().strip()
    assert d.round_mix(rd) == rd / 'mix-dgx'
    assert d.sft_env(rd, d.controller.current, activation_gb=56, preflight_only=False)['CHAINS'] == \
        str(rd / 'mix-dgx' / 'chains.pkl')
    assert not d.recap_for_dgx(rd) and len(ran) == 1               # once
    # a DGX-sized round and the opt-out are never recapped
    _, off = _colab_driver(tmp_path / 'off', monkeypatch, '--no-dgx-recap')
    assert not off.recap_for_dgx(rd)


def test_round_learner_throughput_is_recorded_per_gpu(tmp_path, monkeypatch):
    ei, d = _colab_driver(tmp_path, monkeypatch)
    rd = d.root / 'r0001'
    (rd / 'sft').mkdir(parents=True)
    ei.write_round_manifest(rd, round_id='r0001', base_policy_sha='a' * 64, previous_policy_sha=None,
                            chains_path=_touch(rd / 'mix' / 'chains.pkl'), episode_ids=[], profile='colab')
    steps = lambda first, n, sec: ''.join(json.dumps({'step': first + i, 'of': 100, 'step_seconds': sec,
                                                      'chain_tokens': 300_000}) + '\n' for i in range(n))
    (rd / 'sft' / 'train.dgx-1.jsonl').write_text(steps(1, 30, 50.0))
    (rd / 'sft' / 'train.jsonl').write_text(steps(31, 40, 10.0))
    (rd / 'sft' / 'handback.json').write_text(json.dumps([{'pushed_step': 31}]))
    (rd / 'sft' / 'colab-placement.json').write_text(json.dumps({'host': 'colab-g4', 'gpu': 'NVIDIA A100'}))
    d.record_round_speed(rd)
    table = json.loads((d.root / ei.ROUND_SPEED_FILE).read_text())
    assert table['GB10']['chain_tokens_per_s'] == 6000 and table['NVIDIA A100']['chain_tokens_per_s'] == 30_000
    assert ei.learner_throughput('NVIDIA A100', table)[0] == 30_000
    assert d.expected_hours({'mix': {'forward_tokens': 30_000 * 3600}})['NVIDIA A100'] == 1.0


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x')
    return path


def test_the_round_cap_binds_on_forward_tokens_and_prefers_unseen_aux_units(tmp_path, monkeypatch):
    import numpy as np
    import pickle
    import sft_sources

    def chain(part, task, round_, n, forward):
        return {'part': part, 'task': task, 'round': round_, 'ids': np.zeros(forward, dtype=np.int32),
                'segs': [(4, n, {'sample': np.zeros(n)})]}
    chains = [chain('filled', 't1', 'filled:a', 100, 1000), chain('filled', 't2', 'filled:b', 100, 1000),
              chain('operators', 'op1', '0', 100, 1000), chain('operators', 'op2', '0', 100, 1000)]
    kept, report = sft_sources.cap_target_tokens(chains, 0, max_forward=2000,
                                                 trained_unit_keys={('t1', 'filled:a'), ('op1', '0')})
    assert report['binding'] == 'forward' and report['fraction'] == 0.5
    assert [(c['task'], c['round']) for c in kept] == [('t2', 'filled:b'), ('op2', '0')]
    assert report['parts']['filled']['unseen_episodes'] == 1 and report['parts']['filled']['kept_trained_before'] == 0
    # the tighter bound wins
    assert sft_sources.cap_target_tokens(chains, 100, max_forward=3000)[1]['binding'] == 'target'
    # trained units: the sidecar, else the build's chains.pkl (aux parts included, which rows.jsonl.zst lacks)
    legacy, new = tmp_path / 'legacy', tmp_path / 'new'
    legacy.mkdir()
    with (legacy / 'chains.pkl').open('wb') as handle:
        pickle.dump({'chains': chains[:1]}, handle)
    new.mkdir()
    sft_sources.write_trained_units(new, chains[2:3])
    assert sft_sources.trained_units([legacy, new]) == {('t1', 'filled:a'), ('op1', '0')}


def test_recap_bounds_a_built_mix_and_records_what_it_left_out(tmp_path, monkeypatch):
    import numpy as np
    import pickle
    import sft_sources
    monkeypatch.setenv('NATLANG_ALLOW_UNREGISTERED_DATA', '1')
    monkeypatch.setattr(sft_sources.data_registry, 'record_dataset', lambda *a, **k: None)
    src = tmp_path / 'mix'
    src.mkdir()

    def chain(part, task, round_, n):
        return {'part': part, 'task': task, 'round': round_, 'ids': np.zeros(n, dtype=np.int32),
                'segs': [(0, n, {'sample': np.zeros(n)})]}
    chains = [chain('rl-own', 't1', 'own:1', 100), chain('rl-own', 't2', 'own:2', 100),
              chain('filled', 't3', 'filled:3', 100), chain('filled', 't4', 'filled:4', 100)]
    with (src / 'chains.pkl').open('wb') as handle:
        pickle.dump({'chains': chains, 'schema': 'natlang.sft-chains/1'}, handle)
    out = sft_sources.ZstdWriter(src / 'rows.jsonl.zst')
    for key, task in (('own:1', 't1'), ('own:2', 't2')):
        out.write({'episode': key, 'task_id': task})
    out.close()
    (src / 'build.json').write_text(json.dumps({'admitted_episode_keys': ['own:1', 'own:2'], 'target_cap': {}}))
    seen = tmp_path / 'seen'
    seen.mkdir()
    sft_sources.write_trained_units(seen, [chains[0], chains[2]])
    sft_sources.ZstdWriter(seen / 'rows.jsonl.zst').close()
    dst = tmp_path / 'mix-dgx'
    sft_sources.recap(SimpleNamespace(mix=str(src), out=str(dst), max_target_tokens=200, max_forward_tokens=0,
                                      prefer_unseen=[str(seen)]))
    with (dst / 'chains.pkl').open('rb') as handle:
        assert [c['round'] for c in pickle.load(handle)['chains']] == ['own:2', 'filled:4']
    report = json.loads((dst / 'build.json').read_text())
    assert report['admitted_episode_keys'] == ['own:2'] and report['target_cap']['dropped_episode_keys'] == ['own:1']
    assert sft_sources.trained_in([dst]) == ({'own:2'}, {'t2'})
    assert sft_sources.trained_units([dst]) == {('t2', 'own:2'), ('t4', 'filled:4')}


def _pop_driver(tmp_path, monkeypatch):
    import pop_rl_actor
    monkeypatch.setattr(pop_rl_actor, 'POP', 'pop-host')
    monkeypatch.setattr(pop_rl_actor, 'STATE', tmp_path / 'rl-actor.json')
    ei, d = _remote_driver(tmp_path, monkeypatch, '--rebind-actor', 'pop-host@http://100.1.2.3:18531=24',
                           '--remote-slots', '64')
    monkeypatch.setattr(pop_rl_actor, 'remote_endpoint', lambda host, port, bind_host=None:
                        'http://100.1.2.3:18531' if host == 'pop-host' else f'http://127.0.0.1:{port + 10161}')
    return ei, d, pop_rl_actor


def test_rebind_targets_carry_their_own_slots_and_pop_is_a_standing_actor_with_its_own_unit(tmp_path, monkeypatch):
    ei, d, pop_rl_actor = _pop_driver(tmp_path, monkeypatch)
    assert ei.parse_remote_target('colab-g4@http://127.0.0.1:8570', 64) == ('colab-g4', 'http://127.0.0.1:8570', 64)
    assert ei.parse_remote_target('pop-host@http://100.1.2.3:18531=24', 64) == ('pop-host', 'http://100.1.2.3:18531', 24)
    t = d.remote_targets()
    assert [(x['host'], x['slots'], x['pop'], x['suffix']) for x in t] == [
        ('colab-g4', 64, False, '-remote0'), ('pop-host', 24, True, '-pop')]
    cur = d.controller.current
    unit = d.collection_unit(cur)
    assert d.remote_units(unit) == [f'{unit}-remote0', f'{unit}-pop']
    assert [a[0] for a in d.remote_actors(include_pop=False)] == ['colab-g4']
    _u, work, argv, budget = d.remote_collect_argv(cur, t[1], 'http://100.1.2.3:18531')
    assert work.name.endswith('-pop') and argv[argv.index('--slots') + 1] == '24'
    assert argv[argv.index('--actor') + 1] == 'http://100.1.2.3:18531=24' and budget == 11
    assert argv[argv.index('--tasks') + 1] == str(d.collect_dir(cur) / 'tasks.json')
    # the Colab collector only: Pop's is pop_step's
    started = []
    monkeypatch.setattr(ei, 'endpoint_alive', lambda url, sha, timeout=10: True)
    monkeypatch.setattr(d.c, 'present', lambda u: False)
    monkeypatch.setattr(d.c, 'ledger', lambda unit, *a, **kw: started.append(unit))
    (d.collect_dir(cur)).mkdir(parents=True)
    (d.collect_dir(cur) / 'tasks.json').write_text('[]')
    d.ensure_remote_collection(cur)
    assert started == [f'{unit}-remote0']
    # the publication rebinds Colab in place and leaves Pop to pop_step
    rebinds = []
    monkeypatch.setattr(ei, 'endpoint_alive', lambda url, sha, timeout=10: False)
    monkeypatch.setattr(pop_rl_actor, 'rebind_actor', lambda **kw: rebinds.append(kw))
    d.rebind_remote({'checkpoint_sha256': '1' * 64, 'checkpoint': '/x.pt', 'sglang_export': {'dir': '/x'}}, '/f', 2)
    assert [r['host'] for r in rebinds] == ['colab-g4'] and rebinds[0]['front_url'] == 'http://127.0.0.1:8570'


def test_pop_yields_to_a_held_measurement_and_serves_the_head_again_after_it(tmp_path, monkeypatch):
    ei, d, pop_rl_actor = _pop_driver(tmp_path, monkeypatch)
    cur = d.controller.current
    want = d.collection_unit(cur) + '-pop'
    (d.root / 'collect' / 'v0000-old').mkdir(parents=True)
    d.collect_dir(cur).mkdir(parents=True)
    (d.collect_dir(cur) / 'tasks.json').write_text('[]')
    present, stopped, launched, served = {want}, [], [], []
    monkeypatch.setattr(d.c, 'present', lambda u: u in present)
    monkeypatch.setattr(d.c, 'stop_unit', lambda u: (stopped.append(u), present.discard(u)))
    monkeypatch.setattr(d.c, 'ledger', lambda unit, *a, **kw: (launched.append(unit), present.add(unit)))
    busy = {'v': 'held measurement (pid 7)'}
    monkeypatch.setattr(pop_rl_actor, 'pop_busy', lambda: busy['v'])
    state = {}
    monkeypatch.setattr(pop_rl_actor, 'read_json', lambda path, default=None: dict(state) or default)
    monkeypatch.setattr(pop_rl_actor, 'attested', lambda endpoint, sha, adapters=None: state.get('policy_id') == sha)
    monkeypatch.setattr(ei, 'endpoint_alive', lambda url, sha, timeout=10: state.get('policy_id') == sha)

    def run(argv, **kw):
        served.append(argv)
        settings, _ = pop_rl_actor.pop_sglang('fp8_e4m3', 32, weight_dtype='fp8', overlap_schedule=True)
        state.update(policy_id=argv[argv.index('--policy-id') + 1], sglang=settings, endpoint='http://100.1.2.3:18531')
    monkeypatch.setattr(ei.subprocess, 'run', run)
    d.pop_step()                                    # a held measurement holds Pop: the collector stops, no serve
    assert stopped == [want] and not served and not launched
    busy['v'] = None
    d.pop_step()                                    # Pop is free: served with the collection law, collector started
    assert len(served) == 1 and launched == [want]
    argv = served[0]
    assert argv[argv.index('--weight-dtype') + 1] == 'fp8' and '--overlap-schedule' in argv
    assert argv[argv.index('--max-running') + 1] == '32' and argv[argv.index('--commit') + 1] == 'c' * 40
    d.pop_step()                                    # already serving and collecting: nothing to do
    assert len(served) == 1 and launched == [want]
    with d.POP_LOCK:                                # held-v4 here holds Pop: no serve while it does
        state.clear()
        present.clear()
        d.pop_step()
    assert len(served) == 1
    d.controller.state['phase'] = 'grpo'            # the GRPO phase: no Pop collection
    present.add(want)
    d.pop_step()
    assert want not in present and len(served) == 1


def test_pop_actor_driver_keeps_a_dead_engines_log_tail_outside_its_work_directory(tmp_path):
    import pop_actor_driver
    work = tmp_path / 'held-actor-x'
    work.mkdir()
    (work / 'actor.log').write_text('\n'.join(f'line {i}' for i in range(1000)) +
                                    '\ntorch.OutOfMemoryError: CUDA out of memory.\nSIGQUIT received\n')
    kept = pop_actor_driver.keep_crash_tail(work, lines=50)
    path = Path(kept['path'])
    assert path.parent == tmp_path / 'actor-crashes' and path.name.startswith('held-actor-x-')
    text = path.read_text().splitlines()
    assert len(text) == 50 and text[-1] == 'SIGQUIT received'
    assert kept['last_error'] == 'torch.OutOfMemoryError: CUDA out of memory.'
    assert 'error' in pop_actor_driver.keep_crash_tail(tmp_path / 'gone')


def test_pop_actor_driver_and_serve_pass_the_collection_law(tmp_path):
    import pop_actor_driver
    import pop_rl_actor
    settings, fmt = pop_rl_actor.pop_sglang('fp8_e4m3', 32, weight_dtype='fp8', overlap_schedule=True)
    assert fmt.endswith('-kv-fp8_e4m3-w-fp8')
    assert pop_actor_driver.weight_args(settings) == ['--sglang-weight-dtype', 'fp8', '--sglang-overlap-schedule']
    held, _ = pop_rl_actor.pop_sglang('fp8_e4m3', 20)
    assert pop_actor_driver.weight_args(held) == [] and 'overlap_schedule' not in held   # held settings unchanged


def test_pop_held_processes_are_found_and_the_serve_lock_marks_pop_busy(tmp_path, monkeypatch):
    import pop_rl_actor
    proc = tmp_path / 'proc'
    for pid, argv in ((11, ['/bin/sh', '/ops/grpo-v4/pop-held.sh', '/pin', 'ckpt', '/out', '4', '/c.pt']),
                      (12, ['/bin/sh', '/ops/grpo-v4/pop-held.sh', '/pin', 'compare', '/a', '/b']),
                      (13, ['python', '/r/scripts/neuralese_data/pop_rl_actor.py', 'held-eval', '--spec', 's']),
                      (14, ['python', '/r/scripts/neuralese_data/pop_rl_actor.py', 'serve-checkpoint'])):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / 'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv) + b'\0')
    (proc / 'self').mkdir()
    assert sorted(pop_rl_actor.pop_held_running(proc)) == [11, 13]
    monkeypatch.setattr(pop_rl_actor, 'SERVE_LOCK', tmp_path / 'pop-serve.lock')
    monkeypatch.setattr(pop_rl_actor, 'pop_held_running', lambda: [])
    monkeypatch.setattr(pop_rl_actor, 'pop_held_units', lambda pattern='*popheld*': [])
    monkeypatch.setattr(pop_rl_actor, 'pop_host_address', lambda: '100.1.2.3')
    monkeypatch.setattr(pop_rl_actor, 'held_measure_pids', lambda address: [])
    monkeypatch.setattr(pop_rl_actor, 'pop_side_held', lambda: [])
    assert pop_rl_actor.pop_busy() is None
    import fcntl
    with open(tmp_path / 'pop-serve.lock', 'a') as other:     # another process serving (flock is per open file)
        fcntl.flock(other, fcntl.LOCK_EX)
        assert 'serve lock' in pop_rl_actor.pop_busy()
    monkeypatch.setattr(pop_rl_actor, 'pop_held_running', lambda: [11])
    assert 'held measurement' in pop_rl_actor.pop_busy()


def test_the_standing_teacher_campaign_relaunches_from_the_newest_frontier_whenever_idle(tmp_path, monkeypatch):
    """2026-11-24: the campaign was relaunched only when a round was prepared; it ended about 01:00 and a restarted
    driver joining the already-prepared r0002 left it idle until a hand launch at 02:10. Now whenever the unit is
    idle the newest frontier's campaign runs (teacher-r0002), from the main loop, the waits and a standing thread;
    a campaign of that frontier that ran resumes at most hourly and not after a relaunch that scored nothing."""
    ei, d = _driver(tmp_path, monkeypatch)
    for rid in ('r0001', 'r0002'):
        (d.root / rid).mkdir(parents=True)
        (d.root / rid / 'frontier.json').write_text('{}')
    old = d.root / 'teacher-r0001'
    old.mkdir()
    (old / 'launches.jsonl').write_text('{}\n')
    (old / 'manifest.jsonl').write_text('{}\n' * 40)
    present, launched = {'teacher': True}, []
    monkeypatch.setattr(d.c, 'present', lambda unit: unit == 'aei-teacher' and present['teacher'])

    def launch(runner, work, frontier, groups, unit, covered_extra=(), standing=False, student_rates=()):
        assert standing
        launched.append((Path(work).name, Path(frontier).parent.name, unit))
        Path(work).mkdir(parents=True, exist_ok=True)
        with (Path(work) / 'launches.jsonl').open('a') as handle:
            handle.write('{}\n')
        return 7
    monkeypatch.setattr(ei, 'launch_teacher', launch)
    assert d.ensure_teacher() is None and launched == []            # running: nothing
    present['teacher'] = False
    assert d.ensure_teacher() == d.root / 'teacher-r0002'
    assert launched == [('teacher-r0002', 'r0002', 'aei-teacher')]
    assert d.ensure_teacher() is None and len(launched) == 1        # just launched (and it scored nothing since)
    (d.root / 'teacher-r0002' / 'manifest.jsonl').write_text('{}\n' * 3)
    assert d.ensure_teacher() is None and len(launched) == 1        # scored, but within the hour
    now = time.time()
    monkeypatch.setattr(ei.time, 'time', lambda: now + 3700)
    assert d.ensure_teacher() == d.root / 'teacher-r0002' and len(launched) == 2    # resumes
    monkeypatch.setattr(ei.time, 'time', lambda: now + 9000)
    assert d.ensure_teacher() is None and len(launched) == 2        # that resume scored nothing new: done
    (d.root / 'r0003').mkdir()
    (d.root / 'r0003' / 'frontier.json').write_text('{}')
    (d.root / 'r0003' / 'discarded.json').write_text('{}')          # a discarded round's frontier is not the newest
    assert d.newest_frontier() == d.root / 'r0002' / 'frontier.json'
    (d.root / 'r0004').mkdir()
    (d.root / 'r0004' / 'frontier.json').write_text('{}')
    # a newer frontier while the standing queue runs: it drains (running episodes finish), then the newer one launches
    present['teacher'] = True
    assert d.running_teacher_work() == d.root / 'teacher-r0002'
    assert d.ensure_teacher() is None and len(launched) == 2 and d.teacher_draining
    assert json.loads((d.root / 'teacher-r0002' / 'DRAIN').read_text())['frontier'].endswith('r0004/frontier.json')
    assert 'teacher_drain' in (d.root / 'drive.log').read_text()
    present['teacher'] = False
    assert d.ensure_teacher() == d.root / 'teacher-r0004' and launched[-1][:2] == ('teacher-r0004', 'r0004')
    assert not d.teacher_draining and d.running_teacher_work() == d.root / 'teacher-r0004'
    present['teacher'] = True
    assert d.ensure_teacher() is None and not (d.root / 'teacher-r0004' / 'DRAIN').exists()   # the newest: runs on
    d.a.no_teacher_campaign = True
    (d.root / 'r0005').mkdir()
    (d.root / 'r0005' / 'frontier.json').write_text('{}')
    assert d.ensure_teacher() is None and len(launched) == 3
    # the hooks: every DGX SFT poll and the standing thread check it (a failure is a note, never the caller's end)
    calls = []
    monkeypatch.setattr(d, 'ensure_teacher', lambda round_dir=None: calls.append('teacher'))
    monkeypatch.setattr(d, 'ensure_remote_collection', lambda policy: None)
    monkeypatch.setattr(d, 'sft_handback', lambda *a: calls.append('handback'))
    d.dgx_sft_between(d.root / 'r0002', d.controller.current, 'aei-r0002-sft')
    assert calls == ['teacher', 'handback']
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) > 2:
            raise StopIteration
    monkeypatch.setattr(ei.time, 'sleep', sleep)
    monkeypatch.setattr(d, 'ensure_teacher', lambda round_dir=None: (_ for _ in ()).throw(RuntimeError('ssh')))
    with pytest.raises(StopIteration):
        d.teacher_loop()
    assert sleeps == [600.0] * 3 and 'teacher_check_failed' in (d.root / 'drive.log').read_text()


def _prepared_round(tmp_path, monkeypatch, argv=()):
    ei, d = _driver(tmp_path, monkeypatch, argv=list(argv))
    (tmp_path / 'pin' / '.natlang-pin.json').write_text(json.dumps({'commit': 'b' * 40}))
    monkeypatch.setattr(d.c, 'data_repo', lambda: tmp_path / 'pin')     # the build code's pin
    rd = d.root / 'r0002'
    (rd / 'stages').mkdir(parents=True)
    (rd / 'mix').mkdir()
    (rd / 'mix' / 'chains.pkl').write_bytes(b'old chains')
    (rd / 'mix' / 'build.json').write_text('{}')
    cur = d.controller.current
    size = {'placement': 'colab', 'teacher_budget': 1000, 'max_target_tokens': 5, 'max_forward_tokens': 9}
    ei.write_round_manifest(rd, round_id='r0002', base_policy_sha=cur['sha256'], previous_policy_sha=None,
                            chains_path=rd / 'mix' / 'chains.pkl', episode_ids=['e-old'], prefer_unseen=['/x/mix'],
                            profile_changes=[{'from_profile': 'colab', 'to_profile': 'dgx', 'reason': 'r'}],
                            extra={'frontier_policy': cur['sha256'], 'size': size, 'manual_rebuild': {'by': 'hand'},
                                   'recap': {'at': 1}, 'mix_dir': str(rd / 'mix-dgx'), 'collections': ['/c']})
    for stage in ('frontier', 'data', 'build', 'prepared'):
        (rd / 'stages' / f'{stage}.done').write_text('{}')
    plans, units = [], []
    monkeypatch.setattr(d, 'prepare_plan', lambda rd_, pols, size_=None, prefer_unseen=None: plans.append(
        (rd_, [p['sha256'] for p in pols], size_, prefer_unseen)) or (['frontier'], [['data']], ['build', 'argv']))
    monkeypatch.setattr(d, 'expected_hours', lambda report: {'dgx': 1.0})

    def ledger(unit, budget, priority, command, log, **kw):
        units.append((unit, command))
        (rd / 'mix').mkdir()
        (rd / 'mix' / 'chains.pkl').write_bytes(b'new chains')
        (rd / 'mix' / 'build.json').write_text(json.dumps({'admitted_episode_keys': ['e-new', 'e2']}))
    monkeypatch.setattr(d.c, 'ledger', ledger)
    monkeypatch.setattr(d.c, 'present', lambda unit: False)
    return ei, d, rd, cur, plans, units, size


def test_a_prepared_rounds_mix_rebuilds_alone_from_its_recorded_inputs(tmp_path, monkeypatch):
    """2026-11-24: removing build.done and prepared.done to take new build code reran all of prepare, which first
    waits --snapshot-min-hours (r0002 would have stalled ~2 h). --rebuild-round (or stages/rebuild.request) reruns
    only the build, with the round's recorded size, previous policy and prefer-unseen list, keeps the old mix aside,
    and refreshes chains.sha256 and round.json (every recorded field kept, the DGX recap dropped), once per build
    commit; a failed build restores the old mix; a round whose SFT started is not rebuilt."""
    import hashlib as _h
    ei, d, rd, cur, plans, units, size = _prepared_round(tmp_path, monkeypatch, argv=['--rebuild-round', 'r0002'])
    monkeypatch.setattr(d, 'prepare', lambda *a: (_ for _ in ()).throw(AssertionError('prepare reran')))
    assert d.rebuild_requested(rd) == f'--rebuild-round under build code {"b" * 40}'
    assert d.rebuild_round(rd, d.lineage_policies(), d.rebuild_requested(rd))
    assert plans == [(rd, [cur['sha256']], size, ['/x/mix'])]
    (unit, command), = units
    assert unit.startswith('aei-r0002-rebuild-') and command == ['build', 'argv']
    m = json.loads((rd / 'round.json').read_text())
    new_sha = _h.sha256(b'new chains').hexdigest()
    assert m['chains_sha256'] == new_sha and (rd / 'chains.sha256').read_text().split()[0] == new_sha
    assert m['episode_ids'] == ['e-new', 'e2'] and m['mix_dir'] == str(rd / 'mix') and 'recap' not in m
    assert m['manual_rebuild'] == {'by': 'hand'} and m['profile_changes'][0]['to_profile'] == 'dgx'
    assert m['prefer_unseen'] == ['/x/mix'] and m['size'] == size and m['collections'] == ['/c']
    (old,) = rd.glob('mix.pre-rebuild-*')
    assert (old / 'chains.pkl').read_bytes() == b'old chains'
    assert m['rebuilds'][-1]['build_code'] == 'b' * 40 and m['rebuilds'][-1]['old_mix'] == str(old)
    assert m['rebuilds'][-1]['old_chains_sha256'] == _h.sha256(b'old chains').hexdigest()
    assert d.rebuild_requested(rd) is None                    # a restart with the same argv does not rebuild again
    (rd / 'stages' / 'rebuild.request').write_text('')          # a running driver's request
    assert d.rebuild_requested(rd) == 'stages/rebuild.request'

    def failing(unit, budget, priority, command, log, **kw):
        (rd / 'mix').mkdir()
        (rd / 'mix' / 'chains.pkl').write_bytes(b'partial')
        raise SystemExit('aei-r0002-rebuild failed')
    monkeypatch.setattr(d.c, 'ledger', failing)
    before = (rd / 'round.json').read_text()
    with pytest.raises(SystemExit, match='rebuild failed'):
        d.rebuild_round(rd, d.lineage_policies(), 'stages/rebuild.request')
    assert (rd / 'mix' / 'chains.pkl').read_bytes() == b'new chains' and (rd / 'round.json').read_text() == before
    assert (rd / 'chains.sha256').read_text().split()[0] == new_sha
    # its SFT started: refused, the request consumed, nothing moved
    (rd / 'sft' / 'learner-state').mkdir(parents=True)
    (rd / 'sft' / 'learner-state' / 'learner_manifest.json').write_text('{}')
    monkeypatch.setattr(d.c, 'ledger', lambda *a, **k: (_ for _ in ()).throw(AssertionError('rebuilt')))
    assert not d.rebuild_round(rd, d.lineage_policies(), 'stages/rebuild.request')
    assert not (rd / 'stages' / 'rebuild.request').exists() and (rd / 'round.json').read_text() == before
    assert 'rebuild_refused' in (d.root / 'drive.log').read_text()


def test_the_round_loop_rebuilds_a_requested_round_before_it_trains(tmp_path, monkeypatch):
    ei, d, rd, cur, plans, units, size = _prepared_round(tmp_path, monkeypatch,
                                                         argv=['--rebuild-round', 'r0001', '--max-rounds', '1'])
    rd = d.root / 'r0001'
    calls = []
    monkeypatch.setattr(d.c, 'pin_stage', lambda: None)
    monkeypatch.setattr(d.c, 'require_data_readers', lambda: None)
    monkeypatch.setattr(d, 'record_reader_sources', lambda: None)
    monkeypatch.setattr(d, 'held_loop', lambda: None)
    monkeypatch.setattr(d, 'teacher_loop', lambda: None)
    monkeypatch.setattr(d, 'ensure_teacher', lambda round_dir=None: calls.append('teacher'))
    monkeypatch.setattr(d, 'ensure_collection', lambda p: None)
    monkeypatch.setattr(d, 'ensure_remote_collection', lambda p: None)
    monkeypatch.setattr(d, 'prepare_ahead', lambda rd_, pols: None)
    monkeypatch.setattr(d, 'health_report', lambda rd_, pols: None)

    def prepare(rd_, pols):
        if not (rd_ / 'round.json').exists():
            (rd_ / 'mix').mkdir(parents=True, exist_ok=True)
            (rd_ / 'mix' / 'chains.pkl').write_bytes(b'old')
            ei.write_round_manifest(rd_, round_id=rd_.name, base_policy_sha=pols[-1]['sha256'], previous_policy_sha=None,
                                    chains_path=rd_ / 'mix' / 'chains.pkl', episode_ids=[], extra={'size': size})
        calls.append('prepare')
    monkeypatch.setattr(d, 'prepare', prepare)
    monkeypatch.setattr(d, 'rebuild_round', lambda rd_, pols, reason: calls.append(('rebuild', rd_.name, reason)))
    monkeypatch.setattr(d, 'train_sft', lambda rd_, p: calls.append('sft') or rd_ / 'sft' / 'factors')
    monkeypatch.setattr(d, 'publish', lambda rd_, p, f: {'checkpoint': str(rd_ / 'policy' / 'checkpoint.pt'),
                                                          'checkpoint_sha256': 'e' * 64})
    monkeypatch.setattr(d, 'rebind_remote', lambda *a: None)
    assert d.run() == 0
    assert calls.index('prepare') < calls.index(('rebuild', 'r0001', f'--rebuild-round under build code {"b" * 40}')) \
        < calls.index('sft') and 'teacher' in calls


def test_the_fallow22_fan_in_part_joins_every_round_mix_unless_opted_out(tmp_path, monkeypatch):
    ei, d = _colab_driver(tmp_path, monkeypatch)
    _, _, build = d.prepare_plan(d.root / 'r0001', [d.controller.current])
    aux = [build[i + 1] for i, a in enumerate(build) if a == '--aux']
    assert ei.FALLOW22_AUX in aux and ei.FALLOW22_AUX.endswith('share=0.01,task_tokens=2000')
    _, off = _colab_driver(tmp_path / 'off', monkeypatch, '--no-fallow22-aux')
    _, _, build = off.prepare_plan(off.root / 'r0001', [off.controller.current])
    assert not any(a.startswith('fallow22=') for a in build)


def test_a_held_measure_unit_left_by_a_restarted_driver_keeps_pop_busy_and_is_joined_not_duplicated(tmp_path,
                                                                                                    monkeypatch):
    """2026-11-24: a driver restart killed pop-held.sh but its DGX measure unit popheld-policy-8e634a0f46ce-* ran on;
    pop_busy saw no pop-held.sh process, the Pop loop served Pop for collection and the held failed at 279 episodes.
    The unit, a held_eval measure aimed at Pop and a held process on Pop each keep Pop busy (Pop not answering too),
    and a restarted held step joins the running unit instead of starting a second measurement of the same episodes."""
    import expert_iteration as ei
    import pop_rl_actor
    units = {'*popheld*': ['popheld-policy-8e634a0f46ce-1795481950']}
    monkeypatch.setattr(pop_rl_actor, 'pop_held_units', lambda pattern='*popheld*': list(units.get(pattern, [])))
    monkeypatch.setattr(pop_rl_actor, 'pop_held_running', lambda: [])
    monkeypatch.setattr(pop_rl_actor, 'pop_host_address', lambda: '100.1.2.3')
    measuring, remote = [], []
    monkeypatch.setattr(pop_rl_actor, 'held_measure_pids', lambda address: list(measuring))
    monkeypatch.setattr(pop_rl_actor, 'pop_side_held', lambda: list(remote))
    monkeypatch.setattr(pop_rl_actor, 'SERVE_LOCK', tmp_path / 'pop-serve.lock')
    assert 'popheld-policy-8e634a0f46ce' in pop_rl_actor.pop_busy()
    units.clear()
    measuring.append(7)
    assert 'held_eval measure' in pop_rl_actor.pop_busy()
    measuring.clear()
    remote.append('unreachable')
    assert 'unreachable' in pop_rl_actor.pop_busy()
    remote.clear()
    assert pop_rl_actor.pop_busy() is None
    # the DGX scan: only a held_eval measure naming Pop's address counts (a DGX-series measurement does not)
    proc = tmp_path / 'proc'
    for pid, argv in ((21, ['python', '-m', 'natlang_neuralese.rl.held_eval', 'measure', '--endpoint',
                            'http://100.1.2.3:18531=20']),
                      (22, ['python', '-m', 'natlang_neuralese.rl.held_eval', 'measure', '--endpoint',
                            'http://127.0.0.1:8470=64'])):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / 'cmdline').write_bytes(b'\0'.join(a.encode() for a in argv) + b'\0')
    monkeypatch.undo()
    assert pop_rl_actor.held_measure_pids('100.1.2.3', proc) == [21]
    # a restarted held step waits for the running unit, then resumes with one pop-held.sh
    out = tmp_path / 'held' / 'policy-8e634a0f46ce'
    out.mkdir(parents=True)
    polls = {'n': 0}

    def held_units(pattern='*popheld*'):
        polls['n'] += 1
        return ['popheld-policy-8e634a0f46ce-1795481950'] if pattern == f'popheld-{out.name}-*' and polls['n'] < 3 \
            else []
    monkeypatch.setattr(pop_rl_actor, 'pop_held_units', held_units)
    monkeypatch.setattr(pop_rl_actor, 'stop_pop_fills', lambda: None)
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    started = []

    class Done:
        pid = 1
        def poll(self):
            return 0
    monkeypatch.setattr(ei.subprocess, 'Popen', lambda argv, **kw: started.append(polls['n']) or Done())
    assert ei.pop_held_measure(out, '/c.pt', pin='/pin', samples=4, manifest='/m.json', poll=0) == 'incomplete'
    assert started == [3]                       # launched only after the old unit ended
    stopped = []
    monkeypatch.setattr(pop_rl_actor, 'pop_held_units', lambda pattern='*popheld*': ['popheld-policy-8e634a0f46ce-1'])
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: stopped.append(argv))
    assert ei.join_pop_held(out, preempt=lambda: True, poll=0) == 'preempted'
    assert stopped and stopped[0][-1] == 'popheld-policy-8e634a0f46ce-1.service'


def test_publish_round_records_the_code_whose_merge_wrote_the_bytes(tmp_path, monkeypatch):
    """A VM reproduces the publication's merge only with the same code (r0002 2026-11-24: a VM checkout before
    0462075ea wrote no FP32 master rows): published.json names the publisher's commit (``code``)."""
    import expert_iteration as ei
    from natlang_neuralese.common.code_provenance import code_provenance
    from natlang_neuralese.export import native_hf
    from natlang_neuralese.train import learner_state
    monkeypatch.setattr(learner_state, 'merge_factors_to_checkpoint',
                        lambda base, factors, out: {'checkpoint_sha256': 'p' * 64})
    monkeypatch.setattr(native_hf, 'publish_merged_policy', lambda ckpt, **kw: {'checkpoint_sha256': 'p' * 64})
    (tmp_path / 'factors').mkdir()
    ei.publish_round(SimpleNamespace(out=tmp_path / 'policy', base_checkpoint=tmp_path / 'base.pt',
                                     factors=tmp_path / 'factors', base=None, label='aei-r0001'))
    published = json.loads((tmp_path / 'policy' / 'published.json').read_text())
    assert published['code'] == code_provenance() and published['code']['git_commit']


def _lineage_driver(tmp_path, monkeypatch):
    """A driver whose lineage is v0 -> v1 -> v2 (rounds r0001 and r0002 published), with round r0003 prepared ahead
    while r0002 trained: its frontier was measured on v1 against v0."""
    ei, d = _driver(tmp_path, monkeypatch)
    v0 = d.controller.current
    lineage = [v0, *({'version': i, 'sha256': c * 64, 'checkpoint': f'/p/v{i}.pt'} for i, c in ((1, '1'), (2, '2')))]
    d.controller.state['lineage'] = lineage
    d.controller.save()
    rd = d.root / 'r0003'
    (rd / 'stages').mkdir(parents=True)
    plans, launched = [], []
    monkeypatch.setattr(d, 'prepare_plan', lambda rd_, pols, size=None, **k: plans.append(
        [p['sha256'] for p in pols]) or (['frontier'], [['data']], ['build']))
    monkeypatch.setattr(d.c, 'work_tokenizer', lambda: None)
    monkeypatch.setattr(d.c, 'present', lambda unit: False)
    monkeypatch.setattr(d, 'ensure_teacher', lambda round_dir=None: None)
    monkeypatch.setattr(d, 'expected_hours', lambda report: {})
    monkeypatch.setattr(d, 'prefer_unseen_for', lambda rd_: [])
    monkeypatch.setattr(d, 'round_size', lambda: {'placement': 'dgx', 'teacher_budget': 1})

    def ledger(unit, budget, prio, cmd, log, **kw):
        launched.append(unit.rsplit('-', 1)[1])
        if unit.endswith('-build'):
            (rd / 'mix').mkdir(parents=True, exist_ok=True)
            (rd / 'mix' / 'build.json').write_text(json.dumps({'admitted_episode_keys': ['e']}))
            (rd / 'mix' / 'chains.pkl').write_bytes(b'x')
    monkeypatch.setattr(d.c, 'ledger', ledger)
    return ei, d, rd, lineage, plans, launched


def test_a_round_prepared_ahead_continues_with_its_policy_after_a_restart(tmp_path, monkeypatch):
    """2026-11-24 r0003: prepared ahead on v1 (frontier 03:50, data done 04:56 under the old driver, which died
    before writing data.done); the 04:17 restart after r0002 published v2 began prepare again with v2 and waited
    --snapshot-min-hours for v2's collection. Now the restarted driver continues with the policies the round
    recorded (here inferred from the frontier it measured), never waits or snapshots again, recognises the finished
    data step (its journal and output: a unit launched before exit records) and runs only the build."""
    ei, d, rd, lineage, plans, launched = _lineage_driver(tmp_path, monkeypatch)
    v0, v1, v2 = lineage
    (rd / 'frontier.json').write_text(json.dumps({'policy': v1['sha256'], 'previous_policy': v0['sha256']}))
    (rd / 'stages' / 'frontier.done').write_text('{}')
    (rd / 'neuralese-graph-own').mkdir()
    (rd / 'neuralese-graph-own' / 'chains.pkl').write_bytes(b'data out')
    monkeypatch.setattr(ei, 'unit_last_run_succeeded', lambda unit: unit == 'aei-r0003-data')
    monkeypatch.setattr(ei, 'record_collection', lambda *a, **k: (_ for _ in ()).throw(AssertionError('re-snapshot')))
    monkeypatch.setattr(ei.time, 'sleep', lambda s: (_ for _ in ()).throw(AssertionError('waited for a snapshot')))
    manifest = d.prepare(rd, [v1, v2])                     # what run() passes after r0002 published v2
    assert plans == [[v0['sha256'], v1['sha256']]] and launched == ['build']
    assert manifest['base_policy_sha'] == v1['sha256'] and manifest['previous_policy_sha'] == v0['sha256']
    assert manifest['frontier_policy'] == v1['sha256']
    assert json.loads((rd / 'stages' / 'data.done').read_text())['recovered'] == 'journal'
    assert [p['sha256'] for p in json.loads((rd / 'stages' / 'prepare.json').read_text())['policies']] == \
        [v0['sha256'], v1['sha256']]
    log = (d.root / 'drive.log').read_text()
    assert 'prepare_continues_with' in log and 'step_finished_before_restart' in log


def test_a_restart_during_the_snapshot_wait_waits_on_the_recorded_policy_and_steps_record_their_exit(tmp_path,
                                                                                                      monkeypatch):
    """A round recorded its policies at its start (prepare.json): a driver restarted during the snapshot wait waits
    on that policy's collection, not the new one's. Every step's unit records its exit status for its argv, so a
    step that finished before the driver wrote its marker is not rerun; a failed one (status 1) or one recorded for
    another argv is."""
    ei, d, rd, lineage, plans, launched = _lineage_driver(tmp_path, monkeypatch)
    v0, v1, v2 = lineage
    (rd / 'stages' / 'prepare.json').write_text(json.dumps({'policies': [v0, v1], 'started_at': 1}))
    asked = []

    def collections_of(pols):
        asked.append([p['sha256'] for p in pols])
        out = []
        for p in pols:
            c = tmp_path / 'collect' / p['sha256'][:4]
            c.mkdir(parents=True, exist_ok=True)
            (c / 'runs.jsonl').write_text(json.dumps({'started': 0}) + '\n')
            out.append(c)
        return out
    monkeypatch.setattr(d, 'collections_of', collections_of)
    snapped = []
    monkeypatch.setattr(ei, 'record_collection', lambda work, closed=True: snapped.append(Path(work).name))
    monkeypatch.setattr(ei, 'unit_last_run_succeeded', lambda unit: False)
    # the frontier finished (its exit record) before the driver died; the data run failed; the build never ran
    frontier_argv, data_argv = ['frontier'], ['/bin/sh', str(rd / 'data.sh')]
    (rd / 'stages' / 'frontier.exit').write_text(json.dumps({'rc': 0, 'argv_sha256': ei.argv_digest(frontier_argv)}))
    (rd / 'stages' / 'data.exit').write_text(json.dumps({'rc': 1, 'argv_sha256': ei.argv_digest(data_argv)}))
    cmds = {}
    real = d.c.ledger

    def ledger(unit, budget, prio, cmd, log, **kw):
        cmds[unit.rsplit('-', 1)[1]] = cmd
        return real(unit, budget, prio, cmd, log, **kw)
    monkeypatch.setattr(d.c, 'ledger', ledger)
    d.prepare(rd, [v1, v2])
    assert asked[0] == [v1['sha256']] and snapped == [v0['sha256'][:4], v1['sha256'][:4]]
    assert json.loads((rd / 'stages' / 'prepare.json').read_text())['snapshot_at'] > 0
    assert launched == ['data', 'build'] and json.loads((rd / 'stages' / 'frontier.done').read_text())['recovered'] \
        == 'exit-status'
    assert cmds['data'][:2] == ['/bin/sh', '-c'] and cmds['data'][3:] == ['step', *data_argv]
    assert str(rd / 'stages' / 'data.exit') in cmds['data'][2]
    assert ei.argv_digest(frontier_argv) != ei.argv_digest(['frontier', '--other'])


def test_a_prepare_record_from_a_rolled_back_lineage_starts_over(tmp_path, monkeypatch):
    ei, d, rd, lineage, plans, launched = _lineage_driver(tmp_path, monkeypatch)
    v0, v1, v2 = lineage
    gone = {'version': 9, 'sha256': '9' * 64, 'checkpoint': '/p/v9.pt'}
    (rd / 'stages' / 'prepare.json').write_text(json.dumps({'policies': [v1, gone], 'snapshot_at': 5}))
    (rd / 'stages' / 'frontier.done').write_text('{}')
    (rd / 'frontier.json').write_text('{}')
    stopped = []
    monkeypatch.setattr(d.c, 'present', lambda unit: unit == 'aei-r0003-data' and not stopped)
    monkeypatch.setattr(d.c, 'stop_unit', stopped.append)
    assert [p['sha256'] for p in d.prepare_policies(rd, [v1, v2])] == [v1['sha256'], v2['sha256']]
    assert stopped == ['aei-r0003-data'] and not (rd / 'stages' / 'frontier.done').exists()
    assert not (rd / 'stages' / 'prepare.json').exists() and not (rd / 'frontier.json').exists()


def test_restart_arguments_are_adopted_or_refused_never_shadowed_by_the_state(tmp_path, monkeypatch):
    """2026-11-24 06:20: phase-state.json's reference_policy was sft-v7 while --reference named sft-v9; the stored
    value silently won and every held verdict was queued against sft-v7. Now a restart adopts --reference (and
    re-points an open measurement) unless a held verdict adopted the stored reference; --start that differs from the
    loop's start refuses; --held-every and --prefer-unseen are adopted; each change is in history and the log."""
    import expert_iteration as ei
    v7, v9, v12 = ('7' * 64, '/p/v7.pt'), ('9' * 64, '/p/v9.pt'), ('c' * 64, '/p/v12.pt')
    path = tmp_path / 'phase-state.json'
    ctrl = ei.PhaseController.initialize(path, start_checkpoint=v9[1], start_sha=v9[0], reference_checkpoint=v7[1],
                                         reference_sha=v7[0])
    ctrl.state['measuring'] = {'candidate': {'sha256': 'd' * 64}, 'reference': {'sha256': v7[0], 'checkpoint': v7[1]}}
    ctrl.save()
    changes = ctrl.reconcile_arguments(start_checkpoint=v9[1], start_sha=v9[0], reference_checkpoint=v9[1],
                                       reference_sha=v9[0], held_every=1, held_every_max=8, prefer_unseen=['/m1'])
    assert changes['reference'] == {'from': {'checkpoint': v7[1], 'sha256': v7[0]},
                                    'to': {'checkpoint': v9[1], 'sha256': v9[0]}}
    assert changes['measuring_reference'] == v9[0] and changes['prefer_unseen_added'] == ['/m1']
    assert changes['base_held_every'] == {'from': 2, 'to': 1}
    again = ei.PhaseController(path)
    assert again.state['reference_policy']['sha256'] == v9[0] and again.state['measuring']['reference']['sha256'] == v9[0]
    assert again.state['held_every'] == 1 and again.state['history'][-1]['action'] == 'arguments'
    assert again.reconcile_arguments(start_checkpoint=v9[1], start_sha=v9[0], reference_checkpoint=v9[1],
                                     reference_sha=v9[0], held_every=1, held_every_max=8, prefer_unseen=['/m1']) == {}
    # a held verdict adopted a later reference: the restart's (original) --reference does not undo it
    again.state['reference_policy'] = {'checkpoint': v12[1], 'sha256': v12[0]}
    again.state['history'].append({'action': 'adopt', 'sha256': v12[0]})
    kept = again.reconcile_arguments(start_checkpoint=v9[1], start_sha=v9[0], reference_checkpoint=v9[1],
                                     reference_sha=v9[0])
    assert again.state['reference_policy']['sha256'] == v12[0] and kept['reference_kept']['argument'] == v9[0]
    with pytest.raises(SystemExit, match='another start is another experiment'):
        again.reconcile_arguments(start_checkpoint=v7[1], start_sha=v7[0])
    # a state written before start_policy was recorded: the start comes from the lineage's version-0 entry
    del again.state['start_policy']
    with pytest.raises(SystemExit, match='--start names 777777777777'):
        again.reconcile_arguments(start_checkpoint=v7[1], start_sha=v7[0])
    assert again.reconcile_arguments(start_checkpoint=v9[1], start_sha=v9[0])['start_policy']['recorded']['sha256'] \
        == v9[0]


def test_a_restarted_driver_adopts_its_reference_argument(tmp_path, monkeypatch):
    ref7, ref9 = tmp_path / 'v7.pt', tmp_path / 'v9.pt'
    ref7.write_bytes(b'7')
    ref9.write_bytes(b'9')
    ei, d = _driver(tmp_path, monkeypatch, argv=['--reference', str(ref7), '--reference-sha', '7' * 64])
    assert d.controller.state['reference_policy']['sha256'] == '7' * 64
    ei, d2 = _driver(tmp_path, monkeypatch, argv=['--reference', str(ref9), '--reference-sha', '9' * 64])
    assert d2.controller.state['reference_policy'] == {'checkpoint': str(ref9), 'sha256': '9' * 64}
    assert 'arguments_reconciled' in (d2.root / 'drive.log').read_text()


def test_a_held_pair_refused_for_engine_settings_remeasures_the_reference_on_this_series(tmp_path, monkeypatch):
    """The held loop reused an old sft-v7 result from another engine series; held_eval compare refused the pair and
    the verdict failed every 5 minutes. Now a refusal sets the reference's result aside and measures it again here
    (no reuse), then compares; only a refusal after the candidate was measured again too raises."""
    import subprocess
    ei, d = _driver(tmp_path, monkeypatch)
    cand = {'sha256': 'c' * 64, 'checkpoint': '/p/c.pt'}
    ref = {'sha256': '9' * 64, 'checkpoint': '/p/v9.pt'}
    held = d.c.held
    measured = []

    def measure(name, checkpoint, reuse=True):
        measured.append((name, reuse))
        out = held / name
        out.mkdir(parents=True, exist_ok=True)
        (out / 'result.json').write_text(json.dumps({'complete': True, 'fresh': not reuse}))
        return out
    monkeypatch.setattr(d.c, 'measure_held', measure)
    monkeypatch.setattr(d.c, 'held_series_key', lambda: 'k')
    monkeypatch.setattr(d.c, 'held_measuring', lambda out: False)
    (held / d.held_name(ref)).mkdir(parents=True)
    (held / d.held_name(ref) / 'result.json').write_text(json.dumps({'complete': True, 'reused': True}))
    monkeypatch.setattr(d.c, 'held_result_ok', lambda out: (out / 'result.json').exists())
    refusal = ('ValueError: results served by different engine settings are not paired: '
               '{"kv_cache_dtype": ["fp8_e4m3", "bfloat16"]}')
    runs = []

    def run(argv, **kw):
        runs.append(argv)
        first = json.loads(Path(argv[argv.index('compare') + 1]).read_text())
        if not first.get('fresh'):
            return subprocess.CompletedProcess(argv, 1, stderr=refusal)
        Path(argv[argv.index('--out') + 1]).write_text(json.dumps({'primary': {'delta': 0.0, 'permutation_p': 1.0}}))
        return subprocess.CompletedProcess(argv, 0, stderr='')
    monkeypatch.setattr(ei.subprocess, 'run', run)
    record = d.judge_measurement({'candidate': cand, 'reference': ref})
    assert record['verdict'] in ('neutral', 'success', 'failure') and len(runs) == 2
    assert measured == [(d.held_name(cand), True), (d.held_name(ref), False)]
    assert list(held.glob(f'{d.held_name(ref)}-unpaired-*'))          # the reused result kept aside
    log = (d.root / 'drive.log').read_text()
    assert 'held_pair_refused' in log and '"remeasure": "reference"' in log
    # refused even after both were measured here: a real fault, raised (the held loop retries and notes it)
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: subprocess.CompletedProcess(argv, 1, stderr=refusal))
    with pytest.raises(ei.HeldPairRefused, match='not paired'):
        d.judge_measurement({'candidate': cand, 'reference': ref})
    # other compare failures are not refusals
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: subprocess.CompletedProcess(argv, 2, stderr='boom'))
    with pytest.raises(subprocess.CalledProcessError):
        d.c.compare(held / d.held_name(ref), held / d.held_name(cand), tmp_path / 'x.json')


def test_set_aside_held_results_are_never_reused_and_running_measurements_are_joined(tmp_path, monkeypatch):
    """2026-11-24 09:41: a restarted driver reused the reference result the previous driver had set aside as unpaired
    (held/policy-e15b20a86f1c-unpaired-1795501815), the pair was refused again, and the second set-aside moved the
    whole directory under the running Pop re-measurement (07:41), which then failed. Now set-aside results (and,
    into the series that refused them, their reuse source) are never reused; a reused result moves aside alone,
    leaving a running measurement's files; a restarted driver joins the running Pop unit instead of reusing."""
    import pop_rl_actor
    ei, d = _driver(tmp_path, monkeypatch)
    ops = tmp_path / 'ops'
    monkeypatch.setattr(ei, 'OPS', ops)
    pol = 'e' * 64
    result = lambda **k: json.dumps({'complete': True, 'policy_id': pol, 'samples_per_task': 2, **k})
    source = ops / 'ei' / 'sft-v9' / 'held' / 'candidate'
    source.mkdir(parents=True)
    (source / 'result.json').write_text(result(series={'key': 'other'}))
    other = ops / 'ei' / 'sft-v8' / 'held' / 'start'
    other.mkdir(parents=True)
    (other / 'result.json').write_text(result(series={'key': 'other2'}, v=8))
    held = d.c.held
    out = held / 'policy-eeeeeeeeeeee'
    out.mkdir(parents=True)
    (out / 'result.json').write_bytes((source / 'result.json').read_bytes())
    (out / 'reused.json').write_text(json.dumps({'from': str(source / 'result.json')}))
    for name in ('run.json', 'packet.jsonl'):
        (out / name).write_text('{}')
    (out / 'row-000').mkdir()
    monkeypatch.setattr(d.c, 'held_series_key', lambda: 'mine')
    measuring = {'on': True}
    monkeypatch.setattr(d.c, 'held_measuring', lambda o: measuring['on'])
    d.c.set_held_aside(out, 'unpaired')
    (aside,) = held.glob('policy-eeeeeeeeeeee-unpaired-*')
    assert sorted(p.name for p in aside.iterdir()) == ['aside.json', 'result.json', 'reused.json']
    assert sorted(p.name for p in out.iterdir()) == ['packet.jsonl', 'row-000', 'run.json']     # the measurement's
    rec = json.loads((aside / 'aside.json').read_text())
    assert rec['reason'] == 'unpaired' and rec['series_key'] == 'mine' and rec['reused_from'] == str(source / 'result.json')
    # neither the aside copy nor its source comes back into this series; another cycle's result still may
    allowed = ei.held_reuse_candidates(sorted(ops.glob('ei/*/held/*/result.json')), held.glob('*/aside.json'),
                                       'pop', 'mine')
    assert allowed == [other / 'result.json']
    legacy = held / 'policy-eeeeeeeeeeee-unpaired-1795501815'     # set aside before aside.json existed
    legacy.mkdir()
    (legacy / 'result.json').write_text(result(series={'key': 'x'}))
    assert legacy / 'result.json' not in ei.held_reuse_candidates([legacy / 'result.json'], [], 'pop', 'mine')
    assert ei.held_reuse_candidates([source / 'result.json'], held.glob('*/aside.json'), 'pop', 'theirs') \
        == [source / 'result.json']                                    # another series did not refuse it
    # a restarted driver: the Pop unit measuring into out runs on; pop_held joins it and never reuses around it
    polls = []

    def units(pattern):
        polls.append(pattern)
        if len(polls) < 3:
            return ['popheld-policy-eeeeeeeeeeee-1795502476']
        (out / 'result.json').write_text(result(series={'key': 'mine'}, fresh=True))
        return []
    monkeypatch.setattr(pop_rl_actor, 'pop_held_units', units)
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    monkeypatch.setattr(d.c, 'reuse_held', lambda *a: (_ for _ in ()).throw(AssertionError('reused around a unit')))
    assert d.c.pop_held('policy-eeeeeeeeeeee', '/p/e.pt') == out
    assert json.loads((out / 'result.json').read_text())['fresh'] and polls[0] == 'popheld-policy-eeeeeeeeeeee-*'
    # with nothing measuring, a reused result moves aside with its directory as before
    measuring['on'] = False
    out2 = held / 'policy-ffffffffffff'
    out2.mkdir()
    (out2 / 'result.json').write_text(result())
    d.c.set_held_aside(out2, 'pop')
    assert not out2.exists() and len(list(held.glob('policy-ffffffffffff-pop-*/aside.json'))) == 1


def test_a_failed_sft_launch_is_noted_and_retried_with_backoff_never_ending_the_driver(tmp_path, monkeypatch):
    """2026-11-24 10:31: the preflight unit (4 GB, low, --wait 600) was not admitted behind a stuck low queue,
    launch-sft.sh failed and CalledProcessError ended the driver. A failed launch is noted and retried (backoff
    doubling), the standing services run between attempts, and the preflight asks for high priority and an hour."""
    import subprocess
    ei, d = _driver(tmp_path, monkeypatch)
    rd = d.root / 'r0001'
    (rd / 'mix').mkdir(parents=True)
    (rd / 'chains.sha256').write_text('f' * 64 + '\n')
    env = d.sft_env(rd, d.controller.current, activation_gb=56, preflight_only=True)
    assert (env['PREFLIGHT_PRIORITY'], env['PREFLIGHT_WAIT']) == ('high', '3600')
    calls, sleeps, services = [], [], []
    def run(argv, **kw):
        calls.append(argv)
        if len(calls) < 3:
            raise subprocess.CalledProcessError(1, argv)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(ei.subprocess, 'run', run)
    monkeypatch.setattr(ei.time, 'sleep', sleeps.append)
    monkeypatch.setattr(d, 'standing_services', lambda policy=None: services.append(policy))
    monkeypatch.setattr(d, 'dgx_sft_running', lambda *a: False)
    d.launch_sft(rd, d.controller.current, activation_gb=56, preflight_only=False)
    assert len(calls) == 3 and sleeps == [ei.SFT_LAUNCH_RETRY_SECONDS, 2 * ei.SFT_LAUNCH_RETRY_SECONDS]
    assert len(services) == 2
    log = [json.loads(line) for line in (d.root / 'drive.log').read_text().splitlines()]
    assert [e['attempt'] for e in log if 'sft_launch_failed' in e] == [1, 2]
    assert any(e.get('sft_launched_after') == 3 for e in log)
    # a DGX learner of the round that appeared while it waited ends the retries
    calls.clear()
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: (_ for _ in ()).throw(subprocess.CalledProcessError(1, argv)))
    monkeypatch.setattr(d, 'dgx_sft_running', lambda *a: True)
    d.launch_sft(rd, d.controller.current, activation_gb=56, preflight_only=False)


def test_a_remote_actor_serving_the_newest_policy_gets_a_collector_whenever_it_comes_back(tmp_path, monkeypatch):
    """2026-11-24: the A100 actor came back at 10:00 mid-round and got no collector until a 10:20 driver restart. The
    standing check starts one whenever the phase is SFT; while a Colab SFT trains on a session (its actor yielded)
    no actor is rebound, serving ones still get collectors."""
    ei, d = _driver(tmp_path, monkeypatch, argv=['--rebind-actor', 'colab-g4@http://127.0.0.1:8570=64'])
    started, rebinds = [], []
    monkeypatch.setattr(d, 'ensure_remote_policy', lambda policy: rebinds.append(policy['sha256']))
    monkeypatch.setattr(d, 'start_remote_collector', lambda policy, target: started.append(target['suffix']))
    d.remote_collect_check()
    assert started == ['-remote0'] and rebinds == [d.controller.current['sha256']]
    d.colab_sft_hosts.add('colab-g4')
    d.remote_collect_check()
    assert started == ['-remote0'] * 2 and len(rebinds) == 1
    monkeypatch.setattr(d.controller, 'state', {**d.controller.state, 'phase': 'grpo'})
    d.remote_collect_check()
    assert len(started) == 2


def test_the_health_report_runs_beside_the_main_loop(tmp_path, monkeypatch):
    """2026-11-24: read in the loop, the report held r0005's SFT launch for 25 min; it runs in a thread, one at a
    time."""
    import threading
    ei, d = _driver(tmp_path, monkeypatch)
    release, entered = threading.Event(), []
    monkeypatch.setattr(d, 'health_report', lambda rd, policies: entered.append(rd) or release.wait(10))
    thread = d.start_health_report(d.root / 'r0001', d.lineage_policies())
    assert thread is not None and d.start_health_report(d.root / 'r0002', d.lineage_policies()) is None
    release.set()
    thread.join(5)
    assert entered == [d.root / 'r0001']


def test_selection_reads_the_previous_rates_and_the_lineage_collections_never_every_earlier_one(tmp_path, monkeypatch):
    """Earlier collections move to the HDD (ei_round_tier); a selection globbing every collection's groups read them
    all again. The newest earlier tasks.rates.json stands for them; only the lineage's collections are read."""
    ei, d = _driver(tmp_path, monkeypatch)
    cur = d.controller.current
    first = d.select_command(d.collect_dir(cur) / 'tasks.json')
    assert '--rates-json' not in first and str(d.root / 'collect' / '*' / 'groups' / 'g*') in first
    old = d.root / 'collect' / 'v0000-oldoldoldold'
    old.mkdir(parents=True)
    (old / 'tasks.rates.json').write_text(json.dumps({'t:1': {'successes': 1, 'scored': 2}}))
    (d.collect_dir(cur)).mkdir(parents=True, exist_ok=True)
    (d.collect_dir(cur) / 'runs.jsonl').write_text('{}\n')
    argv = [str(x) for x in d.select_command(d.collect_dir(cur) / 'tasks.json')]
    assert argv[argv.index('--rates-json') + 1] == str(old / 'tasks.rates.json')
    globs = [argv[i + 1] for i, x in enumerate(argv) if x == '--rates-groups']
    assert globs == [str(d.collect_dir(cur) / 'groups' / 'g*'), str(d.root / '*' / 'grpo' / 'groups' / 'g*')]
