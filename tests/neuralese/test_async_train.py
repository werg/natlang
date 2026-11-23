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
    (tmp_path / 'ops' / 'grpo-v4').mkdir(parents=True)
    pin = tmp_path / 'pin'
    (pin / 'scripts' / 'neuralese_data').mkdir(parents=True)
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
    monkeypatch.setattr(d.c, 'wait', finished)
    monkeypatch.setattr(d, 'stop_collections', lambda **kw: None)
    assert d.train_sft(rd, d.controller.current) == rd / 'sft' / 'factors'
    assert not any('run-sft' in argv for argv in launched)
    assert launched and launched[-1][:2] == ['/bin/sh', str(ei.OPS / 'grpo-v4' / 'launch-sft.sh')]
    changes = json.loads((rd / 'round.json').read_text())['profile_changes']
    assert changes[-1]['to_profile'] == 'dgx' and 'slower than the DGX' in changes[-1]['reason']


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
    monkeypatch.setattr(d, 'prepare_plan', lambda rd, policies: (['frontier'], [['data']], ['build']))
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
    monkeypatch.setattr(d, 'prepare_plan', lambda rd, pols: (['frontier'], [['data']], ['build']))
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
    monkeypatch.setattr(d, 'prepare_plan', lambda rd, pols: (['frontier'], [['new-data']], ['build']))
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
    monkeypatch.setattr(d, 'prepare_plan', lambda rd, pols: (_ for _ in ()).throw(StopIteration('snapshot taken')))
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
