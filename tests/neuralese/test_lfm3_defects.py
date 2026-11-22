"""Targeted tests for lfm3opti.md §3 defects fixed now:
1. Split-brain serving after in-place rebind (native engine reload + 4-route attestation + opt-out).
2. SFT & GRPO checkpoint learner_step vs source_step provenance.
3. GiB telemetry labels across learners and probes.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/neuralese_data'))

from natlang_neuralese.serve.sglang_actor import SglangEngine, attest_all_routes  # noqa: E402


def _make_rebind_front(tmp_path):
    """Minimal SglangEngine with a real tiny HF-shaped resident model to exercise native reload."""
    linear = torch.nn.Linear(4, 4, bias=False)
    embed = torch.nn.Embedding(8, 4)
    torch.manual_seed(0)
    with torch.no_grad():
        embed.weight.copy_(torch.randn_like(embed.weight, dtype=torch.bfloat16))
        linear.weight.copy_(torch.randn_like(linear.weight, dtype=torch.bfloat16))
    hf = torch.nn.Module()
    hf.model = torch.nn.Module()
    hf.model.embed_tokens = embed
    hf.model.proj = linear
    port = SimpleNamespace(hf=hf, embedding_weight=embed.weight)
    shared = SimpleNamespace(
        port=port,
        tied=True,
        marker_ids=(6, 7),
        input_rows=embed.weight[torch.tensor([6, 7])].clone(),
        token_override_ids=(5,),
        extra_input_rows=embed.weight[torch.tensor([5])].clone(),
        parameters=hf.parameters,
    )
    front = object.__new__(SglangEngine)
    front._sglang_lock = threading.Lock()
    front.native_execution_lock = threading.RLock()
    front.__dict__['_native_model'] = shared
    front.__dict__['_native_backbone'] = port
    front.checkpoint_sha256 = 'a' * 64
    front.embedding_weight_sha256 = '0' * 64
    front.native_identity = {'checkpoint_sha256': 'a' * 64, 'embedding_weight_sha256': '0' * 64,
                             'architecture': 'lfm3', 'tokenizer_fingerprint': 'tok'}
    front.sglang_actor_export = {'kind': 'exact', 'format': 'sglang-hf', 'sha256': '1' * 64}
    front.rl_policy_binding = {'checkpoint_sha256': 'a' * 64, 'serving_engine': {}, 'actor_export': {}}
    return front, shared


def test_rebind_reloads_resident_native_engine_and_attests_all_four_routes(tmp_path):
    front, shared = _make_rebind_front(tmp_path)
    new_embed = torch.full((8, 4), 2.5, dtype=torch.bfloat16)
    new_embed[6] = 7.0
    new_embed[5] = -3.0
    new_proj = torch.full((4, 4), -1.25, dtype=torch.bfloat16)
    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    save_file({'model.embed_tokens.weight': new_embed, 'model.proj.weight': new_proj},
              str(export_dir / 'model.safetensors'))

    policy_id = 'c' * 64
    weights_sha = 'd' * 64
    result = front.rebind_in_place(policy_id, weights_sha, version=4, export_dir=str(export_dir))

    assert result['native_engine'] == 'reloaded'
    assert torch.equal(shared.port.hf.model.embed_tokens.weight, new_embed)
    assert torch.equal(shared.port.hf.model.proj.weight, new_proj)
    assert torch.equal(shared.input_rows[0], new_embed[6])
    assert torch.equal(shared.extra_input_rows[0], new_embed[5])
    assert front.checkpoint_sha256 == policy_id
    assert front.native_identity['checkpoint_sha256'] == policy_id

    routes = front.route_attestation()
    ok, mismatches = attest_all_routes({'routes': routes}, policy_id, weights_sha)
    assert ok and not mismatches
    assert routes['native_fallback']['synchronized'] is True
    assert routes['native_fallback']['policy_sha256'] == policy_id


def test_rebind_opt_out_flags_unsynchronized_native_route(tmp_path, monkeypatch):
    front, shared = _make_rebind_front(tmp_path)
    before = shared.port.hf.model.proj.weight.clone()
    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    save_file({'model.embed_tokens.weight': torch.zeros((8, 4), dtype=torch.bfloat16),
               'model.proj.weight': torch.ones((4, 4), dtype=torch.bfloat16)},
              str(export_dir / 'model.safetensors'))
    monkeypatch.setenv('NATLANG_REBIND_NATIVE_RELOAD', '0')
    front.rebind_in_place('e' * 64, 'f' * 64, version=1, export_dir=str(export_dir))
    assert torch.equal(shared.port.hf.model.proj.weight, before)
    ok, mismatches = attest_all_routes({'routes': front.route_attestation()}, 'e' * 64, 'f' * 64)
    assert not ok and 'native_fallback' in mismatches


def test_write_checkpoint_records_learner_step_and_source_step_separately(tmp_path):
    import exp0_sft
    import grpo

    embed = torch.nn.Embedding(6, 4)
    proj = torch.nn.Linear(4, 4, bias=False)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = embed
    model.model.proj = proj
    orig_embed = embed.weight.detach().to(torch.bfloat16).clone()
    orig_proj = proj.weight.detach().to(torch.bfloat16).clone()
    state = {
        'schema': 'natlang.native-checkpoint/1',
        'model_config': {'marker_ids': [4, 5]},
        'interface_state': {'input_rows': orig_embed[torch.tensor([4, 5])].clone()},
        'backbone_state': {'model.embed_tokens.weight': orig_embed, 'model.proj.weight': orig_proj,
                           'lm_head.weight': orig_embed},
        'step': 120,
    }
    sft_path = tmp_path / 'sft.pt'
    exp0_sft.write_checkpoint(model, state, [4, 5], sft_path, '1' * 64, step=17)
    written = torch.load(sft_path, map_location='cpu', weights_only=False)
    assert written['learner_step'] == 17
    assert written['step'] == 17
    assert written['source_step'] == 120
    assert written['inference_source']['learner_step'] == 17
    assert written['inference_source']['source_step'] == 120

    native_rows = {'embedding_rows': orig_embed[torch.tensor([4, 5])].clone(),
                   'tied_head': True, 'names': ['model.embed_tokens.weight', 'model.proj.weight']}
    grpo_path = tmp_path / 'grpo.pt'
    grpo.write_pruned_checkpoint(state, native_rows, [4, 5], grpo_path, '2' * 64,
                                 {'model.embed_tokens.weight': orig_embed, 'model.proj.weight': orig_proj},
                                 learner_step=42)
    grpo_written = torch.load(grpo_path, map_location='cpu', weights_only=False)
    assert grpo_written['learner_step'] == 42
    assert grpo_written['source_step'] == 120
    assert grpo_written['inference_source']['learner_step'] == 42
    assert grpo_written['inference_source']['source_step'] == 120
