"""In-place rebind of the SGLang actor front (lfm3opti.md §3, §6 item 4): every route's attestation is measured, the
native route reloads with full name coverage, exact native-state splicing keys on the lineage, native requests drain
at a call boundary before a reload, a lazy native load never serves an older version than the newest rebind, and
native-engine spans are stored by the native path. Plus checkpoint learner_step/source_step provenance.

``fake_front``/``FakeSglang`` give a front whose measurements come from the files and a fake SGLang that can disagree
with what it was asked to load; test_lfm3_lora_core.py drives grpo.WeightSync against them.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/neuralese_data'))

from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402
from natlang_neuralese.serve.sglang_actor import (  # noqa: E402
    SglangEngine, attest_all_routes, export_tensors_identity, native_weights_identity)

LINEAGE = {'checkpoint_sha256': 'a' * 64, 'embedding_weight_sha256': '0' * 64, 'architecture': 'lfm3',
           'tokenizer_fingerprint': 'f' * 64}


class FakeSglang:
    """SGLang's control endpoints as the front and WeightSync use them. ``stale``: report success but keep serving
    the previous directory (what an echoing attestation cannot see). ``shift``: add to every sentinel logprob."""

    def __init__(self, start_dir, logprobs=None):
        self.model_path, self.version = str(start_dir), '0'
        self.stale, self.shift, self.calls = False, 0., []
        self.logprobs = logprobs        # (model_path, ids) -> logprobs of ids[1:], for /generate's sentinel

    def post(self, path, body):
        self.calls.append((path, body))
        if path == 'update_weights_from_disk' and not self.stale:
            self.model_path, self.version = body['model_path'], str(body['weight_version'])
        return json.dumps({'success': True})

    def get(self, path):
        assert path == '/model_info'
        return {'model_path': self.model_path, 'weight_version': self.version}

    def generate(self, body):
        ids = body['input_ids']
        values = [v + self.shift for v in self.logprobs(self.model_path, ids)]
        return {'meta_info': {'input_token_logprobs': [[None, ids[0], None]] + [[v, t, None] for v, t in zip(values, ids[1:])]}}


def fake_native(hf, marker_ids=(), override_ids=()):
    """The parts of a native SharedStateModel the reload and its identity touch, around HF module ``hf``."""
    embedding = hf.model.embed_tokens.weight
    rows = lambda ids: embedding[torch.tensor(list(ids))].detach().clone() if ids else None   # noqa: E731
    return SimpleNamespace(port=SimpleNamespace(hf=hf, embedding_weight=embedding), tied=True,
                           marker_ids=tuple(marker_ids), input_rows=rows(marker_ids),
                           token_override_ids=tuple(override_ids), extra_input_rows=rows(override_ids),
                           eval=lambda: None, parameters=hf.parameters)


def fake_front(native, sglang, start_export=None):
    front = object.__new__(SglangEngine)
    front._sglang_lock = threading.Lock()
    front.native_execution_lock = threading.RLock()
    front.__dict__['_native_lock'] = threading.RLock()
    if native is not None:
        front.__dict__['_native_model'] = native
        front.__dict__['_native_backbone'] = native.port
    front.checkpoint_sha256 = LINEAGE['checkpoint_sha256']
    front.embedding_weight_sha256 = LINEAGE['embedding_weight_sha256']
    front.native_identity = dict(LINEAGE)
    front.sglang_url = 'http://sglang'
    front.sglang_export = {'weights_sha256': sha256_file_hex(Path(start_export) / 'model.safetensors')
                           if start_export else None}
    front.sglang_actor_export = {'kind': 'exact', 'format': 'sglang-hf', 'sha256': front.sglang_export['weights_sha256']}
    front.rl_policy_binding = {'checkpoint_sha256': LINEAGE['checkpoint_sha256'], 'serving_engine': {},
                               'actor_export': {}}
    front._sglang_get = sglang.get
    front._sglang_generate = sglang.generate
    return front


def _tiny_hf():
    torch.manual_seed(0)
    hf = torch.nn.Module()
    hf.model = torch.nn.Module()
    hf.model.embed_tokens = torch.nn.Embedding(8, 4).to(torch.bfloat16)
    hf.model.proj = torch.nn.Linear(4, 4, bias=False).to(torch.bfloat16)
    for p in hf.parameters():
        p.requires_grad_(False)
    return hf


def _export(directory, embed, proj, **extra):
    directory.mkdir(parents=True, exist_ok=True)
    save_file({'model.embed_tokens.weight': embed, 'model.proj.weight': proj, **extra},
              str(directory / 'model.safetensors'))
    return directory


def _expected(directory, version, policy, markers=(6, 7), overrides=(5,)):
    path = directory / 'model.safetensors'
    with safe_open(str(path), 'pt') as handle:
        tensors = {name: handle.get_tensor(name) for name in handle.keys()}
    return {'policy_id': policy, 'version': version, 'model_path': str(directory), 'weights_sha256': sha256_file_hex(path),
            'tensor_identity': export_tensors_identity(tensors, markers, overrides)}


def _front(tmp_path):
    start = _export(tmp_path / 'start', torch.zeros(8, 4, dtype=torch.bfloat16), torch.zeros(4, 4, dtype=torch.bfloat16))
    sglang = FakeSglang(start)
    native = fake_native(_tiny_hf(), (6, 7), (5,))
    return fake_front(native, sglang, start), native, sglang


def _rebind(front, sglang, directory, version, policy, **claims):
    sglang.post('update_weights_from_disk', {'model_path': str(directory), 'weight_version': version})
    return front.rebind_in_place(policy, version=version, export_dir=str(directory), **claims)


def test_rebind_reloads_native_and_every_route_reports_what_it_measured(tmp_path):
    front, native, sglang = _front(tmp_path)
    embed = torch.full((8, 4), 2.5, dtype=torch.bfloat16)
    embed[6], embed[5] = 7., -3.
    export = _export(tmp_path / 'v4', embed, torch.full((4, 4), -1.25, dtype=torch.bfloat16))
    result = _rebind(front, sglang, export, 4, 'c' * 64)
    assert result['native_engine'] == 'reloaded'
    assert torch.equal(native.port.hf.model.embed_tokens.weight, embed)
    assert torch.equal(native.input_rows[0], embed[6]) and torch.equal(native.extra_input_rows[0], embed[5])
    expected = _expected(export, 4, 'c' * 64)
    ok, mismatches = attest_all_routes({'routes': result['routes']}, expected)
    assert ok, mismatches
    assert result['routes']['native_fallback']['tensor_identity'] == native_weights_identity(native)
    # the lineage (what exact native state keys on) is the start checkpoint's, unchanged by the rebind
    assert front.checkpoint_sha256 == LINEAGE['checkpoint_sha256'] and front.native_identity == LINEAGE
    assert front.embedding_weight_sha256 == LINEAGE['embedding_weight_sha256']
    assert result['routes']['lineage'] == LINEAGE


def test_attestation_is_measured_not_an_echo_of_the_rebind_claims(tmp_path):
    front, native, sglang = _front(tmp_path)
    v1 = _export(tmp_path / 'v1', torch.ones(8, 4, dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    expected = _expected(v1, 1, 'c' * 64)
    # a claim that disagrees with the file: the routes report the file the front read, not the claim
    result = _rebind(front, sglang, v1, 1, 'c' * 64, weights_sha256='9' * 64)
    assert result['routes']['generation']['weights_sha256'] == expected['weights_sha256']
    assert attest_all_routes({'routes': result['routes']}, expected)[0]
    # SGLang says it loaded but keeps serving v1: the measured route shows v1's directory and digest
    v2 = _export(tmp_path / 'v2', torch.full((8, 4), 3., dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    sglang.stale = True
    result = _rebind(front, sglang, v2, 2, 'd' * 64)
    ok, mismatches = attest_all_routes({'routes': result['routes']}, _expected(v2, 2, 'd' * 64))
    assert not ok and {'model_path', 'version', 'weights_sha256'} <= set(mismatches['generation'])
    assert 'native_fallback' not in mismatches          # the native route did reload v2: only SGLang is stale


def test_native_reload_refuses_unmatched_names(tmp_path):
    front, native, sglang = _front(tmp_path)
    before = native.port.hf.model.proj.weight.clone()
    missing = tmp_path / 'missing'
    missing.mkdir()
    save_file({'model.embed_tokens.weight': torch.ones(8, 4, dtype=torch.bfloat16)}, str(missing / 'model.safetensors'))
    with pytest.raises(ValueError, match='lacks native parameters'):
        _rebind(front, sglang, missing, 1, 'c' * 64)
    extra = _export(tmp_path / 'extra', torch.ones(8, 4, dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16),
                    **{'model.unknown.weight': torch.ones(2, dtype=torch.bfloat16)})
    with pytest.raises(ValueError, match='unknown tensors'):
        _rebind(front, sglang, extra, 2, 'c' * 64)
    assert torch.equal(native.port.hf.model.proj.weight, before)
    # the producer table and a tied head are not native parameters, and are not unmatched names
    fine = _export(tmp_path / 'fine', torch.ones(8, 4, dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16),
                   **{'model.natlang_producer_embed': torch.ones(8, 4, dtype=torch.bfloat16),
                      'lm_head.weight': torch.ones(8, 4, dtype=torch.bfloat16)})
    assert _rebind(front, sglang, fine, 3, 'c' * 64)['native_engine'] == 'reloaded'


def test_rebind_opt_out_leaves_the_native_route_on_the_start_and_attestation_says_so(tmp_path, monkeypatch):
    front, native, sglang = _front(tmp_path)
    before = native.port.hf.model.proj.weight.clone()
    v1 = _export(tmp_path / 'v1', torch.zeros(8, 4, dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    monkeypatch.setenv('NATLANG_REBIND_NATIVE_RELOAD', '0')
    result = _rebind(front, sglang, v1, 1, 'e' * 64)
    assert torch.equal(native.port.hf.model.proj.weight, before)
    ok, mismatches = attest_all_routes({'routes': result['routes']}, _expected(v1, 1, 'e' * 64))
    assert not ok and 'native_fallback' in mismatches


def test_a_lazy_native_load_applies_the_newest_rebind_under_one_lock(tmp_path, monkeypatch):
    """A rebind landing while the lazy load runs waits for it and then reloads the resident model: the native route
    never attests v2 while holding v1's weights."""
    from natlang_neuralese.model import native_model
    front, native, sglang = _front(tmp_path)
    del front.__dict__['_native_model'], front.__dict__['_native_backbone']
    front.__dict__['_native_loader'] = {'checkpoint': 'start.pt'}
    fresh = fake_native(_tiny_hf(), (6, 7), (5,))
    monkeypatch.setattr(native_model, 'load_native_checkpoint', lambda *a, **k: (fresh, None, None, {}, {}))
    monkeypatch.setattr(native_model, 'native_identity', lambda *a, **k: dict(LINEAGE))
    v1 = _export(tmp_path / 'v1', torch.full((8, 4), 1., dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    v2 = _export(tmp_path / 'v2', torch.full((8, 4), 2., dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    _rebind(front, sglang, v1, 1, 'c' * 64)          # not resident: pending
    assert front.route_attestation()['native_fallback']['resident'] is False
    loading, release = threading.Event(), threading.Event()
    apply = SglangEngine._apply_export_weights_to_native

    def slow_apply(self, model, path):
        loading.set()
        release.wait(10)
        return apply(self, model, path)
    monkeypatch.setattr(SglangEngine, '_apply_export_weights_to_native', slow_apply)
    loader = threading.Thread(target=front._ensure_native)
    loader.start()
    assert loading.wait(10)
    rebind = threading.Thread(target=_rebind, args=(front, sglang, v2, 2, 'd' * 64))
    rebind.start()
    time.sleep(0.3)
    assert rebind.is_alive()                          # waits for the load (one native lock)
    release.set()
    loader.join(10)
    rebind.join(10)
    served = front.route_attestation()['native_fallback']
    assert served['resident'] and served['version'] == 2
    assert torch.equal(fresh.port.hf.model.embed_tokens.weight, torch.full((8, 4), 2., dtype=torch.bfloat16))
    assert served['tensor_identity'] == native_weights_identity(fresh)


def test_a_lazy_load_refuses_weights_other_than_the_claimed_identity(tmp_path, monkeypatch):
    from natlang_neuralese.model import native_model
    front, _, sglang = _front(tmp_path)
    del front.__dict__['_native_model'], front.__dict__['_native_backbone']
    front.__dict__['_native_loader'] = {'checkpoint': 'start.pt'}
    monkeypatch.setattr(native_model, 'load_native_checkpoint',
                        lambda *a, **k: (fake_native(_tiny_hf(), (6, 7), (5,)), None, None, {}, {}))
    monkeypatch.setattr(native_model, 'native_identity', lambda *a, **k: dict(LINEAGE))
    v1 = _export(tmp_path / 'v1', torch.full((8, 4), 1., dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    _rebind(front, sglang, v1, 1, 'c' * 64, tensor_identity='7' * 64)
    with pytest.raises(RuntimeError, match='measured tensor identity'):
        front._ensure_native()


def test_native_calls_drain_at_a_call_boundary_before_a_reload(tmp_path):
    front, native, sglang = _front(tmp_path)
    v1 = _export(tmp_path / 'v1', torch.full((8, 4), 1., dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    running, finish, order = threading.Event(), threading.Event(), []

    def native_call():
        running.set()
        finish.wait(10)
        order.append(('call ends', float(native.port.hf.model.embed_tokens.weight[0, 0])))
    call = threading.Thread(target=front._native_gated, args=(native_call,))
    call.start()
    assert running.wait(10)
    rebind = threading.Thread(target=lambda: (_rebind(front, sglang, v1, 1, 'c' * 64), order.append(('reloaded', None))))
    rebind.start()
    time.sleep(0.3)
    late = threading.Thread(target=front._native_gated, args=(lambda: order.append(('late call', float(
        native.port.hf.model.embed_tokens.weight[0, 0]))),))
    late.start()
    time.sleep(0.3)
    assert rebind.is_alive() and order == []         # the reload waits for the running call; the new call waits too
    finish.set()
    for thread in (call, rebind, late):
        thread.join(10)
    assert [o[0] for o in order] == ['call ends', 'reloaded', 'late call']
    assert order[0][1] != 1.0 and order[2][1] == 1.0   # the running call ran on the old weights, the late one on new


def test_native_engine_spans_are_stored_by_the_native_path(tmp_path, monkeypatch):
    """Native-routed sequences pass ``native_step`` captures; the SGLang override used to index them as a capture
    request (``capture['ids']``) and would have re-run them on a (possibly paused) SGLang."""
    from natlang_neuralese.serve.engine import Engine
    front, _, _ = _front(tmp_path)
    stored = []
    monkeypatch.setattr(Engine, '_native_store_capture',
                        lambda self, ids, chunks, *, text, owner=None: stored.append(chunks) or ('block', SimpleNamespace(id='b1')))
    monkeypatch.setattr(SglangEngine, '_sglang_capture', lambda *a, **k: pytest.fail('a native span went to SGLang'))
    chunk = {'token_ids': torch.tensor([3, 4]), 'embeddings': torch.zeros(2, 4)}
    block, record = front._native_store_capture([3, 4], [chunk], text='x')
    assert stored == [[chunk]] and record.id == 'b1'
    assert front.producer_policy('b1')['engine'] == 'native'


def test_a_block_from_an_earlier_policy_version_splices_exactly_and_keeps_its_producer(tmp_path):
    """Exact splicing keys on the lineage: a state an earlier version produced in the same run is the state the
    consumer reads (no re-encoding from text); the producing version is provenance."""
    front, _, sglang = _front(tmp_path)
    front.model_config = {'tokenizer_fingerprint': LINEAGE['tokenizer_fingerprint']}
    front.sglang_export = {**front.sglang_export, 'architecture': 'lfm3', 'layer_kinds': ['attention']}
    v1 = _export(tmp_path / 'v1', torch.full((8, 4), 1., dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    v2 = _export(tmp_path / 'v2', torch.full((8, 4), 2., dtype=torch.bfloat16), torch.ones(4, 4, dtype=torch.bfloat16))
    _rebind(front, sglang, v1, 1, 'c' * 64)
    produced = dict(LINEAGE)        # what a capture under v1 stamps (execution_native_identity, embedding identity)
    front._note_producer('blk', native=False)

    def require_native(**identity):
        for name, value in produced.items():
            if name in identity and identity[name] != value:
                raise ValueError(name)
    block = SimpleNamespace(require_native=require_native)
    _rebind(front, sglang, v2, 2, 'd' * 64)
    assert front._sglang_exact(block)
    assert front.producer_policy('blk') == {'policy_sha256': 'c' * 64, 'version': 1, 'engine': 'sglang'}


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
                                 learner_step=42, extra={'effective_policy_sha256': '3' * 64})
    grpo_written = torch.load(grpo_path, map_location='cpu', weights_only=False)
    assert grpo_written['learner_step'] == 42
    assert grpo_written['source_step'] == 120
    assert grpo_written['inference_source']['learner_step'] == 42
    assert grpo_written['inference_source']['source_step'] == 120
    assert grpo_written['effective_policy_sha256'] == '3' * 64
