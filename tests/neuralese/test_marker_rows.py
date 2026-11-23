"""The Neuralese role-marker rows in the `lora+rows` learner (DECISIONS 2026-11-23 "marker rows"):
- rows still equal to the base's reserved row start from text (distinct, ordinary-scaled), trained rows are kept;
- the rows train in their own AdamW group at `row_lr_scale` x the LR, and a state saved with one group resumes;
- merged checkpoints carry the FP32 master rows beside their BF16 interface rows, and the next learner starts from them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'tests/neuralese')]

from natlang_neuralese.train.learner_state import export_merged_checkpoint, regroup_optimizer_state  # noqa: E402
from natlang_neuralese.train.lfm3_learner import (  # noqa: E402
    MARKER_PHRASES, AdapterSpec, Lfm3Learner, load_base_model, marker_rows_degenerate, master_rows, text_marker_rows,
)
from natlang_neuralese.model.shared_state import ROLE_MARKERS  # noqa: E402
from natlang_neuralese.tune.targets import LFM3_LORA_FAMILIES  # noqa: E402
from test_lfm3_port import tiny_lfm3_hf  # noqa: E402

MARKERS = (60, 61, 62, 63)


class PhraseTokenizer:
    """Each marker phrase as three distinct ordinary ids below 56."""
    def __init__(self):
        self.ids = {phrase: [3 + 11 * k, 7 + 11 * k, 9 + 11 * k] for k, phrase in enumerate(MARKER_PHRASES.values())}

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': self.ids[text]}


def _model(degenerate=True, seed=0):
    torch.manual_seed(seed)
    model = tiny_lfm3_hf(window=4).float()
    if degenerate:
        with torch.no_grad():
            reserved = model.model.embed_tokens.weight[MARKERS[0]].clone() * 0.4
            for token in MARKERS:   # the base's reserved row, ~0.01% apart (sft-v9: cosine >= 0.99996)
                model.model.embed_tokens.weight[token] = reserved + 1e-5 * torch.randn_like(reserved)
    return model


def _learner(model, **init):
    return Lfm3Learner.from_model(model, AdapterSpec(target_families=LFM3_LORA_FAMILIES, rank=4, alpha=8.0),
                                  special_ids=MARKERS, marker_ids=MARKERS, compile_layers=False, fp8_base=False,
                                  tokenizer=PhraseTokenizer(), marker_names=ROLE_MARKERS, **init)


def test_text_rows_are_distinct_centred_and_ordinary_scaled():
    embedding = _model().model.embed_tokens.weight.detach()
    rows = text_marker_rows(embedding, PhraseTokenizer(), ROLE_MARKERS, ordinary_stop=60)
    ordinary = embedding[:60].float()
    mean = ordinary.mean(0)
    spread = (ordinary - mean).norm(dim=1).mean()
    torch.testing.assert_close((rows - mean).norm(dim=1), spread.expand(4), rtol=1e-5, atol=1e-6)
    assert not marker_rows_degenerate(rows)
    tokenizer = PhraseTokenizer()
    for row, phrase in zip(rows, MARKER_PHRASES.values()):   # the row points where its phrase's tokens point
        direction = embedding[tokenizer.ids[phrase]].float().mean(0) - mean
        assert torch.nn.functional.cosine_similarity(row - mean, direction, dim=0) > 0.9999


def test_degenerate_marker_rows_start_from_text_and_the_frozen_base_keeps_the_served_rows():
    model = _model()
    before = model.model.embed_tokens.weight[list(MARKERS)].detach().clone()
    assert marker_rows_degenerate(before)
    learner = _learner(model)
    expected = text_marker_rows(model.model.embed_tokens.weight.detach(), PhraseTokenizer(), ROLE_MARKERS, ordinary_stop=60)
    assert torch.equal(learner.trainable_rows.rows.detach(), expected)
    assert learner.marker_init['init'] == 'text-v1'
    # the frozen embedding (and so the reference view of the start rows) is the policy that was served
    assert torch.equal(learner.model.model.embed_tokens.weight[list(MARKERS)], before)
    # opt-out, and rows that already differ, are kept as they are
    kept = _learner(_model(), marker_init='none')
    assert torch.equal(kept.trainable_rows.rows.detach(), before) and kept.marker_init is None
    trained = _model(degenerate=False)
    own = trained.model.embed_tokens.weight[list(MARKERS)].detach().clone()
    assert torch.equal(_learner(trained).trainable_rows.rows.detach(), own)
    with pytest.raises(ValueError, match='tokenizer'):
        Lfm3Learner.from_model(_model(), AdapterSpec(rank=4, alpha=8.0), special_ids=MARKERS, marker_ids=MARKERS,
                               compile_layers=False, fp8_base=False)


def test_rows_train_in_their_own_group_and_a_one_group_state_resumes():
    learner = _learner(_model())
    optimizer = learner.build_optimizer(lr=1e-3, row_lr_scale=10.0)
    factors, rows = optimizer.param_groups
    assert rows['params'] == [learner.trainable_rows.rows] and rows['lr'] == pytest.approx(1e-2)
    assert factors['lr_scale'] == 1.0 and rows['lr_scale'] == 10.0
    assert len(factors['params']) + 1 == len(learner.trainable_parameters())
    # one AdamW step moves a row coordinate by ~lr_scale x lr (Adam's first step is lr * sign(grad))
    for parameter in learner.trainable_parameters():
        parameter.grad = torch.ones_like(parameter)
    start = learner.trainable_rows.rows.detach().clone()
    optimizer.step()
    torch.testing.assert_close((learner.trainable_rows.rows.detach() - start).abs().max(), torch.tensor(1e-2), rtol=1e-3, atol=0)
    # a state saved when everything was one group (before 2026-11-23) loads into the two groups, moments in order
    saved = optimizer.state_dict()
    one = {'state': saved['state'], 'param_groups': [{**saved['param_groups'][0], 'params':
           saved['param_groups'][0]['params'] + saved['param_groups'][1]['params']}]}
    fresh = learner.build_optimizer(lr=1e-3)
    fresh.load_state_dict(regroup_optimizer_state(one, fresh))
    assert [len(g['params']) for g in fresh.param_groups] == [len(factors['params']), 1]
    assert fresh.param_groups[1]['lr_scale'] == 10.0
    assert torch.equal(fresh.state[learner.trainable_rows.rows]['exp_avg'], optimizer.state[learner.trainable_rows.rows]['exp_avg'])


def _checkpoint(model, path):
    """A native checkpoint of ``model`` in the real tied layout, BF16, with the marker rows as interface rows."""
    backbone = {k: v.detach().to(torch.bfloat16).clone() for k, v in model.state_dict().items()}
    backbone['lm_head.weight'] = backbone['model.embed_tokens.weight']
    state = {'schema': 'natlang.native-checkpoint/1',
             'model_config': {'marker_ids': list(MARKERS), 'sparse_row_dtype': 'torch.bfloat16', 'dtype': 'torch.bfloat16'},
             'interface_state': {'marker_ids': MARKERS, 'marker_names': ROLE_MARKERS, 'sparse_row_dtype': 'torch.bfloat16',
                                 'input_rows': backbone['model.embed_tokens.weight'][list(MARKERS)].clone(),
                                 'output_rows': None},
             'backbone_state': backbone, 'step': 3}
    torch.save(state, path)
    return state


def test_merged_checkpoints_carry_fp32_master_rows_and_the_next_learner_starts_from_them(tmp_path, monkeypatch):
    learner = _learner(_model())
    learner.state = _checkpoint(learner.model, tmp_path / 'source.pt')
    with torch.no_grad():
        learner.trainable_rows.rows.add_(1e-4 * torch.randn_like(learner.trainable_rows.rows))  # below BF16 resolution
    info = export_merged_checkpoint(learner, learner.state, tmp_path / 'merged.pt', source_sha256='a' * 64, learner_step=1)
    merged = torch.load(info['path'], map_location='cpu', weights_only=False)
    rows = learner.trainable_rows.rows.detach()
    assert merged['interface_state']['input_rows'].dtype == torch.bfloat16      # what every engine loads
    assert torch.equal(merged['interface_state']['input_rows'], rows.to(torch.bfloat16))
    assert torch.equal(merged['trainable_rows_fp32']['rows'], rows)              # the master copy, exact
    assert merged['trainable_rows_fp32']['token_ids'].tolist() == list(MARKERS)
    # the next learner starts its rows from the master copy, not from the BF16 cast
    monkeypatch.setattr('transformers.AutoConfig.from_pretrained', lambda *a, **k: learner.model.config)
    monkeypatch.setattr('transformers.AutoModelForCausalLM.from_pretrained', lambda *a, **k: tiny_lfm3_hf(window=4))
    model, _state, special, markers, _overrides = load_base_model(info['path'], 'unused', device='cpu', dtype=torch.float32)
    nxt = Lfm3Learner.from_model(model, learner.spec, special_ids=special, marker_ids=markers, compile_layers=False,
                                 fp8_base=False, tokenizer=PhraseTokenizer(), marker_names=ROLE_MARKERS)
    assert torch.equal(nxt.trainable_rows.rows.detach(), rows) and nxt.marker_init is None
    # opt-out: no master copy
    monkeypatch.setenv('NATLANG_MERGED_ROW_DTYPE', 'source')
    assert master_rows(learner.factor_tensors()[0]) is None


def test_every_engine_export_loads_the_checkpoints_marker_rows(tmp_path):
    """Distinct (text-initialised) marker rows in a native checkpoint reach every engine: the Python native engine's
    sparse rows, and the embedding of the SGLang HF export (`native_hf.merged_state`) and of the GGUF export that
    llama-server and the page load (`native_gguf.merged_backbone`)."""
    from test_native_serving_adapters import engine_fixture
    from natlang_neuralese.export.native_gguf import merged_backbone
    from natlang_neuralese.export.native_hf import merged_state
    from natlang_neuralese.model.native_model import load_native_checkpoint
    engine_fixture(tmp_path, 'lfm')
    state = torch.load(tmp_path / 'native.pt', map_location='cpu', weights_only=False)
    rows = state['interface_state']['input_rows']
    assert rows.shape[0] == 4
    distinct = torch.randn(rows.shape, generator=torch.Generator().manual_seed(5)).to(rows.dtype)
    state['interface_state']['input_rows'] = distinct
    torch.save(state, tmp_path / 'distinct.pt')
    model, _tokenizer, roles, config, _full = load_native_checkpoint(tmp_path / 'distinct.pt', allow_weights_only=True)
    markers = list(roles.ids)
    assert markers == list(config['marker_ids'])
    assert torch.equal(model.input_rows.detach().to(distinct.dtype), distinct)
    exported, tied = merged_state(model)
    embedding = next(value for name, value in exported.items() if name.endswith('embed_tokens.weight'))
    assert torch.equal(embedding[markers].to(distinct.dtype), distinct)
    hf, _tied = merged_backbone(model)
    assert torch.equal(hf.get_input_embeddings().weight[markers].detach().to(distinct.dtype), distinct)
