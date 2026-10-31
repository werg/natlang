from __future__ import annotations

import hashlib
import json

import pytest

from natlang_neuralese.model.native_model import (
    _checkpoint_bound_config_inventory,
    tokenizer_inventory,
)
from natlang_neuralese.model.native_backbone_restore import validate_backbone_state_coverage
import torch


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_checkpoint_backbone_allows_config_tokenizer_only_base(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text(json.dumps({'model_type': 'lfm3', 'vocab_size': 17}))
    (base / 'tokenizer.json').write_text('{"version":"1.0"}')
    (base / 'tokenizer_config.json').write_text('{"tokenizer_class":"Lfm3Tokenizer"}')
    local = tokenizer_inventory(base)
    full = {
        **local,
        'model-00001-of-00001.safetensors': 'a' * 64,
    }

    verified = _checkpoint_bound_config_inventory(
        base,
        full,
        {'model.embed_tokens.weight': object(), 'model.layers.0.weight': object()},
    )

    assert verified == local
    assert 'model-00001-of-00001.safetensors' not in verified


def test_config_only_restore_rejects_tampered_metadata_and_missing_backbone(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    config = base / 'config.json'
    config.write_text(json.dumps({'model_type': 'lfm3', 'vocab_size': 17}))
    (base / 'tokenizer.json').write_text('{"version":"1.0"}')
    local = tokenizer_inventory(base)
    inventory = {**local, 'model-00001-of-00001.safetensors': 'b' * 64}
    state = {'model.embed_tokens.weight': object()}

    config.write_text(json.dumps({'model_type': 'lfm3', 'vocab_size': 18}))
    with pytest.raises(ValueError, match='config/tokenizer inventory'):
        _checkpoint_bound_config_inventory(base, inventory, state)

    config.write_text(json.dumps({'model_type': 'lfm3', 'vocab_size': 17}))
    with pytest.raises(ValueError, match='token embedding'):
        _checkpoint_bound_config_inventory(base, inventory, {'model.layers.0.weight': object()})


def test_config_only_restore_requires_a_full_saved_weight_inventory(tmp_path):
    base = tmp_path / 'base'
    base.mkdir()
    (base / 'config.json').write_text(json.dumps({'model_type': 'lfm3', 'vocab_size': 17}))
    (base / 'tokenizer.json').write_text('{"version":"1.0"}')
    local = tokenizer_inventory(base)

    with pytest.raises(ValueError, match='safetensors weights'):
        _checkpoint_bound_config_inventory(
            base, local, {'model.embed_tokens.weight': object()})


def test_checkpoint_backbone_requires_exact_all_parameter_and_buffer_shapes():
    expected = {
        'model.embed_tokens.weight': torch.empty((17, 8), device='meta'),
        'model.layers.0.position_cache': torch.empty((1, 16), device='meta'),
    }
    good = {name: torch.empty_like(value, device='cpu') for name, value in expected.items()}

    validate_backbone_state_coverage(expected, good)
    with pytest.raises(ValueError, match='incomplete'):
        validate_backbone_state_coverage(expected, {'model.embed_tokens.weight': good['model.embed_tokens.weight']})
    with pytest.raises(ValueError, match='mismatched shape'):
        validate_backbone_state_coverage(expected, {
            'model.embed_tokens.weight': good['model.embed_tokens.weight'],
            'model.layers.0.position_cache': torch.empty((1, 15)),
        })
