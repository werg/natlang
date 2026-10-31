"""Construct a complete saved native backbone without rereading base weights."""
from pathlib import Path

import torch


def validate_backbone_state_coverage(expected_state, checkpoint_state):
    """Require an exact, shape-compatible checkpoint for every backbone tensor."""
    if not isinstance(checkpoint_state, dict):
        raise ValueError('checkpoint backbone state must be a mapping')
    expected_keys = set(expected_state)
    actual_keys = set(checkpoint_state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(f'checkpoint backbone state is incomplete (missing={missing[:8]}, unexpected={unexpected[:8]})')
    for name, expected in expected_state.items():
        value = checkpoint_state[name]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(expected.shape):
            raise ValueError(f'checkpoint backbone tensor has a mismatched shape or type: {name}')


def restore_backbone(base, architecture, state, *, device, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    base = Path(base)
    if architecture == 'lfm3':
        # Transformers has no lfm3: register the vendored, hash-verified pinned modeling code (no remote code).
        from ..third_party import lfm3
        lfm3.register()
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=False)
    embedding_name = ('model.language_model.embed_tokens.weight' if architecture == 'qwen3_5'
                      else 'model.embed_tokens.weight')
    embedding = state[embedding_name]
    config = AutoConfig.from_pretrained(str(base), trust_remote_code=False)
    if architecture == 'qwen3_5':
        config.text_config.vocab_size = embedding.shape[0]
    config.vocab_size = embedding.shape[0]
    values = ({name: value.detach().clone() for name, value in state.items()}
              if torch.device(device).type == 'cpu' else state)
    # None deliberately supplies no base weight path. The standard HF
    # loader constructs nonpersistent buffers and preserves declared ties.
    # The Auto factory in the installed Transformers release resolves a
    # repository name even with an explicit config and None weight source.
    # Resolve its concrete registered class on meta, without loading bytes.
    model_class = AutoModelForImageTextToText if architecture == 'qwen3_5' else AutoModelForCausalLM
    with torch.device('meta'):
        skeleton = model_class.from_config(config)
    validate_backbone_state_coverage(skeleton.state_dict(), state)
    model_type = type(skeleton)
    del skeleton
    model = model_type.from_pretrained(
        None, config=config, state_dict=values, dtype=dtype,
        device_map={'': device})
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer
