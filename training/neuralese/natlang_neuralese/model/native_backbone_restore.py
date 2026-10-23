"""Construct a complete saved native backbone without rereading base weights."""
from pathlib import Path

import torch


def restore_backbone(base, architecture, state, *, device, dtype):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    base = Path(base)
    tokenizer = AutoTokenizer.from_pretrained(str(base), trust_remote_code=False)
    embedding_name = ('model.language_model.embed_tokens.weight' if architecture == 'qwen3_5'
                      else 'model.embed_tokens.weight')
    embedding = state[embedding_name]
    if architecture in {'maple', 'mellum'}:
        from ..maple.model import MapleConfig, restore_maple_owned_state
        config = MapleConfig.from_dir(base)
        config.dense_experts = (config.model_type != 'maple'
                                and not getattr(config, 'natlang_deployed_ternary', None))
        config.vocab_size = embedding.shape[0]
        # Parameters owned by the live model must not alias CPU checkpoint maps.
        moved = {name: value.detach().to(device=device, copy=True)
                 for name, value in state.items()}
        model = restore_maple_owned_state(config, moved, device)
        model.natlang_precision = 'bf16'
    else:
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
        model_type = type(skeleton)
        del skeleton
        model = model_type.from_pretrained(
            None, config=config, state_dict=values, dtype=dtype,
            device_map={'': device})
        model.requires_grad_(False)
        model.eval()
    return model, tokenizer
