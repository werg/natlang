"""A natlang GGUF of a tiny random LFM3 for the llama.cpp fork's server tests (docs/ENGINE_PARITY.md).

The fork's native features (RL capture, native-state production) need a natlang export: `natlang.identity.*`, the role
markers and the producer adapter that `export/native_gguf.py` writes. This builds a 6-layer, 64-wide LFM3 with every layer
form and LFM3-1.7B's real tokenizer and RoPE parameters (the converter recognizes the tokenizer's pre-tokenizer; a fixture
tokenizer would need its own converter entry), saves it as a native checkpoint and exports it with the fork's converter.

    python scripts/neuralese_data/tiny_lfm3_gguf.py OUT_DIR --llama-cpp ../llama.cpp-neuralese [--tokenizer DIR]

Prints the GGUF path (OUT_DIR/export/model-f32.gguf). Without --tokenizer it downloads LFM3-1.7B's tokenizer and config
files (about 18 MB, no weights).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

REPO = 'LiquidAI/LFM3-1.7B'
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja')
LAYER_TYPES = ('conv', 'full_attention', 'conv', 'sliding_attention', 'conv', 'full_attention')


def tokenizer_dir(path, scratch):
    if path is not None:
        return Path(path)
    from huggingface_hub import hf_hub_download
    for name in ('config.json', *TOKENIZER_FILES):
        hf_hub_download(REPO, name, local_dir=scratch)
    return Path(scratch)


def build(out_dir, llama_cpp, tokenizer=None, outtype='f32'):
    from natlang_neuralese.export import native_gguf
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
    from natlang_neuralese.third_party import lfm3
    out_dir = Path(out_dir)
    source = tokenizer_dir(tokenizer, out_dir / 'tokenizer-source')
    real = json.loads((source / 'config.json').read_text())
    config = lfm3.config_class()(
        vocab_size=real['vocab_size'], hidden_size=64, intermediate_size=128, num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=2, num_key_value_heads=1, head_dim=32, layer_types=list(LAYER_TYPES), sliding_window=16,
        max_position_embeddings=4096, bos_token_id=real['bos_token_id'], eos_token_id=real['eos_token_id'],
        pad_token_id=real['pad_token_id'], use_sandwich_norm=True, tie_word_embeddings=True,
        rope_parameters=real['rope_parameters'], conv_L_cache=real['conv_L_cache'], conv_bias=False, norm_eps=real['norm_eps'])
    torch.manual_seed(0)
    base = out_dir / 'base'
    base.mkdir(parents=True, exist_ok=True)
    lfm3.causal_lm_class()(config).eval().save_pretrained(base, safe_serialization=True)
    for name in TOKENIZER_FILES:
        if (source / name).exists():
            shutil.copy2(source / name, base / name)
    model, _, _, model_config = load_native_model(base, rank=4, dtype=torch.bfloat16)
    checkpoint = out_dir / 'checkpoint.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': model_config, **model_state(model)}, checkpoint)
    del model
    manifest = native_gguf.export(checkpoint, out_dir / 'export', outtypes=(outtype,), llama_cpp=Path(llama_cpp).resolve())
    return out_dir / 'export' / manifest['files'][-1]['file']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('out_dir')
    parser.add_argument('--llama-cpp', required=True, help='the fork checkout whose converter registers LFM3')
    parser.add_argument('--tokenizer', help='a directory with LFM3-1.7B config.json and tokenizer files')
    parser.add_argument('--outtype', default='f32', choices=('f32', 'bf16', 'q8_0'))
    args = parser.parse_args(argv)
    print(build(args.out_dir, args.llama_cpp, args.tokenizer, args.outtype))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
