"""Build a small random LFM2 base for CPU learning-mechanics tests.

The fixture keeps the production tokenizer, vocabulary, reserved control-token IDs,
and complete LFM2 layer-type pattern. It does not qualify pretrained model behavior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


CONTROL_TOKENS = ("<|reserved_7|>", "<|reserved_8|>")


def build(source: Path, output: Path, seed: int, width: int, heads: int, kv_heads: int, ffn: int) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"fixture output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(source, local_files_only=True)
    if config.model_type != "lfm2" or not config.layer_types:
        raise ValueError(f"expected an LFM2 config with layer_types, got {config.model_type!r}")
    config.hidden_size = config.block_dim = config.conv_dim = width
    config.num_attention_heads = config.num_heads = heads
    config.num_key_value_heads = kv_heads
    config.intermediate_size = config.block_ff_dim = ffn
    config.num_hidden_layers = len(config.layer_types)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    if len(tokenizer) > config.vocab_size:
        raise ValueError(f"tokenizer size {len(tokenizer)} exceeds model vocabulary {config.vocab_size}")
    token_ids = tokenizer.convert_tokens_to_ids(CONTROL_TOKENS)
    if any(token_id is None or token_id == tokenizer.unk_token_id for token_id in token_ids):
        raise ValueError("production tokenizer is missing Neuralese reserved control tokens")
    if any(tokenizer.convert_ids_to_tokens(token_id) != token for token, token_id in zip(CONTROL_TOKENS, token_ids)):
        raise ValueError("production tokenizer does not round-trip Neuralese reserved control tokens")

    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="local pinned LFM2.5-350M snapshot")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--ffn", type=int, default=512)
    args = parser.parse_args()
    build(args.source, args.output, args.seed, args.width, args.heads, args.kv_heads, args.ffn)


if __name__ == "__main__":
    main()
