"""HF safetensors export of a native checkpoint for ordinary-token serving on the SGLang fork (training/neuralese/sglang-fork.json).

A native model's ordinary-token computation is its HF backbone plus learned sparse vocabulary rows: the four role
markers (`input_rows`/`output_rows`) and the reasoning-boundary overrides (`extra_input_rows`/`extra_output_rows`),
which replace the embedding row and the readout row of their tokens (`SharedStateModel.embed`/`.logits`). The export
writes those rows into the embedding (and an untied head) of a copy of the backbone state, so the exported HF model is
exact for every ordinary token. The source checkpoint and the loaded model are not modified.

Not represented, recorded in the manifest and enforced by serving: the low-rank producer adapter, which applies only
inside producing spans. Serving through this export bans all four role markers (`banned_token_ids`), so execution stays
in ordinary tokens and the ban is part of every recorded sample's law. Native adapters, coefficient policies and
imported native state are refused by the serving front rather than dropped.

    python -m natlang_neuralese.export.native_hf CHECKPOINT OUT_DIR [--base DIR] [--dtype bfloat16|float32]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

SCHEMA = 'natlang.native-hf-export/1'


def merged_state(model):
    """The backbone's HF state dict with the native sparse rows merged into embedding and head (copies)."""
    hf = model.port.hf
    embedding = hf.get_input_embeddings().weight
    output = hf.get_output_embeddings().weight
    tied = output is embedding
    if tied != model.tied:
        raise ValueError('native and HF backbones disagree on tied embeddings')
    state = {name: tensor.detach() for name, tensor in hf.state_dict().items()}
    modules = {id(module): name for name, module in hf.named_modules()}
    embedding_name = modules[id(hf.get_input_embeddings())] + '.weight'
    output_name = modules[id(hf.get_output_embeddings())] + '.weight'
    merged_embedding = embedding.detach().clone()
    merged_output = None if tied else output.detach().clone()
    # A tied model reads its readout rows from the input rows (SharedStateModel.logits).
    rows = [(model.marker_ids, model.input_rows, None if tied else model.output_rows)]
    if model.token_override_ids:
        rows.append((model.token_override_ids, model.extra_input_rows, None if tied else model.extra_output_rows))
    for ids, inputs, outputs in rows:
        index = torch.tensor(list(ids), device=merged_embedding.device)
        merged_embedding.index_copy_(0, index, inputs.detach().to(merged_embedding.dtype))
        if merged_output is not None:
            merged_output.index_copy_(0, index, outputs.detach().to(merged_output.dtype))
    state[embedding_name] = merged_embedding
    if tied:
        # safetensors stores one tensor; the HF config (tie_word_embeddings) ties the head to it on load.
        state.pop(output_name, None)
    else:
        state[output_name] = merged_output
    return state, tied


def export_loaded(model, tokenizer, roles, config, *, checkpoint, checkpoint_sha256, out_dir, dtype=torch.bfloat16):
    """Write OUT_DIR (config, model.safetensors, tokenizer, natlang-export.json) from an already loaded native model.

    Reuses OUT_DIR when its manifest names the same checkpoint SHA-256 and dtype. Returns the manifest."""
    from safetensors.torch import save_file
    from ..common.hashing import sha256_file_hex
    out_dir = Path(out_dir)
    manifest_path = out_dir / 'natlang-export.json'
    try:
        earlier = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        earlier = None
    dtype_name = str(dtype).replace('torch.', '')
    if (earlier and earlier.get('checkpoint_sha256') == checkpoint_sha256 and earlier.get('dtype') == dtype_name
            and (out_dir / 'model.safetensors').is_file()
            and sha256_file_hex(out_dir / 'model.safetensors') == earlier['weights_sha256']):
        return earlier
    if config['architecture'] != 'lfm3':
        raise ValueError(f"SGLang export is qualified for lfm3 only, not {config['architecture']!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    state, tied = merged_state(model)
    state = {name: tensor.to(dtype).contiguous().cpu() for name, tensor in state.items()}
    partial = out_dir / 'model.safetensors.partial'
    save_file(state, str(partial), metadata={'format': 'pt'})
    os.replace(partial, out_dir / 'model.safetensors')
    hf_config = model.port.hf.config
    hf_config.vocab_size = int(state[next(name for name in state if name.endswith('embed_tokens.weight'))].shape[0])
    hf_config.torch_dtype = dtype_name
    hf_config.save_pretrained(out_dir)
    # The config's auto_map names LFM3's Hub remote code; SGLang reads the config through it (--trust-remote-code)
    # and runs its own model. The vendored files are byte-identical to the pinned revision (third_party/lfm3).
    import shutil
    from ..third_party import lfm3 as vendored
    for name in vendored.verify_sources():
        if name.endswith('.py'):
            shutil.copyfile(Path(vendored.__file__).parent / name, out_dir / name)
    # The tokenizer with its original template (a serving front may have bound the history-reasoning wrapper).
    apply = getattr(tokenizer, '_natlang_history_reasoning_original_apply_chat_template', None)
    bound = tokenizer.__dict__.get('apply_chat_template')
    try:
        if apply is not None:
            tokenizer.apply_chat_template = apply
        tokenizer.save_pretrained(out_dir)
    finally:
        if bound is not None:
            tokenizer.apply_chat_template = bound
    markers = {name: int(getattr(roles, name)) for name in ('produce_open', 'produce_close', 'consume_open', 'consume_close')}
    manifest = {
        'schema': SCHEMA, 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha256, 'dtype': dtype_name,
        'architecture': config['architecture'], 'tokenizer_fingerprint': config.get('tokenizer_fingerprint'),
        'merged_rows': {'marker_ids': list(config['marker_ids']),
                        'token_override_ids': list(config.get('token_override_ids', [])), 'tied': tied},
        'weights_sha256': sha256_file_hex(out_dir / 'model.safetensors'),
        'role_markers': markers,
        'not_represented': {'producer_adapter': 'applies only inside producing spans; serving bans the role markers'},
        'serving': {'banned_token_ids': sorted(markers.values())},
    }
    tmp = manifest_path.with_suffix('.json.partial')
    tmp.write_text(json.dumps(manifest, indent=1) + '\n')
    os.replace(tmp, manifest_path)
    return manifest


def export(checkpoint, out_dir, *, base=None, dtype=torch.bfloat16):
    from ..common.hashing import sha256_file_hex_cached
    from ..model.native_model import load_native_checkpoint
    checkpoint = Path(checkpoint).resolve(strict=True)
    model, tokenizer, roles, config, _state = load_native_checkpoint(str(checkpoint), base=base, device='cpu', dtype=dtype,
                                                                     allow_weights_only=True)
    return export_loaded(model, tokenizer, roles, config, checkpoint=checkpoint,
                         checkpoint_sha256=sha256_file_hex_cached(checkpoint), out_dir=out_dir, dtype=dtype)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint')
    parser.add_argument('out_dir')
    parser.add_argument('--base')
    parser.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    args = parser.parse_args(argv)
    print(json.dumps(export(args.checkpoint, args.out_dir, base=args.base, dtype=getattr(torch, args.dtype)), indent=1))


if __name__ == '__main__':
    main()
