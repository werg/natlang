"""HF safetensors export of a native checkpoint for ordinary-token serving on the SGLang fork (training/neuralese/sglang-fork.json).

A native model's ordinary-token computation is its HF backbone plus learned sparse vocabulary rows: the four role
markers (`input_rows`/`output_rows`) and the reasoning-boundary overrides (`extra_input_rows`/`extra_output_rows`),
which replace the embedding row and the readout row of their tokens (`SharedStateModel.embed`/`.logits`). The export
writes those rows into the embedding (and an untied head) of a copy of the backbone state, so the exported HF model is
exact for every ordinary token. The source checkpoint and the loaded model are not modified.

The low-rank producer adapter applies only to tokens inside a produce span. The export precomputes the producing
embedding of every token, `adapter(embed(t))` with the native module itself, as `model.natlang_producer_embed`,
and names the markers in the config (`natlang_producer`); the fork's LFM3 model feeds producing token t as id
t + vocab_size and looks it up there (NATLANG_FORK.md). Native adapters, coefficient policies and imported native
state are refused by the serving front rather than dropped. `serving.banned_token_ids` is empty: the front samples
from the native law and follows the native marker protocol.

    python -m natlang_neuralese.export.native_hf CHECKPOINT OUT_DIR [--base DIR] [--dtype bfloat16|float32]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

SCHEMA = 'natlang.native-hf-export/5'
PRODUCER_CHUNK = 8192


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


ADAPTER_MODULES = {'self_attn.q_proj': ('qkv_proj', 'q'), 'self_attn.k_proj': ('qkv_proj', 'k'),
                   'self_attn.v_proj': ('qkv_proj', 'v'), 'self_attn.gate_proj': ('gate_proj', None),
                   'self_attn.o_proj': ('o_proj', None), 'conv.in_proj': ('in_proj', None),
                   'conv.out_proj': ('out_proj', None), 'feed_forward.w1': ('gate_up_proj', 'gate'),
                   'feed_forward.w3': ('gate_up_proj', 'up'), 'feed_forward.w2': ('w2', None)}


def adapter_targets(model):
    """Every adaptable native matrix (`native_adapters.target_parameters`) -> the fork's LFM3 module (natlang
    adapter slots, `NatlangAdapters`) and the row offset of its output inside a packed module (q|k|v, w1|w3)."""
    import re
    from ..model.native_adapters import target_parameters
    shapes = {name: tuple(value.shape) for name, value in target_parameters(model).items()}
    targets = {}
    for name, shape in shapes.items():
        match = re.fullmatch(r'port\.hf\.model\.layers\.(\d+)\.(.+)\.weight', name)
        if match is None or match.group(2) not in ADAPTER_MODULES or len(shape) != 2:
            raise ValueError(f'no SGLang adapter module for native matrix {name}')
        layer, (module, part) = match.group(1), ADAPTER_MODULES[match.group(2)]
        sibling = lambda sub: shapes[f'port.hf.model.layers.{layer}.{sub}.weight'][0]
        offset = {None: 0, 'q': 0, 'gate': 0, 'k': sibling('self_attn.q_proj') if part == 'k' else 0,
                  'v': (sibling('self_attn.q_proj') + sibling('self_attn.k_proj')) if part == 'v' else 0,
                  'up': sibling('feed_forward.w1') if part == 'up' else 0}[part]
        targets[name] = {'module': f'layers.{layer}.{module}', 'offset': offset, 'shape': list(shape)}
    return targets


@torch.no_grad()
def producer_embeddings(model):
    """Row t: the producing input embedding of token t (`model.adapter(model.embed(t), producing)`, as native_step)."""
    weight = model.port.embedding_weight
    rows = []
    for start in range(0, weight.shape[0], PRODUCER_CHUNK):
        ids = torch.arange(start, min(start + PRODUCER_CHUNK, weight.shape[0]), device=weight.device)[None]
        rows.append(model.adapter(model.embed(ids), torch.ones_like(ids, dtype=torch.bool))[0].detach())
    return torch.cat(rows)


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
    if (earlier and earlier.get('schema') == SCHEMA and earlier.get('checkpoint_sha256') == checkpoint_sha256 and earlier.get('dtype') == dtype_name
            and (out_dir / 'model.safetensors').is_file()
            and sha256_file_hex(out_dir / 'model.safetensors') == earlier['weights_sha256']):
        return earlier
    if config['architecture'] != 'lfm3':
        raise ValueError(f"SGLang export is qualified for lfm3 only, not {config['architecture']!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    state, tied = merged_state(model)
    state['model.natlang_producer_embed'] = producer_embeddings(model)
    state = {name: tensor.to(dtype).contiguous().cpu() for name, tensor in state.items()}
    partial = out_dir / 'model.safetensors.partial'
    save_file(state, str(partial), metadata={'format': 'pt'})
    os.replace(partial, out_dir / 'model.safetensors')
    hf_config = model.port.hf.config
    hf_config.vocab_size = int(state[next(name for name in state if name.endswith('embed_tokens.weight'))].shape[0])
    hf_config.torch_dtype = dtype_name
    markers = {name: int(getattr(roles, name)) for name in ('produce_open', 'produce_close', 'consume_open', 'consume_close')}
    hf_config.natlang_producer = {**markers, 'offset': hf_config.vocab_size}
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
    manifest = {
        'schema': SCHEMA, 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha256, 'dtype': dtype_name,
        'architecture': config['architecture'], 'tokenizer_fingerprint': config.get('tokenizer_fingerprint'),
        'merged_rows': {'marker_ids': list(config['marker_ids']),
                        'token_override_ids': list(config.get('token_override_ids', [])), 'tied': tied},
        'weights_sha256': sha256_file_hex(out_dir / 'model.safetensors'),
        'role_markers': markers,
        # Native-state layout of this model (`Engine._native_exact`), so a serving front checks imported blocks
        # without loading the native model.
        'layer_kinds': ['attention' if model.port.is_attention(index) else 'gated_delta'
                        if tuple(getattr(model.port, 'layer_types', ()))[index:index + 1] == ('linear_attention',)
                        else 'finite_convolution' for index in range(model.port.num_layers)],
        'adapter_targets': adapter_targets(model),
        'producer_embedding': {'weight': 'model.natlang_producer_embed', 'input_id_offset': hf_config.vocab_size},
        # No marker is banned: produce spans run on the fork (producer table, span export), consume markers and other
        # protocol violations are the native engine's errors, so SGLang samples from the native law.
        'serving': {'banned_token_ids': []},
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
