"""Tuned soft states as llama.cpp native state spans (plan §3.3, §7).

A soft state is a native KV/hybrid state span (`NativeStateBlock`): per token, attention layers hold the key after the
key norm and before RoPE and the value, and LFM2/LFM3 short-convolution layers hold the gated input B*x. LFM3's
sliding and full attention layers are both `attention` states: llama.cpp writes raw keys in HF's layout (no q/k row
permutation) and rotates them with the consumer's own per-layer RoPE at its positions. The llama.cpp fork
imports exactly these quantities for virtual tokens (`llama_state_span_register`; the consumer applies RoPE at its own
positions), so the export carries the final verified tensors, not text and not an approximation.

`export_state_gguf(block, path, ...)` writes one state as GGUF (format: the fork's common/native-state.h). Before
exporting, the state's immutable derivation (parents, gradients, updates) is verified against the restored archive,
as the reference server does when consuming it.
"""
from __future__ import annotations

import sys

from ..common.hashing import sha256_file_hex
from . import TuneMathError
from .export import LLAMA_CPP

# native architectures whose state spans the llama.cpp fork imports (llm_arch_has_native_state)
STATE_ARCHITECTURES = ('lfm2', 'lfm3')


def _gguf():
    path = str(LLAMA_CPP / 'gguf-py')
    if path not in sys.path:
        sys.path.insert(0, path)
    import gguf
    return gguf


def export_state_gguf(block, path, *, state_id, text_sha256, role_tokens):
    """Write one native state span; returns its sha256. Only exact layer kinds the fork imports are accepted."""
    import numpy as np
    block.validate()
    if block.architecture not in STATE_ARCHITECTURES:
        raise TuneMathError(f'llama.cpp state-span export supports {", ".join(STATE_ARCHITECTURES)} states, not {block.architecture}')
    gguf = _gguf()
    writer = gguf.GGUFWriter(str(path), block.architecture)
    writer.add_string('general.type', 'natlang.native_state')
    for key, value in (('id', state_id), ('text_sha256', text_sha256), ('architecture', block.architecture),
                       ('checkpoint_sha256', block.checkpoint_sha256), ('tokenizer_fingerprint', block.tokenizer_fingerprint),
                       ('embedding_weight_sha256', block.embedding_weight_sha256)):
        writer.add_string(f'natlang.state.{key}', value)
    writer.add_int32('natlang.state.n_tokens', len(block.token_ids))
    writer.add_int32('natlang.state.consume_open', int(role_tokens.consume_open))
    writer.add_int32('natlang.state.consume_close', int(role_tokens.consume_close))
    count = len(block.token_ids)
    for index, layer in enumerate(block.layers):
        if layer['kind'] == 'attention':
            # [tokens, heads, head_dim] -> rows of n_embd_k_gqa, head-major as llama.cpp lays out K and V
            for name, key in (('raw_k', 'state_k'), ('v', 'state_v')):
                writer.add_tensor(f'blk.{index}.{key}', layer[name].detach().float().reshape(count, -1).contiguous().numpy().astype(np.float32))
        elif layer['kind'] == 'finite_convolution':
            writer.add_tensor(f'blk.{index}.state_conv', layer['gated_inputs'].detach().float().contiguous().numpy().astype(np.float32))
        else:
            raise TuneMathError(f'layer {index} state kind {layer["kind"]} has no llama.cpp import')
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return sha256_file_hex(path)


def export_soft_states(engine, soft_states, out, *, base_gguf_sha256):
    """Export each trained soft state of a build (`soft_fragment` entries, states already in the engine store)."""
    from ..model.native_state_derivation import verify_state_derivation
    exported = []
    for item in soft_states:
        record = engine.store.get_native_state(item['state'])
        if record is None:
            raise TuneMathError(f'soft state {item["state"]} is not in the store')
        if record.block.derivation is not None:
            verify_state_derivation(engine.store, record.block)
        file = f"exports/{item['state']}.state.gguf"
        target = out / file
        target.parent.mkdir(parents=True, exist_ok=True)
        sha = export_state_gguf(record.block, target, state_id=item['state'], text_sha256=item['textSha256'],
                                role_tokens=engine.role_tokens)
        item['exports'] = {'gguf': {'file': file, 'sha256': sha, 'baseGgufSha256': base_gguf_sha256}}
        exported.append(item)
    return exported
