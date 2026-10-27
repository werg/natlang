"""GGUF export of a native checkpoint for ordinary-token execution on llama.cpp (docs/NATIVE_MODELS.md).

A native model's ordinary-token computation is its HF backbone plus learned sparse vocabulary rows: the four role
markers (`input_rows`/`output_rows`) and the reasoning-boundary overrides (`extra_input_rows`/`extra_output_rows`),
which replace the embedding row and the readout row of their tokens (`SharedStateModel.embed`/`.logits`). Those rows
are merged into the backbone's embedding and head, so the merged backbone is exact for every ordinary token. The
low-rank producer adapter applies only inside producing spans, which llama.cpp does not run: the export records it as
not represented, and serving bans the producer's opening marker so execution stays in ordinary tokens.

The chat template is the tokenizer's own, with the backbone's pinned history-reasoning template arguments, which is
what the native server renders through (`serve/chat.py` `bind_history_reasoning`).

    python -m natlang_neuralese.export.native_gguf CHECKPOINT OUT_DIR [--outtype bf16|f32|q8_0]...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex, sha256_file_hex_cached

SCHEMA = 'natlang.native-gguf/1'
LLAMA_CPP = Path(__file__).resolve().parents[4] / 'vendor' / 'llama.cpp'


def merged_backbone(model):
    """The HF backbone with the native sparse rows written into its embedding and readout; returns (hf, tied)."""
    hf = model.port.hf
    embedding = hf.get_input_embeddings().weight
    output = hf.get_output_embeddings().weight
    tied = output is embedding
    if tied != model.tied:
        raise ValueError('native and HF backbones disagree on tied embeddings')
    rows = [(model.marker_ids, model.input_rows, None if tied else model.output_rows)]
    if model.token_override_ids:
        rows.append((model.token_override_ids, model.extra_input_rows, None if tied else model.extra_output_rows))
    with torch.no_grad():
        for ids, inputs, outputs in rows:
            index = torch.tensor(list(ids), device=embedding.device)
            embedding.index_copy_(0, index, inputs.detach().to(embedding.dtype))
            if outputs is not None:
                output.index_copy_(0, index, outputs.detach().to(output.dtype))
    hf.config.vocab_size = embedding.shape[0]
    return hf, tied


def export(checkpoint, out_dir, *, outtypes=('bf16',), llama_cpp=LLAMA_CPP, python=sys.executable):
    from ..model.native_model import load_native_checkpoint, native_tensor_identity
    checkpoint, out_dir = Path(checkpoint).resolve(strict=True), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, roles, config, state = load_native_checkpoint(str(checkpoint), device='cpu', dtype=torch.float32,
                                                                    allow_weights_only=True)
    if config['architecture'] not in ('lfm2', 'lfm2_moe', 'qwen3', 'qwen2'):
        raise ValueError(f"GGUF export supports HF backbones only, not {config['architecture']!r}")
    checkpoint_sha256 = sha256_file_hex_cached(checkpoint)
    # The native identity states carry (natlang.identity.* in the GGUF; llama-server declares it and stamps produced
    # states with it). It names the embedding path's dtype: an f32 file serves the float32 identity, the others the
    # bfloat16 one the native server uses on CUDA, so states exchange with that server.
    identities = {'float32': native_tensor_identity(model, config, checkpoint_sha256=checkpoint_sha256)}
    if any(outtype != 'f32' for outtype in outtypes):
        reduced = load_native_checkpoint(str(checkpoint), device='cpu', dtype=torch.bfloat16, allow_weights_only=True)
        identities['bfloat16'] = native_tensor_identity(reduced[0], reduced[3], checkpoint_sha256=checkpoint_sha256)
        del reduced
    hf, tied = merged_backbone(model)
    # The producer adapter (inside producing spans, e + up(down(e))) as GGUF tensors the llama.cpp fork applies.
    adapter = getattr(model, 'adapter', None)
    producer = None if adapter is None else {
        'natlang_producer_down.weight': adapter.down.weight.detach().float().cpu().numpy().copy(),
        'natlang_producer_up.weight': adapter.up.weight.detach().float().cpu().numpy().copy()}
    markers = {f'natlang.{name}': int(getattr(roles, name)) for name in ('produce_open', 'produce_close', 'consume_open', 'consume_close')}
    reasoning = dict(getattr(tokenizer, 'natlang_history_reasoning', {}) or {})
    template = tokenizer.chat_template
    if not isinstance(template, str):
        raise ValueError('the tokenizer has no single chat template to export')
    (out_dir / 'chat-template.jinja').write_text(template)
    kwargs = dict(reasoning.get('template_kwargs') or {})
    (out_dir / 'chat-template.kwargs.json').write_text(json.dumps(kwargs, indent=1) + '\n')
    files = []
    with tempfile.TemporaryDirectory(prefix='native-gguf-', dir=out_dir) as scratch:
        hf_dir = Path(scratch) / 'hf'
        hf.save_pretrained(hf_dir, safe_serialization=True)
        # The tokenizer as the native model uses it: role tokens added, the original template.
        apply = getattr(tokenizer, '_natlang_history_reasoning_original_apply_chat_template', None)
        if apply is not None:
            tokenizer.apply_chat_template = apply
        tokenizer.save_pretrained(hf_dir)
        del model, hf
        # The merged backbone's config and tokenizer without weights: what GGUF LoRA conversion of tuned adapters
        # for this base reads (`tune.students.*.hfConfigDir`).
        config_dir = out_dir / 'hf-config'
        shutil.rmtree(config_dir, ignore_errors=True)
        config_dir.mkdir()
        for item in hf_dir.iterdir():
            if item.is_file() and item.suffix not in ('.safetensors', '.bin') and not item.name.endswith('.index.json'):
                shutil.copy2(item, config_dir / item.name)
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(llama_cpp / 'gguf-py'), os.environ.get('PYTHONPATH', '')])}
        for outtype in outtypes:
            target = out_dir / f'model-{outtype}.gguf'
            partial = target.with_suffix('.gguf.partial')
            subprocess.run([python, str(llama_cpp / 'convert_hf_to_gguf.py'), str(hf_dir), '--outfile', str(partial),
                            '--outtype', outtype], check=True, env=env)
            identity = identities['float32' if outtype == 'f32' else 'bfloat16']
            _add_natlang(partial, producer or {}, {**markers, **{f'natlang.identity.{key}': str(value) for key, value in identity.items()}},
                         llama_cpp)
            partial.replace(target)
            files.append({'outtype': outtype, 'file': target.name, 'bytes': target.stat().st_size,
                          'sha256': sha256_file_hex(target), 'native_identity': dict(identity)})
    produce_open = int(roles.produce_open)
    # Files of other tensor types exported earlier from the same checkpoint stay listed.
    try:
        earlier = json.loads((out_dir / 'manifest.json').read_text())
    except (OSError, ValueError):
        earlier = {}
    if earlier.get('checkpoint_sha256') == checkpoint_sha256:
        files = [item for item in earlier.get('files', []) if item['outtype'] not in outtypes and (out_dir / item['file']).exists()] + files
    manifest = {
        'schema': SCHEMA, 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha256,
        'inference_source': state.get('inference_source'),
        'step': ((state.get('inference_source') or {}).get('step') or (state.get('training') or {}).get('global_step')
                 or (state.get('training') or {}).get('step')),
        'architecture': config['architecture'], 'tokenizer_fingerprint': config.get('tokenizer_fingerprint'),
        'merged_rows': {'marker_ids': list(config['marker_ids']), 'token_override_ids': list(config.get('token_override_ids', [])),
                        'tied': tied},
        # The producer adapter is in the GGUF (natlang_producer_down/up); state capture and the native-state protocol
        # are llama-server's (the fork's native state production).
        'producer_adapter': producer is not None,
        'role_markers': markers,
        'serving': {'chat_template': 'chat-template.jinja', 'chat_template_sha256': sha256_file_hex(out_dir / 'chat-template.jinja'),
                    'chat_template_kwargs': kwargs, 'history_reasoning': reasoning.get('policy'),
                    # The producer marker would start a span whose adapter llama.cpp does not apply.
                    'logit_bias': {str(produce_open): -100}},
        'files': files,
        'hf_config': 'hf-config',
        'converter': {'llama_cpp': str(llama_cpp), 'replication': _replication(llama_cpp)},
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


def _add_natlang(path, tensors, metadata, llama_cpp):
    """Rewrite a converted GGUF with natlang's extra F32 tensors and metadata (the producer adapter, the role markers as
    integers, the native identity as strings). Every original key and tensor is copied unchanged."""
    if str(Path(llama_cpp) / 'gguf-py') not in sys.path:
        sys.path.insert(0, str(Path(llama_cpp) / 'gguf-py'))
    import gguf
    reader = gguf.GGUFReader(str(path))
    architecture = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    rewritten = Path(str(path) + '.natlang')
    writer = gguf.GGUFWriter(str(rewritten), architecture)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.') or field.name in metadata:
            continue
        kind = field.types[0]
        writer.add_key_value(field.name, field.contents(), kind, sub_type=field.types[-1] if kind == gguf.GGUFValueType.ARRAY else None)
    for key, value in metadata.items():
        if isinstance(value, str):
            writer.add_string(key, value)
        else:
            writer.add_uint32(key, value)
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type)
    for name, value in tensors.items():
        writer.add_tensor_info(name, value.shape, value.dtype, value.nbytes, gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)
    for value in tensors.values():
        writer.write_tensor_data(value)
    writer.close()
    del reader
    os.replace(rewritten, path)


def _replication(llama_cpp):
    path = Path(llama_cpp) / 'NATLANG_REPLICATION.json'
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('checkpoint')
    parser.add_argument('out_dir')
    parser.add_argument('--outtype', action='append', choices=('f32', 'bf16', 'f16', 'q8_0'),
                        help='GGUF tensor type; repeat for several files (default bf16)')
    args = parser.parse_args(argv)
    manifest = export(args.checkpoint, args.out_dir, outtypes=tuple(args.outtype or ('bf16',)))
    print(json.dumps({'manifest': str(Path(args.out_dir) / 'manifest.json'), 'files': manifest['files']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
