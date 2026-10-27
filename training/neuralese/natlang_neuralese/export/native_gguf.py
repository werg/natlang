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
    from ..model.native_model import load_native_checkpoint
    checkpoint, out_dir = Path(checkpoint).resolve(strict=True), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, roles, config, state = load_native_checkpoint(str(checkpoint), device='cpu', dtype=torch.float32,
                                                                    allow_weights_only=True)
    if config['architecture'] not in ('lfm2', 'lfm2_moe', 'qwen3', 'qwen2'):
        raise ValueError(f"GGUF export supports HF backbones only, not {config['architecture']!r}")
    hf, tied = merged_backbone(model)
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
            partial.replace(target)
            files.append({'outtype': outtype, 'file': target.name, 'bytes': target.stat().st_size,
                          'sha256': sha256_file_hex(target)})
    produce_open = int(roles.produce_open)
    checkpoint_sha256 = sha256_file_hex_cached(checkpoint)
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
        'not_represented': {'producer_adapter': 'low-rank residual inside producing spans; native server only'},
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
