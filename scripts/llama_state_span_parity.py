#!/usr/bin/env python3
"""Forward parity of the llama.cpp fork's native state spans against natlang's native engine (plan §3.3, §7).

    PYTHONPATH=training/neuralese:.:tests/neuralese .venv-neuralese/bin/python scripts/llama_state_span_parity.py WORKDIR
        [--llama ~/llama.cpp-neuralese]

On the real LFM2.5-350M (CPU, F32): the Python engine forwards a prefix and the consume-open marker, splices a
perturbed soft state (not reproducible from text), and forwards the consume-close marker and a suffix;
llama-state-span-dump imports the exported state (common/native-state.h) for the same sequence, as one batch and in
7-token chunks. Prints max/mean |dlogit| and top-1 agreement next to the plain-text backend baseline, and a negative
control (the unperturbed state against the perturbed reference). Results of 2026-10-26: plain text 0.075 max,
state span 0.043 max (both batchings), top-1 1.000; control 2.68.
"""
import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('workdir')
    parser.add_argument('--llama', default=str(Path.home() / 'llama.cpp-neuralese'))
    args = parser.parse_args()
    S, LCPP = Path(args.workdir), Path(args.llama)
    S.mkdir(parents=True, exist_ok=True)
    from test_tune_rows import SNAPSHOTS, real_engine
    from natlang_neuralese.model.shared_state import splice_block
    from natlang_neuralese.tune.soft_export import export_state_gguf
    engine, _ = real_engine(S)
    model_gguf = S / 'lfm-f32.gguf'
    if not model_gguf.exists():
        subprocess.run([sys.executable, str(LCPP / 'convert_hf_to_gguf.py'), str(SNAPSHOTS[0].parent), '--outtype', 'f32',
                        '--outfile', str(model_gguf)], check=True, env={'PYTHONPATH': str(LCPP / 'gguf-py'), 'PATH': '/usr/bin:/bin', 'HF_HUB_OFFLINE': '1'})
    roles = engine.role_tokens
    text = 'Answer only with the capital city of the country named in the question, in lowercase.'
    block = engine.store.get_native_state(engine.native_write_text(text)['id']).block
    generator = torch.Generator().manual_seed(1)
    layers = tuple({key: (value + 0.15 * value.float().std() * torch.randn(value.shape, generator=generator)).to(value.dtype)
                    if isinstance(value, torch.Tensor) else value for key, value in layer.items()} for layer in block.layers)
    state = dataclasses.replace(block, layers=layers, derivation=None)
    encode = lambda value: engine.tokenizer(value, add_special_tokens=False).input_ids
    prefix, suffix = encode('<|im_start|>user\nInstructions: '), encode('\nQuestion: France?<|im_end|>\n<|im_start|>assistant\n')
    with torch.no_grad():
        cache = engine._native_forward(prefix + [roles.consume_open], None)['cache']
        cache = splice_block(engine.shared_model, cache, state, checkpoint_sha256=state.checkpoint_sha256,
                             tokenizer_fingerprint=state.tokenizer_fingerprint, embedding_weight_sha256=state.embedding_weight_sha256)
        reference = engine.shared_model.logits(engine._native_forward([roles.consume_close] + suffix, cache)['h_final'][0]).float().numpy()
        plain = prefix + encode(text) + suffix
        plain_reference = engine.shared_model.logits(engine._native_forward(plain, None)['h_final'][0]).float().numpy()
    for name, value in (('state', state), ('state-text', block)):
        export_state_gguf(value, S / f'{name}.gguf', state_id='nzs1_' + name, text_sha256='0' * 64, role_tokens=roles)
    (S / 'seq-state.json').write_text(json.dumps(prefix + [str(S / 'state.gguf')] + suffix))
    (S / 'seq-text.json').write_text(json.dumps(prefix + [str(S / 'state-text.gguf')] + suffix))
    (S / 'seq-plain.json').write_text(json.dumps(plain))
    start = len(prefix) + 1 + len(block.token_ids)

    def run(sequence, first, out, *extra):
        subprocess.run([str(LCPP / 'build-cpu/bin/llama-state-span-dump'), '-m', str(model_gguf), '--input', str(S / sequence),
                        '--from', str(first), '--out', str(S / out), '--f32-cache', *extra], check=True, capture_output=True)
        return np.fromfile(S / out, dtype=np.float32)

    def compare(name, got, want):
        got = got.reshape(want.shape[0], -1)
        width = min(got.shape[1], want.shape[1])
        delta = np.abs(got[:, :width] - want[:, :width])
        top = (got[:, :width].argmax(1) == want[:, :width].argmax(1)).mean()
        print(f'{name}: rows {want.shape[0]} max|dlogit| {delta.max():.3e} mean {delta.mean():.3e} top1 {top:.3f}')

    compare('plain text (backend baseline)', run('seq-plain.json', 0, 'plain.f32'), plain_reference)
    compare('state span, one batch', run('seq-state.json', start, 'state.f32'), reference)
    compare('state span, chunks of 7', run('seq-state.json', start, 'state7.f32', '--chunk', '7'), reference)
    compare('control: unperturbed state', run('seq-text.json', start, 'control.f32'), reference)


if __name__ == '__main__':
    main()
