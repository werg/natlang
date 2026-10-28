#!/usr/bin/env python3
"""Base parity of the native LFM3 GGUF export (export/native_gguf.py) against Liquid's official BF16 GGUF.

Builds the native model of the pinned LFM3 base (learned role/reasoning rows at their initialization), writes it as a
model-only native checkpoint, exports it to BF16 GGUF with the pinned fork's converter snapshot, and compares every
tensor with the official ``LiquidAI/LFM3-1.7B-GGUF`` BF16 file. Expected: identical tensors except the embedding
(and tied output) rows of the native role markers and reasoning-boundary overrides, which the native model owns;
natlang's extra tensors (producer adapter) are reported, not compared. Equal tensors give equal logits for every
ordinary token; a mismatch anywhere else is a converter or export bug.

  check_lfm3_gguf_base_parity.py --base SNAPSHOT --official GGUF --llama-cpp CONVERTER --work DIR --out REPORT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))


def tensors(path, llama_cpp):
    sys.path.insert(0, str(Path(llama_cpp) / 'gguf-py'))
    from gguf import GGUFReader
    reader = GGUFReader(str(path))
    return {t.name: (t.tensor_type.name, np.asarray(t.data)) for t in reader.tensors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--llama-cpp', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    from natlang_neuralese.export.native_gguf import export
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model
    from natlang_neuralese.train.weight_ema import write_weights_checkpoint

    model, tokenizer, roles, config = load_native_model(str(args.base), device='cpu', dtype=torch.bfloat16)
    checkpoint_dir = args.work / 'base-native'
    write_weights_checkpoint(model, config, CHECKPOINT_SCHEMA, 0,
                             {'source': 'unmodified LFM3 base (GGUF base parity)'}, checkpoint_dir)
    owned = sorted({int(i) for i in config['marker_ids']} | {int(i) for i in config.get('token_override_ids', [])})
    del model
    manifest = export(checkpoint_dir / 'checkpoint.pt', args.work / 'gguf', outtypes=('bf16',),
                      llama_cpp=args.llama_cpp.resolve())
    ours = tensors(args.work / 'gguf' / 'model-bf16.gguf', args.llama_cpp)
    theirs = tensors(args.official, args.llama_cpp)
    report = {'base': str(args.base), 'official': str(args.official), 'native_owned_rows': owned,
              'only_ours': sorted(set(ours) - set(theirs)), 'only_official': sorted(set(theirs) - set(ours)),
              'identical': 0, 'differ': {}}
    for name in sorted(set(ours) & set(theirs)):
        (kind_a, a), (kind_b, b) = ours[name], theirs[name]
        if kind_a != kind_b or a.shape != b.shape:
            report['differ'][name] = {'types': [kind_a, kind_b], 'shapes': [list(a.shape), list(b.shape)]}
            continue
        if np.array_equal(a.view(np.uint8), b.view(np.uint8)):
            report['identical'] += 1
            continue
        rows = sorted({int(r) for r in np.nonzero((a != b).reshape(a.shape[0], -1).any(axis=1))[0]}) if a.ndim >= 2 else []
        report['differ'][name] = {'rows': rows[:64], 'row_count': len(rows),
                                  'only_native_owned_rows': bool(rows) and set(rows) <= set(owned)}
    report['passed'] = (not report['only_official'] and
                        all(v.get('only_native_owned_rows') for v in report['differ'].values()))
    report['gguf_manifest'] = {k: manifest[k] for k in ('files', 'role_markers', 'merged_rows')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'identical', 'only_ours', 'only_official')} |
                     {'differ': {k: v.get('row_count', v) for k, v in report['differ'].items()}}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
