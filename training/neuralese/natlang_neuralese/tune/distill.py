"""Lower a tune distillation export to the student's native graph training rows (plan §5.7).

`python -m natlang_neuralese.tune.distill --export DIR --checkpoint PATH [--device cpu] [--prompt-weight 0.1]` reads
``DIR/turns.jsonl`` (verified training-split `natlang.tune-turn/v1` rows: the teacher's and the tuned stack's) and
writes ``DIR/native-graphs.jsonl`` rendered with the student's own serving template, plus ``DIR/lowered.json`` binding
the input and output hashes. Admission into a production recipe is a separate, explicit registry decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex
from . import TuneMathError
from .rows import TURN_SCHEMA, lower_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--export', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--prompt-weight', type=float, default=.1)
    args = parser.parse_args(argv)
    export = Path(args.export)
    manifest = json.loads((export / 'manifest.json').read_text())
    turns_path = export / 'turns.jsonl'
    if sha256_file_hex(turns_path) != manifest['turns']['sha256']:
        raise TuneMathError('turns.jsonl does not match its export manifest')
    rows = [json.loads(line) for line in turns_path.read_text().splitlines() if line.strip()]
    if any(row.get('schema') != TURN_SCHEMA or row.get('partition') != 'train' for row in rows):
        raise TuneMathError('a distillation export holds only training-split tune turns')
    import torch
    from ..serve import load_native_engine
    engine = load_native_engine(args.checkpoint, device=args.device, dtype=torch.float32)
    graphs = []
    # Production SFT reads bare native graphs: the tune binding is dropped (the base learns the behavior itself) and
    # the row's evidence weight scales its targets, as the tuner weighs it.
    for row in lower_rows(engine, rows, prompt_weight=args.prompt_weight):
        graph = row['graph']
        for node in graph['nodes']:
            node['target_weights'] = [weight * row['evidence_weight'] for weight in node['target_weights']]
        graphs.append(graph)
    out = export / 'native-graphs.jsonl'
    out.write_text(''.join(json.dumps(graph, sort_keys=True) + '\n' for graph in graphs))
    (export / 'lowered.json').write_text(json.dumps({'schema': 'natlang.tune-distill-lowered/v1', 'turns_sha256': manifest['turns']['sha256'],
        'native_graphs': {'path': out.name, 'sha256': sha256_file_hex(out), 'rows': len(graphs)},
        'checkpoint': args.checkpoint, 'prompt_weight': args.prompt_weight}, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
