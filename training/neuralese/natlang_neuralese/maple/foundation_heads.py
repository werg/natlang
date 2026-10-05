"""Untrained Maple port heads for the shared foundation recipe (plans/neuralese/TRAINING_RECIPE.md).

The recipe's stages take a heads checkpoint: it names the frozen backbone (here the Maple student, maple/student.py)
and the port cutoff. A Maple lineage starts from fresh heads at a chosen cutoff, in the trainer's checkpoint layout,
so ``serve.load_engine`` rebuilds exactly this backbone (its student state is pinned by sha256).

    python -m natlang_neuralese.maple.foundation_heads --state NESTED_STATE.pt --cutoff 12 --out heads.pt
"""
import argparse
from pathlib import Path

import torch

from ..model.heads import PortHeads
from ..model.hf_port import qwen_controls
from .maple_port import MaplePortBackbone
from .student import DEFAULT_MODEL, load_student, student_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--state', help='nested-family student state (nested-state.pt); omit for published Maple')
    parser.add_argument('--cutoff', type=int, required=True)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--stop-source', default='final')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('fresh output required')
    state = str(Path(args.state).resolve()) if args.state else None
    model, tokenizer = load_student(args.model, state, device=args.device)
    torch.manual_seed(args.seed)
    backbone = MaplePortBackbone(model, qwen_controls(tokenizer))
    if not 0 < args.cutoff < backbone.num_layers:
        parser.error(f'cutoff must lie inside the {backbone.num_layers}-layer stack')
    heads = PortHeads(backbone, cutoff=args.cutoff, max_length=args.max_length, stop_source=args.stop_source)
    checkpoint = {
        'port_config': {'cutoff': heads.cutoff, 'max_length': heads.max_length, **heads.port_config()},
        'heads': heads.state_dict(),
        'control_rows': backbone.control_rows.detach().cpu(),
        'backbone': student_identity(args.model, state),
        'note': 'untrained Maple port heads for the foundation recipe; not a trained port',
    }
    if not backbone.tied:
        checkpoint['control_head_rows'] = backbone.control_head_rows.detach().cpu()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    print({'out': str(args.out), 'cutoff': heads.cutoff, 'layers': backbone.num_layers, 'tied': backbone.tied})


if __name__ == '__main__':
    main()
