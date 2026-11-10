#!/usr/bin/env python3
"""Queue the verified successful student episodes of a controller's already scored rounds for its next SFT window.

    python3 scripts/neuralese_data/backfill_rollout_distillation.py CONTROL_OUT [--current-round N]

The controller's hook (``rl/rollout_distillation.py``) tops up earlier rounds itself at every scored round; this entry
does it now, for a controller that has not yet run the hook's current code (rounds scored before the 2026-11-10
transition, or by the one-episode-per-task form). It reads the controller's ``config.json`` and ``state.json`` and
writes only packages under ``CONTROL_OUT/memetic/distill`` and one ``rollout-distillation-all.json`` record per round
(the same artefacts the hook writes); a round that already has its record is skipped, so it is safe to repeat and safe
beside a running controller (packages are append-only and a round is queued once). Run it through the memory ledger:
lowering loads captured rollouts, gigabytes per round.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))

from natlang_neuralese.rl import rollout_distillation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('out', type=Path)
    parser.add_argument('--current-round', type=int, help='backfill rounds before this one (default: the state\'s round)')
    args = parser.parse_args()
    config = json.loads((args.out / 'config.json').read_text())
    state = json.loads((args.out / 'state.json').read_text())
    current = state.get('round', 0) if args.current_round is None else args.current_round
    records = rollout_distillation.backfill(config, state, current_round=current)
    print(json.dumps([{key: value for key, value in record.items() if key != 'package'} for record in records if record],
                     indent=1))


if __name__ == '__main__':
    main()
