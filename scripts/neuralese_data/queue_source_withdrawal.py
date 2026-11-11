#!/usr/bin/env python3
"""Queue a source withdrawal plan for the interleaved controller's next SFT window boundary.

The designed way to take rows out of the live joint SFT source (``rl/source_withdrawal.py``): one immutable plan
(``natlang.source-withdrawal-plan/1``: row ids with their family, criterion and optional binding, optional visit caps,
reason, evidence files) is appended to ``<controller out>/memetic/distill/withdrawal-queue.jsonl``. At its next window
boundary the controller builds the static source correction against the window's actual parent checkpoint (filtered leaves,
family map, CE policy, packed geometry, successor pending focus, recipes) and appends the queued distillation packages on
top in the same change; originals stay untouched and an earned row is never withdrawn. A plan is valid for any parent
checkpoint, so it can be queued at any time. The queue is a file of its own: a controller without this code never reads it.

  queue_source_withdrawal.py --controller-out OUT PLAN.json [--dry-run]

``prepare_template_excision.py audit --queue`` writes and queues the plan of the scripted-template families in one step.
This tool changes no recipe, config or controller state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT))

from natlang_neuralese.rl import source_withdrawal  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('plan')
    ap.add_argument('--controller-out', required=True)
    ap.add_argument('--dry-run', action='store_true', help='validate the plan and print what would be queued')
    args = ap.parse_args(argv)
    plan_path = Path(args.plan).resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    source_withdrawal.validate_plan(plan)
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'plan': str(plan_path), 'rows': len(plan['withdraw']),
                          'visit_caps': len(plan.get('visit_cap') or {}), 'reason': plan['reason']}))
        return 0
    source_withdrawal.queue_plan(args.controller_out, plan_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
