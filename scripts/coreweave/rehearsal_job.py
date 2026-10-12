#!/usr/bin/env python3
"""A tiny CPU stand-in for a training stage, for the CoreWeave loop rehearsal (rehearse.py).

It behaves like a trainer where the sandbox machinery can see it:

- it reads a bundled input (proves the fetch);
- it resumes from ``<out>/checkpoint.pt`` when one exists (proves `--resume`);
- it writes a checkpoint by atomic rename (``.pending`` then replace) each step and appends ``train.jsonl`` (proves
  the incremental publish);
- it writes ``report.json`` at the end, and exits with ``--exit-status`` (proves that a nonzero status is recorded
  rather than looping).

The checkpoint is JSON, not torch: the rehearsal tests the CoreWeave mechanism, not the trainer.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--steps', type=int, default=4)
parser.add_argument('--step-seconds', type=float, default=2.5)
parser.add_argument('--exit-status', type=int, default=0)
args = parser.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
data = args.input.read_text()
checkpoint = args.out / 'checkpoint.pt'
state = json.loads(checkpoint.read_text()) if checkpoint.is_file() else {'step': 0, 'resumed_from': None}
start = state['step']
if start:
    state['resumed_from'] = start
print(json.dumps({'event': 'start', 'step': start, 'input_bytes': len(data), 'sandbox': os.environ.get('CW_SANDBOX_ID')}),
      flush=True)
for step in range(start + 1, start + args.steps + 1):
    time.sleep(args.step_seconds)
    state['step'] = step
    pending = checkpoint.with_suffix('.pending')
    pending.write_text(json.dumps(state))
    pending.replace(checkpoint)
    with (args.out / 'train.jsonl').open('a') as log:
        log.write(json.dumps({'step': step}) + '\n')
    print(json.dumps({'event': 'checkpoint', 'step': step}), flush=True)
(args.out / 'report.json').write_text(json.dumps({'final_step': state['step'], 'resumed_from': state['resumed_from'],
                                                  'input_lines': data.count('\n')}))
sys.exit(args.exit_status)
