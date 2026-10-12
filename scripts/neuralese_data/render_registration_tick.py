#!/usr/bin/env python3
"""Timer tick: register finished native-turn renders and answer their coord requests.

Queue file ~/.local/state/natlang/pending-renders.json: a list of
{id, render_dir, source_corpus, request_id?, notes?}. For each entry whose render_dir/receipt.json exists, run
register_native_turn_render.py --via-worktree (idempotent: it replaces the entry of the same id), then, when
request_id is set, close that coord request as dgx-render with row/token counts per rendering and tokenizer and the
omissions, and drop the entry. A failed entry stays queued and is retried on the next tick. A tick holds a lock, so
overlapping runs are harmless. Installed as natlang-render-registration.{service,timer} (OnCalendar=*:05,35,
Persistent=true; templates in scripts/systemd/). Queue an entry with: render_registration_tick.py add ID DIR SOURCE [REQUEST_ID]
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUEUE = Path(os.environ.get('RENDER_QUEUE', Path.home() / '.local/state/natlang/pending-renders.json'))


def load():
    return json.loads(QUEUE.read_text()) if QUEUE.exists() else []


def save(entries):
    tmp = QUEUE.with_suffix('.tmp')
    tmp.write_text(json.dumps(entries, indent=2) + '\n')
    tmp.replace(QUEUE)


def summary(receipt):
    lines = []
    for key, r in sorted(receipt['renders'].items()):
        sup = sum(s.get('supervised_tokens', 0) for s in r['by_split'].values())
        tok = sum(s.get('tokens', 0) for s in r['by_split'].values())
        lines.append(f"{key}: {r['documents']} documents, {sup} supervised tokens, {tok} tokens")
    om = {k: v for k, v in receipt.get('render_omissions', {}).items() if v}
    lines.append('selection: ' + json.dumps(receipt.get('selection')))
    lines.append('omissions (nonzero): ' + (json.dumps(om) if om else 'none'))
    return '\n'.join(lines)


def tick():
    entries = load()
    keep = []
    for e in entries:
        d = Path(e['render_dir'])
        if not (d / 'receipt.json').exists():
            keep.append(e)
            continue
        cmd = [sys.executable, str(ROOT / 'scripts/neuralese_data/register_native_turn_render.py'), '--id', e['id'],
               '--render-dir', str(d), '--source-corpus', e['source_corpus'], '--via-worktree']
        if e.get('notes'):
            cmd += ['--notes', e['notes']]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(f"register failed for {e['id']}: {r.stderr[-800:]}", file=sys.stderr)
            keep.append(e)
            continue
        print('registered', e['id'])
        if e.get('request_id'):
            msg = f"Rendered and registered {e['id']} (source {e['source_corpus']}).\n" + summary(json.loads((d / 'receipt.json').read_text()))
            c = subprocess.run([sys.executable, 'scripts/coord.py', '--as', 'dgx-render', 'close', '-m', msg, e['request_id']],
                               cwd=ROOT, capture_output=True, text=True)
            if c.returncode:
                print('coord close failed:', c.stderr[-400:], file=sys.stderr)
    save(keep)


def main():
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'add':
        a = sys.argv[2:]
        entry = {'id': a[0], 'render_dir': str(Path(a[1]).absolute()), 'source_corpus': a[2]}
        if len(a) > 3:
            entry['request_id'] = a[3]
        with open(QUEUE.with_suffix('.lock'), 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            save([x for x in load() if x['id'] != entry['id']] + [entry])
        return
    with open(QUEUE.with_suffix('.lock'), 'w') as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        tick()


if __name__ == '__main__':
    main()
