#!/usr/bin/env python3
"""Block until something worth attention happens, print it, exit: a gapless watch for an agent's background command.

Sources are log files (``--file PATH=REGEX``) and systemd user units (``--unit NAME=REGEX``). Each source keeps a cursor
in ``--state`` (a byte offset per file, a journal cursor per unit), so the next invocation resumes exactly where the
last one stopped: no event falls between two watches, however long the gap. The first run of a source starts at its
current end. Unit sources also fire when the unit stops being active. Exits 0 after printing the matched lines (all
matches found in one poll), or 0 with ``--timeout`` seconds elapsed and nothing found (prints "quiet").

  watch_events.py --state ~/.cache/natlang-watch/run.json \\
      --file "$W/train.jsonl="'"groups"|grad_spike": true|held eval' --unit grpo-v3-run1-r1='Traceback|oom|Killed'
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def journal(unit, cursor):
    command = ['journalctl', '--user', '-u', unit, '-o', 'json', '--no-pager']
    command += ['--after-cursor', cursor] if cursor else ['-n', '0', '--show-cursor']
    out = subprocess.run(command, capture_output=True, text=True).stdout
    lines, last = [], cursor
    for raw in out.splitlines():
        if raw.startswith('-- cursor: '):
            last = raw[len('-- cursor: '):].strip()
            continue
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        last = record.get('__CURSOR', last)
        message = record.get('MESSAGE')
        if isinstance(message, str):
            lines.append(message)
    return lines, last


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '-q', unit]).returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--state', required=True, help='JSON file holding each source cursor')
    p.add_argument('--file', action='append', default=[], metavar='PATH=REGEX')
    p.add_argument('--unit', action='append', default=[], metavar='NAME=REGEX')
    p.add_argument('--interval', type=float, default=10)
    p.add_argument('--timeout', type=float, default=0, help='seconds; 0 waits for an event indefinitely')
    p.add_argument('--width', type=int, default=400, help='print at most this many characters per line')
    a = p.parse_args()
    state_path = Path(a.state).expanduser()
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    files = [(path, re.compile(rx)) for path, rx in (s.split('=', 1) for s in a.file)]
    units = [(name, re.compile(rx)) for name, rx in (s.split('=', 1) for s in a.unit)]
    for path, _ in files:   # a new source starts at its current end
        state.setdefault('file:' + path, os.path.getsize(path) if os.path.exists(path) else 0)
    for name, _ in units:
        if 'unit:' + name not in state:
            state['unit:' + name] = journal(name, None)[1]
    started = time.time()
    while True:
        found = []
        for path, rx in files:
            key = 'file:' + path
            if not os.path.exists(path):
                continue
            size = os.path.getsize(path)
            if size < state[key]:   # truncated or replaced: read it from the start
                state[key] = 0
            with open(path, 'rb') as handle:
                handle.seek(state[key])
                chunk = handle.read()
            complete = chunk[:chunk.rfind(b'\n') + 1]   # never a half-written line
            state[key] += len(complete)
            found += [f'{Path(path).name}: {line[:a.width]}' for line in complete.decode(errors='replace').splitlines()
                      if rx.search(line)]
        for name, rx in units:
            key = 'unit:' + name
            lines, state[key] = journal(name, state[key])
            found += [f'{name}: {line[:a.width]}' for line in lines if rx.search(line)]
            if not active(name) and not state.get('inactive:' + name):
                state['inactive:' + name] = True
                found.append(f'{name}: no longer active')
            elif active(name):
                state.pop('inactive:' + name, None)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state))
        os.replace(tmp, state_path)
        if found:
            print('\n'.join(found), flush=True)
            return
        if a.timeout and time.time() - started > a.timeout:
            print('quiet', flush=True)
            return
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
