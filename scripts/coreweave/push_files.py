#!/usr/bin/env python3
"""Push local files into a running sandbox (file API, streaming) and verify their sha256 there.

For data appended to a long-lived sandbox (hold.sh) after its launch bundle: a new mixture version, an args file.
Each SOURCE is written to the same repo-relative path under the sandbox checkout (bootstrap NATLANG_REPO), or to DEST
when given as SOURCE=DEST; directories are pushed recursively. Arrival is verified by hashing in the sandbox.

  push_files.py SANDBOX_ID SOURCE[=DEST] ...
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwctl  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
# The sandbox checkout: bootstrap.sh's NATLANG_REPO default (one declaration).
REMOTE_REPO = re.search(r'NATLANG_REPO=\$\{NATLANG_REPO:-([^}]+)\}',
                        (Path(__file__).resolve().parent / 'bootstrap.sh').read_text()).group(1)


def plan(specs):
    items = []
    for spec in specs:
        source, _, dest = spec.partition('=')
        source = Path(source)
        files = [source] if source.is_file() else sorted(p for p in source.rglob('*') if p.is_file())
        for path in files:
            if dest:
                target = dest if source.is_file() else f'{dest.rstrip("/")}/{path.relative_to(source)}'
            else:
                target = f'{REMOTE_REPO}/{path.resolve().relative_to(REPO)}'
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            items.append({'source': str(path), 'target': target, 'sha256': digest, 'bytes': path.stat().st_size})
    return items


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    cwctl.load_env_file()
    sandbox = cwctl.read_sandbox(argv[0])
    items = plan(argv[1:])
    parents = sorted({str(Path(i['target']).parent) for i in items})
    sandbox.exec(['mkdir', '-p', *parents]).result()
    for item in items:
        started = time.time()
        sandbox.write_file_streaming(item['target'], cwctl._chunks(Path(item['source'])), timeout_seconds=7200).result()
        print(json.dumps({'event': 'pushed', 'target': item['target'], 'bytes': item['bytes'],
                          'seconds': round(time.time() - started, 1)}), flush=True)
    check = ('import hashlib,json,sys\nbad=[]\nfor t,s in json.loads(sys.argv[1]):\n'
             ' d=hashlib.file_digest(open(t,"rb"),"sha256").hexdigest()\n bad+=[t] if d!=s else []\n'
             'print(json.dumps({"verified":len(json.loads(sys.argv[1]))-len(bad),"mismatch":bad}))')
    result = sandbox.exec(['python3', '-c', check, json.dumps([[i['target'], i['sha256']] for i in items])]).result()
    print(result.stdout.strip() if hasattr(result, 'stdout') else result)
    return 0


if __name__ == '__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):
        cwctl.reexec_in_venv()
    sys.exit(main(sys.argv[1:]))
