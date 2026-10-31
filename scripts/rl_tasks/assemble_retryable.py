#!/usr/bin/env python3
"""Retry-safe wrapper for immutable broad-task assemblies.

Each execution writes to a fresh output directory. A ledger restart can rerun
the bounded assembly from immutable source builds without colliding with a
partially written directory from an interrupted attempt. The ready pointer is
published only after assemble.py exits successfully.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit('usage: assemble_retryable.py OUT_PARENT/PREFIX [--copy-assets-v3 DIR] [assemble.py arguments excluding --out]')
    prefix = Path(argv.pop(0))
    assets_v3 = None
    if '--copy-assets-v3' in argv:
        index = argv.index('--copy-assets-v3')
        if index + 1 >= len(argv):
            raise SystemExit('--copy-assets-v3 needs a directory')
        assets_v3 = Path(argv[index + 1])
        del argv[index:index + 2]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(f'{prefix}-{stamp}-{__import__("os").getpid()}')
    cmd = [sys.executable, '-I', str(Path(__file__).with_name('assemble.py')), *argv, '--out', str(out)]
    subprocess.run(cmd, check=True)
    asset_proof = None
    if assets_v3 is not None:
        if not assets_v3.is_dir():
            raise SystemExit(f'assets-v3 is not readable: {assets_v3}')
        target = out / 'assets-v3'
        shutil.copytree(assets_v3, target)
        files = {str(path.relative_to(target)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(target.rglob('*')) if path.is_file()}
        proof = {'schema': 'natlang.assets-v3-copy/1', 'source': str(assets_v3), 'files': files,
                 'file_count': len(files)}
        proof_path = out / 'assets-v3.manifest.json'
        proof_path.write_text(json.dumps(proof, indent=2) + '\n')
        asset_proof = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    ready = prefix.with_suffix(prefix.suffix + '.ready.json')
    ready.write_text(json.dumps({'schema': 'natlang.retryable-assembly-ready/1', 'output': str(out),
                                 'qualification_sha256': __import__('hashlib').sha256(
                                     (out / 'qualification.json').read_bytes()).hexdigest(),
                                 'assets_v3_manifest_sha256': asset_proof}, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
