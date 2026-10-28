#!/usr/bin/env python3
"""Private full-state recovery copy of a running sandbox's training state on this machine (AGENTS.md recovery rules).

    recovery_pull.py file SANDBOX REMOTE_FILE DEST_DIR --owner OWNER [--keep 2]
        one file (e.g. a learner's rolling full-state checkpoint.pt, plus its checkpoint-policy.json when present)
    recovery_pull.py tree SANDBOX REMOTE_DIR DEST_DIR --owner OWNER [--keep 2]
        a directory (e.g. a DiLoCo coordinator root), archived in the sandbox as one uncompressed tar first

Before any byte moves, an immutable intent is written under DEST_DIR/intents/: the remote path, its sha256 and size
measured in the sandbox, the training step if the file is a native checkpoint (read in the sandbox from its policy
sidecar), the sandbox id, the owner and the time. An upload or download can finish just after its sandbox
disappears, so the intent is what identifies the object. The data is downloaded as 256 MB parts with retries (one
streamed read of several GB broke on a gateway teardown), reassembled, verified against the intent's sha256 and only
then renamed into DEST_DIR/<stamp>/. The latest verified copy is DEST_DIR/latest (a symlink). Older verified copies
beyond --keep are pruned only after the new one is verified. Sandbox→DGX measured ~9-11 MB/s on 2026-10-28.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cwctl  # noqa: E402

PART = 256 << 20


def run(sandbox, script: str) -> str:
    result = sandbox.exec(['bash', '-c', script]).result()
    if result.returncode != 0:
        raise SystemExit(f'sandbox command failed ({result.returncode}): {result.stderr[-500:]}')
    return result.stdout


def download(sandbox_id: str, remote: str, local: Path, sha256: str, size: int) -> float:
    sandbox = cwctl.read_sandbox(sandbox_id)
    parts = run(sandbox, f'set -e; d="{remote}.parts"; rm -rf "$d"; mkdir -p "$d"; '
                         f'split -b {PART} -d -a 4 "{remote}" "$d/p"; ls "$d"').split()
    started, partial = time.time(), local.with_name(local.name + '.partial')
    try:
        with partial.open('wb') as out:
            for part in parts:
                for attempt in range(8):
                    try:
                        buffer = bytearray()
                        for block in cwctl.read_sandbox(sandbox_id).read_file_streaming(f'{remote}.parts/{part}',
                                                                                         timeout_seconds=1800):
                            buffer += block
                        expected = int(run(sandbox, f'stat -c %s "{remote}.parts/{part}"'))
                        if len(buffer) != expected:
                            raise OSError(f'{part}: {len(buffer)} of {expected} bytes')
                        out.write(buffer)
                        break
                    except Exception as error:  # transient gateway/stream failures: retry this part only
                        print(json.dumps({'retry': part, 'attempt': attempt, 'error': str(error)[:200]}), flush=True)
                        time.sleep(min(60, 5 * 2 ** attempt))
                else:
                    raise SystemExit(f'part {part} failed after retries')
            out.flush()
            os.fsync(out.fileno())
        digest = hashlib.sha256()
        with partial.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1 << 24), b''):
                digest.update(chunk)
        if digest.hexdigest() != sha256 or partial.stat().st_size != size:
            raise SystemExit(f'{local}: downloaded sha256/size differ from the intent')
        partial.replace(local)
    finally:
        partial.unlink(missing_ok=True)
        try:
            run(cwctl.read_sandbox(sandbox_id), f'rm -rf "{remote}.parts"')
        except BaseException:
            pass
    return time.time() - started


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind', choices=('file', 'tree'))
    parser.add_argument('sandbox')
    parser.add_argument('remote')
    parser.add_argument('dest', type=Path)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--keep', type=int, default=2)
    args = parser.parse_args(argv)
    cwctl.load_env_file()
    sandbox = cwctl.read_sandbox(args.sandbox)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    remote = args.remote
    if args.kind == 'tree':
        remote = f'{args.remote.rstrip("/")}.recovery-{stamp}.tar'
        run(sandbox, f'set -e; tar -C "$(dirname "{args.remote}")" -cf "{remote}.tmp" "$(basename "{args.remote}")"; '
                     f'mv "{remote}.tmp" "{remote}"')
    else:
        # A snapshot copy, so a rolling checkpoint replaced mid-download cannot tear the object.
        snap = f'{remote}.recovery-{stamp}'
        run(sandbox, f'set -e; cp "{remote}" "{snap}.tmp"; mv "{snap}.tmp" "{snap}"')
        policy = Path(args.remote).with_name('checkpoint-policy.json')
        remote = snap
    measured = run(sandbox, f'sha256sum "{remote}" | cut -d" " -f1; stat -c %s "{remote}"').split()
    intent = {'schema': 'natlang.coreweave-recovery-intent/1', 'sandbox': args.sandbox, 'source': args.remote,
              'object': remote, 'sha256': measured[0], 'bytes': int(measured[1]), 'kind': args.kind,
              'owner': args.owner, 'at': stamp}
    if args.kind == 'file':
        step = run(sandbox, f'python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get(\'step\', \'\'))" '
                            f'"{policy}" 2>/dev/null || true').strip()
        intent['checkpoint_policy_step'] = step or None
    intents = args.dest / 'intents'
    intents.mkdir(parents=True, exist_ok=True)
    intent_path = intents / f'{stamp}.json'
    with intent_path.open('x') as stream:  # immutable: never overwritten
        json.dump(intent, stream, indent=1, sort_keys=True)
        stream.write('\n')
    target_dir = args.dest / stamp
    target_dir.mkdir(parents=True)
    local = target_dir / Path(args.remote.rstrip('/')).name
    if args.kind == 'tree':
        local = local.with_name(local.name + '.tar')
    try:
        seconds = download(args.sandbox, remote, local, intent['sha256'], intent['bytes'])
    finally:
        try:
            run(cwctl.read_sandbox(args.sandbox), f'rm -f "{remote}"')
        except BaseException:
            pass
    receipt = {**intent, 'local': str(local), 'verified': True, 'seconds': round(seconds, 1),
               'MBps': round(intent['bytes'] / max(seconds, 1e-9) / 1e6, 2)}
    (target_dir / 'receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    latest = args.dest / 'latest'
    tmp = args.dest / '.latest.tmp'
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(stamp)
    tmp.replace(latest)
    verified = sorted(p for p in args.dest.iterdir() if p.is_dir() and (p / 'receipt.json').exists())
    for old in verified[:-args.keep] if args.keep > 0 else []:
        shutil.rmtree(old)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == '__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):
        cwctl.reexec_in_venv()
    sys.exit(main())
