#!/usr/bin/env python3
"""Keep one resumable training run alive on a Colab session, from the DGX, across evictions and session replacements.

A compiler run (T1 OPSD, T2 site compiler) is an admission tenant on a Colab VM that saves its own resumable state in
its run directory. This keeper, a DGX process, makes that state survive the VM:

- every ``--sync-every`` seconds (default 15 min) it mirrors the run directory to ``--mirror`` on the DGX: a hardlink
  snapshot of the directory on the VM (atomically replaced files keep their old inode, so the snapshot holds whole
  files) is rsynced into the mirror; ``--drive-file`` names (a large ``learner-state.pt``) go only to Google Drive
  (the VM uploads inside Google's network, each one when it changed) and the mirror keeps a stamp of what Drive holds,
  not the bytes; the mirror belongs on the HDD, never on the DGX root;
- when the run is not running it (re)starts it detached under ``scripts/colab/admission.py run`` (``--preemptible``
  for a tenant that yields to aei1), after ``--ship`` inputs are present, ``--setup`` ran, and, on a VM without the
  run's state (a replaced session), the newest mirrored state is restored (Drive files fetched on the VM);
- an attempt that ends without a Python traceback in its log (eviction, admission timeout, a lost session) is simply
  restarted; ``--max-failures`` attempts ending in a traceback stop the keeper;
- when ``--done`` exists in the run directory it syncs once more and exits 0.

    remote_run_keeper.py --host colab-g4b --run-dir "$REMOTE_RUN" \\
        --mirror "$LOCAL_MIRROR" --launch "$REMOTE_LAUNCH_SCRIPT" \\
        --label cmp-t1-opsd-v2 --vram-gib 20 --host-ram-gib 24 --done receipt.json \\
        --ship LOCAL=REMOTE ... --setup 'cd ... && git checkout ...'

Set the example variables to the selected session's run and launch paths and the
local durable mirror; the keeper does not require a particular checkout or data root.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO)]

import pop_rl_actor as P  # noqa: E402

q = shlex.quote


def log(**event):
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), **event}, default=str), flush=True)


def remote(host, command, timeout=600, check=True):
    return P.ssh_host(host, command, timeout=timeout, check=check)


def probe(args):
    """(boot id, 'running' | 'rc N' | 'none', done?) of the run on the VM."""
    run = args.run_dir
    out = remote(args.host, f'cat {P.BOOT_ID}; '
                            f'if [ -f {q(run)}/.keeper.rc ]; then echo "rc $(cat {q(run)}/.keeper.rc)"; '
                            f'elif [ -f {q(run)}/.keeper.pid ] && kill -0 "$(cat {q(run)}/.keeper.pid)" 2>/dev/null; '
                            f'then echo running; else echo none; fi; '
                            f'test -f {q(run)}/{q(args.done)} && echo done || echo open', timeout=60)
    boot, state, done = (out.strip().splitlines() + ['', 'none', 'open'])[:3]
    return boot.strip(), state.strip(), done.strip() == 'done'


def ship(args):
    """Every --ship LOCAL=REMOTE present on the VM with LOCAL's bytes (sha256; through Drive when enabled)."""
    cfg = P.host_config('colab', host=args.host)
    for pair in args.ship or ():
        local, remote_path = pair.split('=', 1)
        local = Path(local)
        if local.is_dir():
            P.sync_to_remote(f'{local}/', f'{remote_path}/', args.host)
            continue
        size = local.stat().st_size
        out = remote(args.host, f'stat -c %s {q(remote_path)} 2>/dev/null || echo absent', timeout=60).strip()
        if out == str(size):
            continue
        from natlang_neuralese.common.hashing import sha256_file_hex
        how = P.ship_file(str(local), remote_path, args.host, cfg, sha256_file_hex(local))
        log(event='shipped', file=str(local), to=remote_path, via=how)


def drive_key(args, name):
    return f'compiler/{args.label}/{name}'


def restore(args):
    """A VM without the run directory's state gets the mirror's newest copy (the replaced-session case)."""
    mirror = Path(args.mirror)
    if not mirror.exists() or not any(mirror.iterdir()):
        return False
    present = remote(args.host, f'test -e {q(args.run_dir)}/{q(args.state)} && echo yes || echo no', timeout=60).strip()
    if present == 'yes':
        return False
    remote(args.host, f'mkdir -p {q(args.run_dir)}')
    _rsync(['--exclude=.keeper*', *[f'--exclude={name}' for name in args.drive_file or ()],
            str(mirror) + '/', f'{args.host}:{args.run_dir}/'])
    cfg = P.host_config('colab', host=args.host)
    for name in args.drive_file or ():
        if (mirror / f'.{name}.sig').exists():           # the mirror records it; Drive holds the bytes
            got = P.drive_vm(args.host, cfg, 'get', drive_key(args, name), f'{args.run_dir}/{name}')
            if got.get('event') != 'got':
                raise RuntimeError(f'restore: {drive_key(args, name)} not on Drive ({got})')
    remote(args.host, f'rm -f {q(args.run_dir)}/.keeper.rc {q(args.run_dir)}/.keeper.pid')
    log(event='state restored on a VM without it', host=args.host, run=args.run_dir, mirror=str(mirror))
    return True


def _rsync(argv):
    """Compressed, resumable rsync (never in place: hardlinked mirror files are replaced, not rewritten)."""
    import subprocess
    subprocess.run(['rsync', '-a', '--partial', '-z', '--timeout=300', *argv], check=True, timeout=6 * 3600)


def start(args):
    run = args.run_dir
    admission = f'{P.COLAB_ROOT}/natlang/scripts/colab/admission.py'
    claim = ['python3', admission, 'run', '--vram-gib', str(args.vram_gib), '--host-ram-gib', str(args.host_ram_gib),
             '--label', args.label, '--wait', str(args.admission_wait), *(['--preemptible'] if args.preemptible else []),
             '--', args.launch]
    log_file = f'{run}/.keeper-attempts.log'
    inner = (f'echo "=== attempt $(date -Is)" >> {q(log_file)}; {shlex.join(claim)} >> {q(log_file)} 2>&1; '
             f'echo $? > {q(run)}/.keeper.rc.tmp && mv {q(run)}/.keeper.rc.tmp {q(run)}/.keeper.rc')
    # separate statements: `a && setsid ... &` backgrounds the whole list in a subshell that holds the SSH channel
    # open until the run ends (pop_rl_actor.run_detached)
    remote(args.host, f'mkdir -p {q(run)}; rm -f {q(run)}/.keeper.rc; '
                      f'setsid nohup bash -c {q(inner)} > /dev/null 2>&1 < /dev/null & echo $! > {q(run)}/.keeper.pid')
    log(event='attempt started', host=args.host, run=run)


def attempt_failed(args):
    """Whether the last attempt ended in a Python traceback (a real failure, not an eviction or a timeout)."""
    out = remote(args.host, f"awk '/^=== attempt/{{buf=\"\"}} {{buf=buf $0 \"\\n\"}} END{{printf \"%s\", buf}}' "
                            f"{q(args.run_dir)}/.keeper-attempts.log | grep -c Traceback || true", timeout=60).strip()
    return out not in ('', '0')


def sync(args):
    """Mirror the run directory: a hardlink snapshot on the VM, rsynced; --drive-file names through Drive."""
    run, snap = args.run_dir, args.run_dir.rstrip('/') + '.keeper-snap'
    remote(args.host, f'rm -rf {q(snap)} && cp -al {q(run)} {q(snap)}', timeout=300)
    mirror = Path(args.mirror)
    staged = mirror.with_name(mirror.name + '.pull')
    if mirror.exists() and not staged.exists():
        shutil.copytree(mirror, staged, copy_function=os.link)      # rsync updates the staged copy incrementally
    staged.mkdir(parents=True, exist_ok=True)
    _rsync(['--delete', '--exclude=.keeper*', *[f'--exclude={name}' for name in args.drive_file or ()],
            *[f'--exclude=.{name}.sig' for name in args.drive_file or ()], f'{args.host}:{snap}/', f'{staged}/'])
    cfg = P.host_config('colab', host=args.host)
    for name in args.drive_file or ():
        sig = remote(args.host, f'stat -c "%s %Y" {q(snap)}/{q(name)} 2>/dev/null || echo absent', timeout=60).strip()
        if sig == 'absent':
            continue
        stamp = mirror / f'.{name}.sig'
        if stamp.exists() and stamp.read_text() == sig:
            continue                                        # unchanged since the last put
        started = time.time()
        put = P.drive_vm(args.host, cfg, 'put', f'{snap}/{name}', drive_key(args, name), timeout=4 * 3600)
        if put.get('event') != 'put':
            raise RuntimeError(f'{drive_key(args, name)}: Drive put failed ({put})')
        # Drive holds the bytes (a verified upload replaces the older copy); the DGX mirror keeps only the stamp, so a
        # 14-17 GB state rewritten every sync never lands on a DGX disk (2026-11-26: root fell to 9 GB)
        (staged / f'.{name}.sig').write_text(sig)
        log(event='drive file synced', name=name, seconds=round(time.time() - started), put=put.get('seconds'))
    remote(args.host, f'rm -rf {q(snap)}', timeout=300)
    old = mirror.with_name(mirror.name + '.old')
    shutil.rmtree(old, ignore_errors=True)
    if mirror.exists():
        os.replace(mirror, old)
    os.replace(staged, mirror)
    shutil.rmtree(old, ignore_errors=True)
    (mirror.parent / f'{mirror.name}.synced.json').write_text(json.dumps({'at': time.time(), 'host': args.host,
                                                                          'run': run}))
    log(event='synced', mirror=str(mirror))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--host', required=True)
    p.add_argument('--run-dir', required=True, help='the run directory on the VM')
    p.add_argument('--mirror', required=True, help='its mirror on the DGX')
    p.add_argument('--state', required=True, help='the resumable state inside the run directory (file or directory)')
    p.add_argument('--done', required=True, help='a file in the run directory that marks the run finished')
    p.add_argument('--launch', required=True, help='the VM script that runs (and resumes) the run')
    p.add_argument('--label', required=True)
    p.add_argument('--vram-gib', type=float, required=True)
    p.add_argument('--host-ram-gib', type=float, required=True)
    p.add_argument('--preemptible', action='store_true')
    p.add_argument('--admission-wait', type=int, default=6 * 3600)
    p.add_argument('--ship', action='append', help='LOCAL=REMOTE inputs a fresh VM needs')
    p.add_argument('--setup', help='VM shell command run before each start (idempotent: checkout, env)')
    p.add_argument('--drive-file', action='append', help='run-directory file moved through Drive (large state)')
    p.add_argument('--sync-every', type=int, default=900)
    p.add_argument('--poll', type=int, default=60)
    p.add_argument('--max-failures', type=int, default=3)
    p.add_argument('--allow-root-mirror', action='store_true', help='permit a mirror on the root filesystem')
    args = p.parse_args(argv)
    anchor = Path(args.mirror)
    while not anchor.exists():
        anchor = anchor.parent
    if anchor.stat().st_dev == Path('/').stat().st_dev and not args.allow_root_mirror:
        raise SystemExit(f'{args.mirror} is on the root filesystem: mirrors live on the HDD')
    failures, last_sync, boot = 0, 0.0, None
    while True:
        try:
            now_boot, state, done = probe(args)
        except Exception as error:                          # unreachable (a session being replaced): keep trying
            log(event='unreachable', host=args.host, error=repr(error)[:200])
            time.sleep(args.poll)
            continue
        if boot and now_boot != boot:
            log(event='another VM answers', host=args.host, boot=now_boot[:8], was=boot[:8])
        boot = now_boot
        try:
            if done:
                sync(args)
                log(event='run done', run=args.run_dir)
                return 0
            if state != 'running':
                if state.startswith('rc '):
                    if attempt_failed(args):
                        failures += 1
                        log(event='attempt failed', failures=failures, rc=state)
                        if failures >= args.max_failures:
                            sync(args)
                            log(event='stopping: repeated failures', failures=failures)
                            return 1
                    else:
                        log(event='attempt ended without a traceback (eviction, admission timeout)', rc=state)
                ship(args)
                # a fresh Drive token at ROOT/drive-token, for setup commands that fetch inputs from Drive
                P.drive_token_to(args.host, P.host_config('colab', host=args.host))
                if args.setup:
                    remote(args.host, args.setup, timeout=1800)
                restore(args)
                start(args)
            if time.time() - last_sync >= args.sync_every:
                has_state = remote(args.host, f'test -e {q(args.run_dir)}/{q(args.state)} && echo yes || echo no',
                                   timeout=60).strip() == 'yes'
                if has_state:
                    sync(args)
                    last_sync = time.time()
        except Exception as error:
            log(event='keeper step failed', error=repr(error)[:500])
        time.sleep(args.poll)


if __name__ == '__main__':
    sys.exit(main())
