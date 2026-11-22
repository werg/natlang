#!/usr/bin/env python3
"""Serve a policy checkpoint from the Pop laptop as an extra rollout actor.

    pop_rl_actor.py start --checkpoint CK --policy-id SHA [--port 18531] [--bind-host IP]
    pop_rl_actor.py serve-checkpoint --checkpoint CK --policy-id SHA --commit C [--step N]
    pop_rl_actor.py prefetch --checkpoint CK --policy-id SHA
    pop_rl_actor.py held-eval --spec SPEC.json
    pop_rl_actor.py stop
    pop_rl_actor.py status

start fast-forwards Pop's checkout of the canonical repository (``git pull``; Pop runs it directly), prepares the
model-only view of the checkpoint on the DGX (``prepare-view``, ``native_bundle.prepare_actor_transport`` with scope
``rl-controller-pin``) in a ledger unit, sends it as an exact delta (scripts/coreweave/send_actor_transport.py), and
launches pop_actor_driver.py under Pop's memory ledger: a container bound to Pop's tailscale address serving the
checkpoint with a block store, so a collector fetches its stored blocks over HTTP. The actor attests the training
checkpoint SHA (the transport receipt binds it) and binds ``actor_export`` in every capture (arithmetic on another GPU
differs from the learner's). stop removes the container and keeps the block store.

serve-checkpoint serves a pruned native checkpoint (``exp0_sft.write_checkpoint`` / GRPO's ``step-N/checkpoint.pt``:
already model-only BF16, so no view or transport is prepared) with the same SGLang engine and export format: the file
goes to Pop as is (rsync, sha256-checked by the driver), and the actor runs an exported copy of commit C on Pop
(``git archive``, no worktree), so a pinned experiment's held evaluations on Pop serve its own code. Pop keeps the
newest ``POP_KEEP`` such checkpoints, actor work dirs and code copies.

prefetch copies such a checkpoint to Pop while another evaluation runs, so its serve-checkpoint skips the copy (265-520 s
over Wi-Fi in grpo-v4-run2, 2-4% of an evaluation, and the step most often cut by a restart). held-eval runs one whole
evaluation (serve unless Pop already serves the policy, ``held_eval measure``, ``held_eval compare``) from a spec file:
grpo's ``--held-remote pop`` loop starts it as its own DGX ledger unit, so a restart of the training run neither kills its
episodes nor its copy (grpo-v4-run2: 14 restarts in 40 steps cut 8 of 10 evaluations and 3 copies).
"""
import argparse
import contextlib
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath

# A direct control-plane CLI resolves Pop's canonical machine paths rather than
# inheriting a learner pin. Importing its helpers runs inside that learner's
# process and must preserve the learner's checkout and child-process bindings.
if __name__ == '__main__':
    os.environ.pop('NATLANG_REPO', None)
    os.environ.pop('PYTHONPATH', None)
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.common.jsonio import write_json_durable as write_json  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

W = Path(os.environ.get('NATLANG_LFM3_WORKSPACE', resolve_str('data_nvme', 'lfm3-rl-dgx-20261029')))
# The state, view and receipt root (the former Pop evaluation root: its config.json names the Pop host).
ROOT = Path(os.environ.get('NATLANG_POP_EVAL_ROOT', str(W / 'ops/pop-behavior-eval')))
PYTHON = resolve_str('repo', '.venv-neuralese/bin/python')
LEDGER = str(REPO / 'scripts/memory_ledger.py')
# Pop's SSH host: NATLANG_POP_HOST, else "pop_host" in <root>/config.json. Pop mirrors the DGX data/repo layout.
POP = os.environ.get('NATLANG_POP_HOST') or (json.loads((ROOT / 'config.json').read_text()).get('pop_host')
                                             if (ROOT / 'config.json').is_file() else None)
POP_REPO = resolve_str('repo')
POP_RECOVERY = resolve_str('data_nvme', 'natlang-independent-recovery-20261031')
POP_TRANSPORT_ROOT = POP_RECOVERY + '/matched75-transport-0f8a-1101'   # the peer cache whose delta base Pop holds
POP_EVAL_ROOT = POP_RECOVERY + '/pop-behavior-eval'
POP_BASE = POP_RECOVERY + '/base-config-only-40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
IMAGE = 'natlang:pop-causal-conv1d-torch214-cu134-sm89-v1'
RAW_PREP_GB = 7.5
POP_RESERVE_GB = 0.5
POP_MIN_FREE = 8 * 2**30 + 512 * 2**20       # scratch floor on Pop
RL_PORT = 18531
# Pop ledger admission bound of an actor unit; the ssh that waits on it gets this plus a margin (a 600 s ssh timeout
# under a 900 s admission ended sft-v7's candidate held, 2026-11-21).
POP_ADMISSION_WAIT = 900
STATE = ROOT / 'rl-actor.json'
EXPORT_FORMAT = 'native-pop-rtx4060-bf16'
SGLANG_EXPORT_FORMAT = 'sglang-pop-rtx4060-bf16-vendor-projection'
# Pop's SGLang engine (RTX 4060, 8 GB; measured 2026-11-13, ops decisions.log): the KV pool, not the GPU, capped the
# held evals. BF16 KV held 94.7k tokens (22 KB a token: all 11 attention layers, the sliding ones at full length), so 7.5
# of 10 sequences ran, 13.7 requests waited and each evicted the waiting episodes' prefixes, which were prefilled again
# (prefix hit 41% against 94% on the DGX). A decode step costs about 20 ms whatever the batch, so tokens/s scale with
# running sequences (312 at 7.5, 672 at 16). FP8 KV holds about 180k tokens. FP8 is part of the actor law (attested
# kv_cache_dtype and the export format suffix): held results pair only within one setting.
POP_KV_CACHE_DTYPE = 'fp8_e4m3'
# 20 running (2026-11-17 smoke, POP_HELD_THROUGHPUT_20261117.md): decode tokens/s scale with the batch (18.5 ms a step at
# 1, ~26.5 at 20) and the 280k-token pool at 0.77 held 20 (full usage p99 0.90); attested as max_running.
POP_MAX_RUNNING = 20
# SGLang mem-fraction-static on Pop: weights + KV pool fill this share of the 7.62 GiB GPU. At 0.8 the engine kept
# 0.81 GB after CUDA-graph capture and its prefill transients took the rest: on 2026-11-16 01:3x an 88 MiB allocation
# in the fork's natlang_conv_prefill OOMed with 99 MiB free. 0.77 moves ~0.23 GB from the FP8 pool (318k -> ~280k
# tokens, still over 2x the 131072 context) to that headroom. Pool size only: per-token arithmetic is unchanged.
POP_MEM_FRACTION = 0.77


from natlang_neuralese.common.execution_profile import (  # noqa: E402
    G4_ACTOR_ALONE, G4_ACTOR_BESIDE_LEARNER, G4_VRAM_GIB)

COLAB_ROOT = os.environ.get('NATLANG_COLAB_ROOT', '/content/natlang-work')
COLAB_SGLANG_EXPORT_FORMAT = 'sglang-colab-g4-bf16-vendor-projection'
COLAB_MAX_RUNNING = 128
COLAB_MEM_FRACTION = 0.88


def pop_sglang(kv_cache_dtype=POP_KV_CACHE_DTYPE, max_running=POP_MAX_RUNNING, profile='pop'):
    """(job['sglang'] settings, actor export format) of a remote SGLang engine (Pop or Colab G4)."""
    fp8 = kv_cache_dtype != 'auto'
    if str(profile).lower().replace('-', '_') == 'colab_g4':
        cap = max_running if max_running is not None else COLAB_MAX_RUNNING
        settings = {'kv_cache_dtype': kv_cache_dtype, 'max_running': cap, 'mem_fraction': COLAB_MEM_FRACTION,
                    'max_total_tokens': 2097152 if fp8 else 1048576}
        return settings, COLAB_SGLANG_EXPORT_FORMAT + (f'-kv-{kv_cache_dtype}' if fp8 else '')
    settings = {'kv_cache_dtype': kv_cache_dtype, 'max_running': max_running, 'mem_fraction': POP_MEM_FRACTION,
                'max_total_tokens': 524288 if fp8 else 262144}
    return settings, SGLANG_EXPORT_FORMAT + (f'-kv-{kv_cache_dtype}' if fp8 else '')


def colab_hosts(explicit=None):
    """Ordered Colab G4 SSH host aliases from explicit arg, NATLANG_COLAB_HOSTS, or ROOT/config.json."""
    from natlang_neuralese.common.execution_profile import parse_colab_hosts
    return list(parse_colab_hosts(explicit, ROOT / 'config.json'))


def ssh_host(host, command, *, input=None, timeout=600, check=True):
    """Run an SSH command on any remote host (Pop or Colab G4) over Tailscale."""
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', host, command], input=input,
                            text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'Remote command on {host} failed ({result.returncode}): {command[:200]}: {result.stderr[-2000:]}')
    return result.stdout


def host_reachable(host, timeout=20):
    """True when ``host`` accepts batch SSH over Tailscale within ``timeout`` seconds."""
    if not host:
        return False
    try:
        return subprocess.run(['ssh', '-o', f'ConnectTimeout={min(timeout, 15)}', '-o', 'BatchMode=yes', host, 'true'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def reachable_hosts(profile='pop', hosts=None):
    """Return reachable SSH hosts for ``profile`` ('pop' or 'colab_g4')."""
    key = str(profile).lower().replace('-', '_')
    candidates = list(hosts) if hosts else (colab_hosts() if key == 'colab_g4' else ([POP] if POP else []))
    return [h for h in candidates if host_reachable(h)]


def host_address(host):
    """Resolve the tailscale IP/hostname configured for SSH alias ``host``."""
    out = subprocess.run(['ssh', '-G', host], capture_output=True, text=True, check=True).stdout
    return next(line.split()[1] for line in out.splitlines() if line.startswith('hostname '))


def sync_to_remote(local_path, remote_path, host, *, attempts=None, stall_seconds=120):
    """Resumable rsync from DGX to a remote host (Pop or Colab G4)."""
    ssh_host(host, shlex.join(['mkdir', '-p', str(PurePosixPath(remote_path).parent)]))
    copy_resumable(str(local_path), f'{host}:{remote_path}', attempts=attempts, stall_seconds=stall_seconds)


def sync_from_remote(remote_path, local_path, host, *, attempts=None, stall_seconds=120):
    """Resumable rsync from a remote host (Colab G4 or Pop) back to DGX."""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    copy_resumable(f'{host}:{remote_path}', str(local_path), attempts=attempts, stall_seconds=stall_seconds)


def remote_identity(host, remote_path):
    """(size in bytes, sha256) of a file on ``host``."""
    out = ssh_host(host, f'stat -c %s {shlex.quote(remote_path)} && sha256sum {shlex.quote(remote_path)}', timeout=1800)
    lines = out.split()
    return int(lines[0]), lines[1]


def pull_verified(remote_path, local_path, host, *, attempts=5):
    """Copy a finished artifact (an SFT ``checkpoint.pt``) from ``host`` to ``local_path`` only whole: rsync into
    ``<local>.pull`` (resumable across attempts), check its size and sha256 against the remote file, then rename it
    into place. A failed or truncated copy never appears under the final name (a partial ``checkpoint.pt`` was once
    adopted as the cycle's SFT result). Returns the sha256."""
    from natlang_neuralese.common.hashing import sha256_file_hex
    local_path = Path(local_path)
    temp = local_path.with_name(local_path.name + '.pull')
    size, sha = remote_identity(host, remote_path)
    sync_from_remote(remote_path, temp, host, attempts=attempts)
    got_size = temp.stat().st_size
    got_sha = sha256_file_hex(temp) if got_size == size else None
    if got_size != size or got_sha != sha:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f'{host}:{remote_path} arrived as {got_size} bytes (sha {got_sha}), the remote file is '
                           f'{size} bytes (sha {sha}); the copy was discarded')
    os.replace(temp, local_path)
    return sha


def colab_state_path(host):
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(host))
    return ROOT / f'colab-actor-{safe}.json'


def host_config(profile='pop', *, host=None):
    """Unified remote-host layout for ``pop`` and ``colab_g4``."""
    key = str(profile).lower().replace('-', '_')
    if key == 'colab_g4':
        chosen = host or (colab_hosts()[0] if colab_hosts() else None)
        return {
            'profile': 'colab_g4',
            'host': chosen,
            'root': COLAB_ROOT,
            'repo': f'{COLAB_ROOT}/natlang',
            'python': f'{COLAB_ROOT}/natlang/.venv-neuralese/bin/python',
            'base': f'{COLAB_ROOT}/base-lfm3',
            'checkpoints': f'{COLAB_ROOT}/checkpoints',
            'work_root': f'{COLAB_ROOT}/runs',
            'kv_cache_dtype': POP_KV_CACHE_DTYPE,
            'max_running': COLAB_MAX_RUNNING,
            'mem_fraction': COLAB_MEM_FRACTION,
            'admission': f'{COLAB_ROOT}/natlang/scripts/colab/admission.py',
            # the fork's venv (scripts/colab/bootstrap.py installs training/neuralese/sglang-fork.json's commit there)
            'sglang_python': f'{COLAB_ROOT}/sglang-natlang-venv/bin/python',
            'env': f'{COLAB_ROOT}/natlang-env.sh',
            'vram_gib': G4_ACTOR_ALONE['vram_gib'],
            'host_ram_gib': G4_ACTOR_ALONE['host_ram_gib'],
        }
    return {
        'profile': 'pop',
        'host': host or POP,
        'root': POP_EVAL_ROOT,
        'repo': POP_REPO,
        'python': POP_REPO + '/.venv-neuralese/bin/python',
        'base': POP_BASE,
        'checkpoints': POP_CHECKPOINTS,
        'work_root': POP_EVAL_ROOT,
        'kv_cache_dtype': POP_KV_CACHE_DTYPE,
        'max_running': POP_MAX_RUNNING,
        'mem_fraction': POP_MEM_FRACTION,
        'admission': POP_REPO + '/scripts/memory_ledger.py',
        'vram_gib': 7.5,
        'host_ram_gib': 10.5,
    }


SGLANG_MEMORY_GB = 11   # 8 GiB was OOM-killed at CUDA-graph capture (2026-11-12): front + view on CPU + server


def pop_sglang_budget_gb():
    """Pop ledger budget of an SGLang actor unit (start and serve-checkpoint alike): Pop's 15 GB leave 10.9-11.5 GiB
    admissible (10.9 with the gdm greeter after a desktop logout), and the engine's own limits bound its memory, so
    the unit claims SGLANG_MEMORY_GB - 1.5 (other agents' resident Pop processes left 10.2 GiB admissible on 2026-11-21,
    so 10 + the 0.5 reserve timed out sft-v7's candidate held twice; MemoryMax stays SGLANG_MEMORY_GB - 0.5).
    NATLANG_POP_HELD_BUDGET_GB overrides."""
    return os.environ.get('NATLANG_POP_HELD_BUDGET_GB', str(SGLANG_MEMORY_GB - 1.5))


RL_MEMORY_GB = 8   # container cap and ledger claim; 4 GiB was OOM-killed after 4 h at 3 workers (anon 4.2 GiB, 2026-11-09)


def log(**event):
    print(json.dumps({'at': round(time.time(), 1), 'hook': 'pop_rl_actor', **event}, sort_keys=True), flush=True)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def _stat5(path):
    s = Path(path).stat()
    return {'device': s.st_dev, 'inode': s.st_ino, 'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def prepare_view(args):
    """The ledger unit's work: the model-only actor view of one checkpoint, read in place (never written)."""
    from natlang_neuralese.model.native_bundle import prepare_actor_transport
    before, started = _stat5(args.checkpoint), time.time()
    binding = prepare_actor_transport(args.checkpoint, args.out, scope='rl-controller-pin')
    transport = json.loads(Path(binding['path']).read_text())
    if _stat5(args.checkpoint) != before:
        raise SystemExit('source changed during preparation')
    if args.expect_sha256 and transport['source']['sha256'] != args.expect_sha256:
        raise SystemExit(f'source hashes to {transport["source"]["sha256"]}, expected {args.expect_sha256}')
    write_json(args.receipt, {'schema': 'natlang.pop-eval-preparation/1', 'kind': 'rl', 'transport': binding,
                              'source': transport['source'], 'step': transport['step'], 'view': transport['view'],
                              'view_receipt': transport['view_receipt'], 'code': transport.get('code'),
                              'scope': transport['scope'], 'source_path': str(Path(args.checkpoint).resolve()),
                              'source_stat': before, 'seconds': time.time() - started})


def prepare(checkpoint, expect_sha, wait=None):
    """Preparation receipt of the policy's actor view (reused while its view file is present)."""
    out = ROOT / 'rl-views'
    receipt = ROOT / 'preparations' / f'rl-{expect_sha}.json'
    existing = read_json(receipt)
    if (existing and Path(existing['transport']['path']).is_file()
            and (Path(existing['transport']['path']).parent / existing['view']['path']).is_file()):
        return existing
    # A pruned view keeps its receipts; retire receipts of a view directory without any view before re-preparing.
    stale = out / expect_sha
    if stale.is_dir() and not any(stale.glob('*.pt')):
        retired = ROOT / 'dgx-retirement' / f'stale-receipts-{expect_sha[:12]}-{int(time.time())}'
        retired.mkdir(parents=True, exist_ok=True)
        for item in stale.glob('*.json'):
            os.replace(item, retired / item.name)
    command = ['/usr/bin/ionice', '-c2', '-n6', PYTHON, str(Path(__file__).resolve()), 'prepare-view',
               '--checkpoint', str(checkpoint), '--out', str(out), '--receipt', str(receipt), '--expect-sha256', expect_sha]
    env = [f'NATLANG_REPO={REPO}', f'PYTHONPATH={REPO}/training/neuralese:{REPO}', 'OMP_NUM_THREADS=4',
           f'NATLANG_HASH_CACHE={ROOT}/hash-cache.jsonl', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1']
    unit, status, result = ledger_unit(f'popeval-rl-{expect_sha[:12]}', command, budget_gb=RAW_PREP_GB,
                                       workdir=str(REPO), env=env, **({'wait': wait} if wait else {}))
    if status != 0 or not receipt.is_file():
        raise RuntimeError(f'rl preparation unit {unit} failed ({result}, exit {status}); see journalctl --user -u {unit}')
    return read_json(receipt)


def copy_resumable(source, dest, attempts=None, stall_seconds=120):
    """rsync that fails a stalled stream after ``stall_seconds`` without data (--timeout) and resumes it (--partial):
    a Wi-Fi stall on 2026-11-19 held a copy at 2.5 of 3.5 GB for 35 min. NATLANG_POP_COPY_ATTEMPTS=1 opts out."""
    attempts = attempts or int(os.environ.get('NATLANG_POP_COPY_ATTEMPTS', '5'))
    for attempt in range(attempts):
        try:
            subprocess.run(['rsync', '-a', '--partial', f'--timeout={stall_seconds}', source, dest], check=True,
                           timeout=7200)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == attempts - 1:
                raise
            time.sleep(30)


@contextlib.contextmanager
def copy_lock(sha):
    """One copy of a checkpoint to Pop at a time across DGX processes (a prefetch in the training run, a serve-checkpoint
    in an evaluation unit): two rsyncs into one .partial file would interleave."""
    path = ROOT / 'locks' / f'pop-copy-{sha[:16]}.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def checkpoint_on_pop(checkpoint, sha):
    """Pop's copy of a served checkpoint (``POP_CHECKPOINTS/<sha>.pt``), copied first unless present; the driver
    checks its sha256 before serving."""
    target = f'{POP_CHECKPOINTS}/{sha}.pt'
    with copy_lock(sha):
        if not ssh(shlex.join(['test', '-f', target]) + ' && echo ok', check=False).strip():
            ssh(shlex.join(['mkdir', '-p', POP_CHECKPOINTS]))
            copy_resumable(str(checkpoint), f'{POP}:{target}.partial')
            ssh(shlex.join(['mv', f'{target}.partial', target]))
            return True
    return False


def prefetch(args):
    """Copy a checkpoint to Pop ahead of its serve-checkpoint, beside a running evaluation (only the file moves; the
    engine and its GPU are untouched). Keeps the newest POP_KEEP + 1 checkpoints (this one and what Pop serves
    protected) and refuses when Pop's disk would fall under the scratch floor."""
    sha, target = args.policy_id, f'{POP_CHECKPOINTS}/{args.policy_id}.pt'
    served = (read_json(STATE) or {}).get('policy_id')
    protect = (target,) + ((f'{POP_CHECKPOINTS}/{served}.pt',) if served else ())
    ssh(shlex.join(['mkdir', '-p', POP_CHECKPOINTS]))
    prune_pop_newest(f'{POP_CHECKPOINTS}/*.pt', POP_KEEP + 1, protect=protect)
    free = int(ssh(shlex.join(['df', '--output=avail', '-B1', POP_CHECKPOINTS]), timeout=60).split()[-1])
    if free < Path(args.checkpoint).stat().st_size + POP_MIN_FREE:
        raise RuntimeError(f'Pop has {free / 2**30:.1f} GiB free: no prefetch')
    started = time.time()
    copied = checkpoint_on_pop(args.checkpoint, sha)
    result = {'event': 'Pop checkpoint prefetched' if copied else 'Pop checkpoint already present', 'policy_id': sha,
              'step': args.step, 'copy_seconds': round(time.time() - started, 1)}
    log(**result)
    return result


def ledger_unit(name, command, *, budget_gb, workdir, env=(), wait=86400):
    unit = f'{name}-{int(time.time())}'
    argv = ['/usr/bin/python3', LEDGER, 'run', '--unit', unit, '--budget-gb', str(budget_gb), '--priority', 'low',
            '--class', 'experiment', '--no-resumable', '--wait', str(wait), '--workdir', workdir]
    for item in env:
        argv += ['--env', item]
    subprocess.run(argv + ['--', *command], check=True, timeout=wait + 600)
    while True:
        show = subprocess.run(['systemctl', '--user', 'show', unit + '.service', '-p', 'ActiveState', '-p', 'Result',
                               '-p', 'ExecMainStatus'], capture_output=True, text=True).stdout
        fields = dict(line.split('=', 1) for line in show.split('\n') if '=' in line)
        if fields.get('ActiveState') in ('inactive', 'failed'):
            return unit, int(fields.get('ExecMainStatus') or 0), fields.get('Result')
        time.sleep(15)


def ssh(command, *, input=None, timeout=600, check=True):
    if not POP:
        raise RuntimeError('set NATLANG_POP_HOST or "pop_host" in ' + str(ROOT / 'config.json'))
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', POP, command], input=input,
                            text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'Pop command failed ({result.returncode}): {command[:200]}: {result.stderr[-2000:]}')
    return result.stdout


def pop_python(code, *args, timeout=600):
    return json.loads(ssh(shlex.join(['python3', '-I', '-', *map(str, args)]), input=code, timeout=timeout))


def code_on_dgx():
    from natlang_neuralese.common.code_provenance import code_provenance
    return code_provenance(REPO)


def sync_pop_checkout():
    """Fast-forward Pop's checkout of the canonical repository to origin/main (``git pull --ff-only``), rebuild the
    TypeScript collector when ts-host changed, and return the commit and dirty flag it now runs. Pop runs this checkout
    directly (no runtime copy); a failed pull or build is recorded and the checkout keeps what it has (provenance,
    never a refusal)."""
    git = lambda *args: shlex.join(['git', '-C', POP_REPO, *args])
    out = ssh(git('rev-parse', 'HEAD') + '; ' + git('pull', '--ff-only', '--quiet', 'origin', 'main') + ' 2>&1; '
              + git('rev-parse', 'HEAD') + '; ' + git('status', '--porcelain', '--untracked-files=no') + ' | wc -l',
              check=False, timeout=300)
    lines = out.strip().splitlines()
    try:
        before, commit, dirty = lines[0].strip(), lines[-2].strip(), int(lines[-1]) > 0
    except (IndexError, ValueError):
        raise RuntimeError('cannot read the Pop checkout commit: ' + out[-400:])
    notes = lines[1:-2]
    result = {'schema': 'natlang.code-provenance/1', 'git_commit': commit, 'dirty': dirty, 'pulled': not notes}
    if notes:
        log(event='Pop checkout pull failed; running its current commit', commit=commit, detail=' '.join(notes)[:300])
    if before != commit or not ssh(shlex.join(['test', '-f', POP_REPO + '/ts-host/dist/teacher/cli.js']) + ' && echo ok',
                                   check=False).strip():
        changed = ssh(git('diff', '--name-only', before, commit, '--', 'ts-host', 'scripts/stage-npm-package.mjs'), check=False)
        if changed.strip() or before == commit:
            build = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', POP,
                                    f'cd {shlex.quote(POP_REPO)}/ts-host && npm run build:node'], capture_output=True,
                                   text=True, timeout=1800)
            result['collector_build'] = 'ok' if build.returncode == 0 else 'failed'
            if build.returncode:
                log(event='Pop collector build failed; keeping the previous dist', detail=(build.stderr or build.stdout)[-400:])
    return result


def _pop_upload_json(value, remote_path):
    ssh(shlex.join(['bash', '-c', f'cat > {shlex.quote(remote_path)}']), input=json.dumps(value))
    return remote_path


def remove_pop_work(paths):
    """Delete Pop work-directory entries under POP_EVAL_ROOT. The containers write them as root, so a throwaway
    container of the evaluation image removes them (Pop grants no sudo)."""
    root = PurePosixPath(POP_EVAL_ROOT)
    inside = [str(PurePosixPath('/r') / PurePosixPath(path).relative_to(root)) for path in paths]
    if not inside:
        return
    ssh(shlex.join(['docker', 'run', '--rm', '--name', f'pop-prune-{os.getpid()}-{int(time.time())}',
                    '--entrypoint', '/bin/rm', '-v', f'{POP_EVAL_ROOT}:/r', IMAGE, '-rf', '--', *inside]), timeout=1800)


def send(prep):
    from scripts.coreweave.send_actor_transport import send as send_transport
    receipt = send_transport(prep['transport']['path'], prep['transport']['sha256'], POP, POP_TRANSPORT_ROOT)
    write_json(ROOT / 'transfers' / f'{prep["transport"]["sha256"]}.json', receipt)
    log(event='sent', mode=receipt['mode'], payload_mb=round(receipt['payload_bytes'] / 2**20, 1),
        view=prep['view']['sha256'][:12])
    return receipt


def pop_unit_state(unit):
    out = ssh(shlex.join(['systemctl', '--user', 'show', unit + '.service', '-p', 'ActiveState', '-p', 'ExecMainStatus',
                          '-p', 'LoadState']), check=False)
    return dict(line.split('=', 1) for line in out.split('\n') if '=' in line)


POP_STORAGE = r'''
import json,os,shutil,subprocess,sys
from pathlib import Path, PurePosixPath
root=Path(sys.argv[1]);keep=set(json.loads(sys.argv[2]));need=int(sys.argv[3]);dry=sys.argv[4]=='1'
def free():s=os.statvfs(root);return s.f_bavail*s.f_frsize
hint=root/'.natlang-actor-transfer.json'
if hint.exists():keep.add(Path(json.loads(hint.read_text())['receiver_view']).parent.name)
busy=subprocess.run(['docker','ps','-q'],capture_output=True,text=True).stdout.strip()
removed=[]
if free()<need and not busy:
 stages=sorted((p for p in root.iterdir() if p.is_dir() and len(p.name)==64 and p.name not in keep),key=lambda p:p.stat().st_mtime)
 for stage in stages:
  if free()>=need:break
  if any(stage.glob('*.consumer.json')):continue
  for item in list(stage.glob('*.pt'))+[stage/'.zip-delta',stage/'.delta']:
   if not item.exists():continue
   size=item.stat().st_size if item.is_file() else sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
   removed.append({'path':str(item),'bytes':size})
   if not dry:
    shutil.rmtree(item) if item.is_dir() else item.unlink()
print(json.dumps({'free':free(),'need':need,'removed':removed,'busy':bool(busy),'keep':sorted(keep)}))
'''


def make_pop_room(incoming_bytes, keep):
    """Prune old Pop actor views (never the delta base or an in-use stage) until the next view plus the scratch floor
    fit. Only model-only views and delta caches under the transport root are removed; they are reproducible."""
    need = incoming_bytes + POP_MIN_FREE + 2**30
    result = pop_python(POP_STORAGE, POP_TRANSPORT_ROOT, json.dumps(sorted(keep)), need, '0')
    if result['removed']:
        write_json(ROOT / 'pop-retirement' / f'{int(time.time())}.json', {
            'schema': 'natlang.pop-actor-view-retirement/1', **result,
            'scope': 'derived model-only actor views/delta caches on Pop; sources, receipts and evidence untouched'})
        log(event='pruned Pop views', removed=len(result['removed']), free_gb=round(result['free'] / 2**30, 1))
    if result['free'] < need:
        raise RuntimeError(f'Pop has {result["free"] / 2**30:.1f} GiB free; needs {need / 2**30:.1f} GiB')


def prune_dgx(keep):
    """Remove older DGX actor views except ``keep`` and the peer delta base (sender view)."""
    hint = pop_python('import sys,json,os;p=sys.argv[1]+"/.natlang-actor-transfer.json";'
                      'print(json.dumps(json.load(open(p)) if os.path.exists(p) else None))', POP_TRANSPORT_ROOT)
    removed = []
    for kind in ('views', 'ema', 'rl-views'):
        for directory in sorted((ROOT / kind).glob('*')):
            if directory.name in keep or (hint and str(directory) in hint['sender_view']):
                continue
            for item in directory.glob('*.pt'):
                removed.append({'path': str(item), 'bytes': item.stat().st_size})
                item.unlink()
            delta = directory / '.transfer-deltas'
            if delta.exists():
                import shutil
                shutil.rmtree(delta)
                removed.append({'path': str(delta)})
    if removed:
        write_json(ROOT / 'dgx-retirement' / f'{int(time.time())}.json', {
            'schema': 'natlang.pop-eval-dgx-view-retirement/1', 'removed': removed, 'keep': sorted(keep),
            'scope': 'derived model-only views and delta caches; receipts and evidence kept'})


def pop_host_address():
    """The address the actor binds and the DGX dials: the SSH host's own address (Pop's tailscale IP)."""
    out = subprocess.run(['ssh', '-G', POP], capture_output=True, text=True, check=True).stdout
    return next(line.split()[1] for line in out.splitlines() if line.startswith('hostname '))


def attested(endpoint, policy_id, adapters=None):
    """The actor's info when it serves checkpoint ``policy_id`` and, with ``adapters`` (a release's result.json sha256),
    exactly that default adapter release; a plain checkpoint actor must serve no default adapters."""
    try:
        with urllib.request.urlopen(endpoint + '/v1/neuralese/info', timeout=5) as f:
            info = json.load(f)
    except OSError:
        return None
    binding = info.get('rl_policy_binding') or {}
    served = (binding.get('adapters') or {}).get('result_sha256')
    if binding.get('checkpoint_sha256') == policy_id and binding.get('actor_export') and served == adapters:
        return info
    return None


def pop_containers():
    return ssh("docker ps --format '{{.Names}}'", check=True).split()


def stop_unit(state):
    unit = state.get('unit')
    if unit:
        ssh(shlex.join(['systemctl', '--user', 'stop', unit + '.service']), check=False, timeout=120)
        for _ in range(30):
            if pop_unit_state(unit).get('ActiveState') in ('inactive', 'failed', None):
                break
            time.sleep(2)


def adapter_release(args):
    """(release dir, result.json sha256, largest node rank) of ``--adapter-release`` (a program fit, tune.build output)
    or (None, None, None)."""
    release = getattr(args, 'adapter_release', None)
    if not release:
        return None, None, None
    from natlang_neuralese.common.hashing import sha256_file_hex
    release = Path(release)
    result = json.loads((release / 'result.json').read_text())
    return release, sha256_file_hex(release / 'result.json'), max(item['rank'] for item in result['artifacts'])


def send_adapter_release(job, work, release, adapters, rank):
    """Copy the release to Pop and name it in the job: the driver serves it by default on one SGLang adapter slot."""
    remote = f'{work}/adapter-{adapters[:16]}'
    ssh(shlex.join(['mkdir', '-p', remote]))
    subprocess.run(['rsync', '-a', '--delete', f'{release}/result.json', f'{release}/artifacts',
                    *([f'{release}/states'] if (release / 'states').is_dir() else []), f'{POP}:{remote}/'],
                   check=True, timeout=3600)
    job['adapter_release'] = {'path': remote, 'rank': rank, 'scale': 1.0, 'result_sha256': adapters}


def start(args):
    # A previous actor of this lineage that still serves the same policy (the controller restarted mid-collection)
    # keeps its block store and its code: an actor process keeps the code it started with, and the controller records
    # each actor's commit from its attestation. One serving another policy is replaced; retained-partial custody
    # rehydrates a fresh actor's store.
    previous = read_json(STATE)
    runtime = str(REPO)
    sglang, sglang_format = pop_sglang(args.kv_cache_dtype, args.max_running)
    release, adapters, rank = adapter_release(args)
    # ... with the same engine settings (KV dtype, running cap: held results pair only within one attested engine)
    if (previous and previous['policy_id'] == args.policy_id and previous.get('engine') == args.engine
            and (args.engine != 'sglang' or previous.get('sglang') == sglang) and previous.get('adapters') == adapters
            and attested(previous['endpoint'], args.policy_id, adapters)):
        log(event='Pop RL actor already serves the policy', endpoint=previous['endpoint'])
        return previous
    stop_pop_fills()
    if previous:
        stop_unit(previous)
    host = args.bind_host or pop_host_address()
    endpoint = f'http://{host}:{args.port}'
    ssh('true', timeout=30)
    busy = [name for name in pop_containers() if not name.startswith('poprl-')]
    if busy:
        raise RuntimeError(f'Pop GPU is held by {busy} (an evaluation owns it until it finishes)')
    started = time.time()
    code = {'dgx': code_on_dgx(), 'pop': sync_pop_checkout()}
    prep = prepare(args.checkpoint, args.policy_id, wait=args.prepare_wait)
    make_pop_room(prep['view']['bytes'], {prep['source']['sha256']})
    send(prep)
    source = prep['source']['sha256']
    if source != args.policy_id:
        raise RuntimeError('prepared view binds another source than the controller policy')
    # Keyed by policy only: the actor's code commit is provenance (job['code']), and a relaunch on newer code must keep
    # the block store, or blocks sampled before it are lost at bundling (w5 round 0, 2026-11-10).
    work = f'{POP_EVAL_ROOT}/rl-actor-{prep["step"]}-{source[:12]}'
    name = f'poprl-{source[:12]}'
    transport_json = json.loads(Path(prep['transport']['path']).read_text())
    job = {'mode': 'rl-actor', 'arm': 'rl', 'name': name, 'port': args.port, 'host': host, 'runtime': POP_REPO,
           'code': code['pop'], 'base': POP_BASE, 'image': IMAGE, 'step': prep['step'],
           'source_sha256': source, 'stage': f'{POP_TRANSPORT_ROOT}/{source}', 'view': prep['view']['path'],
           'view_sha256': prep['view']['sha256'], 'work': work, 'min_free_bytes': POP_MIN_FREE,
           'actor_export_format': sglang_format if args.engine == 'sglang' else EXPORT_FORMAT,
           'memory': f'{RL_MEMORY_GB}g', 'engine': args.engine, **({'sglang': sglang} if args.engine == 'sglang' else {}),
           'transport': {'name': Path(prep['transport']['path']).name, 'sha256': prep['transport']['sha256'],
                         'view_receipt_name': prep['view_receipt']['sha256'] + '.json',
                         'view_receipt_sha256': prep['view_receipt']['sha256'], 'scope': transport_json['scope']}}
    # A restart under the same policy reuses this work dir (and its block store); the previous driver copy came from a
    # sealed read-only runtime file, so it is replaced rather than overwritten.
    ssh(shlex.join(['mkdir', '-p', work]) + ' && ' + shlex.join(['rm', '-f', f'{work}/driver.py']))
    # Older policies' actor dirs (block store, caches; 1-3 GB each) are spent once their rounds bundled their blocks to
    # the DGX. Keep this policy's and the newest earlier one (its pre-collection may still be bundling).
    earlier = ssh(f'ls -dt {shlex.quote(POP_EVAL_ROOT)}/rl-actor-* 2>/dev/null || true', check=False).split()
    remove_pop_work([path for path in earlier if path != work][1:])
    subprocess.run(['scp', '-q', str(REPO / 'scripts/neuralese_data/pop_actor_driver.py'), f'{POP}:{work}/driver.py'],
                   check=True, timeout=120)
    if release:
        if args.engine != 'sglang':
            raise RuntimeError('an adapter release is served on SGLang adapter slots (--engine sglang)')
        send_adapter_release(job, work, release, adapters, rank)
    _pop_upload_json(job, f'{work}/job.json')
    unit = f'{name}-{int(time.time())}'
    # The SGLang actor runs on Pop's host (the fork's venv); the native one in its container.
    container = [] if args.engine == 'sglang' else ['--container', name]
    # The SGLang front holds the CPU-loaded view until the server is up, beside the server's own processes; the same
    # admissible budget as serve-checkpoint (pop_sglang_budget_gb: 10.5 was refused beside the gdm greeter, 2026-11-20).
    # On the host (no container) MemoryMax is the real cap: it stays at the 10.5 GB the actor has run with.
    budget = pop_sglang_budget_gb() if args.engine == 'sglang' else str(RL_MEMORY_GB)
    cap = ['--host-max-gb', str(SGLANG_MEMORY_GB - .5)] if args.engine == 'sglang' else []
    ssh(shlex.join(['/usr/bin/python3', POP_REPO + '/scripts/memory_ledger.py', 'run', '--unit', unit,
                        '--budget-gb', budget, *cap, '--reserve-gb', str(POP_RESERVE_GB),
                        '--class', 'experiment', '--no-resumable', '--wait', str(POP_ADMISSION_WAIT), *container, '--workdir', work, '--',
                        '/usr/bin/python3', f'{work}/driver.py', f'{work}/job.json']), timeout=POP_ADMISSION_WAIT + 300)
    result = {'endpoint': endpoint, 'unit': unit, 'container': name, 'policy_id': source, 'work': work, 'runtime': runtime,
              'engine': args.engine, **({'sglang': sglang} if args.engine == 'sglang' else {}),
              'prepared_seconds': round(time.time() - started, 1), 'adapters': adapters}
    write_json(STATE, result)
    try:
        prune_dgx({source})   # older derived DGX views; the one just sent stays as the next delta base
    except Exception as error:
        log(event='DGX view pruning failed', error=repr(error)[:300])
    deadline = time.monotonic() + args.ready_seconds
    while time.monotonic() < deadline:
        if pop_unit_state(unit).get('ActiveState') in ('inactive', 'failed'):
            tail = ssh(shlex.join(['journalctl', '--user', '-u', unit + '.service', '-n', '20', '--no-pager']), check=False)
            raise RuntimeError('Pop RL actor unit exited before readiness: ' + tail[-1200:])
        if attested(endpoint, source, adapters):
            result['ready_seconds'] = round(time.time() - started, 1)
            write_json(STATE, result)
            log(event='Pop RL actor ready', **result)
            return result
        time.sleep(5)
    raise RuntimeError(f'Pop RL actor did not attest {source[:12]} at {endpoint} within {args.ready_seconds} s')


POP_CHECKPOINTS = POP_RECOVERY + '/policy-checkpoints'   # pruned native checkpoints served directly
POP_KEEP = 2


def pop_actor_units():
    out = ssh("systemctl --user list-units 'poprl-*' 'popheld-*' --state=active,activating --plain --no-legend", check=False)
    return [line.split()[0].removesuffix('.service') for line in out.splitlines() if line.strip()]


def pop_fill_units():
    """DGX units drawing fills from Pop's server (expert_iteration ``<cycle>-popfill`` and its tunnel)."""
    out = subprocess.run(['systemctl', '--user', 'list-units', '*-popfill.service', '*-popfill-tunnel.service',
                          '--state=active,activating', '--plain', '--no-legend'], capture_output=True, text=True).stdout
    return [line.split()[0].removesuffix('.service') for line in out.splitlines() if line.strip()]


def stop_pop_fills():
    """A cycle's Pop fill runs on until Pop serves something else (DECISIONS 2026-11-22 "Pop fills through build"):
    every serve stops it first, so no fill draws from a policy it did not start on."""
    for unit in pop_fill_units():
        subprocess.run(['/usr/bin/python3', LEDGER, 'cancel', '--unit', unit], stderr=subprocess.DEVNULL)
        for suffix in ('-waiter', ''):
            subprocess.run(['systemctl', '--user', 'stop', f'{unit}{suffix}.service'], stderr=subprocess.DEVNULL)
        log(event='stopped Pop fill', unit=unit)


def stop_pop_actors():
    """Stop every Pop actor unit and wait until the GPU has no compute process (the driver refuses a busy GPU)."""
    stop_pop_fills()
    for unit in pop_actor_units():
        stop_unit({'unit': unit})
    for _ in range(45):
        if not ssh('nvidia-smi --query-compute-apps=pid --format=csv,noheader', check=False).strip():
            return
        time.sleep(2)
    raise RuntimeError('the Pop GPU still has a compute process after stopping its actors')


def pop_code(commit):
    """An exported copy of ``commit`` on Pop (``git archive`` of Pop's checkout after a fetch): the actor's runtime."""
    dest = f'{POP_EVAL_ROOT}/code-{commit[:12]}'
    q = shlex.quote
    ssh(f'test -f {q(dest)}/.natlang-code.json || (git -C {q(POP_REPO)} fetch -q origin && rm -rf {q(dest)}.partial '
        f'&& mkdir -p {q(dest)}.partial && git -C {q(POP_REPO)} archive {q(commit)} | tar -x -C {q(dest)}.partial '
        f'&& echo {q(json.dumps({"commit": commit}))} > {q(dest)}.partial/.natlang-code.json && mv {q(dest)}.partial {q(dest)})',
        timeout=900)
    return dest


def prune_pop_newest(pattern, keep, protect=()):
    """Keep ``keep`` Pop paths matching ``pattern``: those in ``protect`` (present or about to be), then the newest."""
    paths = ssh(f'ls -dt {pattern} 2>/dev/null || true', check=False).split()
    old = [path for path in paths if path not in protect][max(0, keep - len(protect)):]
    if old:
        ssh(shlex.join(['rm', '-rf', '--', *old]), timeout=600)
    return old


def stop_colab_g4(host):
    """Stop the natlang SGLang actor this DGX started on a Colab G4 VM: its admission process group, by the PID the
    launch recorded (never by pattern: a ``pkill -f`` over ssh matches the remote shell running it)."""
    state_file = colab_state_path(host)
    state = read_json(state_file) or {}
    pid = state.get('pid')
    if pid:
        ssh_host(host, f'kill -TERM -- -{int(pid)} 2>/dev/null; for i in $(seq 30); do kill -0 {int(pid)} 2>/dev/null '
                       f'|| exit 0; sleep 2; done; kill -KILL -- -{int(pid)} 2>/dev/null; true', check=False, timeout=120)
    state_file.unlink(missing_ok=True)
    log(event='Colab G4 actor stopped', host=host, pid=pid, previous=state.get('policy_id'))
    return state


def colab_serve_command(cfg, *, checkpoint, sha, port, work, beside_learner, kv_cache_dtype):
    """(serve argv, admission claim) of a Colab G4 SGLang actor: alone it takes the GPU (mem fraction 0.88, 128
    running); beside the SFT learner under MPS (async-train §1) it is capped at 24 GB KV and 64 running and claims only
    that share, so the learner's admission still fits."""
    sizing = G4_ACTOR_BESIDE_LEARNER if beside_learner else G4_ACTOR_ALONE
    sglang, sglang_format = pop_sglang(kv_cache_dtype, sizing['max_running'], profile='colab_g4')
    memory = (['--sglang-kv-gb', str(sizing['kv_gb'])] if sizing['kv_gb'] else
              ['--sglang-mem-fraction', str(sglang['mem_fraction'])])
    if sizing['kv_gb']:
        sglang = {**{k: v for k, v in sglang.items() if k != 'mem_fraction'}, 'kv_gb': sizing['kv_gb']}
    serve = [cfg['python'], f'{cfg["repo"]}/scripts/neuralese_data/serve_native_checkpoint.py',
             '--store-dir', f'{work}/store', '--checkpoint', checkpoint, '--checkpoint-sha256', sha,
             '--base', cfg['base'], '--host', '0.0.0.0', '--port', str(port), '--prompt-rendering', 'full',
             '--engine', 'sglang', '--sglang-python', cfg['sglang_python'], '--sglang-projection', 'vendor',
             '--sglang-kv-cache-dtype', sglang['kv_cache_dtype'], '--sglang-max-running', str(sglang['max_running']),
             *memory, '--hf-export-dir', f'{work}/sglang-hf', '--actor-export-format', sglang_format]
    return serve, {'vram_gib': sizing['vram_gib'], 'host_ram_gib': sizing['host_ram_gib']}, sglang, sglang_format


def serve_colab_g4(args):
    """Serve a merged BF16 checkpoint on a Colab G4 VM under scripts/colab/admission.py (a VRAM/host-RAM claim that
    lasts while the actor runs) and wait until it attests the checkpoint."""
    cfg = host_config('colab_g4', host=getattr(args, 'host', None))
    host = cfg['host']
    if not host:
        raise RuntimeError('set --host, NATLANG_COLAB_HOSTS, or "colab_hosts" in ' + str(ROOT / 'config.json'))
    sha, commit = args.policy_id, args.commit
    remote_ckpt = f'{cfg["checkpoints"]}/{sha}.pt'
    remote_work = f'{cfg["work_root"]}/actor-{sha[:12]}'
    serve_cmd, claim, sglang, sglang_format = colab_serve_command(
        cfg, checkpoint=remote_ckpt, sha=sha, port=args.port, work=remote_work,
        beside_learner=bool(getattr(args, 'beside_learner', False)), kv_cache_dtype=args.kv_cache_dtype)
    state_file = colab_state_path(host)
    previous = read_json(state_file)
    if (previous and previous.get('policy_id') == sha and previous.get('code') == commit
            and previous.get('sglang') == sglang and attested(previous['endpoint'], sha)):
        log(event='Colab G4 actor already serves the checkpoint', host=host, endpoint=previous['endpoint'])
        return previous
    if previous:
        stop_colab_g4(host)
    started = time.time()
    bind_ip = args.bind_host or host_address(host)
    endpoint = f'http://{bind_ip}:{args.port}'
    ssh_host(host, shlex.join(['mkdir', '-p', cfg['checkpoints'], remote_work]))
    if not ssh_host(host, shlex.join(['test', '-f', remote_ckpt]) + ' && echo ok', check=False).strip():
        sync_to_remote(args.checkpoint, f'{remote_ckpt}.partial', host)
        if remote_identity(host, f'{remote_ckpt}.partial')[1] != sha:
            raise RuntimeError(f'the checkpoint copied to {host} does not hash to {sha}')
        ssh_host(host, shlex.join(['mv', f'{remote_ckpt}.partial', remote_ckpt]))
    copied = time.time()
    ssh_host(host, f'git -C {shlex.quote(cfg["repo"])} fetch -q origin && '
                   f'git -C {shlex.quote(cfg["repo"])} checkout -q -f {shlex.quote(commit)}', timeout=300)
    admitted = [cfg['python'], cfg['admission'], 'run', '--vram-gib', str(claim['vram_gib']),
                '--host-ram-gib', str(claim['host_ram_gib']), '--label', f'actor-{sha[:12]}', '--wait', '300',
                '--workdir', cfg['repo'], '--', *serve_cmd]
    # setsid: the admission process leads its own group (stop_colab_g4 signals it); its PID is the launch's output
    pid = ssh_host(host, f'. {shlex.quote(cfg["env"])} && setsid nohup {shlex.join(admitted)} '
                         f'> {shlex.quote(remote_work + "/actor.log")} 2>&1 < /dev/null & echo $!', timeout=60).strip()
    result = {'endpoint': endpoint, 'host': host, 'profile': 'colab_g4', 'policy_id': sha, 'work': remote_work,
              'code': commit, 'engine': 'sglang', 'sglang': sglang, 'export_format': sglang_format,
              'pid': int(pid.splitlines()[-1]), 'claim': claim, 'copy_seconds': round(copied - started, 1),
              'started': started}
    write_json(state_file, result)
    deadline = time.monotonic() + args.ready_seconds
    while time.monotonic() < deadline:
        if attested(endpoint, sha):
            result['ready_seconds'] = round(time.time() - started, 1)
            write_json(state_file, result)
            log(event='Colab G4 actor ready', **result)
            return result
        time.sleep(5)
    stop_colab_g4(host)
    raise RuntimeError(f'Colab G4 actor on {host} did not attest {sha[:12]} at {endpoint} within {args.ready_seconds} s')


def remap_sft_args(argv, mapping):
    """``exp0_sft`` arguments (the DGX recipe, ``launch.json`` ``exp0_sft_args``) with the values of path flags
    replaced through ``mapping`` (local path -> remote path); every --chains/--checkpoint/--base/--out-dir/--val-chains
    value must be mapped (a DGX path on the remote would fail late or read the wrong file)."""
    out, path_flags = list(argv), ('--chains', '--checkpoint', '--base', '--out-dir', '--val-chains')
    for i, arg in enumerate(out[:-1]):
        if arg in path_flags:
            if out[i + 1] not in mapping:
                raise ValueError(f'{arg} {out[i + 1]} has no remote copy')
            out[i + 1] = mapping[out[i + 1]]
    return out


def run_remote_sft(args):
    """Run the DGX SFT recipe on a Colab G4 VM: exactly the ``exp0_sft`` arguments the DGX preflight
    (``ops/grpo-v4/launch-sft.sh`` with PREFLIGHT_ONLY=1: input identity, held exclusion; ``--launch-json``) recorded,
    with paths mapped to the VM's copies, plus ``--resume`` and the learner-state saves the DGX pulls every
    ``--state-sync-minutes``. The finished checkpoint arrives verified (``pull_verified``) or not at all."""
    import threading
    cfg = host_config('colab_g4', host=args.host)
    host = cfg['host']
    if not host:
        raise RuntimeError('run-sft requires --host or NATLANG_COLAB_HOSTS')
    launch = json.loads(Path(args.launch_json).read_text())
    recipe = shlex.split(launch['exp0_sft_args'])
    local_work = Path(args.work).resolve()
    local_work.mkdir(parents=True, exist_ok=True)
    remote_run = f'{cfg["work_root"]}/sft-{args.run_name}'
    ssh_host(host, shlex.join(['mkdir', '-p', cfg['checkpoints'], remote_run, f'{remote_run}/learner-state']))
    if args.commit:
        ssh_host(host, f'git -C {shlex.quote(cfg["repo"])} fetch -q origin && '
                       f'git -C {shlex.quote(cfg["repo"])} checkout -q -f {shlex.quote(args.commit)}', timeout=300)
    value = lambda flag: recipe[recipe.index(flag) + 1]
    mapping = {value('--base'): cfg['base'], value('--out-dir'): remote_run}
    chains_remote = f'{remote_run}/chains.pkl'
    sync_to_remote(value('--chains'), chains_remote, host)
    if remote_identity(host, chains_remote)[1] != launch['chains_sha256']:
        raise RuntimeError(f'the chains copied to {host} do not hash to {launch["chains_sha256"]}')
    mapping[value('--chains')] = chains_remote
    start_sha = launch['start_checkpoint_sha256']
    remote_ckpt = f'{cfg["checkpoints"]}/{start_sha}.pt'
    if not ssh_host(host, shlex.join(['test', '-f', remote_ckpt]) + ' && echo ok', check=False).strip():
        sync_to_remote(value('--checkpoint'), f'{remote_ckpt}.partial', host)
        if remote_identity(host, f'{remote_ckpt}.partial')[1] != start_sha:
            raise RuntimeError(f'the start checkpoint copied to {host} does not hash to {start_sha}')
        ssh_host(host, shlex.join(['mv', f'{remote_ckpt}.partial', remote_ckpt]))
    mapping[value('--checkpoint')] = remote_ckpt
    vals = [recipe[i + 1] for i, a in enumerate(recipe[:-1]) if a == '--val-chains']
    for idx, vpath in enumerate(vals):
        rv = f'{remote_run}/val-{idx}-{Path(vpath).name}'
        sync_to_remote(vpath, rv, host)
        mapping[vpath] = rv
    local_state = local_work / 'learner-state'
    if (local_state / 'learner_manifest.json').exists():     # exact resume across a preemption (DGX holds the state)
        sync_to_remote(f'{local_state}/', f'{remote_run}/learner-state/', host)
    sft_cmd = [cfg['python'], f'{cfg["repo"]}/scripts/neuralese_data/exp0_sft.py', *remap_sft_args(recipe, mapping),
               *([] if '--resume' in recipe else ['--resume']), '--state-save-minutes', str(args.state_sync_minutes)]
    admitted = [cfg['python'], cfg['admission'], 'run', '--vram-gib', str(args.vram_gib), '--host-ram-gib',
                str(args.host_ram_gib), '--label', f'sft-{args.run_name}', '--wait', '600', '--workdir', cfg['repo'],
                '--', *sft_cmd]
    stop_sync = threading.Event()
    sync_interval = max(60.0, float(args.state_sync_minutes) * 60.0)

    def pull_state(attempts):
        sync_from_remote(f'{remote_run}/learner-state/', f'{local_state}/', host, attempts=attempts)
        sync_from_remote(f'{remote_run}/train.jsonl', str(local_work / 'train.jsonl'), host, attempts=1)

    def sync_loop():
        while not stop_sync.wait(sync_interval):
            try:
                pull_state(2)
            except Exception as err:
                log(event='Colab SFT state sync warning', host=host, error=repr(err)[:200])

    sync_thread = threading.Thread(target=sync_loop, daemon=True)
    sync_thread.start()
    started, rc = time.time(), 1
    try:
        rc = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', host,
                             f'. {shlex.quote(cfg["env"])} && {shlex.join(admitted)}'], timeout=24 * 3600).returncode
    finally:
        stop_sync.set()
        sync_thread.join(timeout=30)
        try:      # the latest learner state, so the DGX can resume a preempted run
            pull_state(3)
        except Exception as err:
            log(event='Colab SFT post-run pull warning', error=repr(err)[:200])
    sha = None
    if rc == 0:
        try:
            sha = pull_verified(f'{remote_run}/checkpoint.pt', local_work / 'checkpoint.pt', host)
        except Exception as err:
            rc = 1
            log(event='Colab SFT checkpoint pull failed', host=host, error=repr(err)[:300])
    log(event='Colab G4 SFT finished', host=host, returncode=rc, seconds=round(time.time() - started, 1),
        checkpoint_sha256=sha)
    return rc


def serve_checkpoint(args):
    """Serve a pruned native checkpoint on Pop or Colab G4 and wait until it attests."""
    if str(getattr(args, 'host_profile', 'pop')).lower().replace('-', '_') == 'colab_g4':
        return serve_colab_g4(args)
    sha, commit = args.policy_id, args.commit
    sglang, sglang_format = pop_sglang(args.kv_cache_dtype, args.max_running)
    release, adapters, rank = adapter_release(args)   # a program fit served by default (TUNING_RESTORE_20261117.md)
    previous = read_json(STATE)
    if (previous and previous['policy_id'] == sha and previous.get('code') == commit and previous.get('sglang') == sglang
            and previous.get('adapters') == adapters and attested(previous['endpoint'], sha, adapters)):
        log(event='Pop actor already serves the checkpoint', endpoint=previous['endpoint'])
        return previous
    host = args.bind_host or pop_host_address()
    endpoint = f'http://{host}:{args.port}'
    started = time.time()
    target = f'{POP_CHECKPOINTS}/{sha}.pt'
    work = f'{POP_EVAL_ROOT}/held-actor-{sha[:12]}'
    stop_pop_actors()
    ssh(shlex.join(['mkdir', '-p', POP_CHECKPOINTS, work]))
    prune_pop_newest(f'{POP_CHECKPOINTS}/*.pt', POP_KEEP, protect=(target,))
    prune_pop_newest(f'{POP_EVAL_ROOT}/held-actor-*', POP_KEEP, protect=(work,))
    checkpoint_on_pop(args.checkpoint, sha)
    copied = time.time()
    runtime = pop_code(commit)
    prune_pop_newest(f'{POP_EVAL_ROOT}/code-*', POP_KEEP, protect=(runtime,))
    name = f'popheld-{sha[:12]}'
    job = {'mode': 'held-eval', 'arm': 'held', 'name': name, 'port': args.port, 'host': host, 'runtime': runtime,
           'code': {'schema': 'natlang.code-provenance/1', 'git_commit': commit, 'dirty': False}, 'base': POP_BASE,
           'image': IMAGE, 'step': args.step, 'source_sha256': sha, 'stage': POP_CHECKPOINTS, 'view': f'{sha}.pt',
           'view_sha256': sha, 'work': work, 'min_free_bytes': 4 * 2**30, 'actor_export_format': sglang_format,
           'memory': f'{RL_MEMORY_GB}g', 'engine': 'sglang', 'sglang': sglang}
    ssh(shlex.join(['rm', '-f', f'{work}/driver.py']))
    subprocess.run(['scp', '-q', str(REPO / 'scripts/neuralese_data/pop_actor_driver.py'), f'{POP}:{work}/driver.py'],
                   check=True, timeout=120)
    if release:
        send_adapter_release(job, work, release, adapters, rank)
    _pop_upload_json(job, f'{work}/job.json')
    unit = f'{name}-{int(time.time())}'
    # Pop's 15 GB leave 10.9-11.5 GiB admissible (10.9 with the gdm greeter and slab after a desktop logout,
    # 2026-11-19: 10.5 + the 0.5 reserve waited out its admission). The engine runs in a docker container capped by
    # its own --memory, so this unit's budget is accounting for the driver, not the engine's limit.
    ssh(shlex.join(['/usr/bin/python3', POP_REPO + '/scripts/memory_ledger.py', 'run', '--unit', unit,
                    '--budget-gb', pop_sglang_budget_gb(), '--reserve-gb', str(POP_RESERVE_GB), '--class', 'experiment',
                    '--no-resumable', '--wait', str(POP_ADMISSION_WAIT), '--workdir', work, '--', '/usr/bin/python3', f'{work}/driver.py', f'{work}/job.json']),
        timeout=POP_ADMISSION_WAIT + 300)
    result = {'endpoint': endpoint, 'unit': unit, 'container': name, 'policy_id': sha, 'work': work, 'runtime': runtime,
              'code': commit, 'engine': 'sglang', 'sglang': sglang, 'copy_seconds': round(copied - started, 1),
              'adapters': adapters}
    write_json(STATE, result)
    deadline = time.monotonic() + args.ready_seconds
    while time.monotonic() < deadline:
        if pop_unit_state(unit).get('ActiveState') in ('inactive', 'failed'):
            tail = ssh(shlex.join(['journalctl', '--user', '-u', unit + '.service', '-n', '20', '--no-pager']), check=False)
            raise RuntimeError('Pop actor unit exited before readiness: ' + tail[-1200:])
        if attested(endpoint, sha, adapters):
            result['ready_seconds'] = round(time.time() - started, 1)
            write_json(STATE, result)
            log(event='Pop actor ready', **result)
            return result
        time.sleep(5)
    raise RuntimeError(f'Pop actor did not attest {sha[:12]} at {endpoint} within {args.ready_seconds} s')


def serving_endpoint(policy, export_format, *, profile='pop', host=None):
    """Remote actor endpoint when it serves checkpoint ``policy`` in ``export_format``, else None."""
    state = read_json(colab_state_path(host) if str(profile).lower().replace('-', '_') == 'colab_g4' and host else STATE) or {}
    try:
        with urllib.request.urlopen(state['endpoint'] + '/v1/neuralese/info', timeout=10) as response:
            binding = json.loads(response.read()).get('rl_policy_binding') or {}
    except (KeyError, OSError, ValueError):
        return None
    served, served_format = binding.get('checkpoint_sha256'), (binding.get('actor_export') or {}).get('format')
    return state['endpoint'] if served == policy and served_format == export_format else None


def complete(result_path):
    try:
        return bool(json.loads(Path(result_path).read_text()).get('complete'))
    except (OSError, ValueError):
        return False


def held_eval_job(args):
    """One held evaluation on Pop from ``--spec`` (grpo ``--held-remote pop``). The Colab G4 held series is superseded
    (async-train §1: the held series runs on the DGX), so a spec naming another host profile is refused."""
    spec = json.loads(Path(args.spec).read_text())
    started, code, serving = time.time(), 1, None
    signal.signal(signal.SIGTERM, lambda *_: None)
    profile = str(spec.get('host_profile') or 'pop').lower().replace('-', '_')
    env = {**os.environ, 'NATLANG_REPO': str(REPO), 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}'}
    python = [sys.executable, '-m', 'natlang_neuralese.rl.held_eval']
    with open(spec['log'], 'a') as handle:
        try:
            if profile != 'pop':
                raise RuntimeError(f'held evaluations run on Pop or the DGX, not {profile}: the Colab G4 held series is '
                                   'superseded by the DGX series (plans/neuralese/async-train.md §1)')
            endpoint = serving_endpoint(spec['policy'], spec['format'])
            if endpoint is None:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), *spec['serve']], cwd=REPO, env=env,
                               stdout=handle, stderr=subprocess.STDOUT, timeout=4 * 3600)
                endpoint = serving_endpoint(spec['policy'], spec['format'])
            if endpoint is None:
                raise RuntimeError(f'Pop does not serve {spec["policy"][:12]}')
            serving = time.time() - started
            code = subprocess.run([*python, 'measure', *spec['measure'], '--endpoint', f'{endpoint}={spec["slots"]}'],
                                  cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            compare = spec.get('compare')
            if compare:
                Path(compare[compare.index('--out') + 1]).unlink(missing_ok=True)
            if code == 0 and compare and all(complete(path) for path in compare[:2]):
                subprocess.run([*python, 'compare', *compare], cwd=REPO, env=env, stdout=handle,
                               stderr=subprocess.STDOUT)
        except Exception as error:
            print(json.dumps({'event': 'held eval job failed', 'error': repr(error)[:500]}), file=handle, flush=True)
        finally:
            write_json(spec['done'], {'unit': spec.get('unit'), 'policy': spec['policy'], 'returncode': code,
                                      'serve_seconds': None if serving is None else round(serving),
                                      'seconds': round(time.time() - started), 'served': serving is not None})
    return code


def stop(args=None):
    if args is not None and str(getattr(args, 'host_profile', 'pop')).lower().replace('-', '_') == 'colab_g4':
        targets = [args.host] if getattr(args, 'host', None) else colab_hosts()
        for h in targets:
            stop_colab_g4(h)
        return
    state = read_json(STATE)
    if state:
        stop_unit(state)
        ssh(shlex.join(['docker', 'rm', '-f', state['container']]), check=False, timeout=60)
        work = state['work']
        ssh(shlex.join(['docker', 'run', '--rm', '-v', f'{work}:/work', 'alpine:3.21', 'sh', '-c',
                            'rm -rf ' + ' '.join(f'/work/{name}' for name in ('inductor', 'triton', 'hf', 'cuda'))]),
                check=False, timeout=120)
        STATE.unlink(missing_ok=True)
        log(event='Pop RL actor stopped', unit=state['unit'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='action', required=True)
    s = commands.add_parser('start')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--policy-id', required=True)
    s.add_argument('--port', type=int, default=RL_PORT)
    s.add_argument('--bind-host')
    s.add_argument('--ready-seconds', type=int, default=900)
    s.add_argument('--engine', choices=('sglang', 'native'), default='sglang')
    s.add_argument('--prepare-wait', type=int, default=900)
    s.add_argument('--adapter-release')
    c = commands.add_parser('serve-checkpoint', help='serve a pruned native checkpoint on Pop or Colab G4')
    for sub in (s, c):
        sub.add_argument('--host-profile', choices=('pop', 'colab_g4'), default='pop')
        sub.add_argument('--host', help='explicit SSH host alias (default: Pop or first Colab G4 host)')
        sub.add_argument('--kv-cache-dtype', choices=('fp8_e4m3', 'auto'), default=POP_KV_CACHE_DTYPE)
        sub.add_argument('--max-running', type=int, default=POP_MAX_RUNNING, help='Pop running cap (a Colab actor '
                         'runs 128 alone, 64 with --beside-learner)')
        sub.add_argument('--beside-learner', action='store_true',
                         help='Colab G4: cap the actor (24 GB KV, 64 running) to share the GPU with the SFT learner')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--policy-id', required=True)
    c.add_argument('--commit', required=True)
    c.add_argument('--step', type=int, default=0)
    c.add_argument('--port', type=int, default=RL_PORT)
    c.add_argument('--bind-host')
    c.add_argument('--ready-seconds', type=int, default=1800)
    c.add_argument('--adapter-release')
    f = commands.add_parser('prefetch')
    f.add_argument('--checkpoint', required=True)
    f.add_argument('--policy-id', required=True)
    f.add_argument('--step', type=int, default=0)
    h = commands.add_parser('held-eval')
    h.add_argument('--spec', required=True)
    rs = commands.add_parser('run-sft', help='run the DGX SFT recipe on a Colab G4 VM with periodic state sync back')
    rs.add_argument('--host', required=True)
    rs.add_argument('--run-name', required=True)
    rs.add_argument('--work', required=True, help='local DGX work directory (receives learner-state and checkpoint.pt)')
    rs.add_argument('--launch-json', required=True,
                    help="the DGX preflight's launch.json (launch-sft.sh PREFLIGHT_ONLY=1): its exp0_sft_args run")
    rs.add_argument('--commit')
    rs.add_argument('--state-sync-minutes', type=float, default=15.0)
    rs.add_argument('--vram-gib', type=float, default=G4_VRAM_GIB - G4_ACTOR_BESIDE_LEARNER['vram_gib'] - 1.0,
                    help='admission claim (default: the GPU less a capped actor beside it)')
    rs.add_argument('--host-ram-gib', type=float, default=40.0)
    rc = commands.add_parser('reachable', help='list reachable remote hosts for a profile')
    rc.add_argument('--host-profile', choices=('pop', 'colab_g4'), default='colab_g4')
    rc.add_argument('--hosts', help='comma-separated SSH host aliases')
    st_cmd = commands.add_parser('stop')
    st_cmd.add_argument('--host-profile', choices=('pop', 'colab_g4'), default='pop')
    st_cmd.add_argument('--host')
    status_cmd = commands.add_parser('status')
    status_cmd.add_argument('--host-profile', choices=('pop', 'colab_g4'), default='pop')
    status_cmd.add_argument('--host')
    p = commands.add_parser('prepare-view', help=argparse.SUPPRESS)
    for flag in ('--checkpoint', '--out', '--receipt'):
        p.add_argument(flag, required=True)
    p.add_argument('--expect-sha256')
    args = parser.parse_args(argv)
    if args.action == 'reachable':
        hosts = colab_hosts(args.hosts) if args.host_profile == 'colab_g4' else ([POP] if POP else [])
        up = reachable_hosts(args.host_profile, hosts)
        print(json.dumps({'profile': args.host_profile, 'configured': hosts, 'reachable': up}))
        return 0
    if args.action == 'status':
        if args.host_profile == 'colab_g4':
            hosts = [args.host] if args.host else colab_hosts()
            print(json.dumps({'profile': 'colab_g4', 'states': {h: read_json(colab_state_path(h)) for h in hosts}}))
        else:
            print(json.dumps({'state': read_json(STATE)}))
        return 0
    if args.action == 'prepare-view':
        prepare_view(args)
        return 0
    if args.action == 'serve-checkpoint':
        print(json.dumps(serve_checkpoint(args)), flush=True)
        return 0
    if args.action == 'prefetch':
        prefetch(args)
        return 0
    if args.action == 'held-eval':
        return held_eval_job(args)
    if args.action == 'run-sft':
        return run_remote_sft(args)
    try:
        if args.action == 'start':
            print(json.dumps(start(args)), flush=True)
        else:
            stop(args)
    except BaseException as error:
        log(event=f'Remote RL actor {args.action} failed', error=repr(error)[:2000])
        if args.action == 'start':
            try:
                stop(args)
            except Exception as cleanup:
                log(event='Remote RL actor cleanup failed', error=repr(cleanup)[:500])
        if isinstance(error, KeyboardInterrupt):
            raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
