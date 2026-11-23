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

serve-checkpoint serves a pruned native checkpoint (``learner_state.export_merged_checkpoint`` / GRPO's ``step-N/checkpoint.pt``:
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
import shutil
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


from natlang_neuralese.common.execution_profile import PROFILE_ALIASES, colab_layout  # noqa: E402


def profile_key(profile):
    """A host profile's name: ``pop`` or ``colab`` (``colab_g4``, the old name, is accepted)."""
    key = str(profile).lower().replace('-', '_')
    return PROFILE_ALIASES.get(key, key)

COLAB_ROOT = os.environ.get('NATLANG_COLAB_ROOT', '/content/natlang-work')
COLAB_SGLANG_EXPORT_FORMAT = 'sglang-colab-bf16-vendor-projection'
COLAB_MAX_RUNNING = 128
COLAB_MEM_FRACTION = 0.88


def pop_sglang(kv_cache_dtype=POP_KV_CACHE_DTYPE, max_running=POP_MAX_RUNNING, profile='pop',
               weight_dtype='bfloat16'):
    """(job['sglang'] settings, actor export format) of a remote SGLang engine (Pop or Colab G4).
    ``weight_dtype`` is ``'bfloat16'`` (held and GRPO) or ``'fp8'`` (SFT collection actors, async-train §3)."""
    if weight_dtype not in ('bfloat16', 'fp8'):
        raise ValueError(f'weight_dtype must be bfloat16 or fp8, got {weight_dtype!r}')
    fp8 = kv_cache_dtype != 'auto'
    w_suffix = '-w-fp8' if weight_dtype == 'fp8' else ''
    w_extra = {'weight_dtype': 'fp8'} if weight_dtype == 'fp8' else {}
    if profile_key(profile) == 'colab':
        cap = max_running if max_running is not None else COLAB_MAX_RUNNING
        settings = {'kv_cache_dtype': kv_cache_dtype, 'max_running': cap, 'mem_fraction': COLAB_MEM_FRACTION,
                    'max_total_tokens': 2097152 if fp8 else 1048576, **w_extra}
        return settings, COLAB_SGLANG_EXPORT_FORMAT + (f'-kv-{kv_cache_dtype}' if fp8 else '') + w_suffix
    settings = {'kv_cache_dtype': kv_cache_dtype, 'max_running': max_running, 'mem_fraction': POP_MEM_FRACTION,
                'max_total_tokens': 524288 if fp8 else 262144, **w_extra}
    return settings, SGLANG_EXPORT_FORMAT + (f'-kv-{kv_cache_dtype}' if fp8 else '') + w_suffix


def ensure_colab_mps(host):
    """Ensure the NVIDIA MPS daemon is running on a Colab G4 VM so an SFT learner and capped collection actor share
    SMs concurrently (async-train §1). Non-fatal when MPS is unavailable in a restricted container; opt out with
    ``NATLANG_COLAB_MPS=0``."""
    if os.environ.get('NATLANG_COLAB_MPS', '1') == '0':
        return {'host': host, 'mps': 'disabled'}
    out = ssh_host(host, 'pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1 && echo running '
                         '|| (nvidia-cuda-mps-control -d >/dev/null 2>&1 && echo started || echo unavailable)',
                   check=False, timeout=30).strip()
    status = out.splitlines()[-1] if out else 'unavailable'
    log(event='Colab G4 MPS checked', host=host, mps=status)
    return {'host': host, 'mps': status}


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
    """Return reachable SSH hosts for ``profile`` ('pop' or 'colab')."""
    key = profile_key(profile)
    candidates = list(hosts) if hosts else (colab_hosts() if key == 'colab' else ([POP] if POP else []))
    return [h for h in candidates if host_reachable(h)]


def ssh_options(host):
    """``ssh -G host``: the effective client options of SSH alias ``host`` (lower-case keys)."""
    out = subprocess.run(['ssh', '-G', host], capture_output=True, text=True, check=True,
                         stdin=subprocess.DEVNULL).stdout
    return dict(line.split(' ', 1) for line in out.splitlines() if ' ' in line)


def host_address(host):
    """Resolve the tailscale IP/hostname configured for SSH alias ``host``."""
    return ssh_options(host)['hostname']


# A host reached only through its SSH ProxyCommand (a Colab runtime: `colab ssh --proxy-mode`, which admits one SSH
# connection, so every command and forward multiplexes over its ControlMaster) has no network address the DGX can
# open: its servers are forwarded to DGX 127.0.0.1:(port + COLAB_FORWARD_OFFSET), e.g. 8570 -> 18731 (front) and
# 9570 -> 19731 (SGLang), the ports the session's tunnel unit forwards.
COLAB_FORWARD_OFFSET = 10161


def _listening(port):
    import socket
    with socket.socket() as sock:
        sock.settimeout(2)
        return sock.connect_ex(('127.0.0.1', int(port))) == 0


def forward_port(host, local, remote):
    """Forward DGX ``127.0.0.1:local`` to ``host``'s ``127.0.0.1:remote`` over the host's ControlMaster (``ssh -O
    forward``: no second connection); a port already forwarded (the tunnel unit's) is kept."""
    if _listening(local):
        return
    subprocess.run(['ssh', '-O', 'forward', '-L', f'{local}:127.0.0.1:{remote}', host], check=True,
                   capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)


def remote_endpoint(host, port, bind_host=None):
    """The DGX-side URL of a server listening on ``host`` at ``port``: ``bind_host`` when given; the host's network
    address when SSH names one (Pop over Tailscale); for a ProxyCommand-only host the forwarded loopback port."""
    if bind_host:
        return f'http://{bind_host}:{port}'
    options = ssh_options(host)
    if options.get('proxycommand', 'none') == 'none':
        return f'http://{options["hostname"]}:{port}'
    forward_port(host, int(port) + COLAB_FORWARD_OFFSET, int(port))
    return f'http://127.0.0.1:{int(port) + COLAB_FORWARD_OFFSET}'


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


def drive_enabled(cfg):
    """Colab transfers go through Google Drive (``natlang/blobs/<sha256>``) unless NATLANG_COLAB_DRIVE=0: the VM
    downloads inside Google's network, the DGX uploads at its full uplink (about 2x the tunnel) and only once per
    blob, so a replacement VM or a later round finds it there."""
    return cfg.get('profile') == 'colab' and os.environ.get('NATLANG_COLAB_DRIVE', '1') != '0'


def drive_token_to(host, cfg):
    """A fresh Drive access token in ROOT/drive-token on the VM (ssh stdin, mode 0600; never on a command line)."""
    from scripts import drive_tier
    path = f'{cfg["root"]}/drive-token'
    ssh_host(host, f'umask 077 && cat > {shlex.quote(path)}.tmp && mv {shlex.quote(path)}.tmp {shlex.quote(path)}',
             input=drive_tier.Drive().token(fresh=True), timeout=120)
    return path


def drive_vm(host, cfg, *args, timeout=3600):
    """``scripts/drive_tier.py`` on the VM (its checkout's copy) with a fresh token; returns its last JSON line."""
    token = drive_token_to(host, cfg)
    out = ssh_host(host, shlex.join([cfg['python'], f'{cfg["repo"]}/scripts/drive_tier.py', '--token-file', token,
                                     *args]), timeout=timeout)
    return json.loads(out.strip().splitlines()[-1])


def ship_file(local_path, remote_path, host, cfg, sha256):
    """``remote_path`` on ``host`` holding ``local_path``'s bytes (``sha256``). Through Drive when enabled (the blob
    uploaded once, then fetched and hash-checked on the VM), else, or when that fails, over the tunnel. Returns how."""
    if drive_enabled(cfg):
        try:
            from scripts import drive_tier
            started, key = time.time(), f'blobs/{sha256}'
            drive = drive_tier.Drive()
            uploaded = None
            if not drive.lookup(key):
                uploaded = drive.upload(local_path, key)['seconds']
            ssh_host(host, shlex.join(['mkdir', '-p', str(PurePosixPath(remote_path).parent)]))
            got = drive_vm(host, cfg, 'get', key, remote_path, '--sha256', sha256)
            if got.get('event') == 'got':
                log(event='Colab file via Drive', host=host, path=remote_path, upload_seconds=uploaded,
                    download_seconds=got.get('seconds'), seconds=round(time.time() - started, 1))
                return 'drive'
        except Exception as error:   # any Drive failure falls back to the tunnel
            log(event='Colab Drive transfer failed: using the tunnel', host=host, error=repr(error)[:300])
    sync_to_remote(local_path, remote_path, host)
    return 'tunnel'


def drive_backup_remote(host, cfg, remote_dir, prefix):
    """The VM puts ``remote_dir`` (a learner-state snapshot, factors) on Drive under ``natlang/<prefix>``, so a lost
    VM loses nothing its last save held. Best effort: a failure is logged, never raised."""
    if not drive_enabled(cfg):
        return None
    try:
        started = time.time()
        out = drive_vm(host, cfg, 'put-dir', remote_dir, prefix)
        log(event='Colab state on Drive', host=host, prefix=prefix, files=out.get('files'),
            seconds=round(time.time() - started, 1))
        return out
    except Exception as error:
        log(event='Colab Drive backup warning', host=host, prefix=prefix, error=repr(error)[:300])
        return None


def colab_checkout(host, cfg, commit):
    """The VM's natlang checkout (bootstrap's clone) at ``commit``, verified: HEAD resolves to the commit and no tracked
    file differs; ts-host is rebuilt when its tree differs from the one last built there (the front serves it)."""
    repo = shlex.quote(cfg['repo'])
    out = ssh_host(host, f'git -C {repo} fetch -q origin && git -C {repo} checkout -q -f {shlex.quote(commit)} && '
                         f'git -C {repo} rev-parse HEAD {shlex.quote(commit + "^{commit}")} HEAD:ts-host && '
                         f'git -C {repo} status --porcelain --untracked-files=no | head -5', timeout=600).split()
    if len(out) != 3 or out[0] != out[1]:
        raise RuntimeError(f'{host}: the checkout is not clean at {commit}: {" ".join(out)[:300]}')
    marker = f'{cfg["root"]}/ts-host-built.tree'
    if ssh_host(host, f'cat {shlex.quote(marker)} 2>/dev/null || true', check=False).strip() != out[2]:
        ssh_host(host, f'. {shlex.quote(cfg["env"])} && cd {repo}/ts-host && npm ci --no-audit --no-fund -q && '
                       f'npm run -s build:node && echo {out[2]} > {shlex.quote(marker)}', timeout=1800)
        log(event='Colab ts-host rebuilt', host=host, commit=out[0], tree=out[2])
    return out[0]


# the VM-side merge of a published round policy (async-train §2.5): ``merge_factors_to_checkpoint`` into a file named
# checkpoint.pt (torch.save names its archive after the file, so the bytes equal the DGX publication's only under
# that name), checked against the DGX's sha256, then renamed to CHECKPOINTS/<sha>.pt; a mismatch leaves nothing
_REMOTE_MERGE = (
    'import json, os, shutil, sys\n'
    'sys.path[:0] = [sys.argv[1] + "/training/neuralese", sys.argv[1]]\n'
    'from natlang_neuralese.train.learner_state import merge_factors_to_checkpoint\n'
    'base, factors, stage, target, expect = sys.argv[2:7]\n'
    'shutil.rmtree(stage, ignore_errors=True)\n'
    'os.makedirs(stage)\n'
    'try:\n'
    '    rec = merge_factors_to_checkpoint(base, factors, os.path.join(stage, "checkpoint.pt"))\n'
    'except ValueError as error:\n'
    '    shutil.rmtree(stage, ignore_errors=True)\n'
    '    print(json.dumps({"ok": False, "error": str(error)[:300]}))\n'
    '    sys.exit(0)\n'
    'ok = rec["checkpoint_sha256"] == expect\n'
    'if ok:\n'
    '    os.replace(os.path.join(stage, "checkpoint.pt"), target)\n'
    'shutil.rmtree(stage, ignore_errors=True)\n'
    'print(json.dumps({"ok": ok, "checkpoint_sha256": rec["checkpoint_sha256"]}))\n')


def remote_merge(host, cfg, *, base_sha, factors_dir, expect_sha):
    """Merge a round's factors over its base checkpoint on ``host`` (both already there or the factors shipped,
    ~0.5 GB instead of the 3.4 GB policy) into ``CHECKPOINTS/<expect_sha>.pt``; True when the result hashes to
    ``expect_sha``. The factors stay on the VM under ROOT/factors/<sha> for the next merge."""
    manifest = json.loads((Path(factors_dir) / 'policy_manifest.json').read_text())
    if manifest.get('base_checkpoint_sha256') != base_sha:
        raise ValueError(f'{factors_dir} was trained from {manifest.get("base_checkpoint_sha256")}, not {base_sha}')
    remote_factors = f'{cfg["root"]}/factors/{manifest["adapter_factors_sha256"]}'
    base = f'{cfg["checkpoints"]}/{base_sha}.pt'
    if not ssh_host(host, f'test -f {shlex.quote(base)} && echo ok', check=False).strip():
        return False
    if not ssh_host(host, f'test -f {shlex.quote(remote_factors + "/policy_manifest.json")} && echo ok',
                    check=False).strip():
        staged = remote_factors + '.partial'
        for name in ('adapter_factors_fp32.safetensors', 'policy_manifest.json'):
            sync_to_remote(str(Path(factors_dir) / name), f'{staged}/{name}', host)
        ssh_host(host, shlex.join(['mv', '-T', staged, remote_factors]))
    started = time.time()
    out = ssh_host(host, shlex.join([cfg['python'], '-c', _REMOTE_MERGE, cfg['repo'], base, remote_factors,
                                     f'{cfg["checkpoints"]}/.merge-{expect_sha[:16]}',
                                     f'{cfg["checkpoints"]}/{expect_sha}.pt', expect_sha]), timeout=3600)
    result = json.loads(out.strip().splitlines()[-1])
    log(event='Colab merged policy', host=host, expect=expect_sha, seconds=round(time.time() - started, 1), **result)
    return bool(result.get('ok'))


def ensure_remote_checkpoint(host, cfg, local_checkpoint, sha):
    """``CHECKPOINTS/<sha>.pt`` on ``host``, hashing to ``sha``: kept when there; else merged on the VM when the DGX
    checkpoint is an async round's publication (``published.json`` beside it names its base and factors, and the VM
    holds the base: the previous round trained from it); else shipped. Returns how it got there."""
    target = f'{cfg["checkpoints"]}/{sha}.pt'
    ssh_host(host, shlex.join(['mkdir', '-p', cfg['checkpoints']]))
    if ssh_host(host, shlex.join(['test', '-f', target]) + ' && echo ok', check=False).strip():
        if remote_identity(host, target)[1] == sha:
            return 'present'
        ssh_host(host, shlex.join(['rm', '-f', target]))     # a file not hashing to its name (an old VM merge)
    published = read_json(Path(local_checkpoint).with_name('published.json')) or {}
    merge = published.get('merge') or {}
    if (published.get('checkpoint_sha256') == sha and merge.get('base_checkpoint_sha256') and published.get('factors_dir')
            and os.environ.get('NATLANG_COLAB_REMOTE_MERGE', '1') != '0'):
        try:
            if remote_merge(host, cfg, base_sha=merge['base_checkpoint_sha256'], factors_dir=published['factors_dir'],
                            expect_sha=sha):
                return 'merged'
        except (OSError, ValueError, RuntimeError) as error:
            log(event='Colab merge failed: shipping the checkpoint', host=host, error=repr(error)[:300])
    started = time.time()
    via = ship_file(str(local_checkpoint), f'{target}.partial', host, cfg, sha)
    if remote_identity(host, f'{target}.partial')[1] != sha:
        raise RuntimeError(f'the checkpoint copied to {host} does not hash to {sha}')
    ssh_host(host, shlex.join(['mv', f'{target}.partial', target]))
    log(event='Colab checkpoint shipped', host=host, sha=sha, via=via, seconds=round(time.time() - started, 1))
    return 'shipped' if via == 'tunnel' else 'drive'


def colab_state_path(host):
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(host))
    return ROOT / f'colab-actor-{safe}.json'


def host_config(profile='pop', *, host=None):
    """Unified remote-host layout for ``pop`` and ``colab``."""
    key = profile_key(profile)
    if key == 'colab':
        chosen = host or (colab_hosts()[0] if colab_hosts() else None)
        return {
            'profile': 'colab',
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
    a Wi-Fi stall on 2026-11-19 held a copy at 2.5 of 3.5 GB for 35 min. NATLANG_POP_COPY_ATTEMPTS=1 opts out.
    Compressed on the wire (``-z``: rsync >= 3.2 negotiates zstd): an SFT mix's chains.pkl shrinks ~12x, a BF16
    checkpoint ~20%, and the Colab link runs at 2.4 MB/s up, 4.8 down (2026-11-22). NATLANG_COPY_COMPRESS=0 opts out."""
    attempts = attempts or int(os.environ.get('NATLANG_POP_COPY_ATTEMPTS', '5'))
    compress = [] if os.environ.get('NATLANG_COPY_COMPRESS', '1') == '0' else ['-z']
    for attempt in range(attempts):
        try:
            subprocess.run(['rsync', '-a', *compress, '--partial', f'--timeout={stall_seconds}', source, dest],
                           check=True, timeout=7200)
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


def stop_colab(host):
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


def colab_hardware(host):
    """The session's attested hardware: ``colab-ready.json`` ``hardware`` (bootstrap's probe: GPU name, VRAM, host
    RAM, compute capability), the compute capability asked of nvidia-smi when the ready file predates that field."""
    cfg = host_config('colab', host=host)
    ready = json.loads(ssh_host(host, shlex.join(['cat', f'{cfg["root"]}/colab-ready.json']), timeout=60))
    hardware = dict(ready.get('hardware') or {})
    if not hardware.get('vram_total_gib'):
        raise RuntimeError(f'{host}: colab-ready.json records no GPU (bootstrap the session first)')
    if not hardware.get('compute_cap'):
        cap = ssh_host(host, 'nvidia-smi --query-gpu=compute_cap --format=csv,noheader', check=False, timeout=60).strip()
        hardware['compute_cap'] = cap.splitlines()[0].strip() if cap else None
    return hardware


def colab_serve_command(cfg, *, checkpoint, sha, port, work, beside_learner, kv_cache_dtype,
                        weight_dtype='bfloat16', layout):
    """(serve argv, admission claim, sglang settings, export format) of a Colab SGLang actor sized from the session's
    hardware (``colab_layout``): alone it takes the GPU (mem fraction 0.88, 128 running); beside the SFT learner under
    MPS (async-train §1) its KV is capped at a quarter of the GPU and it claims only that share, so the learner's
    admission still fits. FP8 weights on a GPU without FP8 GEMMs (sm_80) serve BF16."""
    sizing = layout['actor_beside_learner'] if beside_learner else layout['actor_alone']
    if weight_dtype == 'fp8' and not layout['fp8']:
        log(event='Colab FP8 weights unsupported: BF16', gpu=layout['gpu_name'],
            compute_capability=layout['compute_capability'])
        weight_dtype = 'bfloat16'
    sglang, sglang_format = pop_sglang(kv_cache_dtype, sizing['max_running'], profile='colab',
                                       weight_dtype=weight_dtype)
    memory = (['--sglang-kv-gb', str(sizing['kv_gb'])] if sizing['kv_gb'] else
              ['--sglang-mem-fraction', str(sglang['mem_fraction'])])
    if sizing['kv_gb']:
        sglang = {**{k: v for k, v in sglang.items() if k != 'mem_fraction'}, 'kv_gb': sizing['kv_gb']}
    weight_args = ['--sglang-weight-dtype', 'fp8'] if weight_dtype == 'fp8' else []
    serve = [cfg['python'], f'{cfg["repo"]}/scripts/neuralese_data/serve_native_checkpoint.py',
             '--store-dir', f'{work}/store', '--checkpoint', checkpoint, '--checkpoint-sha256', sha,
             '--base', cfg['base'], '--host', '0.0.0.0', '--port', str(port), '--prompt-rendering', 'full',
             '--engine', 'sglang', '--sglang-python', cfg['sglang_python'], '--sglang-projection', 'vendor',
             '--sglang-kv-cache-dtype', sglang['kv_cache_dtype'], '--sglang-max-running', str(sglang['max_running']),
             *weight_args, *memory, '--hf-export-dir', f'{work}/sglang-hf', '--actor-export-format', sglang_format]
    return serve, {'vram_gib': sizing['vram_gib'], 'host_ram_gib': sizing['host_ram_gib']}, sglang, sglang_format


def _http_json(url, payload=None, *, host=None, timeout=600):
    """GET (``payload is None``) or POST JSON to ``url``; when ``host`` is set and ``url`` targets loopback
    (e.g. a remote SGLang server on ``127.0.0.1``), execute the HTTP request over SSH on ``host``."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    use_ssh = bool(host) and parsed.hostname in ('127.0.0.1', 'localhost', '::1')
    if use_ssh:
        script = (
            'import json, sys, urllib.request\n'
            'url, timeout, body = sys.argv[1], float(sys.argv[2]), sys.stdin.read()\n'
            'req = urllib.request.Request(url, data=(body.encode() if body else None),\n'
            '    headers=({"Content-Type": "application/json"} if body else {}))\n'
            'with urllib.request.urlopen(req, timeout=timeout) as r:\n'
            '    sys.stdout.write(r.read().decode())\n'
        )
        raw = ssh_host(host, shlex.join(['python3', '-I', '-c', script, url, str(timeout)]),
                       input=(json.dumps(payload) if payload is not None else ''), timeout=int(timeout) + 30)
        return json.loads(raw) if raw.strip() else {}
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {'Content-Type': 'application/json'} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode()
    return json.loads(raw) if raw.strip() else {}


def _default_sglang_url(front_url):
    from urllib.parse import urlparse
    parsed = urlparse(front_url)
    port = (parsed.port or 80) + 1000
    return f'http://127.0.0.1:{port}'


def _prepare_remote_rebind_export(*, host, cfg, policy_id, version, export_dir=None, factors_dir=None,
                                  base_checkpoint=None):
    """Stage a versioned HF export directory on ``host`` for in-place rebind.
    When ``factors_dir`` is given (async-train §7.1), ships only the ~130 MB / ~0.5 GB FP32 factors, merges them on
    the remote host over its copy of their base (``remote_merge``: the result must hash to ``policy_id``, the DGX
    publication's sha) and exports that; otherwise syncs ``export_dir`` directly. Returns ``(remote_export_dir, weights_sha256)``."""
    remote_export = f'{cfg["work_root"]}/sglang-hf-v{version}-{policy_id[:12]}'
    if factors_dir is not None:
        manifest = json.loads((Path(factors_dir) / 'policy_manifest.json').read_text())
        remote_merged = f'{cfg["checkpoints"]}/{policy_id}.pt'
        ssh_host(host, shlex.join(['mkdir', '-p', remote_export, cfg['checkpoints']]))
        if base_checkpoint:
            raise ValueError('the VM merges over CHECKPOINTS/<base sha>.pt (the round it trained from), not --base-checkpoint')
        if not (ssh_host(host, f'test -f {shlex.quote(remote_merged)} && echo ok', check=False).strip()
                and remote_identity(host, remote_merged)[1] == policy_id):
            if not remote_merge(host, cfg, base_sha=manifest['base_checkpoint_sha256'], factors_dir=factors_dir,
                                expect_sha=policy_id):
                raise RuntimeError(f'{host}: the merge of {factors_dir} does not hash to the published {policy_id}')
        export_script = (
            'import json, sys\n'
            'sys.path[:0] = [sys.argv[1] + "/training/neuralese", sys.argv[1]]\n'
            'from natlang_neuralese.export.native_hf import export\n'
            'ckpt, base_dir, out_hf = sys.argv[2:5]\n'
            'res = export(ckpt, out_hf, base=base_dir)\n'
            'print(json.dumps({"export_dir": str(out_hf), "weights_sha256": res["weights_sha256"]}))\n'
        )
        out = ssh_host(host, shlex.join([cfg['python'], '-c', export_script, cfg['repo'], remote_merged, cfg['base'],
                                         remote_export]), timeout=1800)
        info = json.loads(out.strip().splitlines()[-1])
        return info['export_dir'], info['weights_sha256']
    if not export_dir:
        raise ValueError('rebind_actor requires export_dir or factors_dir')
    local_export = Path(export_dir)
    if local_export.is_dir():
        ssh_host(host, shlex.join(['mkdir', '-p', remote_export]))
        sync_to_remote(f'{local_export}/', f'{remote_export}/', host)
        weights_file = local_export / 'model.safetensors'
        weights_sha = None
        if weights_file.is_file():
            from natlang_neuralese.common.hashing import sha256_file_hex
            weights_sha = sha256_file_hex(weights_file)
        return remote_export, weights_sha
    # Already a remote path on host
    _, weights_sha = remote_identity(host, f'{export_dir}/model.safetensors')
    return str(export_dir), weights_sha


def rebind_actor(*, front_url, policy_id, version, sglang_url=None, weights_sha256=None, tensor_identity=None,
                 export_dir=None, factors_dir=None, base_checkpoint=None, host=None, host_profile='colab',
                 timeout=600, checkpoint=None):
    """In-place weight update + 4-route measured attestation of a standing SGLang actor (DGX local or remote G4/Pop).
    Pause (retract) -> ``update_weights_from_disk`` -> ``/v1/neuralese/rebind-policy`` -> ``attest_all_routes`` ->
    ``continue_generation`` (always resumed in ``finally``). When ``host`` and ``factors_dir`` are provided, ships only
    FP32 LoRA factors and merges + exports on the remote host (async-train §5, §7.1). ``checkpoint``: the DGX path of
    the merged ``policy_id``; a rebound Colab actor's record (``colab-actor-<host>.json``) then names it, so a session
    the keeper replaces is served again at the policy it was rebound to (``rebound_record``)."""
    from natlang_neuralese.serve.sglang_actor import attest_all_routes
    started = time.time()
    front_url = front_url.rstrip('/')
    sglang_url = (sglang_url or _default_sglang_url(front_url)).rstrip('/')
    target_dir = str(export_dir) if export_dir is not None else None
    if host:
        cfg = host_config(host_profile, host=host)
        target_dir, measured_sha = _prepare_remote_rebind_export(
            host=host, cfg=cfg, policy_id=policy_id, version=version,
            export_dir=export_dir, factors_dir=factors_dir, base_checkpoint=base_checkpoint)
        weights_sha256 = weights_sha256 or measured_sha
    else:
        if target_dir is None:
            raise ValueError('local rebind_actor requires export_dir')
        if weights_sha256 is None and (Path(target_dir) / 'model.safetensors').is_file():
            from natlang_neuralese.common.hashing import sha256_file_hex
            weights_sha256 = sha256_file_hex(Path(target_dir) / 'model.safetensors')
    _http_json(f'{sglang_url}/pause_generation', {'mode': 'retract'}, host=host, timeout=timeout)
    try:
        upd = _http_json(f'{sglang_url}/update_weights_from_disk',
                         {'model_path': target_dir, 'flush_cache': True, 'weight_version': str(version)},
                         host=host, timeout=timeout)
        if not upd.get('success', False):
            raise RuntimeError(f'update_weights_from_disk failed on {sglang_url}: {json.dumps(upd)[:400]}')
        rebind_body = {'policy_id': policy_id, 'version': version, 'export_dir': target_dir}
        if weights_sha256 is not None:
            rebind_body['weights_sha256'] = weights_sha256
        if tensor_identity is not None:
            rebind_body['tensor_identity'] = tensor_identity
        _http_json(f'{front_url}/v1/neuralese/rebind-policy', rebind_body, host=host, timeout=timeout)
        info = _http_json(f'{front_url}/v1/neuralese/info', None, host=host, timeout=min(timeout, 120))
        expected = {'policy_id': policy_id, 'version': version, 'model_path': target_dir,
                    'weights_sha256': weights_sha256, 'tensor_identity': tensor_identity}
        ok, mismatches = attest_all_routes(info, expected)
        if not ok:
            raise RuntimeError(f'rebind_actor route attestation failed for {front_url}: {json.dumps(mismatches)[:600]}')
    finally:
        _http_json(f'{sglang_url}/continue_generation', {}, host=host, timeout=min(timeout, 120))
    result = {'front_url': front_url, 'sglang_url': sglang_url, 'host': host, 'policy_id': policy_id,
              'version': version, 'export_dir': target_dir, 'weights_sha256': weights_sha256,
              'routes_attested': True, 'seconds': round(time.time() - started, 2)}
    if host and profile_key(host_profile) == 'colab':
        state_file = colab_state_path(host)
        record = rebound_record(read_json(state_file), result, checkpoint)
        if record is not None:
            write_json(state_file, record)
    log(event='Actor rebound in place', **result)
    return result


def rebound_record(record, rebound, checkpoint):
    """A Colab actor record (``serve_colab``) after an in-place rebind: its policy and serve arguments name the rebound
    policy (``checkpoint``, its DGX path), so ``restore_colab_actor`` and the session keeper's re-serve after a
    replacement serve that policy, not the one the actor first loaded. None (the record stays) without a record, without
    ``checkpoint`` (a rebind that names no DGX checkpoint cannot be served again), or for another front port."""
    from urllib.parse import urlparse
    serve = (record or {}).get('serve_args')
    if not serve or not checkpoint or urlparse(rebound['front_url']).port != int(serve.get('port') or 0):
        return None
    return {**record, 'policy_id': rebound['policy_id'],
            'serve_args': {**serve, 'checkpoint': str(checkpoint), 'policy_id': rebound['policy_id']},
            'rebound': {k: rebound[k] for k in ('policy_id', 'version', 'export_dir', 'weights_sha256')} |
                       {'at': time.time(), 'from_policy_id': record.get('policy_id')}}


def serve_colab(args):
    """Serve a merged BF16 checkpoint on a Colab G4 VM under scripts/colab/admission.py (a VRAM/host-RAM claim that
    lasts while the actor runs) and wait until it attests the checkpoint."""
    cfg = host_config('colab', host=getattr(args, 'host', None))
    host = cfg['host']
    if not host:
        raise RuntimeError('set --host, NATLANG_COLAB_HOSTS, or "colab_hosts" in ' + str(ROOT / 'config.json'))
    sha, commit = args.policy_id, args.commit
    beside_learner = bool(getattr(args, 'beside_learner', False))
    weight_dtype = getattr(args, 'weight_dtype', 'bfloat16')
    remote_ckpt = f'{cfg["checkpoints"]}/{sha}.pt'
    remote_work = f'{cfg["work_root"]}/actor-{sha[:12]}'
    layout = colab_layout(colab_hardware(host))
    serve_cmd, claim, sglang, sglang_format = colab_serve_command(
        cfg, checkpoint=remote_ckpt, sha=sha, port=args.port, work=remote_work,
        beside_learner=beside_learner, kv_cache_dtype=args.kv_cache_dtype, weight_dtype=weight_dtype, layout=layout)
    state_file = colab_state_path(host)
    previous = read_json(state_file)
    if (previous and previous.get('policy_id') == sha and previous.get('code') == commit
            and previous.get('sglang') == sglang and attested(previous['endpoint'], sha)):
        log(event='Colab G4 actor already serves the checkpoint', host=host, endpoint=previous['endpoint'])
        return previous
    if previous:
        stop_colab(host)
    if beside_learner:
        ensure_colab_mps(host)
    started = time.time()
    endpoint = remote_endpoint(host, args.port, args.bind_host)
    ssh_host(host, shlex.join(['mkdir', '-p', cfg['checkpoints'], remote_work]))
    arrived = ensure_remote_checkpoint(host, cfg, args.checkpoint, sha)
    copied = time.time()
    colab_checkout(host, cfg, commit)
    admitted = [cfg['python'], cfg['admission'], 'run', '--vram-gib', str(claim['vram_gib']),
                '--host-ram-gib', str(claim['host_ram_gib']), '--label', f'actor-{sha[:12]}', '--wait', '300',
                '--workdir', cfg['repo'], '--', *serve_cmd]
    # setsid: the admission process leads its own group (stop_colab signals it); its PID is the launch's output.
    # Separate statements: `. env && setsid ... &` backgrounded the whole list in a subshell that held the SSH channel
    # (the launch timed out) and `$!` named that subshell, whose group stop_colab could not signal
    pid = ssh_host(host, f'. {shlex.quote(cfg["env"])} || exit 1; setsid nohup {shlex.join(admitted)} '
                         f'> {shlex.quote(remote_work + "/actor.log")} 2>&1 < /dev/null & echo $!', timeout=60).strip()
    # the serve's own arguments too: run-sft stops an actor the learner does not fit beside and serves it again after
    result = {'endpoint': endpoint, 'host': host, 'profile': 'colab', 'policy_id': sha, 'work': remote_work,
              'beside_learner': beside_learner, 'gpu': layout['gpu_name'],
              'code': commit, 'engine': 'sglang', 'sglang': sglang, 'export_format': sglang_format,
              'pid': int(pid.splitlines()[-1]), 'claim': claim, 'copy_seconds': round(copied - started, 1),
              'checkpoint_arrived': arrived, 'started': started,
              'serve_args': {'checkpoint': str(args.checkpoint), 'policy_id': sha, 'commit': commit, 'port': args.port,
                             'bind_host': args.bind_host, 'kv_cache_dtype': args.kv_cache_dtype,
                             'weight_dtype': weight_dtype, 'beside_learner': beside_learner,
                             'ready_seconds': args.ready_seconds, 'host': host}}
    write_json(state_file, result)
    deadline = time.monotonic() + args.ready_seconds
    while time.monotonic() < deadline:
        if attested(endpoint, sha):
            result['ready_seconds'] = round(time.time() - started, 1)
            write_json(state_file, result)
            log(event='Colab G4 actor ready', **result)
            return result
        time.sleep(5)
    stop_colab(host)
    raise RuntimeError(f'Colab G4 actor on {host} did not attest {sha[:12]} at {endpoint} within {args.ready_seconds} s')


def colab_sft_fit(host):
    """(fit, reason): whether round SFT on ``host``'s session beats the DGX (``colab_layout`` ``sft_beats_dgx``: an
    L4 does not; an A100, H100 or G4 does). A session not bootstrapped is not fit. ``NATLANG_COLAB_SFT_ANY_GPU=1``
    opts out (any bootstrapped GPU is fit)."""
    try:
        layout = colab_layout(colab_hardware(host))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        return False, f'{host}: no attested hardware ({str(error)[:200]})'
    if os.environ.get('NATLANG_COLAB_SFT_ANY_GPU') == '1':
        return True, layout['sft_reason'] + ' (NATLANG_COLAB_SFT_ANY_GPU=1)'
    return layout['sft_beats_dgx'], layout['sft_reason']


def colab_learner_layout(host):
    """The SFT learner's claims on ``host`` (``colab_layout``): beside the capped actor when this DGX's record of
    the session's actor says it was served beside the learner, else alone."""
    layout = colab_layout(colab_hardware(host))
    actor = read_json(colab_state_path(host)) or {}
    beside = bool(actor.get('beside_learner')) and actor.get('pid') is not None
    return {**layout['learner_beside_actor' if beside else 'learner_alone'], 'beside_actor': beside,
            'gpu': layout['gpu_name']}


def remap_sft_args(argv, mapping):
    """``exp0_sft`` arguments (the DGX recipe, ``launch.json`` ``exp0_sft_args``) with the values of path flags
    replaced through ``mapping`` (local path -> remote path); every --chains/--checkpoint/--base/--out-dir/--val-chains
    value must be mapped (a DGX path on the remote would fail late or read the wrong file)."""
    out, path_flags = list(argv), ('--chains', '--checkpoint', '--base', '--out-dir', '--val-chains',
                                   '--export-factors-dir', '--state-dir')
    for i, arg in enumerate(out[:-1]):
        if arg in path_flags:
            if out[i + 1] not in mapping:
                raise ValueError(f'{arg} {out[i + 1]} has no remote copy')
            out[i + 1] = mapping[out[i + 1]]
    return out


def _learner_probe(remote_run):
    q = shlex.quote
    pid_file, rc_file = f'{remote_run}/learner.pid', f'{remote_run}/learner.rc'
    return (f'if [ -f {q(rc_file)} ]; then echo "rc $(cat {q(rc_file)})"; '
            f'elif [ -f {q(pid_file)} ] && kill -0 "$(cat {q(pid_file)})" 2>/dev/null; then echo running; '
            f'else echo none; fi')


def remote_learner_state(host, remote_run):
    """'running', 'rc N' (finished with status N) or 'none' for the detached learner of ``remote_run``."""
    return ssh_host(host, _learner_probe(remote_run), timeout=60).strip()


def run_detached(host, cfg, remote_run, command, *, poll_seconds=30, away_seconds=1800):
    """Run ``command`` on ``host`` detached from the SSH connection (``setsid nohup``: a dropped ``colab ssh`` link
    used to kill the learner through its stdout pipe) and wait for its exit status; a learner of this run still alive
    from an earlier call is waited for instead of a second one started, and one that already finished with status 0
    is not run again. Output goes to REMOTE_RUN/learner.log. Returns the exit status, or None when the host stayed
    unreachable ``away_seconds`` (the learner may still run)."""
    pid_file, rc_file, log_file = (f'{remote_run}/learner.pid', f'{remote_run}/learner.rc', f'{remote_run}/learner.log')
    q = shlex.quote
    state = remote_learner_state(host, remote_run)
    if state == 'rc 0':
        log(event='Colab learner already finished', host=host, remote_run=remote_run)
        return 0
    if state != 'running':
        inner = f'. {q(cfg["env"])} && {shlex.join(command)}; echo $? > {q(rc_file)}.tmp && mv {q(rc_file)}.tmp {q(rc_file)}'
        # separate statements: `a && setsid ... &` would background the whole list in a subshell that holds the
        # SSH channel open until the learner ends (and `$!` would name that subshell)
        ssh_host(host, f'rm -f {q(rc_file)}; setsid nohup bash -c {q(inner)} >> {q(log_file)} 2>&1 < /dev/null & '
                       f'echo $! > {q(pid_file)}', timeout=60)
        log(event='Colab learner started', host=host, remote_run=remote_run)
    else:
        log(event='Colab learner already running: waiting for it', host=host, remote_run=remote_run)
    last_seen = time.time()
    while True:
        time.sleep(poll_seconds)
        try:
            state = remote_learner_state(host, remote_run)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            if time.time() - last_seen >= away_seconds:
                log(event='Colab learner unreachable', host=host, seconds=round(time.time() - last_seen),
                    error=repr(error)[:200])
                return None
            continue
        last_seen = time.time()
        if state.startswith('rc '):
            return int(state.split()[1] or 1)
        if state == 'none':          # neither running nor finished: killed with its whole group (preemption)
            return 1


def pull_learner_state(host, remote_run, local_state, *, attempts=3):
    """Copy the learner state of ``remote_run`` to ``local_state`` as one save, never a mix of two: the learner
    rotates whole directories (``save_learner_state``: stage -> root), and an rsync of the live directory across a
    rotation pulled files of two saves ("file has vanished", 2026-11-22). A hardlink snapshot whose source directory
    kept its inode throughout holds one save; it arrives beside ``local_state`` and replaces it whole. Returns False
    when the learner has saved nothing yet."""
    q = shlex.quote
    root, snap = f'{remote_run}/learner-state', f'{remote_run}/learner-state.snap'
    out = ssh_host(host, f'for i in 1 2 3 4 5 6 7 8 9 10; do a=$(stat -c %i {q(root)} 2>/dev/null || echo none); '
                         f'rm -rf {q(snap)}; if [ "$a" != none ] && cp -al {q(root)} {q(snap)} 2>/dev/null && '
                         f'[ "$(stat -c %i {q(root)} 2>/dev/null)" = "$a" ]; then '
                         f'test -f {q(snap + "/learner_manifest.json")} && echo ok || echo empty; exit 0; fi; sleep 3; '
                         f'done; echo busy', timeout=120).strip()
    if out == 'empty':
        return False
    if out != 'ok':
        raise RuntimeError(f'{host}: no consistent snapshot of {root} ({out})')
    local_state = Path(local_state)
    staged, old = local_state.with_name(local_state.name + '.pull'), local_state.with_name(local_state.name + '.old')
    shutil.rmtree(staged, ignore_errors=True)
    sync_from_remote(f'{snap}/', f'{staged}/', host, attempts=attempts)
    if not (staged / 'learner_manifest.json').is_file():
        raise RuntimeError(f'the learner state pulled from {host} has no manifest')
    shutil.rmtree(old, ignore_errors=True)
    if local_state.exists():
        os.replace(local_state, old)
    os.replace(staged, local_state)
    shutil.rmtree(old, ignore_errors=True)
    return True


def local_learners(work_dir):
    """PIDs of live ``exp0_sft.py`` processes on this machine whose ``--out-dir`` is ``work_dir``: the DGX learner of a
    round, read from /proc, so it holds whatever systemd or the ledger answer (2026-11-23: a restarted driver saw no
    unit for a DGX learner that kept training and started run-sft beside it)."""
    target = os.path.realpath(str(work_dir))
    found = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            words = [w.decode(errors='replace') for w in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if w]
        except OSError:
            continue
        if not any(w.endswith('exp0_sft.py') for w in words) or '--out-dir' not in words[:-1]:
            continue
        if os.path.realpath(words[words.index('--out-dir') + 1]) == target:
            found.append(int(pid))
    return found


def refuse_beside_local_learner(work_dir, when):
    """SystemExit when a DGX learner trains ``work_dir``: never a second learner of one round (only the hand-back,
    after it stopped the DGX unit, moves a round to a session)."""
    pids = local_learners(work_dir)
    if pids:
        log(event='run-sft refused: a DGX learner trains this run', work=str(work_dir), pids=pids, when=when)
        raise SystemExit(f'run-sft {when}: a DGX learner (pid {pids}) trains {work_dir}; a round has one learner')


def remote_sft_run(cfg, run_name):
    """The VM directory of ``run-sft --run-name RUN_NAME`` (its learner state, logs, factors)."""
    return f'{cfg["work_root"]}/sft-{run_name}'


def state_step(manifest):
    """The step a learner state resumes at (``exp0_sft``: the batch cursor's ``next_step``, else ``learner_step``)."""
    cursor = manifest.get('batch_cursor') or {}
    return int(cursor.get('next_step', manifest.get('learner_step', 0)))


def remote_state_step(host, remote_run):
    """The resume step of ``remote_run``'s learner state on ``host``; None when it holds none."""
    out = ssh_host(host, f'cat {shlex.quote(remote_run + "/learner-state/learner_manifest.json")} 2>/dev/null || true',
                   timeout=60, check=False).strip()
    try:
        return state_step(json.loads(out)) if out else None
    except ValueError:
        return None


def snapshot_learner_state(local_state, *, attempts=20):
    """A hardlink snapshot of the live ``local_state`` holding one save (``save_learner_state`` rotates whole
    directories; a copy across a rotation mixes two saves): taken while the directory kept its inode, beside it as
    ``<name>.push``. The DGX learner's next rotation unlinks its names, never the snapshot's data."""
    local_state = Path(local_state)
    snap = local_state.with_name(local_state.name + '.push')
    for _ in range(attempts):
        shutil.rmtree(snap, ignore_errors=True)
        try:
            inode = local_state.stat().st_ino
            shutil.copytree(local_state, snap, copy_function=os.link)
            if local_state.stat().st_ino == inode and (snap / 'learner_manifest.json').is_file():
                return snap
        except (OSError, shutil.Error):
            pass
        time.sleep(3)
    shutil.rmtree(snap, ignore_errors=True)
    raise RuntimeError(f'no consistent snapshot of {local_state}')


def tree_sha256(root):
    """{relative path: sha256} of every file under ``root``."""
    from natlang_neuralese.common.hashing import sha256_file_hex
    root = Path(root)
    return {str(p.relative_to(root)): sha256_file_hex(p) for p in sorted(root.rglob('*')) if p.is_file()}


def remote_tree_sha256(host, remote_dir):
    """{relative path: sha256} of every file under ``remote_dir`` on ``host``."""
    out = ssh_host(host, f'cd {shlex.quote(remote_dir)} && find . -type f -exec sha256sum {{}} +', timeout=1800)
    pairs = (line.split(None, 1) for line in out.splitlines() if line.strip())
    return {name.strip().removeprefix('./'): sha for sha, name in pairs}


def sft_state_push_needed(host, remote_run, local_state):
    """Whether ``run-sft`` pushes the DGX's ``local_state`` before its learner starts: the DGX holds a state, no learner
    of the run runs on the VM, and the VM holds no state or an older one."""
    manifest = Path(local_state) / 'learner_manifest.json'
    if not manifest.exists() or remote_learner_state(host, remote_run) == 'running':
        return False
    remote_step = remote_state_step(host, remote_run)
    return remote_step is None or state_step(json.loads(manifest.read_text())) > remote_step


def push_learner_state(host, cfg, remote_run, local_state, *, run_name, via='auto'):
    """Place ``local_state`` (the DGX learner's latest save: FP32 factors, optimizer, RNG and batch cursor) as
    ``remote_run``'s learner state on ``host``, verified: a consistent snapshot goes to ``learner-state.incoming`` on
    the VM through Drive (``via`` auto/drive: the DGX uplink is about 2x the tunnel, the VM fetches inside Google's
    network; any Drive failure falls back) or the tunnel, every file's sha256 is checked against the snapshot, and only
    then does it replace the VM's state. Refused while a learner of the run runs there (its own state is newer).
    Returns {step, via, seconds, bytes, files}."""
    if remote_learner_state(host, remote_run) == 'running':
        raise RuntimeError(f'{host}: a learner of {remote_run} is running; its state is not replaced')
    started = time.time()
    snap = snapshot_learner_state(local_state)
    try:
        step = state_step(json.loads((snap / 'learner_manifest.json').read_text()))
        want = tree_sha256(snap)
        size = sum(p.stat().st_size for p in snap.rglob('*') if p.is_file())
        incoming = f'{remote_run}/learner-state.incoming'
        q = shlex.quote
        ssh_host(host, f'mkdir -p {q(remote_run)} && rm -rf {q(incoming)}')
        how = None
        if via in ('auto', 'drive') and drive_enabled(cfg):
            try:
                from scripts import drive_tier
                prefix = f'colab/{run_name}/handback-state'
                drive_tier.put_dir(argparse.Namespace(token_file=None, dir=str(snap), prefix=prefix))
                got = drive_vm(host, cfg, 'get-dir', prefix, incoming)
                if got.get('event') == 'get-dir':
                    how = 'drive'
            except Exception as error:      # any Drive failure falls back to the tunnel
                log(event='learner state via Drive failed: using the tunnel', host=host, error=repr(error)[:300])
                ssh_host(host, f'rm -rf {q(incoming)}', check=False)
        if how is None:
            sync_to_remote(f'{snap}/', f'{incoming}/', host)
            how = 'tunnel'
        got = remote_tree_sha256(host, incoming)
        if got != want:
            ssh_host(host, f'rm -rf {q(incoming)}', check=False)
            bad = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
            raise RuntimeError(f'the learner state on {host} does not match the DGX snapshot ({bad[:5]}); discarded')
        if remote_learner_state(host, remote_run) == 'running':
            raise RuntimeError(f'{host}: a learner of {remote_run} started during the push; its state is kept')
        state = f'{remote_run}/learner-state'
        ssh_host(host, f'rm -rf {q(state + ".old")} && {{ [ ! -e {q(state)} ] || mv -T {q(state)} {q(state + ".old")}; }} '
                       f'&& mv -T {q(incoming)} {q(state)} && rm -rf {q(state + ".old")}')
        record = {'step': step, 'via': how, 'seconds': round(time.time() - started, 1), 'bytes': size,
                  'files': len(want)}
        log(event='learner state pushed and verified', host=host, remote_run=remote_run, **record)
        return record
    finally:
        shutil.rmtree(snap, ignore_errors=True)


# transfer rates for planning a move of a round to a session (measured 2026-11-23, 35a91a22f): the DGX uplink to
# Drive 5.49 MB/s, the tunnel 2.63 up; rsync -z shrinks chains ~12x and a BF16 checkpoint ~20%, an FP32 state barely
SFT_MOVE_RATES_MB_S = {'drive': 5.0, 'tunnel': 2.4}
# checkout, admission, model load, compile (113 s on the DGX) and the Ambient teacher before the first step on a VM
SFT_REMOTE_SETUP_SECONDS = 900.0


def sft_move_seconds(host, cfg, *, run_name, launch, state_bytes, via='auto'):
    """Planned seconds to start ``launch``'s round on ``host`` from a pushed learner state of ``state_bytes``: what the
    VM still lacks (the chains, the start checkpoint unless it is there) plus the state, at the measured rates, plus
    the learner's setup. Returns (seconds, parts)."""
    recipe = shlex.split(launch['exp0_sft_args'])
    value = lambda flag: recipe[recipe.index(flag) + 1]
    rate = SFT_MOVE_RATES_MB_S['drive' if via in ('auto', 'drive') and drive_enabled(cfg) else 'tunnel'] * 1e6
    tunnel = SFT_MOVE_RATES_MB_S['tunnel'] * 1e6
    q = shlex.quote
    chains_remote = f'{remote_sft_run(cfg, run_name)}/chains.pkl'
    ckpt_remote = f'{cfg["checkpoints"]}/{launch["start_checkpoint_sha256"]}.pt'
    have = ssh_host(host, f'for f in {q(chains_remote)} {q(ckpt_remote)}; do test -f "$f" && echo yes || echo no; done',
                    timeout=60, check=False).split()
    have = (have + ['no', 'no'])[:2]
    parts = {'state': round(state_bytes / rate, 1), 'setup': SFT_REMOTE_SETUP_SECONDS,
             'chains': 0.0 if have[0] == 'yes' else round(Path(value('--chains')).stat().st_size / rate, 1),
             'checkpoint': 0.0}
    if have[1] != 'yes':
        published = read_json(Path(value('--checkpoint')).with_name('published.json')) or {}
        factors = Path(published['factors_dir']) / 'adapter_factors_fp32.safetensors' \
            if published.get('factors_dir') else None
        parts['checkpoint'] = round(factors.stat().st_size / tunnel + 120 if factors and factors.exists() else
                                    Path(value('--checkpoint')).stat().st_size / rate, 1)
    return round(sum(parts.values()), 1), parts


def restore_colab_actor(record):
    """Serve again the Colab actor ``run-sft`` stopped for a learner it did not fit beside (its recorded serve
    arguments); a failure is logged, never raised (the round's factors are already home)."""
    try:
        serve_colab(argparse.Namespace(**record['serve_args']))
    except Exception as error:
        log(event='Colab actor restore failed', host=record.get('host'), error=repr(error)[:300])


def run_remote_sft(args):
    """Run the DGX SFT recipe on a Colab G4 VM: exactly the ``exp0_sft`` arguments the DGX preflight
    (``ops/grpo-v4/launch-sft.sh`` with PREFLIGHT_ONLY=1: input identity, held exclusion; ``--launch-json``) recorded,
    with paths mapped to the VM's copies, plus ``--resume`` and the learner-state saves the DGX pulls every
    ``--state-sync-minutes``. The finished checkpoint (or a round's factors) arrives verified (``pull_verified``) or
    not at all. The VM checkout is verified at ``--commit``; the start checkpoint is merged on the VM from the round
    it trained before when it can (``ensure_remote_checkpoint``); the learner runs detached from the SSH link; an
    actor recorded on the session alone (not capped beside the learner) yields the GPU and is served again after."""
    import threading
    refuse_beside_local_learner(args.work, 'at start')
    cfg = host_config('colab', host=args.host)
    host = cfg['host']
    if not host:
        raise RuntimeError('run-sft requires --host or NATLANG_COLAB_HOSTS')
    fit, reason = colab_sft_fit(host)
    if not fit:     # before anything moves or the session's actor yields: its rounds train faster on the DGX
        log(event='Colab session unfit for SFT', host=host, reason=reason)
        raise SystemExit(f'{host}: round SFT here is slower than on the DGX: {reason}')
    launch = json.loads(Path(args.launch_json).read_text())
    recipe = shlex.split(launch['exp0_sft_args'])
    local_work = Path(args.work).resolve()
    local_work.mkdir(parents=True, exist_ok=True)
    remote_run = f'{cfg["work_root"]}/sft-{args.run_name}'
    timings = {}
    t0 = time.time()
    ssh_host(host, shlex.join(['mkdir', '-p', cfg['checkpoints'], remote_run, f'{remote_run}/learner-state']))
    if args.commit:
        colab_checkout(host, cfg, args.commit)
    value = lambda flag: recipe[recipe.index(flag) + 1]
    mapping = {value('--base'): cfg['base'], value('--out-dir'): remote_run}
    chains_remote = f'{remote_run}/chains.pkl'
    t = time.time()
    timings['chains_via'] = ship_file(value('--chains'), chains_remote, host, cfg, launch['chains_sha256'])
    if remote_identity(host, chains_remote)[1] != launch['chains_sha256']:
        raise RuntimeError(f'the chains copied to {host} do not hash to {launch["chains_sha256"]}')
    timings['chains_seconds'] = round(time.time() - t, 1)
    mapping[value('--chains')] = chains_remote
    start_sha = launch['start_checkpoint_sha256']
    t = time.time()
    timings['start_checkpoint'] = ensure_remote_checkpoint(host, cfg, value('--checkpoint'), start_sha)
    timings['start_checkpoint_seconds'] = round(time.time() - t, 1)
    mapping[value('--checkpoint')] = f'{cfg["checkpoints"]}/{start_sha}.pt'
    vals = [recipe[i + 1] for i, a in enumerate(recipe[:-1]) if a == '--val-chains']
    for idx, vpath in enumerate(vals):
        rv = f'{remote_run}/val-{idx}-{Path(vpath).name}'
        from natlang_neuralese.common.hashing import sha256_file_hex
        ship_file(vpath, rv, host, cfg, sha256_file_hex(vpath))
        mapping[vpath] = rv
    local_state = local_work / 'learner-state'
    # exact resume across a preemption or from a DGX learner the round moved from (the DGX holds the newer state):
    # pushed verified whenever it is ahead of the VM's (a VM whose earlier learner failed, 'rc 1', kept a stale state
    # the old rule resumed); never into a learner of this run still running (its own state is newer)
    if sft_state_push_needed(host, remote_run, local_state):
        t = time.time()
        pushed = push_learner_state(host, cfg, remote_run, local_state, run_name=args.run_name,
                                    via=getattr(args, 'state_transfer', 'auto'))
        timings.update(state_via=pushed['via'], state_step=pushed['step'], state_seconds=round(time.time() - t, 1))
    # an async round's recipe exports its factors (--export-factors-dir, --round-id: one argv for the G4 and the DGX
    # learner, expert_iteration.round_sft_env); the factors land beside the run on the VM and are pulled back
    factors_only = '--factors-only' in recipe
    round_id = value('--round-id') if '--round-id' in recipe else None
    if '--export-factors-dir' in recipe:
        mapping[value('--export-factors-dir')] = f'{remote_run}/factors'
    timings['prepare_seconds'] = round(time.time() - t0, 1)
    # no MPS start here: the daemon must be up before the first CUDA client, so the serve that puts an actor beside
    # the learner starts it (serve-checkpoint --beside-learner) and the learner joins it; a session with no actor
    # beside (or one started without MPS) runs the learner as a plain CUDA process
    # the learner's box on this session's hardware: alone, or beside the capped MPS actor its serve started; the
    # activation budget follows (it changes memory, not the step list, so a DGX takeover resumes the same steps)
    learner = colab_learner_layout(host)
    vram_gib = args.vram_gib if args.vram_gib is not None else learner['vram_gib']
    host_ram_gib = args.host_ram_gib if args.host_ram_gib is not None else learner['host_ram_gib']
    # an actor served alone holds the GPU the learner alone needs (the admission would wait out its timeout): it
    # yields for the round and is served again afterwards (between rounds the session decodes, during one it trains)
    # minutes of transfers since the start check: before the actor yields, the sync thread starts (its pulls replace
    # the local learner-state) or the VM learner starts, no DGX learner may train this run
    refuse_beside_local_learner(args.work, 'before the VM learner starts')
    yielded = None
    actor = read_json(colab_state_path(host)) or {}
    if not learner['beside_actor'] and actor.get('pid') is not None:
        yielded = stop_colab(host) if actor.get('serve_args') else None
        if yielded is None:
            log(event='Colab actor without serve arguments: not stopped', host=host, policy=actor.get('policy_id'))
    sft_args = remap_sft_args(recipe, mapping)
    if '--activation-gb' in sft_args:
        sft_args[sft_args.index('--activation-gb') + 1] = str(learner['activation_gib'])
    else:
        sft_args += ['--activation-gb', str(learner['activation_gib'])]
    log(event='Colab SFT layout', host=host, **learner, yielded_actor=bool(yielded), **timings)
    # the data registry lives on the DGX: the run's reads are recorded there (refused when unregistered, as exp0_sft
    # refuses on the DGX), and the VM learner reads its sha-verified copies without a registry of its own
    from natlang_neuralese.common import data_registry
    data_registry.consume([value('--chains'), *vals], run=data_registry.run_name('sft', local_work),
                          how=f'SFT training chains (Colab learner on {host}: sha-verified VM copies)')
    sft_cmd = ['/usr/bin/env', f'{data_registry.ALLOW_ENV}=1', cfg['python'],
               f'{cfg["repo"]}/scripts/neuralese_data/exp0_sft.py', *sft_args,
               *([] if '--resume' in recipe else ['--resume']), '--state-save-minutes', str(args.state_sync_minutes)]
    admitted = [cfg['python'], cfg['admission'], 'run', '--vram-gib', str(vram_gib), '--host-ram-gib',
                str(host_ram_gib), '--label', f'sft-{args.run_name}', '--wait', '600', '--workdir', cfg['repo'],
                '--', *sft_cmd]
    stop_sync = threading.Event()
    sync_interval = max(60.0, float(args.state_sync_minutes) * 60.0)
    pulling = threading.Lock()      # one pull at a time (a 2 GB state takes ~8 min at 4.8 MB/s)

    def pull_state(attempts):
        with pulling:
            pull_learner_state(host, remote_run, local_state, attempts=attempts)
        for name, local_name in (('train.jsonl', 'train.jsonl'), ('learner.log', 'train.log')):
            try:      # the logs follow the state; one not written yet (a learner that failed at start) is no error
                sync_from_remote(f'{remote_run}/{name}', str(local_work / local_name), host, attempts=1)
            except subprocess.CalledProcessError:
                pass

    def sync_loop():
        while not stop_sync.wait(sync_interval):
            try:
                t = time.time()
                pull_state(2)
                log(event='Colab SFT state synced', host=host, seconds=round(time.time() - t, 1))
            except Exception as err:
                log(event='Colab SFT state sync warning', host=host, error=repr(err)[:200])
            # the snapshot the pull took, onto Drive from the VM (inside Google's network): a VM lost between pulls,
            # or a tunnel that fails, still leaves its last save (drive_tier.py get-dir colab/<run>/learner-state)
            drive_backup_remote(host, cfg, f'{remote_run}/learner-state.snap', f'colab/{args.run_name}/learner-state')

    sync_thread = threading.Thread(target=sync_loop, daemon=True)
    sync_thread.start()
    started, rc = time.time(), 1
    try:
        rc = run_detached(host, cfg, remote_run, admitted)
        rc = 1 if rc is None else rc
    finally:
        stop_sync.set()
        sync_thread.join(timeout=30)
        # the latest learner state, so the DGX can resume a failed or preempted run; a finished round needs only its
        # factors (the state's one use is a resume), so the 2 GB state no longer delays them
        try:
            if rc != 0:
                pull_state(3)
            else:     # the logs only (no lock: an in-flight periodic state pull must not hold up the factors)
                for name, local_name in (('train.jsonl', 'train.jsonl'), ('learner.log', 'train.log')):
                    sync_from_remote(f'{remote_run}/{name}', str(local_work / local_name), host, attempts=2)
        except Exception as err:
            log(event='Colab SFT post-run pull warning', error=repr(err)[:200])
    trained = time.time()
    sha = None
    factors_sha = None
    if rc == 0:
        if factors_only:
            # the recipe's DGX factors directory (a DGX takeover writes the same place); blob first, manifest last,
            # staged beside it and renamed into place whole (a reader never sees a manifest without its blob). The VM
            # first puts them on Drive (seconds inside Google's network): a VM lost during the pull keeps them
            drive_backup_remote(host, cfg, f'{remote_run}/factors', f'colab/{args.run_name}/factors')
            local_factors = Path(value('--export-factors-dir'))
            staged = local_factors.with_name(local_factors.name + '.pull')
            staged.mkdir(parents=True, exist_ok=True)
            try:
                factors_sha = pull_verified(f'{remote_run}/factors/adapter_factors_fp32.safetensors',
                                            staged / 'adapter_factors_fp32.safetensors', host)
                pull_verified(f'{remote_run}/factors/policy_manifest.json', staged / 'policy_manifest.json', host)
                manifest = json.loads((staged / 'policy_manifest.json').read_text())
                if manifest.get('adapter_factors_sha256') != factors_sha:
                    raise RuntimeError('pulled adapter_factors_fp32.safetensors does not match policy_manifest.json')
                if local_factors.exists():
                    shutil.rmtree(local_factors)
                os.replace(staged, local_factors)
                # kept on the VM by sha: the next round's start is this round's publication, merged there from them
                keep = f'{cfg["root"]}/factors/{factors_sha}'
                ssh_host(host, f'mkdir -p {shlex.quote(cfg["root"] + "/factors")} && rm -rf {shlex.quote(keep)} && '
                               f'cp -al {shlex.quote(remote_run + "/factors")} {shlex.quote(keep)}', check=False)
            except Exception as err:
                rc = 1
                pending = {'schema': 'natlang.pending-upload/1', 'host': host, 'run_name': args.run_name,
                           'round_id': round_id, 'remote_factors_dir': f'{remote_run}/factors',
                           'error': repr(err)[:300], 'at': round(time.time(), 1)}
                write_json(local_work / 'pending-upload.json', pending)
                try:
                    ssh_host(host, shlex.join(['bash', '-c', f'cat > {shlex.quote(remote_run + "/pending-upload.json")}']),
                             input=json.dumps(pending), check=False, timeout=30)
                except Exception:
                    pass
                log(event='Colab SFT factors pull failed; recorded pending-upload.json', host=host, error=repr(err)[:300])
        else:
            try:
                sha = pull_verified(f'{remote_run}/checkpoint.pt', local_work / 'checkpoint.pt', host)
            except Exception as err:
                rc = 1
                log(event='Colab SFT checkpoint pull failed', host=host, error=repr(err)[:300])
    timings.update(train_seconds=round(trained - started, 1), pull_seconds=round(time.time() - trained, 1))
    if yielded:
        t = time.time()
        restore_colab_actor(yielded)
        timings['actor_restore_seconds'] = round(time.time() - t, 1)
    log(event='Colab G4 SFT finished', host=host, returncode=rc, seconds=round(time.time() - started, 1),
        checkpoint_sha256=sha, adapter_factors_sha256=factors_sha, **timings)
    return rc


def serve_checkpoint(args):
    """Serve a pruned native checkpoint on Pop or Colab G4 and wait until it attests."""
    if profile_key(getattr(args, 'host_profile', 'pop')) == 'colab':
        return serve_colab(args)
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
    state = read_json(colab_state_path(host) if profile_key(profile) == 'colab' and host else STATE) or {}
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
    if args is not None and profile_key(getattr(args, 'host_profile', 'pop')) == 'colab':
        targets = [args.host] if getattr(args, 'host', None) else colab_hosts()
        for h in targets:
            stop_colab(h)
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
        sub.add_argument('--host-profile', choices=('pop', 'colab', 'colab_g4'), default='pop')
        sub.add_argument('--host', help='explicit SSH host alias (default: Pop or first Colab G4 host)')
        sub.add_argument('--kv-cache-dtype', choices=('fp8_e4m3', 'auto'), default=POP_KV_CACHE_DTYPE)
        sub.add_argument('--weight-dtype', choices=('bfloat16', 'fp8'), default='bfloat16',
                         help='SGLang linear weight dtype (fp8 for SFT collection actors; bfloat16 for held and GRPO)')
        sub.add_argument('--max-running', type=int, default=POP_MAX_RUNNING, help='Pop running cap (a Colab actor '
                         'runs 128 alone, 64 with --beside-learner)')
        sub.add_argument('--beside-learner', action='store_true',
                         help='Colab: cap the actor (KV a quarter of the GPU, its running cap scaled; colab_layout) to share '
                         'the GPU with the SFT learner under MPS')
    c.add_argument('--checkpoint', required=True)
    c.add_argument('--policy-id', required=True)
    c.add_argument('--commit', required=True)
    c.add_argument('--step', type=int, default=0)
    c.add_argument('--port', type=int, default=RL_PORT)
    c.add_argument('--bind-host')
    c.add_argument('--ready-seconds', type=int, default=1800)
    c.add_argument('--adapter-release')
    rb = commands.add_parser('rebind', help='in-place rebind + 4-route attestation of a standing SGLang actor')
    rb.add_argument('--front-url', required=True)
    rb.add_argument('--sglang-url', help='SGLang server URL (default: http://127.0.0.1:<front_port + 1000>)')
    rb.add_argument('--policy-id', required=True)
    rb.add_argument('--version', type=int, required=True)
    rb.add_argument('--weights-sha256')
    rb.add_argument('--tensor-identity')
    rb.add_argument('--export-dir', help='HF export directory (local or remote)')
    rb.add_argument('--factors-dir', help='FP32 adapter factors dir to ship and merge locally on --host')
    rb.add_argument('--base-checkpoint', help='base checkpoint path on --host when using --factors-dir')
    rb.add_argument('--host', help='remote SSH host alias (omit for local DGX actor)')
    rb.add_argument('--host-profile', choices=('pop', 'colab', 'colab_g4'), default='colab')
    rb.add_argument('--timeout', type=int, default=600)
    rb.add_argument('--checkpoint', help="the DGX path of the rebound policy's merged checkpoint: a Colab actor's record "
                                         'then names it (the session keeper serves it again after a replacement)')
    f = commands.add_parser('prefetch')
    f.add_argument('--checkpoint', required=True)
    f.add_argument('--policy-id', required=True)
    f.add_argument('--step', type=int, default=0)
    h = commands.add_parser('held-eval')
    h.add_argument('--spec', required=True)
    rs = commands.add_parser('run-sft', help='run the DGX SFT recipe on a Colab G4 VM with periodic state sync back')
    rs.add_argument('--host', required=True)
    rs.add_argument('--run-name', required=True)
    rs.add_argument('--work', required=True, help='local DGX work directory (receives learner-state and checkpoint.pt or factors/)')
    rs.add_argument('--launch-json', required=True,
                    help="the DGX preflight's launch.json (launch-sft.sh PREFLIGHT_ONLY=1): its exp0_sft_args run")
    rs.add_argument('--commit')
    rs.add_argument('--state-sync-minutes', type=float, default=15.0)
    rs.add_argument('--vram-gib', type=float, help='admission claim (default: from the session hardware, the learner '
                    'alone or beside the capped actor the session runs)')
    rs.add_argument('--host-ram-gib', type=float, help='admission claim (default: from the session hardware)')
    rs.add_argument('--state-transfer', choices=('auto', 'drive', 'tunnel'), default='auto',
                    help='how a DGX learner state ahead of the VM\'s is pushed (auto: Drive, the tunnel on failure)')
    rc = commands.add_parser('reachable', help='list reachable remote hosts for a profile')
    rc.add_argument('--host-profile', choices=('pop', 'colab', 'colab_g4'), default='colab')
    rc.add_argument('--hosts', help='comma-separated SSH host aliases')
    st_cmd = commands.add_parser('stop')
    st_cmd.add_argument('--host-profile', choices=('pop', 'colab', 'colab_g4'), default='pop')
    st_cmd.add_argument('--host')
    status_cmd = commands.add_parser('status')
    status_cmd.add_argument('--host-profile', choices=('pop', 'colab', 'colab_g4'), default='pop')
    status_cmd.add_argument('--host')
    p = commands.add_parser('prepare-view', help=argparse.SUPPRESS)
    for flag in ('--checkpoint', '--out', '--receipt'):
        p.add_argument(flag, required=True)
    p.add_argument('--expect-sha256')
    args = parser.parse_args(argv)
    if args.action == 'reachable':
        hosts = colab_hosts(args.hosts) if profile_key(args.host_profile) == 'colab' else ([POP] if POP else [])
        up = reachable_hosts(args.host_profile, hosts)
        print(json.dumps({'profile': args.host_profile, 'configured': hosts, 'reachable': up}))
        return 0
    if args.action == 'status':
        if profile_key(args.host_profile) == 'colab':
            hosts = [args.host] if args.host else colab_hosts()
            print(json.dumps({'profile': 'colab', 'states': {h: read_json(colab_state_path(h)) for h in hosts}}))
        else:
            print(json.dumps({'state': read_json(STATE)}))
        return 0
    if args.action == 'prepare-view':
        prepare_view(args)
        return 0
    if args.action == 'serve-checkpoint':
        print(json.dumps(serve_checkpoint(args)), flush=True)
        return 0
    if args.action == 'rebind':
        print(json.dumps(rebind_actor(
            front_url=args.front_url, sglang_url=args.sglang_url, policy_id=args.policy_id, version=args.version,
            weights_sha256=args.weights_sha256, tensor_identity=args.tensor_identity,
            export_dir=args.export_dir, factors_dir=args.factors_dir, base_checkpoint=args.base_checkpoint,
            host=args.host, host_profile=args.host_profile, timeout=args.timeout, checkpoint=args.checkpoint)),
              flush=True)
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
