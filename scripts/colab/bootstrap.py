#!/usr/bin/env python3
"""Colab G4 VM bootstrap (lfm3opti.md §2, async-train.md §1): notebook cell or CLI entrypoint.

Installs Tailscale with userspace networking (``tailscaled --tun=userspace-networking``), joins the tailnet, starts
``sshd`` with authorized keys, clones the natlang repo at a pinned commit, sets up the environment the DGX uses
(``.venv-neuralese`` with the shared requirements list ``training/neuralese/docker/requirements.txt`` over Colab's
torch, the repository on ``PYTHONPATH`` through ``natlang-env.sh``: the tree is not a package), installs natlang's SGLang
fork at the commit ``training/neuralese/sglang-fork.json`` pins into its own venv, Node >= 22.13 for ts-host and builds
it, fetches the base LFM3-1.7B (not gated; ``HF_TOKEN`` from the environment if set), starts the CUDA MPS daemon (the
SFT learner and a capped actor share the GPU), probes GPU SKU / VRAM / host RAM / driver and writes
``colab-ready.json``.

Secrets never reach the output: the Tailscale auth key goes to tailscale as ``--auth-key=file:`` from a 0600 file
removed after the join, and every logged command and failure message is redacted.

Notebook usage (keys from Colab secrets, never pasted into a cell's output):
    from google.colab import userdata
    from scripts.colab.bootstrap import bootstrap
    info = bootstrap(tailscale_authkey=userdata.get('TAILSCALE_AUTHKEY'), ssh_authorized_key='ssh-ed25519 ...',
                     commit='main')

CLI usage (TAILSCALE_AUTHKEY / SSH_AUTHORIZED_KEY in the environment):
    python scripts/colab/bootstrap.py [--commit SHA] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_REPO_URL = 'https://github.com/werg/natlang.git'
DEFAULT_BASE_MODEL = 'LiquidAI/LFM3-1.7B'
DEFAULT_BASE_REVISION = '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
DEFAULT_COLAB_ROOT = Path(os.environ.get('NATLANG_COLAB_ROOT', '/content/natlang-work'))
NODE_MIN = (22, 13)
NODE_VERSION = '24.11.1'
REDACTED = '<redacted>'

_SECRETS: list[str] = []


def redact(text: str) -> str:
    """``text`` with every registered secret and anything shaped like a Tailscale or HF key replaced."""
    for secret in _SECRETS:
        if secret:
            text = text.replace(secret, REDACTED)
    return re.sub(r'(tskey-[A-Za-z0-9_-]+|hf_[A-Za-z0-9]{20,})', REDACTED, text)


def _log(event: str, **fields) -> None:
    print(redact(json.dumps({'at': round(time.time(), 1), 'hook': 'colab_bootstrap', 'event': event, **fields})),
          flush=True)


def _run(cmd: list[str] | str, *, shell: bool = False, dry_run: bool = False, check: bool = True,
         cwd: str | Path | None = None, env: dict | None = None) -> str:
    rendered = redact(cmd if isinstance(cmd, str) else shlex.join(map(str, cmd)))
    _log('exec', cmd=rendered, dry_run=dry_run)
    if dry_run:
        return ''
    res = subprocess.run(cmd, shell=shell, cwd=cwd, env=env, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(f'bootstrap command failed ({res.returncode}): {rendered}\n{redact(res.stderr[-2000:])}')
    return res.stdout


def tailscaled_start_command(ts_state: Path, root: Path, daemon: str = 'tailscaled') -> str:
    """Start tailscaled unless it runs. ``pgrep -x`` matches the process name exactly: ``pgrep -f tailscaled`` matched
    this very ``sh -c`` command line, so the daemon never started."""
    sock, state = ts_state / 'tailscaled.sock', ts_state / 'tailscaled.state'
    q = lambda p: shlex.quote(str(p))
    return (f'pgrep -x {daemon} >/dev/null 2>&1 || (nohup {daemon} --tun=userspace-networking --statedir={q(ts_state)} '
            f'--state={q(state)} --socket={q(sock)} > {q(root / "tailscaled.log")} 2>&1 & '
            f'for i in $(seq 30); do test -S {q(sock)} && exit 0; sleep 1; done; exit 1)')


def node_ok(version: str) -> bool:
    match = re.match(r'v?(\d+)\.(\d+)', version.strip())
    return bool(match) and (int(match.group(1)), int(match.group(2))) >= NODE_MIN


def env_script(repo_dir: Path, venv_dir: Path, node_bin: Path, root: Path) -> str:
    """natlang-env.sh: what a DGX ledger unit sets (NATLANG_REPO, PYTHONPATH over the repository, HF offline after the
    fetch), plus Node and the MPS pipe directory. Sourced by every remote command (pop_rl_actor colab paths)."""
    return '\n'.join([
        f'export NATLANG_REPO={shlex.quote(str(repo_dir))}',
        f'export PYTHONPATH={shlex.quote(f"{repo_dir}/training/neuralese:{repo_dir}:{repo_dir}/scripts/neuralese_data")}',
        f'export PATH={shlex.quote(str(node_bin))}:{shlex.quote(str(venv_dir / "bin"))}:$PATH',
        'export HF_HUB_OFFLINE=1',
        f'export HF_HOME={shlex.quote(str(root / "hf"))}',
        f'export CUDA_MPS_PIPE_DIRECTORY={shlex.quote(str(root / "mps/pipe"))}',
        f'export CUDA_MPS_LOG_DIRECTORY={shlex.quote(str(root / "mps/log"))}',
        'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1', ''])


def build_bootstrap_plan(
    *,
    tailscale_authkey: str,
    ssh_authorized_key: str,
    hostname: str = 'colab-g4',
    repo_url: str = DEFAULT_REPO_URL,
    commit: str = 'main',
    root: Path = DEFAULT_COLAB_ROOT,
    base_model: str = DEFAULT_BASE_MODEL,
    base_revision: str = DEFAULT_BASE_REVISION,
    ssh_port: int = 22,
) -> dict:
    """Return the structured step plan (inspectable and unit-testable without mutating the host; holds no secret)."""
    root = Path(root)
    repo_dir = root / 'natlang'
    return {
        'root': str(root),
        'repo_dir': str(repo_dir),
        'venv_dir': str(repo_dir / '.venv-neuralese'),
        'sglang_dir': str(root / 'sglang-natlang'),
        'sglang_venv': str(root / 'sglang-natlang-venv'),
        'node_dir': str(root / 'node'),
        'env_file': str(root / 'natlang-env.sh'),
        'base_dir': str(root / 'base-lfm3'),
        'tailscale_state_dir': str(root / 'tailscale'),
        'hostname': hostname,
        'repo_url': repo_url,
        'commit': commit,
        'base_model': base_model,
        'base_revision': base_revision,
        'ssh_port': int(ssh_port),
        'steps': [
            'install_tailscale_userspace',
            'join_tailnet',
            'configure_sshd',
            'clone_and_checkout_repo',
            'setup_python_venv',
            'install_sglang_fork',
            'install_node',
            'build_ts_host',
            'fetch_base_model',
            'start_mps',
            'probe_hardware_and_write_ready',
        ],
    }


def probe_host_hardware() -> dict:
    """Probe GPU SKU, VRAM GiB, host RAM GiB, driver, and Tailscale IPv4."""
    info: dict = {
        'hostname': os.uname().nodename,
        'tailscale_ip': None,
        'gpu_name': None,
        'vram_total_gib': None,
        'vram_free_gib': None,
        'driver_version': None,
        'host_ram_total_gib': None,
        'host_ram_avail_gib': None,
    }
    try:
        ip = subprocess.check_output(['tailscale', 'ip', '-4'], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        info['tailscale_ip'] = ip.splitlines()[0].strip() if ip else None
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.DEVNULL, timeout=15,
        ).strip()
        if out:
            name, tot, free, drv = [x.strip() for x in out.splitlines()[0].split(',')]
            info.update(
                gpu_name=name,
                vram_total_gib=round(float(tot) / 1024.0, 2),
                vram_free_gib=round(float(free) / 1024.0, 2),
                driver_version=drv,
            )
    except Exception:
        pass
    try:
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)
        info['host_ram_total_gib'] = round(int(fields['MemTotal'].split()[0]) / 2**20, 2)
        info['host_ram_avail_gib'] = round(int(fields['MemAvailable'].split()[0]) / 2**20, 2)
    except Exception:
        pass
    return info


def bootstrap(
    *,
    tailscale_authkey: str,
    ssh_authorized_key: str,
    hostname: str = 'colab-g4',
    repo_url: str = DEFAULT_REPO_URL,
    commit: str = 'main',
    root: str | Path = DEFAULT_COLAB_ROOT,
    base_model: str = DEFAULT_BASE_MODEL,
    base_revision: str = DEFAULT_BASE_REVISION,
    ssh_port: int = 22,
    dry_run: bool = False,
) -> dict:
    """Execute the complete Colab G4 bootstrap and return the ready manifest."""
    _SECRETS[:] = [tailscale_authkey, os.environ.get('HF_TOKEN', '')]
    started = time.time()
    plan = build_bootstrap_plan(tailscale_authkey=tailscale_authkey, ssh_authorized_key=ssh_authorized_key,
                                hostname=hostname, repo_url=repo_url, commit=commit, root=Path(root),
                                base_model=base_model, base_revision=base_revision, ssh_port=ssh_port)
    root_path, repo_dir = Path(plan['root']), Path(plan['repo_dir'])
    venv_dir, base_dir, ts_state = Path(plan['venv_dir']), Path(plan['base_dir']), Path(plan['tailscale_state_dir'])
    sglang_dir, sglang_venv, node_dir = Path(plan['sglang_dir']), Path(plan['sglang_venv']), Path(plan['node_dir'])
    if not dry_run:
        root_path.mkdir(parents=True, exist_ok=True)
        ts_state.mkdir(parents=True, exist_ok=True)

    # 1. Tailscale with userspace networking (Colab has no /dev/net/tun)
    _run('command -v tailscaled >/dev/null 2>&1 || curl -fsSL https://tailscale.com/install.sh | sh',
         shell=True, dry_run=dry_run)
    _run(tailscaled_start_command(ts_state, root_path), shell=True, dry_run=dry_run)

    # 2. Join the tailnet: the key goes as a file (never on a command line, in a log or an error message)
    key_file = ts_state / 'authkey'
    if not dry_run:
        key_file.touch(mode=0o600)
        key_file.write_text(tailscale_authkey.strip() + '\n')
    try:
        _run(['tailscale', f'--socket={ts_state / "tailscaled.sock"}', 'up', f'--auth-key=file:{key_file}',
              f'--hostname={hostname}', '--ssh=false'], dry_run=dry_run)
    finally:
        if not dry_run:
            key_file.unlink(missing_ok=True)

    # 3. sshd with the authorized key
    _run('command -v sshd >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq openssh-server rsync)',
         shell=True, dry_run=dry_run)
    if not dry_run:
        ssh_dir = Path.home() / '.ssh'
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
        auth_file = ssh_dir / 'authorized_keys'
        existing = auth_file.read_text() if auth_file.exists() else ''
        if ssh_authorized_key.strip() not in existing:
            auth_file.write_text((existing.rstrip('\n') + '\n' + ssh_authorized_key.strip() + '\n').lstrip('\n'))
        auth_file.chmod(0o600)
        Path('/run/sshd').mkdir(parents=True, exist_ok=True)
    _run(
        f'grep -q "^Port {ssh_port}$" /etc/ssh/sshd_config || echo "Port {ssh_port}" >> /etc/ssh/sshd_config; '
        'sed -i "s/^#*PermitRootLogin.*/PermitRootLogin prohibit-password/" /etc/ssh/sshd_config; '
        'sed -i "s/^#*PasswordAuthentication.*/PasswordAuthentication no/" /etc/ssh/sshd_config; '
        'pgrep -x sshd >/dev/null 2>&1 || /usr/sbin/sshd',
        shell=True, dry_run=dry_run,
    )

    # 4. The repository at the pinned commit
    if not dry_run and (repo_dir / '.git').is_dir():
        _run(['git', '-C', str(repo_dir), 'fetch', '-q', 'origin'], dry_run=dry_run)
    else:
        _run(['git', 'clone', '-q', repo_url, str(repo_dir)], dry_run=dry_run)
    _run(['git', '-C', str(repo_dir), 'checkout', '-q', '-f', commit], dry_run=dry_run)

    # 5. natlang's venv as on the DGX: the shared requirements list over Colab's CUDA torch; the repository is on
    #    PYTHONPATH (natlang-env.sh), not installed (training/neuralese is not a package)
    pip = str(venv_dir / 'bin' / 'pip')
    if dry_run or not (venv_dir / 'bin' / 'python').exists():
        _run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv_dir)], dry_run=dry_run)
    _run([pip, 'install', '-q', '--upgrade', 'pip', 'uv'], dry_run=dry_run)
    _run([pip, 'install', '-q', '-r', str(repo_dir / 'training/neuralese/docker/requirements.txt'),
          'huggingface_hub', 'numpy'], dry_run=dry_run)

    # 6. The SGLang fork (Liquid LFM3 support + natlang hooks) at its pinned commit, in its own venv, as on the DGX
    # a dry run reads the pin of the checkout it runs from (nothing is cloned)
    fork = json.loads(((Path(__file__).resolve().parents[2] if dry_run else repo_dir)
                       / 'training/neuralese/sglang-fork.json').read_text())
    if not dry_run and (sglang_dir / '.git').is_dir():
        _run(['git', '-C', str(sglang_dir), 'fetch', '-q', 'origin'], dry_run=dry_run)
    else:
        _run(['git', 'clone', '-q', fork['repository'], str(sglang_dir)], dry_run=dry_run)
    _run(['git', '-C', str(sglang_dir), 'checkout', '-q', '-f', fork['commit']], dry_run=dry_run)
    uv = str(venv_dir / 'bin' / 'uv')
    if dry_run or not (sglang_venv / 'bin' / 'python').exists():
        _run([uv, 'venv', '-q', '-p', '3.12', str(sglang_venv)], dry_run=dry_run)
    pins = [f'{name.replace("_", "-")}=={version}' for name, version in (fork.get('dependency_pins') or {}).items()
            if not name.startswith('changed') and re.match(r'^[\d.]+$', str(version))]
    _run([uv, 'pip', 'install', '-q', '-p', str(sglang_venv / 'bin' / 'python'), '-e', str(sglang_dir / 'python'), *pins,
          '--extra-index-url', 'https://download.pytorch.org/whl/cu130', '--index-strategy', 'unsafe-best-match'],
         dry_run=dry_run)

    # 7. Node >= 22.13 (ts-host); Colab ships an older one
    node_bin = node_dir / 'bin'
    have = _run('node --version 2>/dev/null || true', shell=True, dry_run=dry_run, check=False)
    if dry_run or not node_ok(have):
        arch = {'x86_64': 'x64', 'aarch64': 'arm64'}.get(os.uname().machine, 'x64')
        tarball = f'node-v{NODE_VERSION}-linux-{arch}'
        _run(f'mkdir -p {shlex.quote(str(node_dir))} && curl -fsSL https://nodejs.org/dist/v{NODE_VERSION}/{tarball}.tar.xz '
             f'| tar -xJ -C {shlex.quote(str(node_dir))} --strip-components=1', shell=True, dry_run=dry_run)
    else:
        node_bin = Path(_run('dirname "$(command -v node)"', shell=True).strip())
    env_file = Path(plan['env_file'])
    if not dry_run:
        env_file.write_text(env_script(repo_dir, venv_dir, node_bin, root_path))
    node_env = {**os.environ, 'PATH': f'{node_bin}:{os.environ.get("PATH", "")}'}
    _run(['npm', 'ci', '--no-audit', '--no-fund'], cwd=str(repo_dir / 'ts-host'), dry_run=dry_run, env=node_env)
    _run(['npm', 'run', 'build:node'], cwd=str(repo_dir / 'ts-host'), dry_run=dry_run, env=node_env)

    # 8. The base model (LiquidAI/LFM3-1.7B is not gated; HF_TOKEN from the environment is used when set)
    fetch_code = ('import os; from huggingface_hub import snapshot_download; '
                  f'snapshot_download({base_model!r}, revision={base_revision!r}, local_dir={str(base_dir)!r}, '
                  "token=os.environ.get('HF_TOKEN') or None)")
    _run([str(venv_dir / 'bin' / 'python'), '-c', fetch_code], dry_run=dry_run,
         env={k: v for k, v in os.environ.items() if k != 'HF_HUB_OFFLINE'})

    # 9. CUDA MPS: the SFT learner and a capped actor share the GPU (async-train §1)
    _run(f'mkdir -p {shlex.quote(str(root_path / "mps/pipe"))} {shlex.quote(str(root_path / "mps/log"))} && '
         f'export CUDA_MPS_PIPE_DIRECTORY={shlex.quote(str(root_path / "mps/pipe"))} '
         f'CUDA_MPS_LOG_DIRECTORY={shlex.quote(str(root_path / "mps/log"))} && '
         '(pgrep -x nvidia-cuda-mps-control >/dev/null 2>&1 || nvidia-cuda-mps-control -d)', shell=True, dry_run=dry_run)

    # 10. Probe hardware and write the ready manifest
    hw = probe_host_hardware() if not dry_run else {'dry_run': True}
    ready = {
        'schema': 'natlang.colab-ready/1',
        'ready': True,
        'dry_run': dry_run,
        'elapsed_seconds': round(time.time() - started, 1),
        'plan': plan,
        'sglang_fork_commit': fork['commit'],
        'hardware': hw,
    }
    if not dry_run:
        (root_path / 'colab-ready.json').write_text(json.dumps(ready, indent=1, sort_keys=True) + '\n')
    _log('ready', hostname=hostname, tailscale_ip=hw.get('tailscale_ip'), gpu=hw.get('gpu_name'),
         vram_gib=hw.get('vram_total_gib'), host_ram_gib=hw.get('host_ram_total_gib'))
    return ready


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--hostname', default='colab-g4')
    parser.add_argument('--repo-url', default=DEFAULT_REPO_URL)
    parser.add_argument('--commit', default='main')
    parser.add_argument('--root', type=Path, default=DEFAULT_COLAB_ROOT)
    parser.add_argument('--base-model', default=DEFAULT_BASE_MODEL)
    parser.add_argument('--base-revision', default=DEFAULT_BASE_REVISION)
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    # secrets come from the environment only: a command-line flag would show in ps and in shell history
    authkey, ssh_key = os.environ.get('TAILSCALE_AUTHKEY', ''), os.environ.get('SSH_AUTHORIZED_KEY', '')
    if not args.dry_run and (not authkey or not ssh_key):
        raise SystemExit('set TAILSCALE_AUTHKEY and SSH_AUTHORIZED_KEY in the environment')
    try:
        ready = bootstrap(tailscale_authkey=authkey, ssh_authorized_key=ssh_key, hostname=args.hostname,
                          repo_url=args.repo_url, commit=args.commit, root=args.root, base_model=args.base_model,
                          base_revision=args.base_revision, ssh_port=args.ssh_port, dry_run=args.dry_run)
    except RuntimeError as error:
        raise SystemExit(redact(str(error)))
    print(json.dumps(ready, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
