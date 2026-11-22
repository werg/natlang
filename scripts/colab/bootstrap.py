#!/usr/bin/env python3
"""Colab G4 VM bootstrap (lfm3opti.md §2): notebook cell or CLI entrypoint.

Installs Tailscale with userspace networking (``tailscaled --tun=userspace-networking``), joins the
tailnet, starts ``sshd`` with authorized keys, clones the natlang repo at a pinned commit, creates
``.venv-neuralese`` (PyTorch, sglang, torchao, natlang_neuralese), builds ``ts-host``, fetches the
base LFM3-1.7B model onto Colab local SSD scratch, probes GPU SKU / VRAM / host RAM / driver, and
writes ``colab-ready.json``.

Notebook usage:
    from scripts.colab.bootstrap import bootstrap
    info = bootstrap(tailscale_authkey='tskey-...', ssh_authorized_key='ssh-ed25519 ...', commit='main')

CLI usage:
    python scripts/colab/bootstrap.py --tailscale-authkey ... --ssh-authorized-key ... [--commit SHA] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_REPO_URL = 'https://github.com/werg/natlang.git'
DEFAULT_BASE_MODEL = 'LiquidAI/LFM3-1.7B'
DEFAULT_BASE_REVISION = '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
DEFAULT_COLAB_ROOT = Path(os.environ.get('NATLANG_COLAB_ROOT', '/content/natlang-work'))


def _log(event: str, **fields) -> None:
    print(json.dumps({'at': round(time.time(), 1), 'hook': 'colab_bootstrap', 'event': event, **fields}), flush=True)


def _run(cmd: list[str] | str, *, shell: bool = False, dry_run: bool = False, check: bool = True,
         cwd: str | Path | None = None, env: dict | None = None) -> str:
    rendered = cmd if isinstance(cmd, str) else shlex.join(map(str, cmd))
    _log('exec', cmd=rendered, dry_run=dry_run)
    if dry_run:
        return ''
    res = subprocess.run(cmd, shell=shell, cwd=cwd, env=env, text=True, capture_output=True)
    if check and res.returncode != 0:
        raise RuntimeError(f'bootstrap command failed ({res.returncode}): {rendered}\n{res.stderr[-2000:]}')
    return res.stdout


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
    sglang_pkg: str = 'sglang[all]',
) -> dict:
    """Return the structured step plan (inspectable and unit-testable without mutating the host)."""
    root = Path(root)
    repo_dir = root / 'natlang'
    venv_dir = repo_dir / '.venv-neuralese'
    base_dir = root / 'base-lfm3'
    ts_state = root / 'tailscale'
    return {
        'root': str(root),
        'repo_dir': str(repo_dir),
        'venv_dir': str(venv_dir),
        'base_dir': str(base_dir),
        'tailscale_state_dir': str(ts_state),
        'hostname': hostname,
        'repo_url': repo_url,
        'commit': commit,
        'base_model': base_model,
        'base_revision': base_revision,
        'ssh_port': int(ssh_port),
        'sglang_pkg': sglang_pkg,
        'steps': [
            'install_tailscale_userspace',
            'join_tailnet',
            'configure_sshd',
            'clone_and_checkout_repo',
            'setup_python_venv',
            'build_ts_host',
            'fetch_base_model',
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
    sglang_pkg: str = 'sglang[all]',
    dry_run: bool = False,
) -> dict:
    """Execute the complete Colab G4 bootstrap and return the ready manifest."""
    started = time.time()
    plan = build_bootstrap_plan(
        tailscale_authkey=tailscale_authkey,
        ssh_authorized_key=ssh_authorized_key,
        hostname=hostname,
        repo_url=repo_url,
        commit=commit,
        root=Path(root),
        base_model=base_model,
        base_revision=base_revision,
        ssh_port=ssh_port,
        sglang_pkg=sglang_pkg,
    )
    root_path = Path(plan['root'])
    repo_dir = Path(plan['repo_dir'])
    venv_dir = Path(plan['venv_dir'])
    base_dir = Path(plan['base_dir'])
    ts_state = Path(plan['tailscale_state_dir'])

    if not dry_run:
        root_path.mkdir(parents=True, exist_ok=True)
        ts_state.mkdir(parents=True, exist_ok=True)

    # 1. Install Tailscale and start tailscaled with userspace networking (Colab lacks /dev/net/tun)
    _run('command -v tailscaled >/dev/null 2>&1 || curl -fsSL https://tailscale.com/install.sh | sh',
         shell=True, dry_run=dry_run)
    ts_sock = ts_state / 'tailscaled.sock'
    ts_state_file = ts_state / 'tailscaled.state'
    _run(
        f'pgrep -f tailscaled >/dev/null 2>&1 || '
        f'(nohup tailscaled --tun=userspace-networking --statedir={shlex.quote(str(ts_state))} '
        f'--state={shlex.quote(str(ts_state_file))} --socket={shlex.quote(str(ts_sock))} '
        f'> {shlex.quote(str(root_path / "tailscaled.log"))} 2>&1 & sleep 3)',
        shell=True, dry_run=dry_run,
    )

    # 2. Join tailnet
    _run([
        'tailscale', f'--socket={ts_sock}', 'up',
        f'--authkey={tailscale_authkey}',
        f'--hostname={hostname}',
        '--ssh=false',
    ], dry_run=dry_run)

    # 3. Configure & start sshd with authorized keys
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

    # 4. Clone repo at pinned commit
    if not dry_run and (repo_dir / '.git').is_dir():
        _run(['git', '-C', str(repo_dir), 'fetch', '-q', 'origin'], dry_run=dry_run)
        _run(['git', '-C', str(repo_dir), 'checkout', '-f', commit], dry_run=dry_run)
    else:
        _run(['git', 'clone', repo_url, str(repo_dir)], dry_run=dry_run)
        _run(['git', '-C', str(repo_dir), 'checkout', '-f', commit], dry_run=dry_run)

    # 5. Create .venv-neuralese and install runtime dependencies (PyTorch, sglang, torchao, editable package)
    pip = str(venv_dir / 'bin' / 'pip')
    py = str(venv_dir / 'bin' / 'python')
    if dry_run or not (venv_dir / 'bin' / 'python').exists():
        _run([sys.executable, '-m', 'venv', '--system-site-packages', str(venv_dir)], dry_run=dry_run)
    _run([pip, 'install', '-q', '--upgrade', 'pip'], dry_run=dry_run)
    _run([pip, 'install', '-q', 'safetensors', 'zstandard', 'huggingface_hub', 'torchao', sglang_pkg], dry_run=dry_run)
    _run([pip, 'install', '-q', '--no-deps', '-e', str(repo_dir / 'training' / 'neuralese')], dry_run=dry_run)

    # 6. Build ts-host
    _run(['npm', 'ci', '--no-audit', '--no-fund'], cwd=str(repo_dir / 'ts-host'), dry_run=dry_run)
    _run(['npm', 'run', 'build:node'], cwd=str(repo_dir / 'ts-host'), dry_run=dry_run)

    # 7. Fetch base LFM3 model onto local SSD scratch
    fetch_code = (
        'from huggingface_hub import snapshot_download; '
        f'snapshot_download({base_model!r}, revision={base_revision!r}, local_dir={str(base_dir)!r})'
    )
    _run([py, '-c', fetch_code], dry_run=dry_run)

    # 8. Probe hardware and emit ready manifest
    hw = probe_host_hardware() if not dry_run else {'dry_run': True}
    ready = {
        'schema': 'natlang.colab-ready/1',
        'ready': True,
        'dry_run': dry_run,
        'elapsed_seconds': round(time.time() - started, 1),
        'plan': plan,
        'hardware': hw,
    }
    if not dry_run:
        (root_path / 'colab-ready.json').write_text(json.dumps(ready, indent=1, sort_keys=True) + '\n')
    _log('ready', hostname=hostname, tailscale_ip=hw.get('tailscale_ip'), gpu=hw.get('gpu_name'),
         vram_gib=hw.get('vram_total_gib'), host_ram_gib=hw.get('host_ram_total_gib'))
    return ready


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tailscale-authkey', default=os.environ.get('TAILSCALE_AUTHKEY', ''))
    parser.add_argument('--ssh-authorized-key', default=os.environ.get('SSH_AUTHORIZED_KEY', ''))
    parser.add_argument('--hostname', default='colab-g4')
    parser.add_argument('--repo-url', default=DEFAULT_REPO_URL)
    parser.add_argument('--commit', default='main')
    parser.add_argument('--root', type=Path, default=DEFAULT_COLAB_ROOT)
    parser.add_argument('--base-model', default=DEFAULT_BASE_MODEL)
    parser.add_argument('--base-revision', default=DEFAULT_BASE_REVISION)
    parser.add_argument('--ssh-port', type=int, default=22)
    parser.add_argument('--sglang-pkg', default='sglang[all]')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.dry_run and (not args.tailscale_authkey or not args.ssh_authorized_key):
        raise SystemExit('Provide --tailscale-authkey and --ssh-authorized-key (or set TAILSCALE_AUTHKEY / SSH_AUTHORIZED_KEY)')
    ready = bootstrap(
        tailscale_authkey=args.tailscale_authkey,
        ssh_authorized_key=args.ssh_authorized_key,
        hostname=args.hostname,
        repo_url=args.repo_url,
        commit=args.commit,
        root=args.root,
        base_model=args.base_model,
        base_revision=args.base_revision,
        ssh_port=args.ssh_port,
        sglang_pkg=args.sglang_pkg,
        dry_run=args.dry_run,
    )
    print(json.dumps(ready, indent=1, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
