#!/usr/bin/env python3
"""Keep one Colab GPU session serving the DGX's Colab actor, and replace or upgrade it unattended.

One unit replaces the G4 poller (``ops/ei/colab/g4-poll.sh``) and the tunnel healer (``g4a-tunnel-heal.sh``). Every
``--check-seconds`` it reads ``colab sessions`` and the tunnel and acts on one decision (``decide``, pure and tested):

- **lost**: the session is missing, runs on a CPU (or hardware it does not rank), or its SSH link has failed for
  ``--tunnel-fail-minutes`` while the healer could not bring it back. The keeper stops what is left of it and asks for
  a new session under the same name in preference order (``--order``, default A100 high-RAM, then G4, then L4), every
  ``--request-minutes`` until one is assigned.
- **upgrade**: while the session is a lower tier than the first, every ``--upgrade-minutes`` it asks for each better
  tier under a second name; when one is assigned it stops the old session at once (Colab offers no capacity probe:
  the assignment GET answers for every entitled GPU, only the POST that assigns says whether there is capacity), so
  two GPU sessions overlap only for the seconds between the assignment and that stop, and renames the new one.
  Never while a round's ``run-sft`` learner runs on the session.
- **heal**: the tunnel unit's journal shows HTTP 429 ("another colab ssh is connected") and SSH fails: end the VM's
  stale sshd sessions through ``colab exec`` (the kernel channel; ``end_stale_sshd.py``). A forward missing on a live
  master is re-added over it (``ssh -O forward``).

A new session is bootstrapped (``bootstrap.py`` at the actor's commit, the SGLang fork shipped as ``git archive``),
its stale host key removed, the tunnel unit restarted, and the actor served again from the last actor record's own
arguments (``pop_rl_actor serve-checkpoint``, sized by ``colab_layout``). The fills reach it through the tunnel's
fixed DGX ports, so they resume on their own. Nothing here prints a token: ``colab`` output is redacted before it
is logged.

    python3 scripts/colab/session_keeper.py run [--session g4a --alias colab-g4 --tunnel-unit colab-g4a-tunnel]
    python3 scripts/colab/session_keeper.py status
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COLAB = os.environ.get('NATLANG_COLAB_CLI', str(Path.home() / '.local/bin/colab'))
COLAB_STORE = Path.home() / '.config/colab-cli/sessions.json'
OPS = Path('/home/werg/data/lfm3-rl-dgx-20261029/ops/ei/colab')
SGLANG_CHECKOUT = Path('/home/werg/sglang-natlang')
VM_ROOT = '/content/natlang-work'
# preference order: (label as `colab sessions` shows it, --gpu value, --high-mem)
DEFAULT_ORDER = (('A100', 'A100', True), ('G4', 'G4', False), ('L4', 'L4', False))
SECRET = re.compile(r'(?i)(token|authorization|cookie|key)(["\'=:\s]+)[^\s"\',]+|[A-Za-z0-9_\-]{40,}')


def redact(text: str) -> str:
    """Colab output with anything token-like masked (the CLI already redacts; this is the second line)."""
    return SECRET.sub(lambda m: (m.group(1) + m.group(2) + '<redacted>') if m.group(1) else '<redacted>', text or '')


def log(event: str, **fields) -> None:
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), 'event': event,
                      **{k: redact(v) if isinstance(v, str) else v for k, v in fields.items()}}), flush=True)


# -- decisions (pure) -----------------------------------------------------------------------------------------------
LINE = re.compile(r'^\[(?P<name>[^\]]*)\]\s+(?P<endpoint>\S+)\s*\|\s*Hardware:\s*(?P<hardware>[^|]+?)\s*\|'
                  r'\s*Shape:\s*(?P<shape>[^|]+?)\s*\|\s*Variant:\s*(?P<variant>[^|\s]+)')


def parse_sessions(text: str) -> list[dict]:
    """``colab sessions`` lines -> [{name, endpoint, hardware, shape, variant}] (``name`` '?' for an assignment the
    local store does not know)."""
    return [m.groupdict() for m in map(LINE.match, (text or '').splitlines()) if m]


def tier(hardware: str | None, order=DEFAULT_ORDER) -> int | None:
    """Rank of a session's hardware in ``order`` (0 best); None for a CPU or any hardware not in the order."""
    label = (hardware or '').strip().upper()
    return next((i for i, (name, _, _) in enumerate(order) if name.upper() == label), None)


@dataclasses.dataclass
class Observation:
    sessions: list[dict]            # parse_sessions of `colab sessions`
    ssh_ok: bool                    # a batch `ssh ALIAS true` over the tunnel's master
    tunnel_429s: int = 0            # HTTP 429 lines in the tunnel unit's journal, last two minutes
    sft_active: bool = False        # a run-sft learner targets the session (no upgrade under a round)
    now: float = 0.0


@dataclasses.dataclass
class Decision:
    action: str                     # ok | heal | replace | upgrade | wait
    reason: str = ''
    tiers: tuple = ()               # replace / upgrade: the order entries to request, best first


def decide(obs: Observation, state: dict, *, session: str, order=DEFAULT_ORDER, tunnel_fail_minutes=15.0,
           upgrade_minutes=20.0, request_minutes=5.0) -> Decision:
    """What the keeper does now. ``state`` (persisted, updated by the caller): ``ssh_fail_since``,
    ``last_upgrade_try``, ``last_request_try``."""
    mine = [s for s in obs.sessions if s['name'] == session]
    others = [s for s in obs.sessions if s['name'] != session and tier(s['hardware'], order) is not None]
    current = mine[0] if mine else None
    rank = tier(current['hardware'], order) if current else None
    lost = (f'session {session} is missing' if current is None else
            f'session {session} runs on {current["hardware"]}' if rank is None else None)
    if lost is None and not obs.ssh_ok and state.get('ssh_fail_since') is not None \
            and obs.now - state['ssh_fail_since'] >= 60 * tunnel_fail_minutes:
        lost = f'session {session} unreachable over SSH for {tunnel_fail_minutes:g} min'
    if lost:
        if others:   # never two GPU sessions: another one (not ours by name) must go first, by hand
            return Decision('wait', f'{lost}, but GPU session(s) {[s["name"] for s in others]} exist')
        if obs.now - state.get('last_request_try', 0) < 60 * request_minutes:
            return Decision('wait', f'{lost}; next request in <= {request_minutes:g} min')
        return Decision('replace', lost, tuple(order))
    if not obs.ssh_ok:
        return Decision('heal' if obs.tunnel_429s >= 3 else 'wait',
                        f'SSH to {session} fails' + (f' ({obs.tunnel_429s} HTTP 429s)' if obs.tunnel_429s else ''))
    if rank and not obs.sft_active and not others \
            and obs.now - state.get('last_upgrade_try', 0) >= 60 * upgrade_minutes:
        return Decision('upgrade', f'{current["hardware"]} is tier {rank}', tuple(order[:rank]))
    return Decision('ok', f'{current["hardware"]} serves')


def note_ssh(state: dict, ok: bool, now: float) -> dict:
    """``ssh_fail_since``: when the current run of SSH failures began (None while SSH works)."""
    return {**state, 'ssh_fail_since': None if ok else (state.get('ssh_fail_since') or now)}


# -- the outside world ------------------------------------------------------------------------------------------------
def run(cmd, *, timeout=600, env=None, check=False) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, env=env)
    except subprocess.TimeoutExpired:
        return 124, f'timeout after {timeout} s'
    if check and p.returncode:
        raise RuntimeError(f'{shlex.join(map(str, cmd))[:200]} -> {p.returncode}: {redact(p.stdout + p.stderr)[-600:]}')
    return p.returncode, p.stdout + p.stderr


class Keeper:
    def __init__(self, a, runner=run):
        self.a, self.run = a, runner
        self.order = parse_order(a.order)
        self.state_path = Path(a.state)

    # state
    def load(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {}

    def save(self, state: dict) -> None:
        tmp = self.state_path.with_suffix('.partial')
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + '\n')
        os.replace(tmp, self.state_path)

    # colab
    def colab(self, *args, timeout=600, env=None):
        return self.run([COLAB, '--auth=adc', *args], timeout=timeout, env=env)

    def sessions(self) -> list[dict] | None:
        rc, out = self.colab('sessions', timeout=120)
        return parse_sessions(out) if rc == 0 else None

    def usage_rate(self) -> float | None:
        rc, out = self.colab('usage', timeout=120)
        m = re.search(r'Usage rate:\s*([0-9.]+)', out or '')
        return float(m.group(1)) if rc == 0 and m else None

    def request(self, name: str, gpu: str, high_mem: bool, *, retries: int) -> str | None:
        """Ask for a session; the hardware it got (as `colab sessions` lists it) or None."""
        env = {**os.environ, 'COLAB_CLI_ASSIGN_RETRIES': str(retries)}
        rc, out = self.colab('new', '-s', name, '--gpu', gpu, *(['--high-mem'] if high_mem else []), env=env)
        got = next((s for s in (self.sessions() or []) if s['name'] == name), None)
        log('request', name=name, gpu=gpu, high_mem=high_mem, rc=rc, got=got and got['hardware'],
            answer=(out or '').strip().splitlines()[-1][-160:] if (out or '').strip() else '')
        if got and got['hardware'].strip().upper() != gpu.upper():
            log('request gave other hardware: stopping it', name=name, hardware=got['hardware'])
            self.colab('stop', '-s', name, timeout=300)
            return None
        return got['hardware'] if got else None

    def stop_session(self, name: str) -> None:
        rc, out = self.colab('stop', '-s', name, timeout=300)
        log('stop session', name=name, rc=rc, answer=(out or '').strip()[-160:])

    def unassign_orphans(self, sessions: list[dict]) -> None:
        """Assignments with no local name on a CPU (what is left when a session is recreated under its name): billed,
        unreachable by name. GPU orphans are only reported (never two GPU sessions: `decide` waits on them)."""
        cpu = [s['endpoint'] for s in sessions if s['name'] == '?' and tier(s['hardware'], self.order) is None]
        if cpu:
            code = ('import sys\nfrom colab_cli.common import state\nfrom colab_cli.auth import AuthProvider\n'
                    'state.auth_provider = AuthProvider.ADC\nfor e in sys.argv[1:]:\n    state.client.unassign(e)\n')
            rc, out = self.run([colab_python(), '-c', code, *cpu], timeout=300)
            log('unassigned orphan CPU VMs', endpoints=cpu, rc=rc)

    def rename(self, old: str, new: str) -> None:
        """Rename a session in the CLI's own store, under its lock (its token never leaves the store)."""
        code = ('import sys\nfrom colab_cli.state import StateStore\ns = StateStore()\nx = s.get(sys.argv[1])\n'
                'x.name = sys.argv[2]\ns.add(x)\ns.remove(sys.argv[1])\n')
        self.run([colab_python(), '-c', code, old, new], timeout=60, check=True)

    # ssh / tunnel
    def ssh_ok(self) -> bool:
        return self.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', self.a.alias, 'true'],
                        timeout=40)[0] == 0

    def tunnel_429s(self) -> int:
        _, out = self.run(['journalctl', '--user', '-u', self.a.tunnel_unit, '--since', '-2min', '--no-pager'],
                          timeout=30)
        return (out or '').count('HTTP 429')

    def heal(self) -> None:
        rc, out = self.colab('exec', '-s', self.a.session, '-f', str(Path(__file__).with_name('end_stale_sshd.py')),
                             timeout=120)
        log('heal: ended stale VM sshd sessions', rc=rc, answer=(out or '')[-300:])

    def reforward(self) -> None:
        """Re-add a forward a reconnected master came back without (seen 2026-11-22 22:03)."""
        if self.run(['systemctl', '--user', 'is-active', '-q', self.a.tunnel_unit], timeout=30)[0]:
            return
        _, listening = self.run(['ss', '-tln'], timeout=30)
        for spec in self.a.forward:
            local, remote = spec.split(':')
            if f'127.0.0.1:{local} ' not in listening:
                rc, _ = self.run(['ssh', '-O', 'forward', '-L', f'{local}:127.0.0.1:{remote}', self.a.alias], timeout=60)
                log('re-forward', local=local, remote=remote, rc=rc)

    def restart_tunnel(self) -> None:
        self.run(['ssh-keygen', '-f', str(Path.home() / '.ssh/known_hosts.colab'), '-R', self.a.alias], timeout=30)
        self.run(['systemctl', '--user', 'restart', self.a.tunnel_unit], timeout=60)

    def sft_active(self) -> bool:
        """A DGX ``pop_rl_actor.py run-sft`` process aimed at this alias (a round's learner on the session)."""
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            words = [w.decode(errors='replace') for w in argv]
            if any(w.endswith('pop_rl_actor.py') for w in words) and 'run-sft' in words \
                    and self.a.alias in words:
                return True
        return False

    # bootstrap and serve
    def actor_record(self) -> dict:
        path = Path(self.a.actor_root) / f'colab-actor-{self.a.alias}.json'
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    def bootstrap_and_serve(self) -> bool:
        """Bootstrap the new VM at the actor's commit and serve the actor again from its record's arguments."""
        record = self.actor_record()
        serve = record.get('serve_args') or self.a.serve_default and json.loads(Path(self.a.serve_default).read_text())
        if not serve:
            log('no actor record to serve again: bootstrap only')
        commit = (serve or {}).get('commit') or run(['git', '-C', str(REPO), 'rev-parse', 'origin/main'],
                                                    check=True)[1].strip()
        fork = json.loads(run(['git', '-C', str(REPO), 'show', f'{commit}:training/neuralese/sglang-fork.json'],
                              check=True)[1])['commit']
        scratch = Path(self.a.scratch)
        scratch.mkdir(parents=True, exist_ok=True)
        archive = scratch / f'sglang-{fork[:12]}.tgz'
        if not archive.exists():
            self.run(['/bin/sh', '-c', f'git -C {shlex.quote(str(SGLANG_CHECKOUT))} archive --format=tar {fork} | '
                                       f'gzip -1 > {shlex.quote(str(archive))}.partial && '
                                       f'mv {shlex.quote(str(archive))}.partial {shlex.quote(str(archive))}'],
                     timeout=600, check=True)
        boot = scratch / f'bootstrap-{commit[:12]}.py'
        boot.write_text(run(['git', '-C', str(REPO), 'show', f'{commit}:scripts/colab/bootstrap.py'], check=True)[1])
        for _ in range(30):      # the restarted tunnel's master
            if self.ssh_ok():
                break
            time.sleep(20)
        alias = self.a.alias
        self.run(['scp', '-q', str(boot), f'{alias}:/content/bootstrap.py'], timeout=300, check=True)
        self.run(['scp', '-q', str(archive), f'{alias}:/content/sglang.tgz'], timeout=1800, check=True)
        start = (f'D={VM_ROOT}/sglang-natlang; mkdir -p $D && tar -xzf /content/sglang.tgz -C $D && echo {fork} > '
                 f'$D/.natlang-archive && cd /content && (setsid nohup python3 /content/bootstrap.py --commit {commit} '
                 f'> {VM_ROOT}/bootstrap.log 2>&1 < /dev/null &)')
        self.run(['ssh', alias, start], timeout=300, check=True)
        log('bootstrap started', commit=commit, sglang_fork=fork)
        deadline = time.time() + 60 * self.a.bootstrap_minutes
        while time.time() < deadline:
            rc, out = self.run(['ssh', alias, f'cat {VM_ROOT}/colab-ready.json'], timeout=60)
            if rc == 0 and '"ready": true' in out:
                break
            time.sleep(30)
        else:
            _, tail = self.run(['ssh', alias, f'tail -5 {VM_ROOT}/bootstrap.log'], timeout=60)
            log('bootstrap did not finish', tail=tail[-600:])
            return False
        log('bootstrap ready')
        if not serve:
            return True
        old = Path(self.a.actor_root) / f'colab-actor-{alias}.json'
        if old.exists():   # its pid named a process on the lost VM: never signal that number on the new one
            os.replace(old, old.with_name(old.stem + f'.lost-{int(time.time())}.json'))
        argv = [str(REPO / '.venv-neuralese/bin/python'), str(REPO / 'scripts/neuralese_data/pop_rl_actor.py'),
                'serve-checkpoint', '--host-profile', 'colab', '--host', alias, '--checkpoint', serve['checkpoint'],
                '--policy-id', serve['policy_id'], '--commit', commit, '--port', str(serve.get('port', 8570)),
                '--kv-cache-dtype', serve.get('kv_cache_dtype', 'fp8_e4m3'),
                '--weight-dtype', serve.get('weight_dtype', 'bfloat16'),
                '--ready-seconds', str(serve.get('ready_seconds', 1800)),
                *(['--beside-learner'] if serve.get('beside_learner') else [])]
        env = {**os.environ, 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}:{REPO}/scripts/neuralese_data'}
        rc, out = self.run(argv, timeout=3 * 3600, env=env)
        log('actor served' if rc == 0 else 'actor serve failed', rc=rc, tail=(out or '')[-600:])
        return rc == 0

    def after_new_session(self, hardware: str, before_rate) -> None:
        after = self.usage_rate()
        log('session assigned', hardware=hardware, usage_rate_before=before_rate, usage_rate_after=after)
        self.restart_tunnel()
        self.bootstrap_and_serve()

    # one cycle
    def step(self) -> Decision:
        state = self.load()
        sessions = self.sessions()
        if sessions is None:
            log('colab sessions failed: no decision this cycle')
            return Decision('wait', 'colab sessions failed')
        self.unassign_orphans(sessions)
        sessions = [s for s in sessions if not (s['name'] == '?' and tier(s['hardware'], self.order) is None)]
        now = time.time()
        ok = self.ssh_ok()
        state = note_ssh(state, ok, now)
        obs = Observation(sessions=sessions, ssh_ok=ok, tunnel_429s=0 if ok else self.tunnel_429s(),
                          sft_active=self.sft_active(), now=now)
        d = decide(obs, state, session=self.a.session, order=self.order,
                   tunnel_fail_minutes=self.a.tunnel_fail_minutes, upgrade_minutes=self.a.upgrade_minutes,
                   request_minutes=self.a.request_minutes)
        if d.action != state.get('last_action') or d.action not in ('ok', 'wait'):
            log('decision', action=d.action, reason=d.reason, tiers=[t[0] for t in d.tiers])
        state['last_action'] = d.action
        if d.action == 'heal':
            self.heal()
        elif d.action == 'ok':
            self.reforward()
        elif d.action == 'replace':
            state['last_request_try'] = now
            self.save(state)
            if any(s['name'] == self.a.session for s in sessions):
                self.stop_session(self.a.session)
            before = self.usage_rate()
            for label, gpu, high_mem in d.tiers:
                got = self.request(self.a.session, gpu, high_mem, retries=3)
                if got:
                    state.update(ssh_fail_since=None, last_upgrade_try=time.time(), hardware=got)
                    self.save(state)
                    self.after_new_session(got, before)
                    break
        elif d.action == 'upgrade':
            state['last_upgrade_try'] = now
            self.save(state)
            nxt = self.a.session + '-next'
            before = self.usage_rate()
            for label, gpu, high_mem in d.tiers:
                got = self.request(nxt, gpu, high_mem, retries=0)
                if got:
                    # the new session is assigned: the old one goes at once (two GPU sessions only for these seconds)
                    self.stop_session(self.a.session)
                    self.rename(nxt, self.a.session)
                    state.update(ssh_fail_since=None, hardware=got)
                    self.save(state)
                    self.after_new_session(got, before)
                    break
        self.save(state)
        return d


def colab_python() -> str:
    """The colab CLI's own interpreter (its package: store and client)."""
    first = Path(COLAB).read_text().splitlines()[0]
    return first[2:].strip() if first.startswith('#!') else sys.executable


def parse_order(text: str):
    """'A100:hm,G4,L4' -> (('A100', 'A100', True), ('G4', 'G4', False), ('L4', 'L4', False))."""
    out = []
    for part in text.split(','):
        gpu, _, flag = part.strip().partition(':')
        out.append((gpu.upper(), gpu.upper(), flag == 'hm'))
    return tuple(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('run', 'status', 'once'))
    p.add_argument('--session', default='g4a', help='colab session name (the SSH alias ProxyCommand names it)')
    p.add_argument('--alias', default='colab-g4', help='SSH alias of the session')
    p.add_argument('--tunnel-unit', default='colab-g4a-tunnel')
    p.add_argument('--forward', action='append', default=None, help='LOCAL:REMOTE forwards of the tunnel unit')
    p.add_argument('--order', default='A100:hm,G4,L4', help='preference order, best first (:hm = --high-mem)')
    p.add_argument('--check-seconds', type=float, default=60)
    p.add_argument('--upgrade-minutes', type=float, default=20)
    p.add_argument('--request-minutes', type=float, default=5)
    p.add_argument('--tunnel-fail-minutes', type=float, default=15)
    p.add_argument('--bootstrap-minutes', type=float, default=45)
    p.add_argument('--state', default=str(OPS / 'session-keeper.json'))
    p.add_argument('--actor-root', default='/home/werg/data/lfm3-rl-dgx-20261029/ops/pop-behavior-eval',
                   help="pop_rl_actor's ROOT (its colab-actor-<alias>.json records the actor to serve again)")
    p.add_argument('--serve-default', default=None, help='serve_args JSON when no actor record exists')
    p.add_argument('--scratch', default=str(OPS / 'keeper-scratch'))
    a = p.parse_args(argv)
    a.forward = a.forward or ['19731:9570', '18731:8570']
    k = Keeper(a)
    if a.command == 'status':
        print(json.dumps({'state': k.load(), 'sessions': k.sessions(), 'ssh_ok': k.ssh_ok()}, indent=1))
        return 0
    while True:
        try:
            k.step()
        except Exception as error:      # a failed cycle is logged and retried, never the end of the keeper
            log('cycle failed', error=repr(error)[:600])
        if a.command == 'once':
            return 0
        time.sleep(a.check_seconds)


if __name__ == '__main__':
    sys.exit(main())
