#!/usr/bin/env python3
"""Restart a live interleaved RL controller at any time: stop it cooperatively, optionally move it to a runtime frozen
from current main, relaunch it under a fresh ledger unit and check that it resumed.

    python3 scripts/neuralese_data/restart_controller.py CONTROL_OUT [--runtime-from-main [--commit SHA] | --use-runtime vNN-MMDD]
        [--grace SECONDS | --patient] [--no-preflight] [--dry-run | --freeze-only]

CONTROL_OUT is the controller's ``out`` directory (it holds config.json, state.json, controller.log).  Every phase
resumes from its durable state (plans/neuralese/CONTROLLER_RESTART.md lists the cost per phase):

1. optional freeze of a successor runtime from main (``freeze_native_runtime.py`` through the ledger; the live controller
   keeps running meanwhile);
2. cooperative stop: SIGTERM to the controller process only.  It stops its collector workers and actor, archives retained
   partial artifacts and returns; a learner or SFT trainer finishes its current update and checkpoints.  After
   ``--grace`` seconds (default 600) the unit is stopped hard (``systemctl stop``: SIGTERM to the whole cgroup, SIGKILL
   after the unit's timeout); the trainer's rolling checkpoint then resumes at its last committed update (at most one
   update is redone).  ``--patient`` waits for the clean boundary instead;
3. with ``--runtime-from-main``: ``rebind-runtime.sh`` (config paths, actor python inventory, learner admission wrapper,
   env.sh), then the controller ``preflight`` under the new runtime; a refusal rolls the rebind back and relaunches on the
   old runtime;
4. relaunch through ``memory_ledger.py run`` as ``<base>-rN`` with the recorded launch spec (``launch.json`` next to the
   config, derived from the old unit the first time), and wait for the resumed controller to rewrite its state.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training/neuralese'))
from natlang_neuralese.common.paths import resolve  # noqa: E402  (stdlib-only)

LEDGER = REPO / 'scripts/memory_ledger.py'
DATA = resolve('data_nvme', 'lfm3-rl-dgx-20261029')
HASH_CACHE = resolve('data_hdd', 'natlang-lfm3-cloud-corrected-20261103', 'hash-cache.jsonl')
OPS = DATA / 'ops/bf95-lowlr-1105'
REBIND = OPS / 'interleave-v49/rebind-runtime.sh'
BINDINGS = OPS / 'interleave-v49/v50-bindings.json'
PYTHON = REPO / '.venv-neuralese/bin/python'
NODE_BIN = Path.home() / '.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin'
GIB = 2 ** 30
RUNTIME_RE = re.compile(r'runtime-native-corrected-mix-(v\d+-\d{4})')


def run(argv, **kwargs):
    return subprocess.run([str(a) for a in argv], text=True, capture_output=True, **kwargs)


def systemctl(*args):
    return run(['systemctl', '--user', *args])


def unit_state(unit):
    """'active', 'activating', 'deactivating', 'failed' or 'gone' (a stopped transient unit disappears)."""
    out = systemctl('show', unit, '-p', 'ActiveState', '-p', 'LoadState').stdout
    fields = dict(line.split('=', 1) for line in out.splitlines() if '=' in line)
    if fields.get('LoadState') == 'not-found':
        return 'gone'
    return fields.get('ActiveState', 'gone')


def units_named(prefix):
    out = systemctl('list-units', '--all', '--no-legend', '--plain', f'{prefix}*').stdout
    return [line.split()[0].removesuffix('.service') for line in out.splitlines() if line.strip()]


def base_unit(unit):
    return re.sub(r'-r\d+$', '', unit)


def pick_unit(out, explicit=None):
    """The controller unit of ``out``: the active ``bf95-lowlr-<name>[-rN]`` (else the newest that exists)."""
    if explicit:
        return explicit
    base = f'bf95-lowlr-{Path(out).name}'
    names = [u for u in units_named(base) if re.fullmatch(re.escape(base) + r'(-r\d+)?', u)]
    active = [u for u in names if unit_state(u) in ('active', 'activating', 'deactivating')]
    pool = active or names
    if not pool:
        recorded = read_json(Path(out) / 'launch.json')
        return recorded['unit'] if recorded else base
    return max(pool, key=lambda u: int((re.search(r'-r(\d+)$', u) or [0, 0])[1]))


def next_unit(unit, known):
    base = base_unit(unit)
    taken = [int(m[1]) for u in [*units_named(base), *known] if (m := re.fullmatch(re.escape(base) + r'-r(\d+)', u))]
    return f'{base}-r{max(taken, default=0) + 1}'


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def save_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)


def derive_launch(unit):
    """The launch spec of a running unit: its ``sh -c`` payload, working directory, memory budget and ledger class."""
    text = systemctl('cat', unit).stdout
    match = re.search(r'^ExecStart=:"/bin/sh" "-c" "(.*)"$', text, re.M)
    if match is None:
        raise SystemExit(f'cannot read the launch command of {unit} (systemctl cat); pass --launch-from FILE')
    props = dict(line.split('=', 1) for line in systemctl('show', unit, '-p', 'MemoryMax', '-p', 'WorkingDirectory').stdout.splitlines())
    claim = {}
    status = run([sys.executable, LEDGER, 'status'])
    if status.returncode == 0:
        claim = (json.loads(status.stdout).get('claims') or {}).get(unit + '.service', {})
    budget = round(int(props['MemoryMax']) / GIB, 1) if props.get('MemoryMax', '').isdigit() else claim.get('budget_gb', 100)
    return {'unit': unit, 'payload': match[1], 'workdir': props.get('WorkingDirectory', str(REPO)), 'budget_gb': budget,
            'priority': claim.get('priority', 'high'), 'fixed_claim': bool(claim.get('fixed_claim', True))}


def ledger_command(spec, unit):
    # The ledger workdir is the runtime the payload runs (AGENTS.md: frozen runtime launches use it as --workdir); a
    # runtime switch rewrites only the payload, so the old unit's WorkingDirectory would otherwise be carried over.
    found = RUNTIME_RE.findall(spec['payload'])
    workdir = str(DATA / f'runtime-native-corrected-mix-{found[0]}') if found else spec['workdir']
    argv = [sys.executable, LEDGER, 'run', '--unit', unit, '--budget-gb', str(spec['budget_gb']),
            '--priority', spec['priority'], '--workdir', workdir, '--wait', '600']
    if spec.get('fixed_claim'):
        argv.append('--fixed-claim')
    return [*argv, '--', '/bin/sh', '-c', spec['payload']]


def current_runtime(spec):
    found = RUNTIME_RE.findall(spec['payload'])
    if not found:
        raise SystemExit('the launch payload names no runtime-native-corrected-mix-vNN-MMDD runtime')
    return found[0]


def next_runtime_tag(current):
    numbers = [int(m[1]) for p in DATA.glob('runtime-native-corrected-mix-v*-*')
               if (m := re.fullmatch(r'runtime-native-corrected-mix-v(\d+)-\d{4}', p.name))]
    return f"v{max(numbers) + 1}-{time.strftime('%m%d')}"


def wait_unit(unit, seconds, poll=5.):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if unit_state(unit) in ('gone', 'failed', 'inactive'):
            return True
        time.sleep(poll)
    return unit_state(unit) in ('gone', 'failed', 'inactive')


def freeze(spec, commit, log):
    """Freeze main (or ``commit``) into the next runtime through the ledger; returns its tag."""
    tag = next_runtime_tag(current_runtime(spec))
    sha = run(['git', '-C', REPO, 'rev-parse', '--verify', (commit or 'HEAD') + '^{commit}'], check=True).stdout.strip()
    new = DATA / f'runtime-native-corrected-mix-{tag}'
    predecessor = DATA / f'runtime-native-corrected-mix-{current_runtime(spec)}'
    unit = f'freeze-controller-{tag}'
    env = f'PATH={NODE_BIN}:{os.environ["PATH"]}'
    payload = (f'cd {REPO} && exec env PYTHONPATH={REPO}/training/neuralese:{REPO} {PYTHON} '
               f'scripts/neuralese_data/freeze_native_runtime.py --commit {sha} --predecessor {predecessor} '
               f'--out {new} --control-bindings {BINDINGS} >> {log} 2>&1')
    started = run([sys.executable, LEDGER, 'run', '--unit', unit, '--budget-gb', '3', '--priority', 'low',
                   '--wait', '900', '--workdir', REPO, '--env', env, '--', '/bin/sh', '-c', payload])
    if started.returncode:
        raise SystemExit('freeze not admitted: ' + (started.stderr or started.stdout)[-400:])
    print(f'freezing {sha[:9]} into {new.name} (unit {unit}, log {log})', flush=True)
    time.sleep(5)
    while unit_state(unit) not in ('gone', 'failed', 'inactive'):
        time.sleep(10)
    if not (new / 'runtime-input-closure.json').is_file():
        raise SystemExit(f'freeze failed: no {new.name}/runtime-input-closure.json; see {log}')
    return tag


def stop_cooperatively(unit, out, grace, patient):
    """SIGTERM the controller process only; escalate to a unit stop after ``grace`` (never with ``patient``)."""
    state_path = Path(out) / 'state.json'
    phase = (read_json(state_path) or {}).get('phase')
    print(f'stopping {unit} (state phase {phase!r}): SIGTERM to the controller, collectors/actor/trainer follow its lead', flush=True)
    active = (read_json(state_path) or {}).get('active_update_round')
    log = Path(active) / 'learner/train.jsonl' if active and phase == 'updating' else None
    if log is not None and log.is_file():
        print(f'  learner: last committed update {(time.time() - log.stat().st_mtime) / 60:.0f} min ago (an update takes about '
              '35-45 min; the clean stop waits for the boundary, an escalated stop redoes the in-flight update)', flush=True)
    systemctl('kill', '--kill-whom=main', '-s', 'SIGTERM', unit)
    began = time.monotonic()
    while True:
        if wait_unit(unit, 5, poll=1.):
            print(f'  stopped cleanly after {time.monotonic() - began:.0f}s', flush=True)
            return 'clean'
        waited = time.monotonic() - began
        if int(waited) % 60 < 5:
            print(f'  {waited:.0f}s: still stopping, phase {(read_json(state_path) or {}).get("phase")!r}', flush=True)
        if not patient and waited >= grace:
            print(f'  grace {grace:.0f}s over: stopping the whole unit (an in-flight update is redone from its rolling '
                  'checkpoint)', flush=True)
            systemctl('stop', unit)
            wait_unit(unit, 600)
            return 'forced'


def held(lock_path):
    with open(lock_path, 'a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False


def preflight(unit, out, runtime):
    """The controller's own dry run under ``runtime`` (collection binding, learner guidance), scratch outputs only. It loads
    the policy checkpoint, so it runs as a ledger unit (the stopped controller's claim is free by now)."""
    scratch, log = Path(out) / 'restart-preflight', Path(out) / 'restart-preflight.log'
    payload = (f'cd {runtime} && exec env NATLANG_REPO={runtime} PYTHONPATH={runtime}/training/neuralese:{runtime} '
               f'HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 NATLANG_HASH_CACHE={os.environ.get("NATLANG_HASH_CACHE", HASH_CACHE)} '
               f'{PYTHON} -m natlang_neuralese.rl.production preflight --config {Path(out) / "config.json"} '
               f'--scratch {scratch} > {log} 2>&1')
    name = f'{unit}-preflight'
    started = run([sys.executable, LEDGER, 'run', '--unit', name, '--budget-gb', '40', '--priority', 'high', '--wait', '600',
                   '--workdir', runtime, '--', '/bin/sh', '-c', payload])
    if started.returncode:
        return False, 'preflight not admitted: ' + (started.stderr or started.stdout)[-300:]
    time.sleep(3)
    wait_unit(name, 1800)
    lines = [line for line in (log.read_text().splitlines() if log.is_file() else []) if line.startswith('{')]
    try:
        report = json.loads(lines[-1])
    except (IndexError, ValueError):
        return False, 'preflight printed no report: ' + (log.read_text()[-500:] if log.is_file() else 'no log')
    return report.get('ok') is True, json.dumps({k: report.get(k) for k in ('status', 'collection', 'learner', 'reason')})[:800]


def wait_resumed(unit, out, launched_at, seconds):
    """Resumed = unit active, controller lock held, state rewritten after launch, no traceback since launch."""
    state_path, log = Path(out) / 'state.json', Path(out) / 'controller.log'
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if unit_state(unit) not in ('active', 'activating'):
            return False, f'{unit} is {unit_state(unit)}; tail of controller.log: ' + log.read_text()[-600:]
        rewritten = state_path.is_file() and state_path.stat().st_mtime > launched_at
        if rewritten and held(Path(out) / 'controller.lock'):
            tail = log.read_text()[-4000:] if log.is_file() else ''
            if 'Traceback' not in tail[tail.rfind('\n', 0, max(0, len(tail) - 3000)):]:
                return True, (read_json(state_path) or {}).get('phase')
        time.sleep(3)
    return False, 'state.json was not rewritten by the new controller in time'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path, help='controller out directory (config.json, state.json, controller.log)')
    p.add_argument('--unit', help='current controller unit (default: bf95-lowlr-<name>[-rN], the active one)')
    p.add_argument('--runtime-from-main', action='store_true', help='freeze main into a successor runtime and rebind to it')
    p.add_argument('--commit', help='commit to freeze (default HEAD of the canonical checkout)')
    p.add_argument('--use-runtime', metavar='TAG', help='rebind to an already frozen runtime (e.g. v58-1107) instead of freezing')
    p.add_argument('--grace', type=float, default=600., help='seconds of cooperative stop before the unit is stopped hard')
    p.add_argument('--patient', action='store_true', help='never escalate: wait for the clean update boundary')
    p.add_argument('--no-preflight', action='store_true')
    p.add_argument('--resume-seconds', type=float, default=300.)
    p.add_argument('--config-patch', type=Path, help='JSON object merged into config.json after the stop (top-level keys; null deletes)')
    p.add_argument('--dry-run', action='store_true', help='print the plan and the launch command; change nothing')
    p.add_argument('--freeze-only', action='store_true', help='only freeze the successor runtime (nothing is stopped)')
    a = p.parse_args(argv)
    out = a.out.resolve()
    unit = pick_unit(out, a.unit)
    spec = read_json(out / 'launch.json') or derive_launch(unit)
    spec['unit'] = unit
    old_tag = current_runtime(spec)
    new_unit = next_unit(unit, spec.get('units', []))
    plan = {'out': str(out), 'unit': unit, 'next_unit': new_unit, 'runtime': old_tag, 'budget_gb': spec['budget_gb'],
            'priority': spec['priority'], 'fixed_claim': spec['fixed_claim']}
    if a.dry_run:
        print(json.dumps({**plan, 'launch': ledger_command(spec, new_unit)[2:]}, indent=1))
        return 0
    with open(out / 'restart.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit('another restart of this controller is running')
        new_tag = None
        if a.use_runtime:
            if not (DATA / f'runtime-native-corrected-mix-{a.use_runtime}/runtime-input-closure.json').is_file():
                raise SystemExit(f'runtime {a.use_runtime} is not a frozen runtime')
            new_tag = a.use_runtime
        elif a.runtime_from_main or a.freeze_only:
            new_tag = freeze(spec, a.commit, out / f'restart-freeze-{time.strftime("%Y%m%d-%H%M%S")}.log')
            print(f'frozen runtime {new_tag}', flush=True)
            if a.freeze_only:
                return 0
        outcome = stop_cooperatively(unit, out, a.grace, a.patient) if unit_state(unit) != 'gone' else 'already stopped'
        if held(out / 'controller.lock'):
            raise SystemExit('controller.lock is still held after the stop: a process of the old controller survives; '
                             'find it (fuser controller.lock) and stop it by PID')
        if a.config_patch:
            patch = read_json(a.config_patch)
            config = read_json(out / 'config.json')
            save_json(out / f'config.pre-patch-{time.strftime("%Y%m%d-%H%M%S")}.json', config)
            for key, value in patch.items():
                if value is None:
                    config.pop(key, None)
                else:
                    config[key] = value
            save_json(out / 'config.json', config)
            print(f'config patched: {sorted(patch)}', flush=True)
        rebound = False
        if new_tag:
            done = run(['bash', REBIND, out, old_tag, new_tag])
            print(done.stdout + done.stderr, flush=True)
            if done.returncode:
                raise SystemExit('rebind failed; the controller is stopped on its old runtime (relaunch with the same command)')
            rebound = True
            spec['payload'] = spec['payload'].replace(f'runtime-native-corrected-mix-{old_tag}', f'runtime-native-corrected-mix-{new_tag}')
            if not a.no_preflight:
                ok, detail = preflight(new_unit, out, DATA / f'runtime-native-corrected-mix-{new_tag}')
                print(f'preflight under {new_tag}: {"ok" if ok else "REFUSED"} {detail}', flush=True)
                if not ok:
                    run(['bash', REBIND, out, new_tag, old_tag])
                    spec['payload'] = spec['payload'].replace(new_tag, old_tag)
                    rebound, new_tag = False, None
                    print(f'rolled back to {old_tag}; relaunching the old runtime', flush=True)
        spec.setdefault('units', []).append(new_unit)
        spec['unit'] = new_unit
        save_json(out / 'launch.json', spec)
        launched_at = time.time()
        started = run(ledger_command(spec, new_unit))
        if started.returncode:
            raise SystemExit('relaunch not admitted by the ledger: ' + (started.stderr or started.stdout)[-500:])
        ok, detail = wait_resumed(new_unit, out, launched_at, a.resume_seconds)
        print(json.dumps({**plan, 'stop': outcome, 'runtime_after': new_tag or old_tag, 'rebound': rebound,
                          'resumed': ok, 'detail': detail}, indent=1))
        return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
