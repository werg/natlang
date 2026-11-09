#!/usr/bin/env python3
"""Restart a live interleaved RL controller at any time: stop it cooperatively, relaunch it under a fresh ledger unit
from the canonical checkout and check that it resumed.

    python3 scripts/neuralese_data/restart_controller.py CONTROL_OUT [--in-place | --to-checkout] [--at-round-boundary]
        [--grace SECONDS | --patient] [--no-preflight] [--dry-run]

CONTROL_OUT is the controller's ``out`` directory (it holds config.json, state.json, controller.log).  Production runs
directly from the canonical checkout (owner, 2026-11-09): a running process keeps its code, every newly
started worker, learner or actor gets current main, and every record names the git commit that produced it
(rl/collection_generations, common/code_provenance).  ``--in-place`` (the controller runs under controller_supervisor.py)
restarts only the controller process, inside its unit, with SIGUSR1: collector workers and the managed actor keep running and
the new controller re-attaches (rl/detached_processes.py).  Every other restart resumes from durable state
(plans/neuralese/CONTROLLER_RESTART.md lists the cost per phase):

1. cooperative stop: SIGTERM to the controller process only.  It stops its collector workers and actor, archives retained
   partial artifacts and returns; a learner or SFT trainer finishes its current update and checkpoints.  After
   ``--grace`` seconds (default 600) the unit is stopped hard (``systemctl stop``: SIGTERM to the whole cgroup, SIGKILL
   after the unit's timeout); the trainer's rolling checkpoint then resumes at its last committed update (at most one
   update is redone).  ``--patient`` waits for the clean boundary instead; ``--at-round-boundary`` first waits for the
   controller to finish its current update and start the next round;
2. ``--to-checkout`` (a controller still launched from a frozen runtime): rewrite its config, learner admission wrapper,
   env.sh, launch.json and state to run from the checkout (``migrate_*`` below; ``--dry-run`` prints every change), then
   the controller ``preflight`` under the checkout; a refusal restores the files and relaunches as before;
3. relaunch through ``memory_ledger.py run`` as ``<base>-rN`` with the recorded launch spec (``launch.json`` next to the
   config, derived from the old unit the first time), and wait for the resumed controller to rewrite its state.

LEGACY (delete once every controller runs from the checkout): ``--runtime-from-main [--commit SHA]``, ``--use-runtime
vNN-MMDD`` and ``--freeze-only`` freeze a runtime from main and rebind to it (``freeze_native_runtime.py``,
``rebind-runtime.sh``).
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
FROZEN_PATH_RE = re.compile(re.escape(str(DATA)) + r'/runtime-native-corrected-mix-v\d+-\d{4}')
# Config values that keep the frozen-runtime paths they record: provenance of a branch and the dev-evaluation runtime
# of the hash-bound trainer parameters (their identity is bound into the recipe), not the code that runs.
MIGRATION_KEEPS = ('branch_origin', 'trainer_parameters', 'seed_trainer_parameters')


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
    # The ledger workdir is the code root the payload runs: the checkout, or (legacy) a frozen runtime; a runtime switch
    # rewrites only the payload, so the old unit's WorkingDirectory would otherwise be carried over.
    found = RUNTIME_RE.findall(spec['payload'])
    workdir = str(DATA / f'runtime-native-corrected-mix-{found[0]}') if found else spec['workdir']
    argv = [sys.executable, LEDGER, 'run', '--unit', unit, '--budget-gb', str(spec['budget_gb']),
            '--priority', spec['priority'], '--workdir', workdir, '--wait', '600']
    if spec.get('fixed_claim'):
        argv.append('--fixed-claim')
    return [*argv, '--', '/bin/sh', '-c', spec['payload']]


def current_runtime(spec):
    """The frozen runtime tag the launch payload runs from, or None when it runs from the checkout."""
    found = RUNTIME_RE.findall(spec['payload'])
    return found[0] if found else None


# ---- migration of a frozen-runtime controller to run from the checkout --------------------------------------------

def checkout_environment():
    return {'NATLANG_REPO': str(REPO), 'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}'}


def _retarget(value):
    """``value`` with every frozen-runtime path replaced by the checkout (strings, lists and dicts)."""
    if isinstance(value, str):
        return FROZEN_PATH_RE.sub(str(REPO), value)
    if isinstance(value, list):
        return [_retarget(item) for item in value]
    if isinstance(value, dict):
        return {key: _retarget(item) for key, item in value.items()}
    return value


def migrate_config(config):
    """(new config, changes): config.json of a controller launched from a frozen runtime, rewritten to run from the
    checkout. Code identity is provenance: the actor Python inventory and rendering-asset hashes are dropped (every actor
    attests its git commit instead), the actor environment and the collector build name the checkout, and the Pop hook no
    longer names a runtime to install. Branch provenance and the hash-bound dev-evaluation runtime stay as recorded."""
    new = {key: (value if key in MIGRATION_KEEPS else _retarget(value)) for key, value in config.items()}
    for key in ('native_actor_python_runtime', 'native_actor_assets'):
        new.pop(key, None)
    if new.get('actor_environment') is not None or new.get('actor_command'):
        new['actor_environment'] = checkout_environment()
    new['runtime'] = f'{REPO}/ts-host'
    for hook in ('after_actor_ready', 'before_collect'):
        command = new.get(hook)
        if isinstance(command, list) and '--runtime' in command:
            at = command.index('--runtime')
            new[hook] = command[:at] + command[at + 2:]
    changes = sorted(key for key in set(config) | set(new) if config.get(key, '<absent>') != new.get(key, '<absent>'))
    return new, changes


SUPERVISOR = 'scripts/neuralese_data/controller_supervisor.py'


def with_supervisor(payload):
    """``payload`` (``... exec env K=V... python -u -m natlang_neuralese.rl.production run ...``) with the controller run
    under controller_supervisor.py: ``--in-place`` restarts then keep the unit, cgroup, claim, workers and actor."""
    if SUPERVISOR in payload:
        return payload
    return re.sub(r'(\S*python\S*)( -u)? -m natlang_neuralese\.rl\.production run ',
                  lambda m: f'{PYTHON} {REPO}/{SUPERVISOR} -- {m[1]}{m[2] or ""} -m natlang_neuralese.rl.production run ',
                  payload, count=1)


def migrate_launch(spec):
    """launch.json with the payload and ledger workdir of the checkout, the controller under the supervisor."""
    return {**spec, 'payload': with_supervisor(_retarget(spec['payload'])), 'workdir': str(REPO)}


def migrate_text(text):
    """A wrapper script or env.sh whose frozen-runtime paths now name the checkout."""
    return FROZEN_PATH_RE.sub(str(REPO), text)


def migration_files(out, config):
    """[(path, kind)] of the files besides config.json and launch.json that name the frozen runtime: the learner
    admission wrappers of the config's commands and the control directory's env.sh."""
    found = []
    for key in ('learner_command', 'seed_command'):
        for part in config.get(key) or []:
            if isinstance(part, str) and part.endswith('admit-learner.py') and Path(part).is_file() and (Path(part), 'wrapper') not in found:
                found.append((Path(part), 'wrapper'))
    env_sh = OPS / 'interleave-v49/env.sh'
    if env_sh.is_file():
        found.append((env_sh, 'env.sh'))
    return found


def migration_record(config, commit):
    return {'schema': 'natlang.rl-code-checkout-transition/1', 'transition': 'run-from-checkout-20261109',
            'from': {'runtime': config.get('actor_environment', {}).get('NATLANG_REPO') or config.get('runtime')},
            'to': {'checkout': str(REPO), 'git_commit': commit}, 'at': time.time()}


def plan_migration(out, spec):
    """Everything ``--to-checkout`` would change, without writing: {file: {'changed': [...]}} and the new launch spec."""
    config = read_json(Path(out) / 'config.json')
    new_config, changes = migrate_config(config)
    files = {str(Path(out) / 'config.json'): {'keys_changed': changes}}
    for path, kind in migration_files(out, config):
        text = path.read_text()
        files[str(path)] = {'kind': kind, 'lines_changed': sum(a != b for a, b in zip(text.splitlines(), migrate_text(text).splitlines()))}
    launch = migrate_launch(spec)
    return {'files': files, 'actor_environment': new_config.get('actor_environment'), 'runtime': new_config.get('runtime'),
            'dropped': [k for k in ('native_actor_python_runtime', 'native_actor_assets') if k in config],
            'launch_payload': launch['payload'], 'launch_workdir': launch['workdir'],
            'state': 'natlang.rl-code-checkout-transition/1 appended to state.json code_transitions'}, new_config, launch


def apply_migration(out, spec, stamp):
    """Write the migrated files (each keeping a ``.pre-checkout-<stamp>`` copy); returns the restore list."""
    out = Path(out)
    plan, new_config, launch = plan_migration(out, spec)
    config = read_json(out / 'config.json')
    restore = []

    def backup(path):
        copy = Path(str(path) + f'.pre-checkout-{stamp}')
        copy.write_bytes(Path(path).read_bytes())
        restore.append((Path(path), copy))
    for path, kind in migration_files(out, config):
        backup(path)
        path.write_text(migrate_text(path.read_text()))
    backup(out / 'config.json')
    save_json(out / 'config.json', new_config)
    state = read_json(out / 'state.json')
    if state is not None:
        backup(out / 'state.json')
        from natlang_neuralese.common.code_provenance import code_provenance
        state.setdefault('code_transitions', []).append(migration_record(config, code_provenance(REPO, fresh=True)['git_commit']))
        save_json(out / 'state.json', state)
    backup(out / 'launch.json') if (out / 'launch.json').is_file() else None
    save_json(out / 'launch.json', launch)
    return restore, launch


def restore_files(restore):
    for path, copy in restore:
        os.replace(copy, path)


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


def restart_in_place(unit, out, seconds):
    """SIGUSR1 to the unit's main process (controller_supervisor.py), which forwards it to the controller: during
    collection the controller detaches (collector workers and the managed actor keep running, rl.detached_processes), during
    an update or SFT window it finishes it and exits at the round boundary; the supervisor then starts the controller
    again from the checkout as it is now, inside the same unit and ledger claim. Nothing is stopped, rebound or re-admitted."""
    log = Path(out) / 'controller.log'
    before = log.stat().st_size if log.is_file() else 0
    launched_at = time.time()
    print(f'restart in place: SIGUSR1 to {unit} (collection detaches at once; an update finishes first)', flush=True)
    systemctl('kill', '--kill-whom=main', '-s', 'SIGUSR1', unit)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if unit_state(unit) not in ('active', 'activating'):
            return False, f'{unit} is {unit_state(unit)}; tail of controller.log: ' + (log.read_text()[-600:] if log.is_file() else '')
        tail = log.read_bytes()[before:].decode(errors='replace') if log.is_file() else ''
        if 'controller_supervisor: restarting' in tail:
            break
        time.sleep(3)
    else:
        return False, 'the controller did not restart in time (an update in flight finishes first; pass a larger --resume-seconds)'
    return wait_resumed(unit, out, launched_at, max(60., deadline - time.monotonic()))


def wait_round_boundary(out, seconds, poll=3.):
    """Wait until the controller has finished its current update/window and begins another round: ``state.round``
    advances (or the controller sits in ``collect`` of a round that has no collection binding yet). A bounded wait: on
    timeout the restart proceeds from wherever the controller is."""
    state_path = Path(out) / 'state.json'
    start = read_json(state_path) or {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = read_json(state_path) or {}
        round_dir = Path(out) / f"round-{state.get('round', 0):06d}"
        if state.get('round') != start.get('round') or (
                state.get('phase') == 'collect' and not (round_dir / 'collection-binding.json').exists()):
            print(f'round boundary: round {state.get("round")} phase {state.get("phase")!r}', flush=True)
            return True
        time.sleep(poll)
    print('round boundary not reached in time; restarting from the current phase', flush=True)
    return False


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path, help='controller out directory (config.json, state.json, controller.log)')
    p.add_argument('--unit', help='current controller unit (default: bf95-lowlr-<name>[-rN], the active one)')
    p.add_argument('--to-checkout', action='store_true',
                   help='move a controller launched from a frozen runtime to run from the checkout (config, admission '
                        'wrapper, env.sh, launch.json, state record); --dry-run prints every change')
    p.add_argument('--in-place', action='store_true',
                   help='restart the controller inside its unit without stopping workers or the actor (needs a controller '
                        'launched under controller_supervisor.py; --to-checkout does that)')
    p.add_argument('--at-round-boundary', action='store_true',
                   help='wait (up to --boundary-wait seconds) for the controller to finish its current update and begin the next round')
    p.add_argument('--boundary-wait', type=float, default=10800.)
    p.add_argument('--runtime-from-main', action='store_true', help='LEGACY: freeze main into a successor runtime and rebind to it')
    p.add_argument('--commit', help='LEGACY: commit to freeze (default HEAD of the canonical checkout)')
    p.add_argument('--use-runtime', metavar='TAG', help='LEGACY: rebind to an already frozen runtime (e.g. v58-1107)')
    p.add_argument('--grace', type=float, default=600., help='seconds of cooperative stop before the unit is stopped hard')
    p.add_argument('--patient', action='store_true', help='never escalate: wait for the clean update boundary')
    p.add_argument('--no-preflight', action='store_true')
    p.add_argument('--resume-seconds', type=float, default=300.)
    p.add_argument('--config-patch', type=Path, help='JSON object merged into config.json after the stop (top-level keys; null deletes)')
    p.add_argument('--dry-run', action='store_true', help='print the plan and the launch command; change nothing')
    p.add_argument('--freeze-only', action='store_true', help='LEGACY: only freeze the successor runtime (nothing is stopped)')
    a = p.parse_args(argv)
    out = a.out.resolve()
    unit = pick_unit(out, a.unit)
    spec = read_json(out / 'launch.json') or derive_launch(unit)
    spec['unit'] = unit
    old_tag = current_runtime(spec)
    legacy = a.runtime_from_main or a.use_runtime or a.freeze_only
    if legacy and old_tag is None:
        raise SystemExit('this controller already runs from the checkout: a plain restart picks up the newest main; '
                         'the frozen-runtime flags apply only to controllers still launched from a frozen runtime')
    if a.to_checkout and legacy:
        raise SystemExit('--to-checkout and the frozen-runtime flags are alternatives')
    if a.in_place and (legacy or a.to_checkout):
        raise SystemExit('--in-place is a restart without any runtime change')
    if a.to_checkout and old_tag is None:
        raise SystemExit('this controller already runs from the checkout (its launch payload names no frozen runtime)')
    new_unit = next_unit(unit, spec.get('units', []))
    plan = {'out': str(out), 'unit': unit, 'next_unit': new_unit, 'runtime': old_tag or 'checkout', 'budget_gb': spec['budget_gb'],
            'priority': spec['priority'], 'fixed_claim': spec['fixed_claim']}
    if a.in_place and SUPERVISOR not in spec['payload']:
        raise SystemExit('this controller does not run under controller_supervisor.py: its next restart is a stop/relaunch '
                         '(plain, or --to-checkout, which also moves it under the supervisor); --in-place works from then on')
    if a.dry_run and a.in_place:
        print(json.dumps({**plan, 'mode': 'in-place', 'signal': 'SIGUSR1 to the unit main process (the supervisor)',
                          'collection': 'workers and actor keep running; the relaunched controller re-attaches',
                          'update': 'finishes first; the controller restarts at the round boundary'}, indent=1))
        return 0
    if a.dry_run:
        report = {**plan, 'launch': ledger_command(spec, new_unit)[2:]}
        if a.to_checkout:
            migration, _, launch = plan_migration(out, spec)
            report = {**plan, 'to': 'checkout', 'migration': migration, 'launch': ledger_command(launch, new_unit)[2:],
                      'preflight': 'production preflight under the checkout after the stop (unless --no-preflight); '
                                   'a refusal restores every file and relaunches the frozen runtime'}
        print(json.dumps(report, indent=1))
        return 0
    if old_tag is not None and not a.to_checkout and not legacy:
        print(f'note: this controller still runs from frozen runtime {old_tag}; --to-checkout moves it to the checkout', flush=True)
    with open(out / 'restart.lock', 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit('another restart of this controller is running')
        if a.in_place:
            ok, detail = restart_in_place(unit, out, a.boundary_wait if a.at_round_boundary else a.resume_seconds)
            print(json.dumps({**plan, 'mode': 'in-place', 'resumed': ok, 'detail': detail}, indent=1))
            return 0 if ok else 1
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
        if a.at_round_boundary:
            wait_round_boundary(out, a.boundary_wait)
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
        restore = []
        if a.to_checkout:
            restore, spec = apply_migration(out, spec, time.strftime('%Y%m%d-%H%M%S'))
            print(f'migrated to run from the checkout {REPO}: {[str(path) for path, _ in restore]}', flush=True)
            if not a.no_preflight:
                ok, detail = preflight(new_unit, out, REPO)
                print(f'preflight under the checkout: {"ok" if ok else "REFUSED"} {detail}', flush=True)
                if not ok:
                    restore_files(restore)
                    spec = read_json(out / 'launch.json') or spec
                    restore = []
                    print('restored the frozen-runtime files; relaunching as before', flush=True)
            rebound = bool(restore)
        elif new_tag:
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
        print(json.dumps({**plan, 'stop': outcome, 'runtime_after': 'checkout' if rebound and a.to_checkout else (new_tag or old_tag or 'checkout'),
                          'rebound': rebound, 'resumed': ok, 'detail': detail}, indent=1))
        return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
