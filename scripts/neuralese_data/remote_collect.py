#!/usr/bin/env python3
"""A remote actor's collection runs its collector on the remote host, beside the actor (DECISIONS 2026-11-25
"collectors run beside their remote actors").

``expert_iteration.py collect --actor URL`` for a Colab actor used to run on the DGX: its node episode servers, the
pool rows and the scoring took ~0.2 GB of DGX host memory per slot (aei1-collect-v0001-remote0, 64 slots: 12-15 GB,
OOM-killed at 00:52 on 2026-11-25) while the VM's 50-160 GB sat unused. ``run`` (the DGX ledger unit, REMOTE_SYNC_GB)
puts the same ``collect`` command on the VM instead:

1. the code: the DGX pin (a clean ``pin_checkout`` of COMMIT) at the same absolute path on the VM, a git worktree of
   the session's clone at COMMIT verified clean (so ``code_provenance`` and ``held_eval.series`` read the commit and
   trees they read on the DGX) plus the pin's ignored build outputs rsynced from the DGX pin (ts-host/dist and
   node_modules, generated .d.nl.ts, .natlang); its venv is the session's (``host_config('colab')['python']``);
2. the data (``data_closure``) at the same absolute paths, symlinks included (``mirror_plan``): the registry roles the
   collector reads (grpo-pool, grpo-memetic-tasks, every held-eval* set with its cases and predecessors), the errata,
   the Neuralese variants, the task order, the custody sources (``custody._protected_source_paths``), the task asset
   roots (``run_bonsai_queue.TASK_ASSET_BUILD_ROOTS``) and every existing path a pool, memetic or variant row names;
   a path the VM-side collector then fails to open is reported (``missing-paths.jsonl``, an audit hook) and mirrored
   before the collector is launched again (at most MISSING_RELAUNCHES times);
3. the work directory: when the VM has none (a replaced session), the DGX copy's small files go up (no rollouts or
   episode workspaces: the collector resumes from rewards and journals), so the collector resumes where it was;
4. ``collect`` there, detached (setsid, under the session's admission claim of host RAM), against the VM-local front;
5. every ``--sync-seconds`` the work directory comes back over the session's ControlMaster (rsync -z: zstd on the
   wire), each group's scoring files (rollouts/rewards/score/versions) staged first and moved into place only after
   its content is pulled again, so a DGX reader (build, health, teacher, frontier) never sees scores whose episodes
   have not arrived;
6. SIGTERM (the driver's stop, a ledger preemption) stops the VM collector and pulls once more; a VM collector that
   exits ends the unit with its exit code (the driver's standing check starts it again). Earlier work directories this
   host collected into are pulled once more at the next start (the last groups of a stopped policy's collection).

    remote_collect.py run --host colab-g4 --work W [--sync-seconds 60] -- PY .../expert_iteration.py collect ...
    remote_collect.py vm --work W -- PY .../expert_iteration.py collect ...      # on the VM (``run`` launches it)
    remote_collect.py closure --tasks T [--neuralese-variants V]                  # the data paths, one per line
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if (REPO / 'training' / 'neuralese').is_dir():     # not the VM wrapper's copy in a work directory (``vm_script``)
    for _p in (str(REPO / 'training' / 'neuralese'), str(REPO), str(HERE), str(REPO / 'scripts')):
        if _p not in sys.path:
            sys.path.insert(0, _p)

REMOTE_SYNC_GB = 3          # the DGX unit: this process and its rsyncs (file lists of ~100k files), no episodes
SYNC_SECONDS = 60
def mirror_roots():
    """Logical coordinator roots, preserved when the standalone wrapper runs on its VM."""
    configured = os.environ.get('NATLANG_COLLECT_MIRROR_ROOTS')
    if configured is not None:
        roots = json.loads(configured)
        if not isinstance(roots, list) or not roots or any(not isinstance(p, str) or not Path(p).is_absolute() for p in roots):
            raise ValueError('NATLANG_COLLECT_MIRROR_ROOTS must be a nonempty JSON list of absolute paths')
    else:
        from natlang_neuralese.common import paths
        roots = [paths.root(name) for name in ('data_nvme', 'data_hdd', 'models')]
        # Use the configured canonical checkout, not a pin NATLANG_REPO may name.
        roots.append(Path(paths._profile().get('repo') or paths.DEFAULTS['repo']).expanduser())
    return tuple(sorted({str(root).rstrip('/') + '/' for root in roots}))


MIRROR_ROOTS = mirror_roots()   # coordinator paths, also passed to the standalone VM wrapper
# a group's scoring outputs: a DGX reader takes a group's outcome from these, so they arrive after its content
MARKERS = ('rollouts.jsonl', 'rewards.jsonl', 'score.json', 'versions.json')
# pin outputs never mirrored: aarch64 native modules of an application the collector does not run
PIN_EXCLUDE = ('applications/pi/',)
# a work directory pushed to a fresh VM: what the collector resumes from (rewards, journals, queues), not the bulk
PUSH_EXCLUDE = ('rollouts.jsonl', 'journal.jsonl', 'row-*/', '*.log', 'actor/', 'missing-paths.jsonl',
                'remote-collect.exit')
# never pulled: an episode's raw job files (the ts-host collector rewrites its partial and appends its evidence while the
# episode runs and removes both when it ends; a pull without --delete kept every in-flight copy on the DGX: 2-3 GB/hour
# for 64 slots on 2026-11-25), the result's byte copy of results.jsonl and atomic-write temp files. What a finished
# episode keeps arrives compacted (``episode_files``: ``*.zst``). NATLANG_REMOTE_PULL_RAW_JOBS=1 opts out.
PULL_EXCLUDE = ('/groups/*/row-*/*/jobs/*.partial.json', '/groups/*/row-*/*/jobs/*.partial.json.evidence-*.jsonl',
                '/groups/*/row-*/*/jobs/*.result.json', '/groups/*/row-*/*/jobs/*.tmp-*')
# a sample's results.jsonl the DGX holds compacted (``episode_files.compact_results``, ``ei_round_tier``: once its group is
# scored) is never pulled raw again: each pull excludes them by path. ``ei_round_tier`` compacts a pulled collection
# only while its live puller's code names this constant.
COMPACTED_RESULTS_EXCLUDED = True
MISSING_RELAUNCHES = 3
LOST_SECONDS = 1800         # a VM that answers no ssh this long is gone: the unit ends (rc 75) and the driver restarts it
SESSION_LOST_RC = 75
PATH_RE = re.compile(rb'(?:' + b'|'.join(re.escape(r.rstrip('/').encode()) for r in MIRROR_ROOTS) + rb')/[A-Za-z0-9._+\-/]+')
MIN_DIR_DEPTH = 5           # a scanned directory reference this shallow (e.g. the checkout's runs/) is never mirrored


def log(**event):
    print(json.dumps({'time': round(time.time(), 1), **event}, default=str), flush=True)


# ---- the data a collection reads --------------------------------------------------------------------------------------
def mirrored(path):
    return str(path).startswith(MIRROR_ROOTS)


def mirror_plan(paths):
    """(links, files): every symlink on the way to each of ``paths`` (``[(link path, its target as written)]``, outermost
    first) and the real files or directories to copy, so the VM resolves each lexical path as the DGX does (a registry
    role path ``data_nvme:x`` reaches the HDD root through a link; a pin's ``runs`` is a link to the checkout's)."""
    links, files = {}, set()

    def walk(path, depth=0):
        if depth > 40:
            raise RuntimeError(f'symlink loop at {path}')
        parts = Path(os.path.normpath(path)).parts
        cur = Path(parts[0])
        for i, part in enumerate(parts[1:], 1):
            nxt = cur / part
            if nxt.is_symlink():
                target = os.readlink(nxt)
                links.setdefault(str(nxt), target)
                cur = Path(walk(os.path.join(str(nxt.parent), target), depth + 1))
            else:
                cur = nxt
        return str(cur)

    for path in paths:
        if not os.path.lexists(path):
            continue
        real = walk(str(path))
        if os.path.exists(real):
            files.add(real)
    return sorted(links.items()), sorted(files)


def scan_paths(files, cache=None):
    """Existing paths under MIRROR_ROOTS that the text of ``files`` names (task rows' service code reads asset roots by
    absolute path); directories shallower than MIN_DIR_DEPTH are left out. ``cache`` (a JSON file): results per
    (path, size, mtime), so a 500 MB pool is read again only when it changed."""
    known = {}
    if cache and Path(cache).exists():
        try:
            known = json.loads(Path(cache).read_text())
        except ValueError:
            known = {}
    out, changed = set(), False
    for f in files:
        try:
            st = os.stat(f)
        except OSError:
            continue
        key = f'{f}|{st.st_size}|{st.st_mtime_ns}'
        if key not in known:
            found = set()
            with open(f, 'rb') as handle:
                for line in handle:
                    found.update(m.decode() for m in PATH_RE.findall(line))
            keep = set()
            for p in found:
                p = p.rstrip('/.')
                if os.path.isfile(p) or (os.path.isdir(p) and len(Path(p).parts) > MIN_DIR_DEPTH):
                    keep.add(p)
            known[key], changed = sorted(keep), True
        out.update(known[key])
    if cache and changed:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(known))
    # a file inside a mirrored directory is covered by it
    dirs = sorted(p for p in out if os.path.isdir(p))
    return sorted(p for p in out if not any(p != d and p.startswith(d + '/') for d in dirs))


def option(argv, flag, default=None):
    """The value of ``flag`` in ``argv`` (``--flag V`` or ``--flag=V``), else ``default``."""
    for i, item in enumerate(argv):
        if item == flag and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(flag + '='):
            return item.split('=', 1)[1]
    return default


def options(argv, flag):
    return [argv[i + 1] for i, item in enumerate(argv) if item == flag and i + 1 < len(argv)] + \
        [item.split('=', 1)[1] for item in argv if item.startswith(flag + '=')]


def held_closure(manifest, seen=None):
    """A held manifest, its own cases file and every earlier manifest it ``also_excludes`` (``held_source_groups``)."""
    seen = set() if seen is None else seen
    path = Path(manifest)
    if str(path) in seen or not path.exists():
        return seen
    seen.add(str(path))
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return seen
    if isinstance(data.get('cases'), dict) and data['cases'].get('path'):
        seen.add(str(path.parent / data['cases']['path']))
    for earlier in data.get('also_excludes') or ():
        if isinstance(earlier, dict) and earlier.get('path'):
            held_closure(earlier['path'], seen)
    engine = path.parent / 'engine.json'
    if engine.exists():
        seen.add(str(engine))
    return seen


def role_files(reg, name):
    """A registry role's files as the VM-side code names them (its lexical ``absolute(entry path)/file``) and as the DGX
    resolves them (``Registry.role``: real paths)."""
    from natlang_neuralese.common import data_registry
    out = []
    for entry, real in reg.role(name):
        spec = (entry.get('roles') or {}).get(name)
        spec = spec if isinstance(spec, dict) else {}
        try:
            base = data_registry.absolute(entry['path'])
            out.append(str(base / spec['file']) if spec.get('file') else str(base))
        except ValueError:
            pass
        out.append(str(real))
    return out


def data_closure(collect_argv, cache=None, pin=None):
    """Every data path (absolute, under MIRROR_ROOTS) the collector of ``collect_argv`` reads, resolved as the code of
    ``pin`` (the collector's checkout; default this one) resolves them: its registry's roles, its custody registry
    (checkout-relative sources: PIN/runs/... through the pin's links); see the module doc."""
    import grpo
    from natlang_neuralese.common import data_registry
    from natlang_neuralese.common import custody
    pin = Path(pin).resolve() if pin else REPO
    reg = data_registry.load(repo=pin)
    paths = set()
    roles = ['grpo-pool', 'grpo-memetic-tasks', *sorted({n for i in reg.git_order
                                                         for n in (reg.git[i].get('roles') or {})
                                                         if n.startswith('held-eval')})]
    role_paths = {name: role_files(reg, name) for name in roles}
    pools = options(collect_argv, '--pool') or role_paths['grpo-pool']
    memetic = option(collect_argv, '--memetic-tasks') or next(iter(role_paths['grpo-memetic-tasks']), None)
    for name in roles:
        paths.update(role_paths[name])
    paths.update(pools)
    held = option(collect_argv, '--held-manifest') or grpo.default_held()
    for manifest in [held, *role_paths.get('held-eval', ()), *[p for n in roles if n.startswith('held-eval')
                                                                for p in role_paths[n]]]:
        paths.update(held_closure(manifest))
    if Path(grpo.ERRATA).is_dir():
        paths.add(grpo.ERRATA)
    variants = option(collect_argv, '--neuralese-variants')
    for flag in ('--tasks', '--neuralese-variants', '--memetic-tasks', '--pass-rates', '--hindsight-trajectories'):
        value = option(collect_argv, flag)
        if value:
            paths.add(value)
    if memetic:
        paths.add(memetic)
    # custody: every file ProtectedWorlds.load reads (registry entries under the pin travel with the code)
    root = pin
    declared = custody._read_protected_registry(custody._source_path(custody.REGISTRY, root))
    paths.update(item[0] for item in custody._protected_source_paths(declared, root))
    from run_bonsai_queue import TASK_ASSET_BUILD_ROOTS
    from natlang_neuralese.common.paths import resolve
    for name, parts in TASK_ASSET_BUILD_ROOTS.items():
        paths.add(os.environ.get(name) or str(resolve(*parts)))
    paths.add(str(data_registry.journal_path()))      # the registry's local layer (read by every registry load)
    # every existing path among grpo's train defaults (collect builds its Collector from them: --pass-rates' base rates,
    # whose absence the collector tolerates silently, the behaviour probe, ...)
    try:
        defaults = vars(grpo.build_parser().parse_args(['train', '--work', '/nonexistent', '--checkpoint', '/nonexistent']))
    except SystemExit:
        defaults = {}
    for value in defaults.values():
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(item, str) and mirrored(item) and os.path.exists(item):
                paths.add(item)
    for name in ('NATLANG_RL_TASK_ASSETS',):
        if os.environ.get(name):
            paths.add(os.environ[name])
    texts = [p for p in [*pools, memetic, variants] if p]
    paths.update(scan_paths(texts, cache))
    return sorted(str(p) for p in paths if p and mirrored(p) and os.path.lexists(p))


# ---- the remote host ----------------------------------------------------------------------------------------------------
def ssh(host, command, *, timeout=600, check=True, input=None, attempts=3):
    """``pop_rl_actor.ssh_host``, again after a connection failure (rc 255: the session's proxy dropped a channel)."""
    import pop_rl_actor
    for attempt in range(attempts):
        try:
            return pop_rl_actor.ssh_host(host, command, timeout=timeout, check=check, input=input)
        except RuntimeError as error:
            if '(255)' not in str(error) or attempt == attempts - 1:
                raise
            time.sleep(10)


def rsync(*args, timeout=3600, check=True):
    """rsync over the host's ControlMaster (ssh config ``ControlMaster auto``), compressed (zstd with rsync >= 3.2),
    resumable and failing a stalled stream after 120 s."""
    compress = [] if os.environ.get('NATLANG_COPY_COMPRESS', '1') == '0' else ['-z']
    # --no-owner/--no-group: the VM side (root) keeps its own ownership (a pin directory owned by the DGX's uid is
    # "dubious" to git there; code_provenance and canonical_checkout_repo would fail)
    return subprocess.run(['rsync', '-a', '--no-owner', '--no-group', *compress, '--partial', '--timeout=120',
                           *map(str, args)], check=check,
                          timeout=timeout, capture_output=True, text=True)


def pin_outputs(pin):
    """The pin's ignored build outputs (``git ls-files -o -i``) that the VM copy needs: not the data links
    (``pin_checkout.LINKED``/``SHARED``, ``data/`` children), not SKIPPED scratch, not PIN_EXCLUDE."""
    import pin_checkout
    listed = subprocess.run(['git', '-C', str(pin), 'ls-files', '-o', '-i', '--exclude-standard', '--directory'],
                            check=True, capture_output=True, text=True).stdout.splitlines()
    out = []
    for rel in listed:
        name = rel.rstrip('/')
        top = name.split('/', 1)[0]
        if (top in pin_checkout.LINKED or any(name == s or name.startswith(s + '/') for s in pin_checkout.SHARED)
                or rel.startswith(pin_checkout.SKIPPED) or rel.startswith(PIN_EXCLUDE) or (Path(pin) / name).is_symlink()):
            continue
        out.append(name)
    if (Path(pin) / '.natlang-pin.json').exists():
        out.append('.natlang-pin.json')
    return sorted(set(out))


def ensure_remote_pin(host, cfg, pin, commit, scratch):
    """The DGX pin at the same path on ``host``: a detached worktree at ``commit`` (verified clean) of a checkout at the
    DGX pin's canonical checkout path (``data_registry.canonical_checkout_repo``; a VM checkout
    sharing the session clone's objects), so checkout-relative registry paths (``runs/...``) resolve where the DGX
    resolves them; the pin's build outputs rsynced over it, its venv the session's and its data links as on the DGX."""
    from natlang_neuralese.common.data_registry import canonical_checkout_repo
    pin = Path(pin)
    owner = str(canonical_checkout_repo(pin))
    q = shlex.quote
    clone, p = q(cfg['repo']), q(str(pin))
    g = 'git -c safe.directory=*'
    steps = (f'set -e; {g} -C {clone} cat-file -e {q(commit)}^{{commit}} 2>/dev/null || {g} -C {clone} fetch -q origin; '
             # the mirror made the directories with the DGX's uid: git (and canonical_checkout_repo) want the VM user's
             f'mkdir -p {q(owner)}; chown "$(id -u):$(id -g)" {q(owner)}; '
             f'if test -e {p}; then chown "$(id -u):$(id -g)" {p}; fi; '
             f'if ! test -e {q(owner)}/.git; then {g} init -q {q(owner)}; '
             f'echo {clone}/.git/objects > {q(owner)}/.git/objects/info/alternates; '
             f'{g} -C {q(owner)} remote add origin {clone}; fi; '
             f'{g} -C {q(owner)} cat-file -e {q(commit)}^{{commit}} 2>/dev/null || {g} -C {q(owner)} fetch -q origin; '
             # a pin made earlier from another checkout (the session clone) is made again from the owner
             f'common="$({g} -C {p} rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"; '
             f'if test -n "$common" && test "$common" != "$({g} -C {q(owner)} rev-parse --path-format=absolute '
             f'--git-common-dir)"; then {g} -C "$(dirname "$common")" worktree remove --force {p} || rm -rf {p}; fi; '
             f'{g} -C {clone} worktree prune; {g} -C {q(owner)} worktree prune; '
             f'if ! test -e {p}/.git; then mkdir -p {q(str(pin.parent))}; '
             f'{g} -C {q(owner)} worktree add -f -q --detach {p} {q(commit)}; fi; '
             f'{g} -C {p} checkout -q -f --detach {q(commit)}; {g} -C {p} rev-parse HEAD; '
             f'{g} -C {p} status --porcelain --untracked-files=no | head -5')
    out = ssh(host, f'flock -w 600 {q(cfg["root"] + "/.tenant-checkout.lock")} bash -c {q(steps)}', timeout=900).split()
    if out != [commit]:
        raise RuntimeError(f'{host}: the collect pin {pin} is not clean at {commit}: {" ".join(out)[:300]}')
    listing = Path(scratch) / 'pin-outputs.txt'
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text('\n'.join(pin_outputs(pin)) + '\n')
    rsync('-r', '--files-from', listing, f'{pin}/', f'{host}:{pin}/', timeout=7200)
    ssh(host, f'chown "$(id -u):$(id -g)" {q(str(pin))}', timeout=60)    # pins mirrored before --no-owner
    # links: the venv is the session's; data roots point where the DGX pin's do (mirrored there by mirror_data)
    links = [('.venv-neuralese', f'{cfg["repo"]}/.venv-neuralese')]
    for name in ('runs', 'data', 'models'):
        if (pin / name).is_symlink():
            links.append((name, os.readlink(pin / name)))
    script = '; '.join(f'mkdir -p {q(t)} 2>/dev/null; test -L {q(str(pin / n))} || ln -sfn {q(t)} {q(str(pin / n))}'
                       if not n.startswith('.venv') else
                       f'test -L {q(str(pin / n))} || ln -sfn {q(t)} {q(str(pin / n))}' for n, t in links)
    ssh(host, script + '; true', timeout=120)
    return {'pin': str(pin), 'commit': commit, 'outputs': sum(1 for _ in listing.open())}


def mirror_data(host, paths, scratch):
    """``paths`` (and the symlinks on their way, ``mirror_plan``) at the same absolute paths on ``host``."""
    links, files = mirror_plan(paths)
    q = shlex.quote
    if links:
        # a link exists already, or its place is free: never replace a real file or directory on the VM
        script = '\n'.join(f'mkdir -p {q(str(Path(l).parent))}; test -L {q(l)} || test -e {q(l)} || ln -s {q(t)} {q(l)}'
                           for l, t in links)
        ssh(host, 'bash -s', input=script + '\ntrue\n', timeout=600)
    listing = Path(scratch) / 'data-files.txt'
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_text(''.join(f[1:] + '\n' for f in files))     # relative to /
    if files:
        rsync('-r', '-L', '--files-from', listing, '/', f'{host}:/', timeout=4 * 3600)
    return {'links': len(links), 'files': len(files)}


def push_work(host, work):
    """A fresh VM's copy of the work directory: what the collector resumes from (PUSH_EXCLUDE left out)."""
    ssh(host, f'mkdir -p {shlex.quote(str(work))}')
    rsync(*[f'--exclude={e}' for e in PUSH_EXCLUDE], f'{work}/', f'{host}:{work}/', timeout=4 * 3600)


def pull_job_excludes():
    """rsync excludes of ``PULL_EXCLUDE`` (none with NATLANG_REMOTE_PULL_RAW_JOBS=1)."""
    if os.environ.get('NATLANG_REMOTE_PULL_RAW_JOBS') == '1':
        return []
    return [f'--exclude={e}' for e in PULL_EXCLUDE]


def compacted_results_excludes(work, stage):
    """``--exclude-from`` for the samples whose results.jsonl the DGX holds as ``results.jsonl.zst`` (none: [])."""
    work = Path(work)
    paths = sorted('/' + str(p.relative_to(work))[:-len('.zst')]
                   for p in work.glob('groups/*/row-*/*/results.jsonl.zst') if p.is_file())
    if not paths:
        return []
    listing = Path(stage) / 'compacted-results.exclude'
    listing.write_text(''.join(f'{p}\n' for p in paths))
    return [f'--exclude-from={listing}']


def pull(host, work, *, timeout=1800):
    """The VM's work directory into the DGX's: content first (markers left out), the markers into a staging tree, the
    content again, then each changed marker moved into place (``os.replace``: atomic), so a DGX reader sees a group's
    scores only with the episodes they name. Returns the number of markers moved."""
    work = Path(work)
    stage = work / '.remote-stage'
    stage.mkdir(parents=True, exist_ok=True)
    content = ['--exclude=/.remote-stage', '--exclude=/.remote-scratch', '--exclude=/.remote-vm',
               '--exclude=/.remote-collect.lock', '--exclude=/remote-collector.json', '--exclude=/collector.json',
               *[f'--exclude=/groups/*/{m}' for m in MARKERS], *pull_job_excludes(),
               *compacted_results_excludes(work, stage)]
    rsync(*content, f'{host}:{work}/', f'{work}/', timeout=timeout)
    # only new or changed markers land in the stage (--compare-dest: one identical in WORK is not transferred)
    rsync('--include=/groups/', '--include=/groups/*/', *[f'--include=/groups/*/{m}' for m in MARKERS],
          '--exclude=*', f'--compare-dest={work.resolve()}/', f'{host}:{work}/', f'{stage}/', timeout=timeout)
    staged = [p for p in sorted(stage.glob('groups/*/*')) if p.name in MARKERS and p.is_file()]
    if staged:
        rsync(*content, f'{host}:{work}/', f'{work}/', timeout=timeout)
    for path in staged:
        dest = work / path.relative_to(stage)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, dest)
    return len(staged)


def vm_script(work):
    """Where ``run`` puts this file on the VM: the ``vm`` wrapper (stdlib only) is the driver's, the collector the pin's
    (a pin older than this module has no copy of it)."""
    return f'{work}/.remote-vm/remote_collect.py'


def vm_command(cfg, pin, work, collect_argv, slots):
    """The shell command that launches ``collect`` on the VM, detached: setsid (its PID leads the process group
    ``stop_remote`` signals), the session's env file (node on PATH), the pin on PYTHONPATH, a host-RAM admission claim,
    the exit code into WORK/remote-collect.exit. Prints the PID."""
    q = shlex.quote
    python = f'{pin}/.venv-neuralese/bin/python'
    claim = round(6 + 0.2 * int(slots), 1)
    inner = [python, vm_script(work), 'vm', '--work', str(work), '--', *collect_argv]
    admitted = [cfg['python'], cfg['admission'], 'run', '--vram-gib', '0', '--host-ram-gib', str(claim),
                '--label', f'collect-{Path(work).name}', '--wait', '86400', '--workdir', str(pin), '--',
                '/usr/bin/env', f'NATLANG_REPO={pin}', f'PYTHONPATH={pin}/training/neuralese:{pin}',
                'NATLANG_COLLECT_MIRROR_ROOTS=' + json.dumps(MIRROR_ROOTS),
                'HF_HUB_OFFLINE=1', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=1', *inner]
    body = (f'{shlex.join(admitted)} >> {q(str(work) + "/collect.log")} 2>&1; '
            f'echo $? > {q(str(work) + "/remote-collect.exit")}')
    return (f'. {q(cfg["env"])} || exit 1; mkdir -p {q(str(work))}; rm -f {q(str(work) + "/remote-collect.exit")}; '
            # separate statements: `cd X && setsid ... &` backgrounds the list in a subshell that holds the ssh channel
            f'cd {q(str(pin))} || exit 1; setsid nohup bash -c {q(body)} > /dev/null 2>&1 < /dev/null & echo $!')


def remote_status(host, work, pid):
    """'running', ('exited', rc) or 'unreachable'."""
    q = shlex.quote
    try:
        out = ssh(host, f'kill -0 {int(pid)} 2>/dev/null && echo running || cat {q(str(work) + "/remote-collect.exit")} '
                        f'2>/dev/null || echo gone', timeout=60, check=False).strip()
    except Exception:
        return 'unreachable'
    if not out:
        return 'unreachable'
    last = out.splitlines()[-1]
    if last == 'running':
        return 'running'
    return ('exited', int(last) if last.lstrip('-').isdigit() else -1)


def stop_remote(host, pid):
    ssh(host, f'kill -TERM -- -{int(pid)} 2>/dev/null; for i in $(seq 20); do kill -0 {int(pid)} 2>/dev/null || exit 0; '
              f'sleep 1; done; kill -KILL -- -{int(pid)} 2>/dev/null; true', check=False, timeout=60)


def missing_on_dgx(work, mirrored_paths):
    """Paths the VM collector failed to open (``missing-paths.jsonl``) that exist here and were not mirrored yet."""
    path = Path(work) / 'missing-paths.jsonl'
    if not path.exists():
        return []
    out = set()
    for line in path.read_text().splitlines():
        try:
            p = json.loads(line)['path']
        except (ValueError, KeyError):
            continue
        if mirrored(p) and os.path.lexists(p) and p not in mirrored_paths:
            out.add(p)
    return sorted(out)


def record_path(work):
    return Path(work) / 'remote-collector.json'


def write_record(work, record):
    from natlang_neuralese.common.jsonio import write_json_durable
    write_json_durable(record_path(work), record)


def sweep_earlier(host, work):
    """Pull once more every earlier work directory (siblings of ``work``) this host collected into and that was not
    pulled after its collector ended (a stop that ran out of time, a session lost mid-pull)."""
    for other in sorted(Path(work).parent.glob('*/remote-collector.json')):
        if other.parent == Path(work):
            continue
        try:
            rec = json.loads(other.read_text())
        except ValueError:
            continue
        if rec.get('host') != host or rec.get('final_pull'):
            continue
        try:
            pull(host, other.parent)
            write_record(other.parent, {**rec, 'final_pull': time.time()})
            log(event='earlier collection pulled', work=str(other.parent))
        except Exception as error:
            log(event='earlier collection pull failed', work=str(other.parent), error=repr(error)[:300])


def run(args):
    import pop_rl_actor
    host, work = args.host, Path(args.work)
    argv = list(args.collect)
    if len(argv) < 3 or not argv[1].endswith('expert_iteration.py') or argv[2] != 'collect':
        raise SystemExit('remote_collect run: the command after -- is PY .../expert_iteration.py collect ...')
    if Path(option(argv, '--work', '')).resolve() != work.resolve():
        raise SystemExit('remote_collect run: --work differs from the collect command\'s --work')
    pin = Path(argv[1]).resolve().parents[2]
    marker = json.loads((pin / '.natlang-pin.json').read_text())
    commit = marker['commit']
    dirty = subprocess.run(['git', '-C', str(pin), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f'{pin} has modified tracked files: a remote collection mirrors a clean pin only')
    cfg = pop_rl_actor.host_config('colab', host=host)
    scratch = work / '.remote-scratch'
    work.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    started = time.time()
    sweep_earlier(host, work)
    code = ensure_remote_pin(host, cfg, pin, commit, scratch)
    paths = data_closure(argv, cache=work.parent / '.remote-closure-cache.json', pin=pin)
    data = mirror_data(host, paths, scratch)
    log(event='remote collection mirrored', host=host, pin=code, data=data, paths=len(paths),
        seconds=round(time.time() - started, 1))
    fresh = ssh(host, f'test -e {shlex.quote(str(work))}/runs.jsonl && echo kept || echo fresh', timeout=60).strip()
    if fresh.endswith('fresh') and (work / 'runs.jsonl').exists():
        push_work(host, work)
        log(event='work directory pushed', host=host, work=str(work))
    ssh(host, f'mkdir -p {shlex.quote(str(work))}/.remote-vm')
    rsync(Path(__file__).resolve(), f'{host}:{vm_script(work)}', timeout=300)
    # a collector an earlier unit started and lost (killed, or its launch timed out): its process group, by the pgid
    # the VM wrapper recorded while it holds the work directory's lock
    pgid = ssh(host, f'cat {shlex.quote(str(work))}/.remote-collect.pgid 2>/dev/null || true', timeout=60).strip()
    if pgid.isdigit() and remote_status(host, work, int(pgid)) == 'running':
        stop_remote(host, int(pgid))
        log(event='earlier remote collector stopped', pgid=int(pgid))
    mirrored_paths, relaunches, rc = set(paths), 0, 0
    slots = int(option(argv, '--slots', '64'))
    while not stop.is_set():
        pid = int(ssh(host, vm_command(cfg, pin, work, argv, slots), timeout=120).strip().splitlines()[-1])
        record = {'host': host, 'pid': pid, 'work': str(work), 'pin': str(pin), 'commit': commit,
                  'launched': time.time(), 'argv': argv, 'final_pull': None}
        write_record(work, record)
        log(event='remote collector launched', host=host, pid=pid, work=str(work), slots=slots)
        last_seen, status = time.time(), 'running'
        while not stop.wait(args.sync_seconds):
            status = remote_status(host, work, pid)
            if status == 'unreachable':
                if time.time() - last_seen > LOST_SECONDS:
                    log(event='remote host lost', host=host, seconds=round(time.time() - last_seen))
                    return SESSION_LOST_RC
                continue
            last_seen = time.time()
            try:
                moved = pull(host, work)
                if moved:
                    log(event='pulled', markers=moved)
            except Exception as error:
                log(event='pull failed', error=repr(error)[:300])
            if status != 'running':
                break
        if stop.is_set():
            stop_remote(host, pid)
            try:
                pull(host, work, timeout=40)
                write_record(work, {**record, 'final_pull': time.time(), 'stopped': True})
            except Exception as error:
                log(event='final pull failed', error=repr(error)[:300])
            return 0
        rc = status[1] if isinstance(status, tuple) else -1
        try:
            pull(host, work)
            write_record(work, {**record, 'final_pull': time.time(), 'rc': rc})
        except Exception as error:
            log(event='final pull failed', error=repr(error)[:300])
        missing = missing_on_dgx(work, mirrored_paths)
        log(event='remote collector exited', rc=rc, missing=missing[:20])
        if rc == 0 or not missing or relaunches >= MISSING_RELAUNCHES:
            return rc
        mirror_data(host, missing, scratch)
        mirrored_paths.update(missing)
        relaunches += 1
    return rc


# ---- on the VM ------------------------------------------------------------------------------------------------------------
def reads(mode, flags):
    """Whether an audited ``open`` (``open()``: a mode string; ``os.open``: flags) only reads."""
    if isinstance(mode, str):
        return not any(c in mode for c in 'wax+')
    if isinstance(flags, int):
        return not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND)
    return True


def descendants(root, proc=Path('/proc')):
    """PIDs below ``root`` (children first-level and deeper), from /proc's parent links."""
    children = {}
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            ppid = int((entry / 'stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(entry.name))
    out, todo = [], [root]
    while todo:
        for child in children.get(todo.pop(), ()):
            out.append(child)
            todo.append(child)
    return out


def vm(args):
    """The VM side: the collector in a child process (``vm-inner``) under this supervisor, which is the children's
    subreaper (PR_SET_CHILD_SUBREAPER): the episode servers and queue workers start their own sessions, so a process
    group signal misses them (2026-11-25: 59 workers and node servers outlived a stopped collector); on SIGTERM or the
    child's exit every descendant is stopped (TERM, then KILL). One collector per work directory (a lock)."""
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    import ctypes
    import fcntl
    guard = open(work / '.remote-collect.lock', 'a')
    try:       # one collector per work directory, whatever launched it
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f'{work}: another collector runs here')
    (work / '.remote-collect.pgid').write_text(f'{os.getpgid(0)}\n')   # ``run`` stops a collector it lost track of
    try:
        ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0)        # PR_SET_CHILD_SUBREAPER
    except (OSError, AttributeError):
        pass
    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'vm-inner', '--work', str(work), '--',
                              *args.collect])
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    while child.poll() is None and not stopping.wait(1):
        pass
    if stopping.is_set() and child.poll() is None:
        child.send_signal(signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in descendants(os.getpid()):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    try:
        while os.waitpid(-1, os.WNOHANG)[0]:     # reap what was reparented here
            pass
    except ChildProcessError:
        pass
    sys.exit(child.returncode if child.returncode is not None and child.returncode >= 0 else 143)


def vm_inner(args):
    """``collect`` in this process with an audit hook that records every path under MIRROR_ROOTS it fails to open
    (WORK/missing-paths.jsonl: the DGX unit mirrors them and launches again)."""
    work = Path(args.work)
    report = work / 'missing-paths.jsonl'
    seen, lock = set(), threading.Lock()
    inside = str(work.resolve()) + '/'

    def hook(event, hook_args):
        if event != 'open' or not hook_args:
            return
        path = hook_args[0]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        path = os.path.abspath(os.fsdecode(path))
        mode = hook_args[1] if len(hook_args) > 1 else 'r'
        flags = hook_args[2] if len(hook_args) > 2 else 0
        if not mirrored(path) or path.startswith(inside) or '/__pycache__/' in path or not reads(mode, flags):
            return
        if path in seen or os.path.exists(path):
            return
        with lock:
            seen.add(path)
            with open(report, 'a') as handle:   # an open under the hook: work is not missing
                handle.write(json.dumps({'path': path, 'at': round(time.time(), 1)}) + '\n')

    argv = list(args.collect)
    script = Path(argv[1])
    sys.addaudithook(hook)
    sys.argv = [str(script), *argv[2:]]
    sys.path[0] = str(script.parent)     # as `python SCRIPT` runs it (expert_iteration imports grpo beside it)
    import runpy
    runpy.run_path(str(script), run_name='__main__')


def closure_cli(args):
    for path in data_closure(['collect', *(['--tasks', args.tasks] if args.tasks else []),
                              *(['--neuralese-variants', args.neuralese_variants] if args.neuralese_variants else [])]):
        print(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--host', required=True)
    r.add_argument('--work', required=True)
    r.add_argument('--sync-seconds', type=float, default=SYNC_SECONDS)
    r.add_argument('collect', nargs=argparse.REMAINDER)
    for name in ('vm', 'vm-inner'):
        v = sub.add_parser(name)
        v.add_argument('--work', required=True)
        v.add_argument('collect', nargs=argparse.REMAINDER)
    c = sub.add_parser('closure')
    c.add_argument('--tasks')
    c.add_argument('--neuralese-variants')
    args = parser.parse_args()
    if getattr(args, 'collect', None) and args.collect[:1] == ['--']:
        args.collect = args.collect[1:]
    if args.cmd == 'run':
        sys.exit(run(args))
    if args.cmd == 'vm':
        return vm(args)
    if args.cmd == 'vm-inner':
        return vm_inner(args)
    return closure_cli(args)


if __name__ == '__main__':
    main()
