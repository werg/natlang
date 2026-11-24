#!/usr/bin/env python3
"""Standing storage rule for expert-iteration drivers: what rounds leave on the NVMe moves to the HDD as it goes cold.

An async driver (``ops/ei/<name>/phase-state.json``) adds about 15 GB per round on the NVMe: a collection per policy
(``collect/vNNNN-<sha12>*``: ``groups/`` and the actor's ``blocks/`` and ``sglang-hf/``) and a round directory
(``rNNNN``: ``policy/`` checkpoint and exports, ``sft/`` states, ``mix/``, ``filled/``). Run by ``storage_sweep.sh``
every 30 minutes; dry run unless ``--apply``.

Collections stay for the last two lineage policies, the active round's base (round builds read the current and
previous policy's collections) and the driver's reference and start policies (a rollback resumes their collection). Checkpoints and factors stay for those and for the held measurement's candidate and
reference, the last applied verdict, the driver's reference and start policies and ``ops/current-policy.json``.

Collections of an unprotected policy: the actor export (``actor/sglang-hf``, rebuilt from the checkpoint whenever an
actor serves it) is deleted, the rest moves to the HDD with a verified copy and a symlink back (``collections_of``,
``--prefer-unseen`` and ``trained_units`` readers resolve through it).

Round directories older than the previous lineage round (never the active round or one prepared after the lineage
head): the exports (``policy/sglang-hf``, ``policy/gguf``: ``publish-round`` regenerates them from the checkpoint) are
deleted with a ``.removed.json`` note; ``sft/learner-state*`` (a finished round's resume state), ``mix/*.pkl``,
``mix.*`` copies and ``filled/`` move to the HDD; ``policy/checkpoint.pt`` and ``sft/factors`` move only when the round's
policy is unprotected (rollback needs the last measured policy's checkpoint, the next merges the head's factors).
``mix/rows.jsonl.zst``, ``trained-units.json``, ``build.json`` and every small file stay.

Each item is skipped while any process (cwd, exe, fds, maps, command line, environment) or an active or
timer-triggered unit names it, or a ``.keep-on-nvme`` marker covers it. A move copies (hashing), syncs, re-reads the
copy from disk and compares every file's size and SHA-256, checks the source did not change and that no process
holds it, then swaps in the symlink; it never runs the HDD below ``--hdd-floor-gb``. Every action is one JSON line
(``~/.local/state/natlang/ei-round-tier.jsonl``).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'training' / 'neuralese'), str(REPO)]
from natlang_neuralese.common.paths import resolve  # noqa: E402
from natlang_neuralese.train.checkpoint_offload import replace_with_link, verified_copy  # noqa: E402
from scripts.prune_stale_checkouts import process_references, references, unit_commands  # noqa: E402

GIB = 2**30
OPS = Path(os.environ.get('NATLANG_OPS_ROOT', Path.home() / 'data' / 'lfm3-rl-dgx-20261029' / 'ops'))
DEST = resolve('data_hdd', 'dgx-moved-ei')
LOG = Path.home() / '.local' / 'state' / 'natlang' / 'ei-round-tier.jsonl'
MARKER = '.keep-on-nvme'
COLLECTION = re.compile(r'^v(\d+)-([0-9a-f]{12})(?:-.*)?$')
ROUND = re.compile(r'^r(\d{4,})$')


def log(**fields):
    line = json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), **fields})
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, 'a') as stream:
            stream.write(line + '\n')
    except OSError:
        pass


# ---------- what is protected ----------

def _shas(node) -> set[str]:
    """Every sha256 value under a state subtree (``sha256`` and ``*_sha256`` keys)."""
    found = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and (key == 'sha256' or key.endswith('_sha256')) and len(value) == 64:
                found.add(value)
            else:
                found |= _shas(value)
    elif isinstance(node, list):
        for value in node:
            found |= _shas(value)
    return found


def protected_policies(state: dict, current_policy: dict | None) -> set[str]:
    lineage = state.get('lineage') or []
    keep = {e['sha256'] for e in lineage[-2:] if e.get('sha256')}
    for key in ('active_round', 'measuring', 'reference_policy', 'start_policy'):
        keep |= _shas(state.get(key))
    verdicts = state.get('applied_verdicts') or []
    if verdicts:
        keep |= _shas(verdicts[-1])
    if current_policy:
        keep |= _shas(current_policy)
    return keep


def plan(driver: Path, state: dict, current_policy: dict | None) -> list[dict]:
    """The actions for one driver directory, oldest first: {'op': 'delete'|'move', 'path', 'why'}."""
    keep = protected_policies(state, current_policy)
    lineage = state.get('lineage') or []
    # round builds read only the current and previous policy's collections (and the active round's base); the
    # reference and start policies are what a rollback returns to, whose collection then resumes (2026-11-24 19:12:
    # aei1 rolled back to r1, whose collections were on the HDD, and its live collectors wrote there)
    keep12 = {e['sha256'][:12] for e in lineage[-2:] if e.get('sha256')} | \
        {s[:12] for key in ('active_round', 'reference_policy', 'start_policy') for s in _shas(state.get(key))}
    by_round = {e.get('round_id'): e for e in lineage if e.get('round_id')}
    last_two = {e.get('round_id') for e in lineage[-2:]}
    active = (state.get('active_round') or {}).get('round_id')
    head = max((int(r[1:]) for r in by_round), default=0)
    actions = []
    collect = driver / 'collect'
    if collect.is_dir():
        for d in sorted(collect.iterdir(), key=lambda p: p.name):
            m = COLLECTION.match(d.name)
            if not m or not d.is_dir() or m.group(2) in keep12:
                continue
            actions.append({'op': 'trim', 'path': str(d.resolve() if d.is_symlink() else d),
                            'why': f'collection of unprotected policy {m.group(2)}: reproducible and diagnostic files'})
            if d.is_symlink():          # moved earlier: only the trim applies
                continue
            export = d / 'actor' / 'sglang-hf'
            if export.is_dir() and not export.is_symlink():
                actions.append({'op': 'delete', 'path': str(export), 'why': f'collection of unprotected policy '
                                f'{m.group(2)}: actor export, rebuilt from the checkpoint when served'})
            actions.append({'op': 'move', 'path': str(d), 'why': f'collection of unprotected policy {m.group(2)}'})
    for r in sorted((p for p in driver.iterdir() if ROUND.match(p.name) and p.is_dir() and not p.is_symlink()),
                    key=lambda p: p.name):
        if r.name in last_two or r.name == active or int(r.name[1:]) >= head:
            continue
        policy = by_round.get(r.name)
        policy_kept = policy is not None and policy.get('sha256') in keep
        for name in ('sglang-hf', 'gguf'):
            export = r / 'policy' / name
            if export.is_dir() and not export.is_symlink():
                actions.append({'op': 'delete', 'path': str(export), 'why': f'{r.name}: {name} export, regenerated '
                                'from policy/checkpoint.pt by publish-round'})
        moves = [*sorted((r / 'sft').glob('learner-state*')), *sorted((r / 'mix').glob('*.pkl')),
                 *sorted(p for p in r.glob('mix.*') if p.is_dir()), r / 'filled']
        if not policy_kept:
            moves += [r / 'policy' / 'checkpoint.pt', r / 'sft' / 'factors']
        for path in moves:
            if path.exists() and not path.is_symlink():
                actions.append({'op': 'move', 'path': str(path),
                                'why': f'{r.name}: older than the previous round' +
                                       ('' if policy_kept else ' (policy unprotected)')})
    return actions


# ---------- moving ----------

def _drop_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _sha(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_inventory(root: Path) -> dict[str, tuple]:
    """rel -> (kind, size, mtime_ns, link target) for everything under ROOT (symlinks not followed)."""
    found = {}
    for directory, children, files in os.walk(root):
        for name in children + files:
            path = os.path.join(directory, name)
            st = os.lstat(path)
            rel = os.path.relpath(path, root)
            if os.path.islink(path):
                found[rel] = ('link', 0, 0, os.readlink(path))
            elif os.path.isdir(path):
                found[rel] = ('dir', 0, 0, None)
            else:
                found[rel] = ('file', st.st_size, st.st_mtime_ns, None)
    return found


FLUSH_BYTES = 64 << 20  # written bytes between a data sync and dropping them from the cache


def _copy_bounded(src: Path, out: Path) -> str:
    """SRC to OUT, hashing SRC; every FLUSH_BYTES written are synced and dropped from the page cache, so dirty pages
    bound for the slow HDD never pile up in the unit's memory cgroup (the sweep was OOM-killed at 1 GB)."""
    digest = hashlib.sha256()
    with open(src, 'rb') as reader, open(out, 'wb') as writer:
        pending = 0
        for block in iter(lambda: reader.read(8 << 20), b''):
            digest.update(block)
            writer.write(block)
            pending += len(block)
            if pending >= FLUSH_BYTES:
                writer.flush()
                os.fdatasync(writer.fileno())
                os.posix_fadvise(writer.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                pending = 0
        writer.flush()
        os.fdatasync(writer.fileno())
        os.posix_fadvise(writer.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    shutil.copystat(src, out)
    return digest.hexdigest()


def copy_tree_verified(src: Path, dest: Path) -> int:
    """SRC's tree to DEST (which must not exist): hashing while copying, then every file re-read from disk and
    compared. An interrupted copy resumes: DEST.partial is kept, and a file there whose size and mtime equal the
    source's is reused (its digest taken from the source; the re-read below still checks it). Returns the bytes
    copied; raises on a mismatch (dropping that file) or a changed source (dropping DEST.partial)."""
    before = tree_inventory(src)
    partial = dest.with_name(dest.name + '.partial')
    partial.mkdir(parents=True, exist_ok=True)
    digests, total = {}, 0
    for rel, (kind, size, mtime, target) in sorted(before.items()):
        out = partial / rel
        if kind == 'dir':
            out.mkdir(exist_ok=True)
        elif kind == 'link':
            if out.is_symlink() and os.readlink(out) == target:
                continue
            if out.is_symlink() or out.exists():
                out.unlink()
            os.symlink(target, out)
        else:
            try:
                st = out.lstat()
                reuse = st.st_size == size and st.st_mtime_ns == mtime and not out.is_symlink()
            except FileNotFoundError:
                reuse = False
            digests[rel] = _sha(src / rel) if reuse else _copy_bounded(src / rel, out)
            total += size
    os.sync()
    for rel, digest in digests.items():
        _drop_cache(partial / rel)
        if (partial / rel).stat().st_size != before[rel][1] or _sha(partial / rel) != digest:
            (partial / rel).unlink(missing_ok=True)
            raise RuntimeError(f'copy of {src / rel} does not verify')
    if tree_inventory(src) != before:
        shutil.rmtree(partial, ignore_errors=True)
        raise RuntimeError(f'{src} changed during the copy')
    os.replace(partial, dest)
    return total


def holders(path: Path) -> list[str]:
    root = str(path)
    held = []
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            found = [os.readlink(f'/proc/{pid}/cwd')]
            found += [os.readlink(f'/proc/{pid}/fd/{fd}') for fd in os.listdir(f'/proc/{pid}/fd')]
        except OSError:
            continue
        if any(f == root or f.startswith(root + '/') for f in found):
            held.append(pid)
    return held


def size_of(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(v[1] for v in tree_inventory(path).values())


def move(path: Path, dest: Path) -> int:
    """PATH (file or directory) to DEST on the HDD, verified, with PATH a symlink to it afterwards."""
    if dest.exists() or dest.is_symlink():
        raise RuntimeError(f'{dest} already exists')
    if path.is_file():
        record = verified_copy(path, dest)
        if holders(path):
            os.unlink(dest)
            raise RuntimeError(f'{path} is held open')
        replace_with_link(path, dest, expect=record['source_identity'])
        return record['bytes']
    copied = copy_tree_verified(path, dest)
    if holders(path):
        shutil.rmtree(dest)
        raise RuntimeError(f'{path} is held open')
    aside = path.with_name(path.name + '.ei-moving')
    os.rename(path, aside)
    try:
        os.symlink(dest, path)
    except OSError:
        os.rename(aside, path)
        raise
    shutil.rmtree(aside)
    return copied


def delete(path: Path) -> int:
    size = size_of(path)
    note = path.with_name(path.name + '.removed.json')
    note.write_text(json.dumps({'removed': str(path), 'bytes': size, 'at': time.time(),
                                'why': 'regenerable export (scripts/ei_round_tier.py)'}) + '\n')
    shutil.rmtree(path)
    return size


# ---------- collection trim ----------
# What a closed collection keeps: the episodes (results.jsonl, rollouts.jsonl, rewards, journals, partial episodes and
# their evidence, error records), memetic evaluations, task and rate files, collect.log. What goes once its policy has
# left the last two lineage entries (reproducible or diagnostic only; readers checked 2026-11-24, DECISIONS):
TRIM_KINDS = {
    # byte copy of the sample's results.jsonl; the run-time job accounting the scorer checks lives in the journal's
    # finish record (task_rollouts output_accounting), and exp1_collect already drops this copy when an episode ends
    'duplicate-result': 'groups/*/row-*/*/jobs/*.result.json',
    'trace': 'groups/*/row-*/*/jobs/*.trace.jsonl',          # ts-host execution traces (audit scripts only)
    'log': ('groups/*/workers/*.log', 'groups/*/row-*/*/collector.log', 'actor.log'),
    # atomic-write leftovers of killed writers (run_bonsai_queue removes these too once the writer is gone)
    'tmp': 'groups/*/row-*/*/jobs/*.tmp-*',
    # the actor's native-state cache: SFT trains imports as pass-through from the recorded write (sft_sources
    # whole-graph rows), the GRPO learner reads only its own run's store, and grpo.collect_blocks drops trained ones
    'native-state-cache': 'actor/blocks/*/nzs1_*.safetensors',
}
TMP_NAME = re.compile(r'\.tmp-\d+-[0-9a-f-]{36}$')
QUIET_SECONDS = 3600


def _same_bytes(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, 'rb') as x, open(b, 'rb') as y:
        while True:
            u, v = x.read(8 << 20), y.read(8 << 20)
            if u != v:
                return False
            if not u:
                return True


def trim_plan(collection: Path) -> list[tuple[Path, str, int]]:
    """(path, kind, bytes) of the files a closed collection can lose; regular files only (never a link or a Drive
    stub). Any doubt keeps the file: a result copy is listed only when its bytes equal results.jsonl's."""
    found = []
    for kind, patterns in TRIM_KINDS.items():
        for pattern in (patterns if isinstance(patterns, tuple) else (patterns,)):
            for path in sorted(collection.glob(pattern)):
                if path.is_symlink() or not path.is_file():
                    continue
                if kind == 'tmp' and not TMP_NAME.search(path.name):
                    continue
                if kind == 'duplicate-result':
                    results = path.parent.parent / 'results.jsonl'
                    try:
                        if results.is_symlink() or not results.is_file() or not _same_bytes(path, results):
                            continue
                    except OSError:
                        continue
                found.append((path, kind, path.stat().st_size))
    return found


def trim(collection: Path, apply: bool) -> dict:
    """Trim one closed collection. Refused (nothing removed) while its collect.log was written in the last hour. The
    list goes to COLLECTION/trimmed.json (appended per pass) before any file is removed. Returns bytes per kind."""
    log_file = collection / 'collect.log'
    if log_file.exists() and time.time() - log_file.stat().st_mtime < QUIET_SECONDS:
        raise RuntimeError('collection written in the last hour: still open')
    if not (collection / 'groups').is_dir():
        raise RuntimeError('not a collection (no groups/)')
    planned = trim_plan(collection)
    by_kind = collections.Counter()
    for _, kind, size in planned:
        by_kind[kind] += size
    if not apply or not planned:
        return dict(by_kind)
    record = collection / 'trimmed.json'
    previous = json.loads(record.read_text()) if record.exists() else {'schema': 'natlang.ei-collection-trim/1',
                                                                          'passes': []}
    previous['passes'].append({'at': time.time(), 'by': 'scripts/ei_round_tier.py', 'kinds': dict(by_kind),
                               'files': [{'path': str(p.relative_to(collection)), 'kind': k, 'bytes': b}
                                         for p, k, b in planned]})
    tmp = record.with_name(record.name + '.partial')
    tmp.write_text(json.dumps(previous) + '\n')
    os.replace(tmp, record)
    removed = collections.Counter()
    for path, kind, size in planned:
        try:
            path.unlink()
            removed[kind] += size
        except FileNotFoundError:
            pass
    return dict(removed)


# ---------- the pass ----------

def keep_marker(path: Path, root: Path) -> bool:
    probe = path
    while True:
        if (probe / MARKER).exists():
            return True
        if probe == root or probe.parent == probe:
            return False
        probe = probe.parent


def run(args) -> dict:
    deadline = time.time() + args.max_minutes * 60
    current = None
    try:
        current = json.loads((OPS / 'current-policy.json').read_text())
    except (OSError, ValueError):
        pass
    procs, units = process_references(), unit_commands()
    summary = {'deleted_bytes': 0, 'moved_bytes': 0, 'actions': 0, 'skipped': 0, 'errors': 0, 'applied': args.apply}
    for state_path in sorted(OPS.glob('ei/*/phase-state.json')):
        driver = state_path.parent
        if driver.is_symlink():
            continue
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError) as error:
            log(event='state unreadable: driver skipped', driver=str(driver), error=repr(error)[:200])
            continue
        for action in plan(driver, state, current):
            if time.time() > deadline:
                log(event='time budget reached', driver=str(driver))
                return summary
            path = Path(action['path'])
            why = references(str(path), procs, units, [])
            if why or keep_marker(path, driver):
                summary['skipped'] += 1
                log(event='kept', path=str(path), reason=(why or ['marker .keep-on-nvme'])[:3])
                continue
            if action['op'] == 'trim':
                try:
                    kinds = trim(path, args.apply)
                except Exception as error:      # open, unreadable or odd: the whole collection stays
                    summary['skipped'] += 1
                    log(event='trim skipped', path=str(path), reason=repr(error)[:200])
                    continue
                if kinds:
                    freed = sum(kinds.values())
                    summary['trimmed_bytes'] = summary.get('trimmed_bytes', 0) + freed
                    summary['actions'] += 1
                    log(event='trimmed' if args.apply else 'would trim', path=str(path), gb=round(freed / GIB, 2),
                        kinds={k: round(v / GIB, 2) for k, v in kinds.items()}, why=action['why'])
                continue
            size = size_of(path)
            if not args.apply:
                log(event='would ' + action['op'], path=str(path), gb=round(size / GIB, 2), why=action['why'])
                summary[('deleted' if action['op'] == 'delete' else 'moved') + '_bytes'] += size
                summary['actions'] += 1
                continue
            try:
                if action['op'] == 'delete':
                    summary['deleted_bytes'] += delete(path)
                else:
                    dest = DEST / driver.name / path.relative_to(driver)
                    probe = dest.parent
                    while not probe.exists():
                        probe = probe.parent
                    free = shutil.disk_usage(probe).free
                    if free - size < args.hdd_floor_gb * GIB:
                        log(event='HDD floor: move refused', path=str(path), gb=round(size / GIB, 2),
                            hdd_free_gb=round(free / GIB, 1))
                        summary['errors'] += 1
                        continue
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    summary['moved_bytes'] += move(path, dest)
                summary['actions'] += 1
                log(event=action['op'] + 'd' if action['op'] == 'delete' else 'moved', path=str(path),
                    gb=round(size / GIB, 2), why=action['why'])
            except Exception as error:      # one item failing keeps its path and the pass going
                summary['errors'] += 1
                log(event='error: kept', path=str(path), error=repr(error)[:300])
    return summary


def _own_tree() -> set[int]:
    """This process and its ancestors."""
    pids, pid = set(), os.getpid()
    while pid > 1 and pid not in pids:
        pids.add(pid)
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def _own_unit() -> str | None:
    try:
        for part in reversed(Path('/proc/self/cgroup').read_text().strip().split('/')):
            if part.endswith('.service'):
                return part
    except OSError:
        pass
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--max-minutes', type=float, default=20.0)
    parser.add_argument('--hdd-floor-gb', type=float, default=100.0)
    parser.add_argument('--trim', action='append', default=[], metavar='COLLECTION',
                        help='trim only these closed collections (e.g. an earlier cycle\'s, outside any driver state); '
                             'no other action runs')
    args = parser.parse_args(argv)
    if args.trim:
        # the paths are on this pass's own command line (and its ledger unit's): those never count as readers
        own_pids, own_unit = _own_tree(), _own_unit()
        procs = [r for r in process_references() if int(r[0]) not in own_pids]
        units = [u for u in unit_commands() if u[0] != own_unit]
        for path in map(Path, args.trim):
            why = references(str(path), procs, units, [])
            if why or keep_marker(path, path):
                log(event='kept', path=str(path), reason=(why or ['marker .keep-on-nvme'])[:3])
                continue
            try:
                kinds = trim(path, args.apply)
                log(event='trimmed' if args.apply else 'would trim', path=str(path),
                    gb=round(sum(kinds.values()) / GIB, 2), kinds={k: round(v / GIB, 2) for k, v in kinds.items()})
            except Exception as error:
                log(event='trim skipped', path=str(path), reason=repr(error)[:200])
        return 0
    summary = run(args)
    root_free = shutil.disk_usage('/').free
    log(event='summary', **summary, deleted_gb=round(summary['deleted_bytes'] / GIB, 1),
        trimmed_gb=round(summary.get('trimmed_bytes', 0) / GIB, 1),
        moved_gb=round(summary['moved_bytes'] / GIB, 1), root_free_gb=round(root_free / GIB, 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
