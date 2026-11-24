#!/usr/bin/env python3
"""Standing storage rule for expert-iteration drivers: what rounds leave on the NVMe moves to the HDD as it goes cold.

An async driver (``ops/ei/<name>/phase-state.json``) adds about 15 GB per round on the NVMe: a collection per policy
(``collect/vNNNN-<sha12>*``: ``groups/`` and the actor's ``blocks/`` and ``sglang-hf/``) and a round directory
(``rNNNN``: ``policy/`` checkpoint and exports, ``sft/`` states, ``mix/``, ``filled/``). Run by ``storage_sweep.sh``
every 30 minutes; dry run unless ``--apply``.

Collections stay for the last two lineage policies and the active round's base (round builds read the current and
previous policy's collections). Checkpoints and factors stay for those and for the held measurement's candidate and
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
    # round builds read only the current and previous policy's collections (and the active round's base)
    keep12 = {e['sha256'][:12] for e in lineage[-2:] if e.get('sha256')} | \
        {s[:12] for s in _shas(state.get('active_round'))}
    by_round = {e.get('round_id'): e for e in lineage if e.get('round_id')}
    last_two = {e.get('round_id') for e in lineage[-2:]}
    active = (state.get('active_round') or {}).get('round_id')
    head = max((int(r[1:]) for r in by_round), default=0)
    actions = []
    collect = driver / 'collect'
    if collect.is_dir():
        for d in sorted(collect.iterdir(), key=lambda p: p.name):
            m = COLLECTION.match(d.name)
            if not m or d.is_symlink() or not d.is_dir() or m.group(2) in keep12:
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


def copy_tree_verified(src: Path, dest: Path) -> int:
    """SRC's tree to DEST (which must not exist): hashing while copying, one sync, then every file re-read from
    disk and compared. Returns the bytes copied; raises (removing DEST) on any mismatch or a changed source."""
    before = tree_inventory(src)
    partial = dest.with_name(dest.name + '.partial')
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    digests, total = {}, 0
    try:
        for rel, (kind, size, _mtime, target) in sorted(before.items()):
            out = partial / rel
            if kind == 'dir':
                out.mkdir(exist_ok=True)
            elif kind == 'link':
                os.symlink(target, out)
            else:
                digest = hashlib.sha256()
                with open(src / rel, 'rb') as reader, open(out, 'wb') as writer:
                    for block in iter(lambda: reader.read(8 << 20), b''):
                        digest.update(block)
                        writer.write(block)
                shutil.copystat(src / rel, out)
                digests[rel] = digest.hexdigest()
                total += size
        os.sync()
        for rel, digest in digests.items():
            _drop_cache(partial / rel)
            if (partial / rel).stat().st_size != before[rel][1] or _sha(partial / rel) != digest:
                raise RuntimeError(f'copy of {src / rel} does not verify')
        if tree_inventory(src) != before:
            raise RuntimeError(f'{src} changed during the copy')
        os.replace(partial, dest)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--max-minutes', type=float, default=20.0)
    parser.add_argument('--hdd-floor-gb', type=float, default=100.0)
    args = parser.parse_args(argv)
    summary = run(args)
    root_free = shutil.disk_usage('/').free
    log(event='summary', **summary, deleted_gb=round(summary['deleted_bytes'] / GIB, 1),
        moved_gb=round(summary['moved_bytes'] / GIB, 1), root_free_gb=round(root_free / GIB, 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
