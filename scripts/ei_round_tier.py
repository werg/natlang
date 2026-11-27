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

Retention (``drop_plan``, default; ``--no-drop`` opts out; DECISIONS 2026-11-25 "EI retention") runs first and deletes
what no consumer reads: closed collections' actor exports and native-state blocks (every policy but the lineage head),
trained rounds' learner states, chains pickles (mix with trained-units.json, filled, own Neuralese graph) and
superseded exports, and the checkpoint of a policy that left the lineage unprotected; teacher campaigns' quiet episodes
are compacted, and so are the scored groups of every collection, running ones included (``compact_groups``: a quiet,
unheld sample's raw job files and its ``results.jsonl``, ``episode_files.compact_sample`` and ``compact_results``,
then a quiet group's ``rollouts.jsonl``, ``compact_rollouts``; ``--no-group-compaction`` or
``NATLANG_RESULTS_COMPACT=0`` / ``NATLANG_ROLLOUTS_COMPACT=0`` / ``NATLANG_EPISODE_COMPACT=0`` opt out; DECISIONS
2026-11-25 "open collections compacted", 2026-11-26 "rollouts compacted"). The async driver runs the same deletions itself at each publication (``retain``).

Each item is skipped while any process (cwd, exe, fds, maps, command line, environment) or an active, failed or
timer-triggered unit names it, or a ``.keep-on-nvme`` marker covers it. A deletion is also skipped while a configured
consumer names it (``prune_stale_checkouts.configured_consumers``: the current policy pointer, the watchdog roster,
``ops/consumers.d`` registrations and the config files they list, the data registry, config and script files named in
unit command lines), so a periodic consumer that is between runs keeps its inputs; ``--no-consumer-configs`` opts out
(DECISIONS 2026-11-25 "configured consumers"). A move copies (hashing), syncs, re-reads the
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
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'training' / 'neuralese'), str(REPO)]
from natlang_neuralese.common.paths import resolve  # noqa: E402
from natlang_neuralese.common.policy_identity import entry_shas  # noqa: E402
from natlang_neuralese.rl import episode_files  # noqa: E402
from natlang_neuralese.train.checkpoint_offload import replace_with_link, verified_copy  # noqa: E402
from scripts.prune_stale_checkouts import (configured_consumers, process_references, references,  # noqa: E402
                                           unit_commands)

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
    # a lineage policy under every file sha of its weights (a rebuilt checkpoint's earlier sha: ``policy_identity``)
    keep = {s for e in lineage[-2:] for s in entry_shas(e)}
    for key in ('active_round', 'measuring', 'reference_policy', 'start_policy'):
        keep |= _shas(state.get(key))
    verdicts = state.get('applied_verdicts') or []
    if verdicts:        # PhaseController records each applied verdict as its policy's sha256 string
        last = verdicts[-1]
        keep |= {last} if isinstance(last, str) and len(last) == 64 else _shas(last)
    if current_policy:
        keep |= _shas(current_policy)
    return keep


def _round_sha(r: Path) -> str | None:
    try:
        return json.loads((r / 'policy' / 'published.json').read_text()).get('checkpoint_sha256')
    except (OSError, ValueError, AttributeError):
        return None


def round_trained(r: Path, state: dict) -> bool:
    """Whether round ``r`` trained: the driver recorded it (``trained_rounds``), or its factors or publication exist."""
    return (r.name in set(state.get('trained_rounds') or []) or (r / 'policy' / 'published.json').exists()
            or (r / 'sft' / 'factors' / 'policy_manifest.json').exists())


def round_pending(r: Path, state: dict) -> bool:
    """A round that is still to train, whatever ``active_round`` says: not discarded, not trained (``round_trained``),
    and either prepared (``stages/prepared.done``) or prepared from the lineage head (round.json ``base_policy_sha`` or
    ``trained_from``). Its mix, fill and Neuralese-graph inputs and learner state are protected. 2026-11-25 14:23: with
    ``active_round`` on a GRPO round that was later dropped, r0011 (prepared, never trained) counted as older than the
    active round and lost its chains and inputs."""
    if (r / 'discarded.json').exists() or round_trained(r, state):
        return False
    if (r / 'stages' / 'prepared.done').exists():
        return True
    lineage = state.get('lineage') or []
    head = (lineage[-1].get('sha256') or '') if lineage else ''
    try:
        record = json.loads((r / 'round.json').read_text())
    except (OSError, ValueError):
        return False
    return bool(head) and head in (record.get('base_policy_sha'), record.get('trained_from'))


def drop_plan(driver: Path, state: dict, current_policy: dict | None) -> list[dict]:
    """Retention (DECISIONS 2026-11-25 "EI retention"): what no consumer reads once a round is trained or a collection
    is closed is deleted, on the NVMe or through an earlier move's link on the HDD. Every action is a 'delete'; an
    action's 'guard' paths must be unreferenced by processes and units too, and its 'alias' paths (the same item
    through the driver's link) by configured consumers.

    - Collections of every policy but the lineage head: ``actor/sglang-hf`` (``serve_native_checkpoint`` re-exports it
      from the checkpoint when the policy is served again) and ``actor/blocks`` (an episode imports only states its own
      calls stored; a resumed partial never continues past a Neuralese write, ``episode_files.partial_continuable``;
      the SFT build trains imports as pass-through). Guarded by ``actor`` (the running actor names ``--store-dir``).
    - Rounds that trained (``trained_rounds``, ``discarded.json``, or older than the active round), never the active
      one nor one still to train (``round_pending``: prepared and not trained, or prepared from the lineage head): ``sft/learner-state*`` (the round's resume state: read by ``--resume``, ``recap_for_dgx``, ``sft_handback``
      and ``rebuild_round`` only before publication); ``*.pkl`` of a mix directory whose ``trained-units.json`` exists
      (``--prefer-unseen`` reads rows.jsonl.zst and trained-units.json, ``sft_sources.trained_units``) or of a
      ``mix.pre-rebuild-*`` copy; ``filled/chains.pkl`` and ``neuralese-graph-own/chains.pkl`` (inputs of that
      round's build only; the fill store stays). Their small files stay.
    - Exports of a trained round whose policy is not the lineage head: ``policy/sglang-hf`` and ``policy/gguf``
      (``publish-round`` writes them; ``rebind_remote`` reads the head's, lfm-latest and Pop serve from the checkpoint).
    - ``policy/checkpoint.pt`` of a trained round whose policy left the lineage (a rollback or a discard) and is
      protected by nothing (``protected_policies``): ``sft/factors`` and ``published.json`` (base + factors) stay.
    """
    keep = protected_policies(state, current_policy)
    lineage = state.get('lineage') or []
    head = (lineage[-1].get('sha256') or '') if lineage else ''
    on_lineage = {s for e in lineage for s in entry_shas(e)}
    trained = set(state.get('trained_rounds') or [])
    active = (state.get('active_round') or {}).get('round_id')
    active_n = int(active[1:]) if active and ROUND.match(active) else None
    actions = []
    collect = driver / 'collect'
    if collect.is_dir():
        for d in sorted(collect.iterdir(), key=lambda p: p.name):
            m = COLLECTION.match(d.name)
            if not m or not d.is_dir() or not head or head.startswith(m.group(2)):
                continue
            base = d.resolve() if d.is_symlink() else d
            for name in ('sglang-hf', 'blocks'):
                path = base / 'actor' / name
                if path.is_dir() and not path.is_symlink():
                    actions.append({'op': 'delete', 'path': str(path), 'guard': [str(d / 'actor'), str(base / 'actor')],
                                    'alias': [str(d / 'actor' / name)],
                                    'why': f'closed collection of {m.group(2)} (not the lineage head): actor {name}'})
    for r in sorted((p for p in driver.iterdir() if ROUND.match(p.name) and p.is_dir()), key=lambda p: p.name):
        if r.name == active or round_pending(r, state):
            continue
        discarded = (r / 'discarded.json').exists()
        if not (r.name in trained or discarded or (active_n is not None and int(r.name[1:]) < active_n)):
            continue
        paths = [*(r / 'sft').glob('learner-state*'), r / 'filled' / 'chains.pkl',
                 r / 'neuralese-graph-own' / 'chains.pkl']
        for mix in [r / 'mix', *sorted(r.glob('mix.*'))]:
            if mix.is_dir() and ((mix / 'trained-units.json').exists() or mix.name.startswith('mix.pre-rebuild-')):
                paths += sorted(mix.glob('*.pkl'))
        sha = _round_sha(r)
        if (r.name in trained or discarded) and sha and sha != head:
            paths += [r / 'policy' / 'sglang-hf', r / 'policy' / 'gguf']
            if sha not in on_lineage and sha not in keep:
                paths.append(r / 'policy' / 'checkpoint.pt')
        for path in paths:
            if path.name.endswith(('.removed.json', '.partial')) or not (path.exists() or path.is_symlink()):
                continue
            if path.is_symlink() and not _under_dest(path):
                continue        # a link to something not ours (a hand-made pointer): left alone
            actions.append({'op': 'delete', 'path': str(path),
                            'why': f'{r.name} trained' + (' (policy off the lineage)' if path.name == 'checkpoint.pt'
                                                          else '') + ': no consumer reads ' + path.name})
    return actions


def _under_dest(link: Path) -> bool:
    try:
        target = link.resolve()
    except OSError:
        return False
    return str(target).startswith(str(DEST) + '/') or str(target).startswith(str(link.parent.resolve()) + '/')


def plan(driver: Path, state: dict, current_policy: dict | None, drop: bool = True,
         group_compaction: bool = True) -> list[dict]:
    """The actions for one driver directory, oldest first: {'op': 'delete'|'move'|'trim'|'compact', 'path', 'why'}.
    ``drop`` (default; ``--no-drop`` opts out) puts the retention deletions (``drop_plan``) and the teacher campaigns'
    episode compaction first; the moves then tier what is left."""
    actions = drop_plan(driver, state, current_policy) if drop else []
    if drop and group_compaction and (driver / 'collect').is_dir():
        # the NVMe's collections first (newest first: the running ones grow); moved ones (on the HDD) after
        newest = sorted((driver / 'collect').iterdir(), key=lambda p: p.name, reverse=True)
        for d in sorted(newest, key=lambda p: p.is_symlink()):
            if COLLECTION.match(d.name) and d.is_dir() and (d / 'groups').is_dir():
                actions.append({'op': 'compact-groups', 'path': str(d),
                                'why': 'scored groups: quiet episodes compacted (episode_files.compact_results)'})
    if drop:
        for t in sorted(driver.glob('teacher-r*')):
            if t.is_dir() and not t.is_symlink() and (t / 'episodes').is_dir() and not (t / COMPACTED).exists():
                actions.append({'op': 'compact', 'path': str(t),
                                'why': 'teacher campaign: quiet episodes compacted (episode_files.compact_sample)'})
    dropped = {a['path'] for a in actions}
    return actions + [a for a in _tier_plan(driver, state, current_policy) if a['path'] not in dropped
                      and not any(a['path'].startswith(p + '/') for p in dropped)]


def _tier_plan(driver: Path, state: dict, current_policy: dict | None) -> list[dict]:
    keep = protected_policies(state, current_policy)
    lineage = state.get('lineage') or []
    # round builds read only the current and previous policy's collections (and the active round's base); the
    # reference and start policies are what a rollback returns to, whose collection then resumes (2026-11-24 19:12:
    # aei1 rolled back to r1, whose collections were on the HDD, and its live collectors wrote there)
    keep12 = {s[:12] for e in lineage[-2:] for s in entry_shas(e)} | \
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
        if r.name in last_two or r.name == active or int(r.name[1:]) >= head or round_pending(r, state):
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


def delete(path: Path, why: str = 'regenerable export (scripts/ei_round_tier.py)') -> int:
    """PATH (file or directory; a link an earlier move left: its HDD target too) removed, with PATH.removed.json
    noting what and why. Returns the bytes freed."""
    target = path.resolve() if path.is_symlink() else path
    size = size_of(target) if target.exists() else 0
    note = path.with_name(path.name + '.removed.json')
    note.write_text(json.dumps({'removed': str(path), **({'target': str(target)} if target != path else {}),
                                'bytes': size, 'at': time.time(), 'why': why}) + '\n')
    if target != path:
        path.unlink()
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()
    return size


COMPACTED = '.episodes-compacted'


def compact_campaign(campaign: Path, quiet_seconds: int = 3600) -> dict:
    """``episode_files.compact_sample`` over a teacher campaign's quiet samples (unchanged ``quiet_seconds``, none
    open). Once every sample was quiet, ``COMPACTED`` marks the campaign done (later passes skip it). Returns the
    compaction counters with 'freed' and 'skipped' (samples still live)."""
    held = episode_files.open_paths(campaign / 'episodes')
    total, skipped, now = collections.Counter(), 0, time.time()
    for sample in episode_files.samples_of(campaign / 'episodes'):
        if episode_files.quiet_sample(sample, now, quiet_seconds, held):
            total.update(episode_files.compact_sample(sample))
        else:
            skipped += 1
    if not skipped:
        (campaign / COMPACTED).write_text(json.dumps({'at': time.time(), **dict(total)}) + '\n')
    return {**dict(total), 'freed': episode_files.freed(total), 'skipped': skipped}


# ---------- scored groups of any collection ----------
GROUP_COMPACTED = '.episodes-compacted'
GROUP_QUIET_SECONDS = 1800


def remote_puller_blocks(collection: Path, procs, units, needle_constant='COMPACTED_RESULTS_EXCLUDED') -> str | None:
    """Why a collection pulled from a remote VM (``remote_collect.py run --work COLLECTION``) is not compacted: a live
    or failed (restartable) puller whose ``remote_collect.py`` predates ``needle_constant`` would pull every compacted
    file raw again (``COMPACTED_RESULTS_EXCLUDED``: a sample's results.jsonl; ``COMPACTED_ROLLOUTS_EXCLUDED``: a
    group's rollouts.jsonl). None when no such puller exists."""
    needle = f'--work {collection} '
    texts = [t for _, kind, t in procs if kind == 'cmdline'] + [t for _, t in units]
    for text in texts:
        if 'remote_collect.py' not in text or needle not in text + ' ':
            continue
        script = next((w for w in text.split() if w.endswith('remote_collect.py')), None)
        try:
            if script and needle_constant in Path(script).read_text():
                continue
        except OSError:
            pass
        return f'remote puller without {needle_constant}: {script}'
    return None


def kept_back(action: dict, why: list, marker: bool) -> bool:
    """An action the pass skips: a ``.keep-on-nvme`` marker keeps anything; holders and references (WHY) keep a move
    or a drop, never a ``compact-groups``, which checks each sample and group itself."""
    return marker or bool(why and action['op'] != 'compact-groups')


GROUP_PASS_SECONDS = 300      # one collection's share of a pass: the moves after it keep their time


def group_quiet(group: Path, now: float, quiet_seconds: float, held: set) -> bool:
    """A group's own files (rollouts, rewards, score, queue, merged and worker journals) unchanged ``quiet_seconds``
    and none held open: no scoring writes it any more."""
    for path in [*group.glob('*'), *group.glob('workers/*')]:
        if path.name.startswith(('row-', '.')) or path.is_dir():
            continue        # the samples (checked one by one: episode_files.quiet_sample), the passes' own markers
        try:
            if now - path.lstat().st_mtime < quiet_seconds:
                return False
        except FileNotFoundError:
            continue
        if str(path.resolve()) in held:
            return False
    return True


def compact_groups(collection: Path, quiet_seconds: int = GROUP_QUIET_SECONDS, deadline: float | None = None,
                   max_seconds: float = GROUP_PASS_SECONDS, rollouts: bool = True) -> dict:
    """The scored groups (``score.json``) of one collection, open or closed: each top-level sample (``row-*/*``) that
    is quiet ``quiet_seconds`` and that no process holds open loses its raw job files and its ``results.jsonl`` to
    ``.zst`` (``episode_files.compact_sample``, ``compact_results``); once every sample is compacted and the group's
    own files are quiet too, its ``rollouts.jsonl`` follows (``episode_files.compact_rollouts``; ``rollouts`` False or
    ``NATLANG_ROLLOUTS_COMPACT=0``: not). A scored group is reused as scored and never scored again
    (``grpo.Collector.reuse_scored``; a JOLT+ rescore drops the stale ``.zst``); its readers resolve the compacted
    files (``episode_files.present``, ``group_rollouts``). A group whose samples were all compacted gets
    ``GROUP_COMPACTED``, one whose rollouts were ``episode_files.ROLLOUTS_MARKER`` (later passes skip what is done).
    Each compacted rollouts file is recorded in the collection's data-registry entry with the raw file's digest.
    Returns counters with 'freed', 'groups' and 'skipped' (samples still live)."""
    total, groups, skipped, packed = collections.Counter(), 0, 0, []
    if not episode_files.enabled():
        return {'freed': 0, 'groups': 0, 'skipped': 0}
    rollouts = rollouts and episode_files.rollouts_enabled()
    held = episode_files.open_paths(collection / 'groups')
    stop = time.time() + max_seconds if deadline is None else min(deadline, time.time() + max_seconds)
    for group in sorted((collection / 'groups').glob('g*')):
        if time.time() > stop:
            break
        if group.is_symlink() or not group.is_dir() or not (group / 'score.json').is_file():
            continue
        episodes_done = (group / GROUP_COMPACTED).exists()
        rollouts_done = not rollouts or (group / episode_files.ROLLOUTS_MARKER).exists()
        if episodes_done and rollouts_done:
            continue
        now, live, stats = time.time(), 0, collections.Counter()
        if not episodes_done:
            for sample in sorted(p for p in group.glob('row-*/*') if p.is_dir() and not p.is_symlink()):
                if not episode_files.quiet_sample(sample, now, quiet_seconds, held):
                    live += 1
                    continue
                stats.update(episode_files.compact_sample(sample))
                stats.update(episode_files.compact_results(sample))
            if not live and not stats.get('errors') and episode_files.results_enabled():
                (group / GROUP_COMPACTED).write_text(json.dumps({'at': time.time(), **dict(stats)}) + '\n')
        if not rollouts_done and not live and not stats.get('errors') and group_quiet(group, now, quiet_seconds, held):
            more, done = episode_files.compact_rollouts(group)
            stats.update(more)
            if done:
                packed.append(done)
            if not more.get('errors'):
                (group / episode_files.ROLLOUTS_MARKER).write_text(json.dumps({'at': time.time(), **dict(more)}) + '\n')
        total.update(stats)
        skipped += live
        groups += 1
    if packed:
        record_packed(collection, packed)
    return {**dict(total), 'freed': episode_files.freed(total), 'groups': groups, 'skipped': skipped}


def record_packed(collection: Path, packed: list) -> None:
    """Compacted rollouts files into the collection's data-registry entry (``data_registry.record_files`` with the raw
    file each replaces), when it has one; bookkeeping never fails the pass."""
    try:
        from natlang_neuralese.common import data_registry as dr
        identity = dr.dataset_id(collection, 'grpo-rollouts')
        if identity not in dr.load().datasets:
            return
        base = Path(collection).resolve()
        dr.record_files(identity, [p for p, _, _ in packed], base, compacted_from=[
            {'path': str(Path(p).resolve().relative_to(base))[:-len(episode_files.SUFFIX)], 'sha256': sha,
             'bytes': size} for p, sha, size in packed])
    except Exception as error:
        log(event='data registry error', path=str(collection), error=repr(error)[:300])


# ---------- collection trim ----------
# What a closed collection keeps: the episodes (results.jsonl, rollouts.jsonl (raw or .zst), rewards, journals, partial episodes and
# their evidence, compacted to .zst by natlang_neuralese.rl.episode_files, error records), memetic evaluations, task and
# rate files, collect.log. What goes once its policy has left the last two lineage entries (reproducible or diagnostic
# only; readers checked 2026-11-24, DECISIONS):
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
    if not apply:
        raw = sum(p.stat().st_size for p in collection.glob('groups/**/jobs/*')
                  if p.is_file() and not p.is_symlink() and episode_files.RAW.search(p.name))
        return {**dict(by_kind), **({'episode-raw (compacts ~30x)': raw} if raw else {})}
    if not planned:
        compacted = episode_files.freed(episode_files.compact_group(collection / 'groups'))
        return {'episode-compaction': compacted} if compacted else {}
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
    compacted = episode_files.freed(episode_files.compact_group(collection / 'groups'))
    if compacted:
        removed['episode-compaction'] += compacted
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
    current = current_policy()
    procs, units = process_references(), unit_commands()
    configured = configured_consumers(units, ops=OPS) if getattr(args, 'consumer_configs', True) else []
    summary = new_summary(args.apply)
    for state_path in sorted(OPS.glob('ei/*/phase-state.json')):
        driver = state_path.parent
        if driver.is_symlink():
            continue
        try:
            state = json.loads(state_path.read_text())
        except (OSError, ValueError) as error:
            log(event='state unreadable: driver skipped', driver=str(driver), error=repr(error)[:200])
            continue
        actions = plan(driver, state, current, drop=getattr(args, 'drop', True),
                       group_compaction=getattr(args, 'group_compaction', True))
        if not apply_plan(driver, actions, args, procs, units, summary, deadline, configured):
            break
    return summary


def current_policy() -> dict | None:
    try:
        return json.loads((OPS / 'current-policy.json').read_text())
    except (OSError, ValueError):
        return None


def new_summary(apply: bool) -> dict:
    return {'deleted_bytes': 0, 'moved_bytes': 0, 'actions': 0, 'skipped': 0, 'errors': 0, 'applied': apply}


def retain(driver: Path, *, apply: bool = True, ops=('delete', 'compact', 'compact-groups'), max_minutes: float = 30.,
           consumer_configs: bool = True) -> dict:
    """The retention of one driver directory, run by the driver itself (``expert_iteration.py async-drive`` at each
    publication and when a collection closes; ``--no-retention`` opts out): ``plan``'s ``ops`` actions (deletions and
    the teacher and scored-group compaction by default; the HDD moves stay with the sweep) under the same reference
    checks."""
    state = json.loads((Path(driver) / 'phase-state.json').read_text())
    summary = new_summary(apply)
    actions = [a for a in plan(Path(driver), state, current_policy()) if a['op'] in ops]
    procs, units = process_references(), unit_commands()
    apply_plan(Path(driver), actions, SimpleNamespace(apply=apply, hdd_floor_gb=100.), procs, units, summary,
               time.time() + max_minutes * 60, configured_consumers(units, ops=OPS) if consumer_configs else [])
    return summary


def apply_plan(driver, actions, args, procs, units, summary, deadline, configured=()) -> bool:
    """Run ``actions`` of one driver (each skipped while referenced, its guards included, or under a keep marker; a
    deletion also while a ``configured`` consumer file names it or a guard); False once the time budget ran out."""
    for action in actions:
        if time.time() > deadline:
            log(event='time budget reached', driver=str(driver))
            return False
        path = Path(action['path'])
        # a compaction checks each sample itself (quiet, none open); a live campaign names the earlier ones on its
        # command line (--covered-by) without reading their job files, so only open files and working dirs count
        refs = ([r for r in procs if r[1] not in ('cmdline', 'environ')], []) if action['op'] == 'compact' else (procs, units)
        why = [w for p in [str(path), *action.get('guard', ())] for w in references(p, *refs, [])]
        if action['op'] == 'delete':        # a consumer's config names what it reads (path or its link alias)
            why += [w for p in [str(path), *action.get('alias', ())] for w in references(p, [], [], list(configured))]
        # a live collection always has holders (its collector's log, a fill's reader): they keep it from a move,
        # but compaction checks each sample and group itself (quiet, none open) and must not be gated by them. Gated
        # here, no group of a live collection was ever compacted (v0013: 0 of 436 groups, root -14 GB/h, 2026-11-27)
        if kept_back(action, why, keep_marker(path, driver)):
            summary['skipped'] += 1
            log(event='kept', path=str(path), reason=(why or ['marker .keep-on-nvme'])[:3])
            continue
        if action['op'] == 'compact-groups':
            # a running collector names and holds its collection: each sample is checked itself (quiet, none open)
            blocked = remote_puller_blocks(path, procs, units)
            if blocked or keep_marker(path, driver):
                summary['skipped'] += 1
                log(event='kept', path=str(path), reason=[blocked or 'marker .keep-on-nvme'])
                continue
            # a puller that would fetch a compacted rollouts.jsonl again leaves the group's rollouts raw
            rollouts_blocked = remote_puller_blocks(path, procs, units, 'COMPACTED_ROLLOUTS_EXCLUDED')
            if rollouts_blocked:
                log(event='rollouts kept raw', path=str(path), reason=[rollouts_blocked])
            if not args.apply:
                continue
            try:
                stats = compact_groups(path, deadline=deadline, rollouts=not rollouts_blocked)
            except Exception as error:
                summary['errors'] += 1
                log(event='error: kept', path=str(path), error=repr(error)[:300])
                continue
            if stats['freed']:
                summary['compacted_bytes'] = summary.get('compacted_bytes', 0) + stats['freed']
                summary['actions'] += 1
                log(event='compacted', path=str(path), gb=round(stats['freed'] / GIB, 2), groups=stats['groups'],
                    skipped=stats['skipped'], why=action['why'])
            continue
        if action['op'] == 'compact':
            if not args.apply:
                continue
            try:
                stats = compact_campaign(path)
            except Exception as error:
                summary['errors'] += 1
                log(event='error: kept', path=str(path), error=repr(error)[:300])
                continue
            if stats['freed']:
                summary['compacted_bytes'] = summary.get('compacted_bytes', 0) + stats['freed']
                summary['actions'] += 1
                log(event='compacted', path=str(path), gb=round(stats['freed'] / GIB, 2), skipped=stats['skipped'],
                    why=action['why'])
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
        target = path.resolve() if path.is_symlink() else path
        size = size_of(target) if target.exists() else 0
        if not args.apply:
            log(event='would ' + action['op'], path=str(path), gb=round(size / GIB, 2), why=action['why'])
            summary[('deleted' if action['op'] == 'delete' else 'moved') + '_bytes'] += size
            summary['actions'] += 1
            continue
        try:
            if action['op'] == 'delete':
                summary['deleted_bytes'] += delete(path, action['why'])
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
    return True


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
    parser.add_argument('--no-drop', dest='drop', action='store_false',
                        help='opt out of the retention deletions and teacher compaction (drop_plan): moves only')
    parser.add_argument('--no-group-compaction', dest='group_compaction', action='store_false',
                        help='opt out of compacting the scored groups of every collection (compact_groups)')
    parser.add_argument('--no-consumer-configs', dest='consumer_configs', action='store_false',
                        help='opt out of the configured consumers (ops/consumers.d registrations, data registry, '
                             'current policy, watchdog, unit config files) as references of a deletion')
    parser.add_argument('--trim', action='append', default=[], metavar='COLLECTION',
                        help='trim only these closed collections (e.g. an earlier cycle\'s, outside any driver state); '
                             'no other action runs')
    parser.add_argument('--retain', metavar='DRIVER', help='only the driver-side retention of DRIVER (retain: '
                        'deletions and teacher compaction, no moves); a dry run lists the paths unless --apply')
    args = parser.parse_args(argv)
    if args.retain:
        summary = retain(Path(args.retain), apply=args.apply, max_minutes=args.max_minutes,
                         consumer_configs=args.consumer_configs)
        log(event='retain summary', driver=args.retain, **summary,
            deleted_gb=round(summary['deleted_bytes'] / GIB, 2),
            compacted_gb=round(summary.get('compacted_bytes', 0) / GIB, 2), root_free_gb=round(shutil.disk_usage('/').free / GIB, 1))
        return 0
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
        compacted_gb=round(summary.get('compacted_bytes', 0) / GIB, 1),
        moved_gb=round(summary['moved_bytes'] / GIB, 1), root_free_gb=round(root_free / GIB, 1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
