#!/usr/bin/env python3
"""Storage retention on the DGX (plans/STORAGE_POLICY.md, training/storage-policy.json).

Every file under the scan roots is classified into a category (checkpoint, optimizer state, export, teacher dump,
corpus, artifact, dataset, log, cache, other) and, from 1 GiB up (``min_report_bytes``), into a tier:

  KEEP        registered corpora and artifacts (their manifests' files and the archived copies they point to),
              files named by run scripts, recipes, registries, plans, certificates and receipts, logs/receipts/labels,
              and the policy's keep paths. Never deleted.
  ACTIVE      in use now: open by a process, in a run some process works in, named by an active memory-ledger
              claim, under an archiver exclude, or in a run changed within ``finished_after_days``. Never deleted.
  PRUNE-AUTO  clearly regenerable: intermediate step checkpoints of finished runs beyond the latest and the best,
              checkpoints of aborted/stopped runs after ``aborted_after_days``, leftover partial writes, dangling
              docker images and build cache. Deleted by ``--apply`` with a deletion manifest.
  PRUNE-ASK   everything else large. Listed for the owner; deleted only when its path is in the approval file.

When in doubt a file is PRUNE-ASK. Symlinks are never followed or deleted on their own; a deleted archived file
(scripts/archive_to_external.py) takes the NVMe symlink that points to it along.

    python3 scripts/storage_retention.py                 # dry run: report + what --apply would delete
    python3 scripts/storage_retention.py --apply         # delete PRUNE-AUTO (+ approved PRUNE-ASK)
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

GIB = 2**30
DAY = 86400
REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / 'training' / 'storage-policy.json'
STATE = Path.home() / '.local' / 'state' / 'natlang' / 'storage'
APPROVALS = Path.home() / '.config' / 'natlang' / 'storage-approved.txt'
LOGICAL_ROOTS = {'runs': REPO / 'runs', 'data': Path.home() / 'data'}  # archive/<name>/... ↔ its NVMe source

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive_to_external as archiver  # noqa: E402  (open files, ledger claims, archiver excludes)


def load_policy(path: Path = POLICY) -> dict:
    return json.loads(Path(path).read_text())


# Scan ----------------------------------------------------------------------------------------------------------------
def walk(root: Path):
    """(path, lstat) for every regular file and symlink under ``root``; symlinks are never followed."""
    stack = [str(root)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) or entry.is_symlink():
                            yield Path(entry.path), entry.stat(follow_symlinks=False)
                    except OSError:
                        continue
        except OSError:
            continue


def logical_path(path: Path, archive_root: Path) -> Path:
    """Where an archived file used to live (its NVMe symlink); other files are their own logical path."""
    if archive_root in path.parents:
        rel = path.relative_to(archive_root).parts
        if rel and rel[0] in LOGICAL_ROOTS:
            return LOGICAL_ROOTS[rel[0]].joinpath(*rel[1:])
    return path


def run_key(path: Path) -> Path:
    """The run a (logical) file belongs to: first level under runs/ or ~/data, two levels elsewhere."""
    for root in LOGICAL_ROOTS.values():
        if root in path.parents:
            rel = path.relative_to(root).parts
            return root / rel[0] if len(rel) > 1 else root
    parts = path.parts
    return Path(*parts[:min(len(parts) - 1, 6)])


def category(path: Path, policy: dict, registered: set[str]) -> str:
    name = path.name.lower()
    text = str(path).lower()
    ckpt = policy['checkpoint']
    if str(path) in registered:
        return 'artifact-registered' if path.suffix == '.nz' or '/artifacts/' in text else 'corpus-registered'
    if '/.sync-history/' in text or '/pull-revisions/' in text:
        return 'sync-revision'  # rsync --backup copies of mirror files a later sync replaced
    if '/.cache/' in text or '/cache/' in text:
        return 'cache'
    if path.suffix == '.gguf' or '/export' in text or '/hf/' in text:
        return 'export'
    if ('optimizer' in name or 'optim_' in name) and path.suffix in ckpt['suffixes']:
        return 'optimizer-state'
    if path.suffix in ckpt['suffixes']:
        if 'teacher' in text or 'logit' in text or 'top_' in name:
            return 'teacher-dump'
        if any(fnmatch.fnmatch(name, p) for p in ckpt['best_patterns'] + ckpt['final_patterns']):
            return 'weights-snapshot'
        return 'checkpoint'
    if path.suffix in ('.jsonl', '.json', '.parquet', '.arrow', '.npy', '.npz', '.tar', '.zst', '.gz', '.db',
                       '.sqlite'):
        if '/vendor/datasets' in text or '/datasets/' in text or path.suffix in ('.parquet', '.arrow'):
            return 'dataset'
        if 'teacher' in text or 'logit' in text:
            return 'teacher-dump'
        return 'corpus-unregistered' if path.suffix == '.jsonl' else 'data-other'
    if path.suffix in ('.log', '.out', '.err') or name.endswith('.log.gz'):
        return 'logs-metrics'
    if path.suffix == '.nz':
        return 'artifact-unregistered'
    return 'other'


# KEEP sets -----------------------------------------------------------------------------------------------------------
def registered_files(policy: dict, repo: Path = REPO) -> set[str]:
    """Absolute paths (and their symlink targets) of every file a corpus or artifact manifest pins."""
    found: set[str] = set()
    for registry in policy['keep']['registries']:
        try:
            entries = json.loads((repo / registry).read_text())
        except (OSError, ValueError):
            continue
        for entry in entries.get('corpora') or entries.get('artifacts') or []:
            manifest = entry.get('manifest')
            if isinstance(manifest, dict):
                manifest = manifest.get('path')
            if not manifest:  # by convention: training/{corpus,artifact}-manifests/<id>.json
                kind = 'artifact' if 'artifacts' in entries else 'corpus'
                conventional = repo / 'training' / f'{kind}-manifests' / f'{entry.get("id")}.json'
                manifest = str(conventional.relative_to(repo)) if conventional.exists() else None
            paths = []
            if manifest:
                try:
                    data = json.loads((repo / manifest).read_text())
                    base = repo / data.get('path', entry.get('path', ''))
                    paths = [base / f['path'] for f in data.get('files', [])]
                except (OSError, ValueError):
                    paths = []
            if not paths:
                base = repo / entry.get('path', '')
                paths = [base / item for item in entry.get('include', [])] or [base]
            for path in paths:
                found.add(str(path))
                try:
                    if path.is_symlink():
                        found.add(os.path.realpath(path))
                except OSError:
                    pass
    return found


def reference_text(policy: dict, repo: Path = REPO, limit=4 << 20) -> str:
    """Run scripts, recipes, registries, plans, certificates and receipts concatenated: a path they name is kept."""
    chunks = []
    for pattern in policy['keep']['reference_texts']:
        for name in glob.glob(str(repo / pattern), recursive=True):
            try:
                if os.path.getsize(name) <= limit:
                    chunks.append(Path(name).read_text(errors='replace'))
            except OSError:
                continue
    return '\n'.join(chunks)


def referenced(path: Path, text: str, repo: Path = REPO) -> bool:
    candidates = {str(path)}
    try:
        candidates.add(str(path.relative_to(repo)))
    except ValueError:
        pass
    for root in LOGICAL_ROOTS.values():
        if root in path.parents:
            candidates.add(str(path.relative_to(root.parent)))
    return any(c in text for c in candidates)


def dir_referenced(path: Path, text: str, repo: Path = REPO) -> bool:
    return referenced(path.parent, text, repo)


# Classification ------------------------------------------------------------------------------------------------------
def step_of(path: Path, regex: str) -> int | None:
    match = re.search(regex, path.name.lower())
    if not match:
        return None
    digits = next((g for g in match.groups() if g), None)
    return int(digits) if digits else None


def classify(entries: list[dict], policy: dict, *, registered: set[str], text: str, active: dict,
             now: float | None = None) -> list[dict]:
    """Tier and reason for each large entry (dicts with path, logical, bytes, mtime, category, run)."""
    now = now or time.time()
    keep = policy['keep']
    ckpt = policy['checkpoint']
    run_mtime: dict[str, float] = defaultdict(float)
    for e in entries:
        run_mtime[e['run']] = max(run_mtime[e['run']], e['mtime'])
    # Step checkpoints per directory: the latest step, best and final stay; the rest are intermediate.
    steps: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for e in entries:
        if e['category'] in ('checkpoint', 'optimizer-state'):
            step = step_of(Path(e['logical']), ckpt['step_regex'])
            if step is not None:
                steps[str(Path(e['logical']).parent)].append((step, e))
    latest = {d: max(s for s, _ in group) for d, group in steps.items()}
    out = []
    for e in entries:
        if e['bytes'] < policy['min_report_bytes'] and not _is_leftover(e, policy):
            continue
        logical = Path(e['logical'])
        name = logical.name.lower()
        age = (now - e['mtime']) / DAY
        run_age = (now - run_mtime[e['run']]) / DAY
        tier, reason = None, None
        if e['path'] in registered or e['logical'] in registered:
            tier, reason = 'KEEP', 'registered corpus/artifact file'
        elif referenced(logical, text) or referenced(Path(e['path']), text):
            tier, reason = 'KEEP', 'named by a run script, recipe, registry, plan or certificate'
        elif any(fnmatch.fnmatch(name, p) for p in keep['name_patterns']) or e['category'] == 'logs-metrics':
            tier, reason = 'KEEP', 'log/receipt/certificate/labels'
        elif any(fnmatch.fnmatch(e['logical'], p) or fnmatch.fnmatch(e['path'], p) for p in keep['path_patterns']):
            tier, reason = 'KEEP', 'policy keep path'
        elif why := active_reason(e, active):
            tier, reason = 'ACTIVE', why
        elif run_age < policy['finished_after_days']:
            tier, reason = 'ACTIVE', f'run changed {run_age:.1f} d ago (< {policy["finished_after_days"]} d)'
        elif _is_leftover(e, policy) and age >= policy['leftover_after_days']:
            tier, reason = 'PRUNE-AUTO', 'leftover partial write'
        elif e['category'] in ('checkpoint', 'optimizer-state', 'weights-snapshot') and \
                any(m in str(e['run']).lower() or m in str(logical.parent).lower()
                    for m in policy['aborted_run_markers']) and age >= policy['aborted_after_days']:
            if dir_referenced(logical, text):
                tier, reason = 'PRUNE-ASK', 'aborted run checkpoint, but its directory is named in a plan/script'
            else:
                tier, reason = 'PRUNE-AUTO', 'checkpoint of an aborted/stopped run'
        elif e['category'] in ('checkpoint', 'optimizer-state'):
            step = step_of(logical, ckpt['step_regex'])
            group = steps.get(str(logical.parent), [])
            protected = any(fnmatch.fnmatch(name, p) for p in ckpt['best_patterns'] + ckpt['final_patterns'])
            if step is not None and len(group) > 1 and step < latest[str(logical.parent)] and not protected:
                tier, reason = 'PRUNE-AUTO', f'intermediate checkpoint (step {step} < latest {latest[str(logical.parent)]})'
        if tier is None and e['category'] == 'sync-revision':
            tier, reason = 'PRUNE-ASK', f'superseded sync revision ({age:.0f} d old); approve its revision directory'
        if tier is None:
            tier, reason = 'PRUNE-ASK', f'large {e["category"]} file without a keep rule'
        out.append(dict(e, tier=tier, reason=reason))
    return out


def _is_leftover(e: dict, policy: dict) -> bool:
    return any(fnmatch.fnmatch(Path(e['logical']).name, p) for p in policy['leftover_patterns'])


def active_state(roots) -> dict:
    held = archiver.open_paths()
    cwds = archiver.process_cwds()
    claims = archiver.ledger_excludes()
    busy = set()
    for item in set(held) | set(cwds):
        if item.startswith('/'):
            busy.add(str(run_key(Path(item))))
    return {'held': held, 'claims': claims, 'busy': busy,
            'excludes': list(archiver.BUILTIN_EXCLUDES) + archiver.user_excludes(),
            'globs': archiver.BUILTIN_EXCLUDE_GLOBS}


def active_reason(e: dict, active: dict) -> str | None:
    for path in {e['path'], e['logical']}:
        if path in active['held']:
            return 'open by a process'
    if str(e['run']) in active['busy']:
        return 'a process works in its run'
    logical = Path(e['logical'])
    for ex in list(active['claims']) + list(active['excludes']):
        ex = Path(ex)
        if ex in LOGICAL_ROOTS.values() or ex == Path.home() or ex == REPO:
            continue
        if logical == ex or ex in logical.parents:
            return f'named by an active claim or exclude ({ex})'
    for pattern in active['globs']:
        if any(parent.match(pattern) for parent in logical.parents):
            return f'archiver exclude {pattern}'
    return None


def revision_dir(path: Path) -> str | None:
    """The revision directory (one sync run's backups) a sync-history file belongs to."""
    parts = path.parts
    for marker in ('.sync-history', 'pull-revisions'):
        if marker in parts and parts.index(marker) + 1 < len(parts):
            return str(Path(*parts[:parts.index(marker) + 2]))
    return None


def scan(policy: dict, roots=None, registered: set[str] | None = None,
         revisions: dict | None = None) -> tuple[list[dict], dict]:
    """Every file under the roots: per-category totals and the entries worth classifying (≥ 64 MiB or leftovers)."""
    roots = [Path(r) for r in (roots or policy['roots']['scan'])]
    archive_root = Path(policy['archive']['root'])
    registered = registered if registered is not None else registered_files(policy)
    totals: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    entries = []
    seen_inodes = set()
    for root in roots:
        for path, st in walk(root):
            if os.path.islink(path):
                totals[str(root)]['symlink'][0] += 1
                continue
            if (st.st_dev, st.st_ino) in seen_inodes:
                continue
            seen_inodes.add((st.st_dev, st.st_ino))
            logical = logical_path(path, archive_root)
            key = path if str(path) in registered else logical
            cat = category(key, policy, registered)
            bucket = totals[str(root)][cat]
            bucket[0] += 1
            bucket[1] += st.st_size
            if cat == 'sync-revision' and revisions is not None:
                key = revision_dir(path)
                if key:
                    revisions[key] = revisions.get(key, 0) + st.st_size
            if st.st_size >= (64 << 20) or any(fnmatch.fnmatch(path.name, p) for p in policy['leftover_patterns']):
                entries.append({'path': str(path), 'logical': str(logical), 'bytes': st.st_size,
                                'mtime': st.st_mtime, 'category': cat, 'run': str(run_key(logical))})
    return entries, {root: {c: {'files': v[0], 'bytes': v[1]} for c, v in cats.items()}
                     for root, cats in totals.items()}


# Apply ---------------------------------------------------------------------------------------------------------------
def approved_paths(path: Path = APPROVALS) -> set[str]:
    try:
        return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith('#')}
    except OSError:
        return set()


def delete(entry: dict, manifest: Path, held: set[str]) -> dict | None:
    """Delete one classified file after re-checking it; also the NVMe symlink that points to an archived copy."""
    path = Path(entry['path'])
    try:
        st = path.lstat()
    except OSError:
        return None
    if os.path.islink(path) or st.st_size != entry['bytes'] or abs(st.st_mtime - entry['mtime']) > 1:
        return None  # changed since the scan: leave it for the next pass
    if entry['path'] in held or entry['logical'] in held:
        return None
    path.unlink()
    removed_link = None
    logical = Path(entry['logical'])
    if logical != path and os.path.islink(logical) and os.path.realpath(logical) == str(path):
        logical.unlink()
        removed_link = str(logical)
    record = {'path': str(path), 'link': removed_link, 'bytes': entry['bytes'], 'category': entry['category'],
              'tier': entry['tier'], 'reason': entry['reason'], 'run': entry['run'], 'time': time.time()}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, 'a') as handle:
        handle.write(json.dumps(record) + '\n')
    return record


def docker_prune(apply: bool) -> dict:
    """Dangling images and build cache: regenerable. Containers and tagged images are never touched."""
    result = {'dangling_images': [], 'applied': apply}
    try:
        listing = subprocess.run(['docker', 'images', '-f', 'dangling=true', '--format', '{{.ID}} {{.Size}}'],
                                 capture_output=True, text=True, timeout=60).stdout.split('\n')
        result['dangling_images'] = [line for line in listing if line.strip()]
        if apply:
            result['image_prune'] = subprocess.run(['docker', 'image', 'prune', '-f'], capture_output=True,
                                                   text=True, timeout=600).stdout.strip().splitlines()[-1:]
            result['builder_prune'] = subprocess.run(['docker', 'builder', 'prune', '-f'], capture_output=True,
                                                     text=True, timeout=600).stdout.strip().splitlines()[-1:]
    except (OSError, subprocess.SubprocessError) as error:
        result['error'] = str(error)
    return result


def run(args) -> dict:
    policy = load_policy(args.policy)
    registered = registered_files(policy)
    revisions: dict = {}
    entries, totals = scan(policy, args.roots, registered, revisions)
    text = reference_text(policy)
    active = active_state(policy['roots']['scan'])
    classified = classify(entries, policy, registered=registered, text=text, active=active)
    approved = approved_paths(Path(args.approvals))
    for e in classified:
        if e['tier'] == 'PRUNE-ASK' and any(e[k] == a or e[k].startswith(a.rstrip('/') + '/')
                                            for a in approved for k in ('path', 'logical')):
            e['approved'] = True
    deleted = []
    if args.apply:
        held = archiver.open_paths()  # fresh, right before deleting
        manifest = Path(args.manifest)
        for e in classified:
            if e['tier'] == 'PRUNE-AUTO' or e.get('approved'):
                record = delete(e, manifest, held)
                if record:
                    deleted.append(record)
        # Approved whole revision directories (sync backups): removed as a unit; rmtree never follows symlinks.
        for directory, size in sorted(revisions.items()):
            if directory in approved and not any(h == directory or h.startswith(directory + '/') for h in held):
                shutil.rmtree(directory, ignore_errors=True)
                record = {'path': directory, 'bytes': size, 'category': 'sync-revision', 'tier': 'PRUNE-ASK',
                          'reason': 'owner-approved sync revision directory', 'time': time.time()}
                with open(manifest, 'a') as handle:
                    handle.write(json.dumps(record) + '\n')
                deleted.append(record)
    docker = docker_prune(args.apply and not args.no_docker)
    tiers: dict = defaultdict(lambda: {'files': 0, 'bytes': 0})
    for e in classified:
        tiers[e['tier']]['files'] += 1
        tiers[e['tier']]['bytes'] += e['bytes']
    return {'schema': 'natlang.storage-retention/1', 'time': time.time(), 'applied': args.apply,
            'totals': totals, 'tiers': dict(tiers), 'entries': classified, 'deleted': deleted,
            'deleted_bytes': sum(r['bytes'] for r in deleted), 'docker': docker,
            'sync_revisions': dict(sorted(revisions.items()))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--apply', action='store_true', help='delete PRUNE-AUTO and approved PRUNE-ASK files')
    p.add_argument('--policy', default=str(POLICY))
    p.add_argument('--roots', nargs='*', default=None)
    p.add_argument('--approvals', default=str(APPROVALS))
    p.add_argument('--manifest', default=str(STATE / 'deletions.jsonl'))
    p.add_argument('--out', default=str(STATE / 'retention-latest.json'))
    p.add_argument('--no-docker', action='store_true')
    args = p.parse_args(argv)
    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    ask = sorted((e for e in report['entries'] if e['tier'] == 'PRUNE-ASK'), key=lambda e: -e['bytes'])
    print(json.dumps({'tiers': {k: {'files': v['files'], 'gb': round(v['bytes'] / GIB, 1)}
                                for k, v in report['tiers'].items()},
                      'deleted_gb': round(report['deleted_bytes'] / GIB, 1), 'report': str(out),
                      'top_ask': [(round(e['bytes'] / GIB, 1), e['path']) for e in ask[:10]]}, indent=1))


if __name__ == '__main__':
    main()
