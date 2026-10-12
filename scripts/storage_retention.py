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

The tier decision is pluggable (applications/ops-retention/DECOMPOSITION.md): ``--classifier crisp`` (default) applies
``crisp_verdict``; ``nl`` asks applications/ops-retention/classifyRetention.nl through ts-host/scripts/retention-classify.mjs
(``--nl-server``/``--nl-model``); ``shadow`` runs both, serves crisp and appends disagreements to
~/.local/state/natlang/storage/retention-classifier-shadow.jsonl.
"""
from __future__ import annotations

import argparse
import fnmatch
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
            real_base = os.path.realpath(base)  # one resolution per entry: a repo symlink into the HDD
            on_nvme = any(str(root) == real_base or real_base.startswith(str(root) + '/')
                          for root in LOGICAL_ROOTS.values())
            for path in paths:
                found.add(str(path))
                found.add(str(Path(real_base) / path.relative_to(base)) if base in path.parents or path == base
                          else str(path))
                # Archived files are NVMe symlinks to their HDD copy: per-file only there (an lstat of 100k+ files
                # on the busy HDD took minutes).
                if on_nvme:
                    try:
                        if path.is_symlink():
                            found.add(os.path.realpath(path))
                    except OSError:
                        pass
    return found


def expand(base: Path, pattern: str):
    """Like glob, but never descends through a symlinked directory (hundreds of runs/ entries link into the busy
    HDD; glob listed each of them). ``**`` matches any depth of real directories."""
    parts = Path(pattern).parts
    frontier = [base]
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        following = []
        for directory in frontier:
            if part == '**':
                stack = [directory]
                while stack:
                    current = stack.pop()
                    following.append(current)
                    try:
                        with os.scandir(current) as entries:
                            stack.extend(Path(e.path) for e in entries if e.is_dir(follow_symlinks=False))
                    except OSError:
                        pass
                continue
            try:
                with os.scandir(directory) as entries:
                    for e in entries:
                        if not fnmatch.fnmatch(e.name, part):
                            continue
                        if last and e.is_file(follow_symlinks=False):
                            following.append(Path(e.path))
                        elif not last and e.is_dir(follow_symlinks=False):
                            following.append(Path(e.path))
            except OSError:
                pass
        frontier = following
    return [str(p) for p in frontier]


def reference_text(policy: dict, repo: Path = REPO, limit=4 << 20) -> str:
    """Run scripts, recipes, registries, plans, certificates and receipts concatenated: a path they name is kept."""
    chunks = []
    for pattern in policy['keep']['reference_texts']:
        for name in expand(repo, pattern):
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


WEIGHTS = ('checkpoint', 'optimizer-state', 'weights-snapshot')


def retention_facts(entries: list[dict], policy: dict, *, registered: set[str], text: str, active: dict,
                    now: float | None = None) -> list[tuple[dict, dict]]:
    """(entry, facts) for each entry worth classifying: the exact facts both classifier implementations read
    (applications/ops-retention/types.ts, RetentionFacts)."""
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
        directory = str(logical.parent)
        step = step_of(logical, ckpt['step_regex']) if e['category'] in ('checkpoint', 'optimizer-state') else None
        out.append((e, {
            'category': e['category'],
            'age_days': round((now - e['mtime']) / DAY, 3),
            'run_age_days': round((now - run_mtime[e['run']]) / DAY, 3),
            'registered': e['path'] in registered or e['logical'] in registered,
            'named_by_reference': referenced(logical, text) or referenced(Path(e['path']), text),
            'directory_named_by_reference': dir_referenced(logical, text),
            'record_file': any(fnmatch.fnmatch(name, p) for p in keep['name_patterns']) or e['category'] == 'logs-metrics',
            'policy_keep_path': any(fnmatch.fnmatch(e['logical'], p) or fnmatch.fnmatch(e['path'], p)
                                    for p in keep['path_patterns']),
            'in_use': active_reason(e, active),
            'partial_write': _is_leftover(e, policy),
            'run_marked_stopped': any(m in str(e['run']).lower() or m in directory.lower()
                                      for m in policy['aborted_run_markers']),
            'step': step,
            'latest_step_in_directory': latest.get(directory),
            'step_files_in_directory': len(steps.get(directory, [])),
            'best_or_final': any(fnmatch.fnmatch(name, p) for p in ckpt['best_patterns'] + ckpt['final_patterns']),
        }))
    return out


def crisp_verdict(f: dict, policy: dict) -> tuple[str, str]:
    """The crisp implementation of the retention decision over one file's facts (classifyRetention.nl is the other)."""
    if f['registered']:
        return 'KEEP', 'registered corpus/artifact file'
    if f['named_by_reference']:
        return 'KEEP', 'named by a run script, recipe, registry, plan or certificate'
    if f['record_file']:
        return 'KEEP', 'log/receipt/certificate/labels'
    if f['policy_keep_path']:
        return 'KEEP', 'policy keep path'
    if f['in_use']:
        return 'ACTIVE', f['in_use']
    if f['run_age_days'] < policy['finished_after_days']:
        return 'ACTIVE', f'run changed {f["run_age_days"]:.1f} d ago (< {policy["finished_after_days"]} d)'
    if f['partial_write'] and f['age_days'] >= policy['leftover_after_days']:
        return 'PRUNE-AUTO', 'leftover partial write'
    if f['category'] in WEIGHTS and f['run_marked_stopped'] and f['age_days'] >= policy['aborted_after_days']:
        if f['directory_named_by_reference']:
            return 'PRUNE-ASK', 'aborted run checkpoint, but its directory is named in a plan/script'
        return 'PRUNE-AUTO', 'checkpoint of an aborted/stopped run'
    if f['category'] in ('checkpoint', 'optimizer-state') and f['step'] is not None and \
            f['step_files_in_directory'] > 1 and f['step'] < f['latest_step_in_directory'] and not f['best_or_final']:
        return 'PRUNE-AUTO', f'intermediate checkpoint (step {f["step"]} < latest {f["latest_step_in_directory"]})'
    if f['category'] == 'sync-revision':
        return 'PRUNE-ASK', f'superseded sync revision ({f["age_days"]:.0f} d old); approve its revision directory'
    return 'PRUNE-ASK', f'large {f["category"]} file without a keep rule'


def classify(entries: list[dict], policy: dict, *, registered: set[str], text: str, active: dict,
             now: float | None = None) -> list[dict]:
    """Tier and reason for each large entry (dicts with path, logical, bytes, mtime, category, run), by the crisp rules."""
    return [dict(e, tier=tier, reason=reason) for e, f in retention_facts(entries, policy, registered=registered,
                                                                          text=text, active=active, now=now)
            for tier, reason in [crisp_verdict(f, policy)]]


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


# Classifier modes (applications/ops-retention/DECOMPOSITION.md) ------------------------------------------------------
CLASSIFIER_MODES = {'crisp': 'crisp', 'nl': 'nl', 'shadow': 'shadow', 'natlang': 'nl', 'natural-language': 'nl'}
NODE_BRIDGE = REPO / 'ts-host' / 'scripts' / 'retention-classify.mjs'


def classifier_setting(args, policy: dict) -> dict:
    """The classifier setting: --classifier, else $NATLANG_RETENTION_CLASSIFIER, else the policy's classifier.mode,
    else crisp. The modes are ts-host's pluggable modes (crisp | nl | shadow)."""
    setting = dict(policy.get('classifier') or {})
    mode = getattr(args, 'classifier', None) or os.environ.get('NATLANG_RETENTION_CLASSIFIER') or setting.get('mode') \
        or 'crisp'
    if mode not in CLASSIFIER_MODES:
        raise SystemExit(f'classifier mode is "crisp", "nl" or "shadow", not {mode!r}')
    setting['mode'] = CLASSIFIER_MODES[mode]
    for key in ('server', 'model', 'shadow_log', 'concurrency'):
        if getattr(args, f'nl_{key}', None) is not None:
            setting[key] = getattr(args, f'nl_{key}')
    setting.setdefault('shadow_log', str(STATE / 'retention-classifier-shadow.jsonl'))
    return setting


def run_bridge(argv: list[str]) -> int:
    return subprocess.call(['node', str(NODE_BRIDGE), *argv])


def nl_classify(classified: list[dict], facts: list[dict], policy: dict, setting: dict, *, bridge=run_bridge) -> dict:
    """Run the natural-language classifier over the classified entries (in place) through the ts-host pluggable.
    shadow: the crisp tiers stay; disagreements go to the shadow log. nl: the served (bounded) verdicts replace the
    crisp ones; each entry keeps its crisp verdict under crisp_tier/crisp_reason. Returns the summary."""
    import tempfile
    mode = setting['mode']
    thresholds = {k: policy[k] for k in ('finished_after_days', 'aborted_after_days', 'leftover_after_days')}
    items = [{'id': str(i), 'path': e['path'], 'logical': e['logical'], 'bytes': e['bytes'], 'facts': f,
              'crisp': {'tier': e['tier'], 'reason': e['reason']}} for i, (e, f) in enumerate(zip(classified, facts))]
    with tempfile.TemporaryDirectory(prefix='retention-classifier-') as work:
        source, out = Path(work) / 'items.json', Path(work) / 'verdicts.json'
        source.write_text(json.dumps({'thresholds': thresholds, 'items': items}))
        argv = ['--input', str(source), '--out', str(out), '--mode', mode, '--shadow-log', str(setting['shadow_log'])]
        argv += [x for key in ('server', 'model', 'concurrency') if setting.get(key) is not None
                 for x in (f'--{key}', str(setting[key]))]
        try:
            status = bridge(argv)
            result = json.loads(out.read_text()) if status == 0 else None
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            status, result = repr(error), None
    if result is None:
        if mode == 'nl':  # the served implementation failed: stop before anything is deleted
            raise RuntimeError(f'natural-language retention classifier failed ({status}); nothing was deleted')
        return {'mode': mode, 'error': f'bridge failed ({status}); the crisp verdicts stand',
                'shadow_log': str(setting['shadow_log'])}
    if mode == 'nl':
        served = {item['id']: item for item in result['classified']}
        for i, e in enumerate(classified):
            verdict = served.get(str(i))
            if verdict is None:
                raise RuntimeError(f'natural-language retention classifier returned no verdict for {e["path"]}')
            e['crisp_tier'], e['crisp_reason'] = e['tier'], e['reason']
            e['tier'], e['reason'], e['served'] = verdict['tier'], verdict['reason'], verdict['served']
    return dict(result['summary'], classifier=result.get('classifier'), shadow_log=str(setting['shadow_log']))


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
    if getattr(args, 'from_report', None):  # reuse a report's walk (hours on the HDD); classification is redone
        previous = json.loads(Path(args.from_report).read_text())
        keys = ('path', 'logical', 'bytes', 'mtime', 'category', 'run')
        entries = [{k: e[k] for k in keys} for e in previous['entries']]
        totals, revisions = previous['totals'], previous.get('sync_revisions', {})
    else:
        entries, totals = scan(policy, args.roots, registered, revisions)
    text = reference_text(policy)
    active = active_state(policy['roots']['scan'])
    pairs = retention_facts(entries, policy, registered=registered, text=text, active=active)
    classified = [dict(e, tier=tier, reason=reason) for e, f in pairs for tier, reason in [crisp_verdict(f, policy)]]
    setting = classifier_setting(args, policy)
    classifier = {'mode': 'crisp'}
    if setting['mode'] != 'crisp':
        classifier = nl_classify(classified, [f for _, f in pairs], policy, setting)
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
            'totals': totals, 'tiers': dict(tiers), 'entries': classified, 'classifier': classifier, 'deleted': deleted,
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
    p.add_argument('--from-report', default=None, help="reuse a report's scan (JSON) instead of walking again")
    p.add_argument('--classifier', choices=sorted(CLASSIFIER_MODES), default=None,
                   help='crisp (default) | nl | shadow; else $NATLANG_RETENTION_CLASSIFIER or the policy classifier.mode')
    p.add_argument('--nl-server', default=None, help='OpenAI-compatible endpoint of the nl classifier')
    p.add_argument('--nl-model', default=None, help='served model id of the nl classifier')
    p.add_argument('--nl-shadow-log', default=None, help='JSONL of disagreements (default: state dir)')
    p.add_argument('--nl-concurrency', type=int, default=None, help='calls in flight (default 4)')
    args = p.parse_args(argv)
    report = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    ask = sorted((e for e in report['entries'] if e['tier'] == 'PRUNE-ASK'), key=lambda e: -e['bytes'])
    print(json.dumps({'tiers': {k: {'files': v['files'], 'gb': round(v['bytes'] / GIB, 1)}
                                for k, v in report['tiers'].items()},
                      'deleted_gb': round(report['deleted_bytes'] / GIB, 1), 'report': str(out),
                      'classifier': report['classifier'],
                      'top_ask': [(round(e['bytes'] / GIB, 1), e['path']) for e in ask[:10]]}, indent=1))


if __name__ == '__main__':
    main()
