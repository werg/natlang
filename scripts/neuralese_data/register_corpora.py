#!/usr/bin/env python3
"""Register corpora unattended: registry entries + immutable manifests, committed and pushed from a private worktree.

Generators (static curriculum campaign, Clef daily pools, S1/view-ask call stages, SWE bulk follow-up) finish while
the shared checkout holds other sessions' uncommitted edits, often to training/neuralese_corpora.json itself. Editing
the registry there and committing would sweep those edits into the commit, or leave the registration uncommitted for
an agent to finish by hand (coordinator decision 2026-10-11: no agent commits registrations by hand). This helper never
touches the shared checkout's files:

  1. under a host-wide lock, reset a private linked worktree (default ~/.cache/natlang-register-wt) to origin/main;
  2. add the entries to its registry (an id already registered with the same content is a no-op unless its manifest
     is missing on main, which is then published; a different entry
     under an existing id is an error unless --replace: published manifests are immutable);
  3. publish each entry's manifest there (scripts/sync_training_corpora.py publish --repo WORKTREE). Data paths are
     resolved through the shared checkout: the first component of an entry path the worktree lacks (gitignored data
     links such as data/neuralese/corpora/ID, runs/...) is symlinked to the shared checkout's;
  4. validate (registry parses, ids unique, every new entry has its manifest) and run scripts/check-main.sh --quick in
     the worktree (it links the main checkout's venv and node packages);
  5. commit only the registry and those manifests, push to origin main;
  6. on a lost push race, a fetch conflict or a red check, back off and redo 1-5 from the new origin/main (the registry
     change is regenerated, never merged textually). Afterwards advance the shared checkout without touching dirty
     files (scripts/git_publish.py --sync-only), best effort.

Python: ``register(entries, message, replace=False, links=None)``; ``registered_ids()`` reads origin/main's registry.
CLI:    register_corpora.py --entries FILE.json [--entries ...] -m SUBJECT [--replace] [--link REL=TARGET ...]
        (FILE holds one entry or a list of entries);
        register_corpora.py --id ID -m SUBJECT   publish the manifest of an entry registered earlier (data now final). Exit 0 registered (or nothing to do), 1 gave up.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REGISTRY = 'training/neuralese_corpora.json'
MANIFESTS = 'training/corpus-manifests'
WORKTREE = Path(os.environ.get('NATLANG_REGISTER_WORKTREE', Path.home() / '.cache/natlang-register-wt'))
LOCK = Path.home() / '.local/state/natlang/register-corpora.lock'
TRAILERS = ('Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
            'Claude-Session: https://claude.ai/code/session_01Q53kXkxDK2onzJmxGfyHnP')


class Retry(Exception):
    """A transient failure: push race, fetch error, red check. The whole attempt is redone after a backoff."""


def git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(['git', *args], cwd=cwd or REPO, capture_output=True, text=True)  # REPO at call time
    if check and result.returncode:
        raise Retry(f'git {" ".join(args)}: {result.stderr.strip()[-500:]}')
    return result


def registered_ids(fetch: bool = True) -> set[str]:
    """Ids registered on origin/main (the shared checkout may lag behind or hold uncommitted edits)."""
    if fetch:
        git('fetch', '-q', 'origin', check=False)
    shown = git('show', f'origin/main:{REGISTRY}')
    return {c['id'] for c in json.loads(shown.stdout)['corpora']}


def link(rel: str, target: str | Path) -> None:
    """Create the gitignored data link REL -> TARGET in the shared checkout (corpora live on NVMe)."""
    path = REPO / rel
    if path.is_symlink() and os.readlink(path) == str(target):
        return
    if path.exists() or path.is_symlink():
        raise ValueError(f'{rel} exists and is not a link to {target}')
    path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, path)


def prepare_worktree() -> Path:
    git('fetch', '-q', 'origin')
    if not (WORKTREE / '.git').exists():
        git('worktree', 'prune', check=False)
        WORKTREE.parent.mkdir(parents=True, exist_ok=True)
        git('worktree', 'add', '-q', '--detach', str(WORKTREE), 'origin/main')
    git('reset', '-q', '--hard', 'origin/main', cwd=WORKTREE)
    git('clean', '-q', '-fd', cwd=WORKTREE)  # untracked only (ignored node_modules links stay); this worktree is ours
    return WORKTREE


def resolve_data(wt: Path, rel: str) -> None:
    """Make WT/REL resolve to the shared checkout's data by linking its first missing component."""
    parts = Path(rel).parts
    for i in range(1, len(parts) + 1):
        here = wt.joinpath(*parts[:i])
        if not here.exists() and not here.is_symlink():
            there = REPO.joinpath(*parts[:i])
            if not there.exists():
                raise ValueError(f'data path missing in the shared checkout: {rel}')
            here.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(there, here)
            return


def attempt(entries: list[dict], message: str, replace: bool, check: bool) -> list[str]:
    wt = prepare_worktree()
    path = wt / REGISTRY
    registry = json.loads(path.read_text())
    index = {c['id']: i for i, c in enumerate(registry['corpora'])}
    changed = []
    for entry in entries:
        if entry['id'] in index:
            if registry['corpora'][index[entry['id']]] == entry:
                # Registered earlier but its manifest never reached main (an older flow left it uncommitted).
                if not (wt / MANIFESTS / f"{entry['id']}.json").exists():
                    changed.append(entry)
                continue
            if not replace:
                raise ValueError(f"{entry['id']} is registered with different content; manifests are immutable: "
                                 'publish a new id (or --replace for an entry whose manifest is unchanged)')
            registry['corpora'][index[entry['id']]] = entry
        else:
            registry['corpora'].append(entry)
        changed.append(entry)
    if not changed:
        return []
    text = json.dumps(registry, indent=2, ensure_ascii=False) + '\n'
    if text != path.read_text():
        path.write_text(text)
    for entry in changed:
        resolve_data(wt, entry['path'])
        published = subprocess.run([sys.executable, str(wt / 'scripts/sync_training_corpora.py'), 'publish',
                                    '--repo', str(wt), '--machine', entry['owner'], '--id', entry['id']],
                                   cwd=wt, capture_output=True, text=True)
        if published.returncode:
            raise ValueError(f"publish {entry['id']}: {published.stderr.strip()[-1500:]}")
    reloaded = json.loads(path.read_text())
    ids = [c['id'] for c in reloaded['corpora']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate registry ids after the change')
    paths = [REGISTRY] + [f"{MANIFESTS}/{e['id']}.json" for e in changed]
    missing = [p for p in paths if not (wt / p).is_file()]
    if missing:
        raise ValueError(f'missing after publish: {missing}')
    if check:
        quick = subprocess.run(['scripts/check-main.sh', '--quick'], cwd=wt, capture_output=True, text=True)
        if quick.returncode:
            raise Retry('check-main --quick is red on origin/main + this registration:\n' + quick.stdout[-1500:])
    git('add', '--', *paths, cwd=wt)
    body = '\n'.join(f"- {e['id']}" for e in changed)
    git('commit', '-q', '-m', message, '-m', 'Registered unattended by scripts/neuralese_data/register_corpora.py:\n'
        + body, '-m', TRAILERS, '--', *paths, cwd=wt)
    pushed = git('push', '-q', 'origin', 'HEAD:main', cwd=wt, check=False)
    if pushed.returncode:
        raise Retry(f'push rejected: {pushed.stderr.strip()[-300:]}')
    return [e['id'] for e in changed]


def register(entries: list[dict], message: str, replace: bool = False, links: dict[str, str] | None = None,
             tries: int = 10, check: bool = True) -> list[str]:
    """Register ENTRIES on origin/main; returns the ids added or replaced (empty: already registered)."""
    for rel, target in (links or {}).items():
        link(rel, target)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for n in range(tries):
            try:
                done = attempt(entries, message, replace, check)
                break
            except Retry as error:
                wait = min(60 * 2 ** n, 1800)
                print(json.dumps({'register_retry': n + 1, 'wait_s': wait, 'reason': str(error)[-600:]}),
                      file=sys.stderr, flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f'registration gave up after {tries} attempts')
    if done:
        subprocess.run([sys.executable, str(REPO / 'scripts/git_publish.py'), '--sync-only'], cwd=REPO,
                       capture_output=True)
        print(json.dumps({'registered': done}), flush=True)
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--entries', type=Path, action='append', default=[])
    ap.add_argument('--id', action='append', default=[], dest='ids',
                    help="an entry already registered on origin/main whose manifest is to be published now")
    ap.add_argument('-m', '--message', required=True)
    ap.add_argument('--replace', action='store_true')
    ap.add_argument('--link', action='append', default=[], help='REL=TARGET data link in the shared checkout')
    ap.add_argument('--tries', type=int, default=10)
    args = ap.parse_args()
    entries = []
    for path in args.entries:
        value = json.loads(path.read_text())
        entries.extend(value if isinstance(value, list) else [value])
    if args.ids:
        registry = json.loads(git('show', f'origin/main:{REGISTRY}').stdout) if git('fetch', '-q', 'origin', check=False) else None
        found = {c['id']: c for c in registry['corpora'] if c['id'] in args.ids}
        if set(args.ids) - set(found):
            print(f'register_corpora: not registered on origin/main: {sorted(set(args.ids) - set(found))}', file=sys.stderr)
            return 1
        entries.extend(found[i] for i in args.ids)
    if not entries:
        ap.error('--entries or --id required')
    links = dict(item.split('=', 1) for item in args.link)
    try:
        register(entries, args.message, args.replace, links, args.tries)
    except (RuntimeError, ValueError) as error:
        print(f'register_corpora: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
