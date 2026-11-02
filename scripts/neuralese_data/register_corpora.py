#!/usr/bin/env python3
"""Register corpus entries and immutable manifests from canonical main.

Build only the requested registry and manifest blobs from the clean selected
paths. The shared publisher uses a temporary index and validates each rebased
plan before pushing; unrelated dirty and staged work is never consumed. A
selected dirty path is an actionable conflict. Manifest reuse retains the
selected-file and full-stat checks, and candidate admissions retain their full
renderer gate. Temporary generated files are not Git worktrees.

Python: register(entries, message, replace=False, links=None)
CLI: register_corpora.py --entries FILE.json -m SUBJECT [--replace] [--link REL=TARGET]
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
LOCK = Path.home() / '.local/state/natlang/register-corpora.lock'
TRAILERS = 'Registration-Tool: scripts/neuralese_data/register_corpora.py'


def git(*args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(['git', *args], cwd=cwd or REPO, capture_output=True, text=True)  # REPO at call time
    if check and result.returncode:
        raise RuntimeError(f'git {" ".join(args)}: {result.stderr.strip()[-500:]}')
    return result


def registered_ids(fetch: bool = True) -> set[str]:
    """Ids registered on origin/main (the shared checkout may lag behind or hold uncommitted edits)."""
    if fetch:
        git('fetch', '-q', 'origin')
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


def _module(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _registry_with_entries(registry, entries, replace):
    import copy
    result = copy.deepcopy(registry)
    corpora = result.get('corpora')
    if not isinstance(corpora, list):
        raise ValueError('corpus registry is malformed')
    index = {entry['id']: i for i, entry in enumerate(corpora)}
    if len(index) != len(corpora):
        raise ValueError('corpus registry contains duplicate ids')
    for entry in entries:
        ident = entry.get('id') if isinstance(entry, dict) else None
        if not isinstance(ident, str) or not ident or '/' in ident or ident in {'.', '..'}:
            raise ValueError('corpus id must be one nonempty path component')
        if ident in index:
            if corpora[index[ident]] != entry:
                if not replace:
                    raise ValueError(f'{ident} is registered with different content; manifests are immutable: '
                                     'publish a new id (or --replace for an entry whose manifest is unchanged)')
                corpora[index[ident]] = entry
        else:
            index[ident] = len(corpora)
            corpora.append(entry)
    return result


def _guard_clean(publisher_git, paths, original_head):
    """Never consume someone else's staged, unstaged, untracked or newly committed selected paths."""
    status = publisher_git.out('status', '--porcelain=v1', '--untracked-files=all', '--', *paths)
    conflicts = [line for line in status.splitlines() if line]
    for path in paths:
        if publisher_git.entry('HEAD', path) != original_head[path]:
            conflicts.append(path + ' (HEAD changed during registration)')
        if original_head[path] is None and ((REPO / path).exists() or (REPO / path).is_symlink()):
            conflicts.append(path + ' (untracked or ignored file)')
    if conflicts:
        raise ValueError('registration selected paths have foreign or concurrent changes; preserve them and '
                         'commit/publish them separately before retrying:\n  ' + '\n  '.join(conflicts))


def attempt(entries: list[dict], message: str, replace: bool, check: bool, *, tries=10) -> list[str]:
    import io
    import contextlib
    import tempfile
    publisher = _module(REPO / 'scripts/git_publish.py', 'natlang_registration_git_publish')
    sync = _module(REPO / 'scripts/sync_training_corpora.py', 'natlang_registration_corpus_sync')
    git('fetch', '-q', 'origin')
    selected = [REGISTRY] + [f"{MANIFESTS}/{entry.get('id')}.json" for entry in entries if isinstance(entry, dict)]
    # Validate ids before allowing them to become Git or filesystem paths.
    for entry in entries:
        ident = entry.get('id') if isinstance(entry, dict) else None
        if not isinstance(ident, str) or not ident or '/' in ident or ident in {'.', '..'}:
            raise ValueError('corpus id must be one nonempty path component')
    if len({entry['id'] for entry in entries}) != len(entries):
        raise ValueError('registration request contains duplicate corpus ids')
    repository = publisher.Git(REPO)
    head_entries = {path: repository.entry('HEAD', path) for path in selected}
    _guard_clean(repository, selected, head_entries)
    upstream_entries = {path: repository.entry('origin/main', path) for path in selected}
    changed_upstream = [path for path in selected if head_entries[path] != upstream_entries[path]]
    if changed_upstream:
        sync_output = io.StringIO()
        with contextlib.redirect_stdout(sync_output), contextlib.redirect_stderr(sync_output):
            status = publisher.main(['-C', str(REPO), '--sync-only'])
        if status != publisher.EXIT_OK:
            raise ValueError('registration paths differ from origin/main; safe checkout sync was blocked. '
                'Preserve the other work and resolve the reported conflict:\n' + sync_output.getvalue()[-4000:])
        head_entries = {path: repository.entry('HEAD', path) for path in selected}
        _guard_clean(repository, selected, head_entries)
        changed_upstream = [path for path in selected if head_entries[path] != repository.entry('origin/main', path)]
        if changed_upstream:
            raise ValueError('registration paths contain unpublished local commits: ' + ', '.join(changed_upstream)
                + '; publish or reconcile those exact paths before retrying')
    registry = json.loads(repository.blob(head_entries[REGISTRY][1]))
    initial = {entry['id']: entry for entry in registry['corpora']}
    proposed = _registry_with_entries(registry, entries, replace)
    changed = [entry for entry in entries if initial.get(entry['id']) != entry or
               head_entries[f"{MANIFESTS}/{entry['id']}.json"] is None]
    if not changed:
        return []
    with tempfile.TemporaryDirectory(prefix='natlang-corpus-registration-') as temporary:
        scratch = Path(temporary)
        registry_file = scratch / 'registry.json'
        manifests_dir = scratch / 'manifests'
        manifests_dir.mkdir()
        registry_file.write_text(json.dumps(proposed, indent=2, ensure_ascii=False) + '\n')
        contents = {REGISTRY: registry_file.read_bytes()}
        for entry in changed:
            manifest_name = f"{MANIFESTS}/{entry['id']}.json"
            generated = manifests_dir / (entry['id'] + '.json')
            publish_cache = LOCK.parent / 'register-corpora-manifest-cache' / (entry['id'] + '.json')
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    value = sync.publish(REPO, entry, publish_cache, manifest_output=generated)
            except (OSError, ValueError) as error:
                raise ValueError(f"publish {entry['id']}: {error}") from error
            existing = head_entries[manifest_name]
            if existing is not None:
                old_bytes = repository.blob(existing[1])
                if json.loads(old_bytes) != value:
                    raise ValueError(f"immutable snapshot changed: {entry['id']}; publish a new id")
                contents[manifest_name] = old_bytes
                generated.write_bytes(old_bytes)
            else:
                contents[manifest_name] = generated.read_bytes()
        _guard_clean(repository, selected, head_entries)
        if check:
            quick = subprocess.run(['scripts/check-main.sh', '--quick'], cwd=REPO, capture_output=True, text=True)
            if quick.returncode:
                failure_path = LOCK.parent / 'register-corpora-last-check-failure.json'
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_failure = failure_path.with_suffix('.tmp')
                temporary_failure.write_text(json.dumps({'time_unix':time.time(), 'checkout':str(REPO),
                    'commit':git('rev-parse','HEAD').stdout.strip(), 'corpus_ids':[e['id'] for e in changed],
                    'python_override':os.environ.get('NATLANG_PYTHON'), 'returncode':quick.returncode,
                    'stdout':quick.stdout, 'stderr':quick.stderr}, indent=2)+'\n')
                temporary_failure.replace(failure_path)
                raise ValueError(f'check-main --quick is red; complete latest failure: {failure_path}')
        candidates = [entry for entry in changed if entry.get('candidate_selection_admission') is not None]
        if candidates:
            build = subprocess.run(['npm', 'run', '--silent', 'build'], cwd=REPO / 'ts-host', capture_output=True, text=True)
            if build.returncode:
                raise ValueError('candidate admission verifier build failed: ' + build.stderr[-1500:])
            for entry in candidates:
                verified = subprocess.run([sys.executable, str(REPO / 'scripts/neuralese_data/render_native_turn_corpus.py'),
                    '--corpus-id', entry['id'], '--source', str(REPO / entry['path'] / 'turns.jsonl'),
                    '--verify-admission-only', '--admission-registry-file', str(registry_file),
                    '--admission-manifest-file', str(manifests_dir / (entry['id'] + '.json'))],
                    cwd=REPO, capture_output=True, text=True)
                if verified.returncode:
                    raise ValueError(f"candidate admission {entry['id']}: {verified.stderr[-1500:]}")
        _guard_clean(repository, selected, head_entries)
        overrides = {path: ('100644', repository.out('hash-object','-w','--no-filters','--stdin',input=data))
                     for path,data in contents.items()}
        def registry_successor(g, _base, upstream):
            try:
                remote_entry = g.entry(upstream, REGISTRY)
                remote_registry = json.loads(g.blob(remote_entry[1]))
                remote_ids = {entry['id']: entry for entry in remote_registry['corpora']}
                for entry in entries:
                    ident = entry['id']
                    if remote_ids.get(ident) != initial.get(ident) and remote_ids.get(ident) != entry:
                        raise ValueError(f'{ident} changed concurrently upstream; reconcile the registry and retry')
                expected = _registry_with_entries(remote_registry, entries, replace)
                if expected == remote_registry:
                    return {REGISTRY: remote_entry}
                text = (json.dumps(expected, indent=2, ensure_ascii=False) + '\n').encode()
                return {REGISTRY: ('100644', g.out('hash-object','-w','--no-filters','--stdin',input=text))}
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise publisher.PublishError(str(error)) from error
        def validate_plan(g, upstream, plan):
            try:
                _guard_clean(g, selected, head_entries)
                # A retry reuses immutable blobs only after the selected file
                # inventory and five-field source identities are checked again.
                for entry in changed:
                    name = f"{MANIFESTS}/{entry['id']}.json"
                    cache = LOCK.parent / 'register-corpora-manifest-cache' / (entry['id'] + '.json')
                    current = sync.publish(REPO, entry, cache,
                        manifest_output=manifests_dir / (entry['id'] + '.json'))
                    if current != json.loads(contents[name]):
                        raise ValueError(f"source snapshot changed during registration: {entry['id']}")
                remote_entry = g.entry(upstream, REGISTRY)
                remote_registry = json.loads(g.blob(remote_entry[1]))
                remote_ids = {entry['id']:entry for entry in remote_registry['corpora']}
                for entry in entries:
                    ident = entry['id']
                    if remote_ids.get(ident) != initial.get(ident) and remote_ids.get(ident) != entry:
                        raise ValueError(f'{ident} changed concurrently upstream; reconcile the registry and retry')
                expected = _registry_with_entries(remote_registry, entries, replace)
                proposed_plan = {path: value for path,value,_ in plan}
                actual = proposed_plan.get(REGISTRY)
                if actual is None or json.loads(g.blob(actual[1])) != expected:
                    raise ValueError('upstream registry merge does not exactly preserve registration semantics')
                for entry in changed:
                    name=f"{MANIFESTS}/{entry['id']}.json"
                    if g.entry(upstream,name) not in (None,overrides[name]) or proposed_plan.get(name) != overrides[name]:
                        raise ValueError(f"immutable manifest {name} changed upstream; inspect it and publish a new id")
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise publisher.PublishError(str(error)) from error
        output = io.StringIO()
        body = '\n'.join(f"- {entry['id']}" for entry in changed)
        args = ['-C',str(REPO),'--no-check','--no-advance','--retries',str(max(0,tries-1)),
                '-m',message,'-m','Registered unattended by scripts/neuralese_data/register_corpora.py:\n'+body,
                '--trailer',TRAILERS,*contents]
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            status = publisher.main(args, content_overrides=overrides, final_overrides=registry_successor,
                                    plan_validator=validate_plan)
        if status != publisher.EXIT_OK:
            raise ValueError('registration was not published; preserve selected paths and resolve the conflict:\n'+output.getvalue()[-4000:])
        # The publisher uses a temporary index and --no-advance. Sync separately:
        # it refuses to overwrite any selected or foreign dirty/staged file.
        sync_output = io.StringIO()
        with contextlib.redirect_stdout(sync_output), contextlib.redirect_stderr(sync_output):
            status = publisher.main(['-C',str(REPO),'--sync-only'])
        if status != publisher.EXIT_OK:
            print('registration was published; checkout was not advanced: '+sync_output.getvalue().strip(), file=sys.stderr)
        return [entry['id'] for entry in changed]


def register(entries: list[dict], message: str, replace: bool = False, links: dict[str, str] | None = None,
             tries: int = 10, check: bool = True) -> list[str]:
    """Register ENTRIES from canonical main, preserving all other work."""
    for rel, target in (links or {}).items():
        link(rel, target)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        done = attempt(entries, message, replace, check, tries=tries)
    if done:
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
        try:
            git('fetch', '-q', 'origin')
            registry = json.loads(git('show', f'origin/main:{REGISTRY}').stdout)
        except (RuntimeError, ValueError) as error:
            print(f'register_corpora: {error}', file=sys.stderr)
            return 1
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
