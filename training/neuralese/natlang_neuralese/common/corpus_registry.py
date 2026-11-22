"""The git layer of the training-data registry, one file per corpus entry.

Layout (2026-11-20; it replaced the single 54 MB ``training/neuralese_corpora.json``, whose every change added a whole
copy to git history):

  training/corpus-registry.json         {schema, policy, coverage_backlog}: small, rarely changes
  training/corpora/<id>.json            one registry entry (the object that was one element of ``corpora``)
  training/corpus-manifests/<id>.json   its immutable file snapshot (unchanged)

A registration changes its own entry file and manifest only. ``load()`` returns the old document shape
(``{schema, policy, coverage_backlog, corpora: [...]}``, entries sorted by id) so readers keep their logic; every
reader and writer goes through this module (``tests/test_data_registry_ratchet.py`` keeps it that way).

An entry whose file list was too long to keep inline (thousands of exact paths) carries ``include_manifest: true``:
its files are exactly its manifest's, and ``include_prefixes`` names the directories they live under (for lookups).

Stdlib only.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ENTRIES = 'training/corpora'
META = 'training/corpus-registry.json'
MANIFESTS = 'training/corpus-manifests'
LEGACY = 'training/neuralese_corpora.json'
SCHEMA = 'natlang.neuralese-corpus-registry/2'


def repo_of(path) -> Path:
    """The checkout of PATH: a checkout root, its ``training/corpora`` directory, the meta file, or the retired
    single-file path (scripts and configs that still name it)."""
    p = Path(path)
    for suffix in (ENTRIES, META, LEGACY):
        if str(p).endswith(suffix):
            return Path(str(p)[:-len(suffix)] or '.')
    return p


def entry_path(repo, identity: str) -> Path:
    if not identity or '/' in identity or identity in {'.', '..'}:
        raise ValueError('corpus id must be one nonempty path component')
    return repo_of(repo) / ENTRIES / f'{identity}.json'


def entry_root(entry, repo=None) -> Path:
    """The local directory of a registry entry (or of a bare ``path`` value): a ``root:rel`` portable path under a
    logical machine root (``data_nvme:...``, used by held-v2/v3 since 2026-11-20) or a checkout-relative one. Every
    reader resolves entry paths here, never as ``repo / entry['path']``."""
    from .data_registry import absolute
    value = entry['path'] if isinstance(entry, dict) else str(entry)
    return absolute(value, repo_of(repo) if repo is not None else Path(__file__).resolve().parents[4])


def entry_text(entry: dict) -> str:
    return json.dumps(entry, indent=2, ensure_ascii=False) + '\n'


def load(path=None, *, sort=True) -> dict:
    """The registry document of the checkout at PATH (default: this module's checkout)."""
    repo = repo_of(path) if path is not None else Path(__file__).resolve().parents[4]
    try:
        meta = json.loads((repo / META).read_text())
    except (OSError, ValueError):
        meta = {'schema': SCHEMA}
    corpora = []
    directory = repo / ENTRIES
    names = sorted(os.listdir(directory)) if directory.is_dir() else []
    for name in names:
        if name.endswith('.json'):
            corpora.append(json.loads((directory / name).read_text()))
    if not sort:
        return {**meta, 'corpora': corpora}
    return {**meta, 'corpora': sorted(corpora, key=lambda e: e['id'])}


def signature(path=None):
    """Changes whenever an entry is added, removed or rewritten (entries are replaced atomically, which renames into
    the directory): cheap cache key for readers that keep the loaded registry."""
    repo = repo_of(path) if path is not None else Path(__file__).resolve().parents[4]
    try:
        s = (repo / ENTRIES).stat()
        m = (repo / META).stat()
        return (s.st_mtime_ns, s.st_ino, m.st_mtime_ns, m.st_size)
    except OSError:
        return None


def write_entries(repo, entries, *, replace=False) -> list[str]:
    """Write ENTRIES as their own files (atomically). An existing different entry needs ``replace``. Returns the ids
    whose file changed."""
    changed = []
    for entry in entries:
        path = entry_path(repo, entry['id'])
        text = entry_text(entry)
        if path.exists():
            current = path.read_text()
            if current == text:
                continue
            if json.loads(current) != entry and not replace:
                raise ValueError(f"{entry['id']} is registered with different content; manifests are immutable: "
                                 'publish a new id (or replace an entry whose manifest is unchanged)')
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.json.{os.getpid()}.tmp')
        tmp.write_text(text)
        tmp.replace(path)
        changed.append(entry['id'])
    return changed


def save(path, document: dict) -> list[str]:
    """Write a whole registry document (a reader that edited ``load()``'s result): every entry that changed, and the
    removal of entries no longer present. Returns the changed ids."""
    repo = repo_of(path)
    meta = {k: v for k, v in document.items() if k != 'corpora'}
    meta_path = repo / META
    meta_text = json.dumps(meta, indent=2, ensure_ascii=False) + '\n'
    if not meta_path.exists() or meta_path.read_text() != meta_text:
        meta_path.write_text(meta_text)
    changed = write_entries(repo, document['corpora'], replace=True)
    keep = {e['id'] for e in document['corpora']}
    directory = repo / ENTRIES
    for name in os.listdir(directory) if directory.is_dir() else []:
        if name.endswith('.json') and name[:-5] not in keep:
            (directory / name).unlink()
            changed.append(name[:-5])
    return changed


def load_ref(repo, ref: str) -> dict:
    """The registry document at git REF (``origin/main``): read with one ``git cat-file --batch``."""
    repo = repo_of(repo)

    def git(*args, data=None):
        return subprocess.run(['git', '-C', str(repo), *args], input=data, capture_output=True, check=True).stdout
    listing = git('ls-tree', '-r', '-z', ref, '--', ENTRIES).decode().split('\0')
    objects = [line.split('\t', 1) for line in listing if line]
    shas = [meta.split()[2] for meta, name in objects if name.endswith('.json')]
    corpora = []
    if shas:
        out = git('cat-file', '--batch', data=('\n'.join(shas) + '\n').encode())
        offset = 0
        for _ in shas:
            header_end = out.index(b'\n', offset)
            size = int(out[offset:header_end].split()[2])
            body = out[header_end + 1:header_end + 1 + size]
            corpora.append(json.loads(body))
            offset = header_end + 1 + size + 1
    try:
        meta = json.loads(git('show', f'{ref}:{META}'))
    except subprocess.CalledProcessError:
        meta = {'schema': SCHEMA}
    return {**meta, 'corpora': sorted(corpora, key=lambda e: e['id'])}


def manifest_files(repo, entry: dict) -> list[str] | None:
    """The exact file list of an ``include_manifest`` entry (its manifest's), else None."""
    if not entry.get('include_manifest'):
        return None
    manifest = json.loads((repo_of(repo) / MANIFESTS / f"{entry['id']}.json").read_text())
    return [f['path'] for f in manifest['files']]
