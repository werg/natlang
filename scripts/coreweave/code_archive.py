"""Reconcile a source-addressed code archive without deleting inputs or untracked work.

The native package inventory is exactly what native_resident.code_inventory authenticates. A restored
workspace may contain tracked Python files deleted in a newer archive. Only previously inventoried
files whose bytes still match that predecessor are removed. Unknown extra files are never deleted;
they make startup fail until their custody is resolved.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath
import tempfile

PACKAGE = 'training/neuralese/natlang_neuralese'


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def archive_inventory(tar):
    expected = {}
    prefix = PACKAGE + '/'
    for member in tar.getmembers():
        name = member.name.removeprefix('./')
        if not name.startswith(prefix) or not name.endswith('.py'):
            continue
        relative = name[len(prefix):]
        if (not member.isfile() or not relative or '..' in PurePosixPath(relative).parts
                or relative in expected):
            raise ValueError('code archive contains an unsafe or duplicate native Python entry')
        with tar.extractfile(member) as stream:
            expected[relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if not expected:
        raise ValueError('code archive lacks the native Python package')
    return expected


def predecessor_inventory(work, manifest_sha256):
    """Authenticate a legacy snapshot's original manifest and its content-addressed archive."""
    if not re.fullmatch('[0-9a-f]{64}', manifest_sha256):
        raise ValueError('legacy predecessor manifest SHA-256 is malformed')
    root = Path(work)/'.coreweave'
    manifest = root/'bundle.json'
    if not manifest.is_file() or digest(manifest) != manifest_sha256:
        raise ValueError('legacy predecessor bundle manifest does not match its bound SHA-256')
    code = json.loads(manifest.read_text())['code']
    sha = code['sha256']
    if not re.fullmatch('[0-9a-f]{64}', sha):
        raise ValueError('legacy predecessor archive SHA-256 is malformed')
    path = root/'cas'/sha
    if not path.is_file() or path.stat().st_size != code['bytes'] or digest(path) != sha:
        raise ValueError('legacy predecessor code archive does not match its manifest')
    with tarfile.open(path) as tar:
        return archive_inventory(tar)


def extract_code_archive(tar, repo, work, archive_sha256, *, predecessor_manifest_sha256=None):
    repo, work = Path(repo), Path(work)
    package = repo / PACKAGE
    expected = archive_inventory(tar)
    inventory_path = work/'.coreweave'/'installed-code-inventory.json'
    plan_path = work/'diloco-resident'/'recipe-plan.json'
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        if inventory.get('schema') != 'natlang.coreweave-code-inventory/1':
            raise ValueError('prior installed code inventory has an unknown schema')
        previous = inventory['files']
    elif plan_path.exists():
        previous = json.loads(plan_path.read_text())['code']
    elif predecessor_manifest_sha256:
        previous = predecessor_inventory(work, predecessor_manifest_sha256)
    else:
        previous = {}
    existing = {str(path.relative_to(package)): path for path in package.rglob('*.py')}
    untracked = sorted(set(existing) - set(previous) - set(expected))
    if untracked:
        raise ValueError('untracked native Python files need explicit custody: ' + ', '.join(untracked[:8]))
    for relative, path in existing.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError('native Python checkout contains a nonregular file: ' + str(path))
        actual = digest(path)
        if actual not in {value for value in (previous.get(relative), expected.get(relative)) if value is not None}:
            raise ValueError('existing native code changed; preserve it: ' + str(path))
    for relative in sorted(set(previous)-set(expected)):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or '..' in pure.parts or not relative.endswith('.py'):
            raise ValueError('prior code inventory contains an unsafe path')
        path = package/relative
        if path.is_symlink():
            raise ValueError('refuse to remove a symlinked prior code file')
        if path.exists():
            if not path.is_file() or digest(path) != previous[relative]:
                raise ValueError('prior tracked code changed; do not remove: ' + str(path))
            path.unlink()
    tar.extractall(repo, filter='data')
    actual = {str(path.relative_to(package)):digest(path) for path in package.rglob('*.py') if path.is_file()}
    if actual != expected:
        extra = sorted(set(actual)-set(expected)); wrong = sorted(k for k in set(actual)&set(expected) if actual[k]!=expected[k])
        raise ValueError(f'native code checkout differs from source archive; extra={extra[:8]} wrong={wrong[:8]}')
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    record = {'schema':'natlang.coreweave-code-inventory/1','archive_sha256':archive_sha256,'files':expected}
    fd, temporary = tempfile.mkstemp(prefix='.code-inventory-', dir=inventory_path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(record,stream,sort_keys=True); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,inventory_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'archive_sha256':archive_sha256,'tracked_native_python_files':len(expected),
            'removed_predecessor_files':len(set(previous)-set(expected))}
