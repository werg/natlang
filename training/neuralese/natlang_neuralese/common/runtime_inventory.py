"""Exact built task runtime, npm and protection-file identity shared by providers and eval."""
from pathlib import Path
import json
import os
import hashlib
import stat
from .hashing import sha256_file_hex


_DEPENDENCY_RECEIPT = 'runtime-dependencies.json'


def _dependency_tree(path):
    """Hash the logical contents of runtime dependency roots, including link identity.

    File bytes use the strict stat-keyed digest cache: unchanged files avoid another
    disk read, while inode/size/mtime/ctime changes require a fresh hash. Walking
    entries on each custody boundary also detects added, removed and retargeted
    dependencies.
    """
    from .hashing import sha256_file_hex_cached
    path = Path(path)
    files = {}
    links = {}
    roots = ('node_modules', '../node_modules', 'vendor')

    def visit(item, logical, ancestors):
        info = item.lstat()
        if item.is_symlink():
            target_text = os.readlink(item)
            target = item.resolve(strict=True)
            links[logical] = {'target': target_text, 'resolved': str(target)}
            if target.is_dir():
                files[logical] = {'kind': 'directory', 'mode': stat.S_IMODE(target.stat().st_mode)}
                real = str(target)
                if real in ancestors:
                    raise ValueError(f'runtime dependency symlink cycle: {item}')
                visit_directory(target, logical, ancestors | {real})
            elif target.is_file():
                target_info = target.stat()
                files[logical] = {'kind': 'file', 'bytes': target_info.st_size,
                                  'mode': stat.S_IMODE(target_info.st_mode),
                                  'sha256': sha256_file_hex_cached(target, trust_archive_receipt=False)}
            else:
                raise ValueError(f'runtime dependency link has unsupported target: {item}')
        elif item.is_dir():
            files[logical] = {'kind': 'directory', 'mode': stat.S_IMODE(info.st_mode)}
            real = str(item.resolve(strict=True))
            if real in ancestors:
                raise ValueError(f'runtime dependency directory cycle: {item}')
            visit_directory(item, logical, ancestors | {real})
        elif item.is_file():
            files[logical] = {'kind': 'file', 'bytes': info.st_size, 'mode': stat.S_IMODE(info.st_mode),
                              'sha256': sha256_file_hex_cached(item, trust_archive_receipt=False)}
        else:
            raise ValueError(f'runtime dependency entry is not a regular file or directory: {item}')

    def visit_directory(folder, logical, ancestors):
        for child in sorted(folder.iterdir(), key=lambda value: value.name):
            visit(child, f'{logical}/{child.name}', ancestors)

    for name in roots:
        root = path / name
        if root.exists() or root.is_symlink():
            visit(root, name, set())
    return {'files': files, 'links': links}


def _dependency_receipt(path, static_inventory):
    from .hashing import canonical_json_sha256_hex
    receipt_path = Path(path) / _DEPENDENCY_RECEIPT
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    if (not isinstance(receipt, dict) or receipt.get('schema') != 'natlang.task-runtime-dependencies/1'
            or not isinstance(receipt.get('tree'), dict) or not isinstance(receipt.get('source_links'), dict)
            or not isinstance(receipt.get('source_inventory_extras'), dict)):
        raise ValueError('frozen task runtime dependency receipt is malformed')
    source_identity = receipt.get('source_inventory_sha256')
    container = receipt_path.parent.parent.name
    if (not isinstance(source_identity, str) or len(source_identity) != 64
            or any(character not in '0123456789abcdef' for character in source_identity)
            or (container != source_identity
                and not container.startswith(f'.preparing-{source_identity[:16]}-'))):
        raise ValueError('frozen task runtime snapshot identity differs from its dependency receipt')
    payload = {key: value for key, value in receipt.items() if key != 'sha256'}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if receipt.get('sha256') != expected:
        raise ValueError('frozen task runtime dependency receipt digest differs')
    reconstructed = dict(static_inventory)
    reconstructed.update({'@dependency-files': receipt['tree'].get('files'),
                          '@dependency-source-links': receipt['source_links']})
    reconstructed.update(receipt['source_inventory_extras'])
    if canonical_json_sha256_hex(reconstructed) != source_identity:
        raise ValueError('task runtime closure changed: dependency receipt differs from its snapshot identity')
    actual = _dependency_tree(path)
    if actual['files'] != receipt['tree'].get('files'):
        raise ValueError('frozen task runtime dependency contents changed')
    if actual['links']:
        raise ValueError('frozen task runtime dependency snapshot unexpectedly contains symlinks')
    return receipt


def snapshot_task_runtime(source, directory):
    """Publish a physical built runtime so async scoring survives live rebuilds."""
    import fcntl
    import os
    import shutil
    import tempfile
    from .hashing import canonical_json_sha256_hex
    source = Path(source).resolve(strict=True)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inventory = task_runtime_inventory(source)
        source_actual_dependencies = _dependency_tree(source)
        source_dependencies = {'files': inventory['@dependency-files'],
                              'links': inventory['@dependency-source-links']}
        if source_actual_dependencies['files'] != source_dependencies['files']:
            raise ValueError('task runtime dependency contents changed before preparation')
        if (not inventory.get('@dependency-receipt-sha256')
                and source_actual_dependencies['links'] != source_dependencies['links']):
            raise ValueError('task runtime dependency links changed before preparation')
        destination = directory / canonical_json_sha256_hex(inventory)
        runtime = destination / 'ts-host'
        if destination.is_symlink():
            raise ValueError('prepared task runtime destination cannot be a symlink')
        if destination.exists():
            if not destination.is_dir() or runtime.is_symlink() or runtime.resolve(strict=True).parent != destination.resolve(strict=True):
                raise ValueError('prepared task runtime destination has an invalid layout')
            existing = task_runtime_inventory(runtime)
            source_static = {key: value for key, value in inventory.items()
                             if not key.startswith('@dependency-')}
            existing_static = {key: value for key, value in existing.items()
                               if not key.startswith('@dependency-')}
            source_extras = {key: value for key, value in inventory.items()
                             if key.startswith('@dependency-')
                             and key not in ('@dependency-files', '@dependency-source-links')}
            existing_receipt = json.loads((runtime / _DEPENDENCY_RECEIPT).read_text())
            if (existing_static != source_static
                    or existing.get('@dependency-files') != source_dependencies['files']
                    or existing.get('@dependency-source-links') != source_dependencies['links']
                    or existing_receipt.get('source_inventory_extras') != source_extras
                    or _dependency_tree(runtime)['files'] != source_dependencies['files']):
                raise ValueError('prepared task runtime changed')
            return runtime
        source_identity = canonical_json_sha256_hex(inventory)
        pending = Path(tempfile.mkdtemp(prefix=f'.preparing-{source_identity[:16]}-', dir=directory))
        try:
            target = pending / 'ts-host'
            for name in inventory:
                if name.startswith('@dependency-'):
                    continue
                copied = target / name
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, copied)
            # Node resolves packages both beside ts-host and in its parent.
            # Physical copies also isolate installs; optional vendor assets serve
            # Python folders. These are existing dependencies, never rebuilt here.
            for name in ('node_modules', '../node_modules', 'vendor'):
                if (source / name).is_dir():
                    shutil.copytree(source / name, target / name, symlinks=False)
            receipt = {'schema': 'natlang.task-runtime-dependencies/1',
                       'source_inventory_sha256': canonical_json_sha256_hex(inventory),
                       'tree': {'files': source_dependencies['files']},
                       'source_links': source_dependencies['links'],
                       'source_inventory_extras': {
                           key: value for key, value in inventory.items()
                           if key.startswith('@dependency-')
                           and key not in ('@dependency-files', '@dependency-source-links')}}
            receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            (target / _DEPENDENCY_RECEIPT).write_text(json.dumps(receipt, sort_keys=True, separators=(',', ':')) + '\n')
            target_inventory = task_runtime_inventory(target)
            source_static = {key: value for key, value in inventory.items()
                             if not key.startswith('@dependency-')}
            target_static = {key: value for key, value in target_inventory.items()
                             if not key.startswith('@dependency-')}
            if (task_runtime_inventory(source) != inventory
                    or _dependency_tree(source) != source_actual_dependencies
                    or source_static != target_static
                    or target_inventory.get('@dependency-files') != source_dependencies['files']
                    or target_inventory.get('@dependency-source-links') != source_dependencies['links']):
                raise ValueError('task runtime changed during preparation')
            pending.rename(destination)
            fd = os.open(directory, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except BaseException:
            shutil.rmtree(pending, ignore_errors=True)
            raise
        return runtime


def task_source_reviews(path):
    """Use the registry carried by a frozen runtime, or beside a live ts-host."""
    path = Path(path)
    inside = path / 'training/source-reviews'
    return inside if (path / 'frozen-runtime.json').is_file() or inside.is_dir() else path.parent / 'training/source-reviews'


def task_runtime_inventory(path):
    path = Path(path)
    frozen = path / 'frozen-runtime.json'
    names = ['prelude.js', 'package.json']
    names += ['package-lock.json', 'frozen-runtime.json'] if frozen.is_file() else ['../package.json', '../package-lock.json']
    reviews = task_source_reviews(path)
    names += [os.path.relpath(reviews/name, path) for name in ('holds.jsonl', 'family-holds.json', 'datasets.json')]
    required = [path/'dist/teacher/cli.js', path/'prelude.js', path/'package.json',
                *(path/name for name in names)]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise ValueError('provider task runtime needs built CLI and pinned npm manifests and source-review tables; missing: '
                         + ', '.join(dict.fromkeys(missing)))
    # Small built files are cheaper to hash than repeatedly scanning the shared
    # append-only cache intended for multi-gigabyte checkpoint artifacts.
    result = {str(p.relative_to(path)): sha256_file_hex(p)
              for p in sorted((path/'dist').rglob('*')) if p.is_file()}
    result.update({name: sha256_file_hex(path/name)
                   for name in names})
    if frozen.is_file():
        declared = json.loads(frozen.read_text())['files']
        for name, digest in result.items():
            if not name.startswith('@dependency-') and name != 'frozen-runtime.json' and declared.get(name) != digest:
                raise ValueError(f'frozen task runtime changed its declared file: {name}')
    dependency_receipt = _dependency_receipt(path, result)
    if dependency_receipt is None:
        dependencies = _dependency_tree(path)
        result['@dependency-files'] = dependencies['files']
        result['@dependency-source-links'] = dependencies['links']
    else:
        result['@dependency-files'] = dependency_receipt['tree']['files']
        result['@dependency-source-links'] = dependency_receipt['source_links']
        result['@dependency-receipt-sha256'] = dependency_receipt['sha256']
    return result


def task_runtime_identity_files(identity):
    """Validate saved TaskEval identity closure; V1 stays historical, V2 binds dependencies."""
    if not isinstance(identity, dict) or not isinstance(identity.get('files'), list):
        raise ValueError('task runtime identity has no file closure')
    schema = identity.get('schema')
    if schema == 'natlang.task-eval-runtime-identity/1':
        return identity['files']
    if schema != 'natlang.task-eval-runtime-identity/2':
        raise ValueError('unknown task runtime identity schema')
    closure = identity.get('dependency_closure')
    if (not isinstance(closure, dict) or set(closure) != {'files', 'source_links'}
            or not isinstance(closure['files'], dict) or not isinstance(closure['source_links'], dict)):
        raise ValueError('task runtime identity dependency closure is malformed')
    dependencies = {}
    def valid_dependency_path(logical):
        parts = logical.split('/')
        if parts[0] == '..':
            return (len(parts) >= 2 and parts[:2] == ['..', 'node_modules']
                    and all(part not in ('', '.', '..') for part in parts[2:]))
        return (parts[0] in ('node_modules', 'vendor')
                and all(part not in ('', '.', '..') for part in parts[1:]))

    for logical, entry in closure['files'].items():
        if not isinstance(logical, str) or not valid_dependency_path(logical) or not isinstance(entry, dict):
            raise ValueError('task runtime dependency entry is malformed')
        if entry.get('kind') == 'file':
            if (type(entry.get('bytes')) is not int or entry['bytes'] < 0
                    or type(entry.get('mode')) is not int
                    or not isinstance(entry.get('sha256'), str)
                    or len(entry['sha256']) != 64
                    or any(character not in '0123456789abcdef' for character in entry['sha256'])):
                raise ValueError('task runtime dependency file binding is malformed')
            dependencies[logical] = (entry['bytes'], entry['sha256'])
        elif entry.get('kind') != 'directory' or type(entry.get('mode')) is not int:
            raise ValueError('task runtime dependency layout entry is malformed')
    for logical, entry in closure['source_links'].items():
        if (not isinstance(logical, str) or not valid_dependency_path(logical)
                or logical not in closure['files'] or not isinstance(entry, dict)
                or set(entry) != {'target', 'resolved'}
                or not isinstance(entry['target'], str) or not entry['target']
                or not isinstance(entry['resolved'], str) or not Path(entry['resolved']).is_absolute()):
            raise ValueError('task runtime dependency link identity is malformed')
    rows = {}
    for row in identity['files']:
        if not isinstance(row, dict) or not isinstance(row.get('path'), str):
            raise ValueError('task runtime identity file row is malformed')
        path = Path(row['path'])
        if (not path.is_absolute() or '..' in path.parts or str(path) != row['path']
                or type(row.get('bytes')) is not int or row['bytes'] < 0
                or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64
                or any(character not in '0123456789abcdef' for character in row['sha256'])):
            raise ValueError('task runtime identity file binding is malformed')
        logical = row.get('runtime_dependency')
        if logical is not None:
            if logical in rows or logical not in dependencies:
                raise ValueError('task runtime dependency file rows are duplicated or undeclared')
            rows[logical] = row
    if set(rows) != set(dependencies):
        raise ValueError('task runtime identity omits dependency file bindings')
    for logical, (size, digest) in dependencies.items():
        row = rows[logical]
        if (row.get('bytes') != size or row.get('sha256') != digest
                or type(row.get('bytes')) is not int):
            raise ValueError('task runtime dependency file binding differs from its closure')
    return identity['files']



def python_runtime_binding(package=None):
    """Bind the existing resident Python closure without a path-dependent identity."""
    from ..train.native_resident import code_inventory
    from .hashing import canonical_json_sha256_hex
    package = Path(package) if package is not None else Path(__file__).resolve().parents[1]
    files = code_inventory(package)
    if not files:
        raise ValueError('native actor Python package has no source inventory')
    return {'schema': 'natlang.python-runtime/1',
            'sha256': canonical_json_sha256_hex(files), 'files': len(files)}
