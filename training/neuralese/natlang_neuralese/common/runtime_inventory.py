"""Exact built task runtime, npm and protection-file identity shared by providers and eval."""
from pathlib import Path
from .hashing import sha256_file_hex


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
        destination = directory / canonical_json_sha256_hex(inventory)
        runtime = destination / 'ts-host'
        if destination.exists():
            if task_runtime_inventory(runtime) != inventory:
                raise ValueError('prepared task runtime changed')
            return runtime
        pending = Path(tempfile.mkdtemp(prefix='.preparing-', dir=directory))
        try:
            target = pending / 'ts-host'
            for name in inventory:
                copied = target / name
                copied.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / name, copied)
            # Node resolves packages both beside ts-host and in its parent.
            # Physical copies also isolate installs; optional vendor assets serve
            # Python folders. These are existing dependencies, never rebuilt here.
            for name in ('node_modules', '../node_modules', 'vendor'):
                if (source / name).is_dir():
                    shutil.copytree(source / name, target / name, symlinks=False)
            if (task_runtime_inventory(source) != inventory
                    or task_runtime_inventory(target) != inventory):
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


def task_runtime_inventory(path):
    path = Path(path)
    names = ['prelude.js', 'package.json', '../package.json', '../package-lock.json',
             '../training/source-reviews/holds.jsonl',
             '../training/source-reviews/family-holds.json',
             '../training/source-reviews/datasets.json']
    required = [path/'dist/teacher/cli.js', path/'prelude.js', path/'package.json',
                *(path/name for name in names)]
    if not all(item.is_file() for item in required):
        raise ValueError('provider task runtime needs built CLI and pinned npm manifests')
    # Small built files are cheaper to hash than repeatedly scanning the shared
    # append-only cache intended for multi-gigabyte checkpoint artifacts.
    result = {str(p.relative_to(path)): sha256_file_hex(p)
              for p in sorted((path/'dist').rglob('*')) if p.is_file()}
    result.update({name: sha256_file_hex(path/name)
                   for name in names})
    return result



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
