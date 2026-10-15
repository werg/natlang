"""Exact built task runtime, npm and protection-file identity shared by providers and eval."""
from pathlib import Path
from .hashing import sha256_file_hex_cached


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
    result = {str(p.relative_to(path)): sha256_file_hex_cached(p)
              for p in sorted((path/'dist').rglob('*')) if p.is_file()}
    result.update({name: sha256_file_hex_cached(path/name)
                   for name in names})
    return result

