#!/usr/bin/env python3
"""Make owned rsync staging files writable; leave published artifacts intact."""
import argparse
import json
import os
from pathlib import Path
import stat


def prepare(root):
    requested = Path(root)
    if requested.is_symlink():
        raise ValueError(f'staging root must be a real directory: {requested}')
    root = requested.resolve(strict=True)
    changed = 0
    for current, directories, _ in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories
                          if name not in {'.git', '.sync-history', 'pull-revisions'}
                          and not (Path(current) / name).is_symlink()]
        for name in list(directories):
            if name not in {'.natlang-priority-partial', '.natlang-sync-partial'}:
                continue
            directory = Path(current) / name
            directories.remove(name)
            for path in directory.iterdir():
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode):
                    continue
                if info.st_uid != os.getuid() or info.st_nlink != 1:
                    raise ValueError(f'staging file is not exclusively owned: {path}')
                if not info.st_mode & stat.S_IWUSR:
                    path.chmod(stat.S_IMODE(info.st_mode) | stat.S_IWUSR)
                    changed += 1
    return changed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps({'writable_staging_files': prepare(args.root)}))
