#!/usr/bin/env python3
"""Make an isolated physical copy of a runtime tree before installation/build work."""
import argparse
import os
from pathlib import Path
import shutil


def clone_runtime_tree(source: Path, destination: Path) -> Path:
    """Copy a tree without shared writable file inodes or external symlinks.

    This is intended for disposable runtime/build candidates that may run npm
    or another package manager. It preserves relative symlinks only when their
    resolved targets remain inside the copied tree. It refuses an existing
    destination so a prior snapshot can never be silently overwritten.
    """
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve(strict=False)
    if not source.is_dir():
        raise ValueError("runtime source must be a directory")
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("runtime source and destination must not contain one another")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Claim the target name exclusively before copying. If another process
    # wins the race after the preflight check, mkdir raises and we never clean
    # up or overwrite that process's directory.
    destination.mkdir()
    try:
        shutil.copytree(source, destination, symlinks=True, copy_function=shutil.copy2,
                        dirs_exist_ok=True)
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            copied = destination / relative
            if path.is_symlink():
                target = os.readlink(path)
                resolved_source_target = (path.parent / target).resolve(strict=False)
                resolved_copy_target = (copied.parent / target).resolve(strict=False)
                if not resolved_source_target.is_relative_to(source):
                    raise ValueError(f"external symlink in runtime source: {relative} -> {target}")
                if not resolved_copy_target.is_relative_to(destination.resolve()):
                    raise ValueError(f"copied symlink escapes isolated runtime: {relative} -> {target}")
                if not copied.is_symlink() or os.readlink(copied) != target:
                    raise ValueError(f"symlink target changed while copying: {relative}")
            elif path.is_file():
                if copied.is_symlink() or not copied.is_file():
                    raise ValueError(f"file type changed while copying: {relative}")
                original_stat, copied_stat = path.stat(), copied.stat()
                if (original_stat.st_dev, original_stat.st_ino) == (copied_stat.st_dev, copied_stat.st_ino):
                    raise ValueError(f"runtime clone shares a writable inode: {relative}")
        return destination
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(clone_runtime_tree(args.source, args.destination))


if __name__ == "__main__":
    main()
