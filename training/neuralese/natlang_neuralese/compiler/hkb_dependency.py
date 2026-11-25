"""The compiler's declared dependency on hindsight-kb (its own repository).

``hindsight_kb`` is imported as an installed package when one is installed; otherwise from the checkout at the logical
root ``hindsight_kb`` (``common.paths``: env ``NATLANG_HINDSIGHT_KB``, ``~/.config/natlang/machine.toml``, then the
default). A pin or Colab VM points the setting at its own checkout. Only ``require()`` touches ``sys.path`` (it adds
``<checkout>/src`` once), and it names the setting when the package is absent.
Every run that uses it records ``provenance()`` (the hindsight-kb commit and dirty flag) next to natlang's own.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

SETTING = 'NATLANG_HINDSIGHT_KB'


def checkout() -> Path:
    from ..common.paths import root
    return root('hindsight_kb')


def require():
    """Import and return the ``hindsight_kb`` package, adding ``<checkout>/src`` to ``sys.path`` if needed."""
    try:
        return importlib.import_module('hindsight_kb')
    except ModuleNotFoundError as error:
        if error.name != 'hindsight_kb':
            raise
    src = checkout() / 'src'
    if not (src / 'hindsight_kb' / '__init__.py').is_file():
        raise ModuleNotFoundError(
            f'the natlang compiler needs hindsight-kb: install it, or set {SETTING} (or `hindsight_kb` in '
            f'~/.config/natlang/machine.toml) to its checkout; {src} has no hindsight_kb package', name='hindsight_kb')
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return importlib.import_module('hindsight_kb')


def provenance() -> dict:
    """``{'path', 'git_commit', 'dirty'}`` of the hindsight-kb in use (a package without a git checkout: commit None)."""
    package = Path(require().__file__).resolve().parent
    top = package.parent.parent
    def git(*args):
        return subprocess.run(['git', '-C', str(top), *args], capture_output=True, text=True, check=False)
    head = git('rev-parse', 'HEAD')
    if head.returncode != 0:
        return {'schema': 'natlang.hindsight-kb-provenance/1', 'path': str(package), 'git_commit': None, 'dirty': None}
    status = git('status', '--porcelain', '--untracked-files=no')
    return {'schema': 'natlang.hindsight-kb-provenance/1', 'path': str(top), 'git_commit': head.stdout.strip(),
            'dirty': bool(status.stdout.strip())}
