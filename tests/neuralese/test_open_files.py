import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = """
import resource
from natlang_neuralese.common.open_files import raise_open_files_limit
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(256, hard), hard))
result = raise_open_files_limit()
print('none' if result is None else 'raised', resource.getrlimit(resource.RLIMIT_NOFILE)[0], hard)
"""


def _probe(**env):
    done = subprocess.run([sys.executable, '-c', PROBE], capture_output=True, text=True, check=True,
                          env={**os.environ, 'PYTHONPATH': str(ROOT / 'training/neuralese'), **env})
    return done.stdout.split(None, 2)


def test_the_soft_open_files_limit_rises_to_the_hard_one_and_can_opt_out():
    """The SGLang front holds two sockets per stream; a unit's soft limit of 1024 ran out under collection
    (`URLError: [Errno 24] Too many open files`, aei1-collect-v0001, 2026-11-24)."""
    import resource
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    _result, soft, _hard = _probe()
    expected = (1 << 20) if hard == resource.RLIM_INFINITY else min(hard, 1 << 20)
    assert int(soft) == max(min(256, hard), expected)
    result, soft, _hard = _probe(NATLANG_RAISE_NOFILE='0')
    assert result == 'none' and int(soft) == min(256, hard)


def test_every_python_server_raises_it_when_it_starts_serving():
    from natlang_neuralese.serve import http
    import inspect
    assert 'raise_open_files_limit()' in inspect.getsource(http.serve)
