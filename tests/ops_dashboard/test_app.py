"""The Tendriller application's own node tests (app/test): domain and reducers, the agent program's duties and live data feeds, the
inbox worker, the browser-side helpers and the pure page model, against the installed @tendriller/core and @chronolog/peer-session
packages. Skipped when node 24 or the app's node_modules are absent."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / 'scripts' / 'ops_dashboard' / 'app'
NODE24 = Path.home() / '.local/share/natlang-dev/node-v24.0.1-linux-arm64/bin/node'


def _node():
    return str(NODE24) if NODE24.exists() else shutil.which('node')


@pytest.mark.skipif(not (APP / 'node_modules' / '@tendriller').exists() or not _node(), reason='app packages or node missing')
def test_app_node_tests():
    done = subprocess.run([_node(), '--test'], cwd=APP, capture_output=True, text=True, timeout=120,
                          env={**os.environ, 'NODE_NO_WARNINGS': '1'})
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
