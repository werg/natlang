import importlib.util
import subprocess
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('serve_native_checkpoint',
    Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/serve_native_checkpoint.py')
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


def test_attestation_binds_interpreter_checkout_and_rejects_dirty_source(tmp_path):
    repo = tmp_path / 'fork'
    package = repo / 'python/sglang'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    subprocess.run(['git', 'init', '-q', str(repo)], check=True)
    subprocess.run(['git', '-C', str(repo), 'add', 'python/sglang/__init__.py'], check=True)
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=test', '-c', 'user.email=test@example.org',
                    'commit', '-qm', 'fixture'], check=True)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    python = tmp_path / 'python'
    python.write_text('#!/bin/sh\nprintf "%s\\n" ' + str(package / '__init__.py') + '\n')
    python.chmod(0o755)
    assert serve.attest_sglang_source(str(python), {'commit': commit})['commit'] == commit
    with pytest.raises(RuntimeError, match='differs from pinned'):
        serve.attest_sglang_source(str(python), {'commit': '0' * 40})
    (package / 'injected.py').write_text('')
    with pytest.raises(RuntimeError, match='uncommitted or untracked'):
        serve.attest_sglang_source(str(python), {'commit': commit})
