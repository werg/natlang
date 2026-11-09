import importlib.util
import subprocess
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('serve_native_checkpoint',
    Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/serve_native_checkpoint.py')
serve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serve)


def test_attestation_records_the_actual_fork_commit_and_warns_never_refuses(tmp_path, capsys):
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
    record = serve.attest_sglang_source(str(python), {'commit': commit})
    assert record['commit'] == commit and record['matches_declared'] is True and record['dirty'] is False
    assert capsys.readouterr().err == ''
    other = serve.attest_sglang_source(str(python), {'commit': '0' * 40})   # a different fork head still launches
    assert other['commit'] == commit and other['declared_commit'] == '0' * 40 and other['matches_declared'] is False
    assert 'sglang_fork_differs_from_declared' in capsys.readouterr().err
    (package / 'injected.py').write_text('')
    dirty = serve.attest_sglang_source(str(python), {'commit': commit})
    assert dirty['dirty'] is True and dirty['commit'] == commit
