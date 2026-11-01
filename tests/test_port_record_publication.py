"""Exercise actual shard publication with real zstd and injected I/O failures."""
import os
import fcntl
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'scripts/neuralese_data/port_record_calls.sh'


def replay_fixture(tmp_path, *, fail_rows=False):
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    guard = repo / 'scripts/disk_guard.sh'
    guard.write_text('#!/bin/sh\nexit 0\n')
    guard.chmod(0o755)
    node = tmp_path / 'node'
    node.write_text('''#!/usr/bin/python3
import json, pathlib, sys
p = pathlib.Path(sys.argv[-1])
p.with_suffix('.turns.jsonl').write_text('{"action":"kept"}\\n')
p.with_suffix('.rows.jsonl').write_text('{"target":"kept"}\\n')
p.with_suffix('.summary.json').write_text(json.dumps({'cases':1}))
''')
    node.chmod(0o755)
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    compressor = binaries / 'zstd'
    compressor.write_text('''#!/bin/sh
if [ "${FAIL_ROWS:-}" = 1 ]; then
  for arg in "$@"; do case "$arg" in *.rows.jsonl) exit 42;; esac; done
fi
exec /usr/bin/zstd "$@"
''')
    compressor.chmod(0o755)
    out = tmp_path / 'out'
    (out / 'tmp').mkdir(parents=True)
    (out / 'logs').mkdir()
    dst = out / 'turns/family/00000'
    dst.parent.mkdir(parents=True)
    shard = tmp_path / 'family/00000.ir.jsonl'
    shard.parent.mkdir()
    shard.write_text('{}\n')
    # Load the real replay function without running the whole driver/signal trap.
    source = DRIVER.read_text()
    complete = source[source.index('shard_complete() {'):source.index('\nbuild() {')]
    function = complete + '\n' + source[source.index('replay() {'):source.index('\nexport -f shard_complete build replay')]
    shell = function + '\nreplay "$1"\n'
    env = {**os.environ, 'REPO': str(repo), 'OUT': str(out), 'NODE': str(node),
           'TS': str(tmp_path), 'RUNTIME': '{"dist_commit":"test"}',
           'PATH': str(binaries) + ':' + os.environ['PATH'], 'FAIL_ROWS': str(int(fail_rows))}
    return shell, env, shard, dst


def test_resume_overwrites_interrupted_compression(tmp_path):
    shell, env, shard, dst = replay_fixture(tmp_path)
    Path(str(dst) + '.turns.jsonl.zst.tmp').write_bytes(b'interrupted')
    Path(str(dst) + '.rows.jsonl.zst.tmp').write_bytes(b'')
    result = subprocess.run(['bash', '-c', shell, 'test', str(shard)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for stream in ('turns', 'rows'):
        decoded = subprocess.check_output(['/usr/bin/zstd', '-q', '-d', '-c', str(dst) + f'.{stream}.jsonl.zst'])
        assert b'kept' in decoded
    assert Path(str(dst) + '.summary.json').is_file()


def test_compression_failure_cannot_publish_completion(tmp_path):
    shell, env, shard, dst = replay_fixture(tmp_path, fail_rows=True)
    result = subprocess.run(['bash', '-c', shell, 'test', str(shard)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not Path(str(dst) + '.summary.json').exists()
    assert 'no completion summary committed' in result.stderr


def test_false_old_summary_is_replayed(tmp_path):
    shell, env, shard, dst = replay_fixture(tmp_path)
    Path(str(dst) + '.summary.json').write_text('{"cases":999}')
    result = subprocess.run(['bash', '-c', shell, 'test', str(shard)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '999' not in Path(str(dst) + '.summary.json').read_text()
    assert Path(str(dst) + '.turns.jsonl.zst').is_file()
    assert Path(str(dst) + '.rows.jsonl.zst').is_file()


def test_second_driver_cannot_clean_live_scratch(tmp_path):
    source, out = tmp_path / 'source', tmp_path / 'out'
    source.mkdir()
    (out / 'tmp').mkdir(parents=True)
    live = out / 'tmp/live-partial'
    live.write_text('valuable in-flight data')
    with (out / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(['bash', str(DRIVER), str(source), str(out)], capture_output=True, text=True)
    assert result.returncode == 75, result.stderr
    assert live.read_text() == 'valuable in-flight data'


def test_runtime_identity_changes_when_script_changes(tmp_path, monkeypatch):
    import importlib.util
    import sys
    sys.path.insert(0, str(ROOT / 'training/neuralese'))
    spec = importlib.util.spec_from_file_location('port_runtime_identity', ROOT / 'scripts/neuralese_data/port_runtime_identity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'task_runtime_inventory', lambda path: {'dist/cli.js': 'authenticated'})
    root = tmp_path / 'runtime'
    script = root / 'ts-host/scripts/inline-curriculum/replay.mjs'
    script.parent.mkdir(parents=True)
    script.write_text('first')
    (root / 'runtime-input-closure.json').write_text('{}')
    reservation = root / 'data/external_pilot/jeff-test-tasks.jsonl'
    reservation.parent.mkdir(parents=True)
    reservation.write_text('{}')
    node = tmp_path / 'node'
    node.write_text('node')
    first = module.identity(root, node)['identity_sha256']
    script.write_text('changed')
    assert module.identity(root, node)['identity_sha256'] != first


def test_predecessor_receipt_detects_changed_summary(tmp_path):
    import json
    root = tmp_path / 'old'
    d = root / 'turns/family'
    d.mkdir(parents=True)
    (root / 'runtime-page.json').write_text('{"dist_commit":"old"}')
    (d / '00000.summary.json').write_text('{"cases":1}')
    (d / '00000.turns.jsonl.zst').write_bytes(b'turns')
    (d / '00000.rows.jsonl.zst').write_bytes(b'rows')
    receipt = tmp_path / 'receipt.json'
    tool = ROOT / 'scripts/neuralese_data/port_predecessor_receipt.py'
    subprocess.run(['python3', str(tool), 'create', str(root), str(receipt)], check=True)
    subprocess.run(['python3', str(tool), 'verify', str(root), str(receipt)], check=True)
    (d / '00000.summary.json').write_text('{"cases":2}')
    assert subprocess.run(['python3', str(tool), 'verify', str(root), str(receipt)], capture_output=True).returncode != 0
    assert json.loads(receipt.read_text())['completed']
