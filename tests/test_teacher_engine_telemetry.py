import importlib.util
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).parents[1] / 'scripts/coreweave/teacher_engine_telemetry.py'
spec = importlib.util.spec_from_file_location('telemetry', SCRIPT)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)


def test_memory_and_pin_counters_keep_units(tmp_path):
    p = tmp_path / 'meminfo'
    p.write_text('MemAvailable: 100 kB\nMlocked: 12 kB\nIgnored: 500 kB\n')
    assert t.counters(p, t.MEMINFO, True) == {'MemAvailable': 102400, 'Mlocked': 12288}
    p.write_text('1234-5678 rw-p 00000000 00:00 0 [rollup]\nRss: 7 kB\nLocked: 5 kB\n')
    assert t.counters(p, {'Rss', 'Locked'}, True) == {'Rss': 7168, 'Locked': 5120}


def test_log_deltas_bound_bytes_and_report_loss_and_replacement(tmp_path):
    p = tmp_path / 'engine.log'
    p.write_bytes(b'abcdefgh')
    offsets = {}
    first = t.log_delta(p, offsets, 4)
    assert (first['text'], first['omitted_bytes'], first['from'], first['to']) == ('efgh', 4, 4, 8)
    assert t.log_delta(p, offsets, 4)['text'] == ''
    with p.open('ab') as stream:
        stream.write(b'ij')
    assert t.log_delta(p, offsets, 4)['text'] == 'ij'
    new = tmp_path / 'replacement'
    new.write_bytes(b'new')
    new.replace(p)
    assert t.log_delta(p, offsets, 4)['text'] == 'new'


def test_append_keeps_prior_samples_durable(tmp_path, monkeypatch):
    synced = []
    monkeypatch.setattr(t.os, 'fsync', lambda fd: synced.append(fd))
    p = tmp_path / 'history.jsonl'
    with p.open('x') as stream:
        t.append_record(stream, {'boot': 'old', 'pin': 12})
        t.append_record(stream, {'boot': 'new', 'pin': 0})
    assert len(synced) == 2
    assert [json.loads(line) for line in p.read_text().splitlines()] == [{'boot': 'old', 'pin': 12}, {'boot': 'new', 'pin': 0}]


def test_actual_stream_exits_on_signal_and_wrong_boot():
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    child = subprocess.Popen([sys.executable, '-u', str(SCRIPT), '--boot-id', boot, '--seconds', '30', '--interval', '.1'], stdout=subprocess.PIPE, text=True)
    try:
        record = json.loads(child.stdout.readline())
        assert record['boot_id'] == boot and record['meminfo_bytes']['MemAvailable'] > 0
        assert record['sampler']['pid'] == child.pid
        child.terminate()
        assert child.wait(timeout=5) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    wrong = subprocess.run([sys.executable, str(SCRIPT), '--boot-id', 'wrong', '--seconds', '.01'], capture_output=True, text=True)
    assert wrong.returncode != 0 and wrong.stdout == ''


def test_sdk_fragment_stream_is_durable_before_container_reset(tmp_path):
    good = json.dumps({'schema': 'natlang.teacher-engine-telemetry/1', 'boot_id': 'old', 'meminfo_bytes': {'Mlocked': 55}}) + '\n'
    class Process:
        @property
        def stdout(self):
            yield good[:13]
            yield good[13:]
            raise ConnectionError('provider reset')
    target = tmp_path / 'observer.jsonl'
    import pytest
    with pytest.raises(ConnectionError):
        t.mirror(Process(), target, 'old')
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert rows[0]['sample']['meminfo_bytes']['Mlocked'] == 55
    assert rows[1]['error_class'] == 'ConnectionError' and rows[1]['samples'] == 1
    with pytest.raises(FileExistsError):
        t.mirror(Process(), target, 'old')


def test_sdk_reset_boot_never_overwrites_old_history(tmp_path):
    import pytest
    good = {'schema': 'natlang.teacher-engine-telemetry/1', 'boot_id': 'old'}
    class Process:
        stdout = [json.dumps(good) + '\n', json.dumps({**good, 'boot_id': 'new'}) + '\n']
    target = tmp_path / 'observer.jsonl'
    with pytest.raises(ValueError, match='identity'):
        t.mirror(Process(), target, 'old')
    rows = [json.loads(line) for line in target.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['sample']['boot_id'] == 'old' and rows[1]['samples'] == 1


def test_sdk_bulk_chunk_has_many_valid_bounded_records(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(t.os, 'fsync', lambda fd: None)
    record = {'schema': 'natlang.teacher-engine-telemetry/1', 'boot_id': 'same', 'log': 'x' * 1024}
    class Process:
        stdout = [(json.dumps(record) + '\n') * 1100]
        def result(self, timeout):
            return SimpleNamespace(returncode=0)
    assert t.mirror(Process(), tmp_path / 'bulk.jsonl', 'same') == 1100


def test_process_state_change_keeps_same_live_birth(tmp_path, monkeypatch):
    proc = tmp_path / 'proc'
    (proc / 'sys/kernel/random').mkdir(parents=True)
    (proc / 'sys/kernel/random/boot_id').write_text('boot')
    (proc / 'meminfo').write_text('MemAvailable: 10 kB\n')
    (proc / 'vmstat').write_text('nr_foll_pin_acquired 123\n')
    child = proc / '123'
    child.mkdir()
    (child / 'cmdline').write_bytes(b'python\0')
    (child / 'comm').write_bytes(b'VLLM::EngineCore')
    (child / 'status').write_text('VmPin: 7 kB\n')
    (child / 'smaps_rollup').write_text('Locked: 3 kB\n')
    seen = []
    def identity(path):
        if path != child:
            return None
        seen.append(1)
        return {'pid': 123, 'ppid': 1, 'start_ticks': 55, 'state': 'R' if len(seen) == 1 else 'S'}
    monkeypatch.setattr(t, 'process_identity', identity)
    result = t.snapshot(proc=proc, cgroup=tmp_path / 'missing', gpu=False)
    assert result['processes'][0]['pid'] == 123
    assert result['processes'][0]['status_bytes']['VmPin'] == 7168
