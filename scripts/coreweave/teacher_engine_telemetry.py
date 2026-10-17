"""Bounded Linux telemetry; one stream, no model construction or recovery policy."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

MEMINFO = {'MemTotal', 'MemFree', 'MemAvailable', 'Buffers', 'AnonPages', 'Slab', 'SReclaimable', 'Mlocked', 'Unevictable', 'Shmem', 'Cached', 'Dirty', 'Writeback', 'SwapFree'}
PROC_STATUS = {'VmRSS', 'VmLck', 'VmPin', 'RssAnon', 'RssFile', 'RssShmem', 'VmSwap'}


def counters(path: Path, selected=None, colon=False):
    result = {}
    try:
        for line in path.read_text().splitlines():
            if colon and ':' not in line:
                continue
            key, value = line.split(':' if colon else None, 1)
            if selected is None or key in selected:
                parts = value.split()
                result[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == 'kB' else 1)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return result


def process_identity(path: Path):
    try:
        raw = (path / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        return {'pid': int(path.name), 'state': fields[0], 'ppid': int(fields[1]), 'start_ticks': int(fields[19])}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def log_delta(path: Path, offsets: dict, limit=4096):
    try:
        with path.open('rb') as stream:
            stat = os.fstat(stream.fileno())
            previous = offsets.get(str(path))
            offset = previous[2] if previous and previous[:2] == (stat.st_dev, stat.st_ino) and previous[2] <= stat.st_size else 0
            begin = max(offset, stat.st_size - limit)
            stream.seek(begin)
            raw = stream.read(limit)
            offsets[str(path)] = (stat.st_dev, stat.st_ino, begin + len(raw))
            return {'path': str(path), 'device': stat.st_dev, 'inode': stat.st_ino, 'from': begin, 'to': begin + len(raw), 'omitted_bytes': begin - offset, 'text': raw.decode('utf-8', 'replace')}
    except (FileNotFoundError, PermissionError):
        return {'path': str(path), 'missing': True}


def snapshot(proc=Path('/proc'), cgroup=Path('/sys/fs/cgroup'), logs=(), offsets=None, gpu=True, smaps=True):
    offsets = offsets if offsets is not None else {}
    boot = (proc / 'sys/kernel/random/boot_id').read_text().strip()
    result = {'schema': 'natlang.teacher-engine-telemetry/1', 'at': time.time(), 'boot_id': boot,
              'sampler': process_identity(proc / str(os.getpid())), 'meminfo_bytes': counters(proc / 'meminfo', MEMINFO, True),
              'vmstat': counters(proc / 'vmstat', {'nr_foll_pin_acquired', 'nr_foll_pin_released', 'nr_mlock', 'nr_unevictable'}),
              'cgroup': {}, 'processes': [], 'logs': [log_delta(Path(p), offsets) for p in logs]}
    for name in ('memory.current', 'memory.max', 'memory.peak', 'memory.stat', 'memory.events'):
        try:
            raw = (cgroup / name).read_text().strip()
            result['cgroup'][name] = counters(cgroup / name) if name in ('memory.stat', 'memory.events') else (int(raw) if raw != 'max' else raw)
        except (FileNotFoundError, PermissionError):
            result['cgroup'][name] = None
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path / 'cmdline').read_bytes()
            comm = (path / 'comm').read_bytes()
            if not any(tag in argv + comm for tag in (b'EngineCore', b'vllm.entrypoints', b'vllm serve', b'teacher_engine_guard.py')):
                continue
            before = process_identity(path)
            status = counters(path / 'status', PROC_STATUS, True)
            rollup = counters(path / 'smaps_rollup', {'Rss', 'Pss', 'Locked', 'Private_Dirty', 'Anonymous'}, True) if smaps else None
            after = process_identity(path)
            if before and after and all(before[key] == after[key] for key in ('pid', 'ppid', 'start_ticks')):
                result['processes'].append({**before, 'status_bytes': status, 'smaps_bytes': rollup})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    if gpu:
        try:
            read = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
            result['gpu'] = {'returncode': read.returncode, 'csv': read.stdout.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            result['gpu'] = {'error': type(error).__name__}
    return result


def append_record(stream, record):
    """Durable owner-host receipt before the next remote sample is requested."""
    stream.write(json.dumps(record, separators=(',', ':')) + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def mirror(process, target: Path, boot_id: str):
    """Consume SDK chunks once; preserve every complete record outside the VM."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as output:
        directory = os.open(target.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        pending = ''
        count = 0
        try:
            for chunk in process.stdout:
                pending += chunk
                while '\n' in pending:
                    line, pending = pending.split('\n', 1)
                    if not line:
                        continue
                    if len(line) > 1 << 20:
                        raise ValueError('telemetry record exceeds the declared bounded stream')
                    record = json.loads(line)
                    if record.get('schema') != 'natlang.teacher-engine-telemetry/1' or record.get('boot_id') != boot_id:
                        raise ValueError('telemetry source identity changed')
                    append_record(output, {'received_at': time.time(), 'sample': record})
                    count += 1
                if len(pending) > 1 << 20:
                    raise ValueError('telemetry record exceeds the declared bounded stream')
            if pending:
                raise ValueError('telemetry stream ended with a partial record')
            result = process.result(timeout=30)
            append_record(output, {'terminal_at': time.time(), 'returncode': result.returncode, 'samples': count, 'stderr_tail': (getattr(result, 'stderr', '') or '')[-4096:]})
            if result.returncode or not count:
                raise RuntimeError('telemetry sampler did not complete successfully')
        except BaseException as error:
            append_record(output, {'error_at': time.time(), 'error_class': type(error).__name__, 'samples': count})
            raise
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--boot-id', required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--interval', type=float, default=10)
    parser.add_argument('--log', action='append', default=[])
    parser.add_argument('--fast-host-only', action='store_true', help='Keep the periodic host stream independent of smaps mmap locks and GPU-driver queries.')
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error('duration and interval must be positive')
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    deadline = time.monotonic() + args.seconds
    offsets = {}
    while not stopped and time.monotonic() < deadline:
        start = time.monotonic()
        record = snapshot(logs=args.log, offsets=offsets, gpu=not args.fast_host_only, smaps=not args.fast_host_only)
        record['sampling_scope'] = 'host-status-and-logs' if args.fast_host_only else 'host-smaps-gpu-and-logs'
        if record['boot_id'] != args.boot_id:
            raise RuntimeError('observer boot differs from its declared container')
        print(json.dumps(record, separators=(',', ':')), flush=True)
        while not stopped and time.monotonic() < min(deadline, start + args.interval):
            time.sleep(min(.2, max(0, min(deadline, start + args.interval) - time.monotonic())))


if __name__ == '__main__':
    main()
