"""Cancel a canonical managed run while its CLI, server and recorder are alive."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


class CheckpointRequested(InterruptedError):
    """An owner interruption, distinct from a task failure."""


class CollectorCheckpoint:
    """One collector owner: stop admission, signal its main, let it retain/archive."""
    def __init__(self):
        self.requested = False
        self.child = None
        self.request_file = None
        self.previous = {}
        self.signalled = False

    def __enter__(self):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            self.previous[sig] = signal.signal(sig, self._request)
        return self

    def __exit__(self, *_):
        for sig, handler in self.previous.items():
            signal.signal(sig, handler)

    def _notify(self):
        if self.request_file is not None:
            temporary = self.request_file.with_suffix('.pending')
            temporary.write_text(json.dumps({'requestedAt': self.requested_at, 'signal': self.signal}) + '\n')
            temporary.replace(self.request_file)
        if self.child is not None and self.child.poll() is None and not self.signalled:
            self.signalled = True
            self.child.send_signal(signal.SIGTERM)

    def _request(self, signum, _frame):
        if self.requested:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        self.requested = True
        self.requested_at = time.time()
        self.signal = signal.Signals(signum).name
        self._notify()

    def run(self, argv, *, request_file=None, on_tick=None, **kwargs):
        if self.requested:
            return subprocess.CompletedProcess(argv, 130)
        self.request_file = Path(request_file) if request_file is not None else None
        self.signalled = False
        try:
            capture = kwargs.pop('capture_output', False)
            data = kwargs.pop('input', None)
            if capture:
                if 'stdout' in kwargs or 'stderr' in kwargs:
                    raise ValueError('capture_output cannot be combined with stdout/stderr')
                kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if data is not None:
                if 'stdin' in kwargs:
                    raise ValueError('input cannot be combined with stdin')
                kwargs['stdin'] = subprocess.PIPE
            self.child = subprocess.Popen(argv, **kwargs)
            if self.requested:
                self._notify()
            if on_tick is None:
                stdout, stderr = self.child.communicate(input=data)
            else:
                # The same owner can decide a post-collection continuation while
                # its collector still owns the last runtime. No second launcher.
                while True:
                    if not self.requested:
                        try:
                            on_tick()
                        except BaseException:
                            self._request(signal.SIGTERM, None)
                            self.child.communicate(input=data)
                            raise
                    try:
                        stdout, stderr = self.child.communicate(input=data, timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        data = None
            return subprocess.CompletedProcess(argv, self.child.returncode, stdout, stderr)
        finally:
            self.child = None
            self.request_file = None


def checkpoint_managed_run(prefix, artifacts, *, cwd, env, timeout=50):
    """Use the runner's authenticated cancellation and actual evidence exporter.

    The prefix addresses one exact instance/workspace. A cancelled incomplete
    case remains unfinished; callers must not turn its packet into a failed
    goal or an attempt. Completed packets retain their original verdicts.
    """
    artifacts = Path(artifacts)
    runs = sorted(artifacts.glob('st_*/run.json'), key=lambda p: p.stat().st_mtime)
    record = {'requestedAt': time.time(), 'kind': 'owned-runtime-checkpoint', 'runs': []}
    deadline = time.monotonic() + timeout
    for path in runs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        run = json.loads(path.read_text())
        run_id = run['runId']
        if run_id != path.parent.name:
            raise ValueError('Managed run metadata does not identify its own directory')
        entry = {'runId': run_id, 'artifactDir': str(path.parent)}
        record['runs'].append(entry)
        with (path.parent / 'checkpoint-cancellation.log').open('ab') as log:
            try:
                result = subprocess.run([*prefix, 'cancel', run_id, '--json'], cwd=cwd, env=env,
                                        stdout=log, stderr=subprocess.STDOUT, timeout=remaining)
                entry['exitCode'] = result.returncode
            except subprocess.TimeoutExpired:
                entry['timedOut'] = True
        entry['fullPackets'] = len(list(path.parent.glob('trajectory-*-full.json.gz'))) + len(list((path.parent/'failure-evidence').glob('trajectory-*-full.json.gz')))
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / 'checkpoint.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def cancel_and_join(child, prefix, artifacts, *, cwd, env):
    """Bounded cooperative retirement before any process-level termination."""
    try:
        checkpoint_managed_run(prefix, artifacts, cwd=cwd, env=env)
    finally:
        if child.poll() is None:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=2)
