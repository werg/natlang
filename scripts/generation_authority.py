"""Short shared lock for generation authority read/modify/write operations."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from pathlib import Path


@contextmanager
def authority_lock(path):
    with path.with_name(path.name + '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _process_argv(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    except (OSError, UnicodeError):
        return []


def verify_worker_process(worker, *, process_argv=None):
    """Resolve a recorded worker PID only when its command matches the binding.

    Successor collectors historically wrote ``worker_pid``/``process_pid``
    rather than ``pid``. Accept those aliases, but never trust a PID by itself:
    require its live argv to match the recorded argv (or the Node child suffix
    of a recorded timeout wrapper).
    """
    process_argv = process_argv or _process_argv
    expected = worker.get('argv')
    if not isinstance(expected, list) or not expected or not all(isinstance(x, str) for x in expected):
        return None

    def matches(actual):
        if actual == expected:
            return True
        # timeout(1) wraps the reviewed Node CLI. Linux /proc exposes the
        # wrapper and child separately, so the child may match the suffix.
        if expected[0].endswith('/timeout'):
            try:
                child = expected.index('/usr/bin/node')
            except ValueError:
                return False
            return actual == expected[child:]
        return False

    seen = set()
    for field in ('process_pid', 'worker_pid', 'pid', 'supervisor_pid'):
        pid = worker.get(field)
        if type(pid) is not int or pid <= 0 or pid in seen:
            continue
        seen.add(pid)
        actual = process_argv(pid)
        if actual and matches(actual):
            return {'pid': pid, 'field': field, 'argv': actual}
    return None


def observe_additional_teacher_processes(teacher, *, observed_at=None, process_argv=None):
    """Add verified live-process observations without changing lifecycle state."""
    observed_at = observed_at or datetime.now(timezone.utc).isoformat()
    workers = teacher.get('workers', [])
    if not isinstance(workers, list):
        return {'actual_live_workers': 0, 'active_processes_unverified': 0}
    active_statuses = {'running', 'active', 'starting'}
    live = 0
    unverified = 0
    for worker in workers:
        if not isinstance(worker, dict):
            continue
        match = verify_worker_process(worker, process_argv=process_argv)
        worker['process_observed_at'] = observed_at
        worker['process_running'] = match is not None
        if match is not None:
            worker['observed_process_pid'] = match['pid']
            worker['observed_process_pid_field'] = match['field']
        else:
            worker.pop('observed_process_pid', None)
            worker.pop('observed_process_pid_field', None)
        if worker.get('status') in active_statuses:
            if match is not None:
                live += 1
            else:
                unverified += 1
    teacher['actual_live_workers'] = live
    teacher['unverified_active_workers'] = unverified
    teacher['process_observed_at'] = observed_at
    return {'actual_live_workers': live, 'active_processes_unverified': unverified}


def reconcile_luna_authority(state, *, observed_at=None, process_argv=None):
    """Derive both Luna views from verified queue bindings, retaining history.

    Call while holding authority_lock. Target concurrency is unchanged. A PID
    alone is insufficient: it can be reused after an old campaign completes.
    """
    process_argv = process_argv or _process_argv
    observed_at = observed_at or datetime.now(timezone.utc).isoformat()
    teacher = state.setdefault('additional_teachers', {}).setdefault('luna', {})
    bindings = {}
    for worker in [*state.get('luna_workers', []), *teacher.get('workers', [])]:
        if not isinstance(worker, dict):
            continue
        key = (worker.get('pid'), worker.get('queue'), worker.get('journal'))
        bindings.setdefault(key, dict(worker))
    live = []
    for worker in bindings.values():
        pid = worker.get('pid')
        argv = process_argv(pid) if type(pid) is int and pid > 0 else []
        def option(name):
            try:
                return argv[argv.index(name) + 1]
            except (ValueError, IndexError):
                return None
        running = (len(argv) >= 4 and argv[1].endswith('/run_bonsai_queue.py') and
                   argv[2] == worker.get('queue') and argv[3] == worker.get('journal') and
                   option('--provider') == 'openai-codex' and option('--model-id') == 'gpt-6-luna' and
                   (not worker.get('runtime') or option('--runtime') == worker['runtime']))
        worker['process_observed_at'] = observed_at
        worker['process_running'] = running
        if running:
            worker['status'] = 'running'
            live.append(worker)
        elif worker.get('status') in ('running', 'active', 'starting'):
            worker['last_registered_status'] = worker['status']
            worker['status'] = 'not_running'
    state['luna_workers'] = list(bindings.values())
    state['luna_actual_live_workers'] = len(live)
    teacher.update({'workers': live, 'actual_live_workers': len(live),
                    'active_campaigns': sorted({str(worker['campaign']) for worker in live if worker.get('campaign')}),
                    'status': 'running' if live else 'idle',
                    'state': 'verified_queue_workers_running' if live else 'no_verified_queue_workers_running',
                    'observed_at': observed_at})
    return {'actual_live_workers': len(live), 'active_campaigns': teacher['active_campaigns']}
