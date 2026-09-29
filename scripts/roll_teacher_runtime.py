"""Move exact queue supervisors after their current journaled case finishes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_bonsai_queue import partial_metrics

def queue_exhausted(entries, rows):
    """Every attempt, including a journaled rejection/timeout, has reached its boundary."""
    done = {row['key'] for row in rows if row['event'] == 'finish'}
    return not (set(entries) - done)


def emit(event, worker, **fields):
    data = dict(event=event, worker=worker['name'], time=time.time(), **fields)
    with AUDIT.open('a') as stream:
        stream.write(json.dumps(data) + '\n')
    print(json.dumps(data), flush=True)
    STATE.write_text(json.dumps(workers, indent=2) + '\n')


def process_live(pid):
    path = Path(f'/proc/{pid}/stat')
    try:
        return path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def children(pid):
    try:
        return [int(value) for value in Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except FileNotFoundError:
        return []


def events(worker):
    rows = []
    for line in Path(worker['journal']).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A concurrent append can expose its final, incomplete line; retry next tick.
            break
    return rows



def main():
    global ROOT, RUNTIME, workers, AUDIT, STATE
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    ROOT = Path(config['root']).resolve()
    os.chdir(ROOT)
    RUNTIME = Path(config['runtime']).resolve()
    workers = config['workers']
    AUDIT = Path(config['audit'])
    STATE = Path(config['state'])
    from scripts.freeze_training_runtime import tree_identity
    assert tree_identity(RUNTIME) == json.loads((RUNTIME / 'frozen-runtime.json').read_text())['files']

    for worker in workers:
        rows = events(worker)
        worker['old_entries'] = {e['key']: e for e in map(json.loads, Path(worker['old_queue']).read_text().splitlines())}
        if not process_live(worker['old_pid']):
            if not queue_exhausted(worker['old_entries'], rows):
                raise RuntimeError(f"old {worker['name']} supervisor disappeared with unfinished queue entries; inspect before resuming")
            worker['old_children'] = []
            worker['phase'] = 'stopping'
            worker['stop_time'] = time.monotonic()
            emit('old_queue_exhausted', worker, supervisor_pid=worker['old_pid'])
            continue
        try:
            command = Path(f"/proc/{worker['old_pid']}/cmdline").read_bytes().decode().replace('\0', ' ')
        except FileNotFoundError:
            if not queue_exhausted(worker['old_entries'], events(worker)):
                raise RuntimeError(f"old {worker['name']} supervisor disappeared with unfinished queue entries")
            worker.update(old_children=[], phase='stopping', stop_time=time.monotonic())
            emit('old_queue_exhausted', worker, supervisor_pid=worker['old_pid'])
            continue
        if 'scripts/run_bonsai_queue.py' not in command or worker['old_queue'] not in command:
            raise RuntimeError(f"unexpected process for {worker['name']}")
        worker['active_key'] = next(row['key'] for row in reversed(rows) if row['event'] == 'start')
        worker['phase'] = 'waiting'
        emit('await_case_boundary', worker, supervisor_pid=worker['old_pid'], key=worker['active_key'])

    for tick in range(72000):
        for worker in workers:
            if worker['phase'] == 'waiting':
                finished = next((row for row in reversed(events(worker)) if row['event'] == 'finish' and
                                 row['key'] == worker['active_key']), None)
                if finished:
                    latest = next(row for row in reversed(events(worker)) if row['event'] == 'start')
                    # Let a next case with saved model progress finish too; do not interrupt it.
                    metrics = partial_metrics(worker['old_entries'][latest['key']])
                    if latest['key'] != worker['active_key'] and metrics['fresh_model_replies']:
                        worker['active_key'] = latest['key']
                        emit('next_case_already_progressing', worker, key=latest['key'], metrics=metrics)
                        continue
                    if latest['key'] != worker['active_key']:
                        emit('preserve_unanswered_next_case_checkpoint', worker, key=latest['key'], metrics=metrics)
                    worker['old_children'] = children(worker['old_pid'])
                    if process_live(worker['old_pid']):
                        os.kill(worker['old_pid'], signal.SIGTERM)
                    worker['phase'] = 'stopping'
                    worker['stop_time'] = time.monotonic()
                    emit('case_finished_stop_old', worker, completion=finished, exact_pid=worker['old_pid'])
                elif not process_live(worker['old_pid']):
                    raise RuntimeError(f"old {worker['name']} supervisor exited before a journaled boundary")
            if worker['phase'] == 'stopping':
                if process_live(worker['old_pid']) or any(process_live(pid) for pid in worker['old_children']):
                    if time.monotonic() - worker['stop_time'] > 30:
                        raise RuntimeError(f"{worker['name']} did not stop; refusing overlapping worker")
                    continue
                command = [sys.executable, str(ROOT / 'scripts/run_bonsai_queue.py'),
                           worker['queue'], worker['journal'], '--runtime', str(RUNTIME), *worker['extra']]
                with Path(worker['log']).open('a') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             cwd=ROOT, start_new_session=True)
                worker['new_pid'] = child.pid
                worker['phase'] = 'running'
                emit('new_supervisor_started', worker, supervisor_pid=child.pid, queue=worker['queue'],
                     journal=worker['journal'], runtime=str(RUNTIME))
        if all(worker['phase'] == 'running' for worker in workers):
            STATE.write_text(json.dumps(workers, indent=2) + '\n')
            print('Both workers migrated at journaled boundaries; no overlapping Luna collectors.', flush=True)
            break
        time.sleep(0.05)
    else:
        raise RuntimeError('Boundary migration timed out; existing supervisors were left running')


if __name__ == '__main__':
    main()
