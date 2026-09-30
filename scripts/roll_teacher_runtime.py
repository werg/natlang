"""Move exact queue supervisors after their current journaled case finishes."""
import json
import hashlib
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
    for line in Path(worker.get('old_journal', worker['journal'])).read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if 'old_entries' not in worker or row.get('key') in worker['old_entries']:
                rows.append(row)
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
    predecessor = config.get('predecessor_rollout_state')
    if predecessor:
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            try:
                prior = json.loads(Path(predecessor).read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                prior = []
            if prior and all(item.get('phase') == 'running' for item in prior):
                by_name = {item['name']: item for item in prior}
                for worker in workers:
                    previous = by_name[worker['name']]
                    if previous['queue'] != worker['old_queue']:
                        raise RuntimeError('Predecessor queue does not match the expected old queue')
                    worker['old_pid'] = previous['new_pid']
                break
            time.sleep(0.5)
        else:
            raise RuntimeError('Predecessor rollout did not finish; no supervisors were signaled')
    if config.get('journal_sources'):
        if not config.get('restart_barrier'):
            raise RuntimeError('Journal migration requires a coordinated stop barrier')
        targets = [Path(worker['journal']).resolve() for worker in workers]
        if len(set(targets)) != len(targets) or any(path.exists() for path in targets):
            raise RuntimeError('Migrated journals must be distinct, new files')
        for source in config['journal_sources']:
            if not Path(source).is_file():
                raise RuntimeError(f'Missing journal source: {source}')

    for worker in workers:
        worker['old_entries'] = {e['key']: e for e in map(json.loads, Path(worker['old_queue']).read_text().splitlines())}
        rows = events(worker)
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

    journals_prepared = not config.get('journal_sources')
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
                worker['phase'] = 'ready'
                emit('old_supervisor_stopped', worker, exact_pid=worker['old_pid'])
        ready_to_restart = not config.get('restart_barrier') or all(
            worker['phase'] in ('ready', 'running') for worker in workers)
        if ready_to_restart and not journals_prepared:
            rows, sources = [], []
            for source in config['journal_sources']:
                raw = Path(source).read_bytes()
                # Parse every completed line after all old writers have stopped.
                rows.extend(json.loads(line) for line in raw.decode().splitlines() if line.strip())
                sources.append(dict(path=source, sha256=hashlib.sha256(raw).hexdigest()))
            payload = ''.join(json.dumps(row) + '\n' for row in rows)
            for worker in workers:
                with Path(worker['journal']).open('x') as stream:
                    stream.write(payload)
                emit('journal_history_carried_forward', worker, sources=sources, events=len(rows))
            journals_prepared = True
        if ready_to_restart:
            for worker in workers:
                if worker['phase'] != 'ready':
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
            print('All workers migrated at journaled boundaries; no overlapping collectors.', flush=True)
            break
        time.sleep(0.05)
    else:
        raise RuntimeError('Boundary migration timed out; existing supervisors were left running')


if __name__ == '__main__':
    main()
