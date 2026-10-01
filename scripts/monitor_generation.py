#!/usr/bin/env python3
"""Persist hourly generation audits and flag idle workers for agent review."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from generation_authority import authority_lock


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def live(pid, signature):
    try:
        command = Path(f'/proc/{int(pid)}/cmdline').read_bytes()
        return signature.encode() in command
    except (OSError, TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authority', required=True, type=Path)
    parser.add_argument('--health-reader', required=True, type=Path)
    parser.add_argument('--spool-reader', required=True, type=Path)
    parser.add_argument('--state-dir', required=True, type=Path)
    args = parser.parse_args()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                authority = read_json(args.authority)
                bonsai_live = live(authority.get('bonsai_supervisor'), 'run_bonsai_queue.py')
                luna = authority.get('luna_workers', [])
                luna_live = [w.get('pid') for w in luna if live(w.get('pid'), 'run_bonsai_queue.py')]
                needs_review = []
                if not bonsai_live:
                    needs_review.append('bonsai_idle: review completion and replenish qualified work')
                if len(luna_live) < authority.get('luna_concurrency', 2):
                    needs_review.append('luna_idle: review completion and replenish qualified work')
                heartbeat = dict(checked_at=utc(), pid=os.getpid(), bonsai_live=bonsai_live,
                                 luna_live_pids=luna_live, needs_agent_review=needs_review,
                                 next_check_at=authority['next_check_at'])
                atomic_json(args.state_dir / 'status.json', heartbeat)
                due = dt.datetime.fromisoformat(authority['next_check_at'].replace('Z', '+00:00'))
                if dt.datetime.now(dt.timezone.utc) >= due:
                    health = subprocess.run(['node', str(args.health_reader)], check=True,
                                            capture_output=True, text=True, timeout=900)
                    metadata = json.loads(health.stdout)
                    report_path = Path(metadata['jsonPath']).resolve()
                    report = read_json(report_path)
                    spool = subprocess.run(['node', str(args.spool_reader), str(report_path)],
                                           check=True, capture_output=True, text=True, timeout=300)
                    spool_metadata = json.loads(spool.stdout)
                    with authority_lock(args.authority):
                        current = read_json(args.authority)
                        # Do not overwrite a newer inspection's baseline or worker authority.
                        if current['checked_at'] == authority['checked_at']:
                            checked = dt.datetime.fromisoformat(report['checked_at'].replace('Z', '+00:00'))
                            current.update(checked_at=report['checked_at'],
                                           next_check_at=(checked + dt.timedelta(hours=1)).isoformat(),
                                           last_status_report=str(report_path),
                                           latest_bonsai_spool=spool_metadata['op'])
                            atomic_json(args.authority, current)
                    event = dict(event='hourly_audit', time=utc(), health=metadata,
                                 spool=spool_metadata,
                                 blocking_reader_errors=report['completeness']['blocking_error_count'],
                                 needs_agent_review=needs_review)
                    with (args.state_dir / 'journal.jsonl').open('a') as stream:
                        stream.write(json.dumps(event) + '\n')
                    print(json.dumps(event), flush=True)
            except Exception as error:
                event = dict(event='monitor_error', time=utc(), error=f'{type(error).__name__}: {error}')
                with (args.state_dir / 'journal.jsonl').open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
                print(json.dumps(event), flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
