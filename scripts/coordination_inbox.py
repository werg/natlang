#!/usr/bin/env python3
"""Append/read a machine-local coordination inbox without committing its contents."""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'post', 'check'])
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--sender', default='operator')
    parser.add_argument('--ack', action='store_true', help='Record that displayed notes were read')
    args = parser.parse_args()
    directory = args.repo / '.coordination'
    directory.mkdir(parents=True, exist_ok=True)
    inbox, receipt = directory / 'inbox.md', directory / 'seen.json'
    with (directory / 'inbox.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not inbox.exists():
            inbox.write_text('# Coordination inbox\n\nAppend notes; keep history. Procedures: plans/MACHINE_COORDINATION.md.\n')
        if args.action == 'post':
            body = sys.stdin.read().strip()
            if not body:
                parser.error('post requires a note on stdin')
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            with inbox.open('a') as output:
                output.write(f'\n## {stamp} — {args.sender}\n\n{body}\n')
        elif args.action == 'check':
            data = inbox.read_bytes()
            offset = json.loads(receipt.read_text()).get('offset', 0) if receipt.exists() else 0
            if offset > len(data):
                offset = 0
            print(data[offset:].decode('utf-8'), end='')
            if args.ack:
                receipt.write_text(json.dumps({'offset': len(data)}) + '\n')


if __name__ == '__main__':
    main()
