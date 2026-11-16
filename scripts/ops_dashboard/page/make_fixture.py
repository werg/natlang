#!/usr/bin/env python3
"""Build a Tendriller preview fixture (streams from the exporter's output) for `tendril-preview PAGE --fixture FILE`.

usage: make_fixture.py OUT_DIR(exporter output) > fixture.json
"""
import json
import sys

out = sys.argv[1]


def lines(name):
    try:
        return [json.loads(l) for l in open(f'{out}/{name}') if l.strip()]
    except OSError:
        return []


def rows(name, stream, durability, last=None):
    recs = lines(name)[-last:] if last else lines(name)
    return {'name': stream, 'durability': durability, 'records': [
        {'id': f'{stream}-{i}', 'type': r['type'], 'data': r['data']} for i, r in enumerate(recs)]}


snap = lines('latest.json')
streams = [{'name': 'ops.snapshot', 'durability': 'durable', 'max_records': 1,
            'records': [{'id': 'snap-1', 'type': 'snapshot', 'data': snap[-1]['data']}] if snap else []},
           rows('machines.jsonl', 'ops.machines', 'durable'), rows('metrics.jsonl', 'training.metrics', 'durable'),
           rows('events.jsonl', 'ops.events', 'durable'), {'name': 'machine.live', 'durability': 'ephemeral', 'records': []}]
print(json.dumps({'streams': streams, 'role': 'owner', 'steps': []}))
