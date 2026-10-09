#!/usr/bin/env python3
"""Observe OpenRouter's free catalog; discovery does not authorize generation."""
import argparse
import datetime as dt
import decimal
import fcntl
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def scan(directory):
    request = urllib.request.Request('https://openrouter.ai/api/v1/models',
                                     headers={'User-Agent': 'natlang-catalog-monitor/1.0'})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    catalog = json.loads(body)
    free = []
    for model in catalog['data']:
        prices = model.get('pricing', {})
        try:
            if not all(decimal.Decimal(str(prices[k])) == 0 for k in ('prompt', 'completion')):
                continue
            if decimal.Decimal(str(prices.get('request', '0'))) != 0:
                continue
        except (KeyError, decimal.InvalidOperation):
            continue
        if 'text' not in model.get('architecture', {}).get('output_modalities', []):
            continue
        free.append({k: model.get(k) for k in
                     ('id', 'name', 'created', 'context_length', 'pricing',
                      'supported_parameters', 'top_provider', 'architecture')})
    free.sort(key=lambda m: (m.get('created') or 0, m['id']), reverse=True)
    previous_path = directory / 'latest.json'
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    old = {m['id']: m for m in previous.get('models', [])}
    current = {m['id']: m for m in free}
    checked = dt.datetime.now(dt.timezone.utc)
    result = {'checked_at': checked.isoformat(), 'source': request.full_url,
              'catalog_sha256': hashlib.sha256(body).hexdigest(), 'models': free,
              'added': sorted(current.keys() - old.keys()),
              'removed': sorted(old.keys() - current.keys()),
              'changed': sorted(k for k in current.keys() & old.keys() if current[k] != old[k]),
              'availability': 'catalog only; authenticated request still required',
              'selection_policy': 'free, working, throughput first; retain quality admission gates'}
    path = directory / (checked.strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    path.write_text(json.dumps(result, indent=2) + '\n')
    temporary = directory / '.latest.tmp'
    temporary.write_text(path.read_text())
    temporary.replace(previous_path)
    print(json.dumps({k: result[k] for k in ('checked_at', 'added', 'removed', 'changed')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval-seconds', type=int, default=10800)
    args = parser.parse_args()
    if args.interval_seconds < 300:
        parser.error('catalog intervals must be at least 300 seconds')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                scan(args.out)
            except Exception as error:
                # No credentials are used or logged by this public catalog observer.
                print(json.dumps({'checked_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                                  'error_type': type(error).__name__}), flush=True)
                if not args.watch:
                    raise
            if not args.watch:
                break
            time.sleep(args.interval_seconds)


if __name__ == '__main__':
    main()
