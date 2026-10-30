#!/usr/bin/env python3
"""Freeze watch and evacuation across CoreWeave accounts (accounts.py; plans/COREWEAVE_SANDBOXES.md "Multiple accounts").

Owner decision 2026-10-30 (insurance): an account counts as frozen only on an explicit quota error from the API
(accounts.quota_error). On the first frozen account, every sandbox in every account gets its newest state pulled to
the DGX at once: each account's sandboxes are read with that account's own key (cwctl binds every sandbox to its
account), pulls run in parallel across accounts, and within an account the sandbox with the most earned work not
yet verified on the DGX goes first. The frozen account itself is attempted too, best effort: a frozen account's
sandboxes may already reject file reads (2026-10-23 incident).

Each pull is recovery_pull.py (immutable intent, sandbox-side sha256, verified parts, ``export_verified`` record).
What a sandbox exports is its launch record's ``exports`` (cwctl.export_targets: the job's declared exports, else its
output trees). An optional ``checkpoint_request`` argv on an export target runs in the sandbox first. The newest
verified state per role is kept in <evacuation_root>/newest-verified.json. Exports keep their normal cadence
otherwise; nothing here adds export traffic until a freeze.

    account_exports.py status                      per-account freeze state and unexported work per sandbox
    account_exports.py evacuate --trigger NAME [--reason TEXT] [--dry-run]
    account_exports.py watch [--once] [--records-interval 60] [--poll-interval 600]
        records every 60 s (free) and a read-only list per account every 600 s; never a create
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cwctl  # noqa: E402

accounts = cwctl.cwaccounts
_SAFE = re.compile(r'[^A-Za-z0-9._-]+')


def _safe(text: str) -> str:
    return _SAFE.sub('_', text.strip('/')) or 'root'


def default_pull(sandbox_id: str, target: dict, dest: Path, owner: str) -> dict:
    """recovery_pull.py in a child process (its own retries, intent and verification); returns its receipt."""
    command = [sys.executable, str(HERE / 'recovery_pull.py'), target['kind'], sandbox_id, target['remote'], str(dest),
               '--owner', owner, '--keep', '2']
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'recovery_pull failed ({result.returncode}): {result.stderr[-800:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def default_checkpoint(sandbox_id: str, argv: list[str]) -> None:
    result = cwctl.read_sandbox(sandbox_id, patience=60).exec(argv, timeout_seconds=1800).result(timeout=1830)
    if result.returncode:
        raise RuntimeError(f'checkpoint request failed ({result.returncode}): {result.stderr[-500:]}')


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    os.replace(tmp, path)


def evacuate(trigger: str, reason: str, *, ledger=None, config=None, pull=default_pull, checkpoint=default_checkpoint,
             dry_run: bool = False, root: Path | None = None) -> dict:
    config = config or accounts.settings()
    ledger = ledger or cwctl.account_ledger(write=False)
    plan = accounts.evacuation_plan(ledger, trigger, reason=reason)
    root = Path(root or config.get('evacuation_root') or accounts.DEFAULTS['evacuation_root']).expanduser()
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    journal = cwctl.STATE / 'evacuations' / f'{stamp}-{trigger}.json'
    result = {**plan, 'journal': str(journal), 'root': str(root), 'dry_run': dry_run, 'results': {}}
    if dry_run:
        return result
    if accounts.lifecycle(cwctl.STATE).get(trigger, {}).get('state') != 'frozen':
        accounts.set_lifecycle(cwctl.STATE, trigger, 'frozen', reason)
    cwctl.record({'event': 'evacuation_started', 'account': trigger, 'reason': reason, 'journal': str(journal)})
    _write_json(journal, result)

    def one_account(name: str, boxes: list[dict]) -> list[dict]:
        done = []
        for box in boxes:  # most unexported work first
            entry = {'sandbox_id': box['sandbox_id'], 'job': box['job'], 'pulls': []}
            if not box['exports']:
                entry['error'] = 'no declared exports: nothing to evacuate (declare "exports" in the job)'
            for target in box['exports']:
                role = target.get('role') or Path(target['remote']).name
                dest = root / name / box['sandbox_id'] / _safe(role)
                try:
                    if target.get('checkpoint_request'):
                        checkpoint(box['sandbox_id'], list(target['checkpoint_request']))
                    receipt = pull(box['sandbox_id'], target, dest, f'evacuation-{trigger}-{role}')
                    entry['pulls'].append({'role': role, 'remote': target['remote'], 'verified': bool(
                        receipt.get('verified')), 'sha256': receipt.get('sha256'), 'bytes': receipt.get('bytes'),
                        'local': receipt.get('local'), 'step': receipt.get('checkpoint_policy_step')})
                except Exception as error:  # noqa: BLE001 - one failed pull never stops the others
                    entry['pulls'].append({'role': role, 'remote': target['remote'], 'verified': False,
                                           'error': f'{type(error).__name__}: {error}'[:800],
                                           'best_effort': box['best_effort']})
            done.append(entry)
        return done

    with ThreadPoolExecutor(max_workers=max(1, len(plan['accounts']))) as pool:
        futures = {name: pool.submit(one_account, name, boxes) for name, boxes in plan['accounts'].items()}
        for name, future in futures.items():
            result['results'][name] = future.result()
    newest_path = root / 'newest-verified.json'
    newest = json.loads(newest_path.read_text()) if newest_path.is_file() else {}
    for name, entries in result['results'].items():
        for entry in entries:
            for item in entry['pulls']:
                if item.get('verified'):
                    newest[f"{entry['job']}/{item['role']}"] = {'account': name, 'sandbox_id': entry['sandbox_id'],
                                                               'at': stamp, **item}
    _write_json(newest_path, newest)
    result['newest_verified'] = str(newest_path)
    result['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    _write_json(journal, result)
    failed = sum(not p.get('verified') for entries in result['results'].values() for e in entries for p in e['pulls'])
    cwctl.record({'event': 'evacuation_finished', 'account': trigger, 'journal': str(journal), 'failed_pulls': failed})
    return result


def newly_frozen(ledger: dict, handled: dict) -> list[str]:
    """Accounts with an explicit quota error that no evacuation has handled since that error."""
    found = []
    for name, account in sorted(ledger['accounts'].items()):
        freeze = account['freeze']
        if not freeze['frozen']:
            continue
        latest = max((s.get('at') or '') for s in freeze['signals'])
        if handled.get(name, '') < latest:
            found.append(name)
    return found


def watch_once(*, refresh: bool, evacuate_fn=evacuate) -> dict:
    if refresh:
        cwctl.refresh_accounts()
    ledger = cwctl.account_ledger()
    handled_path = cwctl.STATE / 'evacuations' / 'handled.json'
    report = {'at': ledger['at'], 'frozen': [], 'evacuations': []}
    if not handled_path.is_file():
        # First start: quota errors already in the records (the default account's 2026-10-23/29/30 cutoffs) are the
        # baseline, not a new freeze; they are recorded as handled without an evacuation.
        baseline = {name: max(s.get('at') or '' for s in a['freeze']['signals'])
                    for name, a in ledger['accounts'].items() if a['freeze']['frozen']}
        _write_json(handled_path, baseline)
        report['baseline'] = baseline
    handled = json.loads(handled_path.read_text())
    for name in newly_frozen(ledger, handled):
        signals = ledger['accounts'][name]['freeze']['signals']
        reason = f'explicit quota error: {signals[-1]["error"]}'
        report['frozen'].append(name)
        outcome = evacuate_fn(name, reason, ledger=ledger)
        report['evacuations'].append(outcome.get('journal'))
        handled[name] = max(s.get('at') or '' for s in signals)
        _write_json(handled_path, handled)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    p = sub.add_parser('evacuate')
    p.add_argument('--trigger', required=True)
    p.add_argument('--reason', default='operator-requested evacuation')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('watch')
    p.add_argument('--once', action='store_true')
    p.add_argument('--records-interval', type=float, default=60)
    p.add_argument('--poll-interval', type=float, default=600)
    args = parser.parse_args(argv)
    cwctl.load_env_file()
    if args.command == 'status':
        print('\n'.join(cwctl.accounts_status(cwctl.account_ledger())))
    elif args.command == 'evacuate':
        print(json.dumps(evacuate(accounts.check_name(args.trigger), args.reason, dry_run=args.dry_run), indent=1))
    else:
        last_poll = 0.0
        while True:
            poll = time.monotonic() - last_poll >= args.poll_interval
            report = watch_once(refresh=poll)
            if poll:
                last_poll = time.monotonic()
            if report['frozen'] or args.once:
                print(json.dumps(report), flush=True)
            if args.once:
                return 0
            time.sleep(args.records_interval)
    return 0


if __name__ == '__main__':
    if not os.environ.get('NATLANG_CW_NO_REEXEC'):
        cwctl.reexec_in_venv()
    sys.exit(main())
