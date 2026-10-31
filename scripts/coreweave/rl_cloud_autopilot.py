#!/usr/bin/env python3
"""Cloud RL autopilot (plans/neuralese/LFM3_RL_CLOUD_20261031.md): wait for GPU entitlement, run the GPU smoke, then
launch the production lineages. A DGX ledger service; one pass per --interval (default an hour).

  rl_cloud_autopilot.py RUN.json --lineages jolt-c1,sd-c1,jolt-c2,sd-c2 [--smoke smoke-jolt] [--interval 3600] [--once]

Each pass, in order (state: ROOT/.cloud/autopilot.json; decisions: the run's decisions log; coord on milestones):
1. Entitlement: while no account is known to be GPU-entitled, ONE minimal GPU sandbox create per pass, rotating over
   the placement accounts; a created probe is stopped at once. CWSANDBOX_GPU_NOT_ALLOWED means "not yet" (never a
   freeze); a quota error is left to the normal freeze path (accounts.py).
2. Smoke: the smoke lineage (one tiny round) on an entitled account; passed when its export is DGX-verified and its
   sandbox completed (the exporter service, rl_cloud.py export, does the upload and GET). A failed smoke stops the
   autopilot with an urgent coord alert: nothing launches on an unproven path.
3. Production: every listed lineage without a running sandbox is seeded (blobs are shared) and launched on an entitled
   account (balanced by rl_cloud.launch's placement over the entitled accounts). A launch refused for entitlement
   marks that account not entitled and is retried in a later pass.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rl_cloud  # noqa: E402

NOT_ALLOWED = 'CWSANDBOX_GPU_NOT_ALLOWED'


def coord(subject, message, urgent=False):
    command = ['python3', str(rl_cloud.REPO / 'scripts' / 'coord.py'), '--as', 'dgx-rl-cloud', 'send', '--to', 'dgx',
               '--subject', subject, '-m', message] + (['--urgent', '--force'] if urgent else [])
    subprocess.run(command, cwd=rl_cloud.REPO, capture_output=True)


def accounts():
    cwctl = rl_cloud.cwctl
    ledger = cwctl.account_ledger(write=False)
    return sorted(name for name, value in ledger['accounts'].items() if value['eligible'])


def refused_for_entitlement(error_text):
    return NOT_ALLOWED in error_text or 'not entitled to create GPU sandboxes' in error_text


def probe(account):
    """One minimal GPU sandbox create in ``account``; a created sandbox is stopped at once. True / False / None (other
    error, unknown)."""
    cwctl = rl_cloud.cwctl
    job = rl_cloud.jobs.validate({**cwctl.smoke_job(True), 'id': f'gpu-entitlement-probe-{account}',
                                  'lifetime_hours': 0.25, 'resources': {'gpu': 1, 'cpu': '1', 'memory': '2Gi'},
                                  'placement': {'where': 'coreweave', 'reason': 'GPU entitlement probe (stopped at once)'}})
    try:
        entry = cwctl.launch(job, False, False, None, account=account)
    except Exception as error:  # noqa: BLE001 -- classified below
        text = f'{type(error).__name__}: {error}'
        return (False if refused_for_entitlement(text) else None), text[:400]
    subprocess.run([sys.executable, str(HERE / 'cwctl.py'), 'stop', entry['sandbox_id']], capture_output=True)
    return True, entry['sandbox_id']


def lineage_state(run, name):
    return rl_cloud.read_json(rl_cloud.hub(run) / 'run-state.json', {}).get('lineages', {}).get(name, {})


def sandbox_status(sandbox_id):
    result = subprocess.run([sys.executable, str(HERE / 'cwctl.py'), 'get', sandbox_id], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith('STATUS'):
            return line.split()[-1]
    return None


def mark_ended(run, name, status):
    path = rl_cloud.hub(run) / 'run-state.json'
    state = rl_cloud.read_json(path, {})
    state['lineages'][name]['status'] = status
    rl_cloud.store_json(path, state)


def launch_on_entitled(run, name, auto, rounds=None):
    """Seed if needed, then launch on the entitled account with the fewest of our running lineages."""
    if rl_cloud.newest_verified(run, name) is None:
        rl_cloud.seed(run, name)
    head = subprocess.run(['git', '-C', str(rl_cloud.REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()
    if rl_cloud.read_json(rl_cloud.hub(run) / 'run-state.json', {}).get('code_sha') != head:
        rl_cloud.bundle(run)  # launches run the current main commit (fixes since the last bundle)
    state = rl_cloud.read_json(rl_cloud.hub(run) / 'run-state.json', {})
    busy = [v.get('account') for v in state.get('lineages', {}).values() if v.get('status') == 'running']
    entitled = sorted(auto['entitled'], key=lambda a: (busy.count(a), a))
    for account in entitled:
        try:
            return rl_cloud.launch(run, name, account=account, rounds=rounds)
        except Exception as error:  # noqa: BLE001
            text = f'{type(error).__name__}: {error}'
            if refused_for_entitlement(text):
                auto['entitled'].remove(account)
                continue
            raise
    return None


def one_pass(run, args, auto):
    path = rl_cloud.hub(run) / 'autopilot.json'
    auto.setdefault('entitled', [])
    auto.setdefault('probes', [])
    if not auto['entitled']:
        candidates = accounts()
        account = candidates[len(auto['probes']) % len(candidates)]
        ok, detail = probe(account)
        auto['probes'].append({'at': time.time(), 'account': account, 'entitled': ok, 'detail': detail})
        rl_cloud.store_json(path, auto)
        if not ok:
            print(json.dumps({'event': 'not_entitled_yet' if ok is False else 'probe_error', 'account': account,
                              'detail': detail}), flush=True)
            return
        auto['entitled'].append(account)
        rl_cloud.note(run, f'GPU entitlement: {account} created a GPU sandbox ({detail}, stopped at once)')
    smoke = auto.setdefault('smoke', {})
    if not smoke.get('passed'):
        live = lineage_state(run, args.smoke)
        if live.get('status') == 'running' and 'check' in live.get('job', ''):
            status = sandbox_status(live['sandbox_id'])
            if status not in ('completed', 'failed', 'terminated'):
                return  # the CPU pipeline check still runs
            mark_ended(run, args.smoke, status)
            live = lineage_state(run, args.smoke)
        if live.get('status') != 'running':
            launched = launch_on_entitled(run, args.smoke, auto, rounds=1)
            smoke.update(started=time.time(), sandbox_id=launched and launched['sandbox_id'])
            rl_cloud.store_json(path, auto)
            if launched:
                coord('cloud RL GPU smoke started', f'smoke-jolt in {launched["sandbox_id"]} ({launched["account"]})')
            return
        status = sandbox_status(live['sandbox_id'])
        verified = [r for r in (rl_cloud.read_json(p) for p in (rl_cloud.hub(run) / 'exports' / args.smoke).glob('[0-9]*.json'))
                    if r and r.get('dgx_verified') and r['trained_rounds'] >= 1]
        if verified and status in ('completed', 'failed', 'terminated', None):
            mark_ended(run, args.smoke, status or 'ended')
            smoke.update(passed=True, at=time.time(), export=verified[-1]['seq'])
            rl_cloud.store_json(path, auto)
            rl_cloud.note(run, f'GPU smoke passed: {args.smoke} export {verified[-1]["seq"]} verified on DGX')
        elif status in ('completed', 'failed', 'terminated'):
            mark_ended(run, args.smoke, status)
            smoke.update(failed=True, at=time.time(), status=status)
            rl_cloud.store_json(path, auto)
            rl_cloud.note(run, f'GPU smoke FAILED ({status}); autopilot stops; logs in {rl_cloud.hub(run)}/logs')
            coord('cloud RL GPU smoke FAILED', f'{args.smoke} sandbox {live["sandbox_id"]} ended {status} without a '
                  f'verified export; autopilot stopped; logs {rl_cloud.hub(run)}/logs', urgent=True)
            raise SystemExit(3)
        return
    launched = []
    for name in args.lineages:
        live = lineage_state(run, name)
        if live.get('status') == 'running':
            status = sandbox_status(live['sandbox_id'])
            if status in ('completed', 'failed', 'terminated'):
                mark_ended(run, name, status)
                rl_cloud.note(run, f'{name} sandbox {live["sandbox_id"]} ended ({status}); relaunching from the '
                                   'newest verified export')
            else:
                continue
        entry = launch_on_entitled(run, name, auto)
        if entry:
            launched.append(f'{name}: {entry["sandbox_id"]} ({entry["account"]})')
    if launched:
        coord('cloud RL launched', '; '.join(launched) + f' (run {run["id"]}, plans/neuralese/LFM3_RL_CLOUD_20261031.md)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('--lineages', required=True)
    parser.add_argument('--smoke', default='smoke-jolt')
    parser.add_argument('--interval', type=float, default=3600)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    args.lineages = args.lineages.split(',')
    run = rl_cloud.load_run(args.run)
    path = rl_cloud.hub(run) / 'autopilot.json'
    while True:
        auto = rl_cloud.read_json(path, {})
        try:
            one_pass(run, args, auto)
        except SystemExit:
            raise
        except Exception as error:  # noqa: BLE001 -- a pass error is logged; the next pass retries
            print(json.dumps({'event': 'pass_error', 'error': f'{type(error).__name__}: {error}'[:800]}), flush=True)
        rl_cloud.store_json(path, auto)
        if args.once:
            return 0
        # entitlement probes hourly; once entitled, the smoke and the launches are followed every 5 minutes
        time.sleep(args.interval if not auto.get('entitled') else min(args.interval, 300))


if __name__ == '__main__':
    sys.exit(main())
