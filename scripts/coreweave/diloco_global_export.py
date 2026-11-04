"""Independent recovery of an elastic DiLoCo session's coordinator: global-weight exports after every merge round.

Until a fenced replica exists, the cohost coordinator is a single copy. With outer momentum 0 (the session's first
rounds) the FP32 global masters plus the session control are the whole coordinator state, so a replacement session can
be seeded exactly from the latest export; after momentum starts, an export loses only the outer momentum, which a new
session restarts honestly at zero. This loop runs on the DGX (the DGX reaches the cohost's service URL; the cohost
itself has no RAM to spare beside its learner): it polls ``/v1/status`` and, when the merge count grew, runs
``native_diloco export-global --coordinator-url`` (fragments over the API) into a model-only checkpoint under
``OUT/global-m<merges>.pt``, keeps the newest ``--keep`` and commits each to the run's cloud blob store.

Usage (under the memory ledger, with the CoreWeave SDK venv; the export itself runs in the learner venv):
  diloco_global_export.py --run RUN.json --url URL --gateway-of SANDBOX --token FILE --parent SEED --out DIR
      [--interval 600] [--keep 2] [--once]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts' / 'coreweave'), str(REPO / 'training' / 'neuralese'), str(REPO)]
CW_PYTHON = Path.home() / '.local/share/natlang/cw-venv/bin/python'
PYTHON = REPO / '.venv-neuralese' / 'bin' / 'python'


def log(event, **fields):
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event, **fields}), flush=True)


def share_token(sandbox):
    import cwctl
    value = cwctl.share_token(sandbox)
    return (value.decode() if isinstance(value, bytes) else value).strip()


def status(url, token, gateway, timeout):
    request = urllib.request.Request(url.rstrip('/') + '/v1/status',
                                     headers={'Authorization': f'Bearer {token}', 'X-Sandbox-Share-Token': gateway})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--url', required=True)
    ap.add_argument('--gateway-of', required=True)
    ap.add_argument('--token', required=True)
    ap.add_argument('--parent', required=True, help='the session seed checkpoint (its bytes on the DGX)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--interval', type=float, default=600)
    ap.add_argument('--keep', type=int, default=2)
    ap.add_argument('--once', action='store_true')
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    secrets = out / '.secrets'
    secrets.mkdir(mode=0o700, exist_ok=True)
    token = Path(a.token).read_text().strip()
    (secrets / 'diloco-token').write_text(token)
    os.chmod(secrets / 'diloco-token', 0o600)
    done = {int(p.stem.split('-m')[1]) for p in out.glob('global-m*.pt') if p.stem.split('-m')[1].isdigit()}
    while True:
        try:
            gateway = share_token(a.gateway_of)
            (secrets / 'diloco-gateway').write_text(gateway)
            os.chmod(secrets / 'diloco-gateway', 0o600)
            value = status(a.url, token, gateway, 600)
            merges = int(value.get('merges') or (value.get('progress') or {}).get('merges') or 0)
            log('status', merges=merges, exported=max(done, default=None))
            if merges and merges not in done:
                target = out / f'global-m{merges:06d}.pt'
                temp = out / f'.global-m{merges:06d}.pt.tmp'
                env = {**os.environ, 'NATLANG_CW_SECRETS': str(secrets),
                       'PYTHONPATH': f'{REPO}/training/neuralese:{REPO}', 'OMP_NUM_THREADS': '4'}
                started = time.monotonic()
                subprocess.run([str(PYTHON), '-m', 'natlang_neuralese.train.native_diloco', 'export-global',
                                '--coordinator-url', a.url, '--parent', a.parent, '--out', str(temp),
                                '--workdir', str(out / '.fetch')], env=env, check=True)
                os.replace(temp, target)
                done.add(merges)
                log('exported', merges=merges, path=str(target), bytes=target.stat().st_size,
                    seconds=round(time.monotonic() - started, 1))
                import rl_cloud
                record = rl_cloud.put_local(rl_cloud.load_run(a.run), target)
                log('committed', merges=merges, blob=record)
                for old in sorted(out.glob('global-m*.pt'))[:-a.keep]:
                    old.unlink()
                    log('pruned', path=str(old))
        except Exception as exc:  # keep polling: a transient outage must not end recovery exports
            log('error', error=f'{type(exc).__name__}: {exc}'[:500])
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == '__main__':
    sys.exit(main())
