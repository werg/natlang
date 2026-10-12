#!/usr/bin/env python3
"""Rehearse the whole CoreWeave loop on DGX with a local S3 stand-in, so the first real launch tests only CoreWeave.

The steps are:

1. Start an S3 stand-in (moto_server from the client venv) as a ledger unit.
2. Bundle the code snapshot (git archive of HEAD) and a tiny input, and upload them.
3. Run `bootstrap.sh` in a docker container built from exactly the request `cwctl.py launch` would send (same
   image, command, env and mounted bootstrap). The container is fresh, with nothing of the host mounted, and the
   DGX roots are mirrored inside it.
4. The job is `rehearsal_job.py`, a CPU stand-in for a training stage. Its checkpoints are published incrementally
   by `bundle.py watch`.
5. On the host: read `exit.json` and the output pointer, then fetch and verify the outputs.
6. Relaunch with `--resume` and a nonzero exit status. Check that the job continues from the published checkpoint,
   that the status is recorded, and that the container does not restart.
7. Plan a gc (dry), then stop the stand-in.

Usage (DGX; every container and the S3 server run as memory-ledger units, priority medium, the smoke class):

    python3 scripts/coreweave/rehearse.py [--image natlang-neuralese:26.09] [--port 59321] [--keep-tmp]

It prints one JSON line per step and a final summary. It exits 0 only if every check passes. No CoreWeave API is
called and no credential is used: the S3 credentials are moto's fake ones, injected the way CoreWeave injects the
sandbox's own.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
VENV = Path(os.environ.get('NATLANG_CW_VENV', Path.home() / '.local/share/natlang/cw-venv'))
BUCKET = 'natlang-cw-rehearsal'
OUT = 'runs/cw-rehearsal/stage'


def log(**event):
    print(json.dumps(event), flush=True)


def ledger(unit: str, budget_gb: float, command: list[str], container: str | None = None) -> None:
    args = ['python3', str(REPO / 'scripts' / 'memory_ledger.py'), 'run', '--unit', unit, '--budget-gb', str(budget_gb),
            '--priority', 'medium', '--wait', '21600', '--no-resumable']
    if container:
        args += ['--container', container]
    subprocess.run(args + ['--', *command], check=True, stdout=subprocess.DEVNULL)


def unit_active(unit: str) -> bool:
    state = subprocess.run(['systemctl', '--user', 'is-active', f'{unit}.service'], capture_output=True, text=True)
    return state.stdout.strip() in ('active', 'activating', 'deactivating')


def wait_unit(unit: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while unit_active(unit):
        if time.monotonic() > deadline:
            raise TimeoutError(f'{unit} still running after {timeout} s')
        time.sleep(2)


def wait_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            if time.monotonic() > deadline:
                raise TimeoutError(f'{url} not up after {timeout} s')
            time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--image', default='natlang-neuralese:26.09')
    parser.add_argument('--port', type=int, default=59321)
    parser.add_argument('--keep-tmp', action='store_true')
    args = parser.parse_args()
    import bundle
    import cwctl
    import job as jobs
    from natlang_neuralese.common.paths import DEFAULTS
    container_repo = DEFAULTS['repo']  # the bootstrap's default NATLANG_REPO inside the container (mirror root)

    stamp = time.strftime('%H%M%S')
    tmp = Path(tempfile.mkdtemp(prefix='cw-rehearse-'))
    endpoint = f'http://127.0.0.1:{args.port}'
    s3_env = {'AWS_ENDPOINT_URL_S3': endpoint, 'AWS_ACCESS_KEY_ID': 'rehearsal', 'AWS_SECRET_ACCESS_KEY': 'rehearsal',
              'AWS_REGION': 'us-east-1', 'NATLANG_CW_S3_ADDRESSING': 'path'}
    os.environ.update(s3_env)
    for name in ('CWSANDBOX_API_KEY', 'COREWEAVE_API_TOKEN', 'AWS_CONTAINER_CREDENTIALS_FULL_URI'):
        os.environ.pop(name, None)  # the rehearsal must never touch real credentials
    s3_unit = f'cw-rehearse-s3-{stamp}'
    checks: dict[str, bool] = {}
    try:
        ledger(s3_unit, 1, [str(VENV / 'bin' / 'moto_server'), '-H', '127.0.0.1', '-p', str(args.port)])
        wait_http(endpoint, 120)
        s3 = bundle.s3_client()
        s3.create_bucket(Bucket=BUCKET)
        log(step='s3', endpoint=endpoint, unit=s3_unit)

        data = tmp / 'input.txt'
        data.write_text(''.join(f'row {i}\n' for i in range(10)))
        pairs = bundle.extra_entries(str(data))
        code, code_source = bundle.code_archive('HEAD', tmp)
        manifest = bundle.build_manifest(pairs, code, 'rehearsal')
        sources = {item['key']: source for item, source in pairs} | {code['key']: code_source}
        bundle.upload(manifest, sources, BUCKET, s3, log=lambda _: None)
        log(step='bundle', bundle=manifest['id'], code=code['git_sha'], files=len(manifest['files']))

        def run(attempt: int, resume: bool, exit_status: int) -> str:
            job = jobs.validate({
                'schema': jobs.SCHEMA, 'id': 'rehearsal', 'kind': 'smoke', 'image': args.image,
                'placement': {'reason': 'local rehearsal of bootstrap, publish, resume and fetch'},
                'resources': {'cpu': '2', 'memory': '4Gi'}, 'lifetime_hours': 0.25, 'bucket': BUCKET,
                'bundle': bundle.manifest_key(manifest), 'region': 'us-east-1', 'outputs': [OUT], 'resume': resume,
                'env': {'NATLANG_CW_PUBLISH_INTERVAL': '1', 'NATLANG_CW_IMAGE_READY': '1'},
                'command': ['python3', 'scripts/coreweave/rehearsal_job.py', '--out', OUT, '--input', str(data),
                            '--exit-status', str(exit_status)]})
            request = jobs.sandbox_request(job, cwctl.bootstrap_text())
            name = f'cw-rehearse-run{attempt}-{stamp}'
            files = tmp / f'files-{attempt}'
            files.mkdir()
            extra = {k: v for k, v in s3_env.items()}
            docker = cwctl.local_docker_command(request, name, files, extra)
            output = tmp / f'run{attempt}.log'
            ledger(name, 4, ['sh', '-c', ' '.join(map(_quote, docker)) + f' > {output} 2>&1'], container=name)
            wait_unit(name, 900)
            text = output.read_text()
            log(step=f'run{attempt}', unit=name, log=str(output), lines=len(text.splitlines()))
            return text

        def fetched_report(label: str) -> dict:
            pointer = bundle.read_json(s3, BUCKET, f'outputs/rehearsal/stage/latest.json')
            target = tmp / f'fetched-{label}'
            out_manifest = bundle.read_json(s3, BUCKET, pointer['manifest_key'])
            bundle.fetch(out_manifest, BUCKET, s3, target, [(f'{container_repo}/{OUT}', str(target))],
                         log=lambda _: None)
            checks[f'{label}_verified'] = bundle.verify(out_manifest, target, [(f'{container_repo}/{OUT}', str(target))]) == []
            return json.loads((target / 'report.json').read_text())

        first = run(1, False, 0)
        report = fetched_report('run1')
        exit1 = bundle.read_json(s3, BUCKET, 'outputs/rehearsal/exit.json') or {}
        incremental = [k for k in (o['Key'] for o in bundle.list_all(s3, BUCKET, 'bundles/'))
                       if (bundle.read_json(s3, BUCKET, k) or {}).get('note') == 'incremental']
        checks.update(run1_code_and_input='"input_bytes": 60' in first, run1_report=report.get('final_step') == 4,
                      run1_exit_recorded=exit1.get('status') == 0, run1_incremental_publishes=len(incremental) >= 2)

        second = run(2, True, 3)
        report = fetched_report('run2')
        exit2 = bundle.read_json(s3, BUCKET, 'outputs/rehearsal/exit.json') or {}
        checks.update(run2_resumed=report.get('resumed_from') == 4 and report.get('final_step') == 8,
                      run2_nonzero_recorded=exit2.get('status') == 3,
                      run2_no_restart=second.count('"bootstrap":"start"') == 1)

        plan = bundle.gc_plan(s3, BUCKET, drop_jobs=['rehearsal'], min_age_hours=0, keep_input_days=0)
        checks['gc_plans_everything_of_a_dropped_job'] = any(i['key'].startswith('outputs/rehearsal/') for i in plan['delete'])
        log(step='gc', delete=len(plan['delete']), delete_bytes=plan['delete_bytes'], incremental_publishes=len(incremental))
    finally:
        subprocess.run(['systemctl', '--user', 'stop', f'{s3_unit}.service'], capture_output=True)
        if not args.keep_tmp and all(checks.values()) and checks:
            shutil.rmtree(tmp, ignore_errors=True)
    ok = bool(checks) and all(checks.values())
    log(step='summary', ok=ok, checks=checks, tmp=None if ok and not args.keep_tmp else str(tmp))
    return 0 if ok else 1


def _quote(part: str) -> str:
    import shlex
    return shlex.quote(part)


if __name__ == '__main__':
    python = VENV / 'bin' / 'python'
    if python.is_file() and Path(sys.prefix).resolve() != VENV.resolve():
        os.execv(str(python), [str(python), *sys.argv])
    sys.exit(main())
