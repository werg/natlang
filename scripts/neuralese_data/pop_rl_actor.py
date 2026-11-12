#!/usr/bin/env python3
"""Serve a policy checkpoint from the Pop laptop as an extra rollout actor.

    pop_rl_actor.py start --checkpoint CK --policy-id SHA [--port 18531] [--bind-host IP]
    pop_rl_actor.py stop
    pop_rl_actor.py status

start fast-forwards Pop's checkout of the canonical repository (``git pull``; Pop runs it directly), prepares the
model-only view of the checkpoint on the DGX (``prepare-view``, ``native_bundle.prepare_actor_transport`` with scope
``rl-controller-pin``) in a ledger unit, sends it as an exact delta (scripts/coreweave/send_actor_transport.py), and
launches pop_actor_driver.py under Pop's memory ledger: a container bound to Pop's tailscale address serving the
checkpoint with a block store, so a collector fetches its stored blocks over HTTP. The actor attests the training
checkpoint SHA (the transport receipt binds it) and binds ``actor_export`` in every capture (arithmetic on another GPU
differs from the learner's). stop removes the container and keeps the block store.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path, PurePosixPath

os.environ.pop('NATLANG_REPO', None)
os.environ.pop('PYTHONPATH', None)
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.common.jsonio import write_json_durable as write_json  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

W = Path(os.environ.get('NATLANG_LFM3_WORKSPACE', resolve_str('data_nvme', 'lfm3-rl-dgx-20261029')))
# The state, view and receipt root (the former Pop evaluation root: its config.json names the Pop host).
ROOT = Path(os.environ.get('NATLANG_POP_EVAL_ROOT', str(W / 'ops/pop-behavior-eval')))
PYTHON = resolve_str('repo', '.venv-neuralese/bin/python')
LEDGER = str(REPO / 'scripts/memory_ledger.py')
# Pop's SSH host: NATLANG_POP_HOST, else "pop_host" in <root>/config.json. Pop mirrors the DGX data/repo layout.
POP = os.environ.get('NATLANG_POP_HOST') or (json.loads((ROOT / 'config.json').read_text()).get('pop_host')
                                             if (ROOT / 'config.json').is_file() else None)
POP_REPO = resolve_str('repo')
POP_RECOVERY = resolve_str('data_nvme', 'natlang-independent-recovery-20261031')
POP_TRANSPORT_ROOT = POP_RECOVERY + '/matched75-transport-0f8a-1101'   # the peer cache whose delta base Pop holds
POP_EVAL_ROOT = POP_RECOVERY + '/pop-behavior-eval'
POP_BASE = POP_RECOVERY + '/base-config-only-40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
IMAGE = 'natlang:pop-causal-conv1d-torch214-cu134-sm89-v1'
RAW_PREP_GB = 7.5
POP_RESERVE_GB = 0.5
POP_MIN_FREE = 8 * 2**30 + 512 * 2**20       # scratch floor on Pop
RL_PORT = 18531
STATE = ROOT / 'rl-actor.json'
EXPORT_FORMAT = 'native-pop-rtx4060-bf16'
SGLANG_EXPORT_FORMAT = 'sglang-pop-rtx4060-bf16-vendor-projection'
SGLANG_MEMORY_GB = 11   # 8 GiB was OOM-killed at CUDA-graph capture (2026-11-12): front + view on CPU + server
RL_MEMORY_GB = 8   # container cap and ledger claim; 4 GiB was OOM-killed after 4 h at 3 workers (anon 4.2 GiB, 2026-11-09)


def log(**event):
    print(json.dumps({'at': round(time.time(), 1), 'hook': 'pop_rl_actor', **event}, sort_keys=True), flush=True)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def _stat5(path):
    s = Path(path).stat()
    return {'device': s.st_dev, 'inode': s.st_ino, 'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def prepare_view(args):
    """The ledger unit's work: the model-only actor view of one checkpoint, read in place (never written)."""
    from natlang_neuralese.model.native_bundle import prepare_actor_transport
    before, started = _stat5(args.checkpoint), time.time()
    binding = prepare_actor_transport(args.checkpoint, args.out, scope='rl-controller-pin')
    transport = json.loads(Path(binding['path']).read_text())
    if _stat5(args.checkpoint) != before:
        raise SystemExit('source changed during preparation')
    if args.expect_sha256 and transport['source']['sha256'] != args.expect_sha256:
        raise SystemExit(f'source hashes to {transport["source"]["sha256"]}, expected {args.expect_sha256}')
    write_json(args.receipt, {'schema': 'natlang.pop-eval-preparation/1', 'kind': 'rl', 'transport': binding,
                              'source': transport['source'], 'step': transport['step'], 'view': transport['view'],
                              'view_receipt': transport['view_receipt'], 'code': transport.get('code'),
                              'scope': transport['scope'], 'source_path': str(Path(args.checkpoint).resolve()),
                              'source_stat': before, 'seconds': time.time() - started})


def prepare(checkpoint, expect_sha, wait=None):
    """Preparation receipt of the policy's actor view (reused while its view file is present)."""
    out = ROOT / 'rl-views'
    receipt = ROOT / 'preparations' / f'rl-{expect_sha}.json'
    existing = read_json(receipt)
    if (existing and Path(existing['transport']['path']).is_file()
            and (Path(existing['transport']['path']).parent / existing['view']['path']).is_file()):
        return existing
    # A pruned view keeps its receipts; retire receipts of a view directory without any view before re-preparing.
    stale = out / expect_sha
    if stale.is_dir() and not any(stale.glob('*.pt')):
        retired = ROOT / 'dgx-retirement' / f'stale-receipts-{expect_sha[:12]}-{int(time.time())}'
        retired.mkdir(parents=True, exist_ok=True)
        for item in stale.glob('*.json'):
            os.replace(item, retired / item.name)
    command = ['/usr/bin/ionice', '-c2', '-n6', PYTHON, str(Path(__file__).resolve()), 'prepare-view',
               '--checkpoint', str(checkpoint), '--out', str(out), '--receipt', str(receipt), '--expect-sha256', expect_sha]
    env = [f'NATLANG_REPO={REPO}', f'PYTHONPATH={REPO}/training/neuralese:{REPO}', 'OMP_NUM_THREADS=4',
           f'NATLANG_HASH_CACHE={ROOT}/hash-cache.jsonl', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1']
    unit, status, result = ledger_unit(f'popeval-rl-{expect_sha[:12]}', command, budget_gb=RAW_PREP_GB,
                                       workdir=str(REPO), env=env, **({'wait': wait} if wait else {}))
    if status != 0 or not receipt.is_file():
        raise RuntimeError(f'rl preparation unit {unit} failed ({result}, exit {status}); see journalctl --user -u {unit}')
    return read_json(receipt)


def ledger_unit(name, command, *, budget_gb, workdir, env=(), wait=86400):
    unit = f'{name}-{int(time.time())}'
    argv = ['/usr/bin/python3', LEDGER, 'run', '--unit', unit, '--budget-gb', str(budget_gb), '--priority', 'low',
            '--class', 'experiment', '--no-resumable', '--wait', str(wait), '--workdir', workdir]
    for item in env:
        argv += ['--env', item]
    subprocess.run(argv + ['--', *command], check=True, timeout=wait + 600)
    while True:
        show = subprocess.run(['systemctl', '--user', 'show', unit + '.service', '-p', 'ActiveState', '-p', 'Result',
                               '-p', 'ExecMainStatus'], capture_output=True, text=True).stdout
        fields = dict(line.split('=', 1) for line in show.split('\n') if '=' in line)
        if fields.get('ActiveState') in ('inactive', 'failed'):
            return unit, int(fields.get('ExecMainStatus') or 0), fields.get('Result')
        time.sleep(15)


def ssh(command, *, input=None, timeout=600, check=True):
    if not POP:
        raise RuntimeError('set NATLANG_POP_HOST or "pop_host" in ' + str(ROOT / 'config.json'))
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', POP, command], input=input,
                            text=True, capture_output=True, timeout=timeout)
    if check and result.returncode:
        raise RuntimeError(f'Pop command failed ({result.returncode}): {command[:200]}: {result.stderr[-2000:]}')
    return result.stdout


def pop_python(code, *args, timeout=600):
    return json.loads(ssh(shlex.join(['python3', '-I', '-', *map(str, args)]), input=code, timeout=timeout))


def code_on_dgx():
    from natlang_neuralese.common.code_provenance import code_provenance
    return code_provenance(REPO)


def sync_pop_checkout():
    """Fast-forward Pop's checkout of the canonical repository to origin/main (``git pull --ff-only``), rebuild the
    TypeScript collector when ts-host changed, and return the commit and dirty flag it now runs. Pop runs this checkout
    directly (no runtime copy); a failed pull or build is recorded and the checkout keeps what it has (provenance,
    never a refusal)."""
    git = lambda *args: shlex.join(['git', '-C', POP_REPO, *args])
    out = ssh(git('rev-parse', 'HEAD') + '; ' + git('pull', '--ff-only', '--quiet', 'origin', 'main') + ' 2>&1; '
              + git('rev-parse', 'HEAD') + '; ' + git('status', '--porcelain', '--untracked-files=no') + ' | wc -l',
              check=False, timeout=300)
    lines = out.strip().splitlines()
    try:
        before, commit, dirty = lines[0].strip(), lines[-2].strip(), int(lines[-1]) > 0
    except (IndexError, ValueError):
        raise RuntimeError('cannot read the Pop checkout commit: ' + out[-400:])
    notes = lines[1:-2]
    result = {'schema': 'natlang.code-provenance/1', 'git_commit': commit, 'dirty': dirty, 'pulled': not notes}
    if notes:
        log(event='Pop checkout pull failed; running its current commit', commit=commit, detail=' '.join(notes)[:300])
    if before != commit or not ssh(shlex.join(['test', '-f', POP_REPO + '/ts-host/dist/teacher/cli.js']) + ' && echo ok',
                                   check=False).strip():
        changed = ssh(git('diff', '--name-only', before, commit, '--', 'ts-host', 'scripts/stage-npm-package.mjs'), check=False)
        if changed.strip() or before == commit:
            build = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', '-o', 'BatchMode=yes', POP,
                                    f'cd {shlex.quote(POP_REPO)}/ts-host && npm run build:node'], capture_output=True,
                                   text=True, timeout=1800)
            result['collector_build'] = 'ok' if build.returncode == 0 else 'failed'
            if build.returncode:
                log(event='Pop collector build failed; keeping the previous dist', detail=(build.stderr or build.stdout)[-400:])
    return result


def _pop_upload_json(value, remote_path):
    ssh(shlex.join(['bash', '-c', f'cat > {shlex.quote(remote_path)}']), input=json.dumps(value))
    return remote_path


def remove_pop_work(paths):
    """Delete Pop work-directory entries under POP_EVAL_ROOT. The containers write them as root, so a throwaway
    container of the evaluation image removes them (Pop grants no sudo)."""
    root = PurePosixPath(POP_EVAL_ROOT)
    inside = [str(PurePosixPath('/r') / PurePosixPath(path).relative_to(root)) for path in paths]
    if not inside:
        return
    ssh(shlex.join(['docker', 'run', '--rm', '--name', f'pop-prune-{os.getpid()}-{int(time.time())}',
                    '--entrypoint', '/bin/rm', '-v', f'{POP_EVAL_ROOT}:/r', IMAGE, '-rf', '--', *inside]), timeout=1800)


def send(prep):
    from scripts.coreweave.send_actor_transport import send as send_transport
    receipt = send_transport(prep['transport']['path'], prep['transport']['sha256'], POP, POP_TRANSPORT_ROOT)
    write_json(ROOT / 'transfers' / f'{prep["transport"]["sha256"]}.json', receipt)
    log(event='sent', mode=receipt['mode'], payload_mb=round(receipt['payload_bytes'] / 2**20, 1),
        view=prep['view']['sha256'][:12])
    return receipt


def pop_unit_state(unit):
    out = ssh(shlex.join(['systemctl', '--user', 'show', unit + '.service', '-p', 'ActiveState', '-p', 'ExecMainStatus',
                          '-p', 'LoadState']), check=False)
    return dict(line.split('=', 1) for line in out.split('\n') if '=' in line)


POP_STORAGE = r'''
import json,os,shutil,subprocess,sys
from pathlib import Path, PurePosixPath
root=Path(sys.argv[1]);keep=set(json.loads(sys.argv[2]));need=int(sys.argv[3]);dry=sys.argv[4]=='1'
def free():s=os.statvfs(root);return s.f_bavail*s.f_frsize
hint=root/'.natlang-actor-transfer.json'
if hint.exists():keep.add(Path(json.loads(hint.read_text())['receiver_view']).parent.name)
busy=subprocess.run(['docker','ps','-q'],capture_output=True,text=True).stdout.strip()
removed=[]
if free()<need and not busy:
 stages=sorted((p for p in root.iterdir() if p.is_dir() and len(p.name)==64 and p.name not in keep),key=lambda p:p.stat().st_mtime)
 for stage in stages:
  if free()>=need:break
  if any(stage.glob('*.consumer.json')):continue
  for item in list(stage.glob('*.pt'))+[stage/'.zip-delta',stage/'.delta']:
   if not item.exists():continue
   size=item.stat().st_size if item.is_file() else sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
   removed.append({'path':str(item),'bytes':size})
   if not dry:
    shutil.rmtree(item) if item.is_dir() else item.unlink()
print(json.dumps({'free':free(),'need':need,'removed':removed,'busy':bool(busy),'keep':sorted(keep)}))
'''


def make_pop_room(incoming_bytes, keep):
    """Prune old Pop actor views (never the delta base or an in-use stage) until the next view plus the scratch floor
    fit. Only model-only views and delta caches under the transport root are removed; they are reproducible."""
    need = incoming_bytes + POP_MIN_FREE + 2**30
    result = pop_python(POP_STORAGE, POP_TRANSPORT_ROOT, json.dumps(sorted(keep)), need, '0')
    if result['removed']:
        write_json(ROOT / 'pop-retirement' / f'{int(time.time())}.json', {
            'schema': 'natlang.pop-actor-view-retirement/1', **result,
            'scope': 'derived model-only actor views/delta caches on Pop; sources, receipts and evidence untouched'})
        log(event='pruned Pop views', removed=len(result['removed']), free_gb=round(result['free'] / 2**30, 1))
    if result['free'] < need:
        raise RuntimeError(f'Pop has {result["free"] / 2**30:.1f} GiB free; needs {need / 2**30:.1f} GiB')


def prune_dgx(keep):
    """Remove older DGX actor views except ``keep`` and the peer delta base (sender view)."""
    hint = pop_python('import sys,json,os;p=sys.argv[1]+"/.natlang-actor-transfer.json";'
                      'print(json.dumps(json.load(open(p)) if os.path.exists(p) else None))', POP_TRANSPORT_ROOT)
    removed = []
    for kind in ('views', 'ema', 'rl-views'):
        for directory in sorted((ROOT / kind).glob('*')):
            if directory.name in keep or (hint and str(directory) in hint['sender_view']):
                continue
            for item in directory.glob('*.pt'):
                removed.append({'path': str(item), 'bytes': item.stat().st_size})
                item.unlink()
            delta = directory / '.transfer-deltas'
            if delta.exists():
                import shutil
                shutil.rmtree(delta)
                removed.append({'path': str(delta)})
    if removed:
        write_json(ROOT / 'dgx-retirement' / f'{int(time.time())}.json', {
            'schema': 'natlang.pop-eval-dgx-view-retirement/1', 'removed': removed, 'keep': sorted(keep),
            'scope': 'derived model-only views and delta caches; receipts and evidence kept'})


def pop_host_address():
    """The address the actor binds and the DGX dials: the SSH host's own address (Pop's tailscale IP)."""
    out = subprocess.run(['ssh', '-G', POP], capture_output=True, text=True, check=True).stdout
    return next(line.split()[1] for line in out.splitlines() if line.startswith('hostname '))


def attested(endpoint, policy_id):
    try:
        with urllib.request.urlopen(endpoint + '/v1/neuralese/info', timeout=5) as f:
            info = json.load(f)
    except OSError:
        return None
    binding = info.get('rl_policy_binding') or {}
    if binding.get('checkpoint_sha256') == policy_id and binding.get('actor_export'):
        return info
    return None


def pop_containers():
    return ssh("docker ps --format '{{.Names}}'", check=True).split()


def stop_unit(state):
    unit = state.get('unit')
    if unit:
        ssh(shlex.join(['systemctl', '--user', 'stop', unit + '.service']), check=False, timeout=120)
        for _ in range(30):
            if pop_unit_state(unit).get('ActiveState') in ('inactive', 'failed', None):
                break
            time.sleep(2)


def start(args):
    # A previous actor of this lineage that still serves the same policy (the controller restarted mid-collection)
    # keeps its block store and its code: an actor process keeps the code it started with, and the controller records
    # each actor's commit from its attestation. One serving another policy is replaced; retained-partial custody
    # rehydrates a fresh actor's store.
    previous = read_json(STATE)
    runtime = str(REPO)
    if previous and previous['policy_id'] == args.policy_id and attested(previous['endpoint'], args.policy_id):
        log(event='Pop RL actor already serves the policy', endpoint=previous['endpoint'])
        return previous
    if previous:
        stop_unit(previous)
    host = args.bind_host or pop_host_address()
    endpoint = f'http://{host}:{args.port}'
    ssh('true', timeout=30)
    busy = [name for name in pop_containers() if not name.startswith('poprl-')]
    if busy:
        raise RuntimeError(f'Pop GPU is held by {busy} (an evaluation owns it until it finishes)')
    started = time.time()
    code = {'dgx': code_on_dgx(), 'pop': sync_pop_checkout()}
    prep = prepare(args.checkpoint, args.policy_id, wait=args.prepare_wait)
    make_pop_room(prep['view']['bytes'], {prep['source']['sha256']})
    send(prep)
    source = prep['source']['sha256']
    if source != args.policy_id:
        raise RuntimeError('prepared view binds another source than the controller policy')
    # Keyed by policy only: the actor's code commit is provenance (job['code']), and a relaunch on newer code must keep
    # the block store, or blocks sampled before it are lost at bundling (w5 round 0, 2026-11-10).
    work = f'{POP_EVAL_ROOT}/rl-actor-{prep["step"]}-{source[:12]}'
    name = f'poprl-{source[:12]}'
    transport_json = json.loads(Path(prep['transport']['path']).read_text())
    job = {'mode': 'rl-actor', 'arm': 'rl', 'name': name, 'port': args.port, 'host': host, 'runtime': POP_REPO,
           'code': code['pop'], 'base': POP_BASE, 'image': IMAGE, 'step': prep['step'],
           'source_sha256': source, 'stage': f'{POP_TRANSPORT_ROOT}/{source}', 'view': prep['view']['path'],
           'view_sha256': prep['view']['sha256'], 'work': work, 'min_free_bytes': POP_MIN_FREE,
           'actor_export_format': SGLANG_EXPORT_FORMAT if args.engine == 'sglang' else EXPORT_FORMAT,
           'memory': f'{RL_MEMORY_GB}g', 'engine': args.engine,
           'transport': {'name': Path(prep['transport']['path']).name, 'sha256': prep['transport']['sha256'],
                         'view_receipt_name': prep['view_receipt']['sha256'] + '.json',
                         'view_receipt_sha256': prep['view_receipt']['sha256'], 'scope': transport_json['scope']}}
    # A restart under the same policy reuses this work dir (and its block store); the previous driver copy came from a
    # sealed read-only runtime file, so it is replaced rather than overwritten.
    ssh(shlex.join(['mkdir', '-p', work]) + ' && ' + shlex.join(['rm', '-f', f'{work}/driver.py']))
    # Older policies' actor dirs (block store, caches; 1-3 GB each) are spent once their rounds bundled their blocks to
    # the DGX. Keep this policy's and the newest earlier one (its pre-collection may still be bundling).
    earlier = ssh(f'ls -dt {shlex.quote(POP_EVAL_ROOT)}/rl-actor-* 2>/dev/null || true', check=False).split()
    remove_pop_work([path for path in earlier if path != work][1:])
    subprocess.run(['scp', '-q', str(REPO / 'scripts/neuralese_data/pop_actor_driver.py'), f'{POP}:{work}/driver.py'],
                   check=True, timeout=120)
    _pop_upload_json(job, f'{work}/job.json')
    unit = f'{name}-{int(time.time())}'
    # The SGLang actor runs on Pop's host (the fork's venv); the native one in its container.
    container = [] if args.engine == 'sglang' else ['--container', name]
    # The SGLang front holds the CPU-loaded view until the server is up, beside the server's own processes.
    budget = SGLANG_MEMORY_GB if args.engine == 'sglang' else RL_MEMORY_GB
    ssh(shlex.join(['/usr/bin/python3', POP_REPO + '/scripts/memory_ledger.py', 'run', '--unit', unit,
                        '--budget-gb', str(budget), '--reserve-gb', str(POP_RESERVE_GB),
                        '--class', 'experiment', '--no-resumable', *container, '--workdir', work, '--',
                        '/usr/bin/python3', f'{work}/driver.py', f'{work}/job.json']))
    result = {'endpoint': endpoint, 'unit': unit, 'container': name, 'policy_id': source, 'work': work, 'runtime': runtime,
              'engine': args.engine,
              'prepared_seconds': round(time.time() - started, 1)}
    write_json(STATE, result)
    try:
        prune_dgx({source})   # older derived DGX views; the one just sent stays as the next delta base
    except Exception as error:
        log(event='DGX view pruning failed', error=repr(error)[:300])
    deadline = time.monotonic() + args.ready_seconds
    while time.monotonic() < deadline:
        if pop_unit_state(unit).get('ActiveState') in ('inactive', 'failed'):
            tail = ssh(shlex.join(['journalctl', '--user', '-u', unit + '.service', '-n', '20', '--no-pager']), check=False)
            raise RuntimeError('Pop RL actor unit exited before readiness: ' + tail[-1200:])
        if attested(endpoint, source):
            result['ready_seconds'] = round(time.time() - started, 1)
            write_json(STATE, result)
            log(event='Pop RL actor ready', **result)
            return result
        time.sleep(5)
    raise RuntimeError(f'Pop RL actor did not attest {source[:12]} at {endpoint} within {args.ready_seconds} s')


def stop(_args=None):
    state = read_json(STATE)
    if state:
        stop_unit(state)
        ssh(shlex.join(['docker', 'rm', '-f', state['container']]), check=False, timeout=60)
        # The store stays: this hook also runs when a controller stops or crashes mid-round, before the collector has
        # bundled the round's blocks (2026-11-08: deleting it lost 22 native states of Pop-sampled rollouts). Only
        # compile caches are removed; the container wrote them as root, so from a throwaway container.
        work = state['work']
        ssh(shlex.join(['docker', 'run', '--rm', '-v', f'{work}:/work', 'alpine:3.21', 'sh', '-c',
                            'rm -rf ' + ' '.join(f'/work/{name}' for name in ('inductor', 'triton', 'hf', 'cuda'))]),
                check=False, timeout=120)
        STATE.unlink(missing_ok=True)
        log(event='Pop RL actor stopped', unit=state['unit'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='action', required=True)
    s = commands.add_parser('start')
    s.add_argument('--checkpoint', required=True)
    s.add_argument('--policy-id', required=True)
    s.add_argument('--port', type=int, default=RL_PORT)
    s.add_argument('--bind-host')
    s.add_argument('--ready-seconds', type=int, default=900)
    s.add_argument('--engine', choices=('sglang', 'native'), default='native',
                   help="Pop's serving engine: the SGLang fork (host venv, see pop_actor_driver.serve_sglang) or the "
                        'native engine in its container')
    s.add_argument('--prepare-wait', type=int, default=900, help='ledger admission bound of the DGX view preparation')
    commands.add_parser('stop')
    commands.add_parser('status')
    p = commands.add_parser('prepare-view', help=argparse.SUPPRESS)
    for flag in ('--checkpoint', '--out', '--receipt'):
        p.add_argument(flag, required=True)
    p.add_argument('--expect-sha256')
    args = parser.parse_args(argv)
    if args.action == 'status':
        print(json.dumps({'state': read_json(STATE)}))
        return 0
    if args.action == 'prepare-view':
        prepare_view(args)
        return 0
    try:
        if args.action == 'start':
            print(json.dumps(start(args)), flush=True)
        else:
            stop()
    except BaseException as error:   # a failed start never fails the caller: its attestation poll drops the endpoint
        log(event=f'Pop RL actor {args.action} failed', error=repr(error)[:2000])
        if args.action == 'start':
            try:
                stop()
            except Exception as cleanup:
                log(event='Pop RL actor cleanup failed', error=repr(cleanup)[:500])
        if isinstance(error, KeyboardInterrupt):
            raise
    return 0


if __name__ == '__main__':
    sys.exit(main())
