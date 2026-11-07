#!/usr/bin/env python3
"""Automated behavioural evaluation of newly earned LFM3 checkpoints on the Pop GPU (fallow register #6/#12).

The DGX watcher turns every window-boundary checkpoint of a production learner (and any checkpoint explicitly marked
for evaluation) into a paired protected-development comparison against the current behavioural parent, so parent
selection uses real task outcomes without manual steps. Operator note: plans/neuralese/POP_BEHAVIOR_EVAL.md.

    pop_behavior_eval.py watch [--poll 300]        the service loop (ledger unit pop-behavior-eval)
    pop_behavior_eval.py once                      one discovery + processing pass, then exit
    pop_behavior_eval.py request CHECKPOINT [--runtime R] [--label L] [--no-ema]
    pop_behavior_eval.py status

Triggers. (1) A controller ``W/<lineage>/state.json`` whose ``last_learner_progress`` names a committed checkpoint
SHA not seen before (the learner writes it at each window boundary/stop); its runtime is the lineage config's
``actor_environment.NATLANG_REPO``. Controllers already present when the watcher was initialised are history, not
work. (2) A request file ``<root>/requests/*.json`` (``request`` writes one). One evaluation runs at a time; when a
newer candidate is queued, older queued candidates are recorded as superseded and skipped.

Per candidate, every step is a separate low-priority ledger unit or an ssh/scp exchange, and every step is
restartable (outputs are content addressed and reused):
  1. prepare the model-only actor view inside the checkpoint's own runtime (pop_eval_prepare.py raw); the source is
     read in place, never copied, pinned or written;
  2. make room on Pop (keep the behavioural parent's view and the peer delta base) and send the view with
     scripts/coreweave/send_actor_transport.py (ZIP delta against the last view Pop has, else full);
  3. run pop_dev16_driver.py on Pop under Pop's memory ledger: the baseline spec (sha f28b1b8c..., cases
     range(0,128,8), one worker, full prompt, 2048 prefill, 0.25 GiB prefix cache, same image and container limits);
  4. collect completed.json and a minimal per-case outcome detail (case id/outcome only) back to the DGX;
  5. make sure the behavioural parent has a receipt under the SAME runtime (promotion pairs only identical runtimes),
     evaluating it once per runtime when missing;
  6. write the paired train/promotion.py comparison and a parent selection; a promoted candidate becomes the new
     behavioural parent (``<root>/behavioural-parent.json``, the pointer the lfm-latest profile follows);
  7. when the checkpoint carries a weight EMA and no newer candidate waits, evaluate the EMA as a second arm
     (pop_eval_prepare.py ema) and compare it with the parent too. An EMA export is evidence only, never a parent.

Receipts are protected-development evidence (training_admission false); the packet's content never enters training,
and only case ids and outcome labels are copied back. Production learners and checkpoints are only read.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.common.hashing import sha256_file_hex as digest  # noqa: E402
from natlang_neuralese.common.jsonio import write_json_durable as write_json  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

W = Path(os.environ.get('NATLANG_LFM3_WORKSPACE', resolve_str('data_nvme', 'lfm3-rl-dgx-20261029')))
ROOT = Path(os.environ.get('NATLANG_POP_EVAL_ROOT', str(W / 'ops/pop-behavior-eval')))
PYTHON = resolve_str('repo', '.venv-neuralese/bin/python')
LEDGER = str(REPO / 'scripts/memory_ledger.py')
HF_BASE = str(Path.home()) + '/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'

# Pop's SSH host: NATLANG_POP_HOST, else "pop_host" in <root>/config.json. Pop mirrors the DGX data/repo layout.
POP = os.environ.get('NATLANG_POP_HOST') or (json.loads((ROOT / 'config.json').read_text()).get('pop_host')
                                             if (ROOT / 'config.json').is_file() else None)
POP_REPO = resolve_str('repo')
POP_RECOVERY = resolve_str('data_nvme', 'natlang-independent-recovery-20261031')
POP_TRANSPORT_ROOT = POP_RECOVERY + '/matched75-transport-0f8a-1101'   # the peer cache whose delta base Pop holds
POP_EVAL_ROOT = POP_RECOVERY + '/pop-behavior-eval'
POP_BASE = POP_RECOVERY + '/base-config-only-40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
POP_PACKET = POP_RECOVERY + '/matched-instruction-baseline-1101/cases.jsonl'
PACKET_SHA256 = '6fa6ca6c2e3a46c4b050da535e1a246c8f34d70f355ea047bcfe19a0653aa560'
IMAGE = 'natlang:pop-causal-conv1d-torch214-cu134-sm89-v1'
POP_PORT = 18530
EMA_ADMISSION_WAIT = 600
# Raw views peaked at 7.2-8.0 GiB (8.0 at the cap: checkpoint page cache); 8 + the 8 GiB reserve missed a 15.9 GiB
# DGX headroom beside the 96 GiB learner by 0.1 GiB (2026-11-06).
RAW_PREP_GB = 7.5
POP_BUDGET_GB = 5.0  # the container is capped at --memory 5g; Pop (14 GiB) shows ~6.8 GiB free after ~4.4 GiB kernel slab, beside a 1.4 GiB reserve (2026-11-06)
# The hard container cap bounds the unit, so the ledger reserve beside it can be small: by 2026-11-07 Pop's kernel held
# ~7.5 GiB (3.7 GiB unreclaimable slab + 3.7 GiB vmalloc, no sudo to reboot) and 6.0 GiB free refused 5.0 + 1.2 default.
POP_RESERVE_GB = 0.5
POP_MIN_FREE = 8 * 2**30 + 512 * 2**20       # the baseline drivers' scratch floor
CASES = list(range(0, 128, 8))
# The exact task spec of the dev16 baselines (parent67 R34 12/16, earned179 R38 11/16).
SPEC = {'packet': {'path': '/packet/cases.jsonl', 'sha256': PACKET_SHA256}, 'cases': CASES, 'runtime': '/repo/ts-host',
        'prompt_rendering': 'full', 'case_seconds': 1200, 'max_model_requests': 64, 'max_turns': 1000, 'workers': 1,
        'min_successes': 0, 'reference_successes': None, 'noise_successes': 1, 'noise_repeats': 1,
        'neuralese_server': True, 'runtime_failure_policy': 'report', 'min_free_mib': 8192}
SPEC_SHA256 = 'f28b1b8c8f8e3e90d1c67196cccbc9ca749aca92802253a47dcae35335846d36'
# A candidate replaces the parent only when it solves strictly more of the 16 cases: a tie keeps the parent
# (one-success flips are within the measured repeat noise of this packet).
DECLARATION = {'schema': 'natlang.behavior-promotion-declaration/1',
               'packet': {'path': POP_PACKET, 'sha256': PACKET_SHA256}, 'cases': CASES, 'compare_to': 'parent',
               'min_delta': 1, 'max_case_regressions': None, 'confirmation': None,
               'note': 'automated Pop dev16 parent selection (scripts/neuralese_data/pop_behavior_eval.py)'}
STAGE = 'behavioural-parent'
# Initial behavioural parent: full67 (BF95 quality source, 12/16 under R34), ops/behavior-parent-reselection-2157-1102.
INITIAL_PARENT = {'sha256': '71c3eb45594dd866d2c3d935a5577188e7e0750bf39cb2484ebbe8e4c7d9a83b', 'step': 67,
                  'label': 'full67', 'path': str(W / 'ops/bf95-quality-source-zero-sgd-v2-1102/checkpoint.pt'),
                  'selection': str(W / 'ops/behavior-parent-reselection-2157-1102.json')}


def log(**event):
    print(json.dumps({'at': round(time.time(), 1), **event}, sort_keys=True), flush=True)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


# ---- state -------------------------------------------------------------------------------------------------------

def load_state():
    for name in ('requests', 'preparations', 'evidence', 'comparisons', 'selections', 'transfers', 'pop-retirement',
                 'dgx-retirement'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    state = read_json(ROOT / 'state.json')
    if state is None:
        state = {'schema': 'natlang.pop-behavior-eval-state/1', 'initialised': time.time(), 'parent': INITIAL_PARENT,
                 'controllers': {}, 'candidates': {}, 'parent_evals': {}}
        # Controllers present now are history: their last checkpoints were evaluated (or deliberately not) by hand.
        for path, progress in controller_progress():
            state['controllers'][str(path)] = progress['checkpoint_sha256']
        write_json(ROOT / 'declaration.json', DECLARATION)
        write_parent_pointer(state['parent'])
        save_state(state)
    return state


def save_state(state):
    write_json(ROOT / 'state.json', state)


def write_parent_pointer(parent):
    path = Path(parent['path'])
    write_json(ROOT / 'behavioural-parent.json', {
        'schema': 'natlang.behavioural-parent-pointer/1', 'label': parent.get('label'),
        'checkpoint': {'path': str(path), 'sha256': parent['sha256'],
                       'bytes': path.stat().st_size if path.exists() else None},
        'global_step': parent.get('step'), 'selection': parent.get('selection'), 'updated': time.time(),
        'scope': 'current behavioural parent by paired protected-development outcomes (Pop dev16); '
                 'read by the lfm-latest native source and by parent selection'})


# ---- discovery ---------------------------------------------------------------------------------------------------

# Cloud SFT-window lineages (rl_cloud.py): the DGX hub copy of each window controller, restored from DGX-verified
# exports only (rl_cloud export), at <run root>/<lineage>/w<NNNNNN>/state.json beside its config.json.
CLOUD = Path(os.environ.get('NATLANG_CLOUD_RUNS', resolve_str('data_hdd')))


def controller_states():
    yield from sorted(W.glob('*/state.json'))
    # The single-learner production line: ops/solo-<date>/<out|wN>/state.json, one controller per SFT window.
    yield from sorted(W.glob('ops/solo-*/out/state.json'))
    yield from sorted(W.glob('ops/solo-*/w[0-9]*/state.json'))
    # The bf95 low-LR restart (2026-11-05): window 0 is its own directory, later windows wN beneath it.
    yield from sorted(W.glob('ops/bf95-lowlr-*/state.json'))
    yield from sorted(W.glob('ops/bf95-lowlr-*/w[0-9]*/state.json'))
    # The interleaved successor (interleave-v49): its controller wN holds the RL rounds, each SFT window runs in
    # wN/sft-window-NNNNNN with its own state and config.
    yield from sorted(W.glob('ops/bf95-lowlr-*/w[0-9]*/sft-window-[0-9]*/state.json'))
    yield from sorted(CLOUD.glob('natlang-lfm3-cloud-*/*/w[0-9][0-9][0-9][0-9][0-9][0-9]/state.json'))


def controller_progress():
    for state_path in controller_states():
        try:
            progress = (read_json(state_path) or {}).get('last_learner_progress') or {}
        except ValueError:
            continue
        if isinstance(progress.get('checkpoint_sha256'), str) and isinstance(progress.get('checkpoint'), str):
            yield state_path, progress


def lineage_runtime(state_path):
    # A cloud/solo window keeps its control beside its state (the solo first window beside its run root); a
    # controller directly under the workspace has its config under ops/<name>.
    candidates = [state_path.parent / 'config.json', state_path.parent.parent / 'config.json'] \
        if state_path.parent.parent != W else []
    candidates.append(W / 'ops' / state_path.parent.name / 'config.json')
    config = read_json(next((c for c in candidates if c.is_file()), candidates[-1])) or {}
    runtime = (config.get('actor_environment') or {}).get('NATLANG_REPO')
    return runtime if runtime and Path(runtime, 'runtime-input-closure.json').is_file() else None


def checkpoint_global_step(progress, checkpoint):
    """Global step from the learner's train.jsonl tail (progress 'step' counts window-local updates)."""
    path = Path(checkpoint).parent / 'train.jsonl'
    try:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - (1 << 20)))
            for line in reversed(f.read().splitlines()):
                if b'"global_step"' in line:
                    return json.loads(line).get('global_step')
    except (OSError, ValueError):
        pass
    return progress.get('step')


def discover(state):
    for state_path, progress in controller_progress():
        key, sha = str(state_path), progress['checkpoint_sha256']
        if state['controllers'].get(key) == sha:
            continue
        state['controllers'][key] = sha
        if sha in state['candidates'] or sha == state['parent']['sha256']:
            continue
        runtime = lineage_runtime(state_path)
        state['candidates'][sha] = {
            'sha256': sha, 'path': progress['checkpoint'], 'runtime': runtime,
            'lineage': state_path.parent.name if state_path.parent.parent == W else
            f'{state_path.parent.parent.parent.name}/{state_path.parent.parent.name}/{state_path.parent.name}',
            'step': checkpoint_global_step(progress, progress['checkpoint']), 'trigger': 'window-boundary',
            'discovered': time.time(), 'status': 'queued' if runtime else 'blocked',
            **({} if runtime else {'reason': 'lineage config names no frozen runtime'}), 'ema': True}
        log(event='discovered', sha256=sha, lineage=state_path.parent.name)
    for request in sorted((ROOT / 'requests').glob('*.json')):
        value = read_json(request)
        done = request.with_suffix('.accepted')
        request.rename(done)
        sha = value.get('sha256') or digest(value['checkpoint'])
        if sha in state['candidates'] and state['candidates'][sha]['status'] not in ('superseded', 'failed'):
            continue
        state['candidates'][sha] = {
            'sha256': sha, 'path': value['checkpoint'], 'runtime': value['runtime'], 'lineage': value.get('label'),
            'step': value.get('step'), 'trigger': 'request', 'discovered': time.time(), 'status': 'queued',
            'ema': value.get('ema', True), 'request': str(done)}
        log(event='requested', sha256=sha, checkpoint=value['checkpoint'])
    save_state(state)


def pick(state):
    running = [c for c in state['candidates'].values() if c['status'] == 'running']
    if running:   # a restarted watcher resumes its in-flight candidate; its finished steps are reused
        return running[0]
    queued = sorted((c for c in state['candidates'].values() if c['status'] == 'queued'), key=lambda c: c['discovered'])
    if not queued:
        return None
    chosen = queued[-1]
    for other in queued[:-1]:
        if other['trigger'] != 'request':   # explicitly marked checkpoints are never dropped
            other.update(status='superseded', reason=f'newer candidate {chosen["sha256"][:12]} queued')
            log(event='superseded', sha256=other['sha256'])
    return chosen


def eval_pending(state):
    """True when an evaluation holds or awaits Pop: a queued/running candidate, or a checkpoint the watcher has not
    discovered yet (read-only: nothing is written to the state)."""
    if any(c['status'] in ('queued', 'running') for c in state['candidates'].values()):
        return True
    return any(state['controllers'].get(str(path)) != progress['checkpoint_sha256']
               and progress['checkpoint_sha256'] not in state['candidates']
               and progress['checkpoint_sha256'] != state['parent']['sha256']
               for path, progress in controller_progress())


def newer_waiting(state, candidate):
    return any(c['status'] == 'queued' and c['discovered'] > candidate['discovered'] for c in state['candidates'].values())


# ---- DGX ledger units --------------------------------------------------------------------------------------------

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


def prepare(kind, checkpoint, runtime, expect_sha, wait=None):
    """Preparation receipt of one actor view; kind rl is the controller's own policy for a Pop RL rollout actor
    (scripts/neuralese_data/pop_rl_actor.py; own view directory, scope rl-controller-pin); ``wait`` bounds admission."""
    out = ROOT / {'raw': 'views', 'rl': 'rl-views', 'ema': 'ema'}[kind]
    receipt = ROOT / 'preparations' / f'{kind}-{expect_sha}.json'
    existing = read_json(receipt)
    if existing and Path(existing['transport']['path']).is_file():
        return existing
    command = [PYTHON, str(REPO / 'scripts/neuralese_data/pop_eval_prepare.py'), kind, '--checkpoint', str(checkpoint),
               '--out', str(out), '--receipt', str(receipt), '--expect-sha256', expect_sha]
    if kind == 'ema':
        command += ['--base', HF_BASE]
    env = [f'NATLANG_REPO={runtime}', f'PYTHONPATH={runtime}/training/neuralese:{runtime}', 'OMP_NUM_THREADS=4',
           f'NATLANG_HASH_CACHE={ROOT}/hash-cache.jsonl', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1']
    try:
        # The EMA arm needs 40 GiB on the DGX, which never fits beside a live learner: a bounded wait, then skip it
        # (2026-11-06: a 24 h admission wait stalled every later candidate behind one EMA view).
        unit, status, result = ledger_unit(f'popeval-{kind}-{expect_sha[:12]}',
                                           ['/usr/bin/ionice', '-c2', '-n6', *command],
                                           budget_gb=40 if kind == 'ema' else RAW_PREP_GB, workdir=runtime, env=env,
                                           **({'wait': wait} if wait else {'wait': EMA_ADMISSION_WAIT} if kind == 'ema' else {}))
    except subprocess.CalledProcessError:
        if kind != 'ema':
            raise
        return {'absent': True, 'skipped': f'not admitted within {EMA_ADMISSION_WAIT} s beside the learner'}
    if status == 3 and kind == 'ema':
        return {'absent': True, 'unit': unit}
    if status != 0 or not receipt.is_file():
        raise RuntimeError(f'{kind} preparation unit {unit} failed ({result}, exit {status}); see journalctl --user -u {unit}')
    return read_json(receipt)


# ---- Pop ---------------------------------------------------------------------------------------------------------

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


def runtime_closure_sha(runtime):
    return digest(Path(runtime) / 'runtime-input-closure.json')


def runtime_files(runtime):
    root = Path(runtime)
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and not p.is_symlink())


def ensure_pop_runtime(runtime):
    """Install the checkpoint's frozen runtime on Pop at the same path when it is missing (exact file inventory)."""
    closure = runtime_closure_sha(runtime)
    probe = pop_python('import sys,os,hashlib,json\np=sys.argv[1]+"/runtime-input-closure.json"\n'
                       'print(json.dumps(hashlib.sha256(open(p,"rb").read()).hexdigest() if os.path.isfile(p) else None))',
                       runtime)
    if probe == closure:
        return closure
    if probe is not None:
        raise RuntimeError(f'Pop holds another runtime at {runtime}; preserve it and investigate')
    log(event='installing runtime on Pop', runtime=runtime)
    parent, name = str(Path(runtime).parent), Path(runtime).name
    staging = f'{runtime}.installing'
    ssh(shlex.join(['mkdir', '-p', parent]))
    ssh(shlex.join(['mkdir', staging]))   # fails when a previous attempt left it: inspect rather than overwrite
    tar = subprocess.Popen(['tar', '--zstd', '-C', parent, '-cf', '-', name], stdout=subprocess.PIPE)
    subprocess.run(['ssh', '-o', 'ConnectTimeout=15', POP, shlex.join(['tar', '--zstd', '-C', staging, '-xf', '-'])],
                   stdin=tar.stdout, check=True, timeout=3600)
    if tar.wait():
        raise RuntimeError('runtime archive failed')
    manifest = _pop_upload_json({path: digest(Path(runtime) / path) for path in runtime_files(runtime)},
                                f'{staging}.manifest.json')
    pop_python(INSTALL_RUNTIME, staging, name, manifest, runtime, timeout=3600)
    log(event='runtime installed on Pop', runtime=runtime)
    return closure


INSTALL_RUNTIME = r'''
import hashlib,json,os,sys
from pathlib import Path
staging,name,manifest,target=sys.argv[1:5];root=Path(staging)/name
want=json.loads(Path(manifest).read_text())
have={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and not p.is_symlink()}
assert have==want,'installed runtime inventory differs from the DGX runtime'
os.rename(root,target);os.rmdir(staging);os.unlink(manifest);print('{}')
'''


def _pop_upload_json(value, remote_path):
    ssh(shlex.join(['bash', '-c', f'cat > {shlex.quote(remote_path)}']), input=json.dumps(value))
    return remote_path


POP_STORAGE = r'''
import json,os,shutil,subprocess,sys
from pathlib import Path
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


def make_pop_room(state, incoming_bytes, keep):
    """Prune old Pop actor views (never the parent's, the delta base or an in-use stage) until the next view plus
    the evaluation scratch floor fit. Only model-only views and delta caches under the transport root are removed;
    they are reproducible from DGX sources and carry no evidence."""
    need = incoming_bytes + POP_MIN_FREE + 2**30
    keep = sorted(set(keep) | {state['parent']['sha256']})
    result = pop_python(POP_STORAGE, POP_TRANSPORT_ROOT, json.dumps(keep), need, '0')
    if result['removed']:
        write_json(ROOT / 'pop-retirement' / f'{int(time.time())}.json', {
            'schema': 'natlang.pop-actor-view-retirement/1', **result,
            'scope': 'derived model-only actor views/delta caches on Pop; sources, receipts and evidence untouched'})
        log(event='pruned Pop views', removed=len(result['removed']), free_gb=round(result['free'] / 2**30, 1))
    if result['free'] < need:
        raise RuntimeError(f'Pop has {result["free"] / 2**30:.1f} GiB free; needs {need / 2**30:.1f} GiB')


def prune_dgx(state):
    """Remove this watcher's own DGX views (and the Pop RL actor's rl-views) except the behavioural parent's and the peer
    delta base (sender view)."""
    keep = {state['parent']['sha256']}
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
            'scope': 'derived model-only views and delta caches of this watcher; receipts and evidence kept'})


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


def run_on_pop(arm, prep, runtime, closure, comparison, attempt=1):
    source = prep['source']['sha256']
    py = prep['python_runtime']['sha256']
    label = f'{arm}-{prep["step"]}-{source[:12]}-py{py[:12]}' + (f'-a{attempt}' if attempt > 1 else '')
    work = f'{POP_EVAL_ROOT}/{label}'
    name = f'popeval-{arm}-{source[:12]}'
    transport_json = json.loads(Path(prep['transport']['path']).read_text())
    job = {'arm': arm, 'name': name, 'port': POP_PORT, 'runtime': runtime, 'runtime_closure_sha256': closure,
           'base': POP_BASE, 'packet': POP_PACKET, 'packet_sha256': PACKET_SHA256, 'spec': SPEC,
           'spec_sha256': SPEC_SHA256, 'image': IMAGE, 'step': prep['step'], 'source_sha256': source,
           'stage': f'{POP_TRANSPORT_ROOT}/{source}', 'view': prep['view']['path'], 'view_sha256': prep['view']['sha256'],
           'work': work, 'min_free_bytes': POP_MIN_FREE, 'comparison': comparison,
           'transport': ({'name': Path(prep['transport']['path']).name, 'sha256': prep['transport']['sha256'],
                          'view_receipt_name': prep['view_receipt']['sha256'] + '.json',
                          'view_receipt_sha256': prep['view_receipt']['sha256'], 'scope': transport_json['scope']}
                         if arm != 'ema' else None)}
    done = pop_python('import sys,os,json;print(json.dumps(os.path.isfile(sys.argv[1]+"/evaluation/completed.json")))', work)
    if not done:
        # A restarted watcher adopts a still-running unit of this arm instead of launching a second one.
        active = ssh(shlex.join(['systemctl', '--user', 'list-units', '--type=service', '--state=active,activating',
                                 '--no-legend', '--plain', f'{name}-*']), check=False).split()
        unit = active[0].removesuffix('.service') if active else None
        if unit is None:
            driver = REPO / 'scripts/neuralese_data/pop_dev16_driver.py'
            ssh(shlex.join(['mkdir', '-p', work]))
            subprocess.run(['scp', '-q', str(driver), f'{POP}:{work}/driver.py'], check=True, timeout=120)
            _pop_upload_json(job, f'{work}/job.json')
            unit = f'{name}-{int(time.time())}'
            ssh(shlex.join(['/usr/bin/python3', POP_REPO + '/scripts/memory_ledger.py', 'run', '--unit', unit,
                            '--budget-gb', str(POP_BUDGET_GB), '--reserve-gb', str(POP_RESERVE_GB), '--class', 'experiment',
                            '--no-resumable', '--container', name,
                            '--workdir', work, '--', '/usr/bin/python3', f'{work}/driver.py', f'{work}/job.json']))
            log(event='Pop evaluation started', arm=arm, unit=unit, step=prep['step'], source=source[:12])
            time.sleep(20)
        while True:
            fields = pop_unit_state(unit)
            if fields.get('ActiveState') in ('inactive', 'failed'):
                break
            time.sleep(60)
        if fields.get('ExecMainStatus') not in ('0', None):
            tail = ssh(shlex.join(['journalctl', '--user', '-u', unit + '.service', '-n', '30', '--no-pager']), check=False)
            raise RuntimeError(f'Pop unit {unit} failed (exit {fields.get("ExecMainStatus")}): {tail[-1500:]}')
    return collect(label, work, prep)


DETAIL = r'''
import json,sys
from pathlib import Path
work=Path(sys.argv[1]);step=int(sys.argv[2])
score=json.loads((work/'evaluation/task-eval'/('step-%07d'%step)/'task-score.json').read_text())
keep=('schema','step','packet_sha256','case_indices','task_successes','cases','outcome_counts','eligible','complete')
detail={k:score.get(k) for k in keep}
detail['cases_detail']=[{'id':r.get('id'),'family':r.get('family'),'outcome':r.get('outcome')} for r in score['cases_detail']]
print(json.dumps(detail))
'''


def collect(label, work, prep):
    out = ROOT / 'evidence' / label
    out.mkdir(parents=True, exist_ok=True)
    for name in ('evaluation/completed.json', 'launch.json'):
        subprocess.run(['scp', '-q', f'{POP}:{work}/{name}', str(out / Path(name).name)], check=True, timeout=120)
    detail = pop_python(DETAIL, work, prep['step'])
    write_json(out / 'detail.json', {**detail, 'scope': 'per-case outcome labels only; protected packet content stays '
                                                         'on Pop; evidence, never training data'})
    completed = read_json(out / 'completed.json')
    log(event='collected', label=label, successes=completed['summary']['task_successes'], cases=completed['summary']['cases'])
    return {'label': label, 'completed': str(out / 'completed.json'), 'detail': str(out / 'detail.json'),
            'successes': completed['summary']['task_successes'], 'cases': completed['summary']['cases'],
            'python_runtime_sha256': prep['python_runtime']['sha256'], 'checkpoint_sha256': prep['source']['sha256'],
            'step': prep['step']}


# A Pop RL rollout actor (pop_rl_actor.py start) owns Pop's one GPU from its start to its stop; the controller's
# teardown hook removes the marker. A marker older than this is a crashed collection, not a live one.
RL_ACTOR_MARKER = ROOT / 'rl-actor-active.json'
RL_ACTOR_MAX_SECONDS = 6 * 3600


def rl_actor_active():
    marker = read_json(RL_ACTOR_MARKER)
    return bool(marker) and time.time() - marker['started'] < RL_ACTOR_MAX_SECONDS


def wait_for_rl_actor():
    """Rollouts have priority over evaluations: wait until no RL actor holds Pop."""
    announced = False
    while rl_actor_active():
        if not announced:
            log(event='waiting for the Pop RL actor to stop', marker=read_json(RL_ACTOR_MARKER))
            announced = True
        time.sleep(60)


def evaluate(state, arm, checkpoint, sha, runtime, comparison, attempt=1):
    wait_for_rl_actor()
    closure = ensure_pop_runtime(runtime)
    kind = 'ema' if arm == 'ema' else 'raw'
    prep = prepare(kind, checkpoint, runtime, sha)
    if prep.get('absent'):
        return None
    keep = {prep['source']['sha256']}
    make_pop_room(state, prep['view']['bytes'], keep)
    send(prep)
    wait_for_rl_actor()
    return run_on_pop(arm, prep, runtime, closure, comparison, attempt)


# An evaluation whose summary is not eligible (a case ended unresolved or as an infrastructure error, e.g. a wall
# timeout) has no task_successes: it is not a model outcome and cannot be paired. It is rerun under a fresh label,
# keeping the earlier evidence, up to this many attempts; the last result is compared (and reported unmatched).
MAX_EVAL_ATTEMPTS = 2


def scored_evaluation(state, arm, checkpoint, sha, runtime, comparison, existing=None):
    result = existing
    while result is None or (result['successes'] is None and result.get('attempt', 1) < MAX_EVAL_ATTEMPTS):
        attempt = result.get('attempt', 1) + 1 if result is not None else 1
        if result is not None:
            log(event='unscored evaluation rerun', arm=arm, sha256=sha, attempt=attempt, previous=result['label'])
        result = evaluate(state, arm, checkpoint, sha, runtime, comparison, attempt)
        if result is None:
            return None
        result['attempt'] = attempt
    return result


# ---- comparison / selection --------------------------------------------------------------------------------------

def compare(candidate_eval, parent_eval, phase_label):
    from natlang_neuralese.train.promotion import compare as promotion_compare, select_parent
    sources = {name: {'path': value, 'sha256': digest(value)} for name, value in (
        ('candidate', candidate_eval['completed']), ('candidate_detail', candidate_eval['detail']),
        ('reference', parent_eval['completed']), ('reference_detail', parent_eval['detail']))}
    receipt = promotion_compare(DECLARATION, phase='development', candidate=read_json(candidate_eval['completed']),
                                reference=read_json(parent_eval['completed']),
                                candidate_detail=read_json(candidate_eval['detail']),
                                reference_detail=read_json(parent_eval['detail']), sources=sources)
    name = f'{phase_label}-{candidate_eval["checkpoint_sha256"][:12]}-vs-{parent_eval["checkpoint_sha256"][:12]}'
    path = ROOT / 'comparisons' / f'{name}.json'
    write_json(path, receipt)
    log(event='compared', arm=phase_label, candidate=candidate_eval['successes'], parent=parent_eval['successes'],
        matched=receipt['matched'], promoted=receipt['promoted'], receipt=str(path))
    return receipt, path, select_parent


def process(state, candidate):
    sha, runtime = candidate['sha256'], candidate['runtime']
    candidate['status'] = 'running'
    # Read the learner's immutable actor-view source pin of this exact checkpoint when it has one (the rolling slot
    # may already hold a newer step); never create one.
    pin = Path(candidate['path']).parent / f'.checkpoint.pt.actor-view-source-{sha}'
    source = str(pin) if pin.is_file() else candidate['path']
    save_state(state)
    parent = state['parent']
    raw = scored_evaluation(state, 'raw', source, sha, runtime,
        f'Automated Pop dev16 evaluation of {candidate.get("lineage")} step {candidate.get("step")} under its own '
        f'runtime {Path(runtime).name}; paired against behavioural parent {parent["label"]}.', candidate.get('raw'))
    candidate['raw'] = raw
    save_state(state)
    key = f'{parent["sha256"]}:{raw["python_runtime_sha256"]}'
    parent_eval = state['parent_evals'].get(key)
    if parent_eval is not None and parent_eval.get('python_runtime_sha256') != raw['python_runtime_sha256']:
        parent_eval = None   # an entry filed under another runtime's key is not a runtime-matched arm: evaluate afresh
    if parent_eval is None or parent_eval['successes'] is None:
        if not Path(parent['path']).is_file():
            raise RuntimeError(f'behavioural parent source {parent["path"]} is unavailable for the runtime-matched arm')
        parent_eval = scored_evaluation(state, 'parent', parent['path'], parent['sha256'], runtime,
            f'Behavioural parent {parent["label"]} re-evaluated under runtime {Path(runtime).name} so candidates of '
            f'this runtime pair under identical limits/runtime.', parent_eval)
        state['parent_evals'][key] = parent_eval
        save_state(state)
    receipt, path, select_parent = compare(raw, parent_eval, 'raw')
    candidate['comparison'] = str(path)
    # A lineage that runs memetic meta-learning (native_memetic_rl) gates its parent on the paired held-program verdict
    # too: NATLANG_MEMETIC_PROMOTION_HISTORY names that controller's memetic/promotion/history.json.
    memetic = os.environ.get('NATLANG_MEMETIC_PROMOTION_HISTORY')
    selection = select_parent(STAGE, [receipt], candidate_path=candidate['path'], reference_path=parent['path'],
                              memetic_history=json.loads(Path(memetic).read_text()) if memetic else None)
    selection_path = ROOT / 'selections' / f'{sha[:12]}-vs-{parent["sha256"][:12]}.json'
    write_json(selection_path, selection)
    candidate['selection'] = str(selection_path)
    if selection['candidate_promoted']:
        # The parent must stay readable for later runtime-matched arms: prefer the learner's immutable actor-view
        # source pin of this exact checkpoint over the rolling slot.
        state['parent'] = {'sha256': sha, 'step': candidate.get('step'), 'label': f'{candidate.get("lineage")}-{candidate.get("step")}',
                           'path': source, 'selection': str(selection_path)}
        state['parent_evals'][f'{sha}:{raw["python_runtime_sha256"]}'] = raw
        write_parent_pointer(state['parent'])
        log(event='promoted', sha256=sha, step=candidate.get('step'))
    save_state(state)
    if candidate.get('ema') and 'ema_eval' not in candidate:
        if newer_waiting(state, candidate):
            candidate['ema_eval'] = {'skipped': 'newer candidate queued'}
        else:
            ema = evaluate(state, 'ema', source, sha, runtime,
                f'Weight EMA of {candidate.get("lineage")} step {candidate.get("step")} (second arm, committed '
                f'decision #2) under runtime {Path(runtime).name}; paired against behavioural parent {parent["label"]}.')
            if ema is None:
                candidate['ema_eval'] = {'absent': True}
            else:
                ema_receipt, ema_path, _ = compare(ema, state['parent_evals'][key], 'ema')
                candidate['ema_eval'] = {**ema, 'comparison': str(ema_path), 'promoted_by_rule': ema_receipt['promoted'],
                                         'note': 'EMA export is evaluation evidence only, never a resumable parent'}
        save_state(state)
    candidate['status'] = 'done'
    candidate['finished'] = time.time()
    save_state(state)
    prune_dgx(state)


def one_pass():
    state = load_state()
    discover(state)
    candidate = pick(state)
    if candidate is None:
        save_state(state)
        return False
    save_state(state)
    try:
        process(state, candidate)
    except Exception as error:   # recorded and retried at most twice; the watcher itself keeps running
        candidate['failures'] = candidate.get('failures', 0) + 1
        candidate['last_error'] = f'{type(error).__name__}: {error}'[:3000]
        candidate['status'] = 'failed' if candidate['failures'] >= 3 else 'queued'
        save_state(state)
        log(event='candidate error', sha256=candidate['sha256'], error=candidate['last_error'][:500])
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    watch = commands.add_parser('watch')
    watch.add_argument('--poll', type=float, default=300)
    commands.add_parser('once')
    commands.add_parser('status')
    request = commands.add_parser('request')
    request.add_argument('checkpoint')
    request.add_argument('--runtime', required=True, help='frozen runtime that trained the checkpoint')
    request.add_argument('--label')
    request.add_argument('--step', type=int)
    request.add_argument('--no-ema', action='store_true')
    request.add_argument('--sha256', help='known SHA-256 of the checkpoint (otherwise hashed at discovery)')
    args = parser.parse_args(argv)
    os.environ.setdefault('NATLANG_HASH_CACHE', str(ROOT / 'hash-cache.jsonl'))
    if args.command == 'request':
        if not Path(args.runtime, 'runtime-input-closure.json').is_file():
            parser.error('--runtime must be a frozen runtime directory (runtime-input-closure.json)')
        value = {'checkpoint': str(Path(args.checkpoint).resolve(strict=True)), 'runtime': str(Path(args.runtime).resolve()),
                 'label': args.label, 'step': args.step, 'ema': not args.no_ema, 'requested': time.time(),
                 **({'sha256': args.sha256} if args.sha256 else {})}
        (ROOT / 'requests').mkdir(parents=True, exist_ok=True)
        write_json(ROOT / 'requests' / f'{int(time.time() * 1000)}.json', value)
        print(json.dumps(value))
    elif args.command == 'status':
        state = load_state()
        print(json.dumps({'parent': state['parent'], 'candidates': {
            sha[:12]: {k: c.get(k) for k in ('status', 'lineage', 'step', 'reason', 'last_error')}
            | {'raw': (c.get('raw') or {}).get('successes'), 'ema': (c.get('ema_eval') or {}).get('successes')}
            for sha, c in state['candidates'].items()}}, indent=1))
    elif args.command == 'once':
        one_pass()
    else:
        while True:
            if not one_pass():
                time.sleep(args.poll)


if __name__ == '__main__':
    main()
