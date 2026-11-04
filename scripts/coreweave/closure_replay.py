"""Replay an rl_cloud lineage start on the DGX inside a container that sees only the shipped closure.

Before a seed or cutover reaches a cloud learner, qualify it locally: the container mounts exactly the files the
sandbox would receive (the closure's files and historical checkpoint blobs at their DGX paths, the frozen runtime,
the Python venv, the HF snapshot and node), a writable copy of the lineage directory, and the parent checkpoint. It
runs the controller (`natlang_neuralese.rl.production run --rounds 1`) CPU-only with no network: the controller and
learner preflight execute until the learner needs CUDA for the model, so every missing input, identity mismatch or
custody refusal surfaces here instead of on a paid GPU. Read the controller log afterwards; a run that stops at the
CUDA model load has passed everything before it.

Usage:
  closure_replay.py --run RUN.json --lineage NAME --closure CLOSURE.json --parent CKPT --config CONFIG --work DIR \
      [--writable SRC:DST ...] [--unit NAME] [--build-only]

--config is the control to start, at its lineage path (a file inside the lineage directory, which the container
sees as the writable copy). --writable binds extra writable directories (e.g. the checkpoint-input offload scratch).
Run it under the memory ledger with the same --unit (memory_ledger.py run --container NAME).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts/coreweave'))

VENV = REPO / '.venv-neuralese'
HF = Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B'
NODE = Path.home() / '.nvm/versions/node/v22.22.0'
IMAGE = 'python:3.12-slim'


def mounts(run, lineage, closure, parent, work, writable, source=None, historical=None):
    """Docker mounts for one lineage. ``source`` stages a lineage tree other than ROOT/<lineage> (a seed not yet
    written there); ``historical`` = (blobs, links) the seed ships, when already resolved (e.g. ``historical_from``)."""
    import rl_cloud
    root = Path(run['root']) / lineage
    if historical is None:
        blobs, skipped, links = rl_cloud.historical_checkpoint_blobs(run, closure, root / 'parent' / 'checkpoint.pt')
    else:
        (blobs, links), skipped = historical, []
    runtime = closure['runtime']
    copy = work / lineage
    if copy.exists():
        shutil.rmtree(copy)
    copy.mkdir(parents=True)
    for p in Path(source or root).iterdir():
        if p.name != 'parent':
            if p.is_dir():
                shutil.copytree(p, copy / p.name, symlinks=True)
            else:
                shutil.copy2(p, copy / p.name, follow_symlinks=False)
    (copy / 'parent').mkdir()
    code = work / 'code'  # the git snapshot the sandbox bundle carries
    if code.exists():
        shutil.rmtree(code)
    code.mkdir()
    subprocess.run(f'git -C {REPO} archive HEAD | tar -x -C {code}', shell=True, check=True)
    out = ['-v', '/usr:/usr:ro', '-v', '/etc:/etc:ro', '-v', f'{code}:{REPO}']
    ro = lambda src, dst=None: out.extend(['-v', f'{src}:{dst or src}:ro'])
    for path in (VENV, HF, runtime, NODE):
        ro(path)
    # The HF cache's snapshot links can point into a shared dedup blob store outside the model directory (the
    # LFM3 safetensors does): mount every resolved target outside HF too, or the base weights are absent.
    for target in sorted({os.path.realpath(p) for p in HF.rglob('*') if p.is_symlink()}):
        if not target.startswith(str(HF) + '/') and os.path.isfile(target):
            ro(target)
    for f in closure['files']:
        ro(f['path'])
    for b in blobs:
        ro(b['path'])
    linkroot = work / 'links'
    if linkroot.exists():
        shutil.rmtree(linkroot)
    for link in links:
        d = linkroot / link['path'].lstrip('/')
        d.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(link['symlink_to'], d)
    for directory in sorted({str(Path(link['path']).parent) for link in links}):
        out.extend(['-v', f'{linkroot / directory.lstrip("/")}:{directory}'])
    out.extend(['-v', f'{copy}:{root}'])
    ro(parent, root / 'parent' / 'checkpoint.pt')
    state = work / 'state'
    state.mkdir(exist_ok=True)
    out.extend(['-v', f'{state}:/replay-state'])
    for spec in writable:
        src, dst = spec.split(':', 1)
        Path(src).mkdir(parents=True, exist_ok=True)
        out.extend(['-v', f'{src}:{dst}'])
    for t in ('/tmp', '/var/tmp'):
        out.extend(['--tmpfs', f'{t}:exec,size=16g'])
    return out, runtime, {'files': len(closure['files']), 'historical': [b['path'] for b in blobs],
                          'skipped_historical': [s['path'] for s in skipped], 'links': links}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    ap.add_argument('--lineage', required=True)
    ap.add_argument('--closure', required=True)
    ap.add_argument('--parent', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--writable', action='append', default=[])
    ap.add_argument('--unit', default='closure-replay')
    ap.add_argument('--build-only', action='store_true')
    a = ap.parse_args(argv)
    run = json.loads(Path(a.run).read_text())
    closure = json.loads(Path(a.closure).read_text())
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    out, runtime, summary = mounts(run, a.lineage, closure, Path(a.parent), work, a.writable)
    env = environment(runtime, Path(a.config).parent)
    for k, v in env.items():
        out.extend(['-e', f'{k}={v}'])
    command = ['docker', 'run', '--rm', '--name', a.unit, '--user', f'{os.getuid()}:{os.getgid()}',
               '--network', 'none', '--shm-size', '8g', *out, '-w', runtime, IMAGE,
               f'{VENV}/bin/python', '-u', '-m', 'natlang_neuralese.rl.production', 'run', '--config', a.config,
               '--rounds', '1']
    (work / 'docker.argv').write_text('\n'.join(command) + '\n')
    print(json.dumps({**summary, 'argv': str(work / 'docker.argv'), 'log': str(work / 'controller.log')}, indent=1),
          flush=True)
    if a.build_only:
        return 0
    with open(work / 'controller.log', 'wb') as log:
        return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode


def environment(runtime, window):
    """The controller environment of a sandbox start, CPU-only."""
    return {'NATLANG_REPO': runtime, 'PYTHONPATH': f'{runtime}/training/neuralese:{runtime}',
           'PATH': f'{NODE}/bin:{VENV}/bin:/usr/bin:/bin', 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
           'NATLANG_HASH_CACHE': '/replay-state/hash-cache.jsonl', 'OMP_NUM_THREADS': '1',
           'MKL_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'HOME': str(Path.home()),
           'NATLANG_CHECKPOINT_MIN_FREE_BYTES': '42949672960', 'NATLANG_DISCRETE_CUDA_MEMORY': '1',
           'NATLANG_NATIVE_PROFILE_FIRST_UPDATE': f'{window}/learner/first-update-profile.json',
           'NATLANG_POLICY_LOGPROB_RECEIPT': f'{window}/learner/behavior-replay-logprobs.json',
           'NATLANG_ROW_INDEX_CACHE': str(Path.home() / '.cache/natlang/row-index')}


if __name__ == '__main__':
    sys.exit(main())
