#!/usr/bin/env python3
"""Pop side of the automated behavioural evaluation (scripts/neuralese_data/pop_behavior_eval.py).

Runs ONE protected dev16 evaluation of one actor on Pop's GPU, described by a job JSON the DGX watcher wrote:

    pop_dev16_driver.py JOB.json

Generalizes the hand-written ``run-pop-expanded-dev16-*.py`` drivers (same image, container limits, serving flags and
task spec) to a job file, so every arm runs under exactly the baseline contract. The job names the actor either as an
exact actor transport (``transport`` set: model-only view of a full training checkpoint) or as a plain model-only
checkpoint (``transport`` null: a weight-EMA export). Stdlib only (Pop's host Python is 3.10).

Protected development measurement only: outputs never enter training (``training_admission`` false in the receipt).

``job['mode'] == 'rl-actor'`` (scripts/neuralese_data/pop_rl_actor.py) serves the same actor as an RL rollout actor
instead: same image, container limits and serving flags, bound to ``job['host']``, binding ``--actor-export-format``
(the learner guards the cross-device engine mismatch), keeping the container until it exits or the unit is stopped.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(job_path):
    job = json.loads(Path(job_path).read_text())
    rl = job.get('mode') == 'rl-actor'
    runtime, base = Path(job['runtime']), Path(job['base'])
    stage, work = Path(job['stage']), Path(job['work'])
    work.mkdir(parents=True, exist_ok=True)
    if not rl and (work / 'evaluation/completed.json').exists():
        print(json.dumps({'state': 'already completed', 'work': str(work)}), flush=True)
        return
    closure = runtime / 'runtime-input-closure.json'
    if digest(closure) != job['runtime_closure_sha256']:
        raise RuntimeError('Pop runtime closure differs from the checkpoint runtime')
    spec_sha = raw = packet = None
    if not rl:
        packet = Path(job['packet'])
        if digest(packet) != job['packet_sha256']:
            raise RuntimeError('protected packet differs from its binding')
        raw = (json.dumps(job['spec'], indent=2, sort_keys=True) + '\n').encode()
        spec_sha = hashlib.sha256(raw).hexdigest()
        if spec_sha != job['spec_sha256']:
            raise RuntimeError('task spec differs from the baseline spec: ' + spec_sha)
    checkpoint = stage / job['view']
    transport = job.get('transport')
    if transport:
        manifest_path = stage / transport['name']
        if digest(manifest_path) != transport['sha256']:
            raise RuntimeError('transport receipt differs')
        manifest = json.loads(manifest_path.read_text())
        if manifest['source']['sha256'] != job['source_sha256'] or manifest['scope'] != transport['scope']:
            raise RuntimeError('transport binds another source or scope')
        if digest(stage / transport['view_receipt_name']) != transport['view_receipt_sha256']:
            raise RuntimeError('transport view receipt differs')
    free = os.statvfs(work).f_bavail * os.statvfs(work).f_frsize
    if free < job['min_free_bytes']:
        raise RuntimeError(f'insufficient Pop scratch headroom: {free} bytes free')
    if digest(checkpoint) != job['view_sha256']:
        raise RuntimeError('actor bytes differ from their binding')
    gpu = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if gpu:
        raise RuntimeError('Pop GPU already has a compute consumer: ' + gpu)
    if subprocess.check_output(['docker', 'ps', '-q'], text=True).strip():
        raise RuntimeError('another container is running on Pop')
    if not rl:
        (work / 'spec.json').write_bytes(raw)
    name, port, image = job['name'], job['port'], job['image']
    server_args = ['serve_native_checkpoint.py', '--checkpoint', '/checkpoint.pt', '--checkpoint-sha256', job['view_sha256']]
    if transport:
        server_args += ['--transport-receipt', '/transport/' + transport['name'],
                        '--transport-receipt-sha256', transport['sha256'],
                        '--transport-view-receipt', '/transport/' + transport['view_receipt_name'],
                        '--transport-view-receipt-sha256', transport['view_receipt_sha256'],
                        '--transport-scope', transport['scope'], '--training-source-sha256', job['source_sha256']]
    host = job.get('host', '127.0.0.1')
    if rl:
        server_args += ['--actor-export-format', job['actor_export_format']]
    server_args += ['--base', '/base', '--device', 'cuda', '--host', host, '--port', str(port),
                    '--prefill-tokens', '2048', '--prefix-block', '2048', '--prefix-cache-gb', '.25', '--threads', '2',
                    '--store-dir', '/work/store', '--prompt-rendering', 'full']
    bootstrap = ("import torch,runpy,sys;torch.cuda.set_per_process_memory_fraction(.90);sys.argv=" + repr(server_args)
                 + ";runpy.run_path('/repo/scripts/neuralese_data/serve_native_checkpoint.py',run_name='__main__')")
    lease = stage / f'{name}.consumer.json'
    lease.write_text(json.dumps({'role': 'rl-rollout-actor' if rl else 'protected-development-evaluation', 'checkpoint_sha256': job['source_sha256'],
                                 'consumer': name, 'started': time.time(),
                                 'release': 'after actual container shutdown; no training admission'}) + '\n')
    memory = job.get('memory', '5g')   # the RL actor measured at most 2.8 GiB of host memory (Pop is short of RAM)
    cmd = ['docker', 'run', '-d', '--name', name, '--gpus', 'all', '--network', 'host', '--cpus', '3', '--memory', memory,
           '--memory-swap', memory, '--read-only', '--tmpfs', '/tmp:rw,size=1g', '--log-opt', 'max-size=10m',
           '--log-opt', 'max-file=2', '-v', f'{runtime}:/repo:ro', '-v', f'{base}:/base:ro',
           '-v', f'{checkpoint}:/checkpoint.pt:ro', '-v', f'{stage}:/transport:ro',
           *([] if rl else ['-v', f'{packet.parent}:/packet:ro']), '-v', f'{work}:/work', '-v', '/usr/bin/node:/usr/bin/node:ro']
    env = {'PYTHONPATH': '/repo/training/neuralese:/repo', 'PYTHONDONTWRITEBYTECODE': '1', 'NATLANG_REPO': '/repo',
           'NATLANG_BF16_REDUCTION': 'fp32-accumulate', 'NATLANG_NATIVE_COMPILE': '0', 'HF_HUB_OFFLINE': '1',
           'TRANSFORMERS_OFFLINE': '1', 'HF_HOME': '/work/hf', 'HF_MODULES_CACHE': '/work/hf/modules',
           'TRITON_CACHE_DIR': '/work/triton', 'TORCHINDUCTOR_CACHE_DIR': '/work/inductor', 'CUDA_CACHE_PATH': '/work/cuda',
           'NATLANG_HASH_CACHE': '/work/hash-cache.jsonl', 'PYTHONUNBUFFERED': '1'}
    for key, value in env.items():
        cmd += ['-e', f'{key}={value}']
    cmd += [image, 'python', '-c', bootstrap]
    if rl:
        import signal
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # the finally clause below stops and removes the container
    try:
        container = subprocess.check_output(cmd, text=True, timeout=60).strip()
        (work / 'launch.json').write_text(json.dumps({
            'container_id': container, 'step': job['step'], 'checkpoint_sha256': job['source_sha256'],
            'arm': job['arm'], 'runtime_receipt_sha256': job['runtime_closure_sha256'],
            'image_id': subprocess.check_output(['docker', 'image', 'inspect', image, '--format', '{{.Id}}'], text=True).strip(),
            'task_spec_sha256': spec_sha, 'mode': job.get('mode', 'evaluation'), 'transport_sha256': (transport or {}).get('sha256'),
            'inference_view_sha256': job['view_sha256'], 'driver_sha256': digest(__file__),
            'comparison': job.get('comparison')}, indent=2) + '\n')
        for _ in range(450):
            if subprocess.check_output(['docker', 'inspect', '-f', '{{.State.Running}}', name], text=True).strip() != 'true':
                raise RuntimeError('actor exited before readiness')
            try:
                with urllib.request.urlopen(f'http://{host}:{port}/v1/neuralese/info', timeout=2) as f:
                    info = json.load(f)
                if info['native_state']['checkpoint_sha256'] != job['source_sha256'] or info['prompt_rendering'] != 'full':
                    raise RuntimeError('actor serves another identity: ' + info['native_state']['checkpoint_sha256'])
                (work / 'actor-info.json').write_text(json.dumps(info, indent=2) + '\n')
                break
            except RuntimeError:
                raise
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError('actor readiness deadline')
        print(json.dumps({'step': job['step'], 'arm': job['arm'], 'state': 'actor ready' + ('' if rl else '; protected evaluation starting')}),
              flush=True)
        if rl:
            (work / 'actor-ready.json').write_text(json.dumps({'container': name, 'host': host, 'port': port,
                                                               'source_sha256': job['source_sha256'], 'at': time.time()}) + '\n')
            # The actor serves until the collection ends: the controller's teardown hook stops this unit (SIGTERM).
            subprocess.run(['docker', 'wait', name], check=False)
            return
        subprocess.run(['docker', 'exec', name, 'python', '/repo/scripts/neuralese_data/evaluate_native_actor.py',
                        '--endpoint', f'http://127.0.0.1:{port}', '--checkpoint-sha256', job['source_sha256'],
                        '--spec', '/work/spec.json', '--spec-sha256', spec_sha, '--out', '/work/evaluation',
                        '--step', str(job['step'])], check=True, timeout=21000)
    finally:
        log = subprocess.run(['docker', 'logs', name], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (work / 'actor.log').write_text(log.stdout)
        subprocess.run(['docker', 'stop', '--time', '15', name], check=False, timeout=60)
        inspect = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', name], text=True, capture_output=True)
        if inspect.returncode == 0 and inspect.stdout.strip() == 'false':
            lease.unlink(missing_ok=True)
            subprocess.run(['docker', 'rm', name], check=True)
    print(json.dumps({'step': job['step'], 'arm': job['arm'], 'state': 'finished; read evaluation/completed.json'}),
          flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
