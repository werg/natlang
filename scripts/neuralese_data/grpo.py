#!/usr/bin/env python3
"""Plain GRPO for LFM3-1.7B (plans/neuralese/APPROACH_REVIEW_20261111.md section 4, "Hour 24-48").

One process runs the whole loop; nothing else (no controller, custody or transitions):

* rollouts: ``--slots`` collector slots run episodes (one ``run_bonsai_queue`` process each, the RL sampling capture) against
  one SGLang actor (``serve_native_checkpoint.py --engine sglang``, started separately). Tasks come shuffled from the pool
  minus the held set (``held_eval.exclusion_ids``); each task is a group of ``--group`` samples. Reward = task outcome
  (no teacher, no privileged twin, no Neuralese bonus, no efficiency shaping).
* overlap: collection never stops. The learner trains on the groups that finished since its last step (at least
  ``--min-groups``, at most ``--max-groups``); unfinished groups keep running and join a later step, so nothing waits on
  stragglers. A group sampled more than ``--max-staleness`` policy versions ago is dropped and counted.
* learner (the Experiment 0 model path, ``exp0_sft`` / ``lfm3_fast_train``): FP32 master weights, AdamW, BF16 compute. Every
  model call is (exact actor prompt ids, sampled ids) (``lower_rollout_turns.episode_turns``; calls of one invocation chain
  into one sequence as in ``exp0_sft.chains_of``). Loss on sampled tokens only:
  ``-min(rho, C) * A * logp + beta * k3(ref || policy)``, ``rho = exp(logp_learner - logp_actor)`` (truncated importance
  sampling against the actor's own sampling logprobs), ``A`` the group-normalized episode reward on every call of the
  episode, the reference the start policy. Summed over tokens, divided by the step's loss tokens.
* weight sync: after each step the BF16 weights are copied once into a pinned host buffer, written as the actor's HF
  safetensors (same layout as ``export/native_hf``; frozen rows and producer table kept) and loaded in place
  (SGLang ``pause_generation`` retract -> ``update_weights_from_disk`` -> ``continue_generation``), in a background thread.
  Episodes keep running across a sync; each token's actor logprob is from the weights that sampled it.
* checkpoints: a pruned native checkpoint (``exp0_sft.write_checkpoint``, servable by ``serve_native_checkpoint``) every
  ``--save-every`` steps and the FP32 training state (masters + AdamW moments) every ``--state-every`` steps, both written
  from pinned host memory in background threads (training never waits on I/O). Keeps the last two native checkpoints plus
  every held-evaluated one, and the last training state.
* held evaluation: the start policy first, then every ``--held-every-hours`` the newest native checkpoint is measured on
  the held set (``python -m natlang_neuralese.rl.held_eval measure``: its own actor and ledger unit, same pinned code) and
  compared with the start policy (``held_eval compare`` -> ``held/step-N/compare-vs-step0.json``).

    grpo.py train --work W --checkpoint C --actor http://127.0.0.1:8310 [...]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import queue
import random
import shutil
import signal
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

from natlang_neuralese.common.paths import resolve_str  # noqa: E402

BASE = str(Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')
POOLS = [resolve_str('data_hdd', 'natlang-rl-composed-20261103', 'memetic-program-source-20261105', 'cases.jsonl'),
         # fallow K1: bgkit KR reader families, 4,734 rows verified by reference/negative/alias replay (verification.json)
         resolve_str('data_hdd', 'natlang-rl-bgkit-kr-20261103', 'natlang-rl-bgkit-kr-20261103', 'cases.jsonl')]
HELD = resolve_str('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'held-v2', 'manifest.json')


def log_line(handle, record):
    text = json.dumps(record)
    print(text, flush=True)
    if handle is not None:
        handle.write(text + '\n')
        handle.flush()


def post(url, body=None, timeout=600):
    data = json.dumps(body or {}).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def git_pin():
    from natlang_neuralese.common.code_provenance import code_provenance
    return code_provenance()


def pin_runtime(work, commit):
    """The run's own copy of the collector runtime (ts-host with its built dist), so episodes keep one harness while main
    moves; every other top-level entry of the checkout is linked (dist resolves ../applications etc. there). The learner
    and the collector's Python are pinned by being this one long-lived process."""
    pin = Path(work) / f'pin-{commit[:12]}'
    if not (pin / 'ts-host' / 'dist' / 'teacher' / 'cli.js').is_file():
        partial = Path(work) / f'pin-{commit[:12]}.partial'
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        subprocess.run(['cp', '-a', str(REPO / 'ts-host'), str(partial / 'ts-host')], check=True)
        for entry in REPO.iterdir():
            if entry.name not in ('ts-host', '.git'):
                (partial / entry.name).symlink_to(entry)
        shutil.rmtree(pin, ignore_errors=True)
        os.replace(partial, pin)
    return pin / 'ts-host'


# ---- rollouts --------------------------------------------------------------------------------------------------------
class Collector:
    """Work-stealing slots over an endless stream of task groups; finished groups are scored and queued for the learner."""

    def __init__(self, args, tasks, work, policy_sha, runtime, first_group=0, cursor=0):
        self.args, self.tasks, self.work, self.policy_sha, self.runtime = args, tasks, work, policy_sha, runtime
        self.lock = threading.Lock()
        self.pending = collections.deque()          # entries not yet taken by a slot
        self.groups = {}                            # group -> {'dir', 'size', 'done', 'versions', 'task_id'}
        self.ready = queue.Queue()                  # finished, scored groups
        self.outstanding = 0                        # groups created and not yet handed to the learner
        self.next_group, self.cursor = first_group, cursor
        self.version = 0                            # the policy version the actor serves now
        self.stop = threading.Event()
        self.generated = 0                          # sampled tokens of finished episodes (throughput)
        self.episodes = 0
        self.env = {**os.environ}
        from natlang_neuralese.rl.held_eval import series
        self.env['NATLANG_MODEL_FACING'] = series()['model_facing']

    def _new_group(self):
        from natlang_neuralese.rl import task_rollouts
        from natlang_neuralese.common.jsonio import read_jsonl
        row = self.tasks[self.cursor % len(self.tasks)]
        self.cursor += 1
        group = self.next_group
        self.next_group += 1
        directory = self.work / 'groups' / f'g{group:06d}'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'cases.jsonl').write_text(json.dumps(row) + '\n')
        a = self.args
        task_rollouts.prepare(SimpleNamespace(
            cases=directory / 'cases.jsonl', out=directory, rows=1, offset=0, group=a.group, temperature=1.,
            case_seconds=a.case_seconds, max_model_requests=a.max_model_requests, max_turns=a.max_turns,
            seed_base=a.seed_base + 100 * group, root_mode='student', match_seeds_from=None,
            policy_checkpoint_sha256=self.policy_sha))
        entries = list(read_jsonl(directory / 'queue.jsonl'))
        for entry in entries:
            entry.update(prompt_rendering='full', context_tokens=a.context_tokens, rl_endpoint=a.actor)
        (directory / 'queue.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n' for e in entries))
        self.groups[group] = {'dir': directory, 'size': len(entries), 'done': 0, 'versions': [], 'task_id': row['id'],
                              'created': time.time()}
        self.outstanding += 1
        self.pending.extend((group, entry) for entry in entries)

    def _take(self):
        with self.lock:
            if not self.pending and self.outstanding < self.args.max_outstanding:
                self._new_group()
            return self.pending.popleft() if self.pending else None

    def slot(self):
        sampling = self.work / 'sampling-config.json'
        while not self.stop.is_set():
            item = self._take()
            if item is None:
                time.sleep(2)
                continue
            group, entry = item
            info = self.groups[group]
            workers = info['dir'] / 'workers'
            workers.mkdir(exist_ok=True)
            key = entry['key']
            queue_file, journal = workers / f'{key}.queue.jsonl', workers / f'{key}.journal.jsonl'
            queue_file.write_text(json.dumps(entry, sort_keys=True) + '\n')
            started = self.version
            command = [sys.executable, str(REPO / 'scripts/run_bonsai_queue.py'), str(queue_file), str(journal),
                       '--runtime', str(self.runtime), '--server', self.args.actor,
                       '--chat-completions-url', self.args.actor + '/v1/chat/completions', '--model-id', 'natlang-rl',
                       '--model-concurrency', '1', '--chat-request-config', str(sampling), '--neuralese-server']
            with (workers / f'{key}.log').open('a') as handle:
                process = subprocess.Popen(command, cwd=REPO, env=self.env, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                while process.poll() is None:
                    if self.stop.wait(2):
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait()
                        return
            with self.lock:
                info['done'] += 1
                info['versions'].append(started)
                complete = info['done'] == info['size']
            if complete:
                try:
                    self._score(group)
                except Exception as error:   # a scoring failure loses one group, never the loop
                    print(json.dumps({'event': 'group scoring failed', 'group': group, 'error': repr(error)[:300]}),
                          flush=True)
                    with self.lock:
                        self.outstanding -= 1

    def _score(self, group):
        from natlang_neuralese.rl import task_rollouts
        info = self.groups[group]
        directory = info['dir']
        with (directory / 'journal.jsonl').open('w') as merged:
            for path in sorted((directory / 'workers').glob('*.journal.jsonl')):
                merged.write(path.read_text())
        task_rollouts.score(directory, directory / 'rollouts.jsonl', neuralese_bonus=0., efficiency_weight=0.)
        (directory / 'versions.json').write_text(json.dumps(info['versions']) + '\n')
        self.ready.put(group)

    def take_groups(self, minimum, maximum, stop=None):
        """At least ``minimum`` finished groups (waits, unless ``stop`` is set), at most ``maximum``."""
        taken = []
        while len(taken) < minimum:
            try:
                taken.append(self.ready.get(timeout=5))
            except queue.Empty:
                if stop is not None and stop.is_set():
                    break
        while len(taken) < maximum:
            try:
                taken.append(self.ready.get_nowait())
            except queue.Empty:
                break
        with self.lock:
            self.outstanding -= len(taken)
        return taken


def load_group(collector, group, stats):
    """(task_id, [episode dicts with 'reward', 'rows'], versions) of one scored group."""
    from natlang_neuralese.common.jsonio import read_jsonl
    from lower_rollout_turns import episode_turns
    info = collector.groups[group]
    episodes = []
    for row in read_jsonl(info['dir'] / 'rollouts.jsonl'):
        reward = row.get('reward')
        rows = episode_turns(row, stats) if row.get('turns') else []
        sampled = sum(len(r['sample']) for r in rows)
        collector.generated += sampled
        collector.episodes += 1
        if reward is None or row.get('unscored'):
            stats['episodes_unscored'] += 1
            continue
        episodes.append({'reward': float(reward['quality']), 'rows': rows, 'task_id': row['task_id'], 'round': group,
                         'outcome': row.get('outcome')})
    return info['task_id'], episodes, info['versions']


def prune_group(collector, group, keep):
    directory = collector.groups.pop(group)['dir']
    if keep:
        return
    for path in directory.iterdir():
        if path.name not in ('score.json', 'rewards.jsonl', 'versions.json', 'cases.jsonl'):
            shutil.rmtree(path) if path.is_dir() else path.unlink()


# ---- host I/O --------------------------------------------------------------------------------------------------------
def safetensors_header(layout, metadata=None):
    """layout: [(name, dtype_str, shape, nbytes)] -> (header bytes, offsets)."""
    header, offset, offsets = {}, 0, []
    for name, dtype, shape, nbytes in layout:
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset + nbytes]}
        offsets.append(offset)
        offset += nbytes
    if metadata:
        header['__metadata__'] = {k: str(v) for k, v in metadata.items()}
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    return struct.pack('<Q', len(raw)) + raw, offsets, offset


class WeightSync:
    """The actor's HF export in one pinned host buffer: snapshot (GPU -> pinned, the only synchronous part), then in a
    background thread write it, load it into SGLang in place, and optionally write a native checkpoint."""

    def __init__(self, model, export_dir, sglang_url, work_dir):
        import torch
        from safetensors import safe_open
        self.export_dir, self.sglang_url = Path(export_dir), sglang_url
        self.weights_dir = Path(work_dir) / 'actor-weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        for path in self.export_dir.iterdir():
            if path.is_file() and path.name != 'model.safetensors':
                shutil.copyfile(path, self.weights_dir / path.name)
        params = dict(model.named_parameters())
        layout, sources = [], []
        with safe_open(str(self.export_dir / 'model.safetensors'), 'pt') as handle:
            fixed = {}
            for name in handle.keys():
                tensor = handle.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError(f'export tensor {name} is {tensor.dtype}, expected bf16')
                if name in params:
                    if tuple(params[name].shape) != tuple(tensor.shape):
                        raise ValueError(f'export/learner shape mismatch {name}')
                    sources.append(name)
                else:
                    fixed[name] = tensor            # the producer table and anything else the learner does not train
                    sources.append(None)
                layout.append((name, 'BF16', tuple(tensor.shape), tensor.numel() * 2))
        missing = set(params) - {name for name in sources if name} - {'lm_head.weight'}
        if missing:
            raise ValueError(f'learner parameters absent from the export: {sorted(missing)[:5]}')
        self.header, offsets, total = safetensors_header(layout, {'format': 'pt'})
        self.buffer = torch.empty(total, dtype=torch.uint8, pin_memory=True)
        self.views = {}
        for (name, _, shape, nbytes), offset, source in zip(layout, offsets, sources):
            view = self.buffer[offset:offset + nbytes].view(torch.bfloat16).view(shape)
            if source is None:
                view.copy_(fixed[name])
            else:
                self.views[name] = view
        self.params = params
        self.thread = None
        self.last = {}

    def snapshot(self):
        import torch
        self.wait()
        for name, view in self.views.items():
            view.copy_(self.params[name].detach(), non_blocking=True)
        torch.cuda.synchronize()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def backbone(self):
        """{name: host bf16 tensor} of the snapshot in native backbone naming (copies, safe to keep)."""
        return {name: view.clone() for name, view in self.views.items()}

    def publish(self, version, on_done, also=None):
        def run():
            timing = {}
            started = time.time()
            partial = self.weights_dir / 'model.safetensors.partial'
            with partial.open('wb') as handle:
                handle.write(self.header)
                handle.write(memoryview(self.buffer.numpy()))
            os.replace(partial, self.weights_dir / 'model.safetensors')
            timing['export_write_s'] = round(time.time() - started, 2)
            t = time.time()
            if self.sglang_url is None:   # --no-actor-sync (a smoke against a shared actor): nothing to update
                answer, ok = 'no actor sync', True
            else:
              try:
                post(self.sglang_url + '/pause_generation', {'mode': 'retract'})
                answer = post(self.sglang_url + '/update_weights_from_disk',
                              {'model_path': str(self.weights_dir), 'flush_cache': True, 'weight_version': str(version)})
                post(self.sglang_url + '/continue_generation', {})
                ok = json.loads(answer).get('success', False)
              except Exception as error:
                answer, ok = repr(error)[:400], False
                try:
                    post(self.sglang_url + '/continue_generation', {})
                except Exception:
                    pass
            timing['actor_update_s'] = round(time.time() - t, 2)
            if also is not None:
                t = time.time()
                timing.update(also() or {})
                timing['checkpoint_s'] = round(time.time() - t, 2)
            self.last = {'version': version, 'ok': ok, 'answer': None if ok else answer, **timing}
            on_done(version, ok)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


class StateSaver:
    """FP32 masters and AdamW moments streamed GPU -> pinned staging -> file in a background thread. The caller waits
    (``wait``) before the next optimizer step changes them."""

    def __init__(self, staging_mb=512):
        import torch
        self.staging = torch.empty(staging_mb << 20, dtype=torch.uint8, pin_memory=True)
        self.stream = torch.cuda.Stream()
        self.thread = None
        self.last = None

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def save(self, path, tensors, metadata):
        import torch
        self.wait()
        torch.cuda.synchronize()
        names = list(tensors)
        layout = [(name, 'F32', tuple(tensors[name].shape), tensors[name].numel() * 4) for name in names]
        header, _, _ = safetensors_header(layout, metadata)

        def run():
            started = time.time()
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix('.partial')
            chunk = self.staging.numel()
            with partial.open('wb') as handle, torch.cuda.stream(self.stream):
                handle.write(header)
                for name in names:
                    flat = tensors[name].detach().reshape(-1).view(torch.uint8)
                    for start in range(0, flat.numel(), chunk):
                        piece = flat[start:start + chunk]
                        self.staging[:piece.numel()].copy_(piece, non_blocking=True)
                        self.stream.synchronize()
                        handle.write(memoryview(self.staging[:piece.numel()].numpy()))
            os.replace(partial, path)
            self.last = {'path': str(path), 'seconds': round(time.time() - started, 1)}
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()


# ---- learner ---------------------------------------------------------------------------------------------------------
def group_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    if std < 1e-6:
        return None
    return [(r - mean) / (std + 1e-6) for r in rewards]


def load_tasks(args, work=None):
    """The shuffled train tasks of the pools (first occurrence of an id wins) minus the held set: its tasks, every row of
    a held ``split_group`` (``held_eval.exclusion_ids``) and every row sharing a ``source_groups`` entry with those.
    Writes ``WORK/pool.json`` (pool sha256s, counts, sha256 of the eligible ids)."""
    import hashlib
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    from natlang_neuralese.common.jsonio import read_jsonl
    from natlang_neuralese.rl.held_eval import exclusion_ids
    rows, seen = [], set()
    for path in args.pool:
        for row in read_jsonl(path):
            if row['id'] not in seen:
                seen.add(row['id'])
                rows.append(row)
    excluded = set(exclusion_ids(args.held_manifest, rows))
    for extra in args.exclude_manifest:
        excluded |= exclusion_ids(extra, rows)
    held_groups = {g for r in rows if r['id'] in excluded for g in r.get('source_groups') or ()}
    excluded |= {r['id'] for r in rows if held_groups.intersection(r.get('source_groups') or ())}
    if args.task_ids:
        wanted = set(json.loads(Path(args.task_ids).read_text()))
        rows = [r for r in rows if r['id'] in wanted]
    eligible = [r for r in rows if r['id'] not in excluded and r.get('split', 'train') == 'train']
    random.Random(args.seed).shuffle(eligible)
    if work is not None:
        record = {'pools': [{'path': str(p), 'sha256': sha256_file_hex_cached(Path(p))} for p in args.pool],
                  'held_manifest': args.held_manifest, 'held_manifest_sha256': sha256_file_hex_cached(Path(args.held_manifest)),
                  'union_rows': len(seen), 'excluded_ids': len(excluded), 'eligible': len(eligible),
                  'eligible_ids_sha256': hashlib.sha256('\n'.join(sorted(r['id'] for r in eligible)).encode()).hexdigest(),
                  'families': len({r.get('family') for r in eligible})}
        (Path(work) / 'pool.json').write_text(json.dumps(record, indent=1) + '\n')
    return eligible, len(excluded)


def train(args):
    import torch
    import exp0_sft
    import lfm3_fast_train
    from natlang_neuralese.common.hashing import sha256_file_hex_cached
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    log = open(work / 'train.jsonl', 'a')
    tasks, excluded = load_tasks(args, work)
    source_sha = args.checkpoint_sha256 or sha256_file_hex_cached(Path(args.checkpoint))
    code = git_pin()
    runtime = pin_runtime(work, code['git_commit']) if args.pin_runtime else REPO / 'ts-host'
    from natlang_neuralese.rl.held_eval import series
    run = {'started': time.time(), 'code': code, 'runtime': str(runtime), 'series': series(), 'argv': sys.argv, 'checkpoint': args.checkpoint,
           'checkpoint_sha256': source_sha, 'pool': args.pool, 'held_manifest': args.held_manifest,
           'excluded_ids': excluded, 'eligible_tasks': len(tasks), 'args': vars(args)}
    with (work / 'runs.jsonl').open('a') as handle:
        handle.write(json.dumps(run) + '\n')
    log_line(log, {'event': 'start', 'code': run['code'], 'eligible_tasks': len(tasks), 'excluded_ids': excluded})

    t0 = time.time()
    actor_ready = {}
    actor_process = []

    def launch_actor():
        # The training actor is a child of this unit (one ledger claim covers learner and actor, so neither waits on the
        # other's admission), served with the pinned held-v2 eval engine (held-v2/engine.json serve_args): training and
        # held measurements sample with the same arithmetic.
        try:
            engine = json.loads((Path(args.held_manifest).parent / 'engine.json').read_text())['serve_args']
            store = work / 'actor'
            store.mkdir(parents=True, exist_ok=True)
            url = f'http://127.0.0.1:{args.actor_port}'
            command = [sys.executable, str(HERE / 'serve_native_checkpoint.py'), '--store-dir', str(store),
                       '--checkpoint', args.checkpoint, '--checkpoint-sha256', source_sha, '--base', args.base,
                       '--port', str(args.actor_port), '--prompt-rendering', 'full', *engine,
                       '--hf-export-dir', str(store / 'sglang-hf'), *args.actor_arg]
            with (work / 'actor.log').open('a') as handle:
                process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            actor_process.append(process)
            deadline = time.time() + 3600
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'actor exited with {process.returncode} (see {work}/actor.log)')
                try:
                    with urllib.request.urlopen(url + '/v1/neuralese/info', timeout=10) as response:
                        info = json.loads(response.read())
                    if (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') == source_sha:
                        actor_ready['url'] = url
                        return
                except OSError:
                    pass
                time.sleep(10)
            raise TimeoutError('actor did not come up in an hour')
        except BaseException as error:
            actor_ready['error'] = error

    def stop_actor(*_):
        for process in actor_process:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
    import atexit
    atexit.register(stop_actor)
    actor_thread = None
    if not args.actor:
        actor_thread = threading.Thread(target=launch_actor, daemon=True)
        actor_thread.start()
    model, state, special = exp0_sft.load_model(args.checkpoint, args.base)
    state = {k: state[k] for k in ('schema', 'model_config', 'interface_state', 'step', 'backbone_state')}
    state['backbone_state'] = {k: v.to(torch.bfloat16) for k, v in state['backbone_state'].items()}
    masters = {name: p.detach().clone().float() for name, p in model.named_parameters()}
    model.to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    reference = None
    if args.kl_beta > 0:
        reference, _, _ = exp0_sft.load_model(args.checkpoint, args.base, dtype=torch.bfloat16)
        reference.eval()
        reference.requires_grad_(False)
        lfm3_fast_train.prepare(reference)
    lfm3_fast_train.prepare(model)
    forward = lfm3_fast_train.hidden
    live = dict(model.named_parameters())
    parameters = [masters[name] for name in live]
    for master in parameters:
        master.grad = torch.zeros_like(master)
    optimizer = torch.optim.AdamW([{'params': [p for p in parameters if p.ndim >= 2], 'weight_decay': args.weight_decay},
                                   {'params': [p for p in parameters if p.ndim < 2], 'weight_decay': 0.}],
                                  lr=args.lr, betas=(0.9, 0.99), eps=1e-8, fused=True)
    special_index = torch.tensor(special, device='cuda')
    embedding_name = next(n for n, p in live.items() if p is model.get_input_embeddings().weight)
    embedding_master = masters[embedding_name]
    saved_rows = embedding_master[special_index].clone()

    step, cursor, first_group = 0, 0, 0
    state_path = Path(args.state_dir) / 'training-state.safetensors'
    if args.resume and state_path.exists():
        from safetensors import safe_open
        with safe_open(str(state_path), 'pt', device='cuda') as handle:
            meta = handle.metadata()
            for name in live:
                masters[name].copy_(handle.get_tensor('master/' + name))
            optimizer.step()        # materialise AdamW state (zero grads), then overwrite it
            for name in live:
                st = optimizer.state[masters[name]]
                st['exp_avg'].copy_(handle.get_tensor('exp_avg/' + name))
                st['exp_avg_sq'].copy_(handle.get_tensor('exp_avg_sq/' + name))
                st['step'].fill_(float(meta['adam_step']))
        step, cursor, first_group = int(meta['step']), int(meta['cursor']), int(meta['next_group'])
        with torch.no_grad():
            for name, p in live.items():
                p.copy_(masters[name])
        log_line(log, {'event': 'resumed', 'step': step, 'cursor': cursor, 'next_group': first_group})

    if actor_thread is not None:
        actor_thread.join()
        if 'error' in actor_ready:
            raise actor_ready['error']
        args.actor = actor_ready['url']
        args.actor_export = args.actor_export or str(work / 'actor' / 'sglang-hf')
    log_line(log, {'event': 'actor ready', 'actor': args.actor, 'init_s': round(time.time() - t0, 1)})
    sglang_url = args.sglang or args.actor.rsplit(':', 1)[0] + ':' + str(int(args.actor.rsplit(':', 1)[1]) + 1000)
    sync = WeightSync(model, args.actor_export, None if args.no_actor_sync else sglang_url, work)
    saver = StateSaver()
    collector = Collector(args, tasks, work, source_sha, runtime, first_group=first_group, cursor=cursor)
    collector.version = step
    (work / 'sampling-config.json').write_text(json.dumps({'guidance': False, 'x_natlang_rl_capture': True, 'x_natlang_rl_top_k': 0}) + '\n')
    versions_live = {'actor': step}

    def on_published(version, ok):
        if ok:
            collector.version = version
            versions_live['actor'] = version
    if step:   # a resumed run first serves its own weights
        sync.snapshot()
        sync.publish(step, on_published)
        sync.wait()
        if not sync.last.get('ok'):
            raise RuntimeError(f'actor weight update failed: {sync.last}')
    log_line(log, {'event': 'learner ready', 'init_s': round(time.time() - t0, 1)})
    slots = [threading.Thread(target=collector.slot, daemon=True) for _ in range(args.slots)]
    for thread in slots:
        thread.start()

    checkpoints = sorted((work / 'checkpoints').glob('step-*'))
    held_steps = set(json.loads((work / 'held-steps.json').read_text())) if (work / 'held-steps.json').exists() else set()
    last_held = time.time() if step else 0.
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    def write_native(step_number):
        directory = work / 'checkpoints' / f'step-{step_number:05d}'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / 'checkpoint.pt'
        changed = exp0_sft.write_checkpoint(None, state, special, path, source_sha, step_number, trained=sync.backbone())
        from natlang_neuralese.common.hashing import sha256_file_hex
        sha = sha256_file_hex(path)
        (directory / 'sha256.txt').write_text(sha + '\n')
        (directory / 'meta.json').write_text(json.dumps({'step': step_number, 'sha256': sha, 'changed_elements': changed,
                                                         'source_sha256': source_sha, 'code': run['code']}) + '\n')
        checkpoints.append(directory)
        for old in checkpoints[:-2]:
            if int(old.name.split('-')[1]) not in held_steps and old.exists():
                shutil.rmtree(old)
        checkpoints[:] = [c for c in checkpoints if c.exists()]
        return {'checkpoint': str(path), 'checkpoint_sha256': sha}

    def start_held(step_number, checkpoint_dir):
        """``held_eval measure`` of a checkpoint (its own actor unit) in a ledger unit, then ``compare`` against step 0."""
        held_steps.add(step_number)
        (work / 'held-steps.json').write_text(json.dumps(sorted(held_steps)) + '\n')
        out = work / 'held' / f'step-{step_number:05d}'
        sha = (checkpoint_dir / 'sha256.txt').read_text().strip() if checkpoint_dir else source_sha
        checkpoint = str(checkpoint_dir / 'checkpoint.pt') if checkpoint_dir else args.checkpoint
        python = f'{sys.executable} -m natlang_neuralese.rl.held_eval'
        script = (f'{python} measure --manifest {args.held_manifest} --checkpoint {checkpoint} --checkpoint-sha256 {sha} '
                  f'--out {out} --samples {args.held_samples} --workers {args.held_workers} --label grpo-step-{step_number} '
                  f'--actor-unit {args.unit_prefix}-heldactor-{step_number}')
        base = work / 'held' / 'step-00000' / 'result.json'
        if step_number:
            script += f' && {python} compare {base} {out}/result.json --out {out}/compare-vs-step0.json'
        command = ['/usr/bin/python3', str(REPO / 'scripts/memory_ledger.py'), 'run', '--unit', f'{args.unit_prefix}-held-{step_number}',
                   '--budget-gb', '12', '--priority', 'medium', '--wait', '86400', '--detach', '--workdir', str(REPO), '--',
                   '/usr/bin/env', f'PYTHONPATH={REPO}/training/neuralese:{REPO}', f'NATLANG_REPO={REPO}', 'HF_HUB_OFFLINE=1',
                   'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'sh', '-c', script]
        result = subprocess.run(command, capture_output=True, text=True)
        log_line(log, {'event': 'held eval queued', 'step': step_number, 'returncode': result.returncode,
                       'output': (result.stdout + result.stderr)[-400:]})

    if args.held_every_hours > 0 and not (work / 'held' / 'step-00000').exists() and not args.skip_base_held:
        start_held(0, None)

    while not stop.is_set() and (not args.max_steps or step < args.max_steps):
        phase = collections.OrderedDict()
        t = time.time()
        groups = collector.take_groups(args.min_groups, args.max_groups, stop)
        phase['wait_groups'] = time.time() - t
        t = time.time()
        stats = collections.Counter()
        chains, rewards, spread, dropped_stale, staleness = [], [], 0, 0, []
        for group in groups:
            task_id, episodes, versions = load_group(collector, group, stats)
            prune_group(collector, group, keep=args.keep_rollouts)
            if not episodes:
                continue
            stale = step - min(versions) if versions else 0
            rewards += [e['reward'] for e in episodes]
            if stale > args.max_staleness:
                dropped_stale += 1
                continue
            staleness.append(stale)
            advantages = group_advantages([e['reward'] for e in episodes]) if len(episodes) >= 2 else None
            if advantages is None:
                continue
            spread += 1
            for episode, advantage in zip(episodes, advantages):
                for chain in exp0_sft.chains_of([episode]):
                    chain['advantage'] = advantage
                    chains.append(chain)
        loss_tokens = sum(length for c in chains for _, length, _ in c['segs'])
        phase['lower'] = time.time() - t
        record = {'step': step + 1, 'groups': len(groups), 'groups_with_spread': spread,
                  'spread_fraction': round(spread / max(1, len(groups)), 3), 'dropped_stale_groups': dropped_stale,
                  'staleness': dict(collections.Counter(staleness)), 'episodes': len(rewards),
                  'reward_mean': round(sum(rewards) / max(1, len(rewards)), 4), 'sequences': len(chains),
                  'loss_tokens': loss_tokens, 'lowering': dict(stats), 'actor_version': versions_live['actor']}
        if not chains:
            log_line(log, {**record, 'event': 'no learning signal in these groups',
                           'phase_s': {k: round(v, 1) for k, v in phase.items()}})
            continue
        # learner
        torch.cuda.reset_peak_memory_stats()
        sums = collections.Counter()
        rho_pre_max, rho_post_max, forward_tokens = 0., 0., 0
        rng = random.Random(args.seed + step)
        rng.shuffle(chains)
        for chain in chains:
            ids, positions, targets = exp0_sft.chain_tensors(chain, 'cuda')
            actor = torch.from_numpy(np.concatenate([row['logp'] for _, _, row in chain['segs']])).cuda().float()
            t = time.time()
            ref = None
            if reference is not None:
                with torch.no_grad():
                    ref = -exp0_sft.token_nll(reference, forward(reference, ids), positions, targets)
                torch.cuda.synchronize()
                phase['reference_forward'] = phase.get('reference_forward', 0.) + time.time() - t
                t = time.time()
            exp0_sft.set_checkpointing(model, len(ids), args.activation_gb)
            h = forward(model, ids)
            logp = -exp0_sft.token_nll(model, h, positions, targets)
            with torch.no_grad():
                log_rho = logp.detach() - actor
                rho = log_rho.clamp(max=20).exp()
                weight = rho.clamp(max=args.rho_cap)
            loss_terms = -(weight * chain['advantage'] * logp)
            if ref is not None:
                d = ref - logp
                kl = d.exp() - d - 1
                loss_terms = loss_terms + args.kl_beta * kl
                sums['kl'] += kl.detach().sum().item()
            (loss_terms.sum() / loss_tokens).backward()
            torch._foreach_add_([masters[name].grad for name in live], [p.grad for p in live.values()])
            for p in live.values():
                p.grad = None
            sums['rho'] += rho.sum().item()
            sums['rho_capped'] += (rho > args.rho_cap).sum().item()
            sums['abs_log_rho'] += log_rho.abs().sum().item()
            sums['logp'] += logp.detach().sum().item()
            rho_pre_max = max(rho_pre_max, rho.max().item())
            rho_post_max = max(rho_post_max, weight.max().item())
            forward_tokens += len(ids)
            del h, logp, loss_terms
            torch.cuda.synchronize()
            phase['policy_forward_backward'] = phase.get('policy_forward_backward', 0.) + time.time() - t
        t = time.time()
        saver.wait()                      # a state save in flight must finish before the masters change
        phase['wait_state_save'] = time.time() - t
        t = time.time()
        embedding_master.grad[special_index] = 0
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.clip).item()
        lr = args.lr * min(1., (step + 1) / args.warmup)
        for g in optimizer.param_groups:
            g['lr'] = lr
        optimizer.step()
        for p in parameters:
            p.grad.zero_()
        with torch.no_grad():
            embedding_master[special_index] = saved_rows
            for name, p in live.items():
                p.copy_(masters[name])
        torch.cuda.synchronize()
        phase['optimizer'] = time.time() - t
        step += 1
        # weight sync and checkpoints (background)
        t = time.time()
        sync.snapshot()                   # waits for the previous publish
        phase['snapshot'] = time.time() - t
        if sync.last and not sync.last.get('ok'):
            raise RuntimeError(f'actor weight update failed: {sync.last}')
        save_native = step % args.save_every == 0 or (args.max_steps and step == args.max_steps)
        sync.publish(step, on_published, also=(lambda s=step: write_native(s)) if save_native else None)
        if args.state_every and step % args.state_every == 0:
            tensors = {}
            for name in live:
                st = optimizer.state[masters[name]]
                tensors['master/' + name] = masters[name]
                tensors['exp_avg/' + name] = st['exp_avg']
                tensors['exp_avg_sq/' + name] = st['exp_avg_sq']
            adam_step = optimizer.state[parameters[0]]['step'].item()
            saver.save(state_path, tensors, {'step': step, 'adam_step': adam_step, 'cursor': collector.cursor,
                                             'next_group': collector.next_group, 'source_sha256': source_sha})
        learner_seconds = sum(v for k, v in phase.items() if k in ('reference_forward', 'policy_forward_backward', 'optimizer'))
        elapsed = time.time() - run['started']
        record.update(
            loss_tokens=loss_tokens, forward_tokens=forward_tokens, lr=lr, grad_norm=round(grad_norm, 4),
            kl_mean=round(sums['kl'] / loss_tokens, 6), mean_logp=round(sums['logp'] / loss_tokens, 4),
            rho_mean=round(sums['rho'] / loss_tokens, 4), rho_pre_cap_max=round(rho_pre_max, 3),
            rho_post_cap_max=round(rho_post_max, 3), rho_capped_fraction=round(sums['rho_capped'] / loss_tokens, 5),
            abs_log_rho_mean=round(sums['abs_log_rho'] / loss_tokens, 5),
            learner_tokens_per_s=round(forward_tokens / max(1e-6, learner_seconds)),
            actor_generated_tokens=collector.generated, actor_episodes=collector.episodes,
            actor_generated_tokens_per_s=round(collector.generated / max(1., elapsed)),
            phase_s={k: round(v, 1) for k, v in phase.items()}, previous_sync=sync.last,
            previous_state_save=saver.last, cuda_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1),
            outstanding_groups=collector.outstanding, elapsed_s=round(elapsed))
        log_line(log, record)
        if save_native and args.held_every_hours > 0 and time.time() - last_held > args.held_every_hours * 3600:
            sync.wait()
            last_held = time.time()
            start_held(step, work / 'checkpoints' / f'step-{step:05d}')
    sync.wait()
    saver.wait()
    collector.stop.set()
    log_line(log, {'event': 'stopped', 'step': step})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--work', required=True)
    t.add_argument('--checkpoint', required=True, help='start policy (native checkpoint); also the KL reference')
    t.add_argument('--checkpoint-sha256')
    t.add_argument('--base', default=BASE)
    t.add_argument('--actor', help='natlang front of an SGLang actor serving --checkpoint (default: a child actor of this '
                   'process with the held-v2 engine arguments)')
    t.add_argument('--actor-port', type=int, default=8460, help='port of the child actor (its SGLang server: +1000)')
    t.add_argument('--actor-arg', action='append', default=[], help='extra serve_native_checkpoint.py argument')
    t.add_argument('--sglang', help='its SGLang server (default: actor port + 1000)')
    t.add_argument('--actor-export', help="the actor's HF export dir (layout and frozen tables; default WORK/actor/sglang-hf)")
    t.add_argument('--pool', action='append', help='task pool cases.jsonl (repeatable; default: POOLS)')
    t.add_argument('--held-manifest', default=HELD)
    t.add_argument('--exclude-manifest', action='append', default=[], help='more held manifests to exclude')
    t.add_argument('--task-ids', help='JSON list: train only on these pool ids (smoke)')
    t.add_argument('--group', type=int, default=8)
    t.add_argument('--slots', type=int, default=96, help='concurrent episodes on the actor (its default runs 128 sequences)')
    t.add_argument('--min-groups', type=int, default=16)
    t.add_argument('--max-groups', type=int, default=48, help='a w5-like episode costs ~90k learner tokens: 48 groups is ~1-2 h of learner')
    t.add_argument('--max-outstanding', type=int, default=0, help='groups in flight or waiting (default max-groups + slots/group)')
    t.add_argument('--max-staleness', type=int, default=2)
    t.add_argument('--case-seconds', type=int, default=1800)
    t.add_argument('--max-model-requests', type=int, default=64)
    t.add_argument('--max-turns', type=int, default=128)
    t.add_argument('--context-tokens', type=int, default=32768)
    t.add_argument('--seed', type=int, default=20261112)
    t.add_argument('--seed-base', type=int, default=30_000_000)
    t.add_argument('--lr', type=float, default=1e-6)
    t.add_argument('--warmup', type=int, default=2)
    t.add_argument('--weight-decay', type=float, default=0.)
    t.add_argument('--clip', type=float, default=1.0)
    t.add_argument('--kl-beta', type=float, default=0.02)
    t.add_argument('--rho-cap', type=float, default=2.0)
    t.add_argument('--activation-gb', type=float, default=14.)
    t.add_argument('--max-steps', type=int, default=0)
    t.add_argument('--save-every', type=int, default=2)
    t.add_argument('--state-every', type=int, default=4)
    t.add_argument('--state-dir', default=resolve_str('data_hdd', 'natlang-grpo-v1'))
    t.add_argument('--resume', action='store_true')
    t.add_argument('--keep-rollouts', action='store_true')
    t.add_argument('--no-actor-sync', action='store_true',
                   help='never update the actor (smoke against a shared actor of the start policy; versions advance nominally)')
    t.add_argument('--no-pin-runtime', dest='pin_runtime', action='store_false',
                   help="episodes run the checkout's live ts-host instead of the run's pinned copy")
    t.add_argument('--held-every-hours', type=float, default=4.)
    t.add_argument('--skip-base-held', action='store_true')
    t.add_argument('--held-samples', type=int, default=2)
    t.add_argument('--held-workers', type=int, default=48)
    t.add_argument('--unit-prefix', default='grpo')
    args = parser.parse_args()
    if args.command == 'train':
        args.pool = args.pool or POOLS
        args.max_outstanding = args.max_outstanding or args.max_groups + -(-args.slots // args.group)
        train(args)


if __name__ == '__main__':
    main()
