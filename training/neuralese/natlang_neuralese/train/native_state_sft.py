"""Shared full-state supervised training for positional-KV/hybrid-state graphs.

The declared recipe supplies admitted, role-aware native graph JSONL. This
stage trains the exact model used by the native serving path; legacy projected
records are rejected by the shared graph parser.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import open_jsonl_text, write_json_durable as write_json
from ..model.native_model import (CHECKPOINT_SCHEMA, load_native_checkpoint, load_native_model, model_state,
                                  native_identity)
from .checkpoint_policy import CheckpointPolicy
from .loop import capture_training_rng_state, restore_training_rng_state

TRAINING_SCHEMA = 'natlang.neuralese-native-sft-training/1'


def _jsonl(path):
    path = Path(path)
    with open_jsonl_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSONL') from exc


def _load_examples(path, tokenizer_fingerprint, *, required_split):
    from ..data.native_graph import parse_graph
    examples, ids = [], set()
    for row in _jsonl(path):
        example = parse_graph(row, tokenizer_fingerprint=tokenizer_fingerprint)
        if example.split != required_split:
            raise ValueError(f'{path}: expected {required_split} rows, got {example.split!r} in {example.id}')
        if example.id in ids:
            raise ValueError(f'{path}: duplicate example id {example.id!r}')
        ids.add(example.id)
        examples.append(example)
    if not examples:
        raise ValueError(f'{path}: no {required_split} native graph rows')
    return examples


def make_native_optimizer(model, *, lr, weight_decay=.1, latent_ns='quintic', in_backward=True):
    """Reuse the production port optimizer and its in-backward MuonSR memory path."""
    from .backbone_policy import configure_backbone_training
    from .optim import PortMuonAdamW, latent_entries, latent_options, latent_split

    backbone_named = [('backbone.' + name, parameter)
                      for name, parameter in configure_backbone_training(model.port, 'latent')]
    # The native role rows and embedding-space adapter are learned parts of the
    # native interface. The frozen vocabulary/readout and final norm follow the
    # existing shared full-layer backbone policy.
    interface_named = [('interface.' + name, parameter) for name, parameter in model.named_parameters()
                       if name.startswith(('input_rows', 'output_rows', 'adapter.')) and parameter.requires_grad]
    named = backbone_named + interface_named
    latent, ordinary = latent_split(named, 'muonsr')
    if not latent:
        raise ValueError('native SFT found no MuonSR backbone/interface matrices')
    # Register finite-gradient checks before the optimizer's post-accumulate
    # hooks. MuonSR updates a matrix as soon as its gradient is complete.
    if in_backward:
        for name, parameter in latent:
            def finite_before_muon(value, name=name):
                if value.grad is None:
                    return
                finite = torch.isfinite(value.grad).all()
                if value.grad.device.type == 'cuda' and hasattr(torch, '_assert_async'):
                    # Keep the per-matrix hook off the host synchronization path
                    # on large dense/MoE models; this precedes update kernels.
                    torch._assert_async(finite, f'nonfinite native gradient in {name}')
                elif not bool(finite):
                    _raise_nonfinite(name)
            parameter.register_post_accumulate_grad_hook(finite_before_muon)
    optimizer = PortMuonAdamW(ordinary, lr=lr, vocab_size=int(model.port.embedding_weight.shape[0]),
        embedding_ids={id(model.port.embedding_weight)},
        latent=latent_entries(latent, 'muonsr', lr), latent_optimizer='muonsr',
        latent_options=latent_options('muonsr', weight_decay=weight_decay, ns=latent_ns))
    if in_backward and optimizer.latent is not None:
        optimizer.latent.step_in_backward(gated=True)
    return optimizer, named


def _raise_nonfinite(name):
    raise FloatingPointError(f'nonfinite native gradient in {name}; update was not committed')


def _source_receipt(path, examples):
    counts = {}
    for example in examples:
        for group in example.source_groups:
            value = counts.setdefault(group, {'examples': 0, 'supervised_tokens': 0,
                                               'supervised_weight': 0.})
            value['examples'] += 1
            value['supervised_tokens'] += sum(weight > 0 for weight in example.target_weights)
            value['supervised_weight'] += sum(example.target_weights)
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file_hex_cached(path),
            'rows': len(examples), 'by_source_group': counts}


def _source_hash(examples):
    digest = hashlib.sha256()
    for item in examples:
        row = {'id': item.id, 'groups': item.source_groups, 'split': item.split,
               'tokenizer': item.tokenizer_fingerprint, 'targets': item.target_weights,
               'nodes': [{'name': node.name, 'token_ids': node.token_ids,
                          'imports': [edge.__dict__ for edge in node.imports],
                          'producing': node.producing, 'logical_producing': node.logical_producing}
                         for node in item.plan.invocations]}
        value = json.dumps(row, sort_keys=True, separators=(',', ':')).encode()
        digest.update(len(value).to_bytes(8, 'big'))
        digest.update(value)
    return digest.hexdigest()


def _row_bindings(examples):
    """Exact semantic identities needed to prove an append-only data continuation."""
    result = {}
    for item in examples:
        value = {'id': item.id, 'groups': item.source_groups, 'split': item.split,
                 'tokenizer': item.tokenizer_fingerprint, 'targets': item.target_weights,
                 'nodes': [{'name': node.name, 'token_ids': node.token_ids,
                            'imports': [edge.__dict__ for edge in node.imports],
                            'producing': node.producing, 'logical_producing': node.logical_producing}
                           for node in item.plan.invocations]}
        result[item.id] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def append_only_rows(previous, current, *, label):
    """Prove a continuation retained every prior row byte-semantically and only added rows."""
    if not previous:
        raise ValueError(f'native SFT checkpoint has no exact {label} row bindings')
    changed = [key for key, value in previous.items() if current.get(key) != value]
    if changed:
        raise ValueError(f'native SFT {label} continuation modified/removed prior rows: ' + ', '.join(changed[:8]))
    return set(current) - set(previous)


def scheduled_indices(examples, *, seed, epoch, prior_ids=None):
    """Stable epoch order; on append, preserve the old permutation as an exact prefix."""
    order = sorted(range(len(examples)), key=lambda i: hashlib.sha256(
        f'{seed}:{epoch}:{examples[i].id}'.encode()).digest())
    if prior_ids is None:
        return order
    old = [index for index in order if examples[index].id in prior_ids]
    new = [index for index in order if examples[index].id not in prior_ids]
    return old + new


def _evaluate(model, examples, *, checkpoint_layers, chunk_tokens, backend=None):
    from ..data.native_graph import graph_loss
    previous_mode = model.training
    model.eval()
    scope = backend.native_modules() if backend is not None else contextlib.nullcontext()
    try:
        with scope, torch.no_grad():
            total_weight, total_loss, total_correct, total_tokens = 0., 0., 0., 0
            for example in examples:
                loss, detail = graph_loss(model, example, checkpoint_layers=checkpoint_layers,
                                          chunk_tokens=chunk_tokens)
                weight = detail['weight']
                total_loss += float(loss) * weight
                total_correct += detail['accuracy'] * weight
                total_weight += weight
                total_tokens += detail['supervised_tokens']
    finally:
        model.train(previous_mode)
    if not math.isfinite(total_loss) or total_weight <= 0:
        raise FloatingPointError('native held evaluation is nonfinite or empty')
    return {'loss': total_loss / total_weight, 'weighted_accuracy': total_correct / total_weight,
            'weight': total_weight, 'supervised_tokens': total_tokens, 'examples': len(examples)}


def _save(policy, model, optimizer, *, config, identity, step, epoch, cursor, source_receipts,
          source_hash, source_row_bindings, held_row_bindings, exposure_counts, global_step, rng, metrics,
          architecture, options, fp8_state=None):
    state = {'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model), 'step': global_step,
             'training': {'schema': TRAINING_SCHEMA, 'identity': identity, 'step': step,
                          'global_step': global_step, 'source_row_bindings': source_row_bindings,
                          'held_row_bindings': held_row_bindings,
                          'exposure_counts': exposure_counts,
                          'cursor': cursor, 'source_hash': source_hash, 'source_receipts': source_receipts,
                          'optimizer': optimizer.state_dict(), 'rng': rng, 'metrics': metrics,
                          'options': options, 'fp8': fp8_state}}
    policy.save(state, metric=metrics.get('held', {}).get('loss'))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-graphs', type=Path, required=True)
    parser.add_argument('--held-native-graphs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--epochs', type=float, default=1.)
    parser.add_argument('--batch-graphs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--weight-decay', type=float, default=.1)
    parser.add_argument('--latent-ns', choices=('quintic', 'cubic5'), default='quintic')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--checkpoint-layers', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--chunk-tokens', type=int, default=128)
    parser.add_argument('--fp8-training', choices=('off', 'te-fp8', 'triton-experts'), default='off')
    parser.add_argument('--fp8-recipe', choices=('delayed', 'current'), default='delayed')
    parser.add_argument('--fp8-amax-history-len', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--task-eval', default=None)
    parser.add_argument('--task-eval-every', type=int, default=0)
    parser.add_argument('--forward-atol', type=float, default=1e-3)
    parser.add_argument('--gradient-atol', type=float, default=1e-3)
    parser.add_argument('--recipe-binding', default=None)
    parser.add_argument('--stage-id', default=None)
    parser.add_argument('--prerequisite-stages', default='[]')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not (math.isfinite(args.epochs) and args.epochs > 0) or min(args.batch_graphs, args.eval_every,
            args.chunk_tokens, args.rank) < 1 or args.checkpoint_every < 0:
        raise ValueError('native SFT requires positive epochs/batch/eval/chunk/rank and nonnegative checkpoint cadence')
    if args.checkpoint_every and args.checkpoint_every % args.eval_every:
        raise ValueError('checkpoint cadence must be an evaluation cadence multiple')
    if args.task_eval_every < 0 or min(args.forward_atol, args.gradient_atol) <= 0:
        raise ValueError('task cadence must be nonnegative and native numerical tolerances positive')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    policy = CheckpointPolicy(args.out, every_steps=args.checkpoint_every, prune_best=False)
    policy.install_signal_handlers()
    resume = None
    parent_continuation = False
    if args.continue_from:
        model, tokenizer, roles, config, parent_state = load_native_checkpoint(args.continue_from, base=args.base,
                                                                          device=args.device)
        resume = parent_state.get('training')
        if not isinstance(resume, dict) or resume.get('schema') != TRAINING_SCHEMA:
            raise ValueError('native SFT continuation lacks exact optimizer/schedule/RNG state')
        parent_continuation = not policy.slot.exists()
        if parent_continuation:
            del parent_state
    elif policy.slot.exists():
        model, tokenizer, roles, config, state = load_native_checkpoint(policy.slot, base=args.base,
                                                                         device=args.device)
        if state.get('schema') != CHECKPOINT_SCHEMA:
            raise ValueError('rolling slot is not a native-state checkpoint')
        resume = state.get('training')
    else:
        if args.base is None:
            raise ValueError('fresh native SFT needs --base')
        model, tokenizer, roles, config = load_native_model(args.base, rank=args.rank,
                                                             device=args.device, dtype=torch.bfloat16)
    if config['rank'] != args.rank and not args.continue_from and not policy.slot.exists():
        raise ValueError('native rank differs from declared stage rank')
    from ..data.text_corpus import tokenizer_fingerprint
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != config['tokenizer_fingerprint']:
        raise ValueError('loaded tokenizer differs from native model configuration')
    train = _load_examples(args.native_graphs, fingerprint, required_split='train')
    held = _load_examples(args.held_native_graphs, fingerprint, required_split='test')
    train_groups = {group for item in train for group in item.source_groups}
    held_groups = {group for item in held for group in item.source_groups}
    if train_groups & held_groups:
        raise ValueError('native train/held source groups overlap: ' + ', '.join(sorted(train_groups & held_groups)[:8]))
    source_receipts = [_source_receipt(args.native_graphs, train), _source_receipt(args.held_native_graphs, held)]
    source_hash = hashlib.sha256((_source_hash(train) + _source_hash(held)).encode()).hexdigest()
    source_row_bindings = _row_bindings(train)
    held_row_bindings = _row_bindings(held)
    training_identity = {'schema': TRAINING_SCHEMA, 'config': config, 'train_source_hash': source_hash,
                'held_sha256': source_receipts[1]['sha256'], 'optimizer': 'shared-muon-sr-in-backward-v1',
                'optimizer_options': {'lr': args.learning_rate, 'weight_decay': args.weight_decay,
                                      'latent_ns': args.latent_ns},
                'batch_graphs': args.batch_graphs, 'seed': args.seed, 'rank': config['rank'],
                'fp8': {'kind': args.fp8_training, 'recipe': args.fp8_recipe,
                        'amax_history_len': args.fp8_amax_history_len}}
    appended_ids = set()
    if resume is not None:
        previous = resume.get('identity') or {}
        stable = ('schema', 'config', 'optimizer', 'optimizer_options',
                  'batch_graphs', 'seed', 'rank', 'fp8')
        if any(previous.get(key) != training_identity.get(key) for key in stable):
            raise ValueError('native SFT continuation changes model/held/optimizer/schedule identity')
        prior_bindings = resume.get('source_row_bindings') or {}
        appended_ids = append_only_rows(prior_bindings, source_row_bindings, label='training-source')
        prior_held = resume.get('held_row_bindings') or {}
        append_only_rows(prior_held, held_row_bindings, label='held-source')
        if appended_ids:
            print(json.dumps({'event': 'native_data_append', 'prior_rows': len(prior_bindings),
                              'new_rows': len(appended_ids), 'optimizer_state': 'restored',
                              'cursor': resume.get('cursor'), 'policy': 'prior epoch order prefix then appended rows'}),
                  flush=True)
        training_identity['prior_source_hash'] = previous.get('train_source_hash')
    if args.fp8_training != 'off':
        arch = config['architecture']
        expected = 'triton-experts' if arch in {'mellum', 'maple'} else 'te-fp8'
        if args.fp8_training != expected:
            raise ValueError(f'{arch} native SFT requires {expected} for the declared FP8 path')
        from .fp8_backend import make_backend
        backend = make_backend(args.fp8_training, model.port.hf, device=args.device, recipe=args.fp8_recipe,
                               amax_history_len=args.fp8_amax_history_len)
        model.port.activation_checkpoint = backend.checkpoint
    else:
        backend = None
    if config['architecture'] in {'mellum', 'maple'} and backend is None:
        raise ValueError('native Mellum/Maple production SFT requires its shared triton-experts FP8 memory path')
    model.port.checkpoint_layers = args.checkpoint_layers
    model.to(args.device)
    model.train()
    optimizer, named = make_native_optimizer(model, lr=args.learning_rate,
                                              weight_decay=args.weight_decay, latent_ns=args.latent_ns)
    task_eval = None
    task_spec = None
    if args.task_eval:
        from ..eval.task_cadence import TaskEval, parse_task_eval
        task_spec = parse_task_eval(args.task_eval)
        from ..serve.engine import Engine
        from ..serve.store import TensorStore
        # Construct the real native serving engine over the same live model,
        # avoiding a second full-width model allocation during evaluation.
        serving_identity = native_identity(model, config, checkpoint_sha256='0' * 64)
        engine = Engine(model.port, None, tokenizer, TensorStore(),
            f"native-state/{config['architecture']}", model_name=f"native-{config['architecture']}",
            device=args.device, prompt_rendering=task_spec.get('prompt_rendering', 'full'),
            shared_model=model, role_tokens=roles, model_config=config,
            checkpoint_sha256=serving_identity['checkpoint_sha256'],
            embedding_weight_sha256=serving_identity['embedding_weight_sha256'])
        task_eval = TaskEval(task_spec, args.out, engine, prompt_rendering=task_spec.get('prompt_rendering', 'full'))
    step, epoch, cursor, global_step = 0, 0, 0, 0
    metrics = {}
    exposure_counts = {key: 0 for key in source_row_bindings}
    schedule_prefix_ids = set(resume['source_row_bindings']) if resume is not None and appended_ids else None
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        if backend is not None:
            if resume.get('fp8') is None:
                raise ValueError('native SFT checkpoint lacks the declared FP8 state')
            backend.restore_extra_state(resume['fp8'])
        if parent_continuation:
            # A new declared phase consumes its current full source set while
            # keeping exact weights/optimizer/RNG; its stage-local cursor begins
            # at zero and the global update counter remains monotonic. The
            # predecessor's exposure is in its own readiness receipt, so this
            # stage starts its own exposure counters at zero.
            global_step = int(resume.get('global_step', resume['step']))
            metrics = {}
        else:
            step, epoch, cursor = int(resume['step']), int(resume['epoch']), int(resume['cursor'])
            global_step = int(resume.get('global_step', step))
            exposure_counts.update({key: int(value) for key, value in (resume.get('exposure_counts') or {}).items()
                                    if key in exposure_counts})
            metrics = dict(resume.get('metrics') or {})
        restore_training_rng_state(resume['rng'], args.device)
        resume['optimizer'] = None
        if 'state' in locals():
            del state
        if 'parent_state' in locals():
            del parent_state
    append_epoch = epoch
    # Each epoch visits every admitted graph once. The order is keyed by epoch
    # and seed; cursor is a consumed prefix so signal-boundary resumes are exact.
    def epoch_order(e):
        prior_ids = schedule_prefix_ids if e == append_epoch else None
        return scheduled_indices(train, seed=args.seed, epoch=e, prior_ids=prior_ids)
    whole_epochs = int(args.epochs)
    fractional_rows = math.ceil((args.epochs - whole_epochs) * len(train))
    target_rows = whole_epochs * len(train) + fractional_rows
    total_updates = whole_epochs * math.ceil(len(train) / args.batch_graphs)
    if fractional_rows:
        total_updates += math.ceil(fractional_rows / args.batch_graphs)
    total_end = total_updates
    if parent_continuation:
        step = 0
    if step > total_end:
        raise ValueError('native checkpoint update is beyond the declared epoch exposure')
    resumed_step = step
    last_saved_step = step if policy.slot.is_file() and not parent_continuation else -1
    eval_at = max(1, args.eval_every)
    started = time.time()
    while step < total_end and not policy.signaled:
        batch = []
        while len(batch) < args.batch_graphs and epoch < math.ceil(args.epochs):
            order = epoch_order(epoch)
            limit = (fractional_rows if epoch == whole_epochs and fractional_rows else len(order))
            while cursor < limit and len(batch) < args.batch_graphs:
                batch.append(train[order[cursor]])
                cursor += 1
            if cursor >= limit:
                epoch += 1
                cursor = 0
        if not batch:
            break
        if backend is not None:
            backend.reset_row_counts()
        optimizer.zero_grad(set_to_none=True)
        if optimizer.latent is not None:
            context = optimizer.latent.in_backward()
        else:
            from contextlib import nullcontext
            context = nullcontext()
        try:
            with context:
                losses = []
                for example in batch:
                    if backend is not None:
                        with backend.autocast():
                            loss, detail = _graph_loss(model, example, args)
                    else:
                        loss, detail = _graph_loss(model, example, args)
                    losses.append(loss)
                torch.stack(losses).mean().backward()
            if optimizer.latent is not None:
                optimizer.latent._armed = None
            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if params:
                norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
                if not torch.isfinite(norm):
                    raise FloatingPointError(f'nonfinite native gradient norm at update {step + 1}')
            optimizer.step()
        except Exception:
            if optimizer.latent is not None:
                optimizer.latent._armed = None
            optimizer.zero_grad(set_to_none=True)
            raise
        step += 1
        global_step += 1
        for example in batch:
            exposure_counts[example.id] += 1
        due_task = task_eval is not None and (step == total_end or
                    bool(args.task_eval_every and step % args.task_eval_every == 0))
        due_eval = step % eval_at == 0 or step == total_end or due_task
        if due_eval:
            metrics = {'train_loss': float(torch.stack([item.detach() for item in losses]).mean()),
                       'held': _evaluate(model, held, checkpoint_layers=args.checkpoint_layers,
                                         chunk_tokens=args.chunk_tokens, backend=backend),
                       'step': step, 'epoch': epoch, 'cursor': cursor,
                       'elapsed_seconds': time.time() - started,
                       'fp8': backend.inventory() if backend is not None else {'enabled': False}}
            write_json(args.out / 'metrics.json', metrics)
        if due_task:
            # Bind the task proof to the checkpoint from these exact weights.
            _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
                  global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
                  source_hash=source_hash, source_row_bindings=source_row_bindings,
                  held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
                  rng=capture_training_rng_state(args.device), metrics=metrics,
                  architecture=config['architecture'], options=vars(args),
                  fp8_state=backend.extra_state() if backend is not None else None)
            last_saved_step = step
            checkpoint_sha = sha256_file_hex_cached(policy.slot)
            serving_identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha)
            engine = task_eval.engine
            engine.checkpoint_sha256 = serving_identity['checkpoint_sha256']
            engine.embedding_weight_sha256 = serving_identity['embedding_weight_sha256']
            engine.native_identity = serving_identity
            saved_rng = capture_training_rng_state(args.device)
            model.eval()
            try:
                if backend is not None:
                    with backend.native_modules():
                        metrics['task_eval'] = task_eval.cadence(step)
                else:
                    metrics['task_eval'] = task_eval.cadence(step)
            finally:
                model.train()
                restore_training_rng_state(saved_rng, args.device)
            write_json(args.out / 'metrics.json', metrics)
            # Keep the exact task receipt resumable with these same weights and
            # optimizer state. TaskEval restores the training RNG above, so
            # this write does not advance the training stream.
            _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
                  global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
                  source_hash=source_hash, source_row_bindings=source_row_bindings,
                  held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
                  rng=capture_training_rng_state(args.device), metrics=metrics,
                  architecture=config['architecture'], options=vars(args),
                  fp8_state=backend.extra_state() if backend is not None else None)
            last_saved_step = step
        if policy.due(step, end=step == total_end) and not due_task:
            _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
                  global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
                  source_hash=source_hash, source_row_bindings=source_row_bindings,
                  held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
                  rng=capture_training_rng_state(args.device), metrics=metrics,
                  architecture=config['architecture'], options=vars(args),
                  fp8_state=backend.extra_state() if backend is not None else None)
            last_saved_step = step
    if (step != resumed_step or policy.signaled) and last_saved_step != step:
        if not metrics.get('held'):
            metrics['held'] = _evaluate(model, held, checkpoint_layers=args.checkpoint_layers,
                                        chunk_tokens=args.chunk_tokens, backend=backend)
        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
              global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
              source_hash=source_hash, source_row_bindings=source_row_bindings,
              held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
              rng=capture_training_rng_state(args.device), metrics=metrics,
              architecture=config['architecture'], options=vars(args),
              fp8_state=backend.extra_state() if backend is not None else None)
        last_saved_step = step
    completed = epoch >= args.epochs and (whole_epochs == int(args.epochs) or fractional_rows > 0)
    # A continuation may inherit a completed checkpoint made before task
    # summaries were checkpointed. At the same final weights, make one real
    # cadence call; do not reuse an unbound log row or enter final_at's retry
    # loop merely to force an eligible score.
    if task_eval is not None and completed and not metrics.get('task_eval'):
        current_sha = sha256_file_hex_cached(policy.slot)
        serving_identity = native_identity(model, config, checkpoint_sha256=current_sha)
        engine = task_eval.engine
        engine.checkpoint_sha256 = serving_identity['checkpoint_sha256']
        engine.embedding_weight_sha256 = serving_identity['embedding_weight_sha256']
        engine.native_identity = serving_identity
        saved_rng = capture_training_rng_state(args.device)
        model.eval()
        try:
            if backend is not None:
                with backend.native_modules():
                    metrics['task_eval'] = task_eval.cadence(step)
            else:
                metrics['task_eval'] = task_eval.cadence(step)
        finally:
            model.train()
            restore_training_rng_state(saved_rng, args.device)
        write_json(args.out / 'metrics.json', metrics)
        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
              global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
              source_hash=source_hash, source_row_bindings=source_row_bindings,
              held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
              rng=capture_training_rng_state(args.device), metrics=metrics,
              architecture=config['architecture'], options=vars(args),
              fp8_state=backend.extra_state() if backend is not None else None)
    checkpoint_sha = sha256_file_hex_cached(policy.slot)
    model_identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha)
    task_summary = metrics.get('task_eval')
    task_receipt_path = None
    if task_summary:
        task_receipt_path = args.out / 'native-task-execution-receipt.json'
        score_path = Path(task_summary['run']) / 'task-score.json' if task_summary.get('run') else None
        task_receipt = {'schema': 'natlang.native-task-execution-receipt/1',
            'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
            'packet_sha256': task_summary.get('packet_sha256'),
            'case_indices': task_summary.get('case_indices'), 'cases': task_summary.get('cases'),
            'complete': task_summary.get('complete'), 'eligible': task_summary.get('eligible'),
            'task_successes': task_summary.get('task_successes'),
            'task_successes_are_qualification_barrier': False,
            'score_path': str(score_path.resolve()) if score_path and score_path.is_file() else None,
            'score_sha256': sha256_file_hex_cached(score_path) if score_path and score_path.is_file() else None}
        task_receipt['passed'] = bool(task_receipt['complete'] and task_receipt['eligible'] and
                                      task_receipt['score_sha256'])
        write_json(task_receipt_path, task_receipt)
    runtime_path = args.out / 'native-runtime-certificate.json'
    if not completed:
        # Operational stops preserve the exact resumable checkpoint. Defer a
        # full capture/splice qualification until the declared data exposure is
        # finished; do not make interruptions pay for a second model pass.
        runtime_report = {'schema': 'natlang.native-runtime-certificate/1', 'passed': False,
                          'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
                          'deferred': True, 'reason': 'declared native graph exposure is incomplete'}
    else:
        try:
            from ..eval.native_state import qualify_existing
            if backend is not None:
                with backend.native_modules():
                    runtime_report = qualify_existing(model, tokenizer, roles, config,
                        checkpoint_sha256=checkpoint_sha, forward_atol=args.forward_atol,
                        gradient_atol=args.gradient_atol,
                        task_execution_receipt=(str(task_receipt_path.resolve()) if task_receipt_path else None),
                        backbone_optimizer_updates=step)
            else:
                runtime_report = qualify_existing(model, tokenizer, roles, config,
                    checkpoint_sha256=checkpoint_sha, forward_atol=args.forward_atol,
                    gradient_atol=args.gradient_atol,
                    task_execution_receipt=(str(task_receipt_path.resolve()) if task_receipt_path else None),
                    backbone_optimizer_updates=step)
        except Exception as error:  # record a real failed check without discarding the trained checkpoint
            runtime_report = {'schema': 'natlang.native-runtime-certificate/1', 'passed': False,
                              'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
                              'error': f'{type(error).__name__}: {error}'}
    write_json(runtime_path, runtime_report)
    task_receipt_ok = bool(task_receipt_path and json.loads(task_receipt_path.read_text()).get('passed'))
    recipe_binding = json.loads(args.recipe_binding) if args.recipe_binding else None
    prerequisites = json.loads(args.prerequisite_stages)
    exposure_by_source_group = {}
    for example in train:
        visits = int(exposure_counts[example.id])
        token_visits = visits * sum(weight > 0 for weight in example.target_weights)
        for group in example.source_groups:
            counts = exposure_by_source_group.setdefault(group, {'graph_visits': 0,
                                                                   'supervised_token_visits': 0})
            counts['graph_visits'] += visits
            counts['supervised_token_visits'] += token_visits
    exposure_counts_sha256 = hashlib.sha256(json.dumps(sorted(exposure_counts.items()),
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    report = {'schema': 'natlang.neuralese-native-sft-report/1', 'completed': completed,
              'numerics_ok': runtime_report.get('passed') is True, 'qualified': False,
              'quality_status': 'measured-for-supervisor',
              'step': step, 'epoch': epoch, 'cursor': cursor, 'training_seconds': time.time() - started,
              'model_config': config, 'train_source': source_receipts[0], 'held_source': source_receipts[1],
              'source_hash': source_hash, 'metrics': metrics, 'checkpoint': str(policy.slot),
              'checkpoint_sha256': checkpoint_sha, 'task_eval': task_summary,
              'runtime_certificate': {'path': str(runtime_path.resolve()),
                                      'sha256': sha256_file_hex_cached(runtime_path),
                                      'passed': runtime_report.get('passed') is True},
              'base_kl': {'status': 'telemetry-only', 'measured': False}}
    write_json(args.out / 'report.json', report)
    write_json(args.out / 'native-training-receipt.json', {
        'schema': 'natlang.native-training-readiness/1', 'model_id': config['architecture'] + ':' +
        config['base_inventory_sha256'], 'lineage': str(args.out.resolve()),
        'stage_id': args.stage_id or args.out.name,
        'checkpoint': {'path': str(policy.slot.resolve()), 'sha256': report['checkpoint_sha256'],
                       'schema': CHECKPOINT_SCHEMA},
        'native_identity': model_identity,
        'runtime_certificate': {'path': str(runtime_path.resolve()),
                                'sha256': sha256_file_hex_cached(runtime_path),
                                'passed': runtime_report.get('passed') is True},
        'prerequisite_stages': prerequisites, 'recipe': recipe_binding,
        'completed': bool(completed and runtime_report.get('passed') is True and task_receipt_ok and
                          all(item.get('completed') is True for item in prerequisites)),
        'task_eval': task_summary,
        'task_execution_receipt': ({'path': str(task_receipt_path.resolve()),
                                     'sha256': sha256_file_hex_cached(task_receipt_path), 'passed': task_receipt_ok}
                                    if task_receipt_path else None),
        'training_receipt': {'path': str((args.out / 'report.json').resolve()),
                                                      'sha256': sha256_file_hex_cached(args.out / 'report.json')},
        'exposure': {'train_rows': len(train), 'held_rows': len(held),
                     'supervised_tokens': sum(sum(w > 0 for w in x.target_weights) for x in train),
                     'graph_visits': sum(exposure_counts.values()),
                     'supervised_token_visits': sum(exposure_counts[x.id] *
                         sum(weight > 0 for weight in x.target_weights) for x in train),
                     'row_update_counts_sha256': exposure_counts_sha256,
                     'by_source_group': exposure_by_source_group,
                     'consumed_rows': min(target_rows, epoch * len(train) + cursor),
                     'consumed_updates': step, 'completed_epochs': epoch}})
    return 0


def _graph_loss(model, example, args):
    from ..data.native_graph import graph_loss
    return graph_loss(model, example, checkpoint_layers=args.checkpoint_layers,
                      chunk_tokens=args.chunk_tokens)


if __name__ == '__main__':
    raise SystemExit(main())
