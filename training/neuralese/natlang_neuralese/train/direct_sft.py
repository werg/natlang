"""Neuralese direct v1 (plans/neuralese/NEURALESE_DIRECT_V1.md): plain whole-trajectory target SFT of a base model
on the native records, one pass on ordinary token embeddings, with the ``NeuraleseProjection`` producing the Neuralese
value at the supervised (Neuralese) positions; the ``TokenChoiceProjection`` is a declared component that this stage
leaves off. Recipe stage kind ``neuralese_direct``.

Per update: one precision point (the recipe's quantization component, sampled as everywhere), one forward over each
window's ``prefix ++ span[:-1]``, the weighted CE of the frozen readout (final norm + lm_head) on the span
(``text_supervision.balanced_position_weights``), plus ``--neuralese-weight`` x the NeuraleseProjection's relative MSE to
the gold next-token embedding on detached hidden states. Backbone attention and expert matrices step on the latent
optimizer (LionSR or MuonSR, in backward); routers, norms and the projections on PortMuonAdamW.

Every ``--eval-every`` updates (declared points): held CE, gold accuracy and agreement against the base model's own
predictions on the same windows, the Neuralese value's error against its soft-token-mixture baseline, and
``behaviour_drift`` with its guard. At the end: the behaviour gate (texts kept). Full state at declared checkpoint points,
on a stop and at the end; a restart resumes from it.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import math
import random
import time
from array import array
from pathlib import Path

import torch

from .loop import StopSignal, TrainingLoop, capture_training_rng_state, restore_training_rng_state
from .checkpoint_policy import CheckpointPolicy

SCHEMA = 'natlang.neuralese-direct/1'
OPERATIONAL = frozenset({'steps', 'checkpoint_every', 'eval_every', 'drift_every', 'drift_gate_every', 'drift_gate_now', 'device', 'value_margin', 'save_state', 'moe_kernel', 'checkpoint_layers', 'drift_rows', 'drift_warn',
                         'drift_stop', 'drift_agreement_warn', 'drift_agreement_stop', 'drift_gate', 'eval_batch',
                         'gate_held_rows'})


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('heads', 'records', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('pieces', 'text-data'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--tokens', type=int, default=8192, help='window capacity: the whole preceding document, system '
                   'prompt included, up to this many tokens')
    p.add_argument('--target-tokens', type=int, default=1024, help='supervised (Neuralese) positions per window')
    p.add_argument('--prefix-tokens', type=int, default=32)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--eval-batch', type=int, default=1)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--checkpoint-every', type=int, default=0)
    p.add_argument('--held-documents', type=int, default=8)
    p.add_argument('--context-weight', type=float, default=1.0)
    p.add_argument('--feedback-weight', type=float, default=.25, help='tool-output weight inside context supervision')
    p.add_argument('--backbone-lr', type=float, default=7.5e-6, help='AdamW lr of routers and norms')
    p.add_argument('--latent-optimizer', choices=('lionsr', 'muonsr'), default='lionsr')
    p.add_argument('--latent-lr', type=float, default=3e-6,
                   help='lionsr: per row ternary scale; muonsr: on AdamW\'s scale')
    p.add_argument('--head-lr', type=float, default=3e-4, help='the projections\' lr (Muon/AdamW)')
    p.add_argument('--neuralese-weight', type=float, default=1.0)
    p.add_argument('--projection-blocks', type=int, default=2)
    p.add_argument('--projection-hidden', type=int, default=0, help='0: width/4')
    p.add_argument('--projection-kernel', type=int, default=4)
    p.add_argument('--token-choice-projection', action=argparse.BooleanOptionalAction, default=False,
                   help='declare the TokenChoiceProjection (the end-to-end Neuralese stage); off in direct v1')
    p.add_argument('--moe-kernel', choices=('loop', 'grouped'), default='loop')
    p.add_argument('--checkpoint-layers', action=argparse.BooleanOptionalAction, default=True,
                   help='recompute each layer in backward (long windows)')
    p.add_argument('--quantization')
    p.add_argument('--monitor', help='JSON (or a path to it) of the monitors without a quantization component: '
                   '{"gate": <quantization.gate schema>, "data": {"path", "sha256"}}; "data" is JSONL of chat '
                   'prompts whose base-model answers and top-k are captured at step 0 as the drift reference')
    p.add_argument('--drift-rows', type=int, default=16)
    p.add_argument('--drift-gate-every', type=int, default=300,
                   help='while the guard warns, run the behaviour gate at most every this many updates (KL still every '
                        '--drift-every); always at the stage end')
    p.add_argument('--drift-gate-now', type=float, default=0.35,
                   help='run the gate at once, whatever --drift-gate-every, when response KL reaches this level')
    p.add_argument('--drift-every', type=int, default=0,
                   help='also run the drift monitor (held KL on --drift-rows, its guard and the gate at a warn) every '
                        'this many updates between evals, logged to drift.jsonl; 0: only at evals')
    # The drift guard (owner 2026-10-11): response KL or agreement past warn runs the behaviour gate at that eval and
    # stops when the gate falls below its reference minus max_gate_drop; response KL past --drift-stop stops at once.
    p.add_argument('--drift-warn', type=float, default=.05)
    p.add_argument('--drift-stop', type=float, default=.5)
    p.add_argument('--drift-agreement-warn', type=float, default=.97)
    p.add_argument('--drift-agreement-stop', type=float, default=0.)
    p.add_argument('--drift-gate', action=argparse.BooleanOptionalAction, default=True,
                   help='run the behaviour gate at an eval whose drift is past warn, and stop on a gate regression')
    p.add_argument('--gate-held-rows', type=int, default=48)
    p.add_argument('--save-state', action=argparse.BooleanOptionalAction, default=True,
                   help='write the full training state (weights, optimizer); smokes pass --no-save-state and keep '
                        'only their logs and report')
    p.add_argument('--value-margin', type=float, default=0.0,
                   help='report-only: value_qualified says whether the final held value_relative_mse_delta is below '
                        'minus this margin (the value requirement is the end-to-end stage\'s, owner 2026-10-11)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    if min(a.steps, a.tokens, a.target_tokens, a.batch, a.eval_batch, a.eval_every, a.held_documents) < 1:
        p.error('positive bounds required')
    if a.checkpoint_every < 0 or a.neuralese_weight < 0:
        p.error('invalid checkpoint cadence or Neuralese weight')
    return a


def context_only_spans(row):
    """A text row's producer-declared context-only spans: ``[[begin, end), ...]`` in its ``token_ids`` (so they need
    native token IDs), sorted and non-overlapping. They stay in the window as history and get zero loss weight."""
    spans = row['context_only_spans']
    ids = row.get('token_ids')
    if ids is None:
        raise ValueError('context_only_spans require native token_ids')
    previous = 0
    for span in spans:
        if (not isinstance(span, (list, tuple)) or len(span) != 2 or any(type(x) is not int for x in span)
                or not previous <= span[0] < span[1] <= len(ids)):
            raise ValueError(f'context_only_spans must be sorted, non-overlapping [begin, end) token spans: {span}')
        previous = span[1]
    return [tuple(span) for span in spans]


def context_only_mask(source, labels, declared, decode):
    """1 at every position of a framed document (``source``: open token, the row's token_ids, close token) that is
    context only: assistant turns answered by a runtime failure (needs role ``labels``) and the declared spans."""
    from .text_supervision import failed_assistant_mask
    mask = failed_assistant_mask(source, labels, decode) if labels is not None else array('B', bytes(len(source)))
    for begin, end in declared:
        for position in range(begin + 1, min(end + 1, len(mask))):  # +1: the leading open token
            mask[position] = 1
    return mask


def segment_ids(roles: torch.Tensor | None, span_start: int, length: int, batch: int, device) -> torch.Tensor:
    """One segment per maximal run of one chat role inside the supervised span (a turn); context positions -1."""
    segments = torch.full((batch, length), -1, dtype=torch.long, device=device)
    if roles is None:
        segments[:, span_start:] = 0
        return segments
    span_roles = roles[:, span_start:length]
    change = torch.ones_like(span_roles, dtype=torch.long)
    change[:, 1:] = (span_roles[:, 1:] != span_roles[:, :-1]).long()
    segments[:, span_start:span_start + span_roles.shape[1]] = change.cumsum(1) - 1
    return segments


def relative_mse(predicted, target, weights=None):
    error = (predicted.float() - target.float()).square().sum(-1)
    scale = target.float().square().sum(-1).clamp_min(1e-12)
    per_position = error / scale
    if weights is None:
        return per_position.mean()
    return (per_position * weights).sum() / weights.sum().clamp_min(1e-12)


def main(argv=None):
    stop = StopSignal().install()
    a = parse(argv)
    from ..common.hashing import sha256_file_hex
    from ..maple.generation_gate import behaviour_report, held_kl
    from ..maple.distill_data import render
    from ..model.neuralese_projection import (NeuraleseProjection, TokenChoiceProjection,
                                               project_token_states, soft_token_mixture)
    from ..serve import load_engine
    from .backbone_policy import configure_backbone_training, plain_named_tensors, shared_parametrized_weights
    from .optim import PortMuonAdamW
    from .quantization import Behaviour, load_component, teacher_shards
    from .text_supervision import balanced_position_weights
    from .text_warmup import (behaviour_drift_level, chunked_readout, evaluation_batches, load_text_rows,
                              prepare_text_windows, select_held_document_windows)
    from ..common.paths import root as path_root

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    code = {str(path.relative_to(Path(__file__).parents[1])): sha256_file_hex(path)
            for path in sorted(Path(__file__).parents[1].rglob('*.py'))}
    inputs = {str(path): sha256_file_hex(path) for path in (a.heads, a.records, a.pieces, a.text_data) if path}
    identity = {'schema': SCHEMA, 'options': {k: v for k, v in options.items() if k not in OPERATIONAL},
                'inputs': inputs, 'code': code}

    engine = load_engine(heads_checkpoint=str(a.heads), device=a.device)
    backbone = engine.backbone
    backbone.checkpoint_layers = a.checkpoint_layers
    from ..maple.model import set_dense_moe_kernel
    set_dense_moe_kernel(a.moe_kernel)
    quant = load_component(a.quantization, root=path_root('repo'))
    monitor = None
    if a.monitor:
        text = Path(a.monitor).read_text() if not a.monitor.lstrip().startswith('{') else a.monitor
        monitor = json.loads(text)
    behaviour = None
    gate_held = None
    if quant is not None:
        preserve = quant.component.get('preserve') or {}
        if float(preserve.get('weight', 0.0)) > 0:
            raise ValueError('neuralese_direct trains no self-distillation: the quantization component\'s preserve '
                             'weight must be 0 (its teacher is only the drift monitor\'s held data)')
        installed = quant.install(backbone.hf, layer_count=backbone.num_layers)
        print(json.dumps({'event': 'quantization_installed', 'groups': installed, 'stage': quant.stage}), flush=True)
        if preserve.get('teacher'):
            gate_held = teacher_shards(preserve['teacher']['artifact'])[1]
        if quant.component.get('gate'):
            behaviour = Behaviour(quant.component['gate'], engine.tokenizer, gate_held, root=path_root('repo'))
    if monitor is not None:
        from ..maple.generation_gate import capture_reference
        if gate_held is None and monitor.get('data'):
            # The monitor-only drift reference of a lineage without a teacher artifact (LFM): the base model's own
            # answers and top-k on the monitor prompts, captured before any update and kept with the run.
            reference_path = a.out / 'monitor-reference.pt'
            if reference_path.is_file():
                gate_held = torch.load(reference_path, map_location='cpu', weights_only=False)
            else:
                data = path_root('repo') / monitor['data']['path'] if not Path(monitor['data']['path']).is_absolute() \
                    else Path(monitor['data']['path'])
                if sha256_file_hex(data) != monitor['data']['sha256']:
                    raise ValueError('monitor prompts changed since the recipe pinned them: ' + str(data))
                prompts = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
                eos = engine.tokenizer.convert_tokens_to_ids('<|im_end|>')
                gate_held = capture_reference(backbone.hf, engine.tokenizer, prompts, render, eos)
                torch.save(gate_held, reference_path)
                print(json.dumps({'event': 'monitor_reference_captured', 'prompts': len(prompts)}), flush=True)
        if behaviour is None and monitor.get('gate'):
            behaviour = Behaviour(monitor['gate'], engine.tokenizer, gate_held, root=path_root('repo'))
    # Trainables: backbone (latent policy), the projections. Embedding, final norm and lm_head stay frozen.
    for parameter in engine.heads.parameters():
        parameter.requires_grad_(False)
    backbone_named = [('backbone.' + n, q) for n, q in configure_backbone_training(backbone, 'latent')]
    width = int(backbone.embedding_weight.shape[1])
    sizes = dict(blocks=a.projection_blocks, hidden=a.projection_hidden or None, kernel=a.projection_kernel)
    neuralese = NeuraleseProjection(width, **sizes).to(a.device)
    token_choice = TokenChoiceProjection(width, **sizes).to(a.device) if a.token_choice_projection else None
    projections = {'neuralese_projection': neuralese, **({'token_choice_projection': token_choice}
                                                         if token_choice is not None else {})}
    head_named = [(key + '.' + n, q) for key, module in projections.items() for n, q in module.named_parameters()]
    latent_named = [(n, q) for n, q in backbone_named if q.ndim >= 2 and '.mlp.gate.' not in n]
    other_backbone = [(n, q) for n, q in backbone_named if not (q.ndim >= 2 and '.mlp.gate.' not in n)]
    from ..maple.qat_convert import row_scale
    latent = [(n, q, a.latent_lr, row_scale(q) if a.latent_optimizer == 'lionsr' else None) for n, q in latent_named]
    optimizer = PortMuonAdamW(head_named + other_backbone, lr=a.head_lr, vocab_size=int(backbone.embedding_weight.shape[0]),
                              latent=latent, latent_optimizer=a.latent_optimizer)
    other_ids = {id(q) for _, q in other_backbone}
    for child in (optimizer.muon, optimizer.auxiliary):  # routers and norms at the backbone lr
        if child is None:
            continue
        for group in list(child.param_groups):
            backbone_params = [q for q in group['params'] if id(q) in other_ids]
            if backbone_params and len(backbone_params) == len(group['params']):
                group['lr'] = a.backbone_lr
            elif backbone_params:
                group['params'] = [q for q in group['params'] if id(q) not in other_ids]
                child.add_param_group({**{k: v for k, v in group.items() if k != 'params'}, 'params': backbone_params,
                                       'lr': a.backbone_lr})
    optimizer.param_groups = optimizer._groups()
    in_backward = optimizer.latent
    in_backward.step_in_backward(gated=True)
    print(json.dumps({'event': 'trainables', 'latent_optimizer': a.latent_optimizer, 'latents': len(latent),
                      'adamw_or_muon_backbone': [n for n, _ in other_backbone][:4] + ['...'],
                      'projections': {k: sum(q.numel() for q in m.parameters()) for k, m in projections.items()}}),
          flush=True)

    # Windows: the whole preceding document (system prompt included) up to --tokens, --target-tokens supervised.
    rows, receipt = load_text_rows(a.records, a.pieces, a.text_data, tokenizer=engine.tokenizer)
    windows, mask_receipt = prepare_text_windows(engine, rows, tokens=a.tokens, prefix_tokens=a.prefix_tokens,
                                                 target_tokens=a.target_tokens, mask_system_prompt=True)
    train_windows = windows['train']
    held, held_selection = select_held_document_windows(windows['test'], a.held_documents)
    if not train_windows or not held:
        raise ValueError('neuralese_direct needs train and held windows')
    print(json.dumps({'event': 'windows', 'train': len(train_windows), 'held': len(held),
                      'held_selection': {k: held_selection.get(k) for k in ('policy', 'limit_documents')},
                      'system_prompt_context_tokens': mask_receipt.get('masked_system_tokens')}), flush=True)
    table = backbone.embedding_weight.detach().float().clone()
    table[backbone.controls.open_id] = backbone.control_rows[0].detach().float()
    table[backbone.controls.close_id] = backbone.control_rows[1].detach().float()

    failed_by_document = {}
    declared_by_document = {}
    for row in rows:  # producer-declared context-only token spans (excluded assistant targets kept as history)
        if row.get('context_only_spans'):
            declared_by_document[hashlib.sha256(row['text'].encode()).hexdigest()] = context_only_spans(row)

    def failed(window):
        """The window's slice of its document's context-only mask (computed once per document): assistant turns
        answered by a runtime failure, plus the producer's declared ``context_only_spans`` (token coordinates in the
        row's ``token_ids``, end exclusive)."""
        source = window['ids']._source
        key = id(source)
        if key not in failed_by_document:
            failed_by_document[key] = context_only_mask(
                source, window['roles']._source if 'roles' in window else None,
                declared_by_document.get(window['document'], ()), engine.tokenizer.decode)
        mask = failed_by_document[key]
        return [mask[i] for i in range(window['start'], window['start'] + len(window['ids']))]

    def tensors(rows_):
        ids = torch.tensor([list(r['ids']) for r in rows_], device=a.device)
        prefix = rows_[0]['prefix']
        span = ids[:, prefix:]
        roles = (torch.tensor([list(r['roles']) for r in rows_], device=a.device)
                 if all('roles' in r for r in rows_) else None)
        exclude = (torch.tensor([failed(r) for r in rows_], device=a.device)[:, prefix:].bool()
                   if roles is not None or declared_by_document else None)
        weights = balanced_position_weights(span, [r.get('supervised_suffix_start') for r in rows_],
                                            roles=None if roles is None else roles[:, prefix:],
                                            context_weight=a.context_weight, feedback_weight=a.feedback_weight,
                                            exclude=exclude)
        return ids, prefix, span, roles, weights

    def forward(rows_, *, gradients):
        """(weighted CE, metrics, per-position hidden/value tensors) of one same-shape batch at the active precision."""
        ids, prefix, span, roles, weights = tensors(rows_)
        start = prefix - 1
        usable = float(weights.sum()) > 0  # a window made only of failed assistant turns teaches nothing
        if not usable:
            weights = torch.ones_like(weights)
        out = backbone.forward_ids(ids[:, :-1], cutoff=engine.heads.cutoff, logits=False)
        hidden = out['h_final']
        segments = segment_ids(roles[:, :-1] if roles is not None else None, start, hidden.shape[1], hidden.shape[0],
                               hidden.device)
        # Both projections read only the supervised positions: segments start inside the span (context is -1), so
        # no convolution tap reaches the context and the slice is exact. Never full-context logits or mixtures.
        hidden, segments = hidden[:, start:], segments[:, start:]
        # Project once over the whole turn so every chunk sees its full causal
        # history, then keep the vocabulary-sized readout chunked as before.
        read = project_token_states(hidden, token_choice, segments)
        ce, training_ce, prediction, _, token_losses = chunked_readout(
            backbone, read, span, backbone.controls.close_id, gradients=gradients, position_weights=weights)
        mixture = soft_token_mixture(backbone.logits, table, hidden.detach())
        value = neuralese(hidden.detach(), mixture.to(hidden.dtype), segments)
        target = table[span]
        value_error = relative_mse(value, target, weights)
        metrics = {'ce': float(ce), 'gold_accuracy': float((prediction == span).float().mean()),
                   'value_relative_mse': float(value_error)}
        if not torch.is_grad_enabled():
            mixture_error = relative_mse(mixture, target, weights)
            nearest = (2 * (value.float() @ table.t()) - table.square().sum(-1)).argmax(-1)
            metrics.update(value_mixture_relative_mse=float(mixture_error),
                           value_relative_mse_delta=float(value_error - mixture_error),
                           value_nearest_accuracy=float((nearest == span).float().mean()))
        loss = training_ce + a.neuralese_weight * value_error
        return (loss if usable else loss * 0.0), metrics, prediction, token_losses

    # Resume.
    policy = CheckpointPolicy(a.out, every_steps=a.checkpoint_every)
    if not a.save_state:
        policy.save = lambda state, **_: None  # a smoke: logs and report only, never a full state
    step, best, base_eval, history = 0, None, None, []
    last_gate = None  # the update at which the guard last ran the behaviour gate
    if policy.slot.exists():
        state = CheckpointPolicy.load(policy.slot)
        if state.get('schema') != SCHEMA or state['identity']['options'] != identity['options'] or \
                state['identity']['inputs'] != identity['inputs']:
            raise ValueError('neuralese_direct resume identity changed (options or inputs)')
        tensors_by_name = plain_named_tensors(backbone.hf)
        with torch.no_grad():
            for name, value in state['student_parameters'].items():
                tensors_by_name[name.removeprefix('backbone.')].copy_(value.to(tensors_by_name[name.removeprefix('backbone.')]))
        for key, module in projections.items():
            module.load_state_dict(state[key])
        from .optim_restore import own_host_tensors
        optimizer.load_state_dict(state['optimizer'])
        own_host_tensors(optimizer)  # no view of the mmap'd state outlives the restore
        optimizer.param_groups = optimizer._groups()
        step, best, base_eval, history = state['step'], state.get('best'), state['base_eval'], state['history']
        last_gate = state.get('last_gate')
        restore_training_rng_state(state['rng'], a.device)
        print(json.dumps({'event': 'resumed', 'step': step, 'code_changed': sorted(
            k for k in code if state['identity']['code'].get(k) != code[k])}), flush=True)
        del state

    def full_state():
        from .backbone_policy import plain_named_tensors as plain
        values = plain(backbone.hf)
        return {'schema': SCHEMA, 'identity': identity, 'step': step,
                'student_parameters': {n: values[n.removeprefix('backbone.')].detach() for n, _ in backbone_named},
                **{key: module.state_dict() for key, module in projections.items()},
                'optimizer': optimizer.state_dict(), 'rng': capture_training_rng_state(a.device),
                'best': best, 'base_eval': base_eval, 'history': history, 'last_gate': last_gate}

    def log(name, value):
        with (a.out / name).open('a') as handle:
            handle.write(json.dumps(value) + '\n')

    gate_by_step = {}  # the full behaviour gate already run at a step (at a warn), reused as the final gate

    def evaluate():
        nonlocal base_eval
        sums, count, agreement = {}, 0, []
        predictions = []
        with torch.no_grad(), contextlib.ExitStack() as scope:
            if quant is not None:
                scope.enter_context(quant.context('bf16', 0.0))
            for batch in evaluation_batches(held, a.eval_batch):
                _, metrics, prediction, _ = forward(batch, gradients=False)
                predictions.append(prediction.cpu())
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + value * len(batch)
                count += len(batch)
            report = {'step': step, **{k: v / count for k, v in sums.items()}}
            if base_eval is None:
                base_eval = {'ce': report['ce'], 'gold_accuracy': report['gold_accuracy'],
                             'predictions_sha256': hashlib.sha256(
                                 b''.join(p.numpy().tobytes() for p in predictions)).hexdigest(),
                             'predictions': [p.tolist() for p in predictions]}
            agree = [float((p == torch.tensor(b)).float().mean()) for p, b in zip(predictions, base_eval['predictions'])]
            report.update(base_ce=base_eval['ce'], ce_delta=report['ce'] - base_eval['ce'],
                          base_agreement=sum(agree) / len(agree))
            if gate_held is not None and a.drift_rows:
                report['behaviour_drift'] = drift_check()
        log('eval.jsonl', report)
        return report

    def drift_check():
        """The drift monitor: held KL on --drift-rows, its guard level, and at a warn the behaviour gate (stop below
        its reference minus max_gate_drop) -- at most every --drift-gate-every updates, at once when response KL
        reaches --drift-gate-now, and at the stage end."""
        nonlocal last_gate
        with torch.no_grad(), contextlib.ExitStack() as scope:
            if quant is not None:
                scope.enter_context(quant.context('bf16', 0.0))
            drift = held_kl(backbone.hf, gate_held, a.drift_rows, 0.25)
        drift['guard'] = behaviour_drift_level(drift, a)
        due = (last_gate is None or step - last_gate >= a.drift_gate_every or step >= a.steps or
               drift['held_response_kl'] >= a.drift_gate_now)
        if drift['guard'] == 'warn' and a.drift_gate and behaviour is not None and not due:
            drift['gate_at_warn'] = {'deferred': True, 'last_gate': last_gate}
        elif drift['guard'] == 'warn' and a.drift_gate and behaviour is not None:
            last_gate = step
            gate = gate_by_step[step] = gate_report()
            floor = behaviour.config['reference']['gate_pass'] - behaviour.config.get('max_gate_drop', 0.0)
            drift['gate_at_warn'] = {k: gate[k] for k in ('gate_pass', 'gate_checks', 'gate_kinds', 'gate_results')}
            drift['gate_floor'] = floor
            if gate['gate_pass'] < floor - 1e-9:
                drift['guard'] = 'stop'
                drift['stop_reason'] = 'behaviour gate below its reference minus max_gate_drop'
        elif drift['guard'] == 'stop':
            drift['stop_reason'] = 'response KL or agreement past the hard stop'
        return drift

    def gate_report():
        if behaviour is None:
            return None
        with torch.no_grad(), contextlib.ExitStack() as scope:
            if quant is not None:
                scope.enter_context(quant.context('bf16', 0.0))
            result = behaviour_report(backbone.hf, engine.tokenizer, behaviour.probes, render, behaviour.eos, gate_held,
                                      a.gate_held_rows, behaviour.config.get('prompt_weight', 0.25))
        result['passed'] = behaviour.passed(result)
        return result

    # Disk: hold the blocks of the next full-state write (its tensors, 5% margin; at least the last write's size) so a
    # write never meets a full disk. The reserve is released just for a write and taken back after it; a run that
    # cannot hold it refuses to start, and one that cannot take it back after a write stops on that complete state.
    from .checkpoint_safety import CheckpointDiskReserve, CheckpointReserveError, _state_tensor_bytes
    reserve = None

    def next_write_bytes():
        written = policy.slot.stat().st_size if policy.slot.exists() else 0
        return max(written, int(1.05 * _state_tensor_bytes(full_state())))

    if a.save_state:
        reserve = CheckpointDiskReserve(a.out / '.checkpoint-space.reserve', next_write_bytes())
        try:
            reserve.acquire()
        except CheckpointReserveError as error:
            print(json.dumps({'event': 'checkpoint_space_preflight_refused', 'step': step, 'error': str(error)}),
                  flush=True)
            reserve.cleanup()
            raise SystemExit(2)
        atexit.register(reserve.cleanup)
        save_full_state = policy.save

        def reserved_save(state, **options):
            reserve.release_space()
            event = save_full_state(state, **options)
            try:
                reserve.acquire()
                reserve.grow_to(next_write_bytes())
            except CheckpointReserveError as error:
                print(json.dumps({'event': 'checkpoint_space_after_write_refused', 'step': state.get('step'),
                                  'error': str(error)}), flush=True)
                stop.request()  # the state just written is complete; no further write is attempted
            return event
        policy.save = reserved_save

    # Step 0: the NeuraleseProjection is exactly the base soft-token mixture on a held batch.
    if step == 0:
        with torch.no_grad():
            ids, prefix, span, roles, _ = tensors(held[:1])
            hidden = backbone.forward_ids(ids[:, :-1], cutoff=engine.heads.cutoff, logits=False)['h_final']
            hidden = hidden[:, prefix - 1:]
            mixture = soft_token_mixture(backbone.logits, table, hidden)
            if not torch.equal(neuralese(hidden, mixture.to(hidden.dtype)), mixture.to(hidden.dtype)):
                raise RuntimeError('NeuraleseProjection is not the soft-token mixture at initialization')
            token_choice_identity = token_choice is None or torch.equal(token_choice(hidden), hidden)
            if not token_choice_identity:
                raise RuntimeError('TokenChoiceProjection is not the identity at initialization')
        # The lineage's identity qualification (recipe identity_qualification, direct certificate).
        (a.out / 'identity-check.json').write_text(json.dumps({
            'passed': True, 'step': 0, 'neuralese_projection_equals_soft_token_mixture': True,
            'token_choice_projection': 'absent' if token_choice is None else 'identity',
            'logits_equal_base': True}, indent=2) + '\n')
        evaluate()
    order = random.Random(f'{a.seed}:direct')
    for _ in range(step):  # the sampler is keyed by step: replay its draws
        order.randrange(len(train_windows))
    loop = TrainingLoop(step, a.steps, stop)
    start_index = step
    saved_step = None
    started = time.time()
    for step_index in loop:
        first = train_windows[order.randrange(len(train_windows))]
        same = [w for w in train_windows if w['prefix'] == first['prefix'] and len(w['ids']) == len(first['ids'])]
        batch = [first] + [same[order.randrange(len(same))] for _ in range(a.batch - 1)]
        point, mix = ('bf16', 0.0) if quant is None else quant.sampled(step_index)[:2]
        tick = time.time()
        with contextlib.ExitStack() as scope:
            if quant is not None:
                scope.enter_context(quant.context(point, mix))
            with shared_parametrized_weights(backbone.hf) as next_pass:
                next_pass()
                with in_backward.in_backward() as armed:
                    loss, metrics, _, _ = forward(batch, gradients=True)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite neuralese_direct loss')
                    loss.backward()
                armed.norm()
        optimizer.step()  # routers, norms and projections (the latents stepped in backward)
        optimizer.zero_grad(set_to_none=True)
        step = step_index + 1
        if reserve is not None and step_index == start_index:  # optimizer slots exist now: size the next write
            try:
                reserve.grow_to(next_write_bytes())
            except CheckpointReserveError as error:
                print(json.dumps({'event': 'checkpoint_space_preflight_refused', 'step': step, 'error': str(error)}),
                      flush=True)
                raise SystemExit(2)
        log('train.jsonl', {'step': step, 'loss': float(loss), 'precision': point, 'mix': mix, **metrics,
                            'seconds': time.time() - tick})
        if step % a.eval_every == 0 or step == a.steps:
            report = evaluate()
            history.append({k: report[k] for k in ('step', 'ce', 'ce_delta', 'value_relative_mse_delta')})
            improved = best is None or report['ce'] < best['ce']
            if (report.get('behaviour_drift') or {}).get('guard') == 'stop':
                report['stopped_by'] = 'behaviour_drift'
                print(json.dumps({'event': 'behaviour_drift_stop', 'step': step}), flush=True)
                stop.request()
            if improved and not stop.requested:
                best = {'step': step, 'ce': report['ce']}
                policy.save(full_state(), metric=report['ce'])
                saved_step = step
            (a.out / 'report.json').write_text(json.dumps({**report, 'status': 'running'}, indent=2) + '\n')
        elif a.drift_every and step % a.drift_every == 0 and gate_held is not None and a.drift_rows:
            drift = drift_check()
            log('drift.jsonl', {'step': step, 'behaviour_drift': drift})
            if drift['guard'] == 'stop':
                print(json.dumps({'event': 'behaviour_drift_stop', 'step': step}), flush=True)
                stop.request()
        if (policy.due(step) or stop.requested) and saved_step != step:  # never a second write of one step:
            policy.save(full_state())  # it would replace the slot and break best-checkpoint.pt's hard link
            saved_step = step
    finished = step >= a.steps
    gate = (gate_by_step.get(step) or gate_report()) if finished else None
    final = json.loads((a.out / 'report.json').read_text()) if (a.out / 'report.json').exists() else {}
    drift_stopped = any((json.loads(line).get('behaviour_drift') or {}).get('guard') == 'stop'
                        for name in ('eval.jsonl', 'drift.jsonl') if (a.out / name).is_file()
                        for line in (a.out / name).open())
    value_delta = final.get('value_relative_mse_delta', math.inf)
    value_qualified = math.isfinite(value_delta) and value_delta < -a.value_margin
    # Owner 2026-10-11: the Neuralese value requirement belongs to the end-to-end stage (consumers reading values
    # against consumers reading tokens); here the value metric and its margin are report-only.
    qualified = bool(finished and final.get('ce_delta', 0.0) < 0 and not drift_stopped and
                     (gate is None or gate['passed']))
    identity_file = a.out / 'identity-check.json'
    final.update(identity_check=json.loads(identity_file.read_text()) if identity_file.is_file() else None)
    status = 'complete' if finished else ('checkpointed_on_drift_guard' if drift_stopped else 'checkpointed_on_signal')
    final.update(status=status, completed=finished, steps=step, best=best,
                 behaviour_gate=gate, drift_stopped=drift_stopped, value_qualified=value_qualified, qualified=qualified,
                 value_gate={'margin': a.value_margin, 'delta': final.get('value_relative_mse_delta'),
                             'report_only': True, 'moved_to': 'end-to-end stage (owner 2026-10-11)'},
                 seconds=time.time() - started, scope='Neuralese direct v1 (plans/neuralese/NEURALESE_DIRECT_V1.md)')
    if saved_step != step:
        policy.save(full_state())
    (a.out / 'report.json').write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps({'event': 'neuralese_direct_' + final['status'], 'step': step, 'qualified': qualified}), flush=True)


if __name__ == '__main__':
    main()
