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
on a stop and at the end; a restart resumes from it. The default ``--window-execution auto`` accumulates gradients
sequentially across independent windows and uses shared activation geometry plus live CUDA headroom to choose whether
to checkpoint layer activations. The packed path remains an explicit diagnostic override.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from array import array
from pathlib import Path

import torch

from .loop import StopSignal, TrainingLoop, capture_training_rng_state, restore_training_rng_state
from ..common.hashing import canonical_json_sha256_hex
from ..common.profiling import UpdateProfiler, backward_phase, region as profile_region, scope as profile_scope
from .checkpoint_policy import CheckpointPolicy

SCHEMA = 'natlang.neuralese-direct/1'
DIRECT_MEMORY_GEOMETRY_VERSION = 'direct-sft-window-v2:fp32-rmsnorm-casts'
# This exact package computed the v2 FP32-RMSNorm geometry but its resume paths
# mislabeled the stored estimator state as v1. Re-label only those checkpoints;
# all other version mismatches reset calibration through AdaptiveGraphMemory.
_V2_GEOMETRY_V1_LABEL_PACKAGE = {
    'train/direct_sft.py': 'b3b38c788879ccde63c28cbc2f3b32144f890a33d6f538aa3a2a88979868a7b0',
    'train/memory_estimator.py': '3b21423ac9c2a1231ee5cb5744e633bb280d12450e37f76b53455435f554cdb8',
}
from .optim import MUON_MIN_SIDE  # noqa: F401  (MuonSR latents: both matrix sides at least this)
# Options added after runs had started, at their defaults, stay out of the resume identity (a run started before them
# resumes; idle-expert skipping is a defect fix that applies to it from then on).
LATER_DEFAULTS = {'latent_skip_idle': True, 'latent_cautious': False, 'latent_beta2_half_life_tokens': 4e6,
                  'latent_ns': 'quintic', 'latent_weight_decay': 0.0, 'grad_spike_clip': 0.0,
                  'epochs': None, 'batch_tokens': 0, 'continue_from': None,
                  'fp8_training': 'off', 'fp8_recipe': 'delayed', 'fp8_amax_history_len': 16,
                  'gradient_audit_steps': 0, 'saved_tensor_audit_steps': 0,
                  'gradient_digest_steps': 0, 'state_hash_steps': 0, 'sdpa_backend': 'default'}
OPERATIONAL = frozenset({'out', 'weight_drift_base', 'stop_at', 'steps', 'checkpoint_every', 'eval_every', 'drift_every', 'drift_gate_every', 'drift_gate_rise', 'drift_ce_windows', 'gate_response_kl', 'device', 'value_margin', 'save_state', 'moe_kernel', 'checkpoint_layers', 'drift_rows', 'drift_warn',
                         'drift_stop', 'drift_agreement_warn', 'drift_agreement_stop', 'drift_gate', 'eval_batch',
                         'gate_held_rows', 'window_execution', 'continue_from', 'continue_identity',
                         'continue_monitor_reference', 'gradient_digest_steps', 'state_hash_steps',
                         'continued_text_data', 'plan_only', 'plan_out'})


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('heads', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('records', 'pieces', 'text-data'):  # the data: prepared --text-data, or --records (+ --pieces)
        p.add_argument('--' + name, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, help='legacy budget in updates (default 1000); a recipe declares --epochs')
    p.add_argument('--epochs', type=float, help='the stage budget: passes over the target-bearing train windows, '
                   'sampled by document without replacement (text_supervision.epoch_schedule); the update count is '
                   'derived from the data and the packing')
    p.add_argument('--batch-tokens', type=int, default=0, help='fit-only: pack each update with windows up to this '
                   'many forward tokens (count x longest, right-padded, length-bucketed); 0: --batch windows per update')
    p.add_argument('--continue-from', type=Path, help="a predecessor neuralese_direct stage's full state (its "
                   'checkpoint.pt): weights, projections and optimizer continue on this stage\'s data from update 0')
    p.add_argument('--continue-identity', type=Path,
                   help='the predecessor stage identity-check.json, pinned separately from a relocated checkpoint')
    p.add_argument('--continue-monitor-reference', type=Path,
                   help='the predecessor stage monitor-reference.pt, pinned separately from a relocated checkpoint')
    p.add_argument('--continued-text-data', type=Path,
                   help='data continuation (owner 2026-10-12): the --text-data this stage trained on before; a resume '
                        'whose checkpoint read it continues on --text-data with the same update counter, weights and '
                        'optimizer state, the seen documents counted as seen (data_continuation)')
    p.add_argument('--plan-only', action='store_true',
                   help='print the data plan (budget, data continuation, QAT range, held set) and exit: no model, no '
                        'writes to --out')
    p.add_argument('--plan-out', type=Path, help='--plan-only: also write the plan and continuation record here')
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
    p.add_argument('--latent-optimizer', choices=('lionsr', 'muonsr', 'adaminisr'), default='lionsr')
    p.add_argument('--latent-skip-idle', action=argparse.BooleanOptionalAction, default=True,
                   help='lionsr/adaminisr: an expert matrix with an exactly zero gradient this update (no routed tokens) '
                        'is not stepped (train/optim.py LionSR)')
    p.add_argument('--latent-cautious', action=argparse.BooleanOptionalAction, default=False,
                   help='lionsr/adaminisr: the cautious mask (a coordinate moves only where the update agrees in sign '
                        'with its gradient), without lr renormalization')
    p.add_argument('--latent-beta2-half-life-tokens', type=float, default=4e6,
                   help="adaminisr: the second moment's half-life in tokens (beta2 from tokens x batch per update; "
                        'Marek et al., arXiv:2507.07101)')
    p.add_argument('--latent-ns', choices=('quintic', 'cubic5'), default='quintic',
                   help="muonsr: the Newton-Schulz variant (cubic5: arXiv:2606.00371, a third fewer matmuls)")
    p.add_argument('--latent-weight-decay', type=float, default=0.0,
                   help='decoupled weight decay of the latents (Mellum2 SFT: 0.1 with Muon)')
    p.add_argument('--grad-spike-clip', type=float, default=0.0,
                   help='spike-only global gradient clip (train/optim.py SpikeClip): an update whose global norm exceeds '
                        'this many times the running median of recent norms is clipped to that threshold (the latents, '
                        'stepping in backward, by the lower bound of the norm so far); every update logs the applied '
                        'and the exact latent norm (grad_clip in train.jsonl). 0: off')
    p.add_argument('--stop-at', type=int, default=0,
                   help='operational: end the stage at this update (eval, gate and report as at the end) while the '
                        'schedules (QAT ramps) follow --steps: a prefix of the declared run; 0: --steps')
    p.add_argument('--weight-drift-base', type=Path,
                   help='report-only: a directory holding the base latents (written before the first update when '
                        'absent); at the end weight-drift.json records each latent\'s drift from it, per expert for '
                        'stacked latents with its idle-update count')
    p.add_argument('--latent-lr', type=float, default=3e-6,
                   help='lionsr: per row ternary scale; muonsr: on AdamW\'s scale')
    p.add_argument('--head-lr', type=float, default=3e-4, help='the projections\' lr (Muon/AdamW)')
    p.add_argument('--neuralese-weight', type=float, default=1.0)
    p.add_argument('--projection-blocks', type=int, default=2)
    p.add_argument('--projection-hidden', type=int, default=0, help='0: width/4')
    p.add_argument('--projection-kernel', type=int, default=4)
    p.add_argument('--token-choice-projection', action=argparse.BooleanOptionalAction, default=False,
                   help='declare the TokenChoiceProjection (the end-to-end Neuralese stage); off in direct v1')
    p.add_argument('--moe-kernel', choices=('invariant', 'loop', 'grouped'), default='invariant')
    p.add_argument('--checkpoint-layers', action=argparse.BooleanOptionalAction, default=None,
                   help='recompute each layer in backward (long windows)')
    p.add_argument('--window-execution', choices=('auto', 'packed', 'sequential'), default='auto',
                   help='auto accumulates independent windows sequentially and chooses checkpointing from shared '
                        'activation geometry plus live CUDA headroom; packed is a diagnostic override')
    p.add_argument('--quantization')
    p.add_argument('--fp8-training', choices=('off', 'te-fp8', 'triton-experts'), default='off',
                   help='FP8 backend (train/fp8_backend.py): te-fp8 = TE dense linears (LFM); triton-experts = '
                        'Triton grouped FP8 expert GEMMs (Mellum, composes with QAT); off = BF16')
    p.add_argument('--fp8-recipe', choices=('delayed', 'current'), default='delayed',
                   help='Transformer Engine scaling recipe when --fp8-training=te-fp8')
    p.add_argument('--fp8-amax-history-len', type=int, default=16,
                   help='FP8 DelayedScaling history length; ignored by CurrentScaling')
    p.add_argument('--gradient-audit-steps', type=int, default=0,
                   help='diagnostic: write per-parameter gradient summaries for this many initial updates; requires '
                        'window_execution=sequential and at least two windows per update')
    p.add_argument('--saved-tensor-audit-steps', type=int, default=0,
                   help='diagnostic: inventory autograd saved tensors for this many initial updates')
    p.add_argument('--gradient-digest-steps', type=int, default=0,
                   help='diagnostic: SHA-256 each named gradient tensor for this many initial updates')
    p.add_argument('--state-hash-steps', type=int, default=0,
                   help='diagnostic: save exact named parameter/optimizer/RNG/FP8 state hashes around initial updates')
    p.add_argument('--sdpa-backend', choices=('default', 'math'), default='default',
                   help='diagnostic attention backend; math forces PyTorch SDPBackend.MATH for direct-SFT forwards')
    from .drift_monitor import add_drift_arguments
    # The drift guard (owner 2026-10-11; train/drift_monitor.py, shared with the end-to-end stage): response KL or
    # agreement past warn runs the behaviour gate and stops when the gate falls below its reference minus
    # max_gate_drop; response KL past --drift-stop stops at once.
    add_drift_arguments(p)
    p.add_argument('--drift-ce-windows', type=int, default=4,
                   help='held CE on this many held windows at every drift check (where held CE bottoms out); 0: off')
    p.add_argument('--save-state', action=argparse.BooleanOptionalAction, default=True,
                   help='write the full training state (weights, optimizer); smokes pass --no-save-state and keep '
                        'only their logs and report')
    p.add_argument('--value-margin', type=float, default=0.0,
                   help='report-only: value_qualified says whether the final held value_relative_mse_delta is below '
                        'minus this margin (the value requirement is the end-to-end stage\'s, owner 2026-10-11)')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    if a.epochs is not None and a.steps is not None:
        p.error('declare the budget as --epochs or --steps, not both')
    if a.epochs is None and a.steps is None:
        a.steps = 1000
    if a.epochs is not None and not (math.isfinite(a.epochs) and a.epochs > 0) or a.batch_tokens < 0:
        p.error('--epochs must be positive and --batch-tokens nonnegative')
    if min(a.steps or 1, a.tokens, a.target_tokens, a.batch, a.eval_batch, a.eval_every, a.held_documents) < 1:
        p.error('positive bounds required')
    if a.checkpoint_every < 0 or a.neuralese_weight < 0:
        p.error('invalid checkpoint cadence or Neuralese weight')
    if min(a.gradient_audit_steps, a.saved_tensor_audit_steps, a.gradient_digest_steps, a.state_hash_steps) < 0:
        p.error('diagnostic audit step counts must be nonnegative')
    if a.gradient_digest_steps and a.window_execution == 'packed':
        p.error('--gradient-digest-steps requires sequential windows so in-backward hooks do not consume gradients')
    if a.text_data is None and a.records is None:
        p.error('give the stage its data: --text-data, or --records with --pieces')
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


def snapshot_latents(directory: Path, named) -> None:
    """The base latents, one file each, for ``latent_drift`` (written before the first update)."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, (name, q) in enumerate(named):
        torch.save(q.detach().cpu(), directory / f'{index:04d}.pt')
    (directory / 'manifest.json').write_text(json.dumps([[n, list(q.shape)] for n, q in named]) + '\n')


@torch.no_grad()
def latent_drift(directory: Path, named, optimizer, step: int) -> dict:
    """Report-only: each latent's drift ``||w - w0||`` from the snapshot and ``||w0||``, per matrix for stacked (expert)
    latents with the optimizer's count of updates in which that matrix was idle (no routed tokens)."""
    manifest = json.loads((directory / 'manifest.json').read_text())
    if [n for n, _ in manifest] != [n for n, _ in named]:
        raise ValueError('weight drift snapshot holds other latents')
    rows, total, reference = [], 0.0, 0.0
    for index, (name, q) in enumerate(named):
        base = torch.load(directory / f'{index:04d}.pt', mmap=True, map_location='cpu').to(q.device)
        delta = q.float() - base.float()
        dims = (-2, -1)
        row = {'name': name, 'shape': list(q.shape)}
        if q.ndim > 2:
            drift = torch.linalg.vector_norm(delta, dim=dims).reshape(-1)
            norm = torch.linalg.vector_norm(base.float(), dim=dims).reshape(-1)
            idle = (getattr(optimizer, 'idle_updates', None) or {}).get(id(q))
            row.update(expert_drift=drift.tolist(), expert_norm=norm.tolist(),
                       expert_idle_updates=None if idle is None else idle.reshape(-1).tolist(),
                       updates=(getattr(optimizer, 'updates', None) or {}).get(id(q)))
            drift, norm = drift.square().sum().sqrt(), norm.square().sum().sqrt()
        else:
            drift, norm = torch.linalg.vector_norm(delta), torch.linalg.vector_norm(base.float())
        row.update(drift=float(drift), norm=float(norm))
        total, reference = total + float(drift) ** 2, reference + float(norm) ** 2
        rows.append(row)
        del base, delta
    return {'schema': 'natlang.latent-weight-drift/1', 'step': step, 'drift': total ** .5, 'norm': reference ** .5,
            'relative': (total / reference) ** .5 if reference else None, 'latents': rows}


def archive_predecessor_state(slot: Path) -> dict | None:
    """After a phase handoff has loaded its predecessor's full state, move that state off the NVMe (coordinator
    2026-10-12: a lineage keeps one full state on the fast disk). The shared mover (scripts/archive_to_external.py
    ``archive_file``) copies it to the HDD archive, verifies size and SHA-256, records it in the archive manifest and
    replaces the file with a symlink, so ``--continue-from`` (a restart before this phase's first save) still resolves.
    The predecessor's ``best-checkpoint.pt`` hard link to the same file is relinked to the copy, so the NVMe blocks are
    freed. A receipt ``<slot>.archived.json`` names the copy and its hash. Already a symlink: nothing to do."""
    import importlib.util
    import os
    if slot.is_symlink() or not slot.is_file():
        return None
    from ..common.paths import root as path_root
    script = Path(os.environ.get('NATLANG_REPO') or path_root('repo')) / 'scripts' / 'archive_to_external.py'
    spec = importlib.util.spec_from_file_location('archive_to_external', script)
    mover = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mover)
    real = slot.resolve()
    roots = [r.resolve() for r in mover.ROOTS]
    if not any(root in real.parents for root in roots):
        print(json.dumps({'event': 'predecessor_state_archive_skipped', 'path': str(slot),
                          'resolved_path': str(real), 'reason': 'outside_configured_archive_roots'}), flush=True)
        return None
    dest = mover.destination(real, roots, mover.ARCHIVE)
    inode = slot.stat().st_ino
    best = slot.with_name('best-checkpoint.pt')
    record = mover.archive_file(slot, dest, mover.ARCHIVE / mover.MANIFEST_NAME)
    if best.is_file() and not best.is_symlink() and best.stat().st_ino == inode:
        link = best.with_name(best.name + '.archive-link')
        os.symlink(dest, link)
        os.replace(link, best)
        record['also_relinked'] = str(best)
    slot.with_name(slot.name + '.archived.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'event': 'predecessor_state_archived', **record}), flush=True)
    return record


class PredecessorArchive:
    """``archive_predecessor_state`` off the critical path (owner 2026-10-12: the synchronous 47 GB copy and hash to
    the HDD left the GPU idle for 20+ minutes at every phase hand-off). A thread copies, verifies and links while the
    stage trains; the mover never touches the source before its verified copy exists. A failure is logged and leaves
    the predecessor where it was."""

    def __init__(self, slot: Path):
        import threading
        self.slot, self.error = slot, None
        self.thread = threading.Thread(target=self._run, name='predecessor-archive', daemon=False)
        self.thread.start()
        print(json.dumps({'event': 'predecessor_state_archive_started', 'path': str(slot)}), flush=True)

    def _run(self):
        try:
            archive_predecessor_state(self.slot)
        except Exception as error:  # the stage keeps training; the predecessor stays on the fast disk
            self.error = repr(error)
            print(json.dumps({'event': 'predecessor_state_archive_failed', 'path': str(self.slot),
                              'error': self.error}), flush=True)

    @property
    def running(self) -> bool:
        return self.thread.is_alive()

    def wait(self) -> None:
        self.thread.join()


INPUT_PATH_OPTIONS = ('heads', 'records', 'pieces', 'text_data')


def same_resume_identity(saved: dict, current: dict) -> bool:
    """A resume continues the same training: the same semantic options and the same input contents. Content, not
    paths: an input read from a staged NVMe copy, the HDD archive or a symlink is the same input when its SHA-256 is
    (v4 2026-10-12: the archived predecessor's path changed between launches). Diagnostic environment (profiling) is
    never part of the identity."""
    # The stateless FP8 expert backend (current per-row scaling) is an execution choice, like --moe-kernel: a run
    # switches it on or off at a resume (owner 2026-10-12: v4 onto FP8 experts at a stop/resume).
    strip = lambda options: {k: v for k, v in options.items() if k not in INPUT_PATH_OPTIONS
                             and not (k == 'fp8_training' and v in ('off', 'triton-experts'))}
    return (strip(saved['options']) == strip(current['options'])
            and sorted(saved['inputs'].values()) == sorted(current['inputs'].values()))


def direct_sft_calibration_kind(fp8_training: str, fp8_recipe: str, mode: str,
                                checkpointed: bool) -> str:
    """Keep activation calibration isolated by execution backend and geometry."""
    suffix = f'{mode}-checkpointed-{int(checkpointed)}'
    if fp8_training == 'off':
        return f'direct-sft-{suffix}'
    if fp8_training == 'te-fp8':
        return f'direct-sft-fp8-{fp8_recipe}-{suffix}'
    if fp8_training == 'triton-experts':
        return f'direct-sft-fp8-experts-{suffix}'
    raise ValueError(f'unknown direct-SFT FP8 backend {fp8_training!r}')


CONTINUATION_SCHEMA = 'natlang.neuralese-direct-data-continuation/1'


def target_bearing(window):
    """A window whose span holds decision-target tokens, or assistant positions in a document without a declared
    suffix (data audit 2026-10-11, DATA_INVENTORY_20261011.md E); context-only windows are not trained."""
    from .text_supervision import ROLE_CODES
    size = len(window['ids']) - window['prefix']
    if window.get('supervised_suffix_start') is not None:
        return window['supervised_suffix_start'] < size
    roles = window.get('roles')
    assistant = (ROLE_CODES.index('assistant_reasoning'), ROLE_CODES.index('assistant_reply'))
    return roles is None or any(roles[i] in assistant for i in range(window['prefix'], len(window['ids'])))


def direct_sampler(targets):
    """The shared cohort-then-(sampling group-)then-document-then-item sampler, equal cohort weights."""
    from .text_supervision import CohortDocumentSampler
    cohorts = sorted({w.get('cohort', 'native') for w in targets})
    grouped = any(w.get('sampling_group') is not None for w in targets)
    return CohortDocumentSampler(targets, {c: 1 / len(cohorts) for c in cohorts},
                                 cohort_key=lambda w: w.get('cohort', 'native'), document_key=lambda w: w['document'],
                                 sampling_group_key=(lambda w: w.get('sampling_group')) if grouped else None)


def _window_length(window):
    return len(window['ids']) - 1


def continued_unseen(targets, continuation):
    """The target windows of documents a data continuation has not seen (by content hash or document id)."""
    hashes, ids = set(continuation['seen_document_sha256']), set(continuation['seen_document_ids'])
    return [w for w in targets if w['document'] not in hashes and w.get('document_id') not in ids]


def direct_schedule(targets, a, continuation=None):
    """The stage's whole update plan, a pure function of the data, the seed, the packing and the continuation.

    Without a continuation: ``text_supervision.epoch_schedule`` over the target-bearing windows. After a data
    continuation (``--continued-text-data``) at update k: k placeholder entries (the updates trained on the previous
    data), then the rest of the first pass on this data's unseen documents (same sampler policy, same seed and
    packing), then the remaining whole passes over all of this data. ``epochs_covered`` counts seen windows plus this
    data's windows trained so far, over seen plus unseen windows. Returns (sampler, plan)."""
    from .text_supervision import epoch_schedule
    sampler = direct_sampler(targets)
    pack = dict(seed=a.seed, length=_window_length, max_tokens=a.batch_tokens, max_items=a.batch)
    if continuation is None:
        return sampler, epoch_schedule(sampler, epochs=a.epochs, updates=None if a.epochs is not None else a.steps,
                                       **pack)
    if a.epochs is None or not float(a.epochs).is_integer():
        raise ValueError('a data continuation continues a whole-epoch budget (--epochs 1, 2, ...)')
    consumed, seen = continuation['consumed_updates'], continuation['seen_windows']
    unseen = continued_unseen(targets, continuation)
    total = seen + len(unseen)
    plan = [([], seen / total * (i + 1) / consumed) for i in range(consumed)]
    if unseen:
        plan += [(batch, (seen + covered * len(unseen)) / total)
                 for batch, covered in epoch_schedule(direct_sampler(unseen), epochs=1.0, **pack)]
    if a.epochs > 1:
        plan += [(batch, 1.0 + covered) for batch, covered in epoch_schedule(sampler, epochs=a.epochs - 1, **pack)]
    return sampler, plan


def held_documents(held):
    return sorted({w['document'] for w in held})


def data_continuation(previous_targets, previous_held, targets, held, *, consumed, previous, quantization_stage, a):
    """Continue the stage's epoch budget on new data after ``consumed`` updates (owner 2026-10-12: "continue our
    training from what we have with more data"). The previous plan is rebuilt exactly (from the previous data and its
    own continuation, if any); the documents of its consumed updates count as seen, and the rest of the epoch is this
    data minus the seen documents, matched by content hash or document id (a partly consumed document counts as
    seen). The update counter, weights and optimizer state continue. The QAT progress is rebased so it continues from
    its value at update k to the stage's end over the new total: progress(t) = p_k + (p1 - p_k) (t - k) / (N - k).
    Learning rates are constant in this trainer (no schedule, no warm-up), so nothing else is rebased. Returns
    (record, plan)."""
    _, previous_plan = direct_schedule(previous_targets, a, previous)
    offset = previous['consumed_updates'] if previous else 0
    if not offset <= consumed <= len(previous_plan):
        raise ValueError(f'data continuation at update {consumed} lies outside the previous plan '
                         f'({offset}..{len(previous_plan)})')
    if consumed and previous_plan[consumed - 1][1] > 1:
        raise ValueError('a data continuation must start inside the first pass of the previous data')
    batches = [batch for batch, _ in previous_plan[offset:consumed]]
    hashes = set(previous['seen_document_sha256'] if previous else ())
    ids = set(previous['seen_document_ids'] if previous else ())
    hashes |= {w['document'] for batch in batches for w in batch}
    ids |= {w['document_id'] for batch in batches for w in batch if w.get('document_id') is not None}
    seen_windows = (previous['seen_windows'] if previous else 0) + sum(len(batch) for batch in batches)
    p0, p1 = quantization_stage['progress'] if quantization_stage else (0.0, 0.0)
    start = quantization_stage['start'] if quantization_stage else 0
    previous_end = len(previous_plan)
    progress_k = p0 + (p1 - p0) * min(1.0, max(0.0, (consumed - start) / max(1, previous_end - start)))
    consumed_documents = [[w['document'] for w in batch] for batch in batches]
    record = {'schema': CONTINUATION_SCHEMA, 'consumed_updates': consumed, 'previous_updates': previous_end,
              # The rebuilt consumed updates (their documents, in order): checkable against the run's execution_plan
              # log events.
              'consumed_documents_sha256': hashlib.sha256(json.dumps(consumed_documents).encode()).hexdigest(),
              'previous_epochs_covered': previous_plan[consumed - 1][1] if consumed else 0.0,
              'seen_documents': len(hashes), 'seen_windows': seen_windows,
              'seen_document_sha256': sorted(hashes), 'seen_document_ids': sorted(ids),
              'previous': ({k: v for k, v in previous.items() if not k.startswith('seen_document_')}
                           if previous else None)}
    _, plan = direct_schedule(targets, a, record)
    unseen = continued_unseen(targets, record)
    record.update(updates=len(plan), plan_documents_sha256=plan_documents_sha256(plan, consumed),
                  unseen_target_windows=len(unseen),
                  unseen_documents=len({w['document'] for w in unseen}),
                  matched_seen_documents=len({w['document'] for w in targets} - {w['document'] for w in unseen}),
                  lr_schedule='constant (direct_sft has no LR schedule or warm-up): unchanged',
                  quantization_rebase=None if quantization_stage is None else {
                      'previous': {'start': start, 'end': previous_end, 'progress': [p0, p1]},
                      'start': consumed, 'end': len(plan), 'progress': [progress_k, p1]},
                  held={'previous_documents': held_documents(previous_held), 'documents': held_documents(held),
                        'rebased': held_documents(previous_held) != held_documents(held)})
    return record, plan


def continuation_summary(record):
    """The record without its seen-document lists (stage report, plan, logs)."""
    return None if record is None else {k: v for k, v in record.items() if not k.startswith('seen_document_')}


def text_engine(heads):
    """The tokenizer side of serve.load_engine (no weights): what load_text_rows and prepare_text_windows read, for
    a data-only plan (--plan-only)."""
    from types import SimpleNamespace
    from transformers import AutoConfig, AutoTokenizer
    from ..model.hf_port import family_controls
    from ..model.lfm2_port import ControlTokens, resolve_base
    saved = dict((torch.load(heads, map_location='cpu', weights_only=False, mmap=True) or {}).get('backbone') or {})
    if saved.get('backbone') == 'maple':
        tokenizer = AutoTokenizer.from_pretrained(saved['base'], trust_remote_code=False)
        controls = family_controls(SimpleNamespace(config=AutoConfig.from_pretrained(saved['base'])), tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(saved.get('student_lora') or resolve_base(saved.get('base')))
        controls = ControlTokens.from_tokenizer(tokenizer)
    import functools
    tokens = functools.lru_cache(maxsize=64)(lambda text: tuple(tokenizer(text, add_special_tokens=False)['input_ids']))
    return SimpleNamespace(tokenizer=tokenizer, backbone=SimpleNamespace(controls=controls),
                           _tokens=lambda text: list(tokens(text)))


def stage_data(engine, a, text_data):
    """(train windows, target-bearing windows, held windows, held selection, mask receipt, rows) of one data file."""
    from .text_warmup import load_text_rows, prepare_text_windows, select_held_document_windows
    rows, _ = load_text_rows(a.records, a.pieces, text_data, tokenizer=engine.tokenizer)
    windows, mask_receipt = prepare_text_windows(engine, rows, tokens=a.tokens, prefix_tokens=a.prefix_tokens,
                                                 target_tokens=a.target_tokens, mask_system_prompt=True)
    held, held_selection = select_held_document_windows(windows['test'], a.held_documents)
    targets = [w for w in windows['train'] if target_bearing(w)]
    return windows['train'], targets, held, held_selection, mask_receipt, rows


def stream_text_rows(text_data, tokenizer):
    """``text_warmup.load_text_rows(text_data=...)`` as a stream, for a data-only plan: the same validation, train/held
    source-group check, exclusion of train rows that duplicate a held text and exact-duplicate dedup (first row
    kept), with content hashes in place of texts; two passes over the file, one row in memory at a time."""
    from ..data.text_corpus import tokenizer_fingerprint
    fingerprint, vocab_size = None, len(tokenizer)
    groups, held_texts = {'train': set(), 'test': set()}, set()
    with Path(text_data).open() as source:
        for line in source:
            row = json.loads(line)
            if (row.get('split') not in ('train', 'test') or not isinstance(row.get('text'), str)
                    or not row['text'].strip() or not row.get('source_groups')):
                raise ValueError('text JSONL needs nonempty text, train/test split and source_groups')
            if 'sampling_group' in row and (not isinstance(row['sampling_group'], str) or not row['sampling_group']):
                raise ValueError('sampling_group must be a nonempty string when present')
            if not isinstance(row.get('text_cohort', 'native'), str) or not row.get('text_cohort', 'native'):
                raise ValueError('text_cohort must be a nonempty string')
            if 'supervised_suffix_start' in row and 'token_ids' not in row:
                raise ValueError('supervised suffix requires native token IDs')
            if 'token_ids' in row:
                ids = row['token_ids']
                fingerprint = fingerprint or tokenizer_fingerprint(tokenizer)
                if (row.get('tokenizer_sha256') != fingerprint or not isinstance(ids, list) or not ids or
                        any(type(i) is not int or i < 0 or i >= vocab_size for i in ids)):
                    raise ValueError('gold token IDs or tokenizer fingerprint mismatch')
                if 'supervised_suffix_start' in row and (type(row['supervised_suffix_start']) is not int or
                                                         not 0 <= row['supervised_suffix_start'] < len(ids)):
                    raise ValueError('gold supervised suffix coordinate is invalid')
            groups[row['split']].update(row['source_groups'])
            if row['split'] == 'test':
                held_texts.add(hashlib.sha256(row['text'].encode()).digest())
    if groups['train'] & groups['test']:
        raise ValueError('text warm-up factual source groups cross train/test')
    seen = {}
    with Path(text_data).open() as source:
        for line in source:
            row = json.loads(line)
            text = hashlib.sha256(row['text'].encode()).digest()
            if row['split'] == 'train' and text in held_texts:
                continue
            if 'token_ids' in row:
                row['token_ids'] = array('I', row['token_ids'])
            key = (row['split'], text, hashlib.sha256(array('I', row.get('token_ids', [])).tobytes()).digest(),
                   row.get('supervised_suffix_start'), row.get('sampling_group'))
            cohort = row.get('text_cohort', 'native')
            if key in seen:
                if seen[key] != cohort:
                    raise ValueError('an identical document belongs to multiple cohorts; resolve ownership before '
                                     'training')
                continue
            seen[key] = cohort
            yield row


def stage_plan_data(engine, a, text_data):
    """The plan-relevant part of ``stage_data`` (target-bearing and held windows as light descriptors: document,
    document id, cohort, sampling group, source groups, offsets and length), streamed row by row (a data-only plan
    holds no text or token arrays). Returns (targets, held, held selection); ``direct_schedule`` over these targets
    equals the one over ``stage_data``'s windows (the trainer checks the plan's hash)."""
    from .text_warmup import prepare_text_windows, select_held_document_windows

    def light(window):
        value = {'ids': range(len(window['ids'])), 'prefix': window['prefix'], 'document': window['document'],
                 'document_id': window.get('document_id'), 'cohort': window.get('cohort', 'native'),
                 'groups': window.get('groups'), 'offset': window.get('offset', 0)}
        if window.get('sampling_group') is not None:
            value['sampling_group'] = window['sampling_group']
        return value
    targets, tests = [], []
    for row in stream_text_rows(text_data, engine.tokenizer):
        windows, _ = prepare_text_windows(engine, [row], tokens=a.tokens, prefix_tokens=a.prefix_tokens,
                                          target_tokens=a.target_tokens, mask_system_prompt=True)
        targets += [light(w) for w in windows['train'] if target_bearing(w)]
        tests += [light(w) for w in windows['test']]
    held, held_selection = select_held_document_windows(tests, a.held_documents)
    return targets, held, held_selection


def plan_documents_sha256(plan, start=0):
    """The documents of each update of ``plan[start:]``, in order, hashed (plan identity across code paths)."""
    return hashlib.sha256(json.dumps([[w['document'] for w in batch] for batch, _ in plan[start:]]).encode()).hexdigest()


def continuation_inputs(a, inputs):
    """The identity inputs the previous data would have had: --text-data's hash replaced by --continued-text-data's."""
    from ..common.hashing import sha256_file_hex_cached
    return {**inputs, str(a.text_data): sha256_file_hex_cached(a.continued_text_data)}


def continuation_mode(a, state, inputs):
    """How the stage continues its saved state (None: a fresh stage): ('none', None), ('resume', record) for a data
    continuation recorded earlier, or ('continue', previous record or None) when the checkpoint trained on
    --continued-text-data and this launch reads --text-data (the record is computed now). Anything else is left to
    the resume identity check, which refuses it."""
    if state is None:
        return 'none', None
    recorded = state.get('data_continuation')
    saved = (state.get('identity') or {}).get('inputs') or {}
    if sorted(saved.values()) == sorted(inputs.values()):
        return ('resume', recorded) if recorded is not None else ('none', None)
    if a.continued_text_data is not None and sorted(saved.values()) == sorted(continuation_inputs(a, inputs).values()):
        return 'continue', recorded
    return 'none', None


def compute_continuation(a, state, engine, targets, held, recorded, quantization_stage):
    """The data continuation record at the saved state's update (data_continuation), from the previous data file."""
    previous_targets, previous_held, _ = stage_plan_data(engine, a, a.continued_text_data)
    if recorded is not None and quantization_stage is not None:
        quantization_stage = {**quantization_stage, **{k: recorded['quantization_rebase'][k]
                                                       for k in ('start', 'progress')}}
    record, _ = data_continuation(previous_targets, previous_held, targets, held, consumed=state['step'],
                                  previous=recorded, quantization_stage=quantization_stage, a=a)
    return record


def identity_inputs(a):
    """The stage's input identity: each data/state file by its cached digest (never a re-read of an unchanged file)."""
    from ..common.hashing import sha256_file_hex_cached
    return {str(path): sha256_file_hex_cached(path)
            for path in (a.heads, a.records, a.pieces, a.text_data, a.continue_from, a.continue_identity,
                         a.continue_monitor_reference) if path}


def plan_only(a):
    """--plan-only: the stage's data plan from its data and saved state, without loading the model or writing to
    --out: the update budget, the data continuation it would make (or has made), the QAT range and the held set.
    With --plan-out, the continuation record (seen lists included) is written there."""
    inputs = identity_inputs(a)
    slot = a.out / 'checkpoint.pt'
    state = CheckpointPolicy.load(slot) if slot.exists() else None
    mode, recorded = continuation_mode(a, state, inputs)
    resumable = state is None or mode != 'none' or sorted(
        ((state.get('identity') or {}).get('inputs') or {}).values()) == sorted(inputs.values())
    quantization_stage = (json.loads(a.quantization).get('stage') if a.quantization else None)
    engine = text_engine(a.heads)
    targets, held, held_selection = stage_plan_data(engine, a, a.text_data)
    record = (compute_continuation(a, state, engine, targets, held, recorded, quantization_stage)
              if mode == 'continue' else recorded)
    _, plan = direct_schedule(targets, a, record)
    step = state['step'] if state is not None else 0
    if quantization_stage is not None:
        quantization_stage = dict(quantization_stage, end=len(plan))
        if record is not None and record.get('quantization_rebase'):
            quantization_stage.update(start=record['quantization_rebase']['start'],
                                      progress=record['quantization_rebase']['progress'])
    summary = {'event': 'direct_plan_only', 'mode': mode, 'resume_accepted': resumable, 'step': step,
               'updates': len(plan), 'remaining_updates': len(plan) - step, 'epochs': a.epochs,
               'batch_tokens': a.batch_tokens, 'target_windows': len(targets),
               'documents': len({w['document'] for w in targets}),
               'epochs_covered_at_step': plan[step - 1][1] if step else 0.0,
               'next_update_documents': [[w['document'] for w in batch] for batch, _ in plan[step:step + 2]],
               'held_documents': held_documents(held), 'held_policy': held_selection.get('policy'),
               'quantization_stage': quantization_stage,
               'lr_schedule': 'constant (no schedule, no warm-up)',
               'data_continuation': continuation_summary(record)}
    print(json.dumps(summary, indent=1), flush=True)
    if a.plan_out is not None:
        pending = a.plan_out.with_name(a.plan_out.name + '.pending')
        pending.write_text(json.dumps({'summary': summary, 'record': record}) + '\n')
        pending.replace(a.plan_out)
    return 0 if resumable else 2


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    a = parse(raw_argv)
    if a.plan_only:
        return plan_only(a)
    stop = StopSignal().install()
    from ..common.hashing import sha256_file_hex
    from ..model.neuralese_projection import (NeuraleseProjection, TokenChoiceProjection,
                                               project_token_states, soft_token_mixture)
    from ..serve import load_engine
    from .backbone_policy import configure_backbone_training, plain_named_tensors, shared_parametrized_weights
    from .optim import PortMuonAdamW
    from .trajectory_state import clip_finite_gradients, gradient_norm
    from .quantization import Behaviour, load_component, teacher_shards
    from .text_supervision import ROLE_CODES, balanced_position_weights
    from .text_warmup import (chunked_readout, evaluation_batches, fused_readout, load_text_rows,
                              prepare_text_windows, select_held_document_windows)
    from ..common.paths import root as path_root

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    if a.continue_from is not None and (a.continue_identity is None or a.continue_monitor_reference is None):
        raise ValueError('--continue-from requires declared identity and monitor reference inputs')
    if a.continue_from is None and (a.continue_identity is not None or a.continue_monitor_reference is not None):
        raise ValueError('continuation sidecars require --continue-from')
    a.out.mkdir(parents=True, exist_ok=True)
    reference = a.out / 'monitor-reference.pt'
    if a.continue_from is not None and not reference.exists():
        # These sidecars are declared inputs: resolving a checkpoint symlink may relocate its parent directory.
        import shutil
        shutil.copyfile(a.continue_monitor_reference, reference)
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    code = {str(path.relative_to(Path(__file__).parents[1])): sha256_file_hex(path)
            for path in sorted(Path(__file__).parents[1].rglob('*.py'))}
    # Inputs by cached digest (never a re-read of an unchanged file, e.g. the HDD-archived predecessor at a resume).
    inputs = identity_inputs(a)
    identity = {'schema': SCHEMA, 'options': {k: v for k, v in options.items() if k not in OPERATIONAL
                                              and not (k in LATER_DEFAULTS and v == LATER_DEFAULTS[k])},
                'inputs': inputs, 'code': code}
    # Data continuation (--continued-text-data): decided from the saved state before the model loads; the record is
    # computed by a --plan-only child (it streams both corpora), then the plan is rebuilt here and its hash checked.
    policy = CheckpointPolicy(a.out, every_steps=a.checkpoint_every)
    resume_state = CheckpointPolicy.load(policy.slot) if policy.slot.exists() else None
    continuation_launch, continuation = continuation_mode(a, resume_state, inputs)
    resume_state = None
    if continuation_launch == 'continue':
        import subprocess
        planned = a.out / 'data-continuation.plan.json'
        subprocess.run([sys.executable, '-m', 'natlang_neuralese.train.direct_sft', *raw_argv, '--plan-only',
                        '--plan-out', str(planned)], check=True)
        continuation = json.loads(planned.read_text())['record']
        planned.unlink()
        print(json.dumps({'event': 'data_continuation_planned', **continuation_summary(continuation)}), flush=True)
    elif a.continued_text_data is not None and continuation is None:
        print(json.dumps({'event': 'data_continuation_unused', 'reason': 'no checkpoint trained on the previous data'
                          if not policy.slot.exists() else 'the checkpoint does not read --continued-text-data'}),
              flush=True)

    engine = load_engine(heads_checkpoint=str(a.heads), device=a.device)
    backbone = engine.backbone
    backbone.checkpoint_layers = True if a.checkpoint_layers is None else a.checkpoint_layers
    from ..maple.model import set_dense_moe_kernel
    set_dense_moe_kernel(a.moe_kernel)
    quant = load_component(a.quantization, root=path_root('repo'))
    fp8_backend = None
    if a.fp8_training != 'off':
        if quant is not None and a.fp8_training == 'te-fp8':
            raise ValueError('experimental TE FP8 training cannot be combined with the separate quantization component')
        from .fp8_backend import make_backend
        fp8_backend = make_backend(a.fp8_training, backbone.hf, device=a.device, recipe=a.fp8_recipe,
                                   amax_history_len=a.fp8_amax_history_len)
        # Use TE's checkpoint wrapper for LFM layer recomputation; ordinary torch checkpoint does not promise to
        # preserve TE's FP8 metadata across a recomputed module call.
        backbone.activation_checkpoint = fp8_backend.checkpoint
    fp8_initial_inventory = fp8_backend.inventory() if fp8_backend is not None else {'enabled': False}
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
    # The monitor (train/drift_monitor.py): the base model's own reference when no teacher holds one (LFM), the
    # monitor's gate when no quantization component declares one, and the stage's response-KL bound on the gate.
    from .drift_monitor import DriftMonitor, parse_monitor, setup_monitor
    behaviour, gate_held = setup_monitor(parse_monitor(a.monitor), a.out, backbone.hf, engine.tokenizer,
                                         behaviour=behaviour, gate_held=gate_held,
                                         gate_response_kl=a.gate_response_kl, root=path_root('repo'))
    monitor = DriftMonitor(backbone.hf, engine.tokenizer, gate_held, behaviour, a,
                           bf16=(lambda: quant.context('bf16', 0.0)) if quant is not None else None)
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
    # The latent split, entries and options are shared with the end-to-end stage (train/optim.py).
    from .optim import beta_for_half_life, latent_entries, latent_options as latent_options_for, latent_split
    latent_named, other_backbone = latent_split(backbone_named, a.latent_optimizer)
    config = backbone.hf.config
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    latent = latent_entries(latent_named, a.latent_optimizer, a.latent_lr, head_dim)
    latent_options = latent_options_for(
        a.latent_optimizer, weight_decay=a.latent_weight_decay, skip_idle=a.latent_skip_idle,
        cautious=a.latent_cautious, ns=a.latent_ns,
        betas=(0.9, beta_for_half_life(a.batch_tokens or a.tokens * a.batch, a.latent_beta2_half_life_tokens)))
    optimizer = PortMuonAdamW(head_named + other_backbone, lr=a.head_lr, vocab_size=int(backbone.embedding_weight.shape[0]),
                              latent=latent, latent_optimizer=a.latent_optimizer, latent_options=latent_options)
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
    from .optim import SpikeClip
    spike = SpikeClip(a.grad_spike_clip) if a.grad_spike_clip else None
    in_backward.step_in_backward(gated=True, spike=spike)
    other_trainables = [q for _, q in head_named + other_backbone]
    parameter_names = {id(parameter): name for name, parameter in head_named + backbone_named}
    optimizer_parameter_names = [[parameter_names[id(parameter)] for parameter in group['params']]
                                 for group in optimizer.param_groups]
    backbone_parameter_dtypes = sorted({str(parameter.dtype) for parameter in backbone.hf.parameters()})
    backbone_floating_buffer_dtypes = sorted({str(buffer.dtype) for buffer in backbone.hf.buffers()
                                              if buffer.is_floating_point()})
    print(json.dumps({'event': 'trainables', 'latent_optimizer': a.latent_optimizer, 'latents': len(latent),
                      'grad_spike_clip': a.grad_spike_clip,
                      'backbone_parameter_dtypes': backbone_parameter_dtypes,
                      'backbone_floating_buffer_dtypes': backbone_floating_buffer_dtypes,
                      'declared_autocast': None,
                      'fp8_training': fp8_initial_inventory,
                      'latent_options': {k: list(v) if isinstance(v, tuple) else v for k, v in latent_options.items()},
                      'adamw_or_muon_backbone': [n for n, _ in other_backbone][:4] + ['...'],
                      'projections': {k: sum(q.numel() for q in m.parameters()) for k, m in projections.items()}}),
          flush=True)

    if a.weight_drift_base is not None and not (a.weight_drift_base / 'manifest.json').is_file():
        snapshot_latents(a.weight_drift_base, latent_named)

    # Windows: the whole preceding document (system prompt included) up to --tokens, --target-tokens supervised.
    train_windows, targets, held, held_selection, mask_receipt, rows = stage_data(engine, a, a.text_data)
    if not train_windows or not held:
        raise ValueError('neuralese_direct needs train and held windows')
    # The budget (data audit 2026-10-11, DATA_INVENTORY_20261011.md E): epochs over the target-bearing train windows
    # (target_bearing), sampled by document without replacement through the shared CohortDocumentSampler, packed into
    # updates of up to --batch-tokens forward tokens (direct_schedule); after a data continuation, the rest of the
    # budget on the unseen documents. Context-only windows (history repeated by later decision documents) are not
    # trained.
    from .text_supervision import epoch_schedule
    if not targets:
        raise ValueError('neuralese_direct found no target-bearing train window')
    sampler, schedule = direct_schedule(targets, a, continuation)
    if continuation is not None and (continuation['updates'] != len(schedule) or plan_documents_sha256(
            schedule, continuation['consumed_updates']) != continuation['plan_documents_sha256']):
        raise ValueError('the data continuation record does not rebuild its plan on this data')
    if a.epochs is not None:
        a.steps = len(schedule)  # derived: the stage's schedules (QAT ramps, cadence ends) follow it
    if quant is not None and quant.stage['end'] == 'derived':
        quant.stage['end'] = a.steps
    if quant is not None and continuation is not None and continuation.get('quantization_rebase'):
        # The QAT progress continues from its value at the continuation update to the stage end over the new total.
        rebase = continuation['quantization_rebase']
        quant.stage.update(start=rebase['start'], end=a.steps, progress=list(rebase['progress']))
    from .memory_estimator import (AdaptiveGraphMemory, DIRECT_SFT_ADJACENT_CONTEXT_DISTANCE,
                                  DIRECT_SFT_ADJACENT_MIN_SAMPLES, DIRECT_SFT_ADJACENT_QUANTILE,
                                  backbone_memory_layout, geometry_bytes)
    from .memory_policy import (choose_layer_checkpointing, effective_cuda_free_bytes, ledger_claim,
                                ledger_reusable_bytes)
    memory_estimator = AdaptiveGraphMemory(geometry_version=DIRECT_MEMORY_GEOMETRY_VERSION)

    def restored_memory_estimator(state):
        estimator_state = state.get('direct_memory_estimator')
        identity_code = (state.get('identity') or {}).get('code') or {}
        relabeled = (isinstance(estimator_state, dict) and
                     estimator_state.get('geometry_version') == 'direct-sft-window-v1' and
                     all(identity_code.get(name) == digest
                         for name, digest in _V2_GEOMETRY_V1_LABEL_PACKAGE.items()))
        if relabeled:
            estimator_state = dict(estimator_state)
            estimator_state['geometry_version'] = DIRECT_MEMORY_GEOMETRY_VERSION
            estimator_state['reset_reason'] = None
            print(json.dumps({'event': 'memory_estimator_geometry_relabel',
                              'from': 'direct-sft-window-v1',
                              'to': DIRECT_MEMORY_GEOMETRY_VERSION,
                              'producer_code': _V2_GEOMETRY_V1_LABEL_PACKAGE,
                              'reason': 'pinned producer used v2 FP32-RMSNorm geometry; resume label alone was stale'}),
                  flush=True)
        return AdaptiveGraphMemory(estimator_state, geometry_version=DIRECT_MEMORY_GEOMETRY_VERSION)
    budget = {'epochs': a.epochs, 'updates': a.steps, 'batch_tokens': a.batch_tokens,
              'train_windows': len(train_windows), 'target_windows': len(targets),
              'documents': sum(len(d) for d in sampler.documents.values()),
              'sampling': sampler.receipt(),
              'windows_per_update': (sum(len(batch) for batch, _ in schedule) /
                                     max(1, sum(1 for batch, _ in schedule if batch))),
              'policy': 'epoch-ordered document shuffle without replacement, target-bearing windows, '
                        'length-bucketed right-padded updates (text_supervision.epoch_schedule)',
              'data_continuation': continuation_summary(continuation)}
    totals = {'supervised_tokens': 0, 'target_tokens': 0, 'windows': 0}
    print(json.dumps({'event': 'windows', 'train': len(train_windows), 'held': len(held), 'budget': budget,
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
    # prepare_text_windows gives each window its own token/label arrays. The raw
    # corpus rows (including their JSON text and source arrays) are no longer
    # needed after extracting these producer-declared context masks; keeping
    # them alive needlessly duplicates a large fraction of the corpus in RAM.
    del rows

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
        """One update's windows as a right-padded batch, any geometry: ``ids`` [B, L] (pad: the close token; causal
        attention and causal convolutions keep every real position exact, and padding carries no weight), and each
        window's supervised span re-aligned to column 0: ``span`` [B, S], ``index`` (the hidden-state position that
        predicts each span token), ``valid``, ``weights`` (``balanced_position_weights`` per window, so a batch of one
        is the unpadded computation), ``segments`` (-1 off the span) and ``suffix`` (decision-target positions)."""
        count = len(rows_)
        longest = max(len(r['ids']) for r in rows_)
        spans = max(len(r['ids']) - r['prefix'] for r in rows_)
        close = backbone.controls.close_id
        ids = torch.full((count, longest), close, dtype=torch.long)
        span = torch.full((count, spans), close, dtype=torch.long)
        index = torch.zeros((count, spans), dtype=torch.long)
        valid = torch.zeros((count, spans), dtype=torch.bool)
        suffix = torch.zeros((count, spans), dtype=torch.bool)
        weights = torch.zeros((count, spans), dtype=torch.float32)
        segments = torch.full((count, spans), -1, dtype=torch.long)
        for b, r in enumerate(rows_):
            row = torch.tensor(list(r['ids']), dtype=torch.long)
            length, prefix = len(row), r['prefix']
            size = length - prefix
            ids[b, :length] = row
            span[b, :size] = row[prefix:]
            index[b, :size] = torch.arange(prefix - 1, length - 1)
            index[b, size:] = prefix - 1
            valid[b, :size] = True
            roles = torch.tensor(list(r['roles']), dtype=torch.long) if 'roles' in r else None
            exclude = (torch.tensor(failed(r), dtype=torch.bool)[prefix:][None]
                       if roles is not None or declared_by_document else None)
            start = r.get('supervised_suffix_start')
            weights[b, :size] = balanced_position_weights(
                row[prefix:][None], [start], roles=None if roles is None else roles[prefix:][None],
                context_weight=a.context_weight, feedback_weight=a.feedback_weight, exclude=exclude)[0]
            if start is not None and start < size:
                suffix[b, start:size] = True if exclude is None else ~exclude[0, start:]
            segments[b, :size] = segment_ids(None if roles is None else roles[None, :-1], prefix - 1, length - 1, 1,
                                             'cpu')[0, prefix - 1:]
        to = lambda t: t.to(a.device, non_blocking=True)
        return to(ids), to(span), to(index), to(valid), to(weights), to(segments), to(suffix)

    def span_states(ids, index):
        """The final hidden states that predict each span token, [B, S, width] (one forward over ``ids[:, :-1]``)."""
        if a.sdpa_backend == 'math':
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                hidden = backbone.forward_ids(ids[:, :-1], cutoff=engine.heads.cutoff, logits=False)['h_final']
        else:
            hidden = backbone.forward_ids(ids[:, :-1], cutoff=engine.heads.cutoff, logits=False)['h_final']
        return hidden.gather(1, index[..., None].expand(-1, -1, hidden.shape[-1]))

    def forward(rows_, *, gradients, accumulation=None):
        """(weighted CE, metrics, predictions, token losses) of one padded batch at the active precision."""
        with profile_region('update/data'):
            ids, span, index, valid, weights, segments, suffix = tensors(rows_)
        usable = float(weights.sum()) > 0  # a window made only of failed assistant turns teaches nothing
        if accumulation is None:
            if not usable:
                weights = valid.float()
        elif not accumulation['has_weight']:
            weights = valid.float()
        usable = float(weights.sum()) > 0
        # Both projections read only the supervised positions: segments start inside the span (context and padding
        # are -1), so no convolution tap reaches the context and the slice is exact. Never full-context logits.
        hidden = span_states(ids, index)
        # Project once over the whole turn so every chunk sees its full causal history, then keep the
        # vocabulary-sized readout chunked as before.
        fused = token_choice is None and os.environ.get('NATLANG_FUSED_READOUT', '1') != '0'
        with profile_scope('readout'):
            read = project_token_states(hidden, token_choice, segments)
            if fused:  # text_warmup.fused_readout: logits once per chunk; the mixture from the same probabilities
                training_ce, prediction, token_losses, mixture = fused_readout(
                    backbone, read, span, backbone.controls.close_id, gradients=gradients, position_weights=weights,
                    mixture_table=table)
            else:
                _, training_ce, prediction, _, token_losses = chunked_readout(
                    backbone, read, span, backbone.controls.close_id, gradients=gradients, position_weights=weights)
        real = valid.sum().clamp_min(1)
        training_ce = training_ce * (span.numel() / real)  # the token mean over real positions (padding excluded)
        ce = (token_losses * valid).sum() / real
        with profile_scope('projection'):
            if not fused:
                mixture = soft_token_mixture(backbone.logits, table, hidden.detach())
            value = neuralese(hidden.detach(), mixture.to(hidden.dtype), segments)
            target = table[span]
            value_error = relative_mse(value, target, weights)
        metrics = {'ce': float(ce), 'gold_accuracy': float(((prediction == span) & valid).sum() / real),
                   'value_relative_mse': float(value_error)}
        if not torch.is_grad_enabled():
            mixture_error = relative_mse(mixture, target, weights)
            nearest = (2 * (value.float() @ table.t()) - table.square().sum(-1)).argmax(-1)
            # These are same-evaluation comparisons: the delta is learned projection MSE minus the
            # contemporaneous soft-mixture MSE, not change from an earlier training step.
            metrics.update(value_mixture_relative_mse=float(mixture_error),
                           value_relative_mse_delta=float(value_error - mixture_error),
                           value_nearest_accuracy=float(((nearest == span) & valid).sum() / real))
        else:
            metrics.update(windows=len(rows_), supervised_tokens=int((weights > 0).sum()),
                           target_tokens=int(suffix.sum()), real_tokens=sum(len(r['ids']) - 1 for r in rows_),
                           forward_tokens=int(ids.shape[0] * (ids.shape[1] - 1)))
        loss = training_ce + a.neuralese_weight * value_error
        if accumulation is not None:
            local_weight = weights.sum()
            training_ce = training_ce * (real / accumulation['valid_positions'])
            value_error = value_error * (local_weight / accumulation['weight_mass'].clamp_min(1e-12))
            loss = training_ce + a.neuralese_weight * value_error
        loss_stats = {'valid_positions': int(valid.sum()), 'weight_mass': float(weights.sum())}
        return (loss if usable else loss * 0.0), metrics, prediction, token_losses, loss_stats

    # Resume.
    if not a.save_state:
        policy.save = lambda state, **_: None  # a smoke: logs and report only, never a full state
    step, best, base_eval, history = 0, None, None, []
    archive = None
    pending_restore = None  # a data continuation with a new held set restores after the base is re-measured

    def restore(state):
        nonlocal step, best, base_eval, history, memory_estimator
        from .text_warmup import copy_restored_values
        tensors_by_name = plain_named_tensors(backbone.hf)
        # The shared copy (its own scope): a loop variable left in main() is a view of the mmap'd state and kept the
        # whole mapping plus the dirtied pages alive for the run (v2 resume: 39 GB host anon, as text_warmup 085713c7).
        copy_restored_values({'backbone.' + n: t for n, t in tensors_by_name.items()}, state['student_parameters'])
        for key, module in projections.items():
            module.load_state_dict(state[key])
        from .optim_restore import own_host_tensors
        optimizer.load_state_dict(state['optimizer'])
        own_host_tensors(optimizer)  # no view of the mmap'd state outlives the restore
        optimizer.param_groups = optimizer._groups()
        step, best, base_eval, history = state['step'], state.get('best'), state['base_eval'], state['history']
        monitor.restore(state)  # the update of the guard's last gate and its response KL
        if spike is not None:
            spike.load_state_dict(state.get('grad_spike_clip'))
        memory_estimator = restored_memory_estimator(state)
        restore_training_rng_state(state['rng'], a.device)
        print(json.dumps({'event': 'resumed', 'step': step, 'code_changed': sorted(
            k for k in code if state['identity']['code'].get(k) != code[k])}), flush=True)

    if policy.slot.exists():
        state = CheckpointPolicy.load(policy.slot)
        previous_options = dict(state.get('identity', {}).get('options') or {})
        current_options = dict(identity['options'])
        previous_epochs = previous_options.pop('epochs', None)
        current_epochs = current_options.pop('epochs', None)
        # A data continuation's checkpoint read --continued-text-data in --text-data's place (continuation_mode).
        expected_inputs = (continuation_inputs(a, identity['inputs']) if continuation_launch == 'continue'
                           else identity['inputs'])
        if state.get('schema') != SCHEMA or not same_resume_identity(
                {'options': previous_options, 'inputs': state['identity']['inputs']},
                {'options': current_options, 'inputs': expected_inputs}):
            raise ValueError('neuralese_direct resume identity changed (options or inputs); a stage that moves to '
                             'new data continues only with --continued-text-data naming the data it trained on')
        if continuation_launch == 'continue' and state['step'] != continuation['consumed_updates']:
            raise ValueError('the checkpoint moved since the data continuation was planned')
        saved_fp8 = state.get('fp8_training')
        stateless = lambda envelope: envelope is None or envelope.get('schema') == 'natlang.neuralese-fp8-experts-training/1'
        if a.fp8_training == 'triton-experts' and stateless(saved_fp8):
            pass  # current scaling: no scale state to restore, and a BF16 state resumes onto it
        elif fp8_backend is not None:
            if saved_fp8 is None:
                raise ValueError('exact FP8 resume checkpoint is missing TE scaling state')
            fp8_backend.restore_extra_state(saved_fp8)
        elif saved_fp8 is not None and not stateless(saved_fp8):
            raise ValueError('an FP8 checkpoint cannot resume through the BF16 backend')
        if previous_epochs != current_epochs:
            # The owner may shorten/extend the epoch budget without throwing away trained updates. Reconstruct
            # both deterministic schedules and verify every consumed batch, not just an epoch counter. Fractional
            # final epochs can repack their last bucket, so not every budget change preserves the prefix.
            if previous_epochs is None or current_epochs is None or quant is not None:
                raise ValueError('epoch budget handoff requires epoch schedules without a rebased QAT curriculum')
            old_schedule = epoch_schedule(sampler, seed=a.seed, length=lambda w: len(w['ids']) - 1,
                                          epochs=previous_epochs, max_tokens=a.batch_tokens, max_items=a.batch)
            consumed = state['step']
            if consumed > min(len(old_schedule), len(schedule)) or any(
                    tuple(map(id, old_schedule[i][0])) != tuple(map(id, schedule[i][0]))
                    for i in range(consumed)):
                raise ValueError('epoch budget handoff changes or discards consumed training batches')
            print(json.dumps({'event': 'epoch_budget_handoff', 'previous_epochs': previous_epochs,
                              'epochs': current_epochs, 'consumed_updates_verified': consumed,
                              'updates': len(schedule)}), flush=True)
            del old_schedule
        if continuation_launch == 'continue' and continuation['held']['rebased']:
            pending_restore = state  # mmap'd; restored once the base is measured on the new held set
        else:
            restore(state)
        del state
        if a.continue_from is not None and a.save_state and not Path(a.continue_from).is_symlink():
            archive = PredecessorArchive(Path(a.continue_from))  # an archive a stop interrupted resumes
    elif a.continue_from is not None:
        # A later phase of the lineage (recipe stage after a neuralese_direct predecessor): the predecessor's weights,
        # projections and optimizer state continue on this stage's data from update 0; held baselines are this
        # stage's own (measured at update 0 on its held windows), the drift reference stays the base model's.
        state = CheckpointPolicy.load(a.continue_from)
        if state.get('schema') != SCHEMA:
            raise ValueError('--continue-from is not a neuralese_direct full state')
        previous_identity = state.get('identity') or {}
        previous_inputs = previous_identity.get('inputs') or {}
        if inputs[str(a.heads)] not in previous_inputs.values():
            raise ValueError('--continue-from was trained from different base heads')
        previous_options = previous_identity.get('options') or {}
        current_options = identity['options']
        # A later phase of one lineage changes its data, its budget and its precision schedule (recipe stages:
        # full-prompt warm-up, then program-header direct_sft with the QAT ramp). Supervised span size is also a
        # declared phase geometry knob: context capacity and masking stay fixed, the new phase rebases held metrics
        # and the memory estimator uses a distinct target-size bin. It changes window grouping, so never an exact
        # in-place resume. Everything else must match.
        phase_keys = {'text_data', 'records', 'pieces', 'epochs', 'quantization', 'batch', 'batch_tokens',
                      'target_tokens', 'fp8_training', 'fp8_recipe', 'fp8_amax_history_len'}
        changed_objective = sorted(key for key in set(previous_options) & set(current_options)
                                   if key not in phase_keys and previous_options[key] != current_options[key])
        if changed_objective:
            raise ValueError('--continue-from training options differ: ' + ', '.join(changed_objective))
        saved_fp8 = state.get('fp8_training')
        previous_fp8 = previous_options.get('fp8_training', 'off')
        if fp8_backend is not None:
            if saved_fp8 is not None:
                fp8_backend.restore_extra_state(saved_fp8)
            elif previous_fp8 != 'off':
                raise ValueError('FP8 predecessor checkpoint is missing TE scaling state')
            else:
                print(json.dumps({'event': 'fp8_scaling_state_initialized_for_phase',
                                  'predecessor_backend': 'off', 'backend': 'te-fp8',
                                  'reason': 'declared precision-backend phase change'}), flush=True)
        elif saved_fp8 is not None:
            raise ValueError('FP8 predecessor requires an explicit TE FP8 phase')
        # Full direct checkpoints keep optimizer-schema names in their canonical ``backbone.*`` form.
        # Keep that same form here; only the destination tensor lookup strips the prefix below.
        expected_backbone = dict(backbone_named)
        saved_backbone = state.get('student_parameters') or {}
        if set(saved_backbone) != set(expected_backbone):
            raise ValueError('--continue-from backbone trainable names differ')
        for name, parameter in expected_backbone.items():
            if tuple(saved_backbone[name].shape) != tuple(parameter.shape):
                raise ValueError('--continue-from backbone shape differs: ' + name)
        for key, module in projections.items():
            saved_projection = state.get(key)
            current_projection = module.state_dict()
            if not isinstance(saved_projection, dict) or set(saved_projection) != set(current_projection):
                raise ValueError('--continue-from projection state differs: ' + key)
            if any(tuple(saved_projection[name].shape) != tuple(value.shape)
                   for name, value in current_projection.items()):
                raise ValueError('--continue-from projection shape differs: ' + key)
        saved_optimizer = state.get('optimizer') or {}
        current_optimizer = optimizer.state_dict()
        if (saved_optimizer.get('format') != current_optimizer.get('format') or
                saved_optimizer.get('schema') != current_optimizer.get('schema')):
            raise ValueError('--continue-from optimizer names, shapes, dtypes, or partitions differ')
        saved_names = state.get('optimizer_parameter_names')
        if saved_names is not None and saved_names != optimizer_parameter_names:
            raise ValueError('--continue-from optimizer parameter names differ')
        from .text_warmup import copy_restored_values
        tensors_by_name = plain_named_tensors(backbone.hf)
        copy_restored_values({'backbone.' + n: t for n, t in tensors_by_name.items()}, state['student_parameters'])
        for key, module in projections.items():
            module.load_state_dict(state[key])
        from .optim_restore import own_host_tensors
        optimizer.load_state_dict(state['optimizer'])
        own_host_tensors(optimizer)
        optimizer.param_groups = optimizer._groups()
        memory_estimator = restored_memory_estimator(state)
        if spike is not None:
            spike.load_state_dict(state.get('grad_spike_clip'))
        print(json.dumps({'event': 'continued', 'from': str(a.continue_from), 'predecessor_step': state['step']}),
              flush=True)
        # The checks above bound views of the mmap'd state in main()'s scope (saved_optimizer is the whole optimizer
        # state): drop them with the state, or the mapping and its dirtied pages stay alive for the run (v4 2026-10-12:
        # ~39 GB host anon, the first update refused with 0.16 GB reusable of a 90 GB claim; as the resume path).
        saved_backbone = saved_projection = saved_optimizer = saved_names = current_optimizer = None
        del state
        if a.save_state:
            archive = PredecessorArchive(Path(a.continue_from))

    def full_state():
        from .backbone_policy import plain_named_tensors as plain
        values = plain(backbone.hf)
        return {'schema': SCHEMA, 'identity': identity, 'step': step,
                'student_parameters': {n: values[n.removeprefix('backbone.')].detach() for n, _ in backbone_named},
                **{key: module.state_dict() for key, module in projections.items()},
                'optimizer': optimizer.state_dict(), 'rng': capture_training_rng_state(a.device),
                'optimizer_parameter_names': optimizer_parameter_names,
                'best': best, 'base_eval': base_eval, 'history': history, **monitor.state(),
                'grad_spike_clip': spike.state_dict() if spike is not None else None,
                'direct_memory_estimator': memory_estimator.state_dict(),
                'data_continuation': continuation,
                'fp8_training': fp8_backend.extra_state() if fp8_backend is not None else None}

    def log(name, value):
        with (a.out / name).open('a') as handle:
            handle.write(json.dumps(value) + '\n')


    def evaluate(log_name='eval.jsonl', check_monitor=True):
        nonlocal base_eval
        sums, count, agreement = {}, 0, []
        predictions = []
        with torch.no_grad(), contextlib.ExitStack() as scope:
            if quant is not None:
                scope.enter_context(quant.context('bf16', 0.0))
            if fp8_backend is not None:
                scope.enter_context(fp8_backend.native_modules())
            for batch in evaluation_batches(held, a.eval_batch):
                _, metrics, prediction, _, _ = forward(batch, gradients=False)
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
            if monitor.active and check_monitor:
                report['behaviour_drift'] = monitor.check(step, min(a.stop_at or a.steps, a.steps))
        log(log_name, report)
        return report

    def drift_ce():
        """Held CE on the first --drift-ce-windows held windows (BF16): where held CE bottoms out, between evals."""
        if not a.drift_ce_windows:
            return None
        total = 0.0
        with torch.no_grad(), contextlib.ExitStack() as scope:
            if quant is not None:
                scope.enter_context(quant.context('bf16', 0.0))
            if fp8_backend is not None:
                scope.enter_context(fp8_backend.native_modules())
            for window in held[:a.drift_ce_windows]:
                total += forward([window], gradients=False)[1]['ce']
        return total / len(held[:a.drift_ce_windows])

    if continuation_launch == 'continue':
        if pending_restore is not None:
            # The held set changed (held windows come from --text-data's test split): the stage's base, its update-0
            # weights (the declared predecessor's, else the base model with identity projections), is measured on
            # the new held windows before the saved state is restored, so ce_delta and agreement stay against base.
            if a.continue_from is not None:
                from .text_warmup import copy_restored_values
                predecessor = CheckpointPolicy.load(a.continue_from)
                tensors_by_name = plain_named_tensors(backbone.hf)
                copy_restored_values({'backbone.' + n: t for n, t in tensors_by_name.items()},
                                     predecessor['student_parameters'])
                for key, module in projections.items():
                    module.load_state_dict(predecessor[key])
                predecessor = tensors_by_name = None
            base_eval = None
            evaluate(log_name='continuation-base-eval.jsonl', check_monitor=False)
            base_eval['drift_ce'] = drift_ce()
            rebased = base_eval
            restore(pending_restore)
            pending_restore = None
            continuation['held'].update(
                previous_base_ce=base_eval['ce'], base_ce=rebased['ce'], base_gold_accuracy=rebased['gold_accuracy'],
                base_drift_ce=rebased['drift_ce'], previous_best=best,
                base_weights='continue_from' if a.continue_from is not None else 'base model')
            base_eval, best = rebased, None
            policy.best_metric = None  # the best checkpoint is chosen on the new held set from here on
            (a.out / 'checkpoint-policy.json').write_text(json.dumps({'best_metric': None}) + '\n')
        continuation['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        sidecar = a.out / 'data-continuation.json'
        sidecar.with_suffix('.pending').write_text(json.dumps(continuation) + '\n')
        sidecar.with_suffix('.pending').replace(sidecar)
        print(json.dumps({'event': 'data_continuation', **continuation_summary(continuation)}), flush=True)

    # Disk: hold the blocks of the next full-state write (its tensors, 5% margin; at least the last write's size) so a
    # write never meets a full disk. The reserve is released just for a write and taken back after it; a run that
    # cannot hold it refuses to start, and one that cannot take it back after a write stops on that complete state.
    from .checkpoint_safety import CheckpointDiskReserve, CheckpointReserveError, _state_tensor_bytes
    reserve = None

    def next_write_bytes():
        written = policy.slot.stat().st_size if policy.slot.exists() else 0
        return max(written, int(1.05 * _state_tensor_bytes(full_state())))

    # While the predecessor's archive runs, its blocks may still be on this disk: a reserve that does not fit yet is
    # deferred and taken as soon as the archive has finished (checked every update), at the latest before a write.
    deferred_reserve = []

    def settle_reserve(wait):
        if not deferred_reserve or (archive.running and not wait):
            return
        archive.wait()
        try:
            reserve.acquire()
            deferred_reserve.clear()
            print(json.dumps({'event': 'checkpoint_reserve_acquired_after_archive', 'step': step,
                              'archive_error': archive.error}), flush=True)
        except CheckpointReserveError as error:
            if wait:  # a write is due: attempt it (the save is atomic), the after-write check stops the run
                print(json.dumps({'event': 'checkpoint_reserve_unavailable_at_write', 'step': step,
                                  'error': str(error)}), flush=True)

    if a.save_state:
        reserve = CheckpointDiskReserve(a.out / '.checkpoint-space.reserve', next_write_bytes())
        try:
            reserve.acquire()
        except CheckpointReserveError as error:
            if archive is None or not archive.running:
                print(json.dumps({'event': 'checkpoint_space_preflight_refused', 'step': step, 'error': str(error)}),
                      flush=True)
                reserve.cleanup()
                raise SystemExit(2)
            deferred_reserve.append(str(error))
            print(json.dumps({'event': 'checkpoint_reserve_deferred_until_archive', 'step': step,
                              'error': str(error)}), flush=True)
        atexit.register(reserve.cleanup)
        save_full_state = policy.save

        def reserved_save(state, **options):
            settle_reserve(wait=True)
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
    if step == 0 and a.continue_from is not None:
        # The lineage's identity qualification is its first stage's; a continued stage carries it forward.
        inherited = a.continue_identity
        inherited_identity = json.loads(inherited.read_text())
        if inherited_identity.get('passed') is not True:
            raise ValueError('the declared predecessor identity check did not pass')
        (a.out / 'identity-check.json').write_text(json.dumps({
            **inherited_identity, 'inherited_from': str(inherited)}, indent=2) + '\n')
        evaluate()
        base_eval['drift_ce'] = drift_ce()
    elif step == 0:
        with torch.no_grad(), (fp8_backend.native_modules() if fp8_backend is not None else contextlib.nullcontext()):
            ids, span, index, _, _, _, _ = tensors(held[:1])
            hidden = span_states(ids, index)
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
        base_eval['drift_ce'] = drift_ce()
    end = min(a.stop_at or a.steps, a.steps)  # --stop-at: a prefix of the declared schedule
    loop = TrainingLoop(step, end, stop)
    start_index = step
    saved_step = None
    started = time.time()
    from .text_warmup import _warmup_update_floor_bytes
    direct_trainables = head_named + backbone_named
    from .fp8_backend import SavedTensorAudit

    def gradient_audit(named):
        result = []
        seen = set()
        for name, parameter in named:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            gradient = parameter.grad
            if gradient is None:
                result.append({'name': name, 'present': False, 'shape': list(parameter.shape),
                               'parameter_dtype': str(parameter.dtype)})
                continue
            value = gradient.detach().float()
            result.append({'name': name, 'present': True, 'shape': list(gradient.shape),
                           'gradient_dtype': str(gradient.dtype), 'l2': float(torch.linalg.vector_norm(value)),
                           'max_abs': float(value.abs().max()),
                           'finite': bool(torch.isfinite(value).all())})
        return result

    def tensor_sha256(value):
        """Digest exact logical tensor bytes plus dtype/shape metadata (including BF16 tensors)."""
        tensor = value.detach().contiguous().cpu()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'sha256': hashlib.sha256(raw).hexdigest()}

    def tree_sha256(value):
        """Canonical digest for nested optimizer/TE/RNG state without serializing it to a side file."""
        if isinstance(value, torch.Tensor):
            return {'tensor': tensor_sha256(value)}
        if isinstance(value, bytes):
            return {'bytes_sha256': hashlib.sha256(value).hexdigest(), 'length': len(value)}
        if isinstance(value, dict):
            return {str(key): tree_sha256(value[key]) for key in sorted(value, key=lambda item: str(item))}
        if isinstance(value, (list, tuple)):
            return [tree_sha256(item) for item in value]
        if value is None or type(value) in (bool, int, float, str):
            return value
        raise TypeError(f'unsupported state-hash value: {type(value).__module__}.{type(value).__qualname__}')

    def state_hashes(named):
        parameters = {}
        seen = set()
        for name, parameter in named:
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters[name] = tensor_sha256(parameter)
        rng = {'torch_cpu': tensor_sha256(torch.get_rng_state())}
        if torch.cuda.is_available():
            rng['torch_cuda'] = [tensor_sha256(value) for value in torch.cuda.get_rng_state_all()]
        rng['python'] = hashlib.sha256(json.dumps(random.getstate(), separators=(',', ':')).encode()).hexdigest()
        optimizer_tree = tree_sha256(optimizer.state_dict())
        fp8_tree = tree_sha256(fp8_backend.extra_state()) if fp8_backend is not None else None
        return {'parameters_sha256': canonical_json_sha256_hex(parameters), 'parameters': parameters,
                'optimizer': {'sha256': canonical_json_sha256_hex(optimizer_tree), 'state': optimizer_tree},
                'rng': {'sha256': canonical_json_sha256_hex(rng), 'states': rng},
                'fp8': None if fp8_tree is None else {'sha256': canonical_json_sha256_hex(fp8_tree),
                                                      'state': fp8_tree}}

    cuda_active = torch.cuda.is_available() and str(a.device).startswith('cuda')

    cache_released = []
    # NATLANG_PROFILE_UPDATES=a:b (common/profiling.py): torch.profiler over updates a..b, breakdown and trace in --out.
    profiler = UpdateProfiler(a.out)

    def execution_plan(batch):
        """Pick sequential accumulation and the least checkpointing that fits live device memory."""
        if a.window_execution == 'auto':
            mode = 'sequential' if len(batch) > 1 else 'packed'
        elif a.window_execution == 'packed':
            mode = 'packed'
        else:
            mode = 'sequential'
        context = max(len(window['ids']) - 1 for window in batch)
        target = max(len(window['ids']) - window['prefix'] for window in batch)
        rows = 1 if mode == 'sequential' else len(batch)
        optimizer_floor = _warmup_update_floor_bytes(direct_trainables, optimizer, bootstrap=False, in_backward=mode == "packed")  # packed: the latents step in backward
        fp8_cache_estimate = fp8_backend.cached_weight_bytes_estimate() if fp8_backend is not None else 0
        fp8_cache_reservation = 0
        if fp8_backend is not None and not fp8_backend.has_executed:
            # Ada keeps distinct rowwise and columnwise FP8 weight representations. Reserve both on the first
            # update before the live allocator can account for those lazily-created TE caches.
            fp8_cache_reservation = fp8_cache_estimate
        planning_floor = optimizer_floor + fp8_cache_reservation
        if cuda_active:
            free, total = torch.cuda.mem_get_info(a.device)
            allocator_backend = torch.cuda.get_allocator_backend()
            allocator_reserved = int(torch.cuda.memory_reserved(a.device))
            allocator_allocated = int(torch.cuda.memory_allocated(a.device))
            inactive_split = None
            if allocator_backend == 'native':
                stats = torch.cuda.memory_stats(a.device)
                value = stats.get('inactive_split_bytes.all.current')
                inactive_split = int(value) if value is not None else None
            device_available = effective_cuda_free_bytes(int(free), int(total), allocator_reserved,
                                                         allocator_allocated,
                                                         inactive_split_bytes=inactive_split)
            available = device_available
            # Under the memory ledger the job's claim, not the shared device (on unified memory: the whole
            # machine), bounds what it may allocate; the guard stops a unit that outgrows its claim under pressure.
            claim = ledger_claim()
            claim_available = None
            if claim is not None:
                claim_available = ledger_reusable_bytes(claim['budget_bytes'], claim['host_bytes'],
                                                        allocator_reserved, allocator_allocated,
                                                        inactive_split_bytes=inactive_split)
                available = min(device_available, claim_available)
        else:
            free = total = allocator_reserved = allocator_allocated = None
            available = device_available = claim = claim_available = None
            allocator_backend = None
            inactive_split = None

        def estimate(checkpointed):
            layout = backbone_memory_layout(backbone, checkpointed=checkpointed)
            one_raw = geometry_bytes(context, 0, **layout,
                                     target_tokens=min(128, target),
                                     vocab_size=int(backbone.embedding_weight.shape[0]))
            kind = direct_sft_calibration_kind(a.fp8_training, a.fp8_recipe, mode, checkpointed)
            predicted_activation = memory_estimator.predict(kind, context, target, one_raw) * rows
            calibration = memory_estimator.calibration_info(kind, context, target)
            return one_raw * rows, predicted_activation, calibration

        if a.checkpoint_layers is not None:
            checkpointed = bool(a.checkpoint_layers)
            raw, predicted, calibration = estimate(checkpointed)
        else:
            checkpointed, raw, predicted, calibration = choose_layer_checkpointing(
                estimate, planning_floor, available if cuda_active else None)
        incremental = predicted + planning_floor
        fits = available is None or incremental <= available
        device_bound = claim_available is None or device_available < claim_available
        if not fits and cuda_active and device_bound and not cache_released:
            # On unified memory (DGX GB10) CUDA allocations only use MemFree: stale page cache of read-once
            # checkpoints and corpora hides reusable memory. Drop it once (the ledger's shared release) and re-measure.
            import subprocess
            cache_released.append(True)
            ledger = Path(os.environ.get('NATLANG_REPO', path_root('repo'))) / 'scripts' / 'memory_ledger.py'
            output = ''
            if ledger.is_file():
                released = subprocess.run(['python3', str(ledger), 'release-cache'], capture_output=True, text=True)
                output = released.stdout[-400:]
            # This process's own reads (model weights, heads, data) are the bulk of the cache it meets; drop them
            # directly too, so a busy or stuck shared release never leaves them pinned.
            sources = {Path(p).resolve().parent for p in (a.heads, a.text_data, a.records, a.pieces) if p}
            name = getattr(backbone.hf.config, '_name_or_path', None)
            if name and Path(name).exists():
                sources.add(Path(name).resolve())
            advised = 0
            for source in sources:
                for path in source.rglob('*'):
                    try:
                        if path.is_file() and path.stat().st_size >= 64 << 20:
                            handle = os.open(path, os.O_RDONLY)
                            os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
                            os.close(handle)
                            advised += path.stat().st_size
                    except OSError:
                        continue
            print(json.dumps({'event': 'page_cache_released', 'ledger': output,
                              'own_files_advised_gb': advised / 2 ** 30}), flush=True)
            return execution_plan(batch)
        if not fits:
            bound = 'reusable CUDA memory' if device_bound else f"the ledger claim of {claim['unit']}"
            raise MemoryError(f'neuralese_direct estimated update exceeds {bound}: '
                              f'{incremental} bytes required, {available} available; no forward was started')
        layout = backbone_memory_layout(backbone, checkpointed=checkpointed)
        return {'mode': mode, 'checkpoint_layers': checkpointed, 'context_tokens': context,
                'target_tokens': target, 'activation_geometry_bytes': raw,
                'predicted_activation_bytes': predicted, 'gradient_optimizer_floor_bytes': optimizer_floor,
                'fp8_weight_cache_estimate_bytes': fp8_cache_estimate,
                'fp8_weight_cache_reservation_bytes': fp8_cache_reservation,
                'estimated_incremental_peak_bytes': incremental, 'reusable_cuda_bytes': available,
                'device_reusable_bytes': device_available, 'ledger_claim_reusable_bytes': claim_available,
                'ledger_unit': claim['unit'] if claim else None,
                'ledger_budget_bytes': claim['budget_bytes'] if claim else None,
                'ledger_host_bytes': claim['host_bytes'] if claim else None,
                'ledger_budget_source': claim['source'] if claim else None,
                'device_free_bytes': int(free) if free is not None else None,
                'device_total_bytes': int(total) if total is not None else None,
                'allocator_backend': allocator_backend,
                'allocator_reserved_bytes': allocator_reserved,
                'allocator_allocated_bytes': allocator_allocated,
                'allocator_inactive_split_bytes': inactive_split,
                'cuda_allocator_config': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                'estimator_margin': memory_estimator.margin, 'estimator_geometry': memory_estimator.geometry_version,
                'fp8_backend': ({'enabled': False} if fp8_backend is None else {
                    **fp8_backend.configuration(), 'cache_estimate_bytes': fp8_cache_estimate,
                    'cache_estimate_reserved_this_update': not fp8_backend.has_executed}),
                'activation_calibration_sample_count': calibration['sample_count'],
                'activation_calibration_source_bins': calibration['source_bins'],
                'activation_calibration_source': calibration['source'],
                'activation_calibration_context_bin_distance': calibration['context_bin_distance'],
                'activation_calibration_quantile': calibration['quantile'],
                'activation_adjacent_context_distance': DIRECT_SFT_ADJACENT_CONTEXT_DISTANCE,
                'activation_adjacent_min_samples': DIRECT_SFT_ADJACENT_MIN_SAMPLES,
                'activation_adjacent_quantile': DIRECT_SFT_ADJACENT_QUANTILE,
                'activation_layout': {key: value for key, value in layout.items()
                                      if key in ('width', 'layers', 'intermediate', 'kv_width', 'dtype_bytes',
                                                 'fp32_norm_saved_bytes_per_token',
                                                 'fp32_norm_transient_bytes_per_token')}}

    for step_index in loop:
        diagnostic_state_before = (state_hashes(direct_trainables)
                                   if step_index < a.state_hash_steps else None)
        if fp8_backend is not None:
            fp8_backend.reset_row_counts()
        batch, covered = schedule[step_index]
        point, mix = ('bf16', 0.0) if quant is None else quant.sampled(step_index)[:2]
        profiler.begin(step_index + 1)
        if deferred_reserve:
            settle_reserve(wait=False)
        tick = time.time()
        with profile_region('update/plan'):
            plan = execution_plan(batch)
        if step_index < a.gradient_audit_steps and (plan['mode'] != 'sequential' or len(batch) < 2):
            raise ValueError('--gradient-audit-steps requires multi-window sequential gradient accumulation')
        if step_index < a.gradient_digest_steps and (plan['mode'] != 'sequential' or len(batch) < 2):
            raise ValueError('--gradient-digest-steps requires multi-window sequential gradient accumulation')
        print(json.dumps({'event': 'execution_plan', 'step': step_index + 1,
                          'source_documents': [window['document'] for window in batch], **plan}), flush=True)
        backbone.checkpoint_layers = plan['checkpoint_layers']
        update_start_allocated = int(torch.cuda.memory_allocated(a.device)) if cuda_active else 0
        if cuda_active:
            torch.cuda.reset_peak_memory_stats(a.device)
        saved_tensor_audit = SavedTensorAudit() if step_index < a.saved_tensor_audit_steps else None
        with contextlib.ExitStack() as scope:
            if saved_tensor_audit is not None:
                scope.enter_context(saved_tensor_audit)
            if quant is not None:
                scope.enter_context(quant.context(point, mix))
            if fp8_backend is not None:
                scope.enter_context(fp8_backend.autocast())
            if a.sdpa_backend == 'math':
                # Keep this active through backward too: activation-checkpoint recomputation happens after the
                # forward returns, and must use the same declared attention backend as the original forward.
                from torch.nn.attention import SDPBackend, sdpa_kernel
                scope.enter_context(sdpa_kernel(backends=[SDPBackend.MATH]))
            if plan['mode'] == 'sequential' and len(batch) > 1:
                _, _, _, valid_all, weights_all, _, _ = tensors(batch)
                valid_positions = valid_all.sum().clamp_min(1)
                has_weight = bool(weights_all.sum() > 0)
                effective_weights = weights_all if has_weight else valid_all.float()
                normalizers = {'valid_positions': valid_positions,
                               'weight_mass': effective_weights.sum(), 'has_weight': has_weight}
                del valid_all, weights_all, effective_weights
                metrics_rows, loss_value = [], 0.0
                # The gated latent hooks are deliberately unarmed here: their gradients accumulate like the
                # other parameters, and the shared optimizer updates every partition exactly once below.
                with shared_parametrized_weights(backbone.hf):
                    for window in batch:
                        with profile_region('update/forward'):
                            part, part_metrics, _, _, part_stats = forward(
                                [window], gradients=True, accumulation=normalizers)
                        if not torch.isfinite(part):
                            raise RuntimeError('nonfinite neuralese_direct loss')
                        with profile_region('update/backward'), backward_phase():
                            part.backward()
                        loss_value += float(part.detach())
                        metrics_rows.append((part_metrics, part_stats))
                        del part
                span_total = int(valid_positions)
                weight_total = float(normalizers['weight_mass'])
                metrics = {
                    'ce': sum(m['ce'] * s['valid_positions'] for m, s in metrics_rows) / span_total,
                    'gold_accuracy': sum(m['gold_accuracy'] * s['valid_positions'] for m, s in metrics_rows) / span_total,
                    'value_relative_mse': sum(m['value_relative_mse'] * s['weight_mass']
                                              for m, s in metrics_rows) / max(weight_total, 1e-12),
                    'windows': sum(m['windows'] for m, _ in metrics_rows),
                    'supervised_tokens': sum(m['supervised_tokens'] for m, _ in metrics_rows),
                    'target_tokens': sum(m['target_tokens'] for m, _ in metrics_rows),
                    'real_tokens': sum(m['real_tokens'] for m, _ in metrics_rows),
                    'forward_tokens': sum(m['forward_tokens'] for m, _ in metrics_rows),
                }
                loss = loss_value
                latent_norm = gradient_norm([parameter for _, parameter, *_ in latent])
                all_trainables = other_trainables + [parameter for _, parameter, *_ in latent]
                if spike is not None:
                    global_norm = clip_finite_gradients(all_trainables, spike=spike)
                    threshold = spike.threshold()
                    exact_coefficient = spike.coefficient(float(global_norm))
                else:
                    global_norm = gradient_norm(all_trainables)
                    if not torch.isfinite(global_norm):
                        raise RuntimeError('nonfinite neuralese_direct gradient norm')
                    threshold, exact_coefficient = None, 1.0
                grad_clip = {'threshold': threshold, 'global_norm': float(global_norm),
                             'latent_norm': float(latent_norm), 'exact_coefficient': exact_coefficient,
                             'latent_applied_norm': float(latent_norm) * exact_coefficient,
                             'latent_exact_norm': float(latent_norm) * exact_coefficient}
                if spike is not None:
                    spike.observe(grad_clip['global_norm'])
                armed = None
            else:
                with shared_parametrized_weights(backbone.hf) as next_pass:
                    next_pass()
                    with in_backward.in_backward() as armed:
                        with profile_region('update/forward'):
                            loss, metrics, _, _, _ = forward(batch, gradients=True)
                        if not torch.isfinite(loss):
                            raise RuntimeError('nonfinite neuralese_direct loss')
                        with profile_region('update/backward'), backward_phase():
                            loss.backward()
                latent_norm = armed.norm()
        gradient_metrics = None
        if step_index < a.gradient_audit_steps:
            gradient_metrics = gradient_audit(head_named + backbone_named)
        gradient_digests = None
        if step_index < a.gradient_digest_steps:
            gradient_digests = {}
            seen_gradient_parameters = set()
            for name, parameter in head_named + backbone_named:
                if id(parameter) in seen_gradient_parameters:
                    continue
                seen_gradient_parameters.add(id(parameter))
                gradient_digests[name] = (None if parameter.grad is None else tensor_sha256(parameter.grad))
        # The global gradient norm: the latents' (stepped in backward, spike-clipped by the norm so far) with the
        # others' (spike-clipped exactly here with --grad-spike-clip).
        if armed is not None:
            if spike is not None:
                global_norm = clip_finite_gradients(other_trainables, consumed_norm=latent_norm, spike=spike)
            else:
                global_norm = torch.linalg.vector_norm(torch.stack([gradient_norm(other_trainables).to(latent_norm.device),
                                                                    latent_norm.double()]))
            grad_clip = armed.clip_record(global_norm)
            if spike is not None:
                spike.observe(grad_clip['global_norm'])
        with profile_region('update/optimizer'):
            optimizer.step()  # routers, norms and projections (the latents stepped in backward)
            optimizer.zero_grad(set_to_none=True)
        step = step_index + 1
        if diagnostic_state_before is not None or gradient_digests is not None:
            diagnostic = {'schema': 'natlang.neuralese-direct-step-diagnostic/1', 'step': step,
                          'sdpa_backend': a.sdpa_backend, 'gradient_sha256_by_name': gradient_digests,
                          'state_before': diagnostic_state_before,
                          'state_after': state_hashes(direct_trainables) if diagnostic_state_before is not None else None}
            (a.out / f'diagnostic-step-{step:06d}.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
        if reserve is not None and step_index == start_index:  # optimizer slots exist now: size the next write
            try:
                reserve.grow_to(next_write_bytes())
            except CheckpointReserveError as error:
                print(json.dumps({'event': 'checkpoint_space_preflight_refused', 'step': step, 'error': str(error)}),
                      flush=True)
                raise SystemExit(2)
        seconds = time.time() - tick
        observed_increment = (max(0, int(torch.cuda.max_memory_allocated(a.device)) - update_start_allocated)
                              if cuda_active else None)
        fp8_rows = fp8_backend.collect_row_counts() if fp8_backend is not None else None
        if observed_increment is not None:
            # Successful updates calibrate only matching packed/sequential and checkpoint geometries.
            fixed_bytes = (plan['gradient_optimizer_floor_bytes'] +
                           plan['fp8_weight_cache_reservation_bytes'])
            observed_activation = max(1, observed_increment - fixed_bytes)
            memory_estimator.observe(
                direct_sft_calibration_kind(a.fp8_training, a.fp8_recipe, plan['mode'],
                                            plan['checkpoint_layers']),
                plan['context_tokens'], plan['target_tokens'], plan['activation_geometry_bytes'],
                observed_activation)
        for key in totals:
            totals[key] += metrics[key]
        log('train.jsonl', {'step': step, 'loss': float(loss), 'precision': point,
                            'precision_policy': point, 'mix': mix,
                            'window_execution': plan['mode'], 'checkpoint_layers': plan['checkpoint_layers'],
                            'memory_plan': plan, 'observed_incremental_peak_bytes': observed_increment,
                            'fp8_training': fp8_backend.configuration() if fp8_backend is not None else {'enabled': False},
                            'fp8_row_padding': fp8_rows,
                            'saved_tensor_audit': saved_tensor_audit.report() if saved_tensor_audit is not None else None,
                            'gradient_audit': gradient_metrics,
                            'backbone_parameter_dtypes': backbone_parameter_dtypes,
                            'backbone_floating_buffer_dtypes': backbone_floating_buffer_dtypes,
                            'declared_autocast': None, **metrics,
                            'epochs': covered, 'supervised_tokens_per_second': metrics['supervised_tokens'] / seconds,
                            'cuda_peak_allocated_gb': torch.cuda.max_memory_allocated(a.device) / 2 ** 30
                            if cuda_active else None,
                            'cuda_peak_reserved_gb': torch.cuda.max_memory_reserved(a.device) / 2 ** 30
                            if cuda_active else None,
                            'grad_clip': grad_clip, 'seconds': seconds})
        if fp8_backend is not None and step == start_index + 1:
            log('fp8-runtime-inventory.jsonl', {'step': step, **fp8_backend.inventory()})
        profiler.update_done(step, seconds, forward_tokens=metrics['forward_tokens'], windows=metrics['windows'],
                             checkpoint_layers=plan['checkpoint_layers'], mode=plan['mode'], precision=point)
        if step % a.eval_every == 0 or step == end:
            with profile_region('update/eval'):
                report = evaluate()
            history.append({k: report[k] for k in ('step', 'ce', 'ce_delta', 'value_relative_mse_delta')})
            improved = best is None or report['ce'] < best['ce']
            if (report.get('behaviour_drift') or {}).get('guard') == 'stop':
                report['stopped_by'] = 'behaviour_drift'
                print(json.dumps({'event': 'behaviour_drift_stop', 'step': step}), flush=True)
                stop.request()
            if improved and not stop.requested:
                best = {'step': step, 'ce': report['ce']}
                with profile_region('update/save'):
                    policy.save(full_state(), metric=report['ce'])
                saved_step = step
            (a.out / 'report.json').write_text(json.dumps({**report, 'status': 'running'}, indent=2) + '\n')
        elif a.drift_every and step % a.drift_every == 0 and monitor.active:
            with profile_region('update/drift'):
                drift = monitor.check(step, end)
                if base_eval is not None and 'drift_ce' not in base_eval:
                    base_eval['drift_ce'] = None  # a lineage resumed from before the measure has no base value
                log('drift.jsonl', {'step': step, 'held_ce_subset': drift_ce(), 'base_held_ce_subset':
                                    base_eval.get('drift_ce') if base_eval else None, 'behaviour_drift': drift})
            if drift['guard'] == 'stop':
                print(json.dumps({'event': 'behaviour_drift_stop', 'step': step}), flush=True)
                stop.request()
        if (policy.due(step) or stop.requested) and saved_step != step:  # never a second write of one step:
            with profile_region('update/save'):
                policy.save(full_state())  # it would replace the slot and break best-checkpoint.pt's hard link
            saved_step = step
        profiler.maybe_finish(step)
    profiler.close(step)
    if archive is not None:
        archive.wait()
    finished = step >= end
    gate = monitor.final_gate(step) if finished else None
    final = json.loads((a.out / 'report.json').read_text()) if (a.out / 'report.json').exists() else {}
    drift_stopped = any((json.loads(line).get('behaviour_drift') or {}).get('guard') == 'stop'
                        for name in ('eval.jsonl', 'drift.jsonl') if (a.out / name).is_file()
                        for line in (a.out / name).open())
    value_delta = final.get('value_relative_mse_delta', math.inf)
    value_qualified = math.isfinite(value_delta) and value_delta < -a.value_margin
    # Owner 2026-10-11: the Neuralese value requirement belongs to the end-to-end stage (consumers reading values
    # against consumers reading tokens); here the value metric and its margin are report-only.
    qualified = bool(finished and step >= a.steps and final.get('ce_delta', 0.0) < 0 and not drift_stopped and
                     (gate is None or gate['passed']))
    identity_file = a.out / 'identity-check.json'
    final.update(identity_check=json.loads(identity_file.read_text()) if identity_file.is_file() else None)
    status = ('complete' if step >= a.steps else 'stopped_at') if finished else ('checkpointed_on_drift_guard' if drift_stopped else 'checkpointed_on_signal')
    final.update(status=status, completed=step >= a.steps, steps=step, stopped_at=a.stop_at or None, best=best,
                 behaviour_gate=gate, drift_stopped=drift_stopped, value_qualified=value_qualified, qualified=qualified,
                 value_gate={'margin': a.value_margin, 'delta': final.get('value_relative_mse_delta'),
                             'report_only': True, 'moved_to': 'end-to-end stage (owner 2026-10-11)'},
                 seconds=time.time() - started, budget=budget, data_continuation=continuation_summary(continuation),
                 epochs_covered=schedule[step - 1][1] if step else 0.0, this_run=dict(totals),
                 cuda_peak_allocated_gb=torch.cuda.max_memory_allocated(a.device) / 2 ** 30 if cuda_active else None,
                 scope='Neuralese direct v1 (plans/neuralese/NEURALESE_DIRECT_V1.md)')
    if saved_step != step:
        policy.save(full_state())
    (a.out / 'report.json').write_text(json.dumps(final, indent=2) + '\n')
    if a.weight_drift_base is not None:
        (a.out / 'weight-drift.json').write_text(json.dumps(
            latent_drift(a.weight_drift_base, latent_named, in_backward, step), indent=1) + '\n')
    print(json.dumps({'event': 'neuralese_direct_' + final['status'], 'step': step, 'qualified': qualified}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
