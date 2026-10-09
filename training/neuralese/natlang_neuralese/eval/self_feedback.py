"""Measure whether projected feedback preserves the crisp next-token channel.

This is a channel-fidelity diagnostic, not a task or runtime certificate. It
compares ordinary token embeddings with the actual Neuralese payloads on the
same projection-generated continuation, then checks whether that continuation
has substantially worse ordinary-text likelihood than a crisp greedy sample.
"""
import argparse
import json
import math
from pathlib import Path

import torch

from ..serve import load_engine
from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
from ..train.output_embedding_projection import sha
from ..train.text_warmup import (
    chunked_readout, load_text_rows, prepare_text_windows,
    select_held_document_windows,
)
from .projected_history import autoregressive_payloads


def _consumer_states(backbone, heads, prefix, history):
    if history.ndim != 3 or history.shape[0] != prefix.shape[0]:
        raise ValueError('feedback history must be an aligned [batch,time,width] tensor')
    if prefix.ndim != 2 or prefix.shape[1] < 1:
        raise ValueError('a nonempty token prefix is required')
    embeds = torch.cat((backbone.embed(prefix), history), dim=1)
    result = backbone.forward_embeds(embeds, logits=False)
    start = prefix.shape[1] - 1
    return result['h_final'][:, start:start + history.shape[1] + 1]


@torch.no_grad()
def _distribution_comparison(backbone, plain_states, projected_states, *, logit_chunk_tokens):
    """Compare normalized distributions in bounded time-by-vocabulary blocks."""
    if plain_states.shape != projected_states.shape or plain_states.ndim != 3:
        raise ValueError('plain and projected states must have identical [batch,time,width] shapes')
    if logit_chunk_tokens < 1:
        raise ValueError('logit_chunk_tokens must be positive')
    agreements = []
    divergences = []
    for start in range(0, plain_states.shape[1], logit_chunk_tokens):
        stop = min(start + logit_chunk_tokens, plain_states.shape[1])
        # Restrict the live vocabulary activation to a few positions. One full
        # vocabulary vector is necessary for exact normalization; no T x V
        # logits are retained across chunks.
        plain_logits = backbone.logits(plain_states[:, start:stop]).float()
        projected_logits = backbone.logits(projected_states[:, start:stop]).float()
        plain_logp = torch.log_softmax(plain_logits, dim=-1)
        projected_logp = torch.log_softmax(projected_logits, dim=-1)
        kl = (plain_logp.exp() * (plain_logp - projected_logp)).sum(dim=-1)
        agreement = plain_logits.argmax(dim=-1).eq(projected_logits.argmax(dim=-1))
        divergences.append(kl)
        agreements.append(agreement)
        del plain_logits, projected_logits, plain_logp, projected_logp
    return torch.cat(divergences, dim=1), torch.cat(agreements, dim=1)


def _mean_and_count(values):
    if values.numel() == 0:
        raise ValueError('self-feedback metrics require at least one scored token')
    return float(values.float().mean()), int(values.numel())


@torch.no_grad()
def self_feedback_window_metrics(backbone, heads, prefix, *, steps=256,
                                 logit_chunk_tokens=128, readout_chunk_tokens=128):
    """Return paired channel-fidelity and ordinary-text continuation metrics."""
    if prefix.ndim != 2 or prefix.shape[1] < 1:
        raise ValueError('prefix must be a nonempty rank-two token tensor')
    if steps < 2:
        raise ValueError('self-feedback comparison needs at least two generated tokens')
    if heads.read_markers:
        raise ValueError('self-feedback requires the raw read profile')
    payloads, generated = autoregressive_payloads(
        backbone, heads, prefix, steps, ('ar_greedy', 'ar_projection'),
        return_generated_tokens=True)
    projection_tokens = generated['ar_projection']
    crisp_tokens = generated['ar_greedy']
    if projection_tokens.shape != crisp_tokens.shape or projection_tokens.shape[1] != steps:
        raise ValueError('both rollouts must return the requested aligned token span')

    # A payload at position j is the feedback used to predict token j+1.
    projected_history = heads.read_embeddings(backbone, payloads['ar_projection'][:, :-1])
    ordinary_projection_history = backbone.embed(projection_tokens[:, :-1])
    ordinary_crisp_history = backbone.embed(crisp_tokens[:, :-1])
    projected_states = _consumer_states(backbone, heads, prefix, projected_history)
    plain_projection_states = _consumer_states(backbone, heads, prefix, ordinary_projection_history)
    plain_crisp_states = _consumer_states(backbone, heads, prefix, ordinary_crisp_history)
    if projected_states.shape[1] != steps:
        raise ValueError('consumer state count must align with generated tokens')

    kl, agreement = _distribution_comparison(
        backbone, plain_projection_states, projected_states,
        logit_chunk_tokens=logit_chunk_tokens)
    _, _, _, _, projection_ce = chunked_readout(
        backbone, plain_projection_states, projection_tokens,
        backbone.controls.close_id, chunk_size=readout_chunk_tokens, gradients=False)
    _, _, _, _, crisp_ce = chunked_readout(
        backbone, plain_crisp_states, crisp_tokens,
        backbone.controls.close_id, chunk_size=readout_chunk_tokens, gradients=False)
    projection_kl, token_count = _mean_and_count(kl)
    agreement_rate, _ = _mean_and_count(agreement)
    projection_text_ce, _ = _mean_and_count(projection_ce)
    crisp_text_ce, _ = _mean_and_count(crisp_ce)
    values = (projection_kl, agreement_rate, projection_text_ce, crisp_text_ce)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('self-feedback evaluation produced a nonfinite metric')
    projection_ids = projection_tokens[0].detach().cpu().tolist()
    crisp_ids = crisp_tokens[0].detach().cpu().tolist()
    projection_close = next((index for index, token in enumerate(projection_ids)
                             if token == backbone.controls.close_id), None)
    crisp_close = next((index for index, token in enumerate(crisp_ids)
                        if token == backbone.controls.close_id), None)
    return {
        'tokens': token_count,
        'kl_plain_to_projected_nats': projection_kl,
        'argmax_agreement': agreement_rate,
        'projection_generated_plain_history_ce': projection_text_ce,
        'crisp_generated_plain_history_ce': crisp_text_ce,
        'quality_ce_gap': projection_text_ce - crisp_text_ce,
        'generated_token_ids': projection_ids,
        'crisp_token_ids': crisp_ids,
        'projection_first_close_index': projection_close,
        'crisp_first_close_index': crisp_close,
        'fixed_span_includes_post_close_positions': any(
            close_index is not None and close_index < steps - 1
            for close_index in (projection_close, crisp_close)),
        'stopping_qualified': False,
        'logit_chunk_tokens': logit_chunk_tokens,
        'max_live_logit_positions': min(logit_chunk_tokens, steps),
    }


def _aggregate(metrics, *, thresholds):
    if not metrics:
        raise ValueError('each evaluation stratum must contain at least one window')
    total = sum(row['tokens'] for row in metrics)
    if total < 1:
        raise ValueError('each evaluation stratum must contain scored tokens')
    keys = ('kl_plain_to_projected_nats', 'argmax_agreement',
            'projection_generated_plain_history_ce', 'crisp_generated_plain_history_ce')
    means = {key: sum(row[key] * row['tokens'] for row in metrics) / total for key in keys}
    means['quality_ce_gap'] = (means['projection_generated_plain_history_ce'] -
                               means['crisp_generated_plain_history_ce'])
    passed = (all(math.isfinite(value) for value in means.values()) and
              means['argmax_agreement'] >= thresholds['min_argmax_agreement'] and
              means['kl_plain_to_projected_nats'] <= thresholds['max_kl_nats'] and
              means['quality_ce_gap'] <= thresholds['max_quality_ce_gap_nats'])
    return {'windows': len(metrics), 'tokens': total, **means, 'passed': passed}


def _validate_thresholds(thresholds):
    expected = {'min_argmax_agreement', 'max_kl_nats', 'max_quality_ce_gap_nats'}
    if not isinstance(thresholds, dict) or set(thresholds) != expected:
        raise ValueError('thresholds must declare agreement, KL and quality-gap limits')
    for name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if not 0 <= thresholds['min_argmax_agreement'] <= 1:
        raise ValueError('minimum argmax agreement must be in [0,1]')
    if thresholds['max_kl_nats'] < 0 or thresholds['max_quality_ce_gap_nats'] < 0:
        raise ValueError('KL and quality CE-gap limits must be nonnegative')
    return {name: float(value) for name, value in thresholds.items()}


def evaluate_windows(backbone, heads, selected_windows, *, steps=256,
                     logit_chunk_tokens=128, readout_chunk_tokens=128,
                     thresholds=None):
    """Evaluate selected first/last windows and token-weighted strata."""
    if thresholds is None:
        thresholds = {
            'min_argmax_agreement': .99,
            'max_kl_nats': .02,
            'max_quality_ce_gap_nats': .05,
        }
    thresholds = _validate_thresholds(thresholds)
    rows = []
    for window in selected_windows:
        parameter = next(iter(backbone.parameters()), None)
        device = parameter.device if parameter is not None else torch.device('cpu')
        ids = torch.tensor([list(window['ids'])], device=device,
                           dtype=torch.long)
        prefix = ids[:, :window['prefix']]
        available_target_span = ids.shape[1] - window['prefix']
        actual_steps = min(steps, available_target_span)
        if actual_steps < 2:
            raise ValueError('selected window has fewer than two target-span tokens for self-feedback comparison')
        result = self_feedback_window_metrics(
            backbone, heads, prefix, steps=actual_steps,
            logit_chunk_tokens=logit_chunk_tokens,
            readout_chunk_tokens=readout_chunk_tokens)
        length_band = ('short' if actual_steps <= 32 else
                       'medium' if actual_steps <= 128 else 'long')
        strata = list(window['strata']) + ['length_band:' + length_band]
        rows.append({
            'document_sha256': window['document'],
            'source_groups': window['groups'],
            'offset': window['offset'],
            'strata': strata,
            'prefix_tokens': window['prefix'],
            'available_target_span_tokens': available_target_span,
            'requested_steps': steps,
            'actual_steps': actual_steps,
            'length_band': length_band,
            **result,
        })
    if not rows:
        raise ValueError('no selected windows')
    stratum_names = {'first', 'last'}
    stratum_names.update('length_band:' + row['length_band'] for row in rows)
    stratum_rows = {}
    for stratum in sorted(stratum_names):
        members = [row for row in rows if stratum in row['strata']]
        stratum_rows[stratum] = _aggregate(members, thresholds=thresholds)
    for row in rows:
        row['proposed_gate_passed'] = _aggregate([row], thresholds=thresholds)['passed']
    return {
        'thresholds': thresholds,
        'windows': rows,
        'strata': stratum_rows,
        'proposed_gate_passed': all(value['passed'] for value in stratum_rows.values()),
    }


def prepare_held_test_windows(engine, rows, *, tokens, prefix_tokens,
                              target_tokens, held_documents,
                              mask_system_prompt=True):
    """Use the trainer's shared text window and leading-system mask policy."""
    split_windows, mask_receipt = prepare_text_windows(
        engine, rows, tokens=tokens, prefix_tokens=prefix_tokens,
        target_tokens=target_tokens, mask_system_prompt=mask_system_prompt)
    selected, selection = select_held_document_windows(
        split_windows['test'], held_documents)
    offsets = {item['document_sha256']: item['selected_window_offsets']
               for item in selection['selected_documents']}
    for window in selected:
        selected_offsets = offsets[window['document']]
        strata = []
        if window['offset'] == selected_offsets[0]:
            strata.append('first')
        if window['offset'] == selected_offsets[-1]:
            strata.append('last')
        window['strata'] = strata
    return selected, selection, mask_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument('--heads', type=Path, help='exact serving heads checkpoint')
    checkpoint.add_argument('--checkpoint', type=Path, help='exact full recurrence checkpoint')
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--pieces', type=Path)
    parser.add_argument('--text-data', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--prefix-tokens', type=int, default=32)
    parser.add_argument('--target-tokens', type=int, default=256)
    parser.add_argument('--held-documents', type=int, default=16)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--mask-system-prompt', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='mask leading system-prompt tokens from score targets (default: true)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--logit-chunk-tokens', type=int, default=128)
    parser.add_argument('--readout-chunk-tokens', type=int, default=128)
    parser.add_argument('--min-argmax-agreement', type=float, default=.99)
    parser.add_argument('--max-kl-nats', type=float, default=.02)
    parser.add_argument('--max-quality-ce-gap-nats', type=float, default=.05)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('fresh diagnostic output required')
    if args.steps < 2 or args.steps > args.target_tokens:
        raise ValueError('steps must be at least two and no greater than target-tokens')
    thresholds = _validate_thresholds({
        'min_argmax_agreement': args.min_argmax_agreement,
        'max_kl_nats': args.max_kl_nats,
        'max_quality_ce_gap_nats': args.max_quality_ce_gap_nats,
    })
    input_paths = [args.checkpoint or args.heads, args.records]
    input_paths.extend(path for path in (args.pieces, args.text_data) if path is not None)
    pins = {str(path.resolve()): sha(path) for path in input_paths}
    package_root = Path(__file__).resolve().parents[1]
    code_paths = {
        'eval/self_feedback.py': Path(__file__).resolve(),
        'eval/projected_history.py': package_root / 'eval/projected_history.py',
        'train/text_warmup.py': package_root / 'train/text_warmup.py',
        'train/trajectory_state.py': package_root / 'train/trajectory_state.py',
        'serve/recurrence_checkpoint.py': package_root / 'serve/recurrence_checkpoint.py',
        'serve/__init__.py': package_root / 'serve/__init__.py',
        'model/heads.py': package_root / 'model/heads.py',
        'data/text_corpus.py': package_root / 'data/text_corpus.py',
        'common/hashing.py': package_root / 'common/hashing.py',
        'train/output_embedding_projection.py': package_root / 'train/output_embedding_projection.py',
    }
    code_hashes = {name: sha(path) for name, path in code_paths.items()}
    checkpoint_step = None
    if args.checkpoint:
        engine, state = load_recurrence_checkpoint(args.checkpoint, device=args.device)
        checkpoint_step = state['step']
    else:
        engine = load_engine(heads_checkpoint=str(args.heads), device=args.device)
    engine.backbone.eval(); engine.heads.eval()
    rows, loader_receipt = load_text_rows(
        args.records, args.pieces, args.text_data, tokenizer=engine.tokenizer)
    selected, selection, mask_receipt = prepare_held_test_windows(
        engine, rows, tokens=args.tokens, prefix_tokens=args.prefix_tokens,
        target_tokens=args.target_tokens, held_documents=args.held_documents,
        mask_system_prompt=args.mask_system_prompt)
    if not selected:
        raise ValueError('no held first/last windows')
    thresholds = {
        'min_argmax_agreement': args.min_argmax_agreement,
        'max_kl_nats': args.max_kl_nats,
        'max_quality_ce_gap_nats': args.max_quality_ce_gap_nats,
    }
    report_metrics = evaluate_windows(
        engine.backbone, engine.heads, selected, steps=args.steps,
        logit_chunk_tokens=args.logit_chunk_tokens,
        readout_chunk_tokens=args.readout_chunk_tokens, thresholds=thresholds)
    if {str(path.resolve()): sha(path) for path in input_paths} != pins:
        raise ValueError('an input changed during evaluation')
    from ..train.trajectory_state import weights_digest
    from ..data.text_corpus import tokenizer_fingerprint
    loaded_weights_sha256 = weights_digest(engine.backbone.state_dict(), engine.heads.state_dict())
    tokenizer_sha256 = tokenizer_fingerprint(engine.tokenizer)
    if {name: sha(path) for name, path in code_paths.items()} != code_hashes:
        raise ValueError('evaluator package code changed during evaluation')
    report = {
        'schema': 'natlang.self-feedback-channel-diagnostic/1',
        'checkpoint': str((args.checkpoint or args.heads).resolve()),
        'checkpoint_sha256': pins[str((args.checkpoint or args.heads).resolve())],
        'checkpoint_step': checkpoint_step,
        'loaded_engine_weights_sha256': loaded_weights_sha256,
        'tokenizer_sha256': tokenizer_sha256,
        'evaluator_package_code_sha256': code_hashes,
        'system_prompt_masking': mask_receipt,
        'pins': pins,
        'loader': loader_receipt,
        'held_selection': selection,
        **report_metrics,
        'requested_steps': args.steps,
        'whole_span_gold_tail_gate': False,
        'foundation_qualified': False,
        'runtime_qualified': False,
        'task_qualified': False,
        'scope': 'Compare actual full-depth Neuralese feedback with ordinary embeddings on the same projection-generated continuation. Quality gap compares ordinary-text CE for projection-generated tokens with crisp greedy tokens on their own ordinary histories. This does not score gold continuation accuracy or certify runtime/task behavior.',
    }
    args.out.mkdir(parents=True)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
