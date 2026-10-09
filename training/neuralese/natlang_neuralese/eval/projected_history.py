"""Measure causal projected-history utility through the production read interface.

This diagnostic does not issue foundation, runtime or task certificates. Gold
history produces next-token payloads; those payloads replace the corresponding
previous-token inputs of a second full-stack consumer. No future gold tokens
are supplied to a projected consumer. Whole-window and last-256 scores remain
separate so easy context cannot hide a difficult tail.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from ..serve import load_engine
from ..train.output_embedding_projection import sha
from ..train.execution import full_depth_projected_feedback_step
from ..train.text_warmup import (
    chunked_readout, document_windows, gold_completion, load_text_rows,
    select_held_document_windows,
)


@torch.no_grad()
def projected_history_metrics(backbone, heads, prefix, span, *, completion=None, live_tokens=None,
                              consumers=None, per_window=False):
    if prefix.ndim != 2 or span.ndim != 2 or not prefix.shape[1] or not span.shape[1]:
        raise ValueError('nonempty aligned prefix and targets required')
    if prefix.shape[0] != span.shape[0]:
        raise ValueError('prefix and target batches must match')
    if heads.read_markers:
        raise ValueError('token-aligned history diagnostic requires the raw read profile')
    completed = gold_completion(backbone, heads, prefix, span) if completion is None else completion
    if not isinstance(completed, dict) or 'top' not in completed:
        raise ValueError('a reusable completion must contain top states')
    top = completed['top']
    if live_tokens is None:
        _, _, live_tokens, _, _ = chunked_readout(
            backbone, top, span, backbone.controls.close_id, gradients=False)
    if live_tokens.shape != span.shape:
        raise ValueError('reused live predictions must align with target tokens')
    available = {'gold','live_greedy','reference','full_projection'}
    if 'sketches' in completed:
        available.add('sketch_projection')
    selected = tuple(('gold','live_greedy','reference','full_projection','sketch_projection')
                     if consumers is None and 'sketch_projection' in available else
                     ('gold','live_greedy','reference','full_projection') if consumers is None else consumers)
    if not selected or len(set(selected)) != len(selected) or any(name not in available for name in selected):
        raise ValueError('consumers must select distinct available history payloads')
    if 'live_greedy' not in selected:
        raise ValueError('matched history consumers require live_greedy as the comparison control')
    all_payloads={}
    for name in selected:
        if name=='gold':all_payloads[name]=backbone.embed(span)
        elif name=='live_greedy':all_payloads[name]=backbone.embed(live_tokens)
        elif name=='reference':all_payloads[name]=heads.content.reference(top)
        elif name=='full_projection':all_payloads[name]=heads.content(torch.zeros_like(top),top)
        else:all_payloads[name]=completed['sketches']
    prefix_embeddings = backbone.embed(prefix)
    scores = {}
    history_inputs = {}
    expected_history_shape=next(iter(all_payloads.values()))[:, :-1].shape
    for name in selected:
        payload = all_payloads[name]
        # Projection at j predicts span[j]; it is used as an input only while
        # predicting span[j+1]. The last prediction is never supplied as input.
        history = heads.read_embeddings(backbone, payload[:, :-1])
        if history.shape != expected_history_shape:
            raise ValueError('read transport changed aligned history geometry')
        out = backbone.forward_embeds(torch.cat((prefix_embeddings, history), 1), logits=False)
        states = out['h_final'][:, prefix.shape[1]-1:]
        _, _, prediction, _, losses = chunked_readout(
            backbone, states, span, backbone.controls.close_id, gradients=False)
        scores[name] = (prediction, losses)
        history_inputs[name] = history
        del out, states
    baseline_prediction, baseline_losses = scores.get('gold', (None, None))
    crisp_prediction, crisp_losses = scores['live_greedy']
    crisp_history = history_inputs['live_greedy']
    rows = {}
    for name, (prediction, losses) in scores.items():
        rows[name] = {}
        for region, start in (('whole', 0), ('last256', max(0, span.shape[1]-256))):
            actual = losses[:, start:]
            region_result = {
                'tokens': actual.numel(), 'ce': float(actual.mean()),
                'gold_accuracy': float((prediction[:, start:]==span[:, start:]).float().mean()),
                'ce_delta_from_live_greedy': float((actual-crisp_losses[:, start:]).mean()),
                'argmax_agreement_with_live_greedy': float((prediction[:, start:]==crisp_prediction[:, start:]).float().mean()),
            }
            if baseline_losses is not None:
                region_result.update(
                    ce_delta_from_gold=float((actual-baseline_losses[:, start:]).mean()),
                    argmax_agreement_with_gold=float((prediction[:, start:]==baseline_prediction[:, start:]).float().mean()))
            history_start=max(0, start-1)
            history=history_inputs[name][:, history_start:]
            control_history=crisp_history[:, history_start:]
            region_result['history_positions']=history.numel() // max(1,history.shape[-1])
            if history.shape[1]:
                region_result['read_history_mse_vs_live_greedy']=float((history.float()-control_history.float()).square().mean())
            rows[name][region] = region_result
    if 'reference' in all_payloads and 'live_greedy' in all_payloads:
        reference=all_payloads['reference'];live_greedy=all_payloads['live_greedy']
        rows['producer_controls'] = {
            'reference_equal_live_greedy': bool(torch.equal(reference, live_greedy)),
            'reference_vs_live_greedy_mse': float((reference.float()-live_greedy.float()).square().mean()),
            'live_greedy_gold_accuracy': float((live_tokens==span).float().mean()),
            'live_greedy_gold_accuracy_last256': float((live_tokens[:, -256:]==span[:, -256:]).float().mean()),
        }
    if per_window:
        rows['windows'] = []
        for index in range(span.shape[0]):
            item = {}
            for name, (prediction, losses) in scores.items():
                item[name] = {}
                for region, start in (('whole', 0), ('last256', max(0, span.shape[1]-256))):
                    actual=losses[index, start:]
                    result={'tokens':actual.numel(),'ce':float(actual.mean()),
                        'gold_accuracy':float((prediction[index, start:]==span[index, start:]).float().mean()),
                        'ce_delta_from_live_greedy':float((actual-crisp_losses[index, start:]).mean()),
                        'argmax_agreement_with_live_greedy':float((prediction[index, start:]==crisp_prediction[index, start:]).float().mean())}
                    if baseline_losses is not None:
                        result.update(ce_delta_from_gold=float((actual-baseline_losses[index, start:]).mean()),
                            argmax_agreement_with_gold=float((prediction[index, start:]==baseline_prediction[index, start:]).float().mean()))
                    history_start=max(0,start-1)
                    history=history_inputs[name][index,history_start:]
                    control_history=crisp_history[index,history_start:]
                    result['history_positions']=history.shape[0]
                    if history.shape[0]:
                        result['read_history_mse_vs_live_greedy']=float((history.float()-control_history.float()).square().mean())
                    item[name][region]=result
            rows['windows'].append(item)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('heads', 'records', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('pieces', 'text-data'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--prefix-tokens', type=int, default=32)
    parser.add_argument('--held-documents', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.out.exists():
        raise ValueError('fresh diagnostic output required')
    # Reject mutable exports that change during this diagnostic.
    checkpoint_sha = sha(args.heads)
    engine = load_engine(heads_checkpoint=str(args.heads), device=args.device)
    engine.backbone.eval(); engine.heads.eval()
    rows, receipt = load_text_rows(args.records, args.pieces, args.text_data, tokenizer=engine.tokenizer)
    windows = []
    for row in rows:
        if row['split'] != 'test':
            continue
        tokens = row['token_ids'] if 'token_ids' in row else engine._tokens(row['text'])
        for window in document_windows(tokens, open_id=engine.backbone.controls.open_id,
                close_id=engine.backbone.controls.close_id, tokens=args.tokens,
                prefix_tokens=args.prefix_tokens):
            windows.append({**window, 'document':hashlib.sha256(row['text'].encode()).hexdigest(),
                            'groups':row['source_groups']})
    held, selection = select_held_document_windows(windows, args.held_documents)
    if not held:
        raise ValueError('no held diagnostic windows')
    result = []
    for window in held:
        ids = torch.tensor([window['ids']], device=args.device)
        scores = projected_history_metrics(engine.backbone, engine.heads,
                    ids[:, :window['prefix']], ids[:, window['prefix']:])
        result.append({'document_sha256':window['document'], 'source_groups':window['groups'],
                       'offset':window['offset'], 'scores':scores})
    if sha(args.heads) != checkpoint_sha:
        raise ValueError('checkpoint changed during diagnostic')
    report = {'schema':'natlang.projected-history-utility/1', 'checkpoint_sha256':checkpoint_sha,
              'inputs':receipt, 'held_selection':selection, 'rows':result,
              'production_read_interface':True, 'future_gold_inputs':False,
              'foundation_qualified':False, 'runtime_qualified':False, 'task_qualified':False,
              'scope':'Gold-history projected previous-token inputs; not autonomous writer generation or task success. Compare full_projection to live_greedy for the crisp prediction control; gold is a teacher-forced history control.'}
    args.out.mkdir(parents=True)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')



AUTOREGRESSIVE_KINDS = ('ar_greedy', 'ar_projection', 'ar_sketch')


def _gold_reference_survival(prediction, losses, span):
    """Score gold targets only while the greedy token history still matches.

    At target position ``t``, the history is still the reference history when
    every generated token before ``t`` matched gold. The first mismatching
    target itself is therefore included: its input context is still exact.
    """
    if prediction.shape != span.shape or losses.shape != span.shape or span.ndim != 2:
        raise ValueError('prediction, loss and gold target shapes must match')
    matches = prediction.eq(span)
    survival = torch.cat((torch.ones_like(matches[:, :1]), matches[:, :-1]), dim=1).cumprod(dim=1).bool()
    surviving_losses = losses[survival]
    surviving_correct = matches[survival]
    first_mismatch = []
    for row in matches:
        mismatch = torch.nonzero(~row, as_tuple=False)
        first_mismatch.append(int(mismatch[0, 0]) if mismatch.numel() else None)
    count = int(survival.sum())
    return {
        'first_token_ce': float(losses[:, 0].mean()),
        'first_token_accuracy': float(matches[:, 0].float().mean()),
        'first_divergence_index_by_window': first_mismatch,
        'exact_prefix_survival_tokens': count,
        'exact_prefix_survival_ce': float(surviving_losses.mean()) if count else None,
        'exact_prefix_survival_accuracy': float(surviving_correct.float().mean()) if count else None,
        'gold_reference_scope': 'gold next-token targets are context-valid through the first mismatching prediction; later targets are scored on divergent histories',
    }


def _token_id_window_fingerprints(prefix, span):
    """Hash each exact token window after one combined device-to-host copy."""
    if prefix.ndim != 2 or span.ndim != 2 or prefix.shape[0] != span.shape[0]:
        raise ValueError('prefix and target token batches must be aligned rank-two tensors')
    prefix_width = prefix.shape[1]
    joined = torch.cat((prefix, span), dim=1).detach().to(device='cpu').contiguous()
    prefix_header = f'natlang.ar-prefix-token-ids/1:{prefix.dtype}:{prefix_width}:'.encode()
    target_header = f'natlang.ar-target-token-ids/1:{span.dtype}:{span.shape[1]}:'.encode()
    output = []
    for row in joined:
        prefix_bytes = row[:prefix_width].numpy().tobytes()
        target_bytes = row[prefix_width:].numpy().tobytes()
        output.append({
            'prefix_token_count': prefix_width,
            'target_token_count': span.shape[1],
            'prefix_token_ids_sha256': hashlib.sha256(prefix_header+prefix_bytes).hexdigest(),
            'target_token_ids_sha256': hashlib.sha256(target_header+target_bytes).hexdigest(),
        })
    return output


def _first_divergence_details(backbone, name, survival, generated_tokens, predictions,
                             token_diagnostics, payloads, span, control_divergences):
    """Compact evidence at the first bad emitted token, without another model pass."""
    rows = []
    for row_index, divergence in enumerate(survival['first_divergence_index_by_window']):
        if divergence is None:
            rows.append(None)
            continue
        gold_embedding = backbone.embed(span[row_index:row_index+1, divergence]).float().reshape(-1)
        emitted_payload = payloads[name][row_index:row_index+1, divergence].float().reshape(-1)
        gold_score, top_score, top_margin = (value[row_index, divergence] for value in token_diagnostics[name])
        greedy_token = (generated_tokens['ar_greedy'][row_index, divergence]
                        if 'ar_greedy' in generated_tokens else span.new_tensor(-1))
        greedy_divergence = control_divergences[row_index] if row_index < len(control_divergences) else None
        control_gold_prefix_valid = greedy_divergence is None or divergence <= greedy_divergence
        if divergence > 0:
            feedback_payload = payloads[name][row_index:row_index+1, divergence-1].float().reshape(-1)
            gold_feedback_embedding = backbone.embed(span[row_index:row_index+1, divergence-1]).float().reshape(-1)
            feedback_values = (
                feedback_payload.norm(), gold_feedback_embedding.norm(),
                (feedback_payload-gold_feedback_embedding).norm(),
                torch.nn.functional.cosine_similarity(feedback_payload[None, :],
                                                       gold_feedback_embedding[None, :]).reshape(()),
            )
        else:
            feedback_payload = gold_feedback_embedding = None
            feedback_values = (span.new_zeros((), dtype=torch.float32),) * 4
        values = torch.stack((
            span[row_index, divergence].float(), generated_tokens[name][row_index, divergence].float(),
            predictions[row_index, divergence].float(), greedy_token.float(),
            (top_score-gold_score).float(), top_margin.float(), emitted_payload.norm(), gold_embedding.norm(),
            (emitted_payload-gold_embedding).norm(),
            torch.nn.functional.cosine_similarity(emitted_payload[None, :], gold_embedding[None, :]).reshape(()),
            *feedback_values,
        )).detach().cpu().tolist()
        rows.append({
            'target_index': divergence,
            'gold_token_id': int(values[0]), 'generated_token_id': int(values[1]),
            'rescored_token_id': int(values[2]),
            'ar_greedy_control_token_id': int(values[3]) if values[3] >= 0 else None,
            'ar_greedy_control_first_divergence_index': greedy_divergence,
            'ar_greedy_control_gold_prefix_valid_at_target': control_gold_prefix_valid,
            'ar_greedy_control_comparability_scope': 'greedy token history still matches gold through prior targets'
                if control_gold_prefix_valid else 'greedy token history had already diverged before this target',
            'generated_logit_minus_gold_logit': values[4], 'top1_logit_margin': values[5],
            'emitted_payload_index': divergence,
            'emitted_payload_l2_norm': values[6], 'gold_embedding_l2_norm': values[7],
            'emitted_payload_minus_gold_embedding_l2_norm': values[8], 'emitted_payload_gold_embedding_cosine': values[9],
            'preceding_feedback_index': divergence-1 if divergence > 0 else None,
            'preceding_feedback_l2_norm': values[10] if divergence > 0 else None,
            'gold_preceding_embedding_l2_norm': values[11] if divergence > 0 else None,
            'preceding_feedback_minus_gold_embedding_l2_norm': values[12] if divergence > 0 else None,
            'preceding_feedback_gold_embedding_cosine': values[13] if divergence > 0 else None,
        })
    return rows


@torch.no_grad()
def autoregressive_payloads(backbone, heads, prefix, steps, kinds=AUTOREGRESSIVE_KINDS, *,
                            return_generated_tokens=False, return_token_diagnostics=False,
                            gold_tokens=None):
    """Self-fed rollouts from one prefix, one position at a time.

    ar_greedy feeds back the crisp greedy token; ar_projection the full-depth projection of the top state (the
    autoregressive Neuralese initialization); ar_sketch the shallow sketch of the shallow state, which is what
    sequence passes approximate in parallel (depth N is exact for the first N positions). Each payload has
    ``steps`` positions aligned with the targets they predict, like the matched-history payloads.
    """
    from ..train.execution import prefill_write_context
    if heads.read_markers:
        raise ValueError('autoregressive history controls require the raw read profile')
    prefix_embeddings = backbone.embed(prefix)
    if return_token_diagnostics and (not return_generated_tokens or gold_tokens is None or
            gold_tokens.ndim != 2 or gold_tokens.shape[0] != prefix.shape[0] or gold_tokens.shape[1] < steps):
        raise ValueError('token diagnostics require generated tokens and aligned gold targets')
    payloads = {};generated_tokens={};token_diagnostics={}
    for kind in kinds:
        if kind not in AUTOREGRESSIVE_KINDS:
            raise ValueError(f'unknown autoregressive control {kind}')
        # A separate prefill per rollout: no-grad caches grow in place.
        context = prefill_write_context(backbone, heads, prefix_embeddings)
        cache, values = context.cache, [];tokens=[];diagnostics=[]
        state, top = context.state[:, None], context.top[:, None]
        for position in range(steps):
            next_top = None
            # The decoded token of each rollout: the sketch head's own argmax (its straight-through choice), the
            # crisp greedy token, and for the projection the full model's greedy reading of the emitting state.
            if kind == 'ar_sketch':
                value = heads.feedback(state)
                decoded = getattr(heads.feedback, 'logits', None)  # latent sketches have no token choice
                scores = decoded(state) if return_generated_tokens and decoded else None
                token = scores.argmax(-1) if scores is not None else None
            elif kind == 'ar_greedy':
                scores = backbone.logits(top)
                token = scores.argmax(-1)
                value = backbone.embed(token)
            else:
                if position < steps - 1:
                    value, next_top, next_cache = full_depth_projected_feedback_step(
                        backbone, heads, top, cache)
                else:
                    value = heads.content(torch.zeros_like(top), top)
                scores = backbone.logits(top) if return_generated_tokens else None
                token = scores.argmax(-1) if scores is not None else None
            if token is not None:
                tokens.append(token)
                if return_token_diagnostics:
                    top_values = scores.topk(min(2, scores.shape[-1]), dim=-1).values
                    gold_score = scores.gather(-1, gold_tokens[:, position:position+1])
                    diagnostics.append((gold_score, top_values[..., :1],
                                        top_values[..., :1] - top_values[..., -1:]))
            values.append(value)
            if position == steps - 1:
                break
            if kind == 'ar_projection':
                top, cache = next_top[:, None], next_cache
                continue  # the shared full-depth transition above already advanced the cache
            history = heads.read_embeddings(backbone, value)
            if kind == 'ar_sketch':
                state, cache = backbone.run_layers(history, range(0, heads.cutoff), cache)
            else:
                top, cache = backbone.run_layers(history, range(backbone.num_layers), cache)
        payloads[kind] = torch.cat(values, 1)
        if tokens:
            generated_tokens[kind] = torch.cat(tokens, 1)
        if diagnostics:
            token_diagnostics[kind] = tuple(torch.cat([item[i] for item in diagnostics], 1) for i in range(3))
        del cache
    if return_token_diagnostics:
        return payloads, generated_tokens, token_diagnostics
    return (payloads,generated_tokens) if return_generated_tokens else payloads


@torch.no_grad()
def autoregressive_history_metrics(backbone, heads, prefix, span, *, steps=256, kinds=AUTOREGRESSIVE_KINDS):
    """Score self-fed histories with the same full-stack consumer as the matched controls.

    Gold-reference CE over the first ``steps`` targets given each generated history, against the teacher-forced
    gold history and against the crisp autoregressive (ar_greedy) run. Once a generated token diverges from gold,
    subsequent gold tokens no longer describe the generated prefix; the reported survival subset isolates scores
    whose prior decoded history (each rollout's own token choice) still exactly matches gold. These are token-fidelity diagnostics, not task success.
    """
    steps = min(int(steps), span.shape[1])
    if steps < 2:
        raise ValueError('autoregressive controls need at least two target positions')
    span = span[:, :steps]
    input_fingerprints = _token_id_window_fingerprints(prefix, span)
    payloads,generated_tokens,token_diagnostics = autoregressive_payloads(
        backbone, heads, prefix, steps, kinds, return_generated_tokens=True,
        return_token_diagnostics=True, gold_tokens=span)
    payloads = {'gold': backbone.embed(span), **payloads}
    prefix_embeddings = backbone.embed(prefix)
    scores = {}
    for name, payload in payloads.items():
        history = heads.read_embeddings(backbone, payload[:, :-1])
        out = backbone.forward_embeds(torch.cat((prefix_embeddings, history), 1), logits=False)
        states = out['h_final'][:, prefix.shape[1]-1:]
        _, _, prediction, _, losses = chunked_readout(
            backbone, states, span, backbone.controls.close_id, gradients=False)
        scores[name] = (prediction, losses)
        del out, states
    gold_prediction, gold_losses = scores['gold']
    crisp_prediction, crisp_losses = scores.get('ar_greedy', (None, None))
    crisp_divergences = (_gold_reference_survival(generated_tokens['ar_greedy'], crisp_losses, span)
                         ['first_divergence_index_by_window'] if crisp_losses is not None else [])
    rows = {}
    for name, (prediction, losses) in scores.items():
        row = {'tokens': losses.numel(), 'ce': float(losses.mean()),
               'gold_accuracy': float((prediction == span).float().mean()),
               'ce_delta_from_gold': float((losses - gold_losses).mean()),
               'argmax_agreement_with_gold': float((prediction == gold_prediction).float().mean())}
        if crisp_losses is not None:
            row['ce_delta_from_ar_greedy'] = float((losses - crisp_losses).mean())
            row['argmax_agreement_with_ar_greedy'] = float((prediction == crisp_prediction).float().mean())
        if name in generated_tokens:
            actual_tokens = generated_tokens[name]
            survival=_gold_reference_survival(actual_tokens, losses, span)
            survival['generated_vs_rescored_prediction_agreement']=float((actual_tokens==prediction).float().mean())
            survival['first_divergence_details_by_window'] = _first_divergence_details(
                backbone, name, survival, generated_tokens, prediction, token_diagnostics, payloads, span,
                crisp_divergences)
            row['gold_reference_after_divergence'] = survival
        rows[name] = row
    return {'schema': 'natlang.autoregressive-history-controls/2', 'steps': steps, 'windows': span.shape[0],
            'input_token_id_fingerprints': input_fingerprints,
            'consumer': 'full stack over prefix + generated history, scored on gold-reference targets',
            'gold_reference_interpretation': 'after a rollout diverges, later gold tokens are not asserted to be valid next-token targets for that generated context',
            'scores': rows}


if __name__ == '__main__':
    main()
