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
from ..train.text_warmup import (
    chunked_readout, document_windows, gold_completion, load_text_rows,
    select_held_document_windows,
)


@torch.no_grad()
def projected_history_metrics(backbone, heads, prefix, span):
    if prefix.ndim != 2 or span.ndim != 2 or not prefix.shape[1] or not span.shape[1]:
        raise ValueError('nonempty aligned prefix and targets required')
    if prefix.shape[0] != span.shape[0]:
        raise ValueError('prefix and target batches must match')
    if heads.read_markers:
        raise ValueError('token-aligned history diagnostic requires the raw read profile')
    completed = gold_completion(backbone, heads, prefix, span)
    top = completed['top']
    projected = heads.content(torch.zeros_like(top), top)
    reference = heads.content.reference(top)
    gold = backbone.embed(span)
    _, _, live_tokens, _, _ = chunked_readout(
        backbone, top, span, backbone.controls.close_id, gradients=False)
    live_greedy = backbone.embed(live_tokens)
    prefix_embeddings = backbone.embed(prefix)
    scores = {}
    for name, payload in (('gold', gold), ('live_greedy', live_greedy), ('reference', reference),
                          ('full_projection', projected), ('sketch_projection', completed['sketches'])):
        # Projection at j predicts span[j]; it is used as an input only while
        # predicting span[j+1]. The last prediction is never supplied as input.
        history = heads.read_embeddings(backbone, payload[:, :-1])
        if history.shape != gold[:, :-1].shape:
            raise ValueError('read transport changed aligned history geometry')
        out = backbone.forward_embeds(torch.cat((prefix_embeddings, history), 1), logits=False)
        states = out['h_final'][:, prefix.shape[1]-1:]
        _, _, prediction, _, losses = chunked_readout(
            backbone, states, span, backbone.controls.close_id, gradients=False)
        scores[name] = (prediction, losses)
        del out, states
    baseline_prediction, baseline_losses = scores['gold']
    rows = {}
    for name, (prediction, losses) in scores.items():
        rows[name] = {}
        for region, start in (('whole', 0), ('last256', max(0, span.shape[1]-256))):
            actual = losses[:, start:]
            rows[name][region] = {
                'tokens': actual.numel(), 'ce': float(actual.mean()),
                'ce_delta_from_gold': float((actual-baseline_losses[:, start:]).mean()),
                'gold_accuracy': float((prediction[:, start:]==span[:, start:]).float().mean()),
                'argmax_agreement_with_gold': float((prediction[:, start:]==baseline_prediction[:, start:]).float().mean()),
            }
    rows['producer_controls'] = {
        'reference_equal_live_greedy': bool(torch.equal(reference, live_greedy)),
        'reference_vs_live_greedy_mse': float((reference.float()-live_greedy.float()).square().mean()),
        'live_greedy_gold_accuracy': float((live_tokens==span).float().mean()),
        'live_greedy_gold_accuracy_last256': float((live_tokens[:, -256:]==span[:, -256:]).float().mean()),
    }
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


if __name__ == '__main__':
    main()
