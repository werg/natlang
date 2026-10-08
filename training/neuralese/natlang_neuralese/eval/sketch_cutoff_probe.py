"""How much next-token signal the sketch head can reach at each shallow cutoff (frozen backbone).

The sketch map is an early exit: the shallow state at the cutoff, one MLP correction, the frozen final norm and the
output head, whose argmax token embedding feeds the next pass. Sketch-only rollout training cannot exceed what that
head can decode at its cutoff. This fits the same head architecture (``CausalFeedbackProjection``, zero-initialized
correction) on frozen states after each requested layer and reports held top-1 accuracy, CE, and agreement with the
full model's own greedy prediction (the crisp control the sketch stands in for), before and after fitting.

    python -m natlang_neuralese.eval.sketch_cutoff_probe --heads H --records R --text-data T --out DIR \\
        --layers 4 6 8 12
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ..model.causal_feedback import CausalFeedbackProjection
from ..model.lfm2_port import PortCache
from ..serve import load_engine
from ..train.text_warmup import document_windows, load_text_rows, select_held_document_windows


@torch.no_grad()
def layer_states(backbone, ids, layers):
    """States after each requested number of layers, and the full model's greedy predictions, for one window."""
    h, cache, start, states = backbone.embed(ids), PortCache.empty(backbone.num_layers), 0, {}
    for layer in sorted(set(layers)) + [backbone.num_layers]:
        h, cache = backbone.run_layers(h, range(start, layer), cache)
        start = layer
        if layer in layers:
            states[layer] = h[0, :-1].to(torch.bfloat16).cpu()
    greedy = torch.cat([backbone.logits(chunk[None]).argmax(-1)[0] for chunk in h[0, :-1].split(512)]).cpu()
    return states, greedy


def head_scores(head, states, targets, greedy, device, chunk=1024):
    total = {'tokens': 0, 'ce': 0., 'gold_top1': 0., 'full_model_agreement': 0.}
    with torch.no_grad():
        for s, t, g in zip(states.split(chunk), targets.split(chunk), greedy.split(chunk)):
            logits = head.logits(s.to(device)).float()
            prediction = logits.argmax(-1).cpu()
            total['ce'] += float(F.cross_entropy(logits, t.to(device), reduction='sum'))
            total['gold_top1'] += float((prediction == t).sum())
            total['full_model_agreement'] += float((prediction == g).sum())
            total['tokens'] += t.numel()
    return {k: (v / total['tokens'] if k != 'tokens' else v) for k, v in total.items()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('heads', 'records', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('pieces', 'text-data'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--layers', type=int, nargs='+', default=[4, 6, 8, 12])
    p.add_argument('--tokens', type=int, default=2048)
    p.add_argument('--prefix-tokens', type=int, default=32)
    p.add_argument('--train-windows', type=int, default=96)
    p.add_argument('--held-documents', type=int, default=16)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--batch-positions', type=int, default=2048)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cuda')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    if a.out.exists():
        raise ValueError('fresh output directory required')
    random.seed(a.seed); torch.manual_seed(a.seed)
    engine = load_engine(heads_checkpoint=str(a.heads), device=a.device)
    backbone = engine.backbone; backbone.eval()
    for q in backbone.parameters():
        q.requires_grad_(False)
    rows, receipt = load_text_rows(a.records, a.pieces, a.text_data, tokenizer=engine.tokenizer)
    split_windows = {'train': [], 'test': []}
    for row in rows:
        tokens = row['token_ids'] if 'token_ids' in row else engine._tokens(row['text'])
        for window in document_windows(tokens, open_id=backbone.controls.open_id, close_id=backbone.controls.close_id,
                                       tokens=a.tokens, prefix_tokens=a.prefix_tokens):
            split_windows['train' if row['split'] == 'train' else 'test'].append(
                {**window, 'document': hashlib.sha256(row['text'].encode()).hexdigest(), 'groups': row['source_groups']})
    held, selection = select_held_document_windows(split_windows['test'], a.held_documents)
    train = random.sample(split_windows['train'], min(a.train_windows, len(split_windows['train'])))

    def collect(windows):
        per_layer, targets, greedy = {k: [] for k in a.layers}, [], []
        for window in windows:
            ids = torch.tensor([window['ids']], device=a.device)
            states, predicted = layer_states(backbone, ids, a.layers)
            for k in a.layers:
                per_layer[k].append(states[k])
            targets.append(ids[0, 1:].cpu()); greedy.append(predicted)
        return {k: torch.cat(v) for k, v in per_layer.items()}, torch.cat(targets), torch.cat(greedy)

    started = time.perf_counter()
    train_states, train_targets, train_greedy = collect(train)
    held_states, held_targets, held_greedy = collect(held)
    collect_seconds = time.perf_counter() - started
    full_model = {'held_gold_top1': float((held_greedy == held_targets).float().mean()),
                  'held_tokens': held_targets.numel(), 'train_tokens': train_targets.numel()}
    print(json.dumps({'event': 'collected', 'seconds': collect_seconds, **full_model}), flush=True)
    results = {}
    for layer in a.layers:
        head = CausalFeedbackProjection(backbone).to(a.device)
        before = head_scores(head, held_states[layer], held_targets, held_greedy, a.device)
        trainable = [q for q in head.parameters() if q.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.)
        states, count = train_states[layer], train_targets.numel()
        curve = []
        for step in range(a.steps):
            index = torch.randint(count, (a.batch_positions,))
            logits = head.logits(states[index].to(a.device)).float()
            loss = F.cross_entropy(logits, train_targets[index].to(a.device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            if (step + 1) % max(1, a.steps // 6) == 0:
                curve.append({'step': step + 1, 'held': head_scores(head, held_states[layer], held_targets,
                                                                    held_greedy, a.device)})
                print(json.dumps({'layer': layer, **curve[-1]}), flush=True)
        results[str(layer)] = {'logit_lens': before, 'fitted': curve[-1]['held'] if curve else before, 'curve': curve}
        del head, optimizer
    report = {'schema': 'natlang.sketch-cutoff-probe/1', 'heads': str(a.heads), 'inputs': receipt,
              'held_selection': selection, 'layers': a.layers, 'num_layers': backbone.num_layers,
              'options': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
              'full_model': full_model, 'results': results,
              'scope': 'frozen backbone; same head architecture as the sketch; diagnostic, not qualification'}
    a.out.mkdir(parents=True)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'event': 'done', 'results': {k: v['fitted'] for k, v in results.items()}}), flush=True)


if __name__ == '__main__':
    main()
