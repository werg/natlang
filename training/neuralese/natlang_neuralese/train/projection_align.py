"""The NeuraleseProjection at end-to-end entry, on one exact direct export: made, measured and gated (owner decisions
2026-10-13).

value(h) = E[chosen(h)] + C(h) (model/neuralese_projection.py) must sit on the input token-embedding manifold when a
lineage enters the end-to-end stage: the nearest embedding of the value is the token the line's readout chooses there,
at high cosine, and the backbone reads a span of values as it reads the chosen tokens. This module is the one
implementation of that entry step and of its exact verifier:

1. ``init`` (the first step of the e2e handoff): the export's NeuraleseProjection with C = 0 (or ``--continue-from``
   an aligned P of this export), written as ``aligned-projection.pt`` bound to the export's manifest SHA-256.
2. ``measure``: the export's frozen backbone runs its line's own rendered text rows (the union text data, test split);
   at every supervised position and a sample of prompt positions: cosine, norm ratio, nearest-token accuracy and rank
   of the chosen token, the same against E[gold], by position kind and confidence bucket; and the functional KL: the
   backbone's next-token distributions on a held span read as a block of values instead of E[chosen] embeddings.
3. ``verify`` (``e2e_acceptance.projection_on_manifold``): the gate's thresholds on those measurements.

A generic learned map h -> embedding was measured first and could not reach the manifold (the chosen token is an
argmax over the vocabulary): NEURALESE_DIRECT_V1.md "Projection alignment fit (2026-10-13)". After entry there is no
alignment loss or anchor: C moves under the consumer's gradient.

CLI: ``python -m natlang_neuralese.train.projection_align init --artifact EXPORT --base SNAPSHOT --text-data JSONL
--out DIR`` (exit 0 only when the gate passes); ``verify --aligned-projection FILE`` re-measures a saved P.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

SCHEMA = 'natlang.neuralese-projection-alignment-report/1'
# Owner 2026-10-13: near-identity on the manifold. The functional-KL bound is the measured achievable one
# (plans/neuralese/NEURALESE_DIRECT_V1.md "The Neuralese value").
DEFAULT_THRESHOLDS = {'min_nearest_chosen': 0.99, 'min_cos_chosen': 0.98, 'max_functional_kl': 0.05}


def text_rows(path, *, held_rows: int, max_tokens: int):
    """The first ``held_rows`` usable test rows of a line's text data JSONL, in file order: token ids (first
    ``max_tokens``) and the supervised suffix start."""
    held = []
    with open(path) as source:
        for line in source:
            row = json.loads(line)
            ids, start = row.get('token_ids'), int(row.get('supervised_suffix_start') or 0)
            if row.get('split') != 'test' or not isinstance(ids, list) or not 1 <= start < max_tokens - 8 or \
                    len(ids) - start < 2:
                continue
            held.append({'id': row.get('id'), 'ids': ids[:max_tokens], 'start': start})
            if len(held) >= held_rows:
                break
    return held


@torch.no_grad()
def collect(backbone, rows, *, prompt_keep: float = 0.25, seed: int = 0, chunk: int = 512):
    """Per-position tensors over ``rows``: h (final hidden state, bf16, detached), chosen, gold, pmax, kind (1: a
    supervised suffix position, 0: a sampled prompt position), row, position."""
    device = backbone.embedding_weight.device
    generator = torch.Generator().manual_seed(seed)
    out = {k: [] for k in ('h', 'chosen', 'gold', 'pmax', 'kind', 'row', 'position')}
    for index, row in enumerate(rows):
        ids = torch.tensor([row['ids']], device=device)
        hidden = backbone.forward_embeds(backbone.embed(ids), logits=False)['h_final'][0, :-1]
        chosen, pmax = [], []
        for s in range(0, hidden.shape[0], chunk):
            logits = backbone.logits(hidden[None, s:s + chunk])[0].float()
            chosen.append(logits.argmax(-1))
            pmax.append(torch.softmax(logits, -1).amax(-1))
            del logits
        chosen, pmax = torch.cat(chosen), torch.cat(pmax)
        position = torch.arange(hidden.shape[0])
        kind = (position + 1 >= row['start']).long()
        keep = ((kind == 1) | (torch.rand(hidden.shape[0], generator=generator) < prompt_keep)) & (position >= 1)
        where = keep.nonzero()[:, 0]
        on = where.to(device)
        out['h'].append(hidden[on].to(torch.bfloat16).cpu())
        out['chosen'].append(chosen[on].cpu())
        out['gold'].append(ids[0, 1:][on].cpu())
        out['pmax'].append(pmax[on].float().cpu())
        out['kind'].append(kind[where])
        out['row'].append(torch.full((len(where),), index))
        out['position'].append(where)
    return {k: torch.cat(v) for k, v in out.items()}


def _bucket_masks(states):
    pmax, kind = states['pmax'], states['kind']
    return {'all': torch.ones_like(kind, dtype=torch.bool), 'suffix': kind == 1, 'prompt': kind == 0,
            'p>=0.9': pmax >= .9, 'p0.5-0.9': (pmax >= .5) & (pmax < .9), 'p<0.5': pmax < .5}


def input_embedding_table(backbone, *, chunk: int = 8192) -> torch.Tensor:
    """What a consumer reads for each token id (``backbone.embed``, control rows included), [vocab, width] FP32."""
    vocab = int(backbone.embedding_weight.shape[0])
    device = backbone.embedding_weight.device
    with torch.no_grad():
        return torch.cat([backbone.embed(torch.arange(s, min(s + chunk, vocab), device=device)[None])[0].float()
                          for s in range(0, vocab, chunk)])


@torch.no_grad()
def measure_geometry(projection, states, table, *, chunk: int = 1024):
    """Held geometry of the values against E[chosen] (and E[gold]) per bucket: means of cosine, norm ratio,
    nearest-token hit and relative squared error; the 10th percentile of cosine; mean rank of the chosen token."""
    from ..model.neuralese_projection import alignment_metrics
    device = table.device
    parts = []
    for s in range(0, len(states['chosen']), chunk):
        chosen = states['chosen'][s:s + chunk].to(device)
        h = states['h'][s:s + chunk].to(device)
        value = projection(h[None], anchor=table[chosen][None])[0]
        parts.append({k: v.cpu() for k, v in alignment_metrics(
            value, chosen, table, gold=states['gold'][s:s + chunk].to(device)).items()})
    metrics = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    report = {}
    for name, mask in _bucket_masks(states).items():
        if not mask.any():
            continue
        row = {'n': int(mask.sum())}
        for key, values in metrics.items():
            row[key] = round(float(values[mask].mean()), 5)
        row['cos_chosen_p10'] = round(float(metrics['cos_chosen'][mask].quantile(.1)), 5)
        report[name] = row
    return report


@torch.no_grad()
def functional_kl(backbone, projection, rows, *, max_rows: int = 64):
    """Mean KL(p_E[chosen] || p_P) over held span positions: the backbone reads each held row's supervised span as a
    block of values (each the value of the state that predicts it) -- once as E[chosen], once as P(h) -- and the
    next-token distributions on the span are compared; per confidence bucket of the producing state."""
    device = backbone.embedding_weight.device
    sums = {}
    for row in rows[:max_rows]:
        ids = torch.tensor([row['ids']], device=device)
        start = row['start']
        inputs = backbone.embed(ids)
        hidden = backbone.forward_embeds(inputs, logits=False)['h_final']
        source = hidden[:, start - 1:ids.shape[1] - 1]
        logits = backbone.logits(source).float()
        pmax = torch.softmax(logits, -1).amax(-1)[0]
        chosen = logits.argmax(-1)
        del logits

        def read(block):
            x = inputs.clone()
            x[:, start:] = block.to(x.dtype)
            return torch.log_softmax(backbone.logits(
                backbone.forward_embeds(x, logits=False)['h_final'][:, start:]).float(), -1)[0]
        anchor = backbone.embed(chosen)
        reference, candidate = read(anchor), read(projection(source, anchor=anchor))
        kl = (reference.exp() * (reference - candidate)).sum(-1)
        for name, mask in (('all', torch.ones_like(pmax, dtype=torch.bool)), ('p>=0.9', pmax >= .9),
                           ('p0.5-0.9', (pmax >= .5) & (pmax < .9)), ('p<0.5', pmax < .5)):
            total, count = sums.get(name, (0.0, 0))
            sums[name] = (total + float(kl[mask].sum()), count + int(mask.sum()))
    return {name: {'n': count, 'kl_mean': round(total / count, 5) if count else None}
            for name, (total, count) in sums.items()}


def verify(report: dict, thresholds: dict) -> dict:
    """The gate on held measurements (``report['held']`` geometry, ``report['functional_kl']``): criteria with
    status pass/fail/pending; missing evidence never passes."""
    from .e2e_acceptance import projection_on_manifold
    return projection_on_manifold(thresholds, report)


def _load(a, *, aligned=None):
    from .direct_export import load_direct_artifact
    dtype = torch.bfloat16 if a.device != 'cpu' else torch.float32
    return load_direct_artifact(a.artifact, base_snapshot=a.base, device=a.device, dtype=dtype,
                                aligned_projection=aligned, allow_unaligned=aligned is None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('init', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--artifact', type=Path, required=True, help='the exact direct export (parent of e2e)')
        p.add_argument('--base', type=Path, required=True, help='its pinned base snapshot')
        p.add_argument('--text-data', type=Path, required=True,
                       help="the line's rendered text data JSONL (token_ids, split, supervised_suffix_start)")
        p.add_argument('--device', default='cuda')
        p.add_argument('--held-rows', type=int, default=200)
        p.add_argument('--max-tokens', type=int, default=2048)
        p.add_argument('--kl-rows', type=int, default=48)
        p.add_argument('--thresholds', default=json.dumps(DEFAULT_THRESHOLDS))
        p.add_argument('--seed', type=int, default=0)
    init = sub.choices['init']
    init.add_argument('--out', type=Path, required=True, help='directory for aligned-projection.pt and its report')
    init.add_argument('--continue-from', type=Path, help='an aligned P of this export to carry (else C = 0)')
    init.add_argument('--hidden', type=int, default=None, help="C's residual MLP width (default 2 x width)")
    sub.choices['verify'].add_argument('--aligned-projection', type=Path, required=True)
    sub.choices['verify'].add_argument('--out', type=Path, help='write the verification report here')
    a = parser.parse_args(argv)
    thresholds = json.loads(a.thresholds)
    log = lambda event: print(json.dumps(event), flush=True)
    started = time.time()
    torch.manual_seed(a.seed)
    held = text_rows(a.text_data, held_rows=a.held_rows, max_tokens=a.max_tokens)
    log({'event': 'rows', 'held': len(held)})
    if a.command == 'verify':
        direct = _load(a, aligned=a.aligned_projection)
        projection = direct.neuralese_projection
    else:
        direct = _load(a)
        from ..model.neuralese_projection import NeuraleseProjection, load_aligned_projection
        width = int(direct.engine.backbone.embedding_weight.shape[1])
        if a.continue_from is not None:
            projection, _ = load_aligned_projection(a.continue_from, direct.manifest['manifest_sha256'])
        else:
            projection = NeuraleseProjection(width, hidden=a.hidden)
    projection = projection.to(a.device).float().eval()
    backbone = direct.engine.backbone
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    table = input_embedding_table(backbone)
    held_states = collect(backbone, held, seed=a.seed)
    log({'event': 'collected', 'split': 'held', 'positions': len(held_states['chosen'])})
    report = {'schema': SCHEMA, 'artifact': str(a.artifact), 'artifact_manifest_sha256':
              direct.manifest['manifest_sha256'], 'text_data': str(a.text_data), 'thresholds': thresholds,
              'held_rows': len(held), 'projection': dict(projection.config),
              'init': ('continue' if getattr(a, 'continue_from', None) else 'zero-correction')
              if a.command == 'init' else 'verify'}
    report['held'] = measure_geometry(projection, held_states, table)
    report['functional_kl'] = functional_kl(backbone, projection, held, max_rows=a.kl_rows)
    report['criteria'] = verify(report, thresholds)
    report['passed'] = all(c['status'] == 'pass' for c in report['criteria'].values())
    report['seconds'] = round(time.time() - started)
    if a.command == 'init':
        from ..model.neuralese_projection import save_aligned_projection
        a.out.mkdir(parents=True, exist_ok=True)
        save_aligned_projection(a.out / 'aligned-projection.pt', projection,
                                artifact_manifest_sha256=direct.manifest['manifest_sha256'], report=report)
        (a.out / 'projection-alignment.json').write_text(json.dumps(report, indent=1) + '\n')
    elif a.out is not None:
        a.out.write_text(json.dumps(report, indent=1) + '\n')
    log({'event': 'done', 'passed': report['passed'], 'held': report['held'].get('all'),
         'functional_kl': report['functional_kl'].get('all')})
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
