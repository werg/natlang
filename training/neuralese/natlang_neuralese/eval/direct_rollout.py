"""Compare ordinary-token and direct neuralese feedback from identical prefixes.

Descriptive diagnostic only: readout tokens from continuous feedback are not a
trained readout operator, and their decoded strings do not grant task admission.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex
from ..serve.direct_write import DirectWriter
from ..train.direct_export import load_direct_artifact
from .direct_runtime import opened_context
from .plain_behavior import _render


def run(artifact, base_snapshot, probes_path, probes_sha256, out, tokens=64, per_kind=4, device='cuda'):
    raw = Path(probes_path).read_bytes()
    import hashlib
    if hashlib.sha256(raw).hexdigest() != probes_sha256:
        raise ValueError('probe snapshot identity changed')
    out = Path(out)
    out.mkdir(exist_ok=False, parents=True)
    (out / 'probes.jsonl').write_bytes(raw)
    probes = [json.loads(line) for line in raw.splitlines() if line.strip()]
    counts, selected = {}, []
    for probe in probes:
        if probe.get('signal') != 'ordinary_competence':
            continue
        kind = probe.get('kind', 'unspecified')
        if counts.get(kind, 0) < per_kind:
            selected.append(probe)
            counts[kind] = counts.get(kind, 0) + 1
    if not selected:
        raise ValueError('no explicitly ordinary probes')
    model = load_direct_artifact(artifact, base_snapshot=base_snapshot, device=device)
    engine = model.engine
    backbone, projection = engine.backbone, model.neuralese_projection
    results = []
    with torch.no_grad():
        for probe in selected:
            ids, _, _ = _render(engine.tokenizer, probe)
            if isinstance(ids, Mapping):
                ids = ids['input_ids']
            if ids.ndim == 1:
                ids = ids[None]
            embeds = backbone.embed(ids.to(device))
            opened = opened_context(engine, embeds)
            neural = DirectWriter(backbone, projection, opened, tokens, engine.stop_ids)
            cache, top = opened.cache, opened.top
            plain_tokens, neural_tokens, observations = [], [], []
            plain_done = False
            for position in range(tokens):
                value = neural.decide()
                if value is None:
                    break
                plain_logits = backbone.logits(top[:, None])[:, -1].float()
                plain_token = int(plain_logits.argmax(-1)[0])
                if not plain_done:
                    plain_tokens.append(plain_token)
                    plain_done = plain_token in engine.stop_ids
                neural_logits = backbone.logits(neural.history[-1][:, None])[:, -1].float()
                neural_token = int(neural_logits.argmax(-1)[0])
                neural_tokens.append(neural_token)
                # This comparison stays descriptive after token decisions diverge:
                # the histories then differ, so later agreement is not accuracy.
                gold_embedding = backbone.embedding_weight[plain_token].float()
                error = ((value[0].float() - gold_embedding).square().sum() /
                         gold_embedding.square().sum().clamp_min(1e-12))
                observations.append({'position': position, 'plain_token': plain_token,
                    'neural_token': neural_token, 'payload_rms': float(value.float().square().mean().sqrt()),
                    'plain_embedding_rms': float(gold_embedding.square().mean().sqrt()),
                    'payload_relative_mse_to_plain_decision': float(error),
                    'neural_next_token_probability': float(neural_logits.softmax(-1).max()),
                    'payload_finite': bool(torch.isfinite(value).all())})
                hidden, cache = backbone.run_layers(backbone.embed(torch.tensor([[plain_token]], device=device)),
                                                    range(backbone.num_layers), cache)
                top = hidden[:, 0]
                final, neural_cache = backbone.run_layers(value[:, None], range(backbone.num_layers), neural.cache)
                neural.advance(value, final[:, 0], neural_cache)
            if not neural.done and neural.count == tokens:
                neural.decide()  # record stop/cap decision after the last emitted value
            # Continue the ordinary baseline to its own stop/cap even if neural stops early.
            while not plain_done and len(plain_tokens) < tokens:
                token = int(backbone.logits(top[:, None])[:, -1].argmax(-1)[0])
                plain_tokens.append(token)
                plain_done = token in engine.stop_ids
                if not plain_done:
                    hidden, cache = backbone.run_layers(backbone.embed(torch.tensor([[token]], device=device)),
                                                        range(backbone.num_layers), cache)
                    top = hidden[:, 0]
            divergence = next((r['position'] for r in observations if r['plain_token'] != r['neural_token']), None)
            row = {'id': probe['id'], 'kind': probe.get('kind'), 'prompt_tokens': ids.shape[-1],
                'plain_tokens': plain_tokens, 'neural_readout_tokens': neural_tokens,
                'plain_text': engine.tokenizer.decode(plain_tokens, skip_special_tokens=False),
                'neural_readout_text': engine.tokenizer.decode(neural_tokens, skip_special_tokens=False),
                'plain_stopped': plain_done, 'neural_stopped': neural.done and not neural.truncated,
                'neural_budget_exhausted': neural.truncated, 'first_token_divergence': divergence,
                'observations': observations}
            results.append(row)
            with (out / 'results.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
    report = {'schema': 'natlang.direct-feedback-diagnostic/1', 'artifact_manifest_sha256':
        sha256_file_hex(Path(artifact) / 'manifest.json'), 'probes_sha256': probes_sha256,
        'tokens': tokens, 'selected_probe_ids': [p['id'] for p in selected],
        'runtime_code': {str(p.relative_to(Path(__file__).parents[1])): sha256_file_hex(p)
                         for p in sorted(Path(__file__).parents[1].rglob('*.py'))},
        'all_payloads_finite': all(o['payload_finite'] for r in results for o in r['observations']),
        'task_qualified': False, 'scope': 'Equal-prefix ordinary vs continuous feedback; not post-divergence accuracy.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'base-snapshot', 'probes', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--probes-sha256', required=True)
    p.add_argument('--tokens', type=int, default=64)
    p.add_argument('--per-kind', type=int, default=4)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if min(a.tokens, a.per_kind) < 1:
        p.error('positive diagnostic bounds required')
    run(a.artifact, a.base_snapshot, a.probes, a.probes_sha256, a.out, a.tokens, a.per_kind, a.device)


if __name__ == '__main__':
    main()
