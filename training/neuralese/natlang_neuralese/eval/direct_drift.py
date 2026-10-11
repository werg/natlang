"""Rescore immutable saved ordinary generations for a direct artifact's drift.

Reuses capture_reference/held_kl and the paired evaluator's renderer/checks;
never regenerates a saved response. Native shape diagnostics stay out of the gate.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import gc
import json
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex, canonical_json_sha256_hex
from ..maple.generation_gate import capture_reference, held_kl
from ..train.direct_export import load_direct_artifact
from .generation_checks import summarize
from .plain_behavior import _load_model, _render


def run(a):
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    rows = {}
    for name, path, expected in [('probes', a.probes, a.probes_sha256),
                                  ('base', a.base_outputs, a.base_outputs_sha256),
                                  ('candidate', a.candidate_outputs, a.candidate_outputs_sha256)]:
        raw = path.read_bytes()
        import hashlib
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('input changed: ' + name)
        (out / (name + '.jsonl')).write_bytes(raw)
        parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows[name] = {r['id']: r for r in parsed}
        if len(rows[name]) != len(parsed):
            raise ValueError('duplicate probe identity in ' + name)
        inputs[name] = {'path': str(path), 'sha256': expected}
    probes = [p for p in rows['probes'].values() if p.get('signal') == 'ordinary_competence']
    if not probes:
        raise ValueError('no ordinary competence probes')
    # This is the existing ordinary evaluator's LFM base loader, not a new teacher stack.
    model, tokenizer = _load_model(None, device=a.device, base=str(a.base_snapshot))
    responses, rendered = [], {}
    for p in probes:
        encoded, messages, conversions = _render(tokenizer, p)
        if isinstance(encoded, Mapping):
            encoded = encoded['input_ids']
        if encoded.ndim == 1:
            encoded = encoded[None]
        ids = encoded[0].tolist()
        record = {'messages': p['messages'], 'normalized_messages': messages,
                  'tool_argument_normalizations': conversions, 'tools': p.get('tools'),
                  'enable_thinking': bool(p.get('enable_thinking')), 'rendered_input_ids': ids}
        digest = canonical_json_sha256_hex(record)
        for role in ('base', 'candidate'):
            row = rows[role][p['id']]
            if row['prompt_sha256'] != digest or row.get('signal') != 'ordinary_competence':
                raise ValueError('saved output rendered prompt mismatch: ' + p['id'])
        rendered[p['id']] = ids
        responses.append(rows['base'][p['id']]['completion_token_ids'])
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    with torch.no_grad():
        reference = capture_reference(model, tokenizer, probes, lambda _, p: rendered[p['id']], eos,
                                      max_tokens=a.reference_tokens, generated_responses=responses)
    torch.save(reference, out / 'reference.pt')
    del model, tokenizer
    gc.collect()
    if a.device.startswith('cuda'):
        torch.cuda.empty_cache()
    restored = load_direct_artifact(a.artifact, base_snapshot=a.base_snapshot, device=a.device)
    with torch.no_grad():
        metrics = held_kl(restored.engine.backbone.hf, reference, rows=len(probes), prompt_weight=.25)
    summaries = {role: summarize([rows[role][p['id']] for p in probes]) for role in ('base', 'candidate')}
    complete = all(not row['response_truncated'] and row['response_tokens'] > 0
                   for row in reference['capture_coverage'])
    response_kl = metrics['held_response_kl']
    passed = (complete and torch.isfinite(torch.tensor(response_kl)).item() and response_kl <= a.max_response_kl
              and summaries['candidate']['gate_pass'] >= summaries['base']['gate_pass'] - a.max_gate_drop)
    package = Path(__file__).parents[1]
    report = {'schema': 'natlang.direct-saved-ordinary-drift/1', 'inputs': inputs,
              'artifact_manifest_sha256': sha256_file_hex(a.artifact / 'manifest.json'),
              'reference_sha256': sha256_file_hex(out / 'reference.pt'),
              'runtime_code': {str(p.relative_to(package)): sha256_file_hex(p) for p in sorted(package.rglob('*.py'))},
              'selected_probe_ids': [p['id'] for p in probes], 'capture_coverage': reference['capture_coverage'],
              'metrics': metrics, 'behavior': summaries, 'coverage_complete': complete,
              'gate': {'signal': 'ordinary_competence', 'max_response_kl': a.max_response_kl,
                       'max_gate_drop': a.max_gate_drop, 'prompt_kl_report_only': True},
              'ordinary_drift_gate_passed': passed, 'neuralese_consumer_qualified': False,
              'scope': 'Additive ordinary drift assessment on saved outputs; original report unchanged.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'base-snapshot', 'probes', 'base-outputs', 'candidate-outputs', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('probes', 'base-outputs', 'candidate-outputs'):
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--reference-tokens', type=int, default=None,
                   help='optional explicit per-row reference cap; by default each rendered prompt gets its full declared response budget')
    p.add_argument('--max-response-kl', type=float, default=.5)
    p.add_argument('--max-gate-drop', type=float, default=.1)
    a = p.parse_args()
    if (a.reference_tokens is not None and a.reference_tokens < 1) or a.max_response_kl < 0 or not 0 <= a.max_gate_drop <= 1:
        p.error('invalid assessment bounds')
    run(a)


if __name__ == '__main__':
    main()
