"""Matrix-growth and attention-logit telemetry for the native trainer (TRAINING_INSIGHTS_20261028.md section 3).

Constant-size Muon steps grow every matrix; growth in Q.K sharpens attention. QK-norm closes that path inside the
model, so this watches for growth anywhere it does not reach, for example imported Neuralese keys at consumer positions.

* Per-matrix Frobenius norm and its ratio to the reference norm. The reference is the norm of the lineage's first
  telemetry-enabled parent (fresh runs: the base weights), inherited unchanged across continuations.
* Names are matched generically (the layer index is stripped), so every matrix kind appears under its module path:
  ``self_attn.q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` / ``gate_proj``, ``feed_forward.w1`` / ``w2`` / ``w3``,
  ``conv.in_proj`` / ``out_proj`` and, as ``embedding``, the vocabulary matrix and the learned role/reasoning rows.
  Only parameters with at least two dimensions are matrices.
* Max pre-softmax attention logit per attention layer on the logged batch, read from the real forward by
  ``model.attention_probe`` (one extra QK product per observed layer on logged steps only). ``layers`` limits the
  cost to that many evenly spaced attention layers; the result states which layers were observed.

Declaration (JSON, ``--matrix-telemetry``)::

    {"every": 100, "norms": true, "attention_logits": true, "layers": null}

Absent, ``DEFAULT`` runs (on by default); ``"off"`` opts out. Telemetry reads weights after the update and never
touches RNG, optimizer state or gradients.
"""

from __future__ import annotations

import json
import re
import time

import torch

SCHEMA = 'natlang.native-matrix-telemetry/1'
_LAYER = re.compile(r'(^|\.)layers\.\d+\.')


DEFAULT = {'every': 100, 'norms': True, 'attention_logits': True, 'layers': None}


def parse_matrix_telemetry(text):
    """On by default (owner 2026-10-28) at ``DEFAULT``; ``"off"`` opts out (``None``)."""
    if text is None or text == '':
        return dict(DEFAULT)
    if text == 'off':
        return None
    spec = json.loads(text) if isinstance(text, str) else dict(text)
    unknown = set(spec) - {'every', 'norms', 'attention_logits', 'layers'}
    if unknown:
        raise ValueError('unknown matrix telemetry options: ' + ', '.join(sorted(unknown)))
    every = spec.get('every')
    if type(every) is not int or every < 1:
        raise ValueError('matrix telemetry every must be a positive integer')
    layers = spec.get('layers')
    if layers is not None and (type(layers) is not int or layers < 1):
        raise ValueError('matrix telemetry layers must be omitted or a positive integer')
    result = {'every': every, 'norms': bool(spec.get('norms', True)),
              'attention_logits': bool(spec.get('attention_logits', True)), 'layers': layers}
    if not (result['norms'] or result['attention_logits']):
        raise ValueError('matrix telemetry needs norms and/or attention_logits')
    return result


def matrix_kind(name):
    """Module-path kind of a parameter name, layer index and wrapper prefixes removed."""
    if name.startswith('interface.'):
        return 'embedding'
    stem = name[:-len('.weight')] if name.endswith('.weight') else name
    stem = stem.removeprefix('backbone.')
    stem = _LAYER.sub(r'\1', stem).removeprefix('model.')
    return stem or name


def matrix_inventory(model, named):
    """(name, kind, parameter) for every matrix: trainable parameters plus the vocabulary embedding."""
    result = [(name, matrix_kind(name), parameter) for name, parameter in named if parameter.ndim >= 2]
    embedding = getattr(model.port, 'embedding_weight', None)
    if embedding is not None and not any(parameter is embedding for _, _, parameter in result):
        result.append(('vocabulary.embedding', 'embedding', embedding))
    return result


@torch.no_grad()
def frobenius_norms(inventory):
    """Frobenius norms in FP32 (no full-size temporaries); host values read after all kernels are queued."""
    values = [torch.linalg.vector_norm(parameter.detach(), dtype=torch.float32) for _, _, parameter in inventory]
    return {name: float(value) for (name, _, _), value in zip(inventory, values)}


def reference_record(norms, *, source, checkpoint_sha256, global_step):
    return {'schema': SCHEMA, 'source': source, 'checkpoint_sha256': checkpoint_sha256,
            'global_step': int(global_step), 'norms': dict(norms)}


def choose_attention_layers(port, count):
    """Evenly spaced attention layer indices (all of them when ``count`` is None or not smaller)."""
    layers = [index for index in range(port.num_layers) if port.is_attention(index)]
    if count is None or count >= len(layers):
        return layers
    if count == 1:
        return [layers[len(layers) // 2]]
    return sorted({layers[round(i * (len(layers) - 1) / (count - 1))] for i in range(count)})


def norm_report(model, named, reference, *, global_step):
    """Per-matrix norm and ratio plus a by-kind summary. A matrix absent from the reference reports no ratio."""
    started = time.monotonic()
    inventory = matrix_inventory(model, named)
    norms = frobenius_norms(inventory)
    matrices = {}
    by_kind = {}
    for name, kind, _ in inventory:
        base = (reference or {}).get('norms', {}).get(name)
        ratio = norms[name] / base if base else None
        matrices[name] = {'kind': kind, 'norm': norms[name], 'ratio': ratio}
        entry = by_kind.setdefault(kind, {'count': 0, 'max_norm': 0.0, 'max_ratio': None, 'max_ratio_name': None,
                                          '_ratio_sum': 0.0, '_ratios': 0})
        entry['count'] += 1
        entry['max_norm'] = max(entry['max_norm'], norms[name])
        if ratio is not None:
            entry['_ratio_sum'] += ratio
            entry['_ratios'] += 1
            if entry['max_ratio'] is None or ratio > entry['max_ratio']:
                entry['max_ratio'], entry['max_ratio_name'] = ratio, name
    for entry in by_kind.values():
        entry['mean_ratio'] = entry['_ratio_sum'] / entry['_ratios'] if entry['_ratios'] else None
        del entry['_ratio_sum'], entry['_ratios']
    return {'matrices': matrices, 'by_kind': by_kind, 'seconds': time.monotonic() - started,
            'reference': {key: (reference or {}).get(key) for key in ('source', 'checkpoint_sha256', 'global_step')}}
