"""Weight-space edits of native checkpoints as re-applicable operators (schema natlang.weight-space-edit/1).

An edit is an ordered list of operators over named tensors of a decoded native checkpoint state. Operators are
expressed against the weights they are applied to: column/row selections and projections are recomputed on each
target (and recorded), so one edit can be applied to the parent it was found on and to any successor checkpoint.
Exact reproduction on a given parent is witnessed by the per-tensor SHA-256 of parent and edited bytes and by an exact
patch (changed flat indices and their new bit patterns), both recorded with the candidate.

Tensor names: ``backbone.<backbone_state key>`` or ``interface.<input_rows|extra_input_rows>`` or
``interface.adapter.<down|up>.weight``.

Operators (vectors live in the spec's ``vectors`` .npz by key; arithmetic is float64, cast back to the tensor dtype
once per operator):

``project_out_columns``
    For an ``nn.Linear`` weight W [out, in] that writes into the space of ``vector`` d:
    ``W[:, J] -= alpha * u (u^T W[:, J])``. With ``gain`` (the weight g of the RMSNorm applied to this projection's
    output, e.g. LFM3's ``post_ffn_norm``) u = normalize(g * d): since d^T (g * y) = (g * d)^T y, removing u from the
    pre-norm output removes the d component of the normed residual write exactly. J: ``{"top_k": K}`` (the K columns
    with largest |u^T W[:, j]|, recomputed on the target), ``{"columns": [...]}`` or ``{"all": true}``.
``project_out_rows``
    ``W[J, :] -= alpha * (W[J, :] v) v^T``: rows J stop reading input direction v.
``rank_one``
    ``W += scale * a b^T``.

A full candidate checkpoint (``write_candidate``) keeps every other tensor and all optimizer, RNG and exposure state
byte-identical and appends a ``weight_space_edit_history`` record to ``training``: a zero-SGD transition with no
optimizer update, graph visit or exposure credit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

SCHEMA = 'natlang.weight-space-edit/1'
TRANSITION_SCHEMA = 'natlang.native-zero-sgd-weight-edit/1'


def load_spec(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    if spec.get('schema') != SCHEMA:
        raise ValueError('not a weight-space edit spec')
    vectors = {}
    if spec.get('vectors'):
        vp = Path(spec['vectors'])
        if not vp.is_absolute():
            vp = path.parent / vp
        raw = vp.read_bytes()
        if hashlib.sha256(raw).hexdigest() != spec.get('vectors_sha256'):
            raise ValueError('edit vectors differ from the declared SHA-256')
        with np.load(vp) as z:
            vectors = {k: z[k].astype(np.float64) for k in z.files}
    spec['_vectors'] = vectors
    spec['_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return spec


def _ref(state, name):
    if name.startswith('backbone.'):
        return state['backbone_state'], name[len('backbone.'):]
    if name.startswith('interface.adapter.'):
        return state['interface_state']['adapter'], name[len('interface.adapter.'):]
    if name.startswith('interface.'):
        return state['interface_state'], name[len('interface.'):]
    raise KeyError(name)


def tensor_sha(t):
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _select(scores, sel):
    if sel.get('all'):
        return np.arange(len(scores))
    if 'columns' in sel or 'rows' in sel:
        return np.asarray(sel.get('columns', sel.get('rows')), dtype=np.int64)
    k = int(sel['top_k'])
    return np.sort(np.argsort(-np.abs(scores), kind='stable')[:k])


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def apply_op(state, op, vectors):
    container, key = _ref(state, op['tensor'])
    original = container[key].detach().cpu()
    base = original.to(torch.float64).numpy()
    W = base.copy()
    kind = op['kind']
    alpha = float(op.get('alpha', 1.0))
    record = {'tensor': op['tensor'], 'kind': kind, 'shape': list(W.shape), 'parent_sha256': tensor_sha(original)}
    if kind == 'project_out_columns':
        d = vectors[op['vector']]
        if op.get('gain'):
            gc, gk = _ref(state, op['gain'])
            d = d * gc[gk].detach().cpu().to(torch.float64).numpy()
        u = _unit(d)
        scores = u @ W
        J = _select(scores, op.get('select', {'all': True}))
        W[:, J] -= alpha * np.outer(u, scores[J])
        record.update(selected=J.tolist(), removed_component_norm=float(np.linalg.norm(scores[J])),
                      total_component_norm=float(np.linalg.norm(scores)))
    elif kind == 'project_out_rows':
        v = _unit(vectors[op['vector']])
        scores = W @ v
        J = _select(scores, op.get('select', {'all': True}))
        W[J, :] -= alpha * np.outer(scores[J], v)
        record.update(selected=J.tolist(), removed_component_norm=float(np.linalg.norm(scores[J])))
    elif kind == 'rank_one':
        W += float(op.get('scale', 1.0)) * np.outer(vectors[op['a']], vectors[op['b']])
    else:
        raise ValueError(f'unknown weight-space edit operator {kind}')
    edited = torch.from_numpy(W).to(original.dtype)
    mask = (edited != original)
    container[key] = edited
    record.update(edited_sha256=tensor_sha(edited), changed_entries=int(mask.sum().item()), alpha=alpha,
                  relative_change=float(np.linalg.norm(W - base) / max(np.linalg.norm(base), 1e-30)))
    return record


def apply_edit(state, spec):
    """Apply every operator in order to a decoded checkpoint state (replacing edited tensors); return op records."""
    return [apply_op(state, op, spec.get('_vectors') or {}) for op in spec['ops']]


def exact_patch(parent_state, child_state, names):
    """Changed flat indices and new bit patterns per tensor: an exact delta valid only on this parent."""
    arrays, entries = {}, []
    for i, name in enumerate(names):
        pc, pk = _ref(parent_state, name)
        cc, ck = _ref(child_state, name)
        a, b = pc[pk].detach().cpu().contiguous(), cc[ck].detach().cpu().contiguous()
        bits = {2: torch.int16, 4: torch.int32}[a.element_size()]
        av, bv = a.reshape(-1).view(bits), b.reshape(-1).view(bits)
        idx = torch.nonzero(av != bv).reshape(-1)
        arrays[f't{i}_idx'] = idx.numpy().astype(np.int64)
        arrays[f't{i}_val'] = bv[idx].numpy()
        entries.append({'tensor': name, 'index': i, 'dtype': str(a.dtype), 'shape': list(a.shape),
                        'changed_entries': int(idx.numel()), 'parent_sha256': tensor_sha(a),
                        'edited_sha256': tensor_sha(b)})
    return arrays, entries


def apply_exact_patch(state, manifest, arrays):
    for entry in manifest['tensors']:
        container, key = _ref(state, entry['tensor'])
        t = container[key].detach().cpu().contiguous().clone()
        if tensor_sha(t) != entry['parent_sha256']:
            raise ValueError(f'exact patch parent differs for {entry["tensor"]}')
        bits = {2: torch.int16, 4: torch.int32}[t.element_size()]
        flat = t.reshape(-1).view(bits)
        i = entry['index']
        flat[torch.from_numpy(arrays[f't{i}_idx'])] = torch.from_numpy(arrays[f't{i}_val'])
        if tensor_sha(t) != entry['edited_sha256']:
            raise ValueError(f'exact patch did not reproduce {entry["tensor"]}')
        container[key] = t


def derive(parent, spec, *, parent_sha256):
    """The edited full state: new tensor objects for edited entries, everything else shared with the parent."""
    child = dict(parent)
    child['backbone_state'] = dict(parent['backbone_state'])
    interface = dict(parent['interface_state'])
    interface['adapter'] = dict(interface['adapter'])
    child['interface_state'] = interface
    records = apply_edit(child, spec)
    training = dict(child.get('training') or {})
    history = list(training.get('weight_space_edit_history') or [])
    history.append({'schema': TRANSITION_SCHEMA, 'parent_checkpoint_sha256': parent_sha256,
                    'edit_spec_sha256': spec['_sha256'], 'edit_name': spec.get('name'),
                    'at_global_step': training.get('global_step', training.get('step')),
                    'optimizer_updates': 0, 'graph_visits': 0,
                    'weights': 'edited tensors listed; all others byte-identical',
                    'optimizer_rng_exposure': 'preserved unchanged',
                    'operators': [{k: v for k, v in r.items() if k != 'selected'} | {'selected_count': len(r.get('selected', []))}
                                  for r in records]})
    training['weight_space_edit_history'] = history
    child['training'] = training
    return child, records


def write_candidate(parent_path, parent_sha256, spec_path, out_dir, *, minimum_free_bytes=40 * 1024**3):
    from ..common.hashing import sha256_file_hex
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..train.checkpoint_safety import PersistentCheckpointReservation, checkpoint_size_upper_bound
    from ..train.trajectory_state import atomic_checkpoint
    parent_path = Path(parent_path).resolve(strict=True)
    if sha256_file_hex(parent_path) != parent_sha256:
        raise ValueError('parent checkpoint SHA-256 differs')
    spec = load_spec(spec_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=False)
    parent = CheckpointPolicy.load(parent_path)
    child, records = derive(parent, spec, parent_sha256=parent_sha256)
    names = [r['tensor'] for r in records]
    names = list(dict.fromkeys(names))
    arrays, entries = exact_patch(parent, child, names)
    np.savez_compressed(out / 'exact-patch.npz', **arrays)
    target = out / 'checkpoint.pt'
    reservation = PersistentCheckpointReservation(target, minimum_free_bytes=minimum_free_bytes)
    try:
        reservation.acquire(checkpoint_size_upper_bound(child))
        atomic_checkpoint(target, child, reservation=reservation)
    finally:
        reservation.close()
    os.chmod(target, 0o444)
    receipt = {'schema': TRANSITION_SCHEMA, 'parent': {'path': str(parent_path), 'sha256': parent_sha256},
               'checkpoint': {'path': str(target), 'sha256': sha256_file_hex(target), 'bytes': target.stat().st_size},
               'edit_spec': {'path': str(Path(spec_path).resolve()), 'sha256': spec['_sha256'], 'name': spec.get('name')},
               'exact_patch': {'path': str(out / 'exact-patch.npz'),
                               'sha256': sha256_file_hex(out / 'exact-patch.npz'), 'tensors': entries},
               'operators': records,
               'changed_weights_total': sum(e['changed_entries'] for e in entries),
               'optimizer_updates': 0, 'graph_visits': 0, 'qualification': 'pending protected development behaviour'}
    (out / 'edit-record.json').write_text(json.dumps(receipt, indent=1) + '\n')
    return receipt


def verify_candidate(record_path):
    """Reload parent and child; the child must equal the parent with the exact patch applied, byte for byte."""
    from ..common.hashing import sha256_file_hex
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..train.native_zero_sgd_adoption import _same
    record = json.loads(Path(record_path).read_text())
    for ref in (record['parent'], record['checkpoint']):
        if sha256_file_hex(Path(ref['path'])) != ref['sha256']:
            raise ValueError(f'{ref["path"]} changed')
    spec = load_spec(record['edit_spec']['path'])
    parent = CheckpointPolicy.load(record['parent']['path'])
    child = CheckpointPolicy.load(record['checkpoint']['path'])
    expected, _ = derive(parent, spec, parent_sha256=record['parent']['sha256'])
    expected['training']['weight_space_edit_history'][-1] = child['training']['weight_space_edit_history'][-1]
    _same(expected, child)
    with np.load(record['exact_patch']['path']) as z:
        arrays = {k: z[k] for k in z.files}
    patched = dict(parent); patched['backbone_state'] = dict(parent['backbone_state'])
    patched['interface_state'] = dict(parent['interface_state'])
    patched['interface_state']['adapter'] = dict(parent['interface_state']['adapter'])
    apply_exact_patch(patched, {'tensors': record['exact_patch']['tensors']}, arrays)
    for entry in record['exact_patch']['tensors']:
        pc, pk = _ref(patched, entry['tensor']); cc, ck = _ref(child, entry['tensor'])
        if tensor_sha(pc[pk]) != tensor_sha(cc[ck]):
            raise ValueError('exact patch reconstruction differs from the candidate')
    return {'verified': True, 'checkpoint_sha256': record['checkpoint']['sha256'],
            'parent_sha256': record['parent']['sha256'], 'tensors': len(record['exact_patch']['tensors'])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    w = sub.add_parser('write', help='apply an edit spec to a full checkpoint and write a zero-SGD candidate')
    w.add_argument('--parent', required=True); w.add_argument('--parent-sha256', required=True)
    w.add_argument('--spec', required=True); w.add_argument('--out', required=True)
    v = sub.add_parser('verify', help='independently rederive a written candidate from its parent')
    v.add_argument('record')
    a = ap.parse_args(argv)
    if a.cmd == 'write':
        print(json.dumps({k: v for k, v in write_candidate(a.parent, a.parent_sha256, a.spec, a.out).items()
                          if k != 'operators'}, indent=1))
    else:
        print(json.dumps(verify_candidate(a.record)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
