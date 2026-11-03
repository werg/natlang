#!/usr/bin/env python3
"""Prepare one actor for the automated Pop behavioural evaluation, inside the checkpoint's own frozen runtime.

Launched by scripts/neuralese_data/pop_behavior_eval.py as a ledger unit whose NATLANG_REPO/PYTHONPATH/workdir are
the runtime that trained the checkpoint, so the transport binds the Python runtime Pop will serve with.

    pop_eval_prepare.py raw  --checkpoint CK --out VIEWS --receipt R.json [--expect-sha256 S]
    pop_eval_prepare.py ema  --checkpoint CK --out EMA   --receipt R.json --base HF_BASE [--expect-sha256 S]

raw: ``native_bundle.prepare_actor_transport(scope='diagnostic-local-pin')``: a model-only view read in place from
the source (no copy, no pin or hardlink added to the learner directory).

ema: the checkpoint's weight EMA (``train/weight_ema.py``) written as a model-only checkpoint ``<sha>.pt`` beside an
EMA view receipt and a transfer manifest that ``scripts/coreweave/send_actor_transport.py`` sends unchanged. It is an
evaluation artifact, never a resumable parent. Exit code 3 when the checkpoint carries no EMA.

Either way the source file is never written; its stat is checked before and after the read.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def stat5(path):
    s = Path(path).stat()
    return {'device': s.st_dev, 'inode': s.st_ino, 'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name('.' + path.name + '.pending')
    with pending.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def immutable(path, raw):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f'{path} exists with other bytes')
        return
    pending = path.with_name('.' + path.name + '.pending')
    with pending.open('wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(pending, path)


def raw_view(args):
    from natlang_neuralese.model.native_bundle import prepare_actor_transport
    binding = prepare_actor_transport(args.checkpoint, args.out, scope='diagnostic-local-pin')
    transport = json.loads(Path(binding['path']).read_text())
    return {'kind': 'raw', 'transport': binding, 'source': transport['source'], 'step': transport['step'],
            'view': transport['view'], 'view_receipt': transport['view_receipt'],
            'python_runtime': transport['python_runtime'], 'scope': transport['scope']}


def ema_view(args):
    import torch
    from natlang_neuralese.common.runtime_inventory import python_runtime_binding
    from natlang_neuralese.model.native_model import load_native_checkpoint, model_state
    from natlang_neuralese.train.native_state_sft import native_trainable_parameters
    from natlang_neuralese.train.trajectory_state import atomic_checkpoint
    from natlang_neuralese.train.weight_ema import EXPORT_SCHEMA, WeightEMA
    source_sha = digest(args.checkpoint)
    source_bytes = Path(args.checkpoint).stat().st_size
    with torch.no_grad():
        model, _, _, config, state = load_native_checkpoint(args.checkpoint, base=args.base, device='cpu')
        training = state.get('training') or {}
        if training.get('ema') is None:
            print(json.dumps({'kind': 'ema', 'absent': True, 'source_sha256': source_sha}), flush=True)
            raise SystemExit(3)
        named = native_trainable_parameters(model)
        ema = WeightEMA.restore(named, training['ema'])
        missing = [event for event in ema.provenance['events'][-1:] if event.get('reinitialized')]
        if missing:
            raise ValueError('checkpoint EMA does not cover every trainable parameter')
        ema.load_into(named)
        step = training.get('global_step', training.get('step', 0))
        export = {'schema': EXPORT_SCHEMA, 'source_checkpoint_sha256': source_sha, 'source_global_step': step,
                  'ema': ema.summary(), 'purpose': 'protected Pop dev16 evaluation of the weight EMA; not a resumable parent'}
        out = Path(args.out) / source_sha
        out.mkdir(parents=True, exist_ok=True)
        pending = out / '.ema-export.pt'
        atomic_checkpoint(pending, {'schema': state['schema'], 'model_config': config, **model_state(model),
                                    'step': step, 'ema_export': export})
    del model, state, training
    view_sha = digest(pending)
    view = out / (view_sha + '.pt')
    os.replace(pending, view)
    view_bytes = view.stat().st_size
    receipt = {'schema': 'natlang.weight-ema-eval-view/1', 'source': {'sha256': source_sha, 'bytes': source_bytes},
               'view': {'path': view.name, 'sha256': view_sha, 'bytes': view_bytes}, 'step': step,
               'ema': export['ema'], 'training_resume': False,
               'logical_identity': 'weight EMA of the source training checkpoint (model-only)'}
    receipt_raw = (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode()
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    immutable(out / (receipt_sha + '.json'), receipt_raw)
    manifest = {'schema': 'natlang.weight-ema-eval-transfer/1', 'scope': 'model-only-ema-export',
                'source': {'sha256': view_sha, 'bytes': view_bytes},
                'training_source': {'sha256': source_sha, 'bytes': source_bytes},
                'view': {'path': view.name, 'sha256': view_sha, 'bytes': view_bytes},
                'view_receipt': {'sha256': receipt_sha, 'bytes': len(receipt_raw)},
                'step': step, 'python_runtime': python_runtime_binding(), 'training_resume': False}
    manifest_raw = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest_path = out / (manifest_sha + '.transport.json')
    immutable(manifest_path, manifest_raw)
    return {'kind': 'ema', 'transport': {'path': str(manifest_path), 'sha256': manifest_sha, 'bytes': len(manifest_raw)},
            'source': manifest['source'], 'training_source': manifest['training_source'], 'step': step,
            'view': manifest['view'], 'view_receipt': manifest['view_receipt'],
            'python_runtime': manifest['python_runtime'], 'scope': manifest['scope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('kind', choices=('raw', 'ema'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--base')
    parser.add_argument('--expect-sha256')
    args = parser.parse_args()
    runtime = os.environ['NATLANG_REPO']
    import natlang_neuralese
    if not natlang_neuralese.__file__.startswith(runtime):
        raise SystemExit(f'natlang_neuralese resolved outside the bound runtime: {natlang_neuralese.__file__}')
    before = stat5(args.checkpoint)
    started = time.time()
    result = raw_view(args) if args.kind == 'raw' else ema_view(args)
    after = stat5(args.checkpoint)
    if before != after:
        raise SystemExit(f'source changed during preparation: {before} {after}')
    training_sha = (result.get('training_source') or result['source'])['sha256']
    if args.expect_sha256 and training_sha != args.expect_sha256:
        raise SystemExit(f'source hashes to {training_sha}, expected {args.expect_sha256} (superseded slot)')
    record = {'schema': 'natlang.pop-eval-preparation/1', **result, 'source_path': str(Path(args.checkpoint).resolve()),
              'source_stat': before, 'runtime': runtime, 'seconds': time.time() - started,
              'purpose': 'protected Pop dev16 evaluation only; no training admission'}
    durable_json(args.receipt, record)
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    sys.exit(main())
