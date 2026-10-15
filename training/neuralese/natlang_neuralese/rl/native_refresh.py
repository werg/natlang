"""Refresh the existing joint semantic curriculum after sealed native RL updates.

Merges independently updated participants from one immutable joint parent. The
semantic trainer preserves their optimizer state and performs all required
VAE/joint/distill exposure before publishing a new shared adapter deployment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from ..common.jsonio import write_json_durable


def refresh(*, adapter, updates, config, out, device='cuda', stop=None):
    from ..train.native_semantic_stage import run
    adapter, out = Path(adapter).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads(adapter.read_text())
    if source.get('schema') != 'natlang.native-semantic-adapter/1':
        raise ValueError('native RL refresh requires its exact shared semantic adapter export')
    parent = source['checkpoint']
    if sha256_file_hex_cached(parent['path']) != parent['sha256']:
        raise ValueError('native RL semantic parent checkpoint changed')
    if set(updates) - set(source['models']):
        raise ValueError('native RL refresh names an unknown participant')
    models = []
    for name, binding in source['models'].items():
        if name in updates:
            path = Path(updates[name]).resolve(strict=True)
            receipt = json.loads((path.parent/'native-training-receipt.json').read_text())
            report_binding = receipt['training_receipt']
            report_path = Path(report_binding['path'])
            if sha256_file_hex_cached(report_path) != report_binding['sha256']:
                raise ValueError('native RL training report changed')
            report = json.loads(report_path.read_text())
            proof = receipt.get('continuation') or {}
            identity = receipt['native_identity']
            if (proof.get('checkpoint_sha256') != parent['sha256'] or proof.get('participant') != name
                    or not proof.get('objective_transition') or not report.get('completed')
                    or not report.get('numerics_ok')
                    or identity['checkpoint_sha256'] != sha256_file_hex_cached(path)):
                raise ValueError('native RL update lacks completed actual optimizer/runtime evidence from this joint parent')
            # The shared trainer rechecks this identity against actual restored
            # tensors; use its bound receipt here to avoid an extra 24GB load.
            models.append({'model_id': name, 'checkpoint': {'path': str(path), 'sha256': identity['checkpoint_sha256']},
                           'native_identity': identity})
        else:
            models.append({'model_id': name, 'checkpoint': binding['checkpoint'],
                           'native_identity': binding['native_identity']})
    manifest = {'schema': 'natlang.native-shared-participants/1', 'models': models,
                'signature': canonical_json_sha256_hex(models)}
    participants = out/'participants.json'
    write_json_durable(participants, manifest)
    completed = out/'report.json'
    own_checkpoint = out/'checkpoint.pt'
    # A partially refreshed stage resumes its own full VAE/model/optimizer slot.
    report = run(config, participants, out, resume=own_checkpoint if own_checkpoint.exists() else parent['path'],
                 refresh_participants=True, device=device, stop=stop)
    if stop is not None and stop.requested:
        raise InterruptedError('semantic refresh checkpointed after stop request; deployment unchanged')
    if not report.get('completed') or not report.get('production_ready'):
        raise RuntimeError('semantic refresh incomplete or runtime/task evidence missing; actors retain previous deployment')
    exported = out/'semantic-adapter.json'
    value = json.loads(exported.read_text())
    deployment = {'schema': 'natlang.native-rl-deployment/1',
        'semantic_adapter': {'path': str(exported), 'sha256': sha256_file_hex_cached(exported)},
        'checkpoint': value['checkpoint'], 'models': value['models'],
        'parent_adapter_sha256': sha256_file_hex_cached(adapter),
        'updated_participants': {name: {'path': str(Path(path).resolve()),
             'sha256': sha256_file_hex_cached(path)} for name, path in updates.items()}}
    write_json_durable(out/'deployment.json', deployment)
    return deployment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--update', action='append', required=True, help='participant=full_native_checkpoint')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    updates = dict(item.split('=', 1) for item in args.update)
    if len(updates) != len(args.update):
        raise ValueError('native RL refresh participant updates must be unique')
    refresh(adapter=args.adapter, updates=updates, config=args.config, out=args.out, device=args.device)


if __name__ == '__main__':
    main()
