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
    from ..train.native_semantic_stage import run,validate_config
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
    qualification_models={name:dict(entry) for name,entry in source['models'].items()}
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
            boundary=report.get('continuation_boundary') or {}
            partial=(boundary==receipt.get('continuation_boundary') and boundary.get('kind')=='semantic-refresh'
                and boundary.get('reached') is True and boundary.get('new_optimizer_updates',0)>0
                and boundary.get('checkpoint_sha256')==identity['checkpoint_sha256'])
            if partial:
                from ..train.native_readiness import _bound
                runtime=json.loads(_bound(receipt['runtime_certificate']).read_text())
                task=json.loads(_bound(receipt['task_execution_receipt']).read_text())
                if (report.get('mandatory_sft')!=receipt.get('mandatory_sft')
                        or (report.get('mandatory_sft') or {}).get('schema')!='natlang.native-sft-progress/1'
                        or runtime.get('passed') is not True or runtime.get('native_identity')!=identity
                        or task.get('passed') is not True or task.get('native_identity')!=identity
                        or task.get('complete') is not True or task.get('eligible') is not True):
                    raise ValueError('partial supervised continuation lacks exact task/runtime/mandatory-exposure evidence')
            if (proof.get('checkpoint_sha256') != parent['sha256'] or proof.get('participant') != name
                    or not proof.get('objective_transition') or not (report.get('completed') or partial)
                    or not report.get('numerics_ok')
                    or identity['checkpoint_sha256'] != sha256_file_hex_cached(path)):
                raise ValueError('native RL update lacks completed actual optimizer/runtime evidence from this joint parent')
            qualification_models[name]={**qualification_models[name],'native_identity':identity,
                'marker_ids':report['model_config']['marker_ids'],
                'raw_communication_rows':report.get('raw_communication_rows') or {}}
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
    from ..train.native_continuation import qualified_refresh_config
    selected,pending=qualified_refresh_config(source,config,qualification_models)
    selected=validate_config(selected)
    selected_path=out/'qualified-curriculum.json'
    write_json_durable(selected_path,selected)
    report = run(selected_path, participants, out, resume=own_checkpoint if own_checkpoint.exists() else parent['path'],
                 refresh_participants=True, device=device, stop=stop)
    if stop is not None and stop.requested:
        raise InterruptedError('semantic refresh checkpointed after stop request; deployment unchanged')
    if not (report.get('completed') or report.get('refresh_ready')) or not report.get('production_ready'):
        raise RuntimeError('semantic refresh incomplete or runtime/task evidence missing; actors retain previous deployment')
    exported = out/'semantic-adapter.json'
    value = json.loads(exported.read_text())
    from ..train.native_readiness import validate_completion
    validate_completion(out/'shared-stage-completion.json',
        {(manifest['signature'],canonical_json_sha256_hex(selected)):{'models':[entry['model_id'] for entry in models],'execution_signature':'semantic-refresh'}})
    deployment = {'schema': 'natlang.native-rl-deployment/1',
        'semantic_adapter': {'path': str(exported), 'sha256': sha256_file_hex_cached(exported)},
        'checkpoint': value['checkpoint'], 'models': value['models'],
        'parent_adapter_sha256': sha256_file_hex_cached(adapter),
        'pending_curriculum':({'path':str(Path(config).resolve()),'sha256':sha256_file_hex_cached(config),
            'raw_communication_deficits':pending} if pending else None),
        'production_curriculum_complete':bool(not pending and report.get('completed') and
            all((entry.get('mandatory_sft') or {}).get('complete') is True for entry in value['models'].values())),
        'semantic_obligations':value.get('semantic_obligations'),
        'shared_stage_completion':{'path':str((out/'shared-stage-completion.json').resolve()),
            'sha256':sha256_file_hex_cached(out/'shared-stage-completion.json')},
        'updated_participants': {name: {'path': str(Path(path).resolve()),
             'sha256': sha256_file_hex_cached(path)} for name, path in updates.items()}}
    if value.get('policy_training_progress'):
        deployment['policy_training_progress']=value['policy_training_progress']
    write_json_durable(out/'deployment.json', deployment)
    return deployment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--update', action='append', default=[], help='participant=full_native_checkpoint')
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
