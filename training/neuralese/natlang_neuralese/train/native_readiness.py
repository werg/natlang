"""Exact native-model readiness and automatic shared-stage rendezvous.

Policy is declared by the recipe: required native stages, numerical contracts,
and the shared-stage command. Two ready models suffice; other model entries
never impose a global barrier. The supervisor may extend the ready set while
the shared trainer runs, preserving its existing model/optimizer identities.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import time

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from ..model.native_model import CHECKPOINT_SCHEMA


def _bound(binding):
    if not isinstance(binding, dict) or not binding.get('path') or not binding.get('sha256'):
        raise ValueError('native readiness artifact needs its exact path and SHA-256')
    path = Path(binding['path']).resolve(strict=True)
    if sha256_file_hex_cached(path) != binding['sha256']:
        raise ValueError('native readiness artifact changed: ' + str(path))
    return path


def validate_ready(path, requirement):
    path = Path(path).resolve(strict=True)
    receipt = json.loads(path.read_text())
    if receipt.get('schema') != 'natlang.native-training-readiness/1':
        raise ValueError('unsupported native training readiness receipt')
    if receipt.get('completed') is not True:
        return None
    if receipt.get('model_id') != requirement['model_id']:
        raise ValueError('native readiness model differs from its declared participant')
    identity = receipt['native_identity']
    checkpoint = receipt['checkpoint']
    if checkpoint.get('schema') != CHECKPOINT_SCHEMA or checkpoint['sha256'] != identity['checkpoint_sha256']:
        raise ValueError('native readiness checkpoint format/identity differs')
    _bound(checkpoint)
    _bound(receipt['recipe'])
    if receipt['recipe']['sha256'] != requirement['recipe_sha256']:
        raise ValueError('native readiness did not finish the declared production recipe')
    required = set(requirement['stage_ids'])
    if not required or receipt['stage_id'] not in required:
        raise ValueError('native readiness must name a declared required native stage')
    observed = {receipt['stage_id']}
    for prerequisite in receipt.get('prerequisite_stages') or []:
        if prerequisite.get('completed') is not True:
            raise ValueError('native readiness prerequisite did not complete')
        proof = json.loads(_bound(prerequisite['exposure_receipt']).read_text())
        if proof.get('completed') is not True:
            raise ValueError('native prerequisite exposure is incomplete')
        observed.add(prerequisite['id'])
    if not required <= observed:
        raise ValueError('native readiness is missing required production stages')
    report = json.loads(_bound(receipt['training_receipt']).read_text())
    exposure = receipt.get('exposure') or {}
    if (report.get('completed') is not True or report.get('checkpoint_sha256') != checkpoint['sha256']
            or exposure.get('consumed_updates', 0) <= 0 or exposure.get('supervised_token_visits', 0) <= 0):
        raise ValueError('native readiness has no bound completed optimizer/data exposure')
    runtime = json.loads(_bound(receipt['runtime_certificate']).read_text())
    if (runtime.get('schema') != 'natlang.native-runtime-certificate/1' or runtime.get('passed') is not True
            or runtime.get('native_identity') != identity or runtime.get('backbone_optimizer_tested') is not True
            or runtime.get('backbone_optimizer_updates', 0) <= 0):
        raise ValueError('native readiness runtime proof does not match the trained model')
    for name, tolerance_name in [('forward_max_abs', 'forward_atol'), ('incremental_max_abs', 'forward_atol'),
                                 ('consumer_ce_abs', 'forward_atol'), ('consumer_gradient_max_abs', 'gradient_atol')]:
        value, tolerance = runtime.get(name), runtime.get(tolerance_name)
        maximum = requirement[tolerance_name]
        if (type(value) not in (float, int) or type(tolerance) not in (float, int)
                or not math.isfinite(value) or not math.isfinite(tolerance)
                or not 0 <= value <= tolerance <= maximum):
            raise ValueError('native runtime exceeds the qualified numerical contract: ' + name)
    task = json.loads(_bound(receipt['task_execution_receipt']).read_text())
    if (task.get('schema') != 'natlang.native-task-execution-receipt/1' or task.get('passed') is not True
            or task.get('native_identity') != identity or task.get('complete') is not True
            or task.get('eligible') is not True):
        raise ValueError('native task execution proof is incomplete or bound to different weights')
    # Task-success counts remain telemetry. The owner wants capable models,
    # adaptive improvements and complete exposure, not a base-relative barrier.
    return {'model_id': receipt['model_id'], 'readiness': {'path': str(path),
            'sha256': sha256_file_hex_cached(path)}, 'checkpoint': checkpoint,
            'native_identity': identity, 'stage_id': receipt['stage_id'], 'recipe': receipt['recipe']}


def ready_participants(config):
    ready, pending = {}, []
    for participant in config['participants']:
        path = Path(participant['readiness'])
        if not path.exists():
            pending.append({'model_id': participant['model_id'], 'reason': 'native training not finished'})
            continue
        try:
            value = validate_ready(path, participant)
        except (ValueError, OSError, KeyError) as error:
            pending.append({'model_id': participant['model_id'], 'reason': str(error), 'repair_required': True})
            continue
        if value is None:
            pending.append({'model_id': participant['model_id'], 'reason': 'native prerequisites incomplete'})
            continue
        if value['model_id'] in ready:
            raise ValueError('shared-stage participants must be distinct native models')
        ready[value['model_id']] = value
    return [ready[key] for key in sorted(ready)], pending


def validate_completion(path, published):
    """An exit code cannot substitute for complete data, models and numerics."""
    from .native_semantic_stage import PairedSource, _exposure_complete
    proof = json.loads(Path(path).read_text())
    key = (proof.get('participants_signature'), proof.get('curriculum_signature'))
    if (proof.get('schema') != 'natlang.native-shared-stage-completion/1'
            or proof.get('completed') is not True or proof.get('production_ready') is not True or key not in published):
        raise ValueError('shared completion does not bind a published participant/curriculum combination')
    checkpoint = _bound(proof['checkpoint'])
    adapter = json.loads(_bound(proof['adapter']).read_text())
    if adapter.get('schema') != 'natlang.native-semantic-adapter/1' or adapter['checkpoint'] != proof['checkpoint']:
        raise ValueError('shared adapter does not bind the actual completed checkpoint')
    curriculum = json.loads(_bound(proof['curriculum']).read_text())
    if canonical_json_sha256_hex(curriculum) != proof['curriculum_signature']:
        raise ValueError('shared curriculum differs from the completed data exposure')
    models = adapter['models']
    if set(models) != set(published[key]['models']):
        raise ValueError('shared adapter omitted or substituted a ready participant')
    source = PairedSource(proof['source_bindings']['train'], split='train')
    if proof['phases'] != curriculum['phases']:
        raise ValueError('shared completion changed its required curriculum')
    complete, _ = _exposure_complete(proof['phases'], source, models, proof['exposure'])
    if not complete or proof.get('missing_exposure'):
        raise ValueError('shared completion left production paired data untrained')
    for name, entry in models.items():
        identity = entry['native_identity']
        if identity['checkpoint_sha256'] != proof['checkpoint']['sha256']:
            raise ValueError('shared participant identity is not bound to the final checkpoint')
        certificate = json.loads(_bound(proof['runtime_certificates'][name]).read_text())
        if (certificate.get('passed') is not True or certificate.get('native_identity') != identity
                or certificate.get('backbone_optimizer_updates', 0) <= 0):
            raise ValueError('shared final model lacks exact trained-state runtime qualification')
        task = json.loads(_bound(proof['task_execution_receipts'][name]).read_text())
        if (task.get('passed') is not True or task.get('complete') is not True
                or task.get('eligible') is not True or task.get('native_identity') != identity):
            raise ValueError('shared final model lacks actual bound natlang task execution')
        if not task.get('score_path') or sha256_file_hex_cached(Path(task['score_path'])) != task.get('score_sha256'):
            raise ValueError('shared task execution score changed or was never written')
    return proof, published[key]['execution_signature']


def run(config_path, *, once=False):
    config = json.loads(Path(config_path).read_text())
    if config.get('schema') != 'natlang.native-shared-stage-controller/1':
        raise ValueError('unsupported shared-stage controller configuration')
    if not once and not config.get('curriculum'):
        raise ValueError('production shared controller requires its live semantic curriculum declaration')
    command = config['command']
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError('shared-stage launch must be an explicit argv, never shell text')
    interval = config.get('poll_seconds', 60)
    if type(interval) not in (int, float) or not 0 < interval <= 60:
        raise ValueError('shared-stage readiness must be checked at least once per minute')
    out = Path(config['out']).resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child, last_success = None, None
        published = {}
        from .loop import StopSignal
        stop = StopSignal()
        if not once:
            stop.install()
        forwarded = False
        pair_cache = None
        state_path = out/'controller.json'
        if state_path.exists():
            last_success = json.loads(state_path.read_text()).get('last_successful_participants')
        while True:
            updated = json.loads(Path(config_path).read_text())
            if updated.get('schema') != config['schema'] or Path(updated['out']).resolve() != out:
                raise ValueError('live shared controller cannot change its schema/output ownership')
            config = updated
            command = config['command']
            if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
                raise ValueError('shared-stage launch must be an explicit argv')
            interval = config.get('poll_seconds', 60)
            if type(interval) not in (int, float) or not 0 < interval <= 60:
                raise ValueError('shared-stage readiness must be checked at least once per minute')
            ready, pending = ready_participants(config)
            curriculum = (json.loads(Path(config['curriculum']).read_text()) if config.get('curriculum') else {})
            if curriculum.get('pairs') and curriculum.get('held_pairs'):
                from .native_semantic_stage import PairedSource
                cache_key = (curriculum['pairs']['sha256'], curriculum['held_pairs']['sha256'])
                if pair_cache is None or pair_cache[0] != cache_key:
                    pair_cache = (cache_key, PairedSource(curriculum['pairs'], split='train'),
                                  PairedSource(curriculum['held_pairs'], split='test'))
                names = {item['model_id'] for item in ready}
                def supported(source):
                    result = set()
                    for entry in source.entries:
                        pair = set(entry['models']) & names
                        if len(pair) >= 2:
                            result.update(pair)
                    return result
                admitted = supported(pair_cache[1]) & supported(pair_cache[2])
                for item in ready:
                    if item['model_id'] not in admitted:
                        pending.append({'model_id': item['model_id'], 'native_ready': True,
                                        'reason': 'paired semantic train/held data is not admitted yet'})
                ready = [item for item in ready if item['model_id'] in admitted]
            signature = canonical_json_sha256_hex(ready)
            curriculum_signature = canonical_json_sha256_hex(curriculum)
            execution_signature = canonical_json_sha256_hex({'controller': config,
                'participants': signature, 'curriculum': curriculum_signature})
            participants = out/'participants.json'
            if len(ready) >= 2:
                manifest = {'schema': 'natlang.native-shared-participants/1', 'models': ready,
                            'signature': signature}
                # The shared trainer polls this exact appendable participant
                # manifest at update boundaries; a third ready model can join
                # without waiting for the current shared stage to finish.
                write_json_durable(participants, manifest)
                published[(signature, curriculum_signature)] = {
                    'execution_signature': execution_signature, 'models': [item['model_id'] for item in ready]}
                if child is None and execution_signature != last_success and not once and not stop.requested:
                    argv = [arg.replace('{participants}', str(participants)).replace('{out}', str(out)) for arg in command]
                    child = subprocess.Popen(argv, cwd=config.get('workdir'), start_new_session=False)
            state = {'schema': 'natlang.native-shared-controller-state/1', 'ready_models': len(ready),
                     'pending': pending, 'participants': str(participants) if len(ready) >= 2 else None,
                     'child_pid': child.pid if child else None, 'last_successful_participants': last_success,
                     'execution_signature': execution_signature, 'curriculum_signature': curriculum_signature}
            if child is not None and stop.requested and not forwarded:
                child.terminate()
                forwarded = True
            if child is not None and child.poll() is not None:
                state['exit_code'] = child.returncode
                if stop.requested:
                    state['child_pid'] = None
                    state['status'] = 'stopped-after-checkpoint-request'
                    write_json_durable(state_path, state)
                    return state
                if child.returncode != 0:
                    write_json_durable(state_path, state)
                    raise RuntimeError('shared-stage command failed; resume its preserved checkpoint')
                completion = out/'shared-stage-completion.json'
                if not completion.is_file():
                    raise RuntimeError('shared-stage command exited without actual completion/exposure receipt')
                proof, last_success = validate_completion(completion, published)
                state['last_successful_participants'] = last_success
                state['child_pid'] = None
                child = None
            write_json_durable(state_path, state)
            if once:
                return state
            if stop.requested and child is None:
                return state
            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--once', action='store_true', help='inspect/publish readiness once without launching training')
    args = parser.parse_args()
    run(args.config, once=args.once)


if __name__ == '__main__':
    main()
