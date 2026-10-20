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
import os
import signal
import uuid
from pathlib import Path
import subprocess
import sys
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


def _validate_training_receipt(path, requirement, *, exposure_only=False):
    if requirement.get('trainer_exit'):
        terminal = Path(requirement['trainer_exit'])
        if not terminal.is_file():
            return None  # The previous optimizer owner has not drained yet.
        exited = json.loads(terminal.read_text())
        if exited.get('returncode') != 0:
            raise ValueError('native participant trainer did not drain successfully')
        if requirement.get('trainer_launch'):
            launch = json.loads(_bound(requirement['trainer_launch']).read_text())
            trainer_recipe = requirement.get('trainer_recipe')
            launch_recipes = {trainer_recipe}
            if requirement.get('resident_adoption'):
                from .native_resident import current
                adoption_path = _bound(requirement['resident_adoption'])
                adoption = current(adoption_path.parent)
                if not adoption or adoption_path.name != 'current.json' or adoption['payload']['recipe']['path'] != trainer_recipe:
                    raise ValueError('resident launch adoption names another current recipe')
                launch_recipes = {adoption['payload']['prior_recipe']['path'],
                                  *(row['recipe']['path'] for row in adoption['history'])}
            if (type(exited.get('at')) not in (int, float) or exited['at'] < launch['at']
                    or not launch_recipes.intersection(launch.get('argv', []))):
                raise ValueError('native participant terminal exit belongs to a different trainer launch')
    path = Path(path).resolve(strict=True)
    receipt = json.loads(path.read_text())
    if receipt.get('schema') != 'natlang.native-training-readiness/1':
        raise ValueError('unsupported native training readiness receipt')
    from .native_resident import current as resident_current
    adoption_directory = path.parent.parent/'resident-data'
    adoption = resident_current(adoption_directory)
    if adoption and adoption['payload']['stage_id'] == receipt.get('stage_id'):
        if ((receipt.get('resident_adoption') or {}).get('generation') != adoption['generation']
                or receipt.get('recipe') != adoption['payload']['recipe']):
            return None  # Prior-generation readiness never certifies a new data adoption.
    foundation_boundary = requirement.get('readiness_boundary') == 'communication-foundation'
    candidate = bool(exposure_only and foundation_boundary and
                     (receipt.get('communication_foundation') or {}).get('complete') is True)
    if receipt.get('completed') is not True and not (foundation_boundary and receipt.get('foundation_ready') is True) and not candidate:
        return None
    if receipt.get('model_id') != requirement['model_id']:
        raise ValueError('native readiness model differs from its declared participant')
    identity = receipt['native_identity']
    checkpoint = receipt['checkpoint']
    if checkpoint.get('schema') != CHECKPOINT_SCHEMA or checkpoint['sha256'] != identity['checkpoint_sha256']:
        raise ValueError('native readiness checkpoint format/identity differs')
    _bound(checkpoint)
    recipe_binding = receipt['recipe']
    if recipe_binding.get('file_sha256'):
        recipe_path = _bound({'path': recipe_binding['path'], 'sha256': recipe_binding['file_sha256']})
        from .recipe import recipe_identity_sha256
        if recipe_identity_sha256(recipe_path) != recipe_binding['sha256']:
            raise ValueError('native production recipe inheritance changed after training')
    else:
        _bound(recipe_binding)
    if receipt['recipe']['sha256'] != requirement['recipe_sha256']:
        raise ValueError('native readiness did not finish the declared production recipe')
    if (requirement.get('recipe_lineage_sha256') is not None and
            recipe_binding.get('lineage_sha256',recipe_binding['sha256'])!=requirement['recipe_lineage_sha256']):
        raise ValueError('native readiness changed its declared preserved recipe lineage')
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
    if ((report.get('completed') is not True and not (foundation_boundary and report.get('foundation_ready') is True) and not candidate)
            or report.get('checkpoint_sha256') != checkpoint['sha256']
            or exposure.get('consumed_updates', 0) <= 0 or exposure.get('supervised_token_visits', 0) <= 0):
        raise ValueError('native readiness has no bound completed optimizer/data exposure')
    obligations = receipt.get('mandatory_sft')
    if obligations is not None or report.get('mandatory_sft') is not None:
        if (not isinstance(obligations, dict) or obligations != report.get('mandatory_sft')
                or obligations.get('schema') != 'natlang.native-sft-progress/1'
                or (not foundation_boundary and obligations.get('complete') is not True)):
            raise ValueError('native stage completion cannot omit mandatory pending source exposure')
    communication = receipt.get('communication_exposure') or {}
    if (communication != report.get('communication_exposure')
            or (not foundation_boundary and communication.get('complete') is not True)
            or any(type(communication.get(key)) is not int or communication[key] <= 0
                   for key in ('graphs', 'required_graph_visits', 'graph_visits'))
            or communication['required_graph_visits'] < communication['graphs']
            or (not foundation_boundary and communication['graph_visits'] < communication['required_graph_visits'])):
        raise ValueError('native readiness requires complete genuine producer/consumer communication training before VAE')
    if foundation_boundary:
        from ..data.native_sources import NativeGraphSource
        from .native_continuation import communication_foundation, validate_communication_foundation
        declared = NativeGraphSource(_bound(requirement['foundation_sources']), identity['tokenizer_fingerprint'],
            required_split='train', role_ids=report['model_config']['marker_ids'])
        foundation = receipt.get('communication_foundation') or {}
        if foundation != report.get('communication_foundation') or foundation.get('complete') is not True:
            raise ValueError('native foundation has no exact complete admitted-cohort receipt')
        validate_communication_foundation(foundation, epochs=requirement.get('foundation_epochs',1.),
            source_sha256=requirement['foundation_sources']['sha256'])
        rows = foundation['rows']
        expected = communication_foundation(declared,
            {key: value['graph_visits'] for key,value in rows.items()},
            epochs=requirement.get('foundation_epochs',1.),
            trained_bindings={key:value['binding'] for key,value in rows.items() if value.get('admitted_to_training')})
        if (expected['complete'] is not True or expected['cohort_sha256'] != foundation.get('cohort_sha256')
                or expected['rows'] != rows):
            raise ValueError('native foundation did not consume the entire declared admitted communication cohort')
        obligations = receipt.get('mandatory_sft')
        if (not isinstance(obligations,dict) or obligations != report.get('mandatory_sft')
                or obligations.get('schema') != 'natlang.native-sft-progress/1' or not obligations.get('stages')):
            raise ValueError('native foundation must retain all unfinished mandatory supervised exposure')
    return path, receipt, report


def validate_foundation_candidate(path, requirement):
    """Authenticate earned per-row exposure only; never qualify or release a model."""
    if requirement.get('readiness_boundary') != 'communication-foundation':
        raise ValueError('deferred qualification requires its explicit finite communication foundation')
    validated = _validate_training_receipt(path, requirement, exposure_only=True)
    if validated is None:
        return None
    path, receipt, _ = validated
    return {'schema': 'natlang.native-foundation-candidate/1', 'qualified': False,
            'model_id': receipt['model_id'], 'stage_id': receipt['stage_id'],
            'training_receipt': {'path': str(path), 'sha256': sha256_file_hex_cached(path)},
            'checkpoint': receipt['checkpoint'], 'native_identity': receipt['native_identity'],
            'recipe': receipt['recipe'], 'communication_foundation': receipt['communication_foundation'],
            'mandatory_sft': receipt['mandatory_sft']}


def validate_foundation_candidate_document(candidate, requirement):
    """Verify transported exposure evidence without turning it into readiness."""
    from .native_continuation import validate_communication_foundation
    def digest(value):
        return isinstance(value,str) and len(value)==64 and all(char in '0123456789abcdef' for char in value)
    identity=(candidate or {}).get('native_identity') or {}
    if not all(digest(identity.get(key)) for key in ('checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256')):
        raise ValueError('foundation candidate lacks exact checkpoint/tokenizer/embedding identities')
    if (not isinstance(candidate, dict) or candidate.get('schema') != 'natlang.native-foundation-candidate/1'
            or candidate.get('qualified') is not False or candidate.get('model_id') != requirement['model_id']
            or requirement.get('readiness_boundary') != 'communication-foundation'
            or candidate.get('stage_id') not in requirement['stage_ids']
            or (candidate.get('checkpoint') or {}).get('sha256') != (candidate.get('native_identity') or {}).get('checkpoint_sha256')
            or (candidate.get('checkpoint') or {}).get('schema') != CHECKPOINT_SCHEMA
            or (candidate.get('mandatory_sft') or {}).get('schema') != 'natlang.native-sft-progress/1'
            or not candidate['mandatory_sft'].get('stages')):
        raise ValueError('foundation candidate changed its unqualified participant/exposure contract')
    source = requirement.get('foundation_sources_sha256') or (requirement.get('foundation_sources') or {}).get('sha256')
    if not digest(source):
        raise ValueError('foundation candidate needs its declared admitted source identity')
    validate_communication_foundation(candidate.get('communication_foundation'),
        epochs=requirement.get('foundation_epochs') or 1., source_sha256=source)
    return candidate


def validate_foundation_qualification_request(path, model_id, foundation):
    """Check an explicit shared scheduling request, never a qualification certificate."""
    path = Path(path).resolve(strict=True)
    request = json.loads(path.read_bytes())
    contract = request.get('contract') or {}
    models = {value['model_id']:value for value in (contract.get('models') or {}).values()}
    candidates = request.get('candidates') or {}
    if (request.get('schema') != 'natlang.native-foundation-qualification-request/1'
            or request.get('readiness_granted') is not False or len(candidates) < 2
            or not set(candidates) <= set(models) or model_id not in candidates):
        raise ValueError('shared qualification request needs two actual declared participant candidates')
    for name,candidate in candidates.items():
        validate_foundation_candidate_document(candidate,models[name])
    if candidates[model_id]['communication_foundation']['cohort_sha256'] != (foundation or {}).get('cohort_sha256'):
        raise ValueError('shared qualification request names another current communication cohort')
    return {'path':str(path),'sha256':sha256_file_hex_cached(path),'bytes':path.stat().st_size}


def validate_ready(path, requirement):
    validated = _validate_training_receipt(path, requirement)
    if validated is None:
        return None
    path, receipt, report = validated
    checkpoint, identity = receipt['checkpoint'], receipt['native_identity']
    runtime = json.loads(_bound(receipt['runtime_certificate']).read_text())
    if (runtime.get('schema') != 'natlang.native-runtime-certificate/1' or runtime.get('passed') is not True
            or runtime.get('native_identity') != identity or runtime.get('backbone_optimizer_tested') is not True
            or runtime.get('backbone_optimizer_updates', 0) <= 0):
        raise ValueError('native readiness runtime proof does not match the trained model')
    from ..eval.native_gradients import validate_complete_proof
    if runtime.get('qualification_profile')!=report.get('qualification_profile')or not isinstance(report.get('qualification_profile'),dict):
        raise ValueError('native readiness lacks its exact communication execution profile')
    validate_complete_proof(runtime.get('complete_gradient_qualification'),identity,
        gradient_atol=requirement['gradient_atol'],execution_profile=report['qualification_profile'])
    for name, tolerance_name in [('forward_max_abs', 'forward_atol'), ('incremental_max_abs', 'forward_atol'),
                                 ('consumer_ce_abs', 'forward_atol'), ('consumer_gradient_max_abs', 'gradient_atol')]:
        value, tolerance = runtime.get(name), runtime.get(tolerance_name)
        maximum = requirement[tolerance_name]
        if (type(value) not in (float, int) or type(tolerance) not in (float, int)
                or not math.isfinite(value) or not math.isfinite(tolerance)
                or not 0 <= value <= tolerance <= maximum):
            raise ValueError('native runtime exceeds the qualified numerical contract: ' + name)
    task = json.loads(_bound(receipt['task_execution_receipt']).read_text())
    from ..eval.native_state import validate_task_execution_receipt
    if not validate_task_execution_receipt(task, identity):
        raise ValueError('native task execution proof is incomplete or bound to different weights')
    # Task-success counts remain telemetry. The owner wants capable models,
    # adaptive improvements and complete exposure, not a base-relative barrier.
    # Carry the supervisor's declared policy, never reconstruct it from a
    # receipt when a legacy participant needs provenance backfill later.
    return {'model_id': receipt['model_id'], 'readiness': {'path': str(path),
            'sha256': sha256_file_hex_cached(path)}, 'checkpoint': checkpoint,
            'native_identity': identity, 'stage_id': receipt['stage_id'], 'recipe': receipt['recipe'],
            'readiness_boundary': 'stage-complete' if receipt.get('completed') else 'communication-foundation',
            'readiness_requirement':json.loads(json.dumps(requirement)),
            'mandatory_sft': receipt.get('mandatory_sft'),
            'communication_foundation': receipt.get('communication_foundation')}


def _manage_raw_bootstrap(config):
    value = config.get('manage_raw_bootstrap', True)
    if type(value) is not bool:
        raise ValueError('manage_raw_bootstrap must be an explicit boolean')
    if not value and not config.get('raw_rl_bootstrap'):
        raise ValueError('external RAW ownership requires its bound release plan')
    return value


def _role_local_released(config):
    if not config.get('raw_rl_bootstrap'):return False
    from ..rl.native_role_release import local_released
    return local_released(config)


def ready_participants(config):
    if config.get('raw_rl_bootstrap'):
        from ..rl.native_raw_rounds import raw_out,validate_release
        raw_plan_path=_bound(config['raw_rl_bootstrap'])
        raw_plan=json.loads(raw_plan_path.read_text())
        from ..rl import native_role_release as roles
        if roles.ownership(raw_plan)[1]is not None:
            try:
                if not roles.local_released(config):
                    return [],[{'reason':'local source-host optimizer has not released current checkpoint'}]
                return roles.ready_participants(raw_plan)
            except (ValueError,OSError,KeyError)as error:
                return [],[{'reason':'source-host role handoff needs repair: '+str(error),'repair_required':True}]
        raw_state=raw_out(config)/'state.json'
        if raw_state.exists():
            try:released=validate_release(raw_state,_bound(config['raw_rl_bootstrap']))
            except (ValueError,OSError,KeyError) as error:
                return [],[{'reason':'raw optimizer handoff needs repair: '+str(error),'repair_required':True}]
            if released is None:
                return [],[{'model_id':item['model_id'],'reason':'raw optimizer owner has not released current checkpoint'}
                           for item in config['participants']]
            return [released[key] for key in sorted(released)],[]
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
    proof = json.loads(Path(path).read_text())
    key = (proof.get('participants_signature'), proof.get('curriculum_signature'))
    if (proof.get('schema') != 'natlang.native-shared-stage-completion/1'
            or not (proof.get('completed') is True or proof.get('refresh_ready') is True)
            or proof.get('production_ready') is not True or key not in published):
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
    from .native_semantic_refresh import validate_boundary
    validate_boundary(proof,adapter,curriculum)
    transport=json.loads(_bound(proof['semantic_runtime_certificate']).read_text())
    expected={(a,b) for a in models for b in models if a!=b}
    directions=transport.get('directions') or []
    if (transport.get('schema')!='natlang.native-semantic-transport-certificate/1'
            or transport.get('passed') is not True or transport.get('adapter_sha256')!=proof['adapter']['sha256']
            or {(row['source'],row['target']) for row in directions}!=expected
            or len(directions)!=len(expected)
            or any(row.get('passed') is not True or row.get('mode')!='learned-semantic-state-deltas'
                or row.get('wire_layer_tensors')!=0
                or row.get('source_native_identity')!=models[row['source']]['native_identity']
                or row.get('target_native_identity')!=models[row['target']]['native_identity'] for row in directions)):
        raise ValueError('shared boundary lacks actual compact cross-model learned transport qualification')
    for name, entry in models.items():
        identity = entry['native_identity']
        if identity['checkpoint_sha256'] != proof['checkpoint']['sha256']:
            raise ValueError('shared participant identity is not bound to the final checkpoint')
        certificate = json.loads(_bound(proof['runtime_certificates'][name]).read_text())
        if (certificate.get('passed') is not True or certificate.get('native_identity') != identity
                or certificate.get('backbone_optimizer_updates', 0) <= 0):
            raise ValueError('shared final model lacks exact trained-state runtime qualification')
        from ..eval.native_gradients import validate_complete_proof
        if not isinstance(certificate.get('qualification_profile'),dict):
            raise ValueError('shared final model lacks its complete communication execution profile')
        validate_complete_proof(certificate.get('complete_gradient_qualification'),identity,
            gradient_atol=curriculum.get('gradient_atol',1e-5),execution_profile=certificate['qualification_profile'])
        task = json.loads(_bound(proof['task_execution_receipts'][name]).read_text())
        from ..eval.native_state import validate_task_execution_receipt
        if (not validate_task_execution_receipt(task, identity)
                or task.get('semantic_adapter_sha256')!=proof['adapter']['sha256']):
            raise ValueError('shared final model lacks actual bound natlang task execution')
        if not task.get('score_path') or sha256_file_hex_cached(Path(task['score_path'])) != task.get('score_sha256'):
            raise ValueError('shared task execution score changed or was never written')
    return proof, published[key]['execution_signature']


def rl_bootstrap_command(config, completion, proof):
    """Exact optional next-stage plumbing after real shared qualification."""
    plan = _bound(config['rl_bootstrap'])
    value = json.loads(plan.read_text())
    if value.get('schema') != 'natlang.native-rl-bootstrap/1' or len(value.get('models') or []) < 2:
        raise ValueError('rl_bootstrap must declare a typed plan for both native learners')
    out = Path(config.get('rl_out', str(Path(config['out'])/'rl'))).resolve()
    argv = [sys.executable, '-m', 'natlang_neuralese.rl.native_rounds',
            '--bootstrap', str(plan), '--shared-completion', str(Path(completion).resolve()),
            '--out', str(out), '--device', config.get('rl_device', 'cuda')]
    signature = canonical_json_sha256_hex({'plan_sha256': config['rl_bootstrap']['sha256'],
        'shared_checkpoint': proof['checkpoint'], 'adapter': proof['adapter'], 'out': str(out)})
    return argv, signature


def raw_bootstrap_command(config):
    from ..rl.native_raw_rounds import raw_out
    plan=_bound(config['raw_rl_bootstrap']);value=json.loads(plan.read_text())
    if value.get('schema')!='natlang.native-raw-rl-bootstrap/1' or not value.get('models'):
        raise ValueError('raw bootstrap needs its declared actual students and original-owner handoffs')
    out=raw_out(config)
    return [sys.executable,'-m','natlang_neuralese.rl.native_rounds','--raw-bootstrap',str(plan),'--out',str(out)]


_CHILD_OWNER_ENV = 'NATLANG_NATIVE_CONTROLLER_LAUNCH'


def _owned_process_identity(pid, launch, *, root=True):
    """Exact local process identity; a recycled PID is never an owned child."""
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] in {'Z', 'X'} or proc.stat().st_uid != os.getuid():
            return None
        argv = [part.decode() for part in (proc/'cmdline').read_bytes().split(b'\0') if part]
        environment = (proc/'environ').read_bytes().split(b'\0')
        cwd = str(Path(proc/'cwd').resolve())
        if ((root and (argv != launch['argv'] or cwd != launch['cwd'])) or
                (_CHILD_OWNER_ENV + '=' + launch['token']).encode() not in environment):
            return None
        return {'pid': int(pid), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'start_ticks': fields[19], 'argv': argv, 'cwd': cwd}
    except (OSError, ValueError, UnicodeError):
        return None


def _owned_tree(launch):
    result = []
    for proc in Path('/proc').iterdir():
        if proc.name.isdigit():
            identity = _owned_process_identity(int(proc.name), launch, root=False)
            if identity is not None:
                result.append(identity)
    return result


def _terminate_owned(launch, identity):
    try:
        descriptor = os.pidfd_open(identity['pid'])
    except ProcessLookupError:
        return
    try:
        if _owned_process_identity(identity['pid'], launch, root=False) == identity:
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
    except ProcessLookupError:
        pass
    finally:
        os.close(descriptor)


class _RecoveredChild:
    """Drain an exact surviving launch tree; disappearance requires resume."""
    def __init__(self, launch, identity, survivors):
        self.launch, self.identity = launch, identity
        self.pid = identity['pid'] if identity is not None else survivors[0]['pid']
        self.returncode, self.forwarded = None, set()

    def _drain(self, survivors):
        for identity in survivors:
            key = (identity['boot_id'], identity['pid'], identity['start_ticks'])
            if key not in self.forwarded:
                _terminate_owned(self.launch, identity)
                self.forwarded.add(key)

    def poll(self):
        survivors = _owned_tree(self.launch)
        if self.identity is not None and self.identity in survivors:
            return None
        # The original supervisor vanished. Its descendants still own artifacts
        # and may hold learner state: checkpoint/drain them before any retry.
        self._drain(survivors)
        if not survivors:
            # waitpid is unavailable after reparenting. Never invent success.
            self.returncode = 1
        return self.returncode

    def terminate(self):
        if self.identity is not None and _owned_process_identity(self.identity['pid'], self.launch) == self.identity:
            _terminate_owned(self.launch, self.identity)
        else:
            self._drain(_owned_tree(self.launch))


def _recover_child(launch):
    if not launch or launch.get('status') not in {'launching', 'running'}:
        return None
    survivors = _owned_tree(launch)
    matches = [identity for identity in survivors
               if identity['argv'] == launch['argv'] and identity['cwd'] == launch['cwd']]
    saved = launch.get('identity')
    if saved is not None:
        # A fork may briefly retain its parent's argv before exec. The saved
        # boot/start identity identifies the original root among descendants.
        identity = next((entry for entry in matches if entry == saved), None)
        if identity is None and any(entry['pid'] == saved['pid'] for entry in matches):
            raise RuntimeError('native controller child process identity changed')
    else:
        # For a crash before the initial PID receipt, identify the unique root
        # of matching processes, rather than a just-forked pre-exec child.
        pids = {entry['pid'] for entry in matches}
        roots = []
        for entry in matches:
            try:
                parent = int((Path('/proc')/str(entry['pid'])/'stat').read_text().rsplit(')', 1)[1].split()[1])
            except (OSError, ValueError):
                continue
            if parent not in pids:
                roots.append(entry)
        if len(roots) > 1:
            raise RuntimeError('multiple processes claim the exact native controller launch; resolve ownership')
        identity = roots[0] if roots else None
    return _RecoveredChild(launch, identity, survivors) if survivors else None


def _launch_child(argv, kind, config, out, signature, rl_signature, *, semantic_completion=None):
    # Persist the unique launch before spawning, so a supervisor killed between
    # Popen and its next state write can rediscover the exact child safely.
    launch = {'schema': 'natlang.native-controller-child/1', 'status': 'launching',
        'token': uuid.uuid4().hex, 'argv': argv, 'cwd': str(Path(config.get('workdir') or '.').resolve()),
        'out': str(out), 'child_kind': kind, 'execution_signature': signature,
        'rl_signature': rl_signature, 'semantic_completion':str(semantic_completion) if semantic_completion else None}
    path = out/'child-launch.json'
    write_json_durable(path, launch)
    try:
        child = subprocess.Popen(argv, cwd=config.get('workdir'), start_new_session=False,
                                 env={**os.environ, _CHILD_OWNER_ENV: launch['token']})
    except OSError:
        write_json_durable(path, {**launch, 'status': 'launch-failed'})
        raise
    # Popen's exec handshake can finish while /proc still exposes a transient
    # cmdline/environ view. Persisting that first miss loses the root identity
    # needed to adopt a live child after controller restart. The durable launch
    # token already exists, so interruption during this observation is safe.
    # This is a bounded observation, not a prerequisite for running the child:
    # wrappers may legitimately expose different argv, and a vanished process
    # must never leave the controller stuck before supervision starts. Recovery
    # can still find and drain token-owned descendants without a root identity.
    identity = _owned_process_identity(child.pid, launch)
    for _ in range(100):
        if (identity is not None or child.poll() is not None
                or not (Path('/proc') / str(child.pid)).exists()):
            break
        time.sleep(.001)
        identity = _owned_process_identity(child.pid, launch)
    launch.update(status='running', identity=identity)
    write_json_durable(path, launch)
    return child, launch


def _retry_failure(previous, kind, signature, interval, *, exit_code=None, error=None):
    """Persist recoverable execution failure without inventing stage completion."""
    now = time.time()
    return {'child_kind': kind, 'exit_code': exit_code,
        'attempts': (previous.get('attempts', 0) if previous else 0) + 1,
        'last_failed_at': now, 'next_retry_at': now + interval,
        'reason': (f'{kind} launch failed: {error}' if error is not None else
                   f'{kind} child failed; retry the preserved state after repair'),
        'execution_signature': signature}


def _run(config_path, *, once=False, stop):
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
        child, child_kind, last_success, rl_last_success, active_rl_signature = None, None, None, None, None
        published = {}
        semantic_publication=None
        forwarded = False
        pair_cache = None
        retry = None
        state_path = out/'controller.json'
        if state_path.exists():
            previous_state = json.loads(state_path.read_text())
            last_success = previous_state.get('last_successful_participants')
            rl_last_success = previous_state.get('last_successful_rl_bootstrap')
            retry = previous_state.get('retry')
            semantic_publication=previous_state.get('last_semantic_publication')
        launch_path = out/'child-launch.json'
        launch = json.loads(launch_path.read_text()) if launch_path.exists() else None
        if not once and launch and launch.get('status') in {'launching', 'running'}:
            if launch.get('out') != str(out):
                raise ValueError('native controller child belongs to another output owner')
            child = _recover_child(launch)
            if child is not None:
                child_kind = launch['child_kind']
                active_rl_signature = launch.get('rl_signature')
            else:
                retry = _retry_failure(retry, launch['child_kind'], launch['execution_signature'], interval,
                                       error='previous owned child ended while controller was absent')
                write_json_durable(launch_path, {**launch, 'status': 'ended-unobserved'})
        elif not once and launch is None and state_path.exists() and previous_state.get('child_pid'):
            # Historical states lack ownership evidence: do not signal or replay
            # over a possibly surviving worker using a bare PID.
            if Path('/proc', str(previous_state['child_pid'])).exists():
                raise RuntimeError('historical native controller child lacks exact ownership; drain it before restarting')
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
            manage_raw = _manage_raw_bootstrap(config)
            ready, pending = ready_participants(config)
            curriculum = (json.loads(Path(config['curriculum']).read_text()) if config.get('curriculum') else {})
            # Pending external owners need polling, not a full paired-row index.
            # Once two owners qualify, validate every train/held row before launch.
            if len(ready) >= 2 and curriculum.get('pairs') and curriculum.get('held_pairs'):
                from .native_semantic_stage import PairedSource,validate_config
                curriculum=validate_config(curriculum)
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
            retry_ready = retry is None or time.time() >= retry['next_retry_at']
            if (manage_raw and child is None and len(ready)<2 and config.get('raw_rl_bootstrap') and not last_success
                    and not semantic_publication and not once and not stop.requested and retry_ready
                    and not _role_local_released(config)):
                try:
                    child,launch=_launch_child(raw_bootstrap_command(config),'native-raw-rl',config,out,execution_signature,None)
                    child_kind='native-raw-rl'
                except (OSError,ValueError) as error:
                    retry=_retry_failure(retry,'native-raw-rl',execution_signature,interval,error=error)
            if len(ready) >= 2:
                manifest = {'schema': 'natlang.native-shared-participants/1', 'models': ready,
                            'signature': signature}
                # The shared trainer polls this exact appendable participant
                # manifest at update boundaries; a third ready model can join
                # without waiting for the current shared stage to finish.
                write_json_durable(participants, manifest)
                published[(signature, curriculum_signature)] = {
                    'execution_signature': execution_signature, 'models': [item['model_id'] for item in ready]}
                completion = out/'shared-stage-completion.json'
                semantic_out=out
                semantic_pending=False
                if not config.get('rl_bootstrap'):
                    # Recover a successfully published child even when the
                    # controller died before its following state commit.
                    recovered=Path((launch or {}).get('semantic_completion') or completion)
                    if child is None and recovered.is_file():
                        try:
                            candidate,candidate_success=validate_completion(recovered,published)
                        except Exception:
                            pass  # ordinary launch/retry path reports invalid evidence
                        else:
                            semantic_publication={'path':str(recovered.resolve()),'sha256':sha256_file_hex_cached(recovered)}
                            last_success=candidate_success
                            retry=None;retry_ready=True
                    if semantic_publication:
                        try:
                            previous_path=_bound(semantic_publication)
                            previous=json.loads(previous_path.read_text())
                            previous_key=(previous['participants_signature'],previous['curriculum_signature'])
                            previous_adapter=json.loads(_bound(previous['adapter']).read_text())
                            previous_published={previous_key:{'models':list(previous_adapter['models']),
                                'execution_signature':last_success}}
                            previous,_=validate_completion(previous_path,previous_published)
                            semantic_pending=(previous.get('semantic_obligations') or {}).get('complete') is False
                        except Exception as error:
                            retry=_retry_failure(retry,'native-semantic',execution_signature,interval,error=error)
                            retry_ready=False
                            pending.append({'reason':'semantic publication needs repair: '+str(error),'repair_required':True})
                        else:
                            if semantic_pending or execution_signature!=last_success:
                                if not any('{out}' in arg for arg in command) or not any('{participants}' in arg for arg in command):
                                    raise ValueError('autonomous semantic continuation requires {out} and {participants} command placeholders')
                                parent={'completion':semantic_publication,'checkpoint':previous['checkpoint'],'adapter':previous['adapter']}
                                semantic_out=out/'semantic-continuations'/canonical_json_sha256_hex(parent)
                                semantic_out.mkdir(parents=True,exist_ok=True)
                                participants=semantic_out/'participants.json'
                                manifest['semantic_parent']=parent
                                write_json_durable(participants,manifest)
                                completion=semantic_out/'shared-stage-completion.json'
                if (child is None and config.get('rl_bootstrap') and completion.is_file()
                        and not once and not stop.requested and retry_ready):
                    proof = None
                    try:
                        candidate = json.loads(completion.read_text())
                        candidate_key = (candidate.get('participants_signature'), candidate.get('curriculum_signature'))
                        if candidate_key == (signature, curriculum_signature):
                            proof, last_success = validate_completion(completion, published)
                        elif last_success == execution_signature:
                            last_success = None
                    except Exception as error:
                        last_success = None
                        pending.append({'reason': 'shared completion rejected: ' + str(error), 'repair_required': True})
                    if proof is not None:
                        try:
                            argv, active_rl_signature = rl_bootstrap_command(config, completion, proof)
                        except Exception as error:
                            pending.append({'reason': 'RL bootstrap requires repair: ' + str(error), 'repair_required': True})
                        else:
                            if active_rl_signature != rl_last_success:
                                try:
                                    child, launch = _launch_child(argv, 'native-rl', config, out, execution_signature, active_rl_signature)
                                    child_kind = 'native-rl'
                                except OSError as error:
                                    retry = _retry_failure(retry, 'native-rl', execution_signature, interval, error=error)
                                    retry_ready = False
                if child is None and (execution_signature != last_success or semantic_pending) and not once and not stop.requested and retry_ready:
                    argv = [arg.replace('{participants}', str(participants)).replace('{out}', str(semantic_out)) for arg in command]
                    try:
                        child, launch = _launch_child(argv, 'native-semantic', config, out, execution_signature, None,semantic_completion=completion)
                        child_kind = 'native-semantic'
                    except OSError as error:
                        retry = _retry_failure(retry, 'native-semantic', execution_signature, interval, error=error)
            state = {'schema': 'natlang.native-shared-controller-state/1', 'ready_models': len(ready),
                     'manage_raw_bootstrap': manage_raw,
                     'pending': pending, 'participants': str(participants) if len(ready) >= 2 else None,
                     'child_pid': child.pid if child else None, 'child_kind': child_kind,
                     'child_ownership': launch if child else None,
                     'last_successful_participants': last_success,
                     'last_successful_rl_bootstrap': rl_last_success,
                     'last_semantic_publication':semantic_publication,
                     'execution_signature': execution_signature, 'curriculum_signature': curriculum_signature,
                     'retry': retry, 'status': 'running' if child else 'retry_wait' if retry else 'monitoring'}
            if child is not None and stop.requested and not forwarded:
                child.terminate()
                forwarded = True
            if child is not None and child.poll() is not None and not isinstance(child, _RecoveredChild):
                survivors = _recover_child(launch)
                if survivors is not None:
                    child = survivors
                    state['child_pid'] = child.pid
            if child is not None and child.poll() is not None:
                state['exit_code'] = child.returncode
                launch={**launch,'status':'exited','exit_code':child.returncode}
                write_json_durable(launch_path,launch)
                if stop.requested:
                    state['child_pid'] = None
                    state['status'] = 'stopped-after-checkpoint-request'
                    write_json_durable(state_path, state)
                    return state
                if child.returncode != 0:
                    retry = _retry_failure(retry, child_kind, execution_signature, interval, exit_code=child.returncode)
                    state.update(retry=retry, status='retry_wait', child_pid=None, child_kind=None)
                    child, child_kind = None, None
                    forwarded = False
                    write_json_durable(state_path, state)
                    time.sleep(interval)
                    continue
                if child_kind == 'native-raw-rl':
                    from ..rl.native_role_release import seal_local
                    sealed=seal_local(config)
                    from ..rl.native_raw_rounds import raw_out,validate_release
                    if sealed is None and validate_release(raw_out(config)/'state.json',_bound(config['raw_rl_bootstrap'])) is None:
                        retry=_retry_failure(retry,child_kind,execution_signature,interval,
                            error='raw owner exited without current checkpoint/readiness release')
                        state.update(retry=retry,status='retry_wait',child_pid=None,child_kind=None)
                        child,child_kind=None,None;forwarded=False
                        write_json_durable(state_path,state);time.sleep(interval);continue
                elif child_kind == 'native-rl':
                    rl_last_success = active_rl_signature
                    state['last_successful_rl_bootstrap'] = rl_last_success
                else:
                    completion = Path(launch.get('semantic_completion') or out/'shared-stage-completion.json')
                    try:
                        if not completion.is_file():
                            raise RuntimeError('shared-stage command exited without actual completion/exposure receipt')
                        proof, last_success = validate_completion(completion, published)
                    except Exception as error:
                        retry = _retry_failure(retry, child_kind, execution_signature, interval, exit_code=0)
                        retry['reason'] = 'shared completion rejected: ' + type(error).__name__ + ': ' + str(error)
                        state.update(retry=retry, status='retry_wait', child_pid=None, child_kind=None)
                        child, child_kind = None, None
                        forwarded = False
                        write_json_durable(state_path, state)
                        time.sleep(interval)
                        continue
                    state['last_successful_participants'] = last_success
                    semantic_publication={'path':str(completion.resolve()),'sha256':sha256_file_hex_cached(completion)}
                    state['last_semantic_publication']=semantic_publication
                retry = None
                state['retry'] = None
                state['child_pid'] = None
                state['child_kind'] = None
                child, child_kind = None, None
            if stop.requested and child is None:
                state['status'] = 'stopped-after-checkpoint-request'
            write_json_durable(state_path, state)
            if once:
                return state
            if stop.requested and child is None:
                return state
            time.sleep(interval)


def run(config_path, *, once=False):
    from .loop import StopSignal
    stop = StopSignal()
    if not once:
        stop.install()
    try:
        return _run(config_path, once=once, stop=stop)
    finally:
        if not once:
            stop.restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--once', action='store_true', help='inspect/publish readiness once without launching training')
    args = parser.parse_args()
    run(args.config, once=args.once)


if __name__ == '__main__':
    main()
