"""Exact native-model readiness receipts and owned child-process supervision.

``validate_ready`` authenticates a completed native training receipt against its declared requirement;
``_launch_child``/``_recover_child`` give provider workers exact, restart-safe ownership of their children.
"""
from __future__ import annotations

import json
import math
import os
import signal
import uuid
from pathlib import Path
import subprocess
import time

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from ..model.native_model import CHECKPOINT_SCHEMA


def _bound(binding):
    if not isinstance(binding, dict) or not binding.get('path') or not binding.get('sha256'):
        raise ValueError('native readiness artifact needs its exact path and SHA-256')
    path = Path(binding['path']).resolve(strict=True)
    if sha256_file_hex_cached(path) != binding['sha256']:
        raise ValueError('native readiness artifact changed: ' + str(path))
    return path


def _validate_training_receipt(path, requirement):
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
    if receipt.get('completed') is not True and not (foundation_boundary and receipt.get('foundation_ready') is True):
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
    if ((report.get('completed') is not True and not (foundation_boundary and report.get('foundation_ready') is True))
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


def _launch_child(argv, kind, config, out, signature, rl_signature):
    # Persist the unique launch before spawning, so a supervisor killed between
    # Popen and its next state write can rediscover the exact child safely.
    launch = {'schema': 'natlang.native-controller-child/1', 'status': 'launching',
        'token': uuid.uuid4().hex, 'argv': argv, 'cwd': str(Path(config.get('workdir') or '.').resolve()),
        'out': str(out), 'child_kind': kind, 'execution_signature': signature,
        'rl_signature': rl_signature}
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
