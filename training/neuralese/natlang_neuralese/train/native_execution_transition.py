"""Pure zero-SGD execution-profile derivation and exact independent verification.

This module does not publish a controller head or assert numerical/behavioral
qualification.  Callers must separately prove the resulting actor and data
collection before using a new policy for learning.
"""
from __future__ import annotations

SCHEMA = 'natlang.native-zero-sgd-execution-transition/1'
SOURCE = 'source-v1'
TARGET = 'ffn-stable-v1'


def derive(parent, *, parent_sha256):
    from ..model.native_model import CHECKPOINT_SCHEMA
    from ..rl.native_raw_policy import require_raw_training
    if (not isinstance(parent_sha256, str) or len(parent_sha256) != 64
            or any(c not in '0123456789abcdef' for c in parent_sha256)):
        raise ValueError('transition requires the exact full parent SHA256')
    if not isinstance(parent, dict) or parent.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('execution transition requires a full native checkpoint')
    config = parent.get('model_config'); interface = parent.get('interface_state'); training = parent.get('training')
    if not all(isinstance(x, dict) for x in (config, interface, training)):
        raise ValueError('execution transition requires full config, interface and training state')
    if config.get('architecture') != 'lfm3':
        raise ValueError('execution transition is currently restricted to LFM3')
    require_raw_training(parent)
    if config.get('native_shared_execution', SOURCE) != SOURCE or interface.get('native_shared_execution', SOURCE) != SOURCE:
        raise ValueError('execution transition requires a consistent source-v1 parent')
    step = training.get('global_step', training.get('step'))
    if type(step) is not int or step < 0:
        raise ValueError('execution transition requires its actual nonnegative earned step')
    options = training.get('options') or {}
    if not isinstance(options, dict):
        raise ValueError('checkpoint training options are malformed')
    history = list(options.get('native_shared_execution_history') or [])
    history.append({'from': SOURCE, 'to': TARGET, 'at_global_step': step,
                    'parent_checkpoint_sha256': parent_sha256,
                    'weights_optimizer_rng_exposure': 'preserved',
                    'qualification': 'pending'})
    child = dict(parent)
    child['model_config'] = {**config, 'native_shared_execution': TARGET}
    child['interface_state'] = {**interface, 'native_shared_execution': TARGET}
    child['training'] = {**training, 'options': {**options, 'native_shared_execution_history': history}}
    return child


def verify_states(parent, child, *, parent_sha256):
    from .native_zero_sgd_adoption import _same
    expected = derive(parent, parent_sha256=parent_sha256)
    _same(expected, child)
    return {'schema': SCHEMA, 'parent_sha256': parent_sha256,
            'source_execution': SOURCE, 'target_execution': TARGET,
            'global_step': parent['training'].get('global_step', parent['training'].get('step')),
            'optimizer_updates': 0, 'graph_visits': 0,
            'actor_identity': 'must be derived from serialized child bytes',
            'fresh_collection_required': True,
            'qualification': 'pending CUDA runtime, packed replay and held behavior'}


def verify_files(parent_ref, child_ref):
    """Independently reread and compare full checkpoint bytes and derive both identities."""
    from pathlib import Path
    import torch
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    from ..rl.native_raw_policy import saved_native_identity
    def stamp(path):
        st = path.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    old = _verify_historical_file_ref(parent_ref)
    new = _verify_historical_file_ref(child_ref)
    if old == new or parent_ref['sha256'] == child_ref['sha256']:
        raise ValueError('execution transition requires distinct retained parent and child bytes')
    old_stamp, new_stamp = stamp(old), stamp(new)
    parent = torch.load(old, map_location='cpu', weights_only=False, mmap=True)
    child = torch.load(new, map_location='cpu', weights_only=False, mmap=True)
    report = verify_states(parent, child, parent_sha256=parent_ref['sha256'])
    old_identity = saved_native_identity(parent, checkpoint_sha256=parent_ref['sha256'])
    new_identity = saved_native_identity(child, checkpoint_sha256=child_ref['sha256'])
    if old_identity == new_identity:
        raise ValueError('execution transition did not produce a distinct native actor identity')
    if (stamp(old) != old_stamp or stamp(new) != new_stamp
            or _verify_historical_file_ref(parent_ref) != old
            or _verify_historical_file_ref(child_ref) != new):
        raise ValueError('checkpoint changed during independent execution-transition verification')
    return {**report, 'parent': dict(parent_ref), 'checkpoint': dict(child_ref),
            'actor_identity': {'old': old_identity, 'new': new_identity}}


def publish_diagnostic(state_path, *, proof_ref, out):
    """Register an independently reverified candidate without moving the production head.

    This is deliberately not an authorization to collect or train on the
    candidate: it creates a persistent selection hold.  Only a future typed
    qualification admission may clear that hold.
    """
    import fcntl
    import json
    from pathlib import Path
    from ..common.jsonio import write_json_durable
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref, _strong_binding
    state_path = Path(state_path).resolve(strict=True)
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('execution transition publication receipt already exists')
    proof_path = _verify_historical_file_ref(proof_ref)
    proof = json.loads(proof_path.read_text())
    if proof.get('schema') != SCHEMA:
        raise ValueError('execution transition publication requires its typed proof')
    checked = verify_files(proof['parent'], proof['checkpoint'])
    if proof != checked:
        raise ValueError('execution transition proof differs from independent full-state rederivation')
    def stamp(path):
        s = Path(path).stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    observed = {name: stamp(proof[name]['path']) for name in ('parent', 'checkpoint')}
    proof_stamp = stamp(proof_path)
    with (state_path.parent / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text())
        if state.get('active_update_round') or state.get('phase') in {'updating', 'publishing', 'learner_pending'}:
            raise ValueError('cannot publish an execution candidate during an owned update')
        if out.exists() or state.get('execution_transition_candidate'):
            raise ValueError('execution transition candidate or receipt already exists')
        parent_path = Path(proof['parent']['path']).resolve(strict=True)
        if Path(state['checkpoint']).resolve(strict=True) != parent_path:
            raise ValueError('execution candidate parent differs from the current controller head')
        if (stamp(proof_path) != proof_stamp or any(stamp(proof[k]['path']) != observed[k] for k in observed)
                or _verify_historical_file_ref(proof_ref) != proof_path
                or any(_verify_historical_file_ref(proof[k]) != Path(proof[k]['path']).resolve(strict=True)
                       for k in observed)):
            raise ValueError('execution candidate evidence changed before publication')
        result = {'schema':'natlang.native-execution-candidate-publication/1', 'proof':dict(proof_ref),
                  'parent':proof['parent'], 'checkpoint':proof['checkpoint'],
                  'actor_identity':proof['actor_identity'], 'optimizer_updates':0, 'graph_visits':0,
                  'selection_hold':True, 'requirements':['cuda-forward-backward','fresh-actor-collection',
                      'complete-packed-replay','held-behavior']}
        write_json_durable(out, result)
        next_state = dict(state)
        next_state['execution_transition_candidate'] = _strong_binding(out)
        write_json_durable(state_path, next_state)
        return result


def verify_candidate_publication(publication_ref):
    """Reauthenticate a held diagnostic candidate and its exact source ancestry."""
    import json
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    publication = json.loads(_verify_historical_file_ref(publication_ref).read_text())
    if publication.get('schema') != 'natlang.native-execution-candidate-publication/1':
        raise ValueError('unsupported execution candidate publication')
    proof_ref = publication.get('proof') or {}
    proof = json.loads(_verify_historical_file_ref(proof_ref).read_text())
    checked = verify_files(proof['parent'], proof['checkpoint'])
    if (proof != checked or publication.get('parent') != checked['parent']
            or publication.get('checkpoint') != checked['checkpoint']
            or publication.get('actor_identity') != checked['actor_identity']
            or publication.get('optimizer_updates') != 0 or publication.get('graph_visits') != 0
            or publication.get('selection_hold') is not True
            or publication.get('requirements') != ['cuda-forward-backward','fresh-actor-collection',
                'complete-packed-replay','held-behavior']):
        raise ValueError('execution candidate publication differs from its exact held derivation')
    return {'publication': publication, 'proof': checked,
            'authenticated_ancestors': [checked['parent']['sha256'], checked['checkpoint']['sha256']]}


def prepare_candidate(parent_ref, *, output):
    """Stage a new full candidate with a physical reservation; never publish it.

    This operation is intentionally separate from diagnostic controller
    publication and must not be invoked until a candidate is authorized to be
    created and the storage floor can be met.
    """
    import os
    from pathlib import Path
    import torch
    from ..common.jsonio import write_json_durable
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref, _strong_binding
    from .checkpoint_safety import PersistentCheckpointReservation, checkpoint_size_upper_bound
    from .trajectory_state import atomic_checkpoint
    parent_path = _verify_historical_file_ref(parent_ref)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    parent = torch.load(parent_path, map_location='cpu', weights_only=False, mmap=True)
    candidate = derive(parent, parent_sha256=parent_ref['sha256'])
    target = output / 'checkpoint.pt'
    reservation = PersistentCheckpointReservation(target, minimum_free_bytes=40 * 1024**3)
    try:
        reservation.acquire(checkpoint_size_upper_bound(candidate))
        atomic_checkpoint(target, candidate, reservation=reservation)
    finally:
        reservation.close()  # a failed nonempty pending inode remains custody evidence
    os.chmod(target, 0o444)
    child_ref = _strong_binding(target)
    proof = verify_files(parent_ref, child_ref)
    write_json_durable(output / 'proof.json', proof)
    return proof


def require_qualified_for_training(training):
    """Fail closed on this transition's held checkpoint before optimizer work."""
    history = ((training or {}).get('options') or {}).get('native_shared_execution_history') or []
    if not isinstance(history, list):
        raise ValueError('native execution transition history is malformed')
    for entry in history:
        if (isinstance(entry, dict) and entry.get('from') == SOURCE and entry.get('to') == TARGET
                and entry.get('qualification') == 'pending'):
            raise ValueError('FFN execution transition is held pending authenticated CUDA, fresh replay and held behavior qualification')
