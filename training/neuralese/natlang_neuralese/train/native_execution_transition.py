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
