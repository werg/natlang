"""Pure zero-SGD execution-profile derivation and exact independent verification.

This module does not publish a controller head or assert numerical/behavioral
qualification.  Callers must separately prove the resulting actor and data
collection before using a new policy for learning.
"""
from __future__ import annotations

from collections import OrderedDict
from ..rl.native_direct_sft_recovery import with_historical_file_verification_session

# Repeated admission layers in one process authenticate the same full tensors.
# Cache only a completed independent comparison, keyed by both exact references
# and their current five-field filesystem identities. Cross-process reuse is
# separately authenticated by native_transition_comparison_cache; both layers
# retain no tensors and require fresh file authentication per admission.
_VERIFIED_FILE_TRANSITIONS = OrderedDict()

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
            'qualification': 'pending CUDA runtime, packed replay and protected development behavior',
            'untouched_confirmation': False}


@with_historical_file_verification_session
def verify_files(parent_ref, child_ref, *, proof_ref=None):
    """Freshly hash both checkpoints and verify their exact full-state transition.

    A cross-process comparison memo is usable only with an independently
    authenticated proof; checkpoint byte hashes remain fresh per operation.
    """
    from pathlib import Path
    import copy
    import json
    import torch
    from ..common.jsonio import canonical_json_bytes
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    from ..rl.native_raw_policy import saved_native_identity
    from . import native_transition_comparison_cache as comparison_cache
    def stamp(path):
        st = path.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    old = _verify_historical_file_ref(parent_ref)
    new = _verify_historical_file_ref(child_ref)
    if old == new or parent_ref['sha256'] == child_ref['sha256']:
        raise ValueError('execution transition requires distinct retained parent and child bytes')
    old_stamp, new_stamp = stamp(old), stamp(new)
    expected = None
    proof_path = proof_stamp = None
    if proof_ref is not None:
        proof_path = _verify_historical_file_ref(proof_ref)
        proof_stamp = stamp(proof_path)
        expected = json.loads(proof_path.read_text())
        _verify_historical_file_ref(proof_ref)
        if not isinstance(expected, dict) or expected.get('schema') != SCHEMA:
            raise ValueError('execution transition comparison needs a typed independent proof')
    try:
        verifier = comparison_cache.verifier_identity()
    except (OSError, ValueError, AttributeError, TypeError):
        verifier = None
    key = (str(old), str(new), old_stamp, new_stamp,
           json.dumps(parent_ref, sort_keys=True), json.dumps(child_ref, sort_keys=True), verifier,
           None if proof_path is None else str(proof_path), proof_stamp,
           None if proof_ref is None else json.dumps(proof_ref, sort_keys=True))
    persistent_key = ({'parent': dict(parent_ref), 'checkpoint': dict(child_ref),
                       'parent_path': str(old), 'checkpoint_path': str(new),
                       'parent_identity': list(old_stamp), 'checkpoint_identity': list(new_stamp),
                       'verifier': verifier, 'proof': dict(proof_ref),
                       'proof_path': str(proof_path), 'proof_identity': list(proof_stamp)}
                      if verifier is not None and proof_ref is not None else None)
    result = None
    persistent_hit = False
    if verifier is not None and key in _VERIFIED_FILE_TRANSITIONS:
        result = copy.deepcopy(_VERIFIED_FILE_TRANSITIONS[key])
        _VERIFIED_FILE_TRANSITIONS.move_to_end(key)
    elif persistent_key is not None:
        result = comparison_cache.lookup(persistent_key, expected)
        persistent_hit = result is not None
    if result is not None:
        if (stamp(old) != old_stamp or stamp(new) != new_stamp
                or _verify_historical_file_ref(parent_ref) != old
                or _verify_historical_file_ref(child_ref) != new):
            raise ValueError('checkpoint changed during cached execution-transition verification')
        if expected is not None and canonical_json_bytes(result) != canonical_json_bytes(expected):
            raise ValueError('execution transition proof differs from independent full-state rederivation')
        if proof_ref is not None:
            if _verify_historical_file_ref(proof_ref) != proof_path or stamp(proof_path) != proof_stamp:
                raise ValueError('execution transition proof changed during cached comparison')
        if comparison_cache.verifier_identity() != verifier:
            raise ValueError('transition verifier changed during cached comparison')
        if persistent_key is not None and not persistent_hit:
            comparison_cache.store(persistent_key, result)
        return copy.deepcopy(result)
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
    result = {**report, 'parent': dict(parent_ref), 'checkpoint': dict(child_ref),
              'actor_identity': {'old': old_identity, 'new': new_identity}}
    if expected is not None and canonical_json_bytes(result) != canonical_json_bytes(expected):
        raise ValueError('execution transition proof differs from independent full-state rederivation')
    if proof_ref is not None:
        if _verify_historical_file_ref(proof_ref) != proof_path or stamp(proof_path) != proof_stamp:
            raise ValueError('execution transition proof changed during full comparison')
    if verifier is not None:
        if comparison_cache.verifier_identity() != verifier:
            raise ValueError('transition verifier changed during full comparison')
        _VERIFIED_FILE_TRANSITIONS[key] = copy.deepcopy(result)
        while len(_VERIFIED_FILE_TRANSITIONS) > 8:
            _VERIFIED_FILE_TRANSITIONS.popitem(last=False)
        if persistent_key is not None:
            comparison_cache.store(persistent_key, result)
    return result


def publish_diagnostic(state_path, *, proof_ref, out):
    """Register an independently reverified candidate without moving the production head.

    Exact-candidate diagnostic collection and forward qualification are
    allowed under the hold. Production selection and optimizer work require
    the separate typed qualification and selection publication.
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
    checked = verify_files(proof['parent'], proof['checkpoint'], proof_ref=proof_ref)
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
                      'complete-packed-replay','paired-protected-dev-behavior']}
        write_json_durable(out, result)
        next_state = dict(state)
        next_state['execution_transition_candidate'] = _strong_binding(out)
        write_json_durable(state_path, next_state)
        return result


@with_historical_file_verification_session
def verify_candidate_publication(publication_ref):
    """Reauthenticate a held diagnostic candidate and its exact source ancestry."""
    import json
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    publication = json.loads(_verify_historical_file_ref(publication_ref).read_text())
    if publication.get('schema') != 'natlang.native-execution-candidate-publication/1':
        raise ValueError('unsupported execution candidate publication')
    proof_ref = publication.get('proof') or {}
    proof = json.loads(_verify_historical_file_ref(proof_ref).read_text())
    checked = verify_files(proof['parent'], proof['checkpoint'], proof_ref=proof_ref)
    if (proof != checked or publication.get('parent') != checked['parent']
            or publication.get('checkpoint') != checked['checkpoint']
            or publication.get('actor_identity') != checked['actor_identity']
            or publication.get('optimizer_updates') != 0 or publication.get('graph_visits') != 0
            or publication.get('selection_hold') is not True
            or publication.get('requirements') != ['cuda-forward-backward','fresh-actor-collection',
                'complete-packed-replay','paired-protected-dev-behavior']):
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


def require_qualified_for_training(training, *, admission_ref=None, checkpoint_path=None):
    """Authenticate an external selection; never rewrite the immutable candidate."""
    from pathlib import Path
    from ..common.hashing import sha256_file_hex_cached
    training = training or {}
    options = training.get('options') or {}
    history = options.get('native_shared_execution_history') or []
    if not isinstance(history, list):
        raise ValueError('native execution transition history is malformed')
    targets = [i for i, entry in enumerate(history) if isinstance(entry, dict)
               and entry.get('from') == SOURCE and entry.get('to') == TARGET
               and ('qualification' in entry)]
    if not targets:
        if admission_ref is not None:
            raise ValueError('execution admission has no matching checkpoint transition')
        return training
    if len(targets) != 1:
        raise ValueError('checkpoint has ambiguous FFN execution transition ancestry')
    index = targets[0]; entry = history[index]; recorded = entry.get('qualification')
    if recorded == 'pending':
        if admission_ref is None or checkpoint_path is None:
            raise ValueError('FFN execution transition is held pending authenticated CUDA, fresh replay and development behavior qualification')
        selection = verify_selected_publication(admission_ref)
        if sha256_file_hex_cached(Path(checkpoint_path).resolve(strict=True)) != selection['proof']['checkpoint']['sha256']:
            raise ValueError('execution admission is for a different immutable candidate checkpoint')
        if selection['proof']['parent']['sha256'] != entry.get('parent_checkpoint_sha256'):
            raise ValueError('execution admission changed the exact transition parent')
        replacement = {**entry, 'qualification': {'admission':dict(admission_ref),
            'candidate_checkpoint_sha256':selection['proof']['checkpoint']['sha256']}}
        updated_history = list(history);updated_history[index] = replacement
        return {**training, 'options':{**options, 'native_shared_execution_history':updated_history}}
    if not isinstance(recorded, dict) or not isinstance(recorded.get('admission'), dict):
        raise ValueError('execution transition has an unsupported qualification record')
    if admission_ref is not None and admission_ref != recorded['admission']:
        raise ValueError('explicit execution admission differs from the earned transition')
    selection = verify_selected_publication(recorded['admission'])
    if (recorded.get('candidate_checkpoint_sha256') != selection['proof']['checkpoint']['sha256']
            or entry.get('parent_checkpoint_sha256') != selection['proof']['parent']['sha256']
            or type(training.get('global_step', training.get('step'))) is not int
            or training.get('global_step', training.get('step')) < entry['at_global_step']):
        raise ValueError('earned checkpoint has invalid execution qualification ancestry')
    if training.get('global_step', training.get('step')) == entry['at_global_step']:
        if checkpoint_path is None:
            raise ValueError('zero-update execution continuation requires its actual checkpoint')
        _verify_zero_update_continuation(checkpoint_path, selection['proof']['checkpoint'], training)
    return training



def _optimizer_history(value):
    """LR is a live recipe control; retain every state tensor and group member."""
    if isinstance(value, dict):
        return {key: ([{k: v for k, v in group.items() if k != 'lr'} for group in item]
                      if key == 'param_groups' else _optimizer_history(item))
                for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_optimizer_history(item) for item in value)
    return value


@with_historical_file_verification_session
def _verify_zero_update_continuation(checkpoint_path, candidate_ref, training):
    """Authenticate a metadata-only save before its first earned optimizer step.

    A saved qualification is not sufficient: independently check the actual full
    tensors, optimizer history, RNG and positive exposure against the selected candidate.
    Source appends may add zero-credit rows; no earned visit may change.
    """
    from pathlib import Path
    import torch
    from .native_zero_sgd_adoption import _same
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    path = Path(checkpoint_path).resolve(strict=True)
    candidate = _verify_historical_file_ref(candidate_ref)
    def stamp(p):
        st = p.stat()
        return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    observed = stamp(path), stamp(candidate)
    source = torch.load(candidate, map_location='cpu', weights_only=False, mmap=True)
    child = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    _same(training, child['training'])
    for key in ('schema', 'model_config', 'backbone_state', 'interface_state'):
        _same(source[key], child[key])
    _same(_optimizer_history(source['training']['optimizer']),
          _optimizer_history(child['training']['optimizer']))
    _same(source['training']['rng'], child['training']['rng'])
    step = lambda t: t.get('global_step', t.get('step'))
    if step(source['training']) != step(child['training']):
        raise ValueError('zero-update execution continuation changed its earned step')
    def exposure(state):
        counts = state.get('exposure_counts')
        if not isinstance(counts, dict) or any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError('zero-update execution continuation has invalid exposure')
        return {key: value for key, value in counts.items() if value}
    if exposure(source['training']) != exposure(child['training']):
        raise ValueError('zero-update execution continuation changed earned exposure')
    if observed != (stamp(path), stamp(candidate)):
        raise ValueError('zero-update execution checkpoint changed during verification')


def validate_numerical_certificate(certificate, *, identity, model_config):
    """Validate a real CUDA runtime and complete gradient certificate for this policy."""
    import math
    from ..common.hashing import canonical_json_sha256_hex
    from ..eval.native_gradients import validate_complete_proof
    if (not isinstance(certificate, dict) or certificate.get('schema') != 'natlang.native-runtime-certificate/1'
            or certificate.get('passed') is not True or certificate.get('native_identity') != identity
            or certificate.get('checkpoint_sha256') != identity.get('checkpoint_sha256')
            or not str(certificate.get('execution_device', '')).startswith('cuda')
            or certificate.get('channel_restore_exact') is not True
            or certificate.get('gradient_scope') != 'complete-production-trainables'):
        raise ValueError('execution candidate lacks its actual CUDA complete runtime certificate')
    if model_config.get('native_shared_execution') != TARGET:
        raise ValueError('runtime certificate has a different candidate execution policy')
    profile = certificate.get('qualification_profile')
    if (not isinstance(profile, dict) or profile.get('model_config_sha256') != canonical_json_sha256_hex(model_config)):
        raise ValueError('runtime qualification has a different model execution profile')
    for name, limit in (('forward_max_abs', 1e-4), ('incremental_max_abs', 1e-4),
                        ('consumer_ce_abs', 1e-4), ('consumer_gradient_max_abs', 1e-5)):
        value = certificate.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError('runtime certificate exceeds its unchanged numerical gate: ' + name)
    validate_complete_proof(certificate.get('complete_gradient_qualification'), identity,
                            gradient_atol=1e-5, execution_profile=profile)
    return True


def validate_complete_replay_probe(receipt, *, identity, graph_id, compiled_replay):
    """Join a full-episode CUDA replay report to independently compiled graph actions.

    The caller must authenticate the sealed collection, recompute graph_id and
    compile the actual rollout (including its native-state bundle) before
    calling this check; it never treats a self-reported graph ID as custody.
    """
    import math
    if (not isinstance(receipt, dict) or receipt.get('schema') != 'natlang.native-policy-behavior-logprob-probe/1'
            or receipt.get('policy') != identity or receipt.get('graph_id') != graph_id
            or not str(receipt.get('execution_device', '')).startswith('cuda')
            or receipt.get('all_episode_actions') is not True or receipt.get('passed') is not True
            or type(receipt.get('atol')) not in (int,float) or not 0 <= receipt['atol'] <= 1e-4):
        raise ValueError('candidate lacks its exact complete-episode replay probe')
    actions = compiled_replay.actions
    decisions = getattr(compiled_replay, 'decision_actions', ())
    positions = receipt.get('positions')
    decision_positions = receipt.get('decision_positions', [])
    if (getattr(compiled_replay, 'identity', None) != identity or not (actions or decisions)
            or not isinstance(positions, list) or receipt.get('action_count') != len(actions)
            or len(positions) != len(actions) or not isinstance(decision_positions,list)
            or len(decision_positions)!=len(decisions)
            or (decisions and receipt.get('decision_count')!=len(decisions))
            or (not decisions and receipt.get('decision_count',0)!=0)):
        raise ValueError('candidate replay does not cover every sampled graph action')
    errors = []
    for index, (action, row) in enumerate(zip(actions, positions, strict=True)):
        if (not isinstance(row, dict) or row.get('action_index') != index
                or row.get('token_id') != action.get('token_id')
                or row.get('behavior_logp') != action.get('behavior_logp')):
            raise ValueError('candidate replay action differs from the authenticated sampled graph')
        values = [row.get(key) for key in ('behavior_logp','learner_logp','delta')]
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
            raise ValueError('candidate replay contains a nonfinite log probability')
        delta = values[1] - values[0]
        if delta != values[2]:
            raise ValueError('candidate replay delta is not its actual log probability difference')
        errors.append(abs(delta))
    for index,(action,row) in enumerate(zip(decisions,decision_positions,strict=True)):
        tokens=[[item['token_id'] for item in option] for option in action['options']]
        behavior=action['behavior_logps']
        if (not isinstance(row,dict) or row.get('decision_index')!=index
                or row.get('turn')!=action.get('turn') or row.get('chosen')!=action.get('chosen')
                or row.get('temperature')!=action.get('temperature')
                or row.get('option_nodes')!=action.get('option_nodes')
                or row.get('option_token_ids')!=tokens or row.get('behavior_logps')!=behavior
                or row.get('option_queries')!=[[item['query'] for item in option] for option in action['options']]
                or not isinstance(row.get('learner_logps'),list) or not isinstance(row.get('deltas'),list)
                or len(row['learner_logps'])!=len(behavior) or len(row['deltas'])!=len(behavior)
                or len(tokens)!=len(behavior)):
            raise ValueError('candidate decision replay differs from authenticated option branches')
        for old,current,delta in zip(behavior,row['learner_logps'],row['deltas'],strict=True):
            if (any(type(x) not in (int,float) or not math.isfinite(x) for x in (old,current,delta))
                    or current-old!=delta):
                raise ValueError('candidate decision replay has invalid log probability differences')
            errors.append(abs(delta))
    maximum = max(errors)
    if maximum != receipt.get('max_abs_delta') or maximum > receipt['atol']:
        raise ValueError('candidate replay exceeds the unchanged 1e-4 parity gate')
    return True


DEV_PACKET_ID = 'natlang-dev-task-packet-v2-20261028'
DEV_PACKET_SHA256 = '6fa6ca6c2e3a46c4b050da535e1a246c8f34d70f355ea047bcfe19a0653aa560'


def validate_dev_behavior(candidate, baseline, *, candidate_identity, baseline_identity,
                          packet_ref, protection_root):
    """Authenticate paired observations on the registered protected development packet.

    This is operational selection evidence, not untouched confirmation or a
    capability claim. Callers must make and record the selection judgment.
    """
    import json
    from pathlib import Path
    from ..common.custody import ProtectedWorlds
    from ..common.hashing import sha256_file_hex
    from ..eval.portable_task_evidence import validate as validate_portable_task
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    root = Path(protection_root).resolve(strict=True)
    packet = _verify_historical_file_ref(packet_ref)
    if packet_ref.get('sha256') != DEV_PACKET_SHA256:
        raise ValueError('development behavior uses a different declared packet')
    corpus = json.loads((root / 'training/neuralese_corpora.json').read_text())
    entries = corpus.get('corpora', corpus) if isinstance(corpus, dict) else corpus
    matches = [row for row in entries if isinstance(row, dict) and row.get('id') == DEV_PACKET_ID]
    if (len(matches) != 1 or matches[0].get('admission') != 'evaluation-only-never-training'
            or matches[0].get('training_admission') is not False
            or 'role development' not in matches[0].get('split', '')
            or (root / matches[0]['path'] / 'cases.jsonl').resolve(strict=True) != packet):
        raise ValueError('development packet is not the registered evaluation-only source')
    protection = ProtectedWorlds.load(root=root)
    sources = [json.loads(value) for value in protection.sources]
    records = [item for source in sources if source.get('id') == 'registered-evaluation-only-packets'
               for item in source.get('packets', []) if item.get('id') == DEV_PACKET_ID]
    if len(records) != 1 or records[0].get('cases_sha256') != DEV_PACKET_SHA256:
        raise ValueError('development packet is absent from the current protected-worlds union')
    new = validate_portable_task(candidate,identity=candidate_identity,packet_path=packet)['score']
    old = validate_portable_task(baseline,identity=baseline_identity,packet_path=packet)['score']
    if (new.get('packet_sha256') != DEV_PACKET_SHA256 or old.get('packet_sha256') != DEV_PACKET_SHA256
            or new.get('case_indices') != old.get('case_indices')
            or [row.get('id') for row in new.get('cases_detail', [])]
                != [row.get('id') for row in old.get('cases_detail', [])]):
        raise ValueError('development behavior did not evaluate the same protected cases')
    if type(new.get('task_successes')) is not int or type(old.get('task_successes')) is not int:
        raise ValueError('development behavior lacks actual outcome counts')
    _verify_historical_file_ref(packet_ref)
    changes = [{'id': before['id'], 'parent': before.get('outcome'), 'candidate': after.get('outcome')}
               for before, after in zip(old['cases_detail'], new['cases_detail'], strict=True)
               if before.get('outcome') != after.get('outcome')]
    return {'role':'protected-development', 'packet_sha256':DEV_PACKET_SHA256,
            'protected_worlds_sha256':protection.digest,
            'parent_successes':old['task_successes'], 'candidate_successes':new['task_successes'],
            'case_outcome_changes':changes, 'untouched_confirmation':False}


def validate_fresh_sealed_replay(collection_ref, request_ref, replay_ref, *, identity, roles):
    """Recheck a sealed collection and join its actual graph to the full replay probe."""
    import json
    from pathlib import Path
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.jsonio import read_jsonl
    from ..rl.episode_graph import whole_episode_turns
    from ..rl.graph_replay import NativeBundleStore
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    from ..rl.native_replay import compile_replay
    from ..rl.production import validate_actor_receipts
    from ..rl.sealed_collection import resume_sealed_collection
    request = json.loads(_verify_historical_file_ref(request_ref).read_text())
    probe = json.loads(_verify_historical_file_ref(replay_ref).read_text())
    expected = request.get('expected_native') or {}
    if (request.get('policy_id') != identity.get('checkpoint_sha256')
            or expected.get('native_identity') != identity
            or request.get('root_mode', 'student') != 'student'):
        raise ValueError('fresh collection does not bind the candidate native actor')
    report = resume_sealed_collection(collection_ref, request, validate_actors=validate_actor_receipts)
    proof = json.loads(_verify_historical_file_ref(collection_ref).read_text())
    rollout_path = _verify_historical_file_ref(proof['files']['rollouts.jsonl'])
    matches = []
    for rollout in read_jsonl(rollout_path):
        if canonical_json_sha256_hex(rollout) == probe.get('graph_id'):
            matches.append(rollout)
            if len(matches) > 1:
                raise ValueError('fresh collection duplicates the replayed graph')
    if len(matches) != 1:
        raise ValueError('complete replay graph is absent from the authenticated fresh collection')
    bundle = report.get('block_bundle')
    store = NativeBundleStore(Path(bundle), policy_id=identity['checkpoint_sha256']) if bundle else None
    def fetch(block_id):
        if store is None:
            raise ValueError('fresh replay omitted its native-state artifact bundle')
        return store.state(block_id)
    compiled = compile_replay(whole_episode_turns(matches[0]), identity=identity, roles=roles, fetch_block=fetch)
    validate_complete_replay_probe(probe, identity=identity, graph_id=probe['graph_id'], compiled_replay=compiled)
    for reference in (collection_ref, request_ref, replay_ref, proof['files']['rollouts.jsonl']):
        _verify_historical_file_ref(reference)
    return True


@with_historical_file_verification_session
def qualify_candidate(publication_ref, evidence_ref, *, protection_root):
    """Recompute candidate evidence; never change a controller or its hold."""
    import json
    from types import SimpleNamespace
    import torch
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    candidate = verify_candidate_publication(publication_ref)
    proof = candidate['proof']
    evidence = json.loads(_verify_historical_file_ref(evidence_ref).read_text())
    required = {'runtime_certificate','collection_proof','collection_request','replay_probe',
                'dev_candidate','dev_parent','dev_packet'}
    if (evidence.get('schema') != 'natlang.native-execution-qualification-evidence/1'
            or set(evidence) != required | {'schema'}):
        raise ValueError('execution qualification must bind every required evidence artifact exactly')
    def read(name):
        return json.loads(_verify_historical_file_ref(evidence[name]).read_text())
    child_path = _verify_historical_file_ref(proof['checkpoint'])
    child = torch.load(child_path, map_location='cpu', weights_only=False, mmap=True)
    config = child['model_config']
    identity = proof['actor_identity']['new']
    previous = proof['actor_identity']['old']
    markers = config.get('marker_ids')
    if not isinstance(markers, (list, tuple)) or len(markers) != 4 or any(type(x) is not int for x in markers):
        raise ValueError('candidate has an invalid native role-token inventory')
    roles = SimpleNamespace(produce_open=markers[0], produce_close=markers[1],
                            consume_open=markers[2], consume_close=markers[3])
    validate_numerical_certificate(read('runtime_certificate'), identity=identity, model_config=config)
    validate_fresh_sealed_replay(evidence['collection_proof'], evidence['collection_request'],
                                 evidence['replay_probe'], identity=identity, roles=roles)
    dev = validate_dev_behavior(evidence['dev_candidate'], evidence['dev_parent'], candidate_identity=identity,
                                baseline_identity=previous, packet_ref=evidence['dev_packet'],
                                protection_root=protection_root)
    # Recheck both the evidence envelope and every bound record after the long
    # independent source and tensor reads. Nothing is promoted here.
    _verify_historical_file_ref(evidence_ref)
    _verify_historical_file_ref(publication_ref)
    for reference in evidence.values():
        if isinstance(reference, dict):
            _verify_historical_file_ref(reference)
    verify_candidate_publication(publication_ref)
    return {'schema':'natlang.native-execution-qualification/1', 'candidate':dict(publication_ref),
            'evidence':dict(evidence_ref), 'checkpoint':proof['checkpoint'],
            'actor_identity':identity, 'parent_identity':previous,
            'protected_development':dev, 'protection_root':str(protection_root),
            'runtime_and_replay_verified':True, 'selection_hold':True,
            'untouched_confirmation':False}


def _with_recorded_protection_union(checked, report):
    """The protected-worlds union digest in a qualification is provenance of when it was made: the registry only
    grows as held packets are registered (2026-11-03: two registrations made every earlier qualification unverifiable).
    The recheck itself requires the development packet to be in the current union (validate_dev_behavior); every
    other field still compares exactly."""
    recorded = (report.get('protected_development') or {}).get('protected_worlds_sha256')
    current = checked.get('protected_development')
    if not isinstance(current, dict) or recorded is None:
        return checked
    return {**checked, 'protected_development': {**current, 'protected_worlds_sha256': recorded}}


@with_historical_file_verification_session
def verify_qualification(qualification_ref):
    import json
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    report = json.loads(_verify_historical_file_ref(qualification_ref).read_text())
    if report.get('schema') != 'natlang.native-execution-qualification/1':
        raise ValueError('unsupported execution qualification receipt')
    checked = qualify_candidate(report['candidate'], report['evidence'], protection_root=report['protection_root'])
    if report != _with_recorded_protection_union(checked, report):
        raise ValueError('execution qualification differs from independently rechecked evidence')
    _verify_historical_file_ref(qualification_ref)
    return report  # the rechecked qualification as recorded (its union digest is provenance)


@with_historical_file_verification_session
def verify_selected_publication(reference):
    """Recheck a selected transition and its complete diagnostic qualification."""
    import json
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref
    from pathlib import Path
    publication = json.loads(_verify_historical_file_ref(reference).read_text())
    if publication.get('schema') == 'natlang.native-execution-source-adopted-selection/1':
        if set(publication)!={'schema','selected_execution','source_adoption'}:
            raise ValueError('unsupported source-adopted execution selection')
        selected=verify_selected_publication(publication['selected_execution'])
        if selected['publication'].get('schema')!='natlang.native-execution-selection-publication/1':
            raise ValueError('source adoption requires an original execution selection')
        adoption_ref=publication['source_adoption']
        adoption=json.loads(_verify_historical_file_ref(adoption_ref).read_text())
        from .native_zero_sgd_adoption import verify as verify_adoption
        prior=_verify_historical_file_ref(adoption['prior_recipe'])
        successor=_verify_historical_file_ref(adoption['successor_recipe'])
        checked=verify_adoption(adoption['parent'],adoption['checkpoint'],prior_recipe=prior,
                                successor_recipe=successor,stage_id=adoption['stage_id'])
        if adoption!={**checked,'prior_recipe':adoption['prior_recipe'],'successor_recipe':adoption['successor_recipe']}:
            raise ValueError('source adoption differs from independently rechecked full state')
        candidate=selected['proof']
        if (adoption['parent']['sha256']!=candidate['checkpoint']['sha256']
                or _verify_historical_file_ref(adoption['parent'])!=_verify_historical_file_ref(candidate['checkpoint'])
                or adoption['global_step']!=candidate['global_step']):
            raise ValueError('source adoption does not descend from the exact selected execution checkpoint')
        _verify_historical_file_ref(adoption_ref)
        _verify_historical_file_ref(reference)
        # The original qualification remains attached to the original candidate;
        # the adopted child is authorized solely by the independently replayed
        # metadata-only source transition.
        proof={**candidate,'checkpoint':adoption['checkpoint']}
        return {'publication':publication,'qualification':selected['qualification'],'proof':proof,
                'original_execution_selection':selected,'source_adoption':adoption,
                'authenticated_ancestors':[*selected['authenticated_ancestors'],adoption['checkpoint']['sha256']]}
    if (publication.get('schema') != 'natlang.native-execution-selection-publication/1'
            or publication.get('selection_hold') is not False
            or publication.get('optimizer_updates') != 0 or publication.get('graph_visits') != 0
            or not isinstance(publication.get('decision_reason'), str) or not publication['decision_reason'].strip()):
        raise ValueError('unsupported or incomplete execution selection publication')
    qualification = verify_qualification(publication['qualification'])
    candidate_publication = verify_candidate_publication(qualification['candidate'])
    candidate = candidate_publication['proof']
    if (publication.get('candidate') != qualification['candidate']
            or publication.get('parent') != candidate['parent']
            or publication.get('checkpoint') != candidate['checkpoint']
            or publication.get('actor_identity') != candidate['actor_identity']
            or publication.get('protected_development') != qualification['protected_development']
            or publication.get('untouched_confirmation') is not False):
        raise ValueError('selected execution differs from its exact qualified candidate')
    controller_path = Path(publication.get('controller_state_path', '')).resolve(strict=True)
    before = controller_path.stat()
    controller = json.loads(controller_path.read_text())
    after = controller_path.stat()
    sig = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    same_ref = lambda item: (isinstance(item, dict) and item.get('sha256') == reference.get('sha256')
        and Path(item.get('path', '')).resolve() == Path(reference.get('path', '')).resolve())
    if (sig(before) != sig(after)
            or not any(same_ref(item) for item in controller.get('execution_transition_history', []))
            or not any(same_ref(item) for item in controller.get('direct_sft_refresh_history', []))):
        raise ValueError('execution selection is not recorded by its durable controller')
    _verify_historical_file_ref(reference)
    return {'publication':publication, 'qualification':qualification, 'proof':candidate,
            'authenticated_ancestors':[candidate['parent']['sha256'],candidate['checkpoint']['sha256']]}


@with_historical_file_verification_session
def publish_selected(state_path, *, qualification_ref, out, decision_reason):
    """Select an independently qualified candidate under the stopped controller lock."""
    import fcntl
    import json
    from pathlib import Path
    from ..common.jsonio import write_json_durable
    from ..common.custody import REGISTRY, _protected_worlds_snapshot, _read_protected_registry
    from ..rl.native_direct_sft_recovery import _verify_historical_file_ref, _strong_binding
    from ..rl.production import invalidate_checkpoint_readiness
    if not isinstance(decision_reason, str) or not decision_reason.strip():
        raise ValueError('execution selection requires an explicit operational decision reason')
    state_path = Path(state_path).resolve(strict=True); out = Path(out).resolve()
    if out.exists():
        raise ValueError('execution selection receipt already exists')
    # Expensive tensor, graph and protection checks happen before the lock.
    qualification = verify_qualification(qualification_ref)
    candidate_publication = verify_candidate_publication(qualification['candidate'])
    candidate = candidate_publication['proof']
    root = Path(qualification['protection_root']).resolve(strict=True)
    registry_path = root / REGISTRY
    protection_declaration = _read_protected_registry(registry_path)
    protection_snapshot = _protected_worlds_snapshot(registry_path, protection_declaration, root)
    original_state = json.loads(state_path.read_text())
    round_number = original_state.get('round')
    if type(round_number) is not int or round_number < 0:
        raise ValueError('execution selection requires the actual controller round')
    attempt = original_state.get('collection_attempt', 0)
    if type(attempt) is not int or attempt < 0:
        raise ValueError('execution selection has an invalid collection attempt')
    old_selection = state_path.parent / f'round-{round_number:06d}-selection.json'
    old_round = state_path.parent / (f'round-{round_number:06d}' + (f'-attempt-{attempt}' if attempt else ''))
    old_refs = {}
    for name, path in (('selection',old_selection),('collection_binding',old_round/'collection-binding.json')):
        if path.exists():old_refs[name] = _strong_binding(path)
    old_presence = (old_selection.exists(), old_round.exists())
    next_round = round_number + int(any(old_presence))
    def round_bound(number):
        return ((state_path.parent / f'round-{number:06d}').exists()
                or (state_path.parent / f'round-{number:06d}-selection.json').exists()
                or any(state_path.parent.glob(f'round-{number:06d}-attempt-*')))
    if next_round != round_number and round_bound(next_round):
        raise ValueError('next candidate round already has a bound packet or collection')
    def stamp(path):
        st=Path(path).stat();return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns
    evidence = json.loads(_verify_historical_file_ref(qualification['evidence']).read_text())
    collection = json.loads(_verify_historical_file_ref(evidence['collection_proof']).read_text())
    request = json.loads(_verify_historical_file_ref(evidence['collection_request']).read_text())
    collection_binding = json.loads(_verify_historical_file_ref(collection['files']['collection-binding.json']).read_text())
    cases_ref = {'path':request['cases'], 'sha256':collection_binding['cases_sha256']}
    refs = [qualification_ref,qualification['candidate'],candidate_publication['publication']['proof'],
            qualification['evidence'],candidate['parent'],candidate['checkpoint'],cases_ref,
            *[v for v in evidence.values() if isinstance(v,dict)],*old_refs.values(),
            *list((collection.get('files') or {}).values())]
    if 'blocks/manifest.json' in collection['files']:
        from ..rl.graph_replay import verified_bundle_path
        manifest_path = _verify_historical_file_ref(collection['files']['blocks/manifest.json'])
        manifest = json.loads(manifest_path.read_text())
        for ident, entry in manifest.get('blocks', {}).items():
            path = verified_bundle_path(manifest_path, ident, policy_id=candidate['checkpoint']['sha256'],
                                        manifest=manifest)
            refs.append({'path':str(path), 'sha256':entry['sha256'], 'bytes':entry['bytes']})
    for name in ('dev_candidate','dev_parent'):
        task_path = _verify_historical_file_ref(evidence[name])
        task = json.loads(task_path.read_text())
        from ..eval.portable_task_evidence import FILES, SCHEMA
        if task.get('schema') != SCHEMA or set(task.get('files',{})) != set(FILES):
            raise ValueError('invalid portable development evidence')
        for filename in FILES:
            item=task['files'][filename]
            refs.append({'path':str(task_path.parent/filename),'sha256':item['sha256'],'bytes':item['bytes']})
    from .native_readiness import _bound
    def checked_path(ref):
        path = _bound(ref)
        if 'bytes' in ref and (type(ref['bytes']) is not int or ref['bytes'] < 0 or path.stat().st_size != ref['bytes']):
            raise ValueError('execution qualification dependency changed its declared size')
        return path
    checked_paths = [(ref,checked_path(ref)) for ref in refs]
    stamps = [(ref,path,stamp(path)) for ref,path in checked_paths]
    with (state_path.parent/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads(state_path.read_text())
        if state != original_state or out.exists():
            raise ValueError('controller changed while qualifying the execution candidate')
        if state.get('phase') not in {'stopped_collecting','requested_stop'}:
            raise ValueError('execution selection requires the stopped collector boundary')
        registered = state.get('execution_transition_candidate')
        if (not isinstance(registered, dict)
                or registered.get('sha256') != qualification['candidate'].get('sha256')
                or checked_path(registered) != checked_path(qualification['candidate'])):
            raise ValueError('execution selection conflicts with the registered candidate')
        if (state.get('active_update_round') is not None or state.get('pending_native_checkpoint')
                or state.get('pending_native_round_dir') or state.get('active_update_partial')
                or state.get('partial_update_round') is not None):
            raise ValueError('execution selection conflicts with a pending update or candidate')
        if Path(state['checkpoint']).resolve(strict=True) != Path(candidate['parent']['path']).resolve(strict=True):
            raise ValueError('execution candidate no longer descends from the owner head')
        partial = state.get('partial_checkpoint')
        if partial and Path(partial).exists() and Path(partial).resolve() != Path(state['checkpoint']).resolve():
            raise ValueError('a different earned partial checkpoint still belongs to the controller')
        if (any(stamp(path) != before for _,path,before in stamps)
                or _protected_worlds_snapshot(registry_path, _read_protected_registry(registry_path), root) != protection_snapshot):
            raise ValueError('execution qualification dependencies changed before controller selection')
        if (old_selection.exists(), old_round.exists()) != old_presence:
            raise ValueError('current collection became bound during qualification')
        if next_round != round_number and round_bound(next_round):
            raise ValueError('next execution round became bound during qualification')
        result = {'schema':'natlang.native-execution-selection-publication/1',
            'controller_state_path':str(state_path),
            'candidate':qualification['candidate'], 'qualification':dict(qualification_ref),
            'parent':candidate['parent'], 'checkpoint':candidate['checkpoint'], 'actor_identity':candidate['actor_identity'],
            'protected_development':qualification['protected_development'], 'untouched_confirmation':False,
            'decision_reason':decision_reason, 'selection_hold':False, 'optimizer_updates':0, 'graph_visits':0,
            'previous_round':round_number, 'next_round':next_round, 'preserved_old_collection':old_refs}
        write_json_durable(out,result)
        publication_ref = _strong_binding(out)
        next_state=dict(state)
        next_state.update(checkpoint=str(Path(candidate['checkpoint']['path']).resolve(strict=True)),
                          actor_policy_pending_load=candidate['checkpoint']['sha256'],
                          phase='collect',round=next_round,collection_attempt=0,production_complete=False,
                          execution_transition_admission=publication_ref)
        next_state.pop('execution_transition_candidate',None)
        next_state.setdefault('execution_transition_history',[]).append(publication_ref)
        next_state.setdefault('direct_sft_refresh_history',[]).append(publication_ref)
        invalidate_checkpoint_readiness(next_state,candidate['checkpoint'],reason='native-execution-policy-transition')
        write_json_durable(state_path,next_state)
        return result


def seal_fresh_collection(*, out, request_ref, proof_out):
    """Bind and independently validate a completed diagnostic collection."""
    import json
    import os
    from pathlib import Path
    from ..common.jsonio import read_jsonl, write_json_durable
    from ..rl.native_direct_sft_recovery import _strong_binding, _verify_historical_file_ref
    from ..rl.production import validate_actor_receipts
    from ..rl.sealed_collection import resume_sealed_collection
    out = Path(out).resolve(strict=True); proof_out = Path(proof_out).resolve()
    if proof_out.exists():raise ValueError('sealed collection proof already exists')
    request = json.loads(_verify_historical_file_ref(request_ref).read_text())
    if Path(request.get('out','')).resolve(strict=True) != out:
        raise ValueError('fresh collection request names a different output directory')
    names = {'collection-binding.json','score.json','rollouts.jsonl','packet.jsonl',
             'queue.jsonl','journal.jsonl','preparation.json','sampling-config.json'}
    report = json.loads((out/'score.json').read_text())
    if report.get('block_bundle'):
        bundle = Path(report['block_bundle']).resolve(strict=True)
        if bundle != (out/'blocks/manifest.json').resolve(strict=True):
            raise ValueError('fresh collection block bundle is outside its declared archive')
        names.add('blocks/manifest.json')
    for row in read_jsonl(out/'queue.jsonl'):
        path = Path(row['output'])
        if path.exists():
            try:names.add(str(path.resolve(strict=True).relative_to(out)))
            except ValueError:raise ValueError('fresh collection result is outside its archive') from None
    files = {name:_strong_binding(out/name) for name in sorted(names)}
    proof = {'schema':'natlang.sealed-collection-continuation/1','out':str(out),'files':files}
    temporary = proof_out.with_name(proof_out.name+'.pending')
    if temporary.exists():raise ValueError('sealed collection proof pending path already exists')
    write_json_durable(temporary,proof)
    # Validation is the same authenticated full collection path used for
    # continuing an earned round, not a declaration that collection passed.
    resume_sealed_collection(_strong_binding(temporary),request,validate_actors=validate_actor_receipts)
    _verify_historical_file_ref(request_ref)
    if proof_out.exists():raise ValueError('sealed collection proof was published concurrently')
    os.link(temporary,proof_out)  # no overwrite if another publisher wins
    os.unlink(temporary)
    fd=os.open(proof_out.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return proof


def publish_source_adopted_selection(selected_ref, adoption_ref, *, out):
    """Publish a typed composition after independently verifying both chains."""
    from pathlib import Path
    from ..common.jsonio import write_json_durable
    from ..rl.native_direct_sft_recovery import _strong_binding
    output=Path(out).resolve()
    if output.exists():
        raise ValueError('source-adopted execution selection already exists')
    publication={'schema':'natlang.native-execution-source-adopted-selection/1',
                 'selected_execution':dict(selected_ref),'source_adoption':dict(adoption_ref)}
    # Verify through the same reader using a temporary exclusive envelope, then
    # publish only the exact verified bytes to a new immutable path.
    import tempfile, json, os
    output.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.execution-composition-',dir=output.parent)
    temporary=Path(name)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump(publication,stream,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        verify_selected_publication(_strong_binding(temporary))
        # Hard link is exclusive and cannot replace a concurrent publication.
        os.link(temporary,output)
        directory=os.open(output.parent,os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return _strong_binding(output)
