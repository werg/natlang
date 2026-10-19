"""Exact saved-update observations; interrupted work grants no qualification."""
from pathlib import Path
import json
from ..common.hashing import canonical_json_sha256_hex as digest
from ..common.jsonio import write_json_durable
from ..train.native_readiness import _bound
from ..train.native_resident import binding

SCHEMA = 'natlang.native-owned-interrupted-update-recovery/1'
OBSERVATION = 'natlang.native-interrupted-checkpoint-observation/1'
STOP = 'natlang.native-supervised-interruption/1'


def request_operational_stop(intent, trainer, original_request, *, reason, out):
    """Send one exact pidfd signal and persist its performed action custody."""
    import os
    import signal
    import time
    from .native_raw_handoff import _captured_owner_controls, observe, same_process
    _, _, identities = _captured_owner_controls(intent)
    if (not isinstance(reason, str) or not reason.strip()
            or trainer not in identities or trainer == identities[0]):
        raise ValueError('operational stop needs its actual captured trainer and reason')
    before = observe(trainer['pid'])
    if not same_process(before or {}, trainer) or before.get('argv_hex') != trainer['argv_hex']:
        raise ValueError('operational stop trainer identity changed')
    fd = os.pidfd_open(trainer['pid'])
    try:
        after = observe(trainer['pid'])
        if not same_process(after or {}, trainer) or after.get('argv_hex') != trainer['argv_hex']:
            raise ValueError('operational stop trainer changed while opening pidfd')
        value = {'schema': STOP, 'at': time.time(), 'signal': 'SIGTERM', 'reason': reason,
                 'trainer': trainer, 'captured_owner_intent': intent, 'original_request': original_request,
                 'actor_source': binding(__file__)}
        plan = Path(out).with_name(Path(out).stem + '-intent.json')
        if Path(out).exists() or plan.exists():
            raise ValueError('reconcile existing operational stop; never repeat its signal blindly')
        write_json_durable(plan, value)
        signal.pidfd_send_signal(fd, signal.SIGTERM)
        value.update(performed=True, syscall='pidfd_send_signal', action_intent=binding(plan), sent_at=time.time())
        write_json_durable(out, value)
    finally:
        os.close(fd)
    return binding(out)


def verify_checkpoint_pin(recovery):
    """Verify real bytes and independent storage before the first acknowledgement."""
    pin = _bound(recovery['checkpoint']); slot = _bound(recovery['original_slot'])
    a, b = pin.stat(), slot.stat()
    if (recovery['checkpoint']['sha256'] != recovery['original_slot']['sha256']
            or any('bytes' in ref and ref['bytes'] != a.st_size
                   for ref in (recovery['checkpoint'], recovery['original_slot']))
            or pin.is_symlink() or slot.is_symlink() or a.st_size != b.st_size
            or a.st_size != recovery['checkpoint_bytes']
            or [a.st_dev, a.st_ino] != recovery['independent_pin_inode']
            or [b.st_dev, b.st_ino] != recovery['original_slot_inode']
            or (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)):
        raise ValueError('interrupted recovery needs its actual independent complete checkpoint pin')
    return {'schema': 'natlang.interrupted-pin-verification/1', 'verifier': binding(__file__),
        'checkpoint': recovery['checkpoint'], 'original_slot': recovery['original_slot'],
        'bytes': a.st_size, 'pin_inode': [a.st_dev, a.st_ino],
        'original_inode': [b.st_dev, b.st_ino], 'independent': True}


def verify_payload_evidence(value, record):
    """Historical validation retains byte-bound proof after obsolete payload pruning."""
    pin = record['checkpoint_pin']
    expected = {'schema': 'natlang.interrupted-pin-verification/1', 'verifier': binding(__file__),
        'checkpoint': record['checkpoint'], 'original_slot': pin['original_slot'],
        'bytes': pin['bytes'], 'pin_inode': pin['pin_inode'],
        'original_inode': pin['original_inode'], 'independent': True}
    if value != expected:
        raise ValueError('interrupted acknowledgement lost its verified full payload evidence')


def capture_provider_environment(owner, out):
    """Capture only the entry identity variable from the exact live process."""
    import time
    from .native_raw_handoff import observe, same_process
    before = observe(owner['pid'])
    if not same_process(owner, before) or before.get('argv_hex') != owner.get('argv_hex'):
        raise ValueError('provider environment needs its exact live observed owner')
    raw = (Path('/proc') / str(owner['pid']) / 'environ').read_bytes()
    values = [part.partition(b'=')[2].decode() for part in raw.split(b'\0')
              if part.startswith(b'NATLANG_WORKER_ENTRY_SHA256=')]
    after = observe(owner['pid'])
    if (not same_process(owner, after) or after.get('argv_hex') != owner.get('argv_hex')
            or len(values) != 1 or len(values[0]) != 64
            or any(char not in '0123456789abcdef' for char in values[0])):
        raise ValueError('provider environment lost its actual entry SHA binding')
    value = {'schema': 'natlang.native-provider-entry-environment/1', 'at': time.time(),
             'owner': owner, 'entry_sha256': values[0]}
    if Path(out).exists():
        existing = json.loads(Path(out).read_text())
        if any(existing.get(key) != value[key] for key in ('schema', 'owner', 'entry_sha256')):
            raise ValueError('provider environment observation changed on retry')
        return binding(out)
    write_json_durable(out, value)
    return binding(out)


def observe_checkpoint(checkpoint, out):
    """Decode the pinned full state once, without constructing a model."""
    from ..train.checkpoint_policy import CheckpointPolicy
    from .native_raw_policy import require_raw_training
    path = _bound(checkpoint)
    state = CheckpointPolicy.load(path)
    training = require_raw_training(state)
    rows = training.get('source_row_bindings') or {}
    counts = training.get('exposure_counts') or {}
    order = training.get('source_row_order') or []
    held = training.get('held_row_bindings') or {}
    if (not rows or set(counts) != set(rows) or len(order) != len(rows)
            or set(order) != set(rows) or not held
            or any(type(n) is not int or n < 0 for n in counts.values())
            or state.get('step') != training.get('global_step')
            or not isinstance(training.get('sft_progress'), dict)):
        raise ValueError('interrupted checkpoint lost its complete ordinary exposure or counters')
    options = training.get('options') or {}
    value = {'schema': OBSERVATION, 'checkpoint': checkpoint,
        'observer': binding(__file__), 'model_config': state['model_config'],
        'global_step': training['global_step'], 'stage_step': training['step'],
        'continuation': options.get('continuation'),
        'resident_adoption': options.get('resident_adoption'),
        'source_receipts': training['source_receipts'],
        'source_row_bindings_sha256': digest(rows), 'held_row_bindings_sha256': digest(held),
        'exposure_counts_sha256': digest(sorted(counts.items())),
        'sft_progress_sha256': digest(training['sft_progress']),
        'train_rows': len(rows), 'held_rows': len(held),
        'optimizer_and_rng_present': True, 'qualification_credit': False}
    out = Path(out)
    if out.exists():
        if json.loads(out.read_text()) != value:
            raise ValueError('interrupted checkpoint observation changed on retry')
    else:
        write_json_durable(out, value)
    return binding(out)


def bridge_invocation(spec, argv, recovery, intent):
    """Accept a provider-bound versioned entry, never an arbitrary argv rewrite."""
    if argv[1:4] == ['-u', '-m', 'natlang_neuralese.rl.native_bridge']:
        return True
    entry = recovery.get('provider_entry') or {}
    environment = json.loads(_bound(recovery['provider_environment']).read_text())
    stop = json.loads(_bound(recovery['operational_stop']).read_text())
    from .native_raw_handoff import same_process
    return (len(argv) >= 3 and argv[1] == '-u' and argv[2] == entry.get('path')
            and binding(_bound(entry)) == {k: entry[k] for k in ('path', 'sha256')}
            and spec.get('env', {}).get('NATLANG_WORKER_ENTRY_SHA256') == entry['sha256']
            and environment.get('schema') == 'natlang.native-provider-entry-environment/1'
            and same_process(environment.get('owner') or {}, intent['owner'])
            and environment['owner'].get('argv_hex') == intent['owner'].get('argv_hex')
            and environment.get('entry_sha256') == entry['sha256']
            and intent['at'] <= environment['at'] <= stop['at'])


def verify_controls(recovery, *, receipt, report, certificate, spec, intent, identities,
                    terminal, parent):
    """Retain actual stop/failure, current source authority and inherited capacity."""
    from .native_raw_handoff import same_process, process_argv
    from ..train import native_resident as resident
    stop = json.loads(_bound(recovery['operational_stop']).read_text())
    if (stop.get('schema') != STOP or stop.get('signal') not in ('SIGTERM', 'SIGINT', 'SIGUSR1')
            or not isinstance(stop.get('reason'), str) or not stop['reason'].strip()
            or stop.get('captured_owner_intent') != recovery['captured_owner_intent']
            or stop.get('original_request') != recovery['original_request']
            or type(stop.get('at')) not in (int, float)
            or not intent['at'] <= stop['at'] <= terminal['at']
            or receipt.get('completed') is not False or report.get('completed') is not False
            or certificate.get('passed') is not False or certificate.get('deferred') is not True
            or certificate.get('reason') != 'operational stop; qualification deferred until resume'):
        raise ValueError('interrupted recovery needs its actual unqualified operational stop')
    action = json.loads(_bound(stop['action_intent']).read_text())
    if (stop.get('actor_source') != binding(__file__) or stop.get('performed') is not True
            or stop.get('syscall') != 'pidfd_send_signal' or not stop['at'] <= stop['sent_at'] <= terminal['at']
            or action != {k: v for k, v in stop.items()
                          if k not in ('performed', 'syscall', 'action_intent', 'sent_at')}):
        raise ValueError('interrupted recovery lost its performed exact supervisor stop action')
    trainers = [item for item in identities if same_process(item, stop.get('trainer') or {})
                and item.get('argv_hex') == stop.get('trainer', {}).get('argv_hex')]
    if len(trainers) != 1:
        raise ValueError('interrupted recovery lost its captured actual trainer')
    trainer = trainers[0]
    argv = process_argv(trainer)
    def argument(flag):
        if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv):
            raise ValueError('interrupted recovery lost its trainer argument: ' + flag)
        return argv[argv.index(flag) + 1]
    if (argv.count('-m') != 1 or argv[argv.index('-m') + 1] not in (
            'natlang_neuralese.train.native_state_sft', 'natlang_neuralese.train.native_preference')
            or Path(argument('--out')).resolve() != Path(recovery['original_slot']['path']).parent.resolve()):
        raise ValueError('interrupted recovery changed its actual shared trainer invocation')
    evidence = recovery['resident']
    initial = json.loads(_bound(evidence['initial']).read_text())
    code = resident.code_inventory(Path(spec['runtime']) / 'training/neuralese/natlang_neuralese')
    capacity = initial['geometry_qualification']
    resident.geometry_capacity(capacity, identity=initial['geometry_identity'], code=code)
    from ..train.native_state_sft import build_parser
    args = build_parser().parse_args(argv[argv.index('-m') + 2:])
    identity = initial['geometry_identity']
    for name in ('batch_graphs', 'batch_tokens', 'checkpoint_layers', 'chunk_tokens',
                 'checkpoint_input_offload', 'cotangent_accumulator_backend',
                 'ffn_chunk_tokens', 'share_causal_prefixes'):
        if name in identity and getattr(args, name) != identity[name]:
            raise ValueError('interrupted recovery changed its executed capacity profile: ' + name)
    if 'fp8' in identity and identity['fp8']['kind'] != args.fp8_training:
        raise ValueError('interrupted recovery changed its executed FP8 capacity profile')
    if (not same_process(initial.get('owner') or {}, trainer)
            or initial['geometry_identity'].get('model_config') != report['model_config']
            or Path(argument('--resident-data-directory')).resolve() != Path(evidence['directory']).resolve()
            or Path(argument('--resident-geometry-qualification')).resolve() != _bound(capacity).resolve()):
        raise ValueError('interrupted recovery lost its existing actual resident capacity')
    observed = json.loads(_bound(recovery['saved_frame_observation']).read_text())
    if (observed.get('schema') != OBSERVATION or observed.get('observer') != binding(__file__)
            or observed.get('checkpoint') != recovery['checkpoint']
            or observed.get('optimizer_and_rng_present') is not True
            or observed.get('qualification_credit') is not False
            or observed.get('model_config') != report['model_config']
            or observed.get('global_step') != recovery['saved_global_step']
            or observed.get('stage_step') != recovery['saved_stage_step']
            or (observed.get('continuation') or {}).get('checkpoint_sha256') != parent):
        raise ValueError('interrupted recovery lost its observed exact full saved state')
    if evidence.get('current'):
        current = json.loads(_bound(evidence['current']).read_text())
        payload = current.get('payload') or {}
        sources = observed['source_receipts']
        adoption = observed.get('resident_adoption') or {}
        if (current.get('schema') != resident.SCHEMA or resident.digest(payload) != current.get('generation')
                or current['generation'] != evidence.get('generation') or payload['plan']['code'] != code
                or adoption.get('generation') != current['generation']
                or adoption.get('history') != current.get('history')
                or observed['global_step'] < current['at_global_step']
                or payload['rows'] != observed['train_rows'] or payload['held_rows'] != observed['held_rows']
                or any(payload[name]['sha256'] != sources[i]['sha256']
                       for i, name in enumerate(('train', 'held')))):
            raise ValueError('interrupted recovery changed its actual adopted sources')
    else:
        plan = json.loads(_bound(evidence['plan']).read_text())
        inputs = plan['stage_inputs'][initial['stage_id']]
        if (resident.digest(plan) != initial['plan_sha256'] or plan['code'] != code
                or initial['rows'] != observed['train_rows'] or initial['held_rows'] != observed['held_rows']
                or any(inputs[name]['sha256'] != observed['source_receipts'][i]['sha256']
                       for i, name in enumerate(('native_graphs', 'held_native_graphs')))):
            raise ValueError('interrupted recovery changed its actual initial sources')
    return evidence
