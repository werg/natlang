"""Native trainer binding for the shared DiLoCo coordinator and learners.

A DiLoCo stage runs N copies of the ordinary native trainer
(``native_state_sft``), each on its own GPU and its own stable shard of the
source rows, all continuing from one parent full-state checkpoint. Learners
keep their inner MuonSR/AdamW history; the coordinator owns the global FP32
trainable parameters and the outer Nesterov momentum.

  init    seed a coordinator from the parent checkpoint and the learner argv
  export  write one ordinary full-state continuation checkpoint: the
          coordinator's global weights, every learner's exposure merged onto
          the parent's, and learner 0's inner optimizer/RNG

    python -m natlang_neuralese.train.native_diloco init --root R --parent P \
        --trainer-argv JSON --settings JSON [--base B] [--quorum Q] ...
    python -m natlang_neuralese.train.native_diloco export --root R --parent P \
        --learner-checkpoint W0 [--learner-checkpoint W1 ...] --out DIR [--base B]
    python -m natlang_neuralese.train.native_diloco export-global (--root R | --coordinator-url URL) --parent P \
        --out DIR [--ema-out DIR] [--base B]

``init --outer-schedule '[{"from_round":0,"lr":1.0,"momentum":0.0},{"from_round":10,"lr":0.7,"momentum":0.9}]'``
replaces ``--outer-lr``/``--outer-momentum`` with a per-fragment merge-round schedule (coordinator.outer_entry).
``export-global`` writes model-only checkpoints (weights only, never a resumable parent) of the current FP32 global
masters and, with ``--ema-out``, of the global-version EMA, cast to each parameter's dtype, while the coordinator
keeps serving: each fragment file is replaced atomically, so every fragment is read at one committed version, and
the versions read are recorded.

``--settings`` is the stage's ``diloco`` block; its coordinator URL is not
part of the identity, because the coordinator's address exists only once it runs.
The CoreWeave launcher is ``scripts/coreweave/diloco.py``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex_cached
from .diloco.coordinator import global_ema, global_masters, initialize_coordinator, read_manifest
from .diloco.protocol import TensorSurface
from .diloco.worker import DilocoLearner, DilocoSettings

IDENTITY_SCHEMA = 'natlang.native-diloco-identity/1'
EXPORT_SCHEMA = 'natlang.native-diloco-export/1'
GLOBAL_EXPORT_SCHEMA = 'natlang.native-diloco-global-export/1'
ARGUMENT = '--diloco'


def trainer_argv(argv):
    """The learner's command line without its ``--diloco`` block (which names the coordinator URL)."""
    argv = list(argv)
    result = []
    index = 0
    while index < len(argv):
        if argv[index] == ARGUMENT:
            index += 2
            continue
        if not argv[index].startswith(ARGUMENT + '='):
            result.append(argv[index])
        index += 1
    return result


def identity(parent_checkpoint, argv, settings):
    """What every learner and the coordinator must share exactly."""
    return {'schema': IDENTITY_SCHEMA, 'parent_checkpoint_sha256': sha256_file_hex_cached(parent_checkpoint),
            'trainer_argv': trainer_argv(argv), 'diloco': settings.identity()}



# These arguments name authenticated data/recipe inputs, not the distributed
# optimizer surface or transport layout. A change still requires an actual
# full-checkpoint source-correction proof in the native trainer.
_SOURCE_ARGUMENTS = frozenset({'--native-graphs','--held-native-graphs','--foundation-native-graphs',
    '--continued-native-graphs','--continued-held-native-graphs','--source-correction',
    '--source-classification-migration','--ce-supervision-policy','--sft-packed-geometry',
    '--memory-qualification','--recipe-binding','--pending-native-sources','--withdrawn-native-graphs',
    '--retired-native-withdrawal'})

def _fixed_trainer_argv(argv):
    result=[];i=0
    while i<len(argv):
        value=argv[i];name=value.split('=',1)[0]
        if name in _SOURCE_ARGUMENTS:
            if '=' not in value:
                if i+1>=len(argv) or argv[i+1].startswith('--'):raise ValueError('source identity argument lacks a value')
                i+=1
        else:result.append(value)
        i+=1
    return result

def source_identity_transition(original, current, proof):
    """Keep the original coordinator identity, recording a checked data-only transition."""
    if (not isinstance(proof,dict) or proof.get('schema')!='natlang.diloco-source-identity-transition/1'
            or proof.get('correction_verified') is not True or not all(isinstance(proof.get(k),str) and re.fullmatch('[0-9a-f]{64}',proof[k])
                 for k in ('checkpoint_sha256','correction_sha256','previous_source_hash','source_hash'))
            or (proof['previous_source_hash']==proof['source_hash'] and proof.get('transition_kind')!='consumed-proof-retirement')):
        raise ValueError('coordinator identity change requires a verified full-checkpoint source correction')
    if (original.get('schema')!=current.get('schema') or original.get('parent_checkpoint_sha256')!=current.get('parent_checkpoint_sha256')
            or original.get('diloco')!=current.get('diloco')
            or _fixed_trainer_argv(original['trainer_argv'])!=_fixed_trainer_argv(current['trainer_argv'])):
        raise ValueError('source correction cannot change coordinator surface, parent, transport or fixed trainer arguments')
    return {**proof,'original_identity':original,'current_identity':current}

def attach_learner(args, named, *, parent_continuation, saved_state, argv=None, source_transition=None):
    """Join the stage's coordinator after the trainer restored its parent or own state."""
    if args.diloco is None:
        if saved_state is not None:
            raise ValueError('checkpoint is a DiLoCo learner state; continue it with its --diloco block')
        return None
    if args.continue_from is None:
        raise ValueError('DiLoCo learners continue from the coordinator parent checkpoint (--continue-from)')
    settings = DilocoSettings.from_dict(json.loads(args.diloco))
    current_identity=identity(args.continue_from,sys.argv[1:] if argv is None else argv,settings)
    transition=None;transport=None;effective=current_identity
    if source_transition is not None:
        if parent_continuation or saved_state is None:
            raise ValueError('source identity transition requires an earned DiLoCo checkpoint')
        from .diloco.worker import transport_from_env
        transport=transport_from_env(settings);manifest=transport.json('GET','/v1/manifest')
        original=manifest['identity']
        if current_identity!=original:
            prior=source_transition.get('original_identity')
            if prior is not None:
                if prior!=original:
                    raise ValueError('carried identity transition does not match this coordinator')
                previous_identity=source_transition.get('current_identity')
                proof={k:v for k,v in source_transition.items() if k not in ('original_identity','current_identity','replay_verified_checkpoint_sha256')}
                if previous_identity!=current_identity:
                    # Only retire consumed correction/predecessor switches without
                    # inventing another data generation; trainer already verified
                    # their checkpoint history and current bound source.
                    retired={'--withdrawn-native-graphs','--retired-native-withdrawal','--source-correction','--continued-native-graphs','--continued-held-native-graphs'}
                    def without_retired(argv):
                        out=[];i=0
                        while i<len(argv):
                            name=argv[i].split('=',1)[0]
                            if name in retired:
                                if '=' not in argv[i]:i+=1
                            else:out.append(argv[i])
                            i+=1
                        return out
                    if without_retired(previous_identity['trainer_argv'])!=without_retired(current_identity['trainer_argv']):
                        raise ValueError('same-source replay changed more than consumed proof arguments')
                    proof={**proof,'previous_source_hash':proof['source_hash'],
                           'checkpoint_sha256':source_transition['replay_verified_checkpoint_sha256'],
                           'transition_kind':'consumed-proof-retirement'}
                transition=source_identity_transition(original,current_identity,proof)
            else:
                transition=source_identity_transition(original,current_identity,source_transition)
            effective=original
    learner = DilocoLearner(settings, surface=TensorSurface(dict(named)),
                            identity=effective, output=args.out / 'diloco', transport=transport)
    if parent_continuation:
        learner.start()
    elif saved_state is None:
        raise ValueError('rolling checkpoint lacks its DiLoCo learner state')
    else:
        learner.resume(saved_state)
    if transition is not None:
        learner.record_identity_transition(transition)
        learner.identity_transition=transition
    return learner


def _trainable(path, base):
    from ..model.native_model import load_native_checkpoint
    from .native_state_sft import native_trainable_parameters
    model, *_ = load_native_checkpoint(path, base=base, device='cpu')
    return model, native_trainable_parameters(model)


def _checkpoint_surface(path, base):
    """Map the full checkpoint's optimizer-bound trainable surface without a CPU model copy.

    The checkpoint is mmap-backed; coordinator fragments cast only as needed.
    Learners still independently verify the complete name/shape fingerprint.
    """
    from ..model.native_model import CHECKPOINT_SCHEMA, base_inventory
    from ..common.hashing import canonical_json_sha256_hex
    from .checkpoint_policy import CheckpointPolicy
    state=CheckpointPolicy.load(path)
    config=state.get('model_config') or {};training=state.get('training') or {}
    if state.get('schema')!=CHECKPOINT_SCHEMA or config.get('architecture')!='lfm3':
        raise ValueError('streamed coordinator seed requires an LFM3 full native checkpoint')
    if not training.get('rng') or not training.get('exposure_counts') or not training.get('source_receipts'):
        raise ValueError('streamed coordinator seed requires full RNG/exposure/source custody')
    inventory=base_inventory(base or config.get('base'))
    if inventory!=config.get('base_inventory') or canonical_json_sha256_hex(inventory)!=config.get('base_inventory_sha256'):
        raise ValueError('coordinator checkpoint base inventory differs')
    groups=(training.get('optimizer') or {}).get('param_names')
    if not isinstance(groups,dict) or not groups or not all(isinstance(v,list) for v in groups.values()):
        raise ValueError('coordinator checkpoint lacks named optimizer bindings')
    names=[name for group in groups.values() for name in group]
    if not names or len(names)!=len(set(names)) or not all(isinstance(name,str) for name in names):
        raise ValueError('coordinator checkpoint parameter bindings are invalid')
    backbone=state.get('backbone_state') or {};interface=state.get('interface_state') or {};masters={}
    for name in names:
        if name.startswith('backbone.'):
            value=backbone.get(name.removeprefix('backbone.'))
        elif name.startswith('interface.adapter.'):
            value=(interface.get('adapter') or {}).get(name.removeprefix('interface.adapter.'))
        elif name.startswith('interface.'):
            value=interface.get(name.removeprefix('interface.'))
        else:raise ValueError('unknown coordinator parameter binding '+name)
        if not isinstance(value,torch.Tensor) or not value.is_floating_point():
            raise ValueError('missing floating coordinator parameter '+name)
        masters[name]=value
    return masters


def _coordinator_surface(path, base):
    """Stream LFM3 full checkpoints; retain the generic loader for other architectures."""
    from .checkpoint_policy import CheckpointPolicy
    state=CheckpointPolicy.load(path)
    if (state.get('model_config') or {}).get('architecture')=='lfm3':
        return _checkpoint_surface(path, base)
    _, named=_trainable(path, base)
    return {name:parameter.detach() for name,parameter in named}


DEFAULT_COORDINATOR_EMA = {'decay': 0.99, 'warmup': 0}


def parse_coordinator_ema(text):
    """``--ema``: on by default for every new coordinator run (owner 2026-10-28, improvements ship on by default);
    ``off`` opts out. An existing run keeps the EMA setting its manifest recorded at init."""
    if text is None or text == '':
        return dict(DEFAULT_COORDINATOR_EMA)
    if text == 'off':
        return None
    return json.loads(text)


def init(args):
    settings = DilocoSettings.from_dict(json.loads(args.settings))
    masters = _coordinator_surface(args.parent, args.base)
    manifest = initialize_coordinator(
        args.root, run_id=settings.run_id, masters=masters,
        identity=identity(args.parent, json.loads(args.trainer_argv), settings),
        workers=settings.workers, min_quorum=args.quorum or settings.workers, fragment_bytes=args.fragment_mib << 20,
        outer_lr=args.outer_lr, outer_momentum=args.outer_momentum, grace_seconds=args.grace_seconds,
        max_staleness=args.max_staleness,
        seed_source={'path': str(args.parent), 'sha256': sha256_file_hex_cached(args.parent)},
        ema=parse_coordinator_ema(args.ema),
        outer_schedule=json.loads(args.outer_schedule) if args.outer_schedule else None)
    print(json.dumps({'run_id': manifest['run_id'], 'fragments': len(manifest['fragments']),
                      'fingerprint': manifest['fingerprint']}, sort_keys=True))


def merged_exposure(parent_counts, learner_counts):
    """Parent exposure plus every learner's own visits (learners train disjoint shards)."""
    merged = dict(parent_counts)
    for counts in learner_counts:
        for key, value in counts.items():
            gained = value - parent_counts.get(key, 0)
            if gained < 0:
                raise ValueError(f'learner exposure for {key} is below its parent')
            merged[key] = merged.get(key, 0) + gained
    return merged


def uploaded(root, name):
    """ROOT/uploads/NAME after re-verifying the sha256 the file store committed for it."""
    from .diloco.coordinator import FileStore
    store = FileStore(Path(root))
    store.check(name)
    expected = store.committed(name)
    if expected is None or sha256_file_hex_cached(store.root / name) != expected:
        raise ValueError(f'{name}: no verified upload in {store.root}')
    return store.root / name


def _retire_learner_calibration(training, learners):
    """Archive local calibration and force a fresh window on the merged weights."""
    calibration = {str(item['diloco']['worker']): item.get('quantization') for item in learners}
    training.pop('quantization', None)
    declarations = {str(item['diloco']['worker']): {
        'declaration': (item.get('options') or {}).get('quantization'),
        'history': (item.get('options') or {}).get('quantization_history') or []}
        for item in learners}
    return {'action': 'recalibrate-merged-weights', 'learners': calibration,
            'learner_declarations': declarations}


def export(args):
    from .checkpoint_policy import CheckpointPolicy
    args.learner_checkpoint = [*args.learner_checkpoint, *(uploaded(args.root, name) for name in args.learner_upload)]
    if not args.learner_checkpoint:
        raise ValueError('export needs --learner-checkpoint or --learner-upload for every learner')
    from .native_continuation import collect_sft_progress
    from ..model.native_model import model_state

    manifest = read_manifest(args.root)
    parent_sha = sha256_file_hex_cached(args.parent)
    if manifest['identity']['parent_checkpoint_sha256'] != parent_sha:
        raise ValueError('coordinator run continues a different parent checkpoint')
    masters, info = global_masters(args.root)
    parent = CheckpointPolicy.load(args.parent)['training']
    learners = [CheckpointPolicy.load(path)['training'] for path in args.learner_checkpoint]
    workers = [state['diloco']['worker'] for state in learners]
    if sorted(workers) != list(range(manifest['workers'])):
        raise ValueError(f'export needs every learner of the run exactly once, got {sorted(workers)}')
    if any(state['diloco']['fingerprint'] != manifest['fingerprint'] for state in learners):
        raise ValueError('learner checkpoint belongs to another coordinator run')
    first = args.learner_checkpoint[workers.index(0)]
    model, named = _trainable(first, args.base)
    with torch.no_grad():
        for name, parameter in named:
            if masters[name].shape != parameter.shape:
                raise ValueError(f'global parameter {name} has a different shape')
            parameter.copy_(masters[name].to(parameter.dtype))
    state = CheckpointPolicy.load(first)
    training = state['training']
    updates = sum(item['global_step'] - parent['global_step'] for item in learners)
    training['exposure_counts'] = merged_exposure(parent['exposure_counts'], [item['exposure_counts'] for item in learners])
    training['global_step'] = parent['global_step'] + updates
    training['step'] = parent['step'] + updates
    training.pop('diloco')
    # Learner calibration describes different local weights and batches, not the merged masters.
    # Retain each exact state as custody and force calibration on the merged continuation.
    calibration = _retire_learner_calibration(training, learners)
    training['options'] = {**training['options'], 'diloco_export': {
        'schema': EXPORT_SCHEMA, 'run_id': manifest['run_id'], 'coordinator_fingerprint': manifest['fingerprint'],
        'fragment_versions': info['versions'], 'parent_checkpoint_sha256': parent_sha,
        'learner_updates': {str(item['diloco']['worker']): item['global_step'] - parent['global_step'] for item in learners},
        'inner_optimizer_and_rng_from_learner': 0,
        'quantization_calibration': calibration}}
    training['sft_progress'] = collect_sft_progress(training)
    state.update(model_state(model), step=training['global_step'])
    CheckpointPolicy(args.out).save(state)
    ema_out = args.ema_out
    if ema_out is None and manifest.get('ema') is not None:
        # On by default: a run that keeps a global EMA also exports it beside the continuation checkpoint.
        ema_out = Path(str(args.out) + '-ema')
    if ema_out is not None and str(ema_out) != 'off':
        args.ema_out = Path(ema_out)
        export_ema(args, manifest, model, named, first, training['global_step'])
    print(json.dumps(training['options']['diloco_export'], sort_keys=True))


def export_ema(args, manifest, model, named, first_checkpoint, global_step):
    """Model-only checkpoint of the coordinator's EMA of global versions (an evaluation / release artifact)."""
    from .checkpoint_policy import CheckpointPolicy
    from .weight_ema import EXPORT_SCHEMA as EMA_EXPORT_SCHEMA, write_weights_checkpoint
    averages, info = global_ema(args.root)
    with torch.no_grad():
        for name, parameter in named:
            if averages[name].shape != parameter.shape:
                raise ValueError(f'global EMA parameter {name} has a different shape')
            parameter.copy_(averages[name].to(parameter.dtype))
    state = CheckpointPolicy.load(first_checkpoint)
    export = {'schema': EMA_EXPORT_SCHEMA, 'source': 'diloco-coordinator-global-ema', 'run_id': manifest['run_id'],
              'coordinator_fingerprint': manifest['fingerprint'], 'fragment_versions': info['versions'],
              'ema': info['ema'], 'source_global_step': global_step,
              'parent_checkpoint_sha256': manifest['identity']['parent_checkpoint_sha256'],
              'dtype_cast': 'FP32 coordinator EMA rounded to each parameter dtype (nearest)'}
    write_weights_checkpoint(model, state['model_config'], state['schema'], global_step, export, args.ema_out)


def export_global(args):
    """Model-only checkpoints of the running coordinator's global masters (and EMA) for evaluation and restore."""
    from .checkpoint_policy import CheckpointPolicy
    from .weight_ema import write_weights_checkpoint

    if getattr(args, 'coordinator_url', None):
        from .diloco.coordinator import TOKEN_ENV, TOKEN_SECRET, credential, remote_global
        from .diloco.worker import GATEWAY_SECRET, GATEWAY_TOKEN_ENV, Transport
        transport = Transport(args.coordinator_url, credential(TOKEN_ENV, TOKEN_SECRET), getattr(args, 'timeout', 600.0),
                              gateway=credential(GATEWAY_TOKEN_ENV, GATEWAY_SECRET))
        workdir = getattr(args, 'workdir', None) or args.out.with_name(args.out.name + '.fetch')
        manifest = transport.json('GET', '/v1/manifest')

        def read(key):
            values, info, _ = remote_global(transport, key, workdir)
            return values, info
        readers = {'global': lambda: read('theta'), 'ema': lambda: read('ema')}
    else:
        manifest = read_manifest(args.root)
        readers = {'global': lambda: global_masters(args.root), 'ema': lambda: global_ema(args.root)}
    parent_sha = sha256_file_hex_cached(args.parent)
    if manifest['identity']['parent_checkpoint_sha256'] != parent_sha:
        raise ValueError('coordinator run continues a different parent checkpoint')
    model, named = _trainable(args.parent, args.base)
    state = CheckpointPolicy.load(args.parent)
    parent_step = (state.get('training') or {}).get('global_step', state.get('step', 0))
    for kind, out in (('global', args.out), ('ema', args.ema_out)):
        if out is None:
            continue
        values, info = readers[kind]()
        with torch.no_grad():
            for name, parameter in named:
                if values[name].shape != parameter.shape:
                    raise ValueError(f'global {kind} parameter {name} has a different shape')
                parameter.copy_(values[name].to(parameter.dtype))
        export = {'schema': GLOBAL_EXPORT_SCHEMA, 'source': f'diloco-coordinator-{kind}', 'run_id': manifest['run_id'],
                  'coordinator_fingerprint': manifest['fingerprint'], 'fragment_versions': info['versions'],
                  'parent_checkpoint_sha256': parent_sha, 'parent_global_step': parent_step,
                  'dtype_cast': f'FP32 coordinator {kind} rounded to each parameter dtype (nearest)',
                  **({'ema': info['ema']} if kind == 'ema' else {})}
        write_weights_checkpoint(model, state['model_config'], state['schema'], parent_step, export, out)
        print(json.dumps({'kind': kind, 'out': str(out), 'versions': info['versions']}, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    start = commands.add_parser('init')
    start.add_argument('--root', type=Path, required=True)
    start.add_argument('--parent', type=Path, required=True)
    start.add_argument('--base', type=Path)
    start.add_argument('--trainer-argv', required=True, help='JSON list: the learner trainer arguments')
    start.add_argument('--settings', required=True, help='JSON: the stage diloco block (coordinator_url may be absent)')
    start.add_argument('--quorum', type=int)
    start.add_argument('--fragment-mib', type=int, default=64)
    start.add_argument('--outer-lr', type=float, default=0.7)
    start.add_argument('--outer-momentum', type=float, default=0.9)
    start.add_argument('--grace-seconds', type=float, default=8.0)
    start.add_argument('--max-staleness', type=int, default=2)
    start.add_argument('--ema', help='JSON {"decay": 0.99, "warmup": 0} (the default): keep an EMA of the global '
                                     'versions; "off" opts out')
    start.add_argument('--outer-schedule', help='JSON [{"from_round": 0, "lr": 1.0, "momentum": 0.0}, ...]: '
                       'per-fragment merge-round outer settings (replaces --outer-lr/--outer-momentum)')
    finish = commands.add_parser('export')
    finish.add_argument('--root', type=Path, required=True)
    finish.add_argument('--parent', type=Path, required=True)
    finish.add_argument('--base', type=Path)
    finish.add_argument('--learner-checkpoint', type=Path, action='append', default=[])
    finish.add_argument('--learner-upload', action='append', default=[],
                        help='a learner full state uploaded to this coordinator (ROOT/uploads/NAME; sha256 re-verified)')
    finish.add_argument('--out', type=Path, required=True)
    finish.add_argument('--ema-out',
                        help='model-only checkpoint of the coordinator EMA of global versions (default: <out>-ema '
                             'when the run keeps an EMA; "off" skips it)')
    snapshot = commands.add_parser('export-global')
    source = snapshot.add_mutually_exclusive_group(required=True)
    source.add_argument('--root', type=Path, help='the coordinator root on this volume')
    source.add_argument('--coordinator-url',
                        help='a running coordinator: fragments over its API (bearer token and gateway share token from '
                             '$NATLANG_CW_SECRETS/diloco-token and diloco-gateway, as learners receive them)')
    snapshot.add_argument('--timeout', type=float, default=600.0)
    snapshot.add_argument('--workdir', type=Path, help='download scratch for --coordinator-url (default: <out>.fetch)')
    snapshot.add_argument('--parent', type=Path, required=True)
    snapshot.add_argument('--base', type=Path)
    snapshot.add_argument('--out', type=Path, required=True)
    snapshot.add_argument('--ema-out', type=Path)
    args = parser.parse_args(argv)
    {'init': init, 'export': export, 'export-global': export_global}[args.command](args)


if __name__ == '__main__':
    main()
