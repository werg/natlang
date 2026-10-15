"""Executable, resumable all-pair native semantic adaptation curriculum.

The readiness controller supplies an appendable participant manifest. Data,
losses, learning rates and complete-epoch requirements can change while this
process runs. Exposure is tracked per phase, source row and unordered model
pair, so adding a model cannot masquerade as already-completed training.
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
from pathlib import Path
import time

import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from ..common.jsonio import open_jsonl_text, write_json_durable
from ..model.native_model import (SHARED_CHECKPOINT_SCHEMA, load_native_checkpoint, model_state,
    restore_model_state, native_identity)
from ..model.semantic_vae import SemanticStateVAE, StateLayout
from .checkpoint_policy import CheckpointPolicy
from .loop import capture_training_rng_state, restore_training_rng_state
from .native_readiness import _bound
from .native_semantic import PAIR_SCHEMA, SemanticParticipant, parse_pair, producer, joint_loss
from .native_state_sft import make_native_optimizer, append_only_rows

CONFIG_SCHEMA = 'natlang.native-semantic-training/1'


def pair_key(first, second):
    return json.dumps(sorted((first, second)), separators=(',', ':'))


class PairedSource:
    """Compact row identities; tensors and graph plans remain streamed."""
    def __init__(self, binding, *, split):
        self.binding = dict(binding)
        self.path = _bound(binding)
        self.entries, self.groups, self.bindings = [], set(), {}
        for row in self.rows():
            if (row.get('schema') != PAIR_SCHEMA or row.get('split') != split
                    or not isinstance(row.get('models'), dict) or len(row['models']) < 2
                    or not row.get('source_groups')):
                raise ValueError('paired semantic source has an invalid schema, split, groups or model pair')
            key = row['id']
            if key in self.bindings:
                raise ValueError('duplicate paired semantic source identity: ' + key)
            self.bindings[key] = canonical_json_sha256_hex(row)
            self.entries.append({'id': key, 'models': sorted(row['models'])})
            self.groups.update(row['source_groups'])
        if not self.entries:
            raise ValueError('paired semantic source is empty')
        self.sha256 = sha256_file_hex_cached(self.path)
        if self.sha256 != binding['sha256']:
            raise ValueError('paired semantic source changed while indexing')

    def rows(self):
        with open_jsonl_text(self.path) as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def get(self, identity):
        # A whole phase streams sequentially below; this fallback supports a
        # supervisor-selected individual repair without retaining full graphs.
        for row in self.rows():
            if row['id'] == identity:
                if canonical_json_sha256_hex(row) != self.bindings[identity]:
                    raise ValueError('paired semantic source changed after validation')
                return row
        raise ValueError('paired semantic source row disappeared: ' + identity)


def validate_config(config):
    if config.get('schema') != CONFIG_SCHEMA:
        raise ValueError('unsupported native semantic training configuration')
    phases = config.get('phases') or []
    if (not phases or len({phase['id'] for phase in phases}) != len(phases)
            or any(phase.get('kind') not in {'vae', 'joint', 'distill'}
                   or type(phase.get('epochs')) is not int or phase['epochs'] < 1 for phase in phases)):
        raise ValueError('semantic phases need distinct IDs, real objectives and positive complete epochs')
    kinds = [phase['kind'] for phase in phases]
    if any(kinds.index(kind) > kinds.index(later) for kind, later in [('vae', 'joint'), ('joint', 'distill')]
           if kind in kinds and later in kinds):
        raise ValueError('semantic curriculum must learn VAE before joint adaptation before distillation')
    if any(kind not in kinds for kind in ('vae', 'joint', 'distill')):
        raise ValueError('complete semantic curriculum requires VAE, joint and behavioral distillation phases')
    for key in ('backbone_lr', 'vae_lr'):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('semantic learning rates must be finite and positive: ' + key)
    for key in ('checkpoint_every', 'eval_every'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError('semantic checkpoints/evaluation require positive declared cadence: ' + key)
    return config


def needed_pairs(entry, participants, counts, epochs):
    return [(a, b) for a, b in itertools.combinations(sorted(set(entry['models']) & set(participants)), 2)
            if counts.get(pair_key(a, b), 0) < epochs]


def first_work(phases, entries, participants, exposure, cursors=None):
    cursors = {} if cursors is None else cursors
    for phase in phases:
        counts = exposure.get(phase['id'], {})
        for index in range(cursors.get(phase['id'], 0), len(entries)):
            entry = entries[index]
            pairs = needed_pairs(entry, participants, counts.get(entry['id'], {}), phase['epochs'])
            if pairs:
                cursors[phase['id']] = index
                return phase, entry['id'], pairs
        cursors[phase['id']] = len(entries)
    return None


def _adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
        if 'qat_latent_lr' in group:
            group['qat_latent_lr'] = lr


def load_participant(name, binding, config, device):
    path = _bound(binding['checkpoint'])
    model, tokenizer, _, model_config, state = load_native_checkpoint(path, device=device,
        participant=binding['checkpoint'].get('participant'))
    identity = native_identity(model, model_config, checkpoint_sha256=binding['checkpoint']['sha256'])
    if identity != binding['native_identity']:
        raise ValueError('semantic participant actual checkpoint/tokenizer/embedding identity differs: ' + name)
    backend = None
    declaration = config.get('models', {}).get(name, {})
    kind = declaration.get('fp8_training', 'triton-experts' if model_config['architecture'] in {'mellum', 'maple'} else 'off')
    if kind != 'off':
        from .fp8_backend import make_backend
        backend = make_backend(kind, model.port.hf, device=device, recipe=declaration.get('fp8_recipe', 'delayed'),
                               amax_history_len=declaration.get('fp8_amax_history_len', 16))
        model.port.activation_checkpoint = backend.checkpoint
    elif model_config['architecture'] in {'mellum', 'maple'}:
        raise ValueError('native Mellum semantic training requires the shared triton-experts memory path')
    optimizer, _ = make_native_optimizer(model, lr=config['backbone_lr'],
        weight_decay=config.get('weight_decay', .1), latent_ns=config.get('latent_ns', 'quintic'), in_backward=False)
    inherited = state.get('training', {})
    if inherited.get('optimizer') and config.get('inherit_optimizer', True):
        optimizer.load_state_dict(inherited['optimizer'])
    if backend is not None and inherited.get('fp8') is not None:
        backend.restore_extra_state(inherited['fp8'])
    _adjust_lr(optimizer, config['backbone_lr'])
    model.port.checkpoint_layers = True
    model.train()
    return SemanticParticipant(model, tokenizer, model_config, identity, optimizer, backend)


class ExternalArtifacts:
    """Use the existing exact native block bundle, including path/hash checks."""
    def __init__(self, config):
        self.bundles = {}
        for name, binding in config.get('external_bundles', {}).items():
            path = _bound(binding)
            value = json.loads(path.read_text())
            if value.get('schema') != 'natlang.rl-block-bundle/1':
                raise ValueError('semantic external observations require a verified state bundle')
            self.bundles[name] = (path, value)

    def __call__(self, name, block_id):
        if name not in self.bundles:
            raise ValueError('semantic model has no declared external native-state bundle: ' + name)
        root, manifest = self.bundles[name]
        entry = manifest['blocks'][block_id]
        path = (root.parent/entry['path']).resolve(strict=True)
        if path.parent != root.parent or path.stat().st_size != entry['bytes']:
            raise ValueError('semantic external native artifact path/size differs from its exact bundle')
        if sha256_file_hex_cached(path) != entry['sha256']:
            raise ValueError('semantic external native artifact bytes changed')
        from ..serve.store import decode_native_state
        block, actual_id = decode_native_state(path.read_bytes())
        if actual_id != block_id:
            raise ValueError('semantic external native artifact content ID changed')
        return block


def _first_view(source, participants, name):
    for row in source.rows():
        if name in row['models']:
            example, view = parse_pair(row, participants)[name]
            return example, view
    raise ValueError('ready semantic participant has no admitted paired training trajectory: ' + name)


def _layout(source, participants, name, resolver):
    participant = participants[name]
    example, view = _first_view(source, participants, name)
    with torch.no_grad():
        external = {key: resolver(name, value['id']) for key, value in example.external_states.items()}
        block = producer(participant, example, view, supervise=False, external_blocks=external)[0]
    return StateLayout.from_block(block)


def _vae_optimizer(vae, config, saved=None):
    named = dict(vae.named_parameters())
    if saved:
        groups = [{'params': [named[name] for name in group], 'lr': config['vae_lr']}
                  for group in saved['parameter_groups']]
        if set(itertools.chain.from_iterable(saved['parameter_groups'])) != set(named):
            raise ValueError('semantic optimizer parameter identities differ from the restored VAE')
    else:
        groups = [{'params': list(named.values()), 'lr': config['vae_lr']}]
    optimizer = torch.optim.AdamW(groups, lr=config['vae_lr'], weight_decay=config.get('vae_weight_decay', .01))
    if saved:
        optimizer.load_state_dict(saved['state'])
    _adjust_lr(optimizer, config['vae_lr'])
    return optimizer


def _optimizer_names(vae, optimizer):
    names = {id(parameter): name for name, parameter in vae.named_parameters()}
    return [[names[id(parameter)] for parameter in group['params']] for group in optimizer.param_groups]


def _manifest(path):
    value = json.loads(Path(path).read_text())
    if value.get('schema') != 'natlang.native-shared-participants/1' or len(value.get('models') or []) < 2:
        raise ValueError('shared semantic training needs at least two ready native models')
    if canonical_json_sha256_hex(value['models']) != value.get('signature'):
        raise ValueError('shared participant manifest signature differs from its exact model bindings')
    models = {item['model_id']: item for item in value['models']}
    if len(models) != len(value['models']):
        raise ValueError('shared semantic participant identities must be distinct')
    return value, models


def _sources(config, old=None):
    train = PairedSource(config['pairs'], split='train')
    held = PairedSource(config['held_pairs'], split='test')
    if train.groups & held.groups:
        raise ValueError('semantic train/held source groups overlap')
    if old:
        append_only_rows(old['train'], train.bindings, label='semantic train')
        append_only_rows(old['held'], held.bindings, label='semantic held')
    return train, held


def _native_modules(participants):
    stack = contextlib.ExitStack()
    for participant in participants.values():
        if participant.backend is not None:
            stack.enter_context(participant.backend.native_modules())
    return stack


def save_shared(policy, participants, vae, optimizer, *, training, device):
    with _native_modules(participants):
        state = {'schema': SHARED_CHECKPOINT_SCHEMA,
            'participants': {name: {'model_config': entry.config, 'model_state': model_state(entry.model),
                'source_identity': entry.identity, 'optimizer': entry.optimizer.state_dict(),
                'fp8': entry.backend.extra_state() if entry.backend is not None else None}
                for name, entry in participants.items()},
            'vae_specification': vae.specification(), 'vae_state': vae.state_dict(),
            'vae_optimizer': {'state': optimizer.state_dict(), 'parameter_groups': _optimizer_names(vae, optimizer)},
            'training': {**training, 'rng': capture_training_rng_state(device)}}
        policy.save(state)
    checkpoint_sha = sha256_file_hex_cached(policy.slot)
    # Final durable bytes bind all participants, rather than publishing
    # shape-only adapters usable against arbitrary unrelated checkpoints.
    identities = {name: native_identity(entry.model, entry.config, checkpoint_sha256=checkpoint_sha)
                  for name, entry in participants.items()}
    tokenizers = {}
    for name, entry in participants.items():
        if not hasattr(entry.tokenizer, 'save_pretrained'):
            continue  # mechanism tests may use a small in-memory tokenizer
        directory = policy.out/'semantic-tokenizers'/canonical_json_sha256_hex(name)
        entry.tokenizer.save_pretrained(directory)
        tokenizers[name] = {'path': str(directory.relative_to(policy.out)),
            'files': {str(path.relative_to(directory)): sha256_file_hex_cached(path)
                      for path in directory.rglob('*') if path.is_file()}}
    export = {'schema': 'natlang.native-semantic-adapter/1',
        'checkpoint': {'path': str(policy.slot.resolve()), 'sha256': checkpoint_sha},
        'specification_sha256': canonical_json_sha256_hex(vae.specification()),
        'models': {name: {'native_identity': identity,
                        'checkpoint': {'path': str(policy.slot.resolve()), 'sha256': checkpoint_sha,
                                       'participant': name}} for name, identity in identities.items()},
        'step': training['step'], 'communication': 'shared-semantic-state-deltas'}
    export['tokenizers'] = tokenizers
    write_json_durable(policy.out/'semantic-adapter.json', export)
    return export


def _exposure_complete(phases, source, participants, exposure):
    missing = []
    admitted_pairs = 0
    for entry in source.entries:
        available = list(itertools.combinations(sorted(set(entry['models']) & set(participants)), 2))
        admitted_pairs += len(available)
        for phase in phases:
            for a, b in available:
                actual = exposure.get(phase['id'], {}).get(entry['id'], {}).get(pair_key(a, b), 0)
                if actual < phase['epochs']:
                    missing.append({'row': entry['id'], 'phase': phase['id'], 'models': [a, b],
                                    'visits': actual, 'required': phase['epochs']})
    if not admitted_pairs:
        raise ValueError('ready participants have no paired admitted training data in common')
    return not missing, missing


def measure_tasks(participants, export, config, out, step, device):
    """Serve the same live trained weights; do not allocate a second backbone."""
    from ..eval.native_state import task_execution_receipt
    from ..eval.task_cadence import TaskEval, parse_task_eval
    from ..model.shared_state import RoleTokens
    from ..serve.engine import Engine
    from ..serve.store import TensorStore
    results = {}
    rng = capture_training_rng_state(device)
    try:
        with _native_modules(participants):
            for name, participant in participants.items():
                declaration = config.get('models', {}).get(name, {}).get('task_eval', config.get('task_eval'))
                if not declaration:
                    continue
                spec = parse_task_eval(json.dumps(declaration))
                identity = export['models'][name]['native_identity']
                mode = participant.model.training
                participant.model.eval()
                try:
                    engine = Engine(participant.model.port, None, participant.tokenizer, TensorStore(),
                        'native-state/' + participant.config['architecture'], model_name='native-' + name,
                        device=device, prompt_rendering=spec.get('prompt_rendering', 'full'),
                        shared_model=participant.model, role_tokens=RoleTokens(*participant.model.marker_ids),
                        model_config=participant.config, checkpoint_sha256=identity['checkpoint_sha256'],
                        embedding_weight_sha256=identity['embedding_weight_sha256'])
                    engine.native_identity = identity
                    directory = Path(out)/('participant-' + canonical_json_sha256_hex(name))
                    evaluator = TaskEval(spec, directory, engine, prompt_rendering=spec.get('prompt_rendering', 'full'))
                    summary = evaluator.cadence(step)
                    receipt = task_execution_receipt(summary, identity,
                        checkpoint_sha256=export['checkpoint']['sha256'])
                    path = directory/'native-task-execution-receipt.json'
                    write_json_durable(path, receipt)
                    results[name] = {'path': str(path.resolve()), 'sha256': sha256_file_hex_cached(path),
                                     'passed': receipt['passed'], 'task_successes': receipt['task_successes']}
                finally:
                    participant.model.train(mode)
    finally:
        restore_training_rng_state(rng, device)
    return results


def run(config_path, participants_path, out, *, resume=None, max_updates=None, device='cuda'):
    """Run the actual curriculum; max_updates is an explicit incomplete pause."""
    config_path, out = Path(config_path), Path(out)
    config = validate_config(json.loads(config_path.read_text()))
    manifest, bindings = _manifest(participants_path)
    prior = CheckpointPolicy.load(resume) if resume else None
    if prior and prior.get('schema') != SHARED_CHECKPOINT_SCHEMA:
        raise ValueError('semantic continuation needs its complete shared model/VAE/optimizer checkpoint')
    training = dict(prior['training']) if prior else {'step': 0, 'exposure': {}, 'config_history': []}
    train, held = _sources(config, training.get('source_bindings'))
    resolver = ExternalArtifacts(config)
    participants = {name: load_participant(name, binding, config, device) for name, binding in bindings.items()}
    if prior:
        if set(prior['participants']) - set(participants):
            raise ValueError('semantic continuation cannot silently remove existing ready models')
        for name, saved in prior['participants'].items():
            participant = participants[name]
            if participant.config != saved['model_config'] or participant.identity != saved['source_identity']:
                raise ValueError('semantic continuation changed the declared model lineage: ' + name)
            with _native_modules({name: participant}):
                restore_model_state(participant.model, saved['model_state'])
            participant.optimizer.load_state_dict(saved['optimizer'])
            if participant.backend is not None:
                participant.backend.restore_extra_state(saved['fp8'])
        vae = SemanticStateVAE.from_specification(prior['vae_specification']).to(device)
        vae.load_state_dict(prior['vae_state'], strict=True)
        optimizer = _vae_optimizer(vae, config, prior['vae_optimizer'])
        restore_training_rng_state(training['rng'], device)
        del prior
    else:
        vae = SemanticStateVAE({name: _layout(train, participants, name, resolver) for name in participants},
            latent_width=config.get('latent_width', 128), hidden_width=config.get('hidden_width', 512)).to(device)
        optimizer = _vae_optimizer(vae, config)
    policy = CheckpointPolicy(out, every_steps=config['checkpoint_every'], prune_best=False).install_signal_handlers()
    started_step, started = training['step'], time.time()
    trainable = {name: {key for key, value in entry.model.named_parameters() if value.requires_grad}
                 for name, entry in participants.items()}
    # Existing participant values are never replaced while their optimizer is
    # alive; a new declaration requires an explicit continuation or restart.
    initial_bindings = dict(bindings)
    last_config = None
    stream, cached_row = None, None
    cursors = {}
    previous_manifest = manifest['signature']
    metrics = {}
    while not policy.signaled:
        updated = validate_config(json.loads(config_path.read_text()))
        config_sha = canonical_json_sha256_hex(updated)
        if config_sha != last_config:
            if (updated.get('latent_width', 128), updated.get('hidden_width', 512)) != (vae.latent_width, vae.hidden_width):
                raise ValueError('semantic latent width/layout change requires explicit adapter migration')
            if updated['pairs'] != config['pairs'] or updated['held_pairs'] != config['held_pairs']:
                train, held = _sources(updated, {'train': train.bindings, 'held': held.bindings})
                stream = None
            config = updated
            resolver = ExternalArtifacts(config)
            cursors = {}
            training['config_history'].append({'step': training['step'], 'sha256': config_sha, 'config': config})
            last_config = config_sha
        manifest, bindings = _manifest(participants_path)
        if manifest['signature'] != previous_manifest:
            cursors = {}
            previous_manifest = manifest['signature']
        for name, binding in bindings.items():
            if name in initial_bindings and binding != initial_bindings[name]:
                raise ValueError('ready model changed while joint optimizer was active: ' + name)
            if name not in participants:
                participants[name] = load_participant(name, binding, config, device)
                initial_bindings[name] = binding
                trainable[name] = {key for key, value in participants[name].model.named_parameters() if value.requires_grad}
            if name not in vae.layouts:
                added = vae.add_model(name, _layout(train, participants, name, resolver))
                optimizer.add_param_group({'params': [value for _, value in added], 'lr': config['vae_lr']})
        if set(participants) - set(bindings):
            raise ValueError('ready-model manifest removed an active semantic participant')
        work = first_work(config['phases'], train.entries, participants, training['exposure'], cursors)
        if work is None:
            break
        phase, row_id, pairs = work
        if stream is None:
            # Keep compressed JSONL sequential between neighboring updates.
            # A new epoch/new phase intentionally restarts the stream once.
            stream = train.rows()
        row = cached_row if cached_row and cached_row['id'] == row_id else None
        if row is None:
            for candidate in stream:
                if candidate['id'] == row_id:
                    row = candidate
                    break
        if row is None:
            row = train.get(row_id)
            stream = None
        cached_row = row
        if canonical_json_sha256_hex(row) != train.bindings[row_id]:
            raise ValueError('semantic training row changed under its exposure cursor')
        active = {name for pair in pairs for name in pair}
        selected = {name: participant for name, participant in participants.items() if name in active}
        options = {**config.get('options', {}), **phase.get('options', {})}
        for name, participant in participants.items():
            participant.optimizer.zero_grad(set_to_none=True)
            _adjust_lr(participant.optimizer, options.get('backbone_lr', config['backbone_lr']))
            for key, value in participant.model.named_parameters():
                value.requires_grad_(phase['kind'] != 'vae' and name in active and key in trainable[name])
        optimizer.zero_grad(set_to_none=True)
        vae.requires_grad_(phase['kind'] != 'distill')
        _adjust_lr(optimizer, options.get('vae_lr', config['vae_lr']))
        with contextlib.ExitStack() as stack:
            for participant in selected.values():
                if participant.backend is not None:
                    stack.enter_context(participant.backend.autocast())
            loss, details = joint_loss(vae, selected, row, phase=phase['kind'], options=options,
                                       fetch_external=resolver)
            loss.backward()
        parameters = [parameter for participant in selected.values() for parameter in participant.model.parameters()
                      if parameter.requires_grad and parameter.grad is not None]
        parameters += [parameter for parameter in vae.parameters() if parameter.requires_grad and parameter.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(parameters, options.get('max_grad_norm', float('inf')))
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('nonfinite semantic gradient; no model optimizer was stepped')
        # No post-accumulate optimizer hooks are installed: all producer,
        # consumer and VAE adjoints finish before any parameter mutation.
        if phase['kind'] != 'vae':
            for name, participant in selected.items():
                participant.optimizer.step()
                updates = training.setdefault('model_updates', {})
                updates[name] = updates.get(name, 0) + 1
        if phase['kind'] != 'distill':
            optimizer.step()
        training['step'] += 1
        counts = training['exposure'].setdefault(phase['id'], {}).setdefault(row_id, {})
        # joint_loss includes every unordered pair of selected participants.
        for a, b in itertools.combinations(sorted(selected), 2):
            key = pair_key(a, b)
            counts[key] = counts.get(key, 0) + 1
        metrics = {'step': training['step'], 'phase': phase['id'], 'source_row': row_id,
                   'loss': float(loss.detach()), 'gradient_norm': float(norm), 'details': details,
                   'seconds': time.time() - started}
        training['source_bindings'] = {'train': train.bindings, 'held': held.bindings}
        training['participants_signature'] = manifest['signature']
        if training['step'] % config['eval_every'] == 0:
            held_values = []
            rng = capture_training_rng_state(device)
            modes = {name: entry.model.training for name, entry in participants.items()}
            vae_mode = vae.training
            try:
                for entry in participants.values():
                    entry.model.eval()
                vae.eval()
                with torch.no_grad(), _native_modules(participants):
                    for held_row in held.rows():
                        if len(set(held_row['models']) & set(participants)) >= 2:
                            value, _ = joint_loss(vae, participants, held_row, phase=phase['kind'], options=options,
                                                  fetch_external=resolver)
                            held_values.append(float(value))
            finally:
                for name, entry in participants.items():
                    entry.model.train(modes[name])
                vae.train(vae_mode)
                restore_training_rng_state(rng, device)
            if not held_values:
                raise ValueError('ready semantic participants have no paired held evaluation data')
            metrics['held_loss'] = sum(held_values) / len(held_values)
            metrics['held_rows'] = len(held_values)
        write_json_durable(out/'metrics.json', metrics)
        due_task = bool(config.get('task_eval')) and training['step'] % config.get('task_eval_every', config['eval_every']) == 0
        if policy.due(training['step']) or due_task:
            current = save_shared(policy, participants, vae, optimizer, training=training, device=device)
            if due_task:
                metrics['task_execution'] = measure_tasks(participants, current, config, out, training['step'], device)
                write_json_durable(out/'metrics.json', metrics)
        if max_updates is not None and training['step'] - started_step >= max_updates:
            break
    # Restore the declared trainable set before serializing/serving; phase
    # freezing is transient and must not become the next continuation's policy.
    for name, participant in participants.items():
        for key, parameter in participant.model.named_parameters():
            parameter.requires_grad_(key in trainable[name])
    vae.requires_grad_(True)
    training['source_bindings'] = {'train': train.bindings, 'held': held.bindings}
    training['participants_signature'] = manifest['signature']
    export = save_shared(policy, participants, vae, optimizer, training=training, device=device)
    complete, missing = _exposure_complete(config['phases'], train, participants, training['exposure'])
    certificates = {}
    tasks = {}
    if complete:
        from ..eval.native_state import qualify_existing
        from ..model.shared_state import RoleTokens
        tasks = measure_tasks(participants, export, config, out, training['step'], device)
        rng = capture_training_rng_state(device)
        try:
            with _native_modules(participants):
                for name, participant in participants.items():
                    certificate = qualify_existing(participant.model, participant.tokenizer,
                        RoleTokens(*participant.model.marker_ids), participant.config,
                        checkpoint_sha256=export['checkpoint']['sha256'],
                        forward_atol=config.get('forward_atol', 1e-4),
                        gradient_atol=config.get('gradient_atol', 1e-5),
                        task_execution_receipt=tasks.get(name, {}).get('path'),
                        backbone_optimizer_updates=training.get('model_updates', {}).get(name, 0))
                    path = out/('runtime-' + canonical_json_sha256_hex(name) + '.json')
                    write_json_durable(path, certificate)
                    certificates[name] = {'path': str(path.resolve()), 'sha256': sha256_file_hex_cached(path),
                                          'passed': certificate['passed']}
        finally:
            restore_training_rng_state(rng, device)
        complete = all(value['passed'] for value in certificates.values())
    configuration = out/'completed-curriculum.json'
    write_json_durable(configuration, config)
    report = {'schema': 'natlang.native-shared-stage-completion/1', 'completed': complete,
        'participants_signature': manifest['signature'], 'step': training['step'],
        'curriculum_signature': canonical_json_sha256_hex(config),
        'curriculum': {'path': str(configuration.resolve()), 'sha256': sha256_file_hex_cached(configuration)},
        'checkpoint': export['checkpoint'], 'adapter': {'path': str((out/'semantic-adapter.json').resolve()),
            'sha256': sha256_file_hex_cached(out/'semantic-adapter.json')},
        'source_bindings': {'train': train.binding, 'held': held.binding},
        'phases': config['phases'], 'exposure': training['exposure'], 'missing_exposure': missing,
        'runtime_certificates': certificates,
        'task_execution_receipts': tasks,
        'production_ready': bool(complete and set(export['tokenizers']) == set(participants)
                                 and set(tasks) == set(participants)
                                 and all(value['passed'] for value in tasks.values())),
        'task_lift': 'not-claimed', 'metrics': metrics}
    write_json_durable(out/'shared-stage-completion.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--participants', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    declaration = validate_config(json.loads(Path(args.config).read_text()))
    if not declaration.get('task_eval'):
        raise ValueError('production semantic training requires actual natlang task evaluation for the trained weights')
    if args.max_updates is not None and args.max_updates <= 0:
        raise ValueError('explicit semantic interruption must permit at least one update')
    resume = args.resume
    if resume is None and (Path(args.out)/'checkpoint.pt').is_file():
        resume = str(Path(args.out)/'checkpoint.pt')
        print(json.dumps({'event': 'native_semantic_continue', 'checkpoint': resume}), flush=True)
    report = run(args.config, args.participants, args.out, resume=resume,
                 max_updates=args.max_updates, device=args.device)
    print(json.dumps({'completed': report['completed'], 'step': report['step'],
                      'checkpoint': report['checkpoint']}), flush=True)
    return 0 if report['production_ready'] else 75


if __name__ == '__main__':
    raise SystemExit(main())
