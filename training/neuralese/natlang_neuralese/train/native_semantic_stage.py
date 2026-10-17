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
    config.setdefault('checkpoint_seconds', 3600.)
    if (type(config['checkpoint_seconds']) not in (int, float)
            or not math.isfinite(config['checkpoint_seconds']) or config['checkpoint_seconds'] < 0):
        raise ValueError('semantic checkpoint_seconds must be finite and nonnegative')
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
    from .native_learning_rates import validate_learned_row_rate
    from .native_optimizer_migration import validate_native_optimizer_migration
    if any(phase.get('options',{}).get('native_optimizer_migration') is not None for phase in phases):
        raise ValueError('native optimizer migration must be declared globally or per model before phase training')
    declarations=[config,config.get('options') or {},*config.get('models',{}).values(),
                  *[phase.get('options') or {} for phase in phases]]
    for declaration in declarations:
        validate_native_optimizer_migration(declaration.get('native_optimizer_migration'))
        if 'checkpoint_input_offload' in declaration and type(declaration['checkpoint_input_offload']) is not bool:
            raise ValueError('semantic checkpoint_input_offload must be a boolean')
        directory = declaration.get('checkpoint_input_offload_directory')
        if directory is not None and not isinstance(directory,str):
            raise ValueError('semantic checkpoint_input_offload_directory must be a string or omitted')
        for name in ('learned_row_learning_rate','interface_row_learning_rate'):
            if name in declaration:
                validate_learned_row_rate(declaration[name],name=name)
    for key in ('backbone_lr', 'vae_lr'):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('semantic learning rates must be finite and positive: ' + key)
    for key in ('checkpoint_every', 'eval_every'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError('semantic checkpoints/evaluation require positive declared cadence: ' + key)
    adaptive=config.get('adaptive_refresh')
    if adaptive is not None:
        if not isinstance(adaptive,dict):raise ValueError('adaptive_refresh must declare phase work')
        for phase in phases:
            value=(phase.get('options') or {}).get('refresh_updates',adaptive.get('updates_per_phase'))
            if type(value) is not int or value<=0:raise ValueError('adaptive refresh needs positive updates for every phase')
    if config.get('held_eval_rows') is not None and (type(config['held_eval_rows']) is not int or config['held_eval_rows']<=0):
        raise ValueError('held_eval_rows must be a positive whole-graph count')
    from .native_semantic_policy import validate_policy_training
    validate_policy_training(config.get('policy_training'),[phase['id'] for phase in phases])
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


def _adjust_lr(optimizer, lr, learned_row_lr=None, interface_row_lr=None):
    if hasattr(optimizer,'_native_named_parameters'):
        from .native_learning_rates import apply_native_learning_rates
        return apply_native_learning_rates(optimizer,backbone_lr=lr,learned_row_lr=learned_row_lr,interface_row_lr=interface_row_lr)
    for group in optimizer.param_groups:
        group['lr'] = lr
        if 'qat_latent_lr' in group:
            group['qat_latent_lr'] = lr


def load_participant(name, binding, config, device):
    from .native_optimizer_migration import native_optimizer_migration_option
    declaration = config.get('models', {}).get(name, {})
    migration=native_optimizer_migration_option(config,name)
    path = _bound(binding['checkpoint'])
    model, tokenizer, _, model_config, state = load_native_checkpoint(path, device=device,
        base=declaration.get('base'), participant=binding['checkpoint'].get('participant'))
    identity = native_identity(model, model_config, checkpoint_sha256=binding['checkpoint']['sha256'])
    if identity != binding['native_identity']:
        raise ValueError('semantic participant actual checkpoint/tokenizer/embedding identity differs: ' + name)
    backend = None
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
    inherited = state.get('training') or state.get('shared_training') or {}
    from .recipe import training_recipe_binding,native_recipe_provenance
    recipe_binding=training_recipe_binding(state)
    declared=native_recipe_provenance(binding.get('recipe'))
    if declared is not None:
        if recipe_binding is not None and recipe_binding!=declared:
            raise ValueError('semantic participant recipe provenance differs from its actual checkpoint: '+name)
        if recipe_binding is None:
            if not binding.get('readiness'):
                raise ValueError('semantic participant needs exact readiness to recover missing saved recipe provenance')
            readiness=json.loads(_bound(binding['readiness']).read_text())
            if (readiness.get('recipe')!=declared or readiness.get('native_identity')!=identity
                    or (readiness.get('checkpoint') or {}).get('sha256')!=binding['checkpoint']['sha256']):
                raise ValueError('semantic participant recipe recovery differs from its actual readiness/checkpoint')
            from .native_readiness import validate_ready
            requirement=binding.get('readiness_requirement')
            if (not isinstance(requirement,dict) or requirement.get('model_id')!=name
                    or requirement.get('recipe_sha256')!=declared['sha256']
                    or not requirement.get('stage_ids')
                    or binding.get('stage_id') not in requirement['stage_ids']):
                raise ValueError('semantic recipe recovery requires its declared original readiness policy')
            checked=validate_ready(_bound(binding['readiness']),requirement)
            if checked is None or checked['stage_id']!=binding['stage_id']:
                raise ValueError('semantic participant recipe recovery requires qualified original readiness')
        recipe_binding=declared
    if inherited.get('optimizer') and config.get('inherit_optimizer', True):
        optimizer.load_state_dict(inherited['optimizer'],native_optimizer_migration=migration)
    if backend is not None and inherited.get('fp8') is not None:
        backend.restore_extra_state(inherited['fp8'])
    _adjust_lr(optimizer, config['backbone_lr'], declaration.get('learned_row_learning_rate',config.get('learned_row_learning_rate')),
        declaration.get('interface_row_learning_rate',config.get('interface_row_learning_rate')))
    from ..common.checkpoint_offload import configure_checkpoint_storage_options
    configure_checkpoint_storage_options(model.port,declaration,config,inherited.get('options') or {})
    model.port.checkpoint_layers = True
    model.train()
    result = SemanticParticipant(model, tokenizer, model_config, identity, optimizer, backend,recipe_binding=recipe_binding)
    from .native_continuation import collect_sft_progress
    result.sft_progress = collect_sft_progress(inherited)
    result.global_step = int(inherited.get('global_step', inherited.get('step', 0)))
    if binding.get('readiness'):
        proof=json.loads(_bound(binding['readiness']).read_text())
        foundation=proof.get('communication_foundation') or {}
        if proof.get('foundation_ready') and foundation.get('complete'):
            frame=result.sft_progress.get(proof['stage_id'])
            if frame is None:
                raise ValueError('foundation checkpoint lost its mandatory supervised progress: '+name)
            frame['completed_foundations']=sorted(set(frame.get('completed_foundations') or [])|{foundation['cohort_sha256']})
    result.continuation_proof = (inherited.get('options') or {}).get('continuation') or {}
    return result


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
    from ..common.checkpoint_offload import checkpoint_input_storage_options
    with _native_modules(participants):
        state = {'schema': SHARED_CHECKPOINT_SCHEMA,
            'participants': {name: {'model_config': entry.config, 'model_state': model_state(entry.model),
                'source_identity': entry.identity, 'optimizer': entry.optimizer.state_dict(),
                'sft_progress': entry.sft_progress, 'global_step': entry.global_step,
                'training_options': {**checkpoint_input_storage_options(entry.model.port),
                    **({'recipe_binding':entry.recipe_binding} if entry.recipe_binding is not None else {})},
                'fp8': entry.backend.extra_state() if entry.backend is not None else None}
                for name, entry in participants.items()},
            'vae_specification': vae.specification(), 'vae_state': vae.state_dict(),
            'vae_optimizer': {'state': optimizer.state_dict(), 'parameter_groups': _optimizer_names(vae, optimizer)},
            'training': {**training, 'rng': capture_training_rng_state(device)}}
        from ..model.native_shared_checkpoint import save_shared_checkpoint
        save_shared_checkpoint(policy, state)
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
        from ..data.text_corpus import tokenizer_fingerprint
        history_profile = getattr(entry.tokenizer, 'natlang_history_reasoning', None)
        declared_profile = entry.config.get('history_reasoning_policy')
        if declared_profile is not None and (not isinstance(history_profile, dict)
                or history_profile.get('policy') != declared_profile):
            raise ValueError('semantic participant tokenizer profile differs from its checkpoint configuration')
        if isinstance(history_profile, dict):
            history_binding = {'schema': 'natlang.raw-native-tokenizer-history/1',
                'checkpoint_sha256': checkpoint_sha,
                'tokenizer_fingerprint': tokenizer_fingerprint(entry.tokenizer),
                'history_reasoning': dict(history_profile)}
            write_json_durable(directory/'natlang-history-reasoning.json', history_binding)
        tokenizers[name] = {'path': str(directory.relative_to(policy.out)),
            'files': {str(path.relative_to(directory)): sha256_file_hex_cached(path)
                      for path in directory.rglob('*') if path.is_file()}}
    from .native_continuation import progress_obligations, raw_communication_rows
    export = {'schema': 'natlang.native-semantic-adapter/1',
        'checkpoint': {'path': str(policy.slot.resolve()), 'sha256': checkpoint_sha},
        'specification_sha256': canonical_json_sha256_hex(vae.specification()),
        'models': {name: {'native_identity': identity,
                        **({'history_reasoning_policy': participants[name].tokenizer.natlang_history_reasoning['policy']}
                           if isinstance(getattr(participants[name].tokenizer,'natlang_history_reasoning',None),dict) else {}),
                        'mandatory_sft':progress_obligations(participants[name].sft_progress),
                        'raw_communication_rows':raw_communication_rows(participants[name].sft_progress),
                        'marker_ids':list(participants[name].model.marker_ids),
                        **({'recipe_binding':participants[name].recipe_binding} if participants[name].recipe_binding is not None else {}),
                        'checkpoint': {'path': str(policy.slot.resolve()), 'sha256': checkpoint_sha,
                                       'participant': name}} for name, identity in identities.items()},
        'step': training['step'], 'communication': 'shared-semantic-state-deltas'}
    export['tokenizers'] = tokenizers
    export['semantic_obligations']=training.get('semantic_obligations')
    export['refresh_progress']=training.get('refresh_cycle')
    if training.get('policy_training_phase'):
        from .native_semantic_policy import checkpoint_progress
        export['policy_training_progress']=checkpoint_progress(training)
    history=training.get('config_history') or []
    if history:
        qualified=history[-1]['config']
        export['qualified_curriculum']={'config':qualified,'sha256':canonical_json_sha256_hex(qualified)}
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


def measure_tasks(participants, export, config, out, step, device, vae=None):
    """Serve the same live trained weights; do not allocate a second backbone."""
    from ..eval.native_state import task_execution_receipt
    from ..eval.task_cadence import TaskEval, parse_task_eval
    from ..model.shared_state import RoleTokens
    from ..serve.engine import Engine
    from ..serve.store import TensorStore
    results = {}
    channel=None
    if vae is not None:
        from ..model.semantic_channel import SemanticChannel
        channel=SemanticChannel(vae,{name:entry['native_identity'] for name,entry in export['models'].items()},
            {name:entry.tokenizer for name,entry in participants.items()},
            adapter_sha256=sha256_file_hex_cached(Path(out)/'semantic-adapter.json'))
    rng = capture_training_rng_state(device)
    vae_mode=vae.training if vae is not None else None
    try:
        if vae is not None:vae.eval()
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
                    if channel is not None:
                        from ..serve import bind_native_semantic_channel
                        bind_native_semantic_channel(engine,channel,name)
                    directory = Path(out)/('participant-' + canonical_json_sha256_hex(name))
                    evaluator = TaskEval(spec, directory, engine, prompt_rendering=spec.get('prompt_rendering', 'full'))
                    summary = evaluator.cadence(step)
                    receipt = task_execution_receipt(summary, identity,
                        checkpoint_sha256=export['checkpoint']['sha256'])
                    if channel is not None:
                        receipt.update(communication='shared-semantic-state-deltas',semantic_adapter_sha256=channel.adapter_sha256)
                    path = directory/'native-task-execution-receipt.json'
                    write_json_durable(path, receipt)
                    results[name] = {'path': str(path.resolve()), 'sha256': sha256_file_hex_cached(path),
                                     'passed': receipt['passed'], 'task_successes': receipt['task_successes']}
                finally:
                    participant.model.train(mode)
    finally:
        if vae is not None:vae.train(vae_mode)
        restore_training_rng_state(rng, device)
    return results


def run(config_path, participants_path, out, *, resume=None, max_updates=None, device='cuda', refresh_participants=False, stop=None):
    """Run the actual curriculum; max_updates is an explicit incomplete pause."""
    config_path, out = Path(config_path), Path(out)
    config = validate_config(json.loads(config_path.read_text()))
    manifest, bindings = _manifest(participants_path)
    semantic_parent=manifest.get('semantic_parent')
    if semantic_parent is not None:
        if not isinstance(semantic_parent,dict) or set(semantic_parent)!={'checkpoint','adapter','completion'}:
            raise ValueError('semantic continuation parent must bind its exact checkpoint, adapter and completion')
        parent_checkpoint=_bound(semantic_parent['checkpoint'])
        parent_adapter=json.loads(_bound(semantic_parent['adapter']).read_text())
        parent_completion=json.loads(_bound(semantic_parent['completion']).read_text())
        if (parent_adapter.get('checkpoint')!=semantic_parent['checkpoint']
                or parent_completion.get('checkpoint')!=semantic_parent['checkpoint']
                or parent_completion.get('adapter')!=semantic_parent['adapter']
                or parent_completion.get('production_ready') is not True):
            raise ValueError('semantic continuation parent is not its exact qualified publication')
        from .native_readiness import validate_completion
        validate_completion(_bound(semantic_parent['completion']),
            {(parent_completion['participants_signature'],parent_completion['curriculum_signature']):
                {'models':list(parent_adapter['models']),'execution_signature':'semantic-parent'}})
        if resume is None:resume=parent_checkpoint
    prior = CheckpointPolicy.load(resume) if resume else None
    if prior and prior.get('schema') != SHARED_CHECKPOINT_SCHEMA:
        raise ValueError('semantic continuation needs its complete shared model/VAE/optimizer checkpoint')
    training = dict(prior['training']) if prior else {'step': 0, 'exposure': {}, 'config_history': []}
    if semantic_parent is not None:
        parent_identity={role+'_sha256':binding['sha256'] for role,binding in semantic_parent.items()}
        if (sha256_file_hex_cached(resume)!=semantic_parent['checkpoint']['sha256']
                and training.get('semantic_parent_identity')!=parent_identity):
            raise ValueError('semantic continuation checkpoint belongs to a different publication cycle')
        training['semantic_parent_identity']=parent_identity
    train, held = _sources(config, training.get('source_bindings'))
    resolver = ExternalArtifacts(config)
    participants = {name: load_participant(name, binding, config, device) for name, binding in bindings.items()}
    if all(binding.get('readiness') for binding in bindings.values()):
        from .native_continuation import qualified_refresh_config,raw_communication_rows
        qualified_refresh_config({},config_path,{name:{'native_identity':participant.identity,
            'marker_ids':list(participant.model.marker_ids),
            'raw_communication_rows':raw_communication_rows(participant.sft_progress)}
            for name,participant in participants.items()})
    changed_models = set()
    if prior:
        prior_sha = sha256_file_hex_cached(resume)
        if set(prior['participants']) - set(participants):
            raise ValueError('semantic continuation cannot silently remove existing ready models')
        for name, saved in prior['participants'].items():
            participant = participants[name]
            if participant.config != saved['model_config']:
                raise ValueError('semantic continuation changed model geometry: ' + name)
            from .recipe import native_recipe_provenance
            previous_recipe=native_recipe_provenance((saved.get('training_options') or {}).get('recipe_binding'))
            if previous_recipe is not None and participant.recipe_binding is not None:
                if previous_recipe.get('lineage_sha256',previous_recipe['sha256'])!=participant.recipe_binding.get('lineage_sha256',participant.recipe_binding['sha256']):
                    raise ValueError('semantic refresh changed preserved participant recipe lineage: '+name)
            same_source = participant.identity == saved['source_identity']
            same_export = (bindings[name]['checkpoint']['sha256'] == prior_sha
                           and bindings[name]['checkpoint'].get('participant') == name)
            if participant.recipe_binding is None and previous_recipe is not None:
                if not same_source and not same_export:
                    raise ValueError('updated semantic participant lost its current recipe provenance: '+name)
                participant.recipe_binding=previous_recipe
            if (same_source or same_export) and previous_recipe is not None and participant.recipe_binding!=previous_recipe:
                raise ValueError('unchanged semantic participant changed its saved recipe provenance: '+name)
            if not same_source and not same_export:
                if not refresh_participants:
                    raise ValueError('semantic continuation changed the declared model lineage: ' + name)
                proof = getattr(participant, 'continuation_proof', {})
                if (proof.get('checkpoint_sha256') != prior_sha or proof.get('participant') != name
                        or not proof.get('objective_transition')):
                    raise ValueError('semantic refresh requires an exact optimizer-preserving update from this shared parent: ' + name)
                changed_models.add(name)
                # Current participant weights, optimizer and FP8 were loaded
                # from its actual RL update. NEVER replace them with old state.
            else:
                with _native_modules({name: participant}):
                    restore_model_state(participant.model, saved['model_state'])
                from .native_optimizer_migration import native_optimizer_migration_option
                migration=native_optimizer_migration_option(config,name)
                participant.optimizer.load_state_dict(saved['optimizer'],native_optimizer_migration=migration)
                participant.sft_progress = saved.get('sft_progress', participant.sft_progress)
                participant.global_step = saved.get('global_step', participant.global_step)
                if participant.backend is not None:
                    participant.backend.restore_extra_state(saved['fp8'])
                from ..common.checkpoint_offload import configure_checkpoint_storage_options
                configure_checkpoint_storage_options(participant.model.port,
                    config.get('models',{}).get(name,{}),config,saved.get('training_options') or {})
        if changed_models and not config.get('adaptive_refresh'):
            raise ValueError('changed participant weights require declared adaptive_refresh work in every semantic phase')
        if changed_models:
            training.setdefault('participant_refreshes', []).append({
                'step': training['step'], 'parent_sha256': prior_sha,
                'changed_models': sorted(changed_models), 'participants': bindings,
                'required_phases': [phase['id'] for phase in config['phases']]})
        vae = SemanticStateVAE.from_specification(prior['vae_specification']).to(device)
        vae.load_state_dict(prior['vae_state'], strict=True)
        optimizer = _vae_optimizer(vae, config, prior['vae_optimizer'])
        restore_training_rng_state(training['rng'], device)
        del prior
    else:
        vae = SemanticStateVAE({name: _layout(train, participants, name, resolver) for name in participants},
            latent_width=config.get('latent_width', 128), hidden_width=config.get('hidden_width', 512)).to(device)
        optimizer = _vae_optimizer(vae, config)
    policy = CheckpointPolicy(out, every_steps=config['checkpoint_every'],
        every_seconds=config['checkpoint_seconds'], prune_best=False)
    if stop is None:
        policy.install_signal_handlers()
    else:
        policy.stop = stop
    started_step, started = training['step'], time.time()
    trainable = {name: {key for key, value in entry.model.named_parameters() if value.requires_grad}
                 for name, entry in participants.items()}
    # Existing participant values are never replaced while their optimizer is
    # alive; a new declaration requires an explicit continuation or restart.
    initial_bindings = dict(bindings)
    last_config = None
    stream, cached_row = None, None
    stream_phase,stream_index=None,None
    policy_source=None
    cursors = {}
    previous_manifest = manifest['signature']
    metrics = {}
    while not policy.signaled:
        updated = validate_config(json.loads(config_path.read_text()))
        config_sha = canonical_json_sha256_hex(updated)
        if config_sha != last_config:
            prior_kinds={phase['id']:phase['kind'] for item in training.get('config_history',[]) for phase in item['config']['phases']}
            if any(phase['id'] in prior_kinds and prior_kinds[phase['id']]!=phase['kind'] for phase in updated['phases']):
                raise ValueError('a changed semantic objective needs a new phase ID; retain earlier phase exposure')
            if (updated.get('latent_width', 128), updated.get('hidden_width', 512)) != (vae.latent_width, vae.hidden_width):
                raise ValueError('semantic latent width/layout change requires explicit adapter migration')
            if updated['pairs'] != config['pairs'] or updated['held_pairs'] != config['held_pairs']:
                train, held = _sources(updated, {'train': train.bindings, 'held': held.bindings})
                stream = None
            config = updated
            from .native_semantic_policy import PolicySource
            if config.get('policy_training'):
                policy_source=PolicySource(config['policy_training'],previous_sources=training.get('policy_source_history',[]))
                training['policy_obligations']=policy_source.synchronize(training)
            elif training.get('policy_obligations') and not training['policy_obligations']['complete']:
                raise ValueError('semantic continuation cannot drop unfinished policy episode obligations')
            else:policy_source=None
            resolver = ExternalArtifacts(config)
            cursors = {}
            training['config_history'].append({'step': training['step'], 'sha256': config_sha, 'config': config})
            last_config = config_sha
        manifest, bindings = _manifest(participants_path)
        if manifest.get('semantic_parent')!=semantic_parent:
            raise ValueError('semantic parent changed inside a running cycle; checkpoint and resume the declared parent')
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
        from . import native_semantic_refresh as refresh_schedule
        obligation_signature=canonical_json_sha256_hex({'phases':config['phases'],'source':train.binding,'participants':sorted(participants)})
        if training.get('semantic_obligation_signature')!=obligation_signature:
            training['semantic_obligations']=refresh_schedule.exposure_summary(config['phases'],train,participants,training['exposure'])
            training['semantic_obligation_signature']=obligation_signature
        refresh_cycle=refresh_schedule.initialize(training,config,manifest)
        if refresh_cycle is not None:
            work=refresh_schedule.work(config['phases'],train.entries,participants,training['exposure'],
                refresh_cycle,training.setdefault('refresh_source_cursors',{}))
        else:
            work=first_work(config['phases'],train.entries,participants,training['exposure'],cursors)
        from .native_semantic_policy import policy_cycle_complete
        policy_cycle=(policy_source.cycle(training,refresh_cycle['signature']) if policy_source is not None and refresh_cycle is not None else None)
        policy_next=policy_source.next(training) if policy_source is not None else None
        policy_due=bool(policy_next is not None and (policy_cycle is None or not policy_cycle_complete(policy_cycle))
            and (work is None or training.get('semantic_updates',0)-training.get('last_policy_semantic_update',0)
                 >=policy_source.declaration['every_semantic_updates']))
        if policy_due:
            policy_index,policy_entry=policy_next
            policy_loaded=policy_source.load(policy_entry,device=device)
            rollout,behavior,_=policy_loaded
            active=set()
            for turn in rollout['turns']:
                matches=[name for name,identity in behavior.bindings.items() if identity==(turn.get('sampling') or {}).get('policy')]
                if len(matches)!=1:raise ValueError('semantic policy source lost its exact participant identity')
                active.add(matches[0])
            if active-set(participants):raise ValueError('semantic policy source names an absent ready participant')
            origin=training.get('policy_execution_origin')
            if origin is None:
                # The exact shared parent, including current codec, is durable
                # before the first policy update. No frozen foreign model copy.
                parent_export=save_shared(policy,participants,vae,optimizer,training=training,device=device)
                origin={'checkpoint_sha256':parent_export['checkpoint']['sha256'],
                    'adapter_sha256':sha256_file_hex_cached(out/'semantic-adapter.json'),
                    'step':training['step'],'models':{name:{'native_identity':entry['native_identity']}
                        for name,entry in parent_export['models'].items()}}
                training['policy_execution_origin']=origin
            phase={'id':policy_source.declaration['id'],'kind':'policy','epochs':policy_source.declaration['epochs'],
                   'options':policy_source.declaration.get('options',{})}
            row_id=policy_entry['id'];refresh_row_index=None
        elif work is None:
            break
        else:
            phase,row_id,pairs=work[:3]
            refresh_row_index=work[3] if refresh_cycle is not None else None
        if not policy_due and (stream_phase!=phase['id'] or (refresh_row_index is not None and stream_index is not None
                and refresh_row_index<stream_index)):
            stream,cached_row=None,None
        if not policy_due:stream_phase,stream_index=phase['id'],refresh_row_index
        if not policy_due and stream is None:
            # Keep compressed JSONL sequential between neighboring updates.
            # A new epoch/new phase intentionally restarts the stream once.
            stream = train.rows()
        row = cached_row if not policy_due and cached_row and cached_row['id'] == row_id else None
        if not policy_due and row is None:
            for candidate in stream:
                if candidate['id'] == row_id:
                    row = candidate
                    break
        if not policy_due and row is None:
            row = train.get(row_id)
            stream = None
        if not policy_due:cached_row = row
        if not policy_due and canonical_json_sha256_hex(row) != train.bindings[row_id]:
            raise ValueError('semantic training row changed under its exposure cursor')
        if not policy_due:active = {name for pair in pairs for name in pair}
        selected = {name: participant for name, participant in participants.items() if name in active}
        options = {**config.get('options', {}), **phase.get('options', {})}
        for name, participant in participants.items():
            participant.optimizer.zero_grad(set_to_none=True)
            if name in active:participant.model.port.checkpoint_layers=True
            _adjust_lr(participant.optimizer, options.get('backbone_lr', config['backbone_lr']),
                       options.get('learned_row_learning_rate',config.get('models',{}).get(name,{}).get(
                           'learned_row_learning_rate',config.get('learned_row_learning_rate'))),
                       options.get('interface_row_learning_rate',config.get('models',{}).get(name,{}).get(
                           'interface_row_learning_rate',config.get('interface_row_learning_rate'))))
            from ..common.checkpoint_offload import configure_checkpoint_storage_options
            configure_checkpoint_storage_options(participant.model.port,options,
                config.get('models',{}).get(name,{}),config)
            for key, value in participant.model.named_parameters():
                value.requires_grad_(phase['kind'] != 'vae' and name in active and key in trainable[name])
        optimizer.zero_grad(set_to_none=True)
        vae.requires_grad_(phase['kind'] != 'distill')
        _adjust_lr(optimizer, options.get('vae_lr', config['vae_lr']))
        for participant in selected.values():
            if participant.backend is not None:participant.backend.reset_row_counts()
        with contextlib.ExitStack() as stack:
            for participant in selected.values():
                if participant.backend is not None:
                    stack.enter_context(participant.backend.autocast())
            if policy_due:
                from .native_semantic_policy import policy_loss
                loss,details,actual_active=policy_loss(policy_source,policy_entry,participants,vae,training,origin,
                    device=device,loaded=policy_loaded)
                if actual_active!=active:raise ValueError('semantic policy forward changed its actual participant inventory')
            else:
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
                participant.global_step += 1
                updates = training.setdefault('model_updates', {})
                updates[name] = updates.get(name, 0) + 1
        if phase['kind'] != 'distill':
            optimizer.step()
        training['step'] += 1
        backend_execution={name:participant.backend.collect_row_counts() for name,participant in selected.items()
                           if participant.backend is not None}
        for name,execution in backend_execution.items():
            total=training.setdefault('phase_backend_execution',{}).setdefault(phase['id'],{}).setdefault(name,{})
            total['updates']=total.get('updates',0)+1
            for key,value in execution.items():total[key]=total.get(key,0)+value
        counts = training['exposure'].setdefault(phase['id'], {}).setdefault(row_id, {})
        # joint_loss includes every unordered pair of selected participants.
        consumed_obligations=0
        for a, b in (() if policy_due else itertools.combinations(sorted(selected), 2)):
            key = pair_key(a, b)
            consumed_obligations+=int(counts.get(key,0)<phase['epochs'])
            counts[key] = counts.get(key, 0) + 1
        if policy_due:
            policy_source.committed(training,policy_index,policy_entry)
            training['last_policy_semantic_update']=training.get('semantic_updates',0)
            if policy_cycle is not None:policy_cycle['completed_updates']+=1
        else:training['semantic_updates']=training.get('semantic_updates',0)+1
        if refresh_cycle is not None and not policy_due:
            refresh_schedule.committed(training,refresh_cycle,phase,refresh_row_index,len(train.entries))
        obligations=training['semantic_obligations']
        obligations['remaining_pair_visits']-=consumed_obligations
        if obligations['remaining_pair_visits']<0:raise RuntimeError('semantic exposure counter underflow')
        obligations['complete']=obligations['remaining_pair_visits']==0
        metrics = {'step': training['step'], 'phase': phase['id'], 'source_row': row_id,
                   'loss': float(loss.detach()), 'gradient_norm': float(norm), 'details': details,
                   'seconds': time.time() - started,
                   'backend_execution':backend_execution,
                   'learning_rates':{name:participant.optimizer.native_learning_rates
                                     for name,participant in participants.items()}}
        training['source_bindings'] = {'train': train.bindings, 'held': held.bindings}
        training['participants_signature'] = manifest['signature']
        if training['step'] % config['eval_every'] == 0:
            held_metrics,coverage=refresh_schedule.held_evaluation(participants,vae,held,
                phase='joint' if policy_due else phase['kind'],options=options,resolver=resolver,coverage=training.get('held_coverage'),
                rows=config.get('held_eval_rows'))
            training['held_coverage']=coverage
            metrics['held']=held_metrics;metrics['held_loss']=held_metrics['loss'];metrics['held_rows']=held_metrics['rows']
        write_json_durable(out/'metrics.json', metrics)
        due_task = bool(config.get('task_eval')) and training['step'] % config.get('task_eval_every', config['eval_every']) == 0
        if policy.due(training['step']) or due_task:
            current = save_shared(policy, participants, vae, optimizer, training=training, device=device)
            if due_task:
                metrics['task_execution'] = measure_tasks(participants, current, config, out, training['step'], device, vae=vae)
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
    from . import native_semantic_refresh as refresh_schedule
    training['semantic_obligations']=refresh_schedule.exposure_summary(config['phases'],train,participants,training['exposure'])
    refresh_cycle=refresh_schedule.initialize(training,config,manifest)
    from .native_semantic_policy import policy_cycle_complete,checkpoint_progress
    policy_complete=not training.get('policy_obligations') or training['policy_obligations']['complete']
    refresh_reached=refresh_schedule.complete(refresh_cycle) and policy_cycle_complete(training.get('policy_refresh_cycle'))
    exposure_complete, missing = _exposure_complete(config['phases'], train, participants, training['exposure'])
    final_coverage_key=canonical_json_sha256_hex({'train':train.binding,'held':held.binding,'phases':config['phases'],
        **({'policy_training':config['policy_training'],'policy_progress':checkpoint_progress(training),
             'policy_exposure':training['exposure'].get(config['policy_training']['id'],{})}
           if config.get('policy_training') else {})})
    publication_boundary=refresh_cycle is None or refresh_reached
    final_due=exposure_complete and policy_complete and publication_boundary and training.get('full_held_curriculum')!=final_coverage_key
    if final_due:
        metrics['held'],training['held_coverage']=refresh_schedule.held_evaluation(participants,vae,held,
            phase='joint',options=config.get('options',{}),resolver=resolver,coverage=training.get('held_coverage'),final=True)
        training['full_held_curriculum']=final_coverage_key
    elif (refresh_reached or exposure_complete) and not metrics.get('held'):
        metrics['held'],training['held_coverage']=refresh_schedule.held_evaluation(participants,vae,held,
            phase='joint',options=config.get('options',{}),resolver=resolver,coverage=training.get('held_coverage'),rows=config.get('held_eval_rows'))
    export = save_shared(policy, participants, vae, optimizer, training=training, device=device)
    complete, missing = _exposure_complete(config['phases'], train, participants, training['exposure'])
    complete=bool(complete and publication_boundary and policy_complete)
    certificates = {}
    tasks = {}
    semantic_certificate=None
    if complete or refresh_reached:
        from ..eval.native_backend import qualify_complete_runtime
        from ..model.shared_state import RoleTokens
        tasks = measure_tasks(participants, export, config, out, training['step'], device, vae=vae)
        rng = capture_training_rng_state(device)
        try:
            for name, participant in participants.items():
                declaration=config.get('models',{}).get(name,{})
                fp8={'kind':declaration.get('fp8_training','triton-experts'if participant.config['architecture']in{'mellum','maple'}else'off'),
                    'recipe':declaration.get('fp8_recipe','delayed'),'amax_history_len':declaration.get('fp8_amax_history_len',16)}
                try:
                    certificate = qualify_complete_runtime(participant.model, participant.tokenizer,
                        RoleTokens(*participant.model.marker_ids), participant.config,
                        optimizer=participant.optimizer,backend=participant.backend,fp8_profile=fp8,device=device,
                        checkpoint_sha256=export['checkpoint']['sha256'],
                        forward_atol=config.get('forward_atol', 1e-4),
                        gradient_atol=config.get('gradient_atol', 1e-5),
                        task_execution_receipt=tasks.get(name, {}).get('path'),
                        backbone_optimizer_updates=training.get('model_updates', {}).get(name, 0))
                except Exception as error:
                    certificate={'schema':'natlang.native-runtime-certificate/1','passed':False,
                        'native_identity':native_identity(participant.model,participant.config,checkpoint_sha256=export['checkpoint']['sha256']),
                        'error':f'{type(error).__name__}: {error}'}
                path = out/('runtime-' + canonical_json_sha256_hex(name) + '.json')
                write_json_durable(path, certificate)
                certificates[name] = {'path': str(path.resolve()), 'sha256': sha256_file_hex_cached(path),
                                      'passed': certificate['passed']}
        finally:
            restore_training_rng_state(rng, device)
        if set(export['tokenizers'])==set(participants):
            from ..model.semantic_channel import SemanticChannel
            from ..eval.native_semantic_runtime import qualify_semantic_transport
            texts={}
            for row in train.rows():
                for name in participants:
                    if name in texts or name not in row['models']:continue
                    _,view=parse_pair(row,participants)[name]
                    text=view['export']['text']
                    if text:texts[name]=text
                if set(texts)==set(participants):break
            channel=SemanticChannel(vae,{name:entry['native_identity'] for name,entry in export['models'].items()},
                {name:entry.tokenizer for name,entry in participants.items()},
                adapter_sha256=sha256_file_hex_cached(out/'semantic-adapter.json'))
            with _native_modules(participants):
                transport=qualify_semantic_transport(participants,channel,texts=texts,
                    rtol=config.get('semantic_runtime_rtol',1e-5),atol=config.get('forward_atol',1e-4))
            path=out/'semantic-runtime-certificate.json'
            write_json_durable(path,transport)
            semantic_certificate={'path':str(path.resolve()),'sha256':sha256_file_hex_cached(path),'passed':transport['passed']}
        numerics_ready=all(value['passed'] for value in certificates.values())
        complete = bool(exposure_complete and policy_complete and numerics_ready and (metrics.get('held') or {}).get('scope')=='full')
    configuration = out/'completed-curriculum.json'
    write_json_durable(configuration, config)
    current_checks=bool(certificates and all(value['passed'] for value in certificates.values())
        and set(export['tokenizers'])==set(participants) and set(tasks)==set(participants)
        and all(value['passed'] for value in tasks.values())
        and semantic_certificate is not None and semantic_certificate['passed'])
    report = {'schema': 'natlang.native-shared-stage-completion/1', 'completed': complete,
        'exposure_complete':exposure_complete,'refresh_ready':bool(refresh_reached and current_checks),
        'refresh_progress':refresh_cycle,'semantic_obligations':training['semantic_obligations'],
        'participants_signature': manifest['signature'], 'step': training['step'],
        'curriculum_signature': canonical_json_sha256_hex(config),
        'curriculum': {'path': str(configuration.resolve()), 'sha256': sha256_file_hex_cached(configuration)},
        'checkpoint': export['checkpoint'], 'adapter': {'path': str((out/'semantic-adapter.json').resolve()),
            'sha256': sha256_file_hex_cached(out/'semantic-adapter.json')},
        'source_bindings': {'train': train.binding, 'held': held.binding},
        'phases': config['phases'], 'exposure': training['exposure'], 'missing_exposure': missing,
        'runtime_certificates': certificates,
        'semantic_runtime_certificate':semantic_certificate,
        'task_execution_receipts': tasks,
        'production_ready':bool((complete or refresh_reached) and current_checks),
        'task_lift': 'not-claimed', 'metrics': metrics}
    if training.get('policy_training_phase'):
        report['policy_training_progress']=checkpoint_progress(training)
    write_json_durable(out/'shared-stage-completion.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--participants', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--refresh-participants', action='store_true')
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
                 max_updates=args.max_updates, device=args.device, refresh_participants=args.refresh_participants)
    print(json.dumps({'completed': report['completed'], 'step': report['step'],
                      'checkpoint': report['checkpoint']}), flush=True)
    return 0 if report['production_ready'] else 75


if __name__ == '__main__':
    raise SystemExit(main())
