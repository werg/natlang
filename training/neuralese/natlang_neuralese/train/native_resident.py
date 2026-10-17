"""Prepared, data-only adoption for the existing native SFT runner.

current.json is the sole commit record. Other files are recoverable projections;
checkpoint exposure remains authoritative for weights and completed updates.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex as digest, sha256_file_hex_cached as sha, sha256_file_hex
from ..common.jsonio import write_json_durable as write_json

SCHEMA = 'natlang.native-resident-adoption/1'


def binding(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def bound(value):
    path = Path(value['path']).resolve(strict=True)
    if sha(path) != value['sha256']:
        raise ValueError('resident artifact changed: ' + str(path))
    return path


def code_inventory(package):
    package = Path(package)
    return {str(p.relative_to(package)): sha256_file_hex(p) for p in sorted(package.rglob('*.py'))}


@contextmanager
def locked(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def current(directory):
    path = Path(directory) / 'current.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('schema') != SCHEMA or digest(value['payload']) != value['generation']:
        raise ValueError('resident commit is corrupt')
    return value


def repair(directory, state):
    """Idempotently project the already committed generation, never invent one."""
    directory = Path(directory)
    payload = state['payload']
    write_json(directory.parent / 'recipe-plan.json', payload['plan'])
    write_json(directory / 'active-binding.json', {'generation': state['generation'],
        'recipe': payload['recipe'], 'stage_id': payload['stage_id'],
        'train': payload['train'], 'held': payload['held'], 'adopted_at_step': state['at_global_step'],
        'checkpoint_required_for_weights': True})


def recover(directory):
    with locked(directory):
        state = current(directory)
        if state:
            # Verify the recipe before repairing anything a runner will consume.
            bound({'path': state['payload']['recipe']['path'],
                   'sha256': state['payload']['recipe']['file_sha256']})
            from .recipe import recipe_identity_sha256, load_recipe
            if (recipe_identity_sha256(state['payload']['recipe']['path']) != state['payload']['recipe']['sha256']
                    or load_recipe(Path(state['payload']['recipe']['path'])) != state['payload']['plan']['recipe']):
                raise ValueError('resident committed recipe policy/inheritance changed')
            repair(directory, state)
        return state


def _source_witness(source, package=None):
    """Bind the existing validation-cache entries; no second graph parser."""
    from ..common import row_index
    if source.kind == 'manifest':
        return [w for child in source.children for w in _source_witness(child, package)]
    package = Path(package or Path(__file__).resolve().parents[1])
    names = ('data/native_sources.py', 'data/native_graph_schema.py', 'model/native_graph_plan.py',
             'data/native_pair_schema.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')
    key = {'source_sha256': source.file_sha256, 'tokenizer': source.tokenizer_fingerprint,
           'split': source.required_split, 'policy_sha256': digest({n: sha256_file_hex(package/n) for n in names}),
           'kind': source.kind, 'pair_model': source.pair_model, 'role_ids': list(source.role_ids)}
    path = row_index.entry_path('native-graphs', digest(key))
    value = row_index.load(path)
    if not value or value.get('key') != key or value.get('metadata_sha256') != digest(source.entries):
        raise ValueError('resident preparation requires the exact completed source index: ' + str(source.path))
    return [{'cache': binding(path), 'key': key, 'metadata_sha256': value['metadata_sha256']}]


def _check_indexes(witnesses):
    from ..common import row_index
    for witness in witnesses:
        cache = bound(witness['cache'])
        if cache != row_index.entry_path('native-graphs', digest(witness['key'])).resolve():
            raise ValueError('resident cache must be staged in the learner row-index cache')
        value = row_index.load(cache)
        if (not value or value.get('key') != witness['key']
                or value.get('metadata_sha256') != witness['metadata_sha256']
                or digest(value.get('entries')) != witness['metadata_sha256']):
            raise ValueError('resident warmed source index changed')


def _contract(plan, proposed, stage_id):
    """One stage may append active data and/or mandatory pending declarations."""
    from .recipe import _data_continuation, _pending_source_continuation
    old_inputs, new_inputs = plan['stage_inputs'], proposed['stage_inputs']
    if plan['code'] != proposed['code'] or plan.get('device') != proposed.get('device'):
        raise ValueError('resident append cannot change runtime or device')
    if set(old_inputs) != set(new_inputs):
        raise ValueError('resident append cannot change stage membership')
    for sid in old_inputs:
        if sid != stage_id and old_inputs[sid] != new_inputs[sid]:
            raise ValueError('resident append cannot change another stage')
    old, new = old_inputs[stage_id], new_inputs[stage_id]
    if old.get('foundation_native_graphs') != new.get('foundation_native_graphs'):
        raise ValueError('resident append cannot change fixed foundation')
    if old.get('withdrawn_native_graphs') != new.get('withdrawn_native_graphs'):
        raise ValueError('resident append cannot withdraw graphs')
    pending_role = 'pending_native_sources'
    pending_changed = old.get(pending_role) != new.get(pending_role)
    if pending_changed and (not isinstance(new.get(pending_role), dict) or
            not _pending_source_continuation(old, {**old, pending_role: new[pending_role]})):
        raise ValueError('resident pending declarations must preserve all prior rows, quotas and dependencies')
    without_pending = lambda values: {k:v for k,v in values.items() if k != pending_role}
    active_changed = without_pending(old) != without_pending(new)
    if active_changed and not _data_continuation(without_pending(old), without_pending(new)):
        raise ValueError('resident append requires the exact training-source predecessor')
    if not active_changed and not pending_changed:
        raise ValueError('resident request must add active data or mandatory pending work')
    recipes = []
    for recipe in (plan['recipe'], proposed['recipe']):
        value = copy.deepcopy(recipe)
        value.pop('input_bindings', None)
        for stage in value['stages']:
            if stage['id'] == stage_id:
                if stage['kind'] != 'native_state_sft':
                    raise ValueError('resident append supports supervised native graphs only')
                for key in ('native_graphs', 'continued_native_graphs', 'held_native_graphs', 'continued_held_native_graphs', pending_role):
                    stage.get('inputs', {}).pop(key, None)
        recipes.append(value)
    if recipes[0] != recipes[1]:
        raise ValueError('resident append cannot change parameters, runtime, objective or policy')


def pending_declarations(plan, stage_id, tokenizer_fingerprint):
    """Authenticate the same declaration consumed by checkpoint progress/readiness."""
    from .native_pending_sources import load_pending_sources
    reference = plan['stage_inputs'][stage_id].get('pending_native_sources')
    return ([load_pending_sources(bound(reference), tokenizer_fingerprint=tokenizer_fingerprint)]
            if reference else [])


def validate_source_obligations(plan, stage_id, train, declarations, withdrawal_history=()):
    """Reject withdrawn active/pending targets before the sole commit point."""
    from .native_continuation import _retain_pending_sources, reject_withdrawn_targets
    history = copy.deepcopy(list(withdrawal_history))
    reference = plan['stage_inputs'][stage_id].get('withdrawn_native_graphs')
    if reference:
        proof = json.loads(bound(reference).read_text())
        if proof.get('schema') != 'natlang.native-graph-withdrawal/1':
            raise ValueError('resident withdrawal proof schema differs')
        history.append({'rows': proof['rows']})
    frame = {'source_row_bindings':train.bindings, 'source_row_order':train.ordered_ids(),
        'exposure_counts':{}, 'options':{'withdrawal_history':history,
        '_pending_native_source_declarations':declarations}}
    reject_withdrawn_targets(frame, train.bindings)
    return _retain_pending_sources(frame, {})


def geometry_capacity(value, *, identity, code):
    certificate = json.loads(bound(value).read_text())
    if (certificate.get('schema') != 'natlang.native-resident-geometry-qualification/1'
            or certificate.get('passed') is not True or certificate.get('identity') != identity
            or certificate.get('runtime_code_sha256') != digest(code)):
        raise ValueError('resident geometry qualification differs from actual model/runtime/packing')
    bound(certificate['evidence'])
    limits = certificate.get('limits') or {}
    if set(limits) != {'forward_tokens', 'kv_slots'} or any(type(v) is not int or v < 1 for v in limits.values()):
        raise ValueError('resident geometry qualification needs positive measured forward/KV-slot limits')
    return limits


def new_geometry(source, first):
    maximum = {'forward_tokens': 0, 'kv_slots': 0}
    for example in source.iter_indices(range(first, len(source))):
        maximum['forward_tokens'] = max(maximum['forward_tokens'], example.plan.token_count)
        maximum['kv_slots'] = max(maximum['kv_slots'], len(example.plan.slot_sources))
    return maximum



def geometry_from_evidence(source, first, evidence, *, package, source_preparation=None):
    """Reuse a bound completed CPU parse, conservatively covering a new suffix.

    Legacy tiered prewarm receipts parsed TRAIN only. Their held indexes are
    admission evidence, never held geometry. A prepared resident request covers
    both splits, but its union maximum is deliberately retained for each split.
    """
    report = json.loads(bound(evidence).read_text())
    split = source.required_split
    if report.get('schema') == SCHEMA:
        payload = report['payload']
        if digest(payload) != report.get('generation'):
            raise ValueError('geometry evidence resident request is corrupt')
        field, prior = ('train', 'prior_train') if split == 'train' else ('held', 'prior_held')
        source_binding, prior_binding = payload[field], payload[prior]
        witnesses, maximum = payload['indexes'], payload['geometry']
        expected_rows = payload['rows' if split == 'train' else 'held_rows']
        if payload.get('delta_validation') is not None:
            raise ValueError('delta preparation is not a full-source geometry evidence receipt')
        expected_entries = payload['entries_sha256' if split == 'train' else 'held_entries_sha256']
        start = None
    else:
        if report.get('passed') is not True or split != 'train':
            raise ValueError('legacy geometry evidence covers completed training geometry only')
        source_binding, prior_binding = report['source'], report['prior_train']
        witnesses = report['indexes']
        start = report['old_train_rows']
        expected_rows = start + report['new_train_rows']
        expected_entries = None
        maximum = {key: report['maxima'][key][key] for key in ('forward_tokens', 'kv_slots')}
    if source_preparation is None:
        from .native_source_preparation import NativeSourcePreparation
        source_preparation = NativeSourcePreparation(source.tokenizer_fingerprint, source.role_ids)
        source_preparation._remember(source)
    covered = source_preparation.load(source_binding, split=split)
    prior_source = source_preparation.load(prior_binding, split=split)
    if start is None:
        start = len(prior_source)
    if (type(start) is not int or start != len(prior_source) or len(covered) != expected_rows
            or covered.entries[:start] != prior_source.entries
            or expected_entries is not None and digest(covered.entries) != expected_entries):
        raise ValueError('geometry evidence source prefix/count/metadata differs')
    _check_indexes(witnesses)
    recorded = {digest(w) for w in witnesses}
    if any(digest(w) not in recorded for w in _source_witness(covered, package)):
        raise ValueError('geometry evidence lacks exact source/tokenizer/split/policy indexes')
    measured = {entry['id']: entry for entry in covered.entries[start:]}
    requested = source.entries[first:]
    if any(measured.get(entry['id']) != entry for entry in requested):
        raise ValueError('geometry evidence does not cover every exact appended row')
    if set(maximum) != {'forward_tokens', 'kv_slots'} or any(type(v) is not int or v < 1 for v in maximum.values()):
        raise ValueError('geometry evidence needs positive physical and total-slot maxima')
    if maximum['forward_tokens'] < max((e['forward_tokens'] for e in covered.entries[start:]), default=0):
        raise ValueError('geometry evidence understates indexed physical geometry')
    if report.get('schema') != SCHEMA:
        descriptors = report['maxima']
        indices = sorted({v['index'] for v in descriptors.values()})
        if any(type(index) is not int or not 0 <= index < len(covered) for index in indices):
            raise ValueError('geometry maximum must name an exact covered-source row')
        actual = {}
        for index, example in zip(indices, covered.iter_indices(indices)):
            entry = covered.entries[index]
            actual[index] = {'index': index, 'id': example.id, 'binding': entry['binding'],
                'forward_tokens': example.plan.token_count, 'kv_slots': len(example.plan.slot_sources)}
        if any(actual.get(value['index']) != value for value in descriptors.values()):
            raise ValueError('geometry evidence maximum row identity/physical/total slots differ')
    return maximum


def process_owner():
    return {'pid': os.getpid(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'start_ticks': Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19]}


def owner_alive(owner):
    try:
        return (owner['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and owner['start_ticks'] == Path('/proc', str(owner['pid']), 'stat').read_text().split(') ', 1)[1].split()[19])
    except (KeyError, OSError):
        return False


def retire(directory, checkpoint, terminal):
    """After the optimizer owner drains, allow an ordinary runtime handoff.

    A normal checkpoint must already contain this adoption. No special save is
    requested here; stale checkpoints cannot retire the recovery authority.
    """
    import torch
    directory = Path(directory)
    with locked(directory):
        state = current(directory)
        if state is None:
            return
        initial = json.loads((directory/'initial.json').read_text())
        if owner_alive(initial.get('owner') or {}):
            raise ValueError('resident retirement requires the actual optimizer process to exit')
        exit_record = json.loads(bound(terminal).read_text())
        if exit_record.get('returncode') != 0 or exit_record.get('at', 0) < state['at']:
            raise ValueError('resident retirement requires the drained optimizer owner')
        saved = torch.load(bound(checkpoint), map_location='cpu', weights_only=False, mmap=True)
        from ..model.native_model import CHECKPOINT_SCHEMA
        from .native_state_sft import TRAINING_SCHEMA
        training = saved.get('training') or {}
        if (saved.get('schema') != CHECKPOINT_SCHEMA or training.get('schema') != TRAINING_SCHEMA
                or not training.get('optimizer') or not training.get('rng')
                or training.get('global_step', -1) < state['at_global_step']):
            raise ValueError('resident retirement requires actual full optimizer/RNG checkpoint state')
        if ((training.get('options', {}).get('resident_adoption') or {}).get('generation') != state['generation']
                or (training.get('source_receipts') or [{}])[0].get('sha256') != state['payload']['train']['sha256']):
            raise ValueError('resident retirement requires a checkpoint of the adopted generation')
        # The published request remains after adoption. Clear only this exact
        # retired generation under the same writer lock; preserve a newer request.
        pointer = directory/'pending.json'
        clear_pending = False
        if pointer.exists():
            request = json.loads(pointer.read_text())
            pending = json.loads(bound(request).read_text())
            if pending.get('schema') != SCHEMA or digest(pending.get('payload')) != pending.get('generation'):
                raise ValueError('resident pending request is corrupt before retirement')
            clear_pending = (pending['generation'] == state['generation']
                             and pending['payload'] == state['payload'])
        history = directory/'retired'; history.mkdir(exist_ok=True)
        write_json(history/(state['generation']+'.json'), {**state, 'checkpoint': checkpoint, 'terminal': terminal})
        if clear_pending:
            pointer.unlink()
        (directory/'current.json').unlink()
        descriptor = os.open(directory, os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)


def _delta_payload(directory, previous, plan, proposed, stage_id, recipe_path,
                   tokenizer_fingerprint, role_ids, package, delta_sources):
    """Validate only explicit new leaves; the learner validates the full union."""
    from .recipe import native_recipe_binding
    if not isinstance(delta_sources, dict) or set(delta_sources) != {'train'}:
        raise ValueError('resident delta preparation requires an explicit train delta')
    prior = plan['stage_inputs'][stage_id]
    actual = proposed['stage_inputs'][stage_id]
    if actual['held_native_graphs'] != prior['held_native_graphs']:
        raise ValueError('resident delta preparation requires unchanged held source')
    train_ref = binding(bound(actual['native_graphs']))
    prior_ref = binding(bound(prior['native_graphs']))
    held_ref = binding(bound(prior['held_native_graphs']))
    delta_ref = binding(bound(delta_sources['train']))
    wrapper = json.loads(bound(train_ref).read_text())
    expected = [{'kind': 'native-graphs', **prior_ref}, {'kind': 'native-graphs', **delta_ref}]
    if wrapper != {'schema': 'natlang.native-graph-sources/1', 'sources': expected}:
        raise ValueError('resident delta source must be exact prior source followed by declared delta')
    initial = json.loads((directory/'initial.json').read_text())
    prior_counts = previous['payload'] if previous else initial
    rows, held_rows = prior_counts.get('rows'), prior_counts.get('held_rows')
    if type(rows) is not int or rows < 1 or type(held_rows) is not int or held_rows < 1:
        raise ValueError('resident delta preparation lacks authenticated prior row counts')
    from .native_source_preparation import NativeSourcePreparation
    preparation = NativeSourcePreparation(tokenizer_fingerprint, role_ids)
    delta = preparation.load(delta_ref, split='train')
    capacity = initial.get('geometry_qualification')
    if not capacity:
        raise ValueError('resident delta append needs explicit geometry qualification')
    limits = geometry_capacity(capacity, identity=initial['geometry_identity'], code=plan['code'])
    geometry = new_geometry(delta, 0)
    if any(geometry[key] > limits[key] for key in limits):
        raise ValueError('resident delta exceeds qualified whole-graph geometry')
    pending_declarations(proposed, stage_id, tokenizer_fingerprint)
    return {'predecessor': previous['generation'] if previous else digest(plan),
        'stage_id': stage_id, 'plan': proposed,
        'recipe': native_recipe_binding(recipe_path, plan['recipe_sha256']),
        'train': train_ref, 'prior_train': prior_ref, 'held': held_ref, 'prior_held': held_ref,
        'prior_recipe': previous['payload']['prior_recipe'] if previous else initial['recipe'],
        'tokenizer_fingerprint': tokenizer_fingerprint, 'role_ids': list(role_ids),
        'indexes': _source_witness(delta, package), 'rows': rows + len(delta), 'held_rows': held_rows,
        'geometry_qualification': capacity, 'geometry_identity': initial['geometry_identity'],
        'geometry': geometry,
        'delta_validation': {'schema': 'natlang.native-resident-delta-validation/1',
            'source': delta_ref, 'rows': len(delta), 'entries_sha256': digest(
                [{**entry, 'training_priority':entry.get('training_priority',0)} for entry in delta.entries]),
            'prior_rows': rows, 'prior_held_rows': held_rows, 'prior_plan_sha256': digest(plan)}}


def prepare(directory, recipe_path, stage_id, *, tokenizer_fingerprint, role_ids,
            runtime_package=None, geometry_evidence=None, publish=True, delta_sources=None):
    """CPU preparation; callers decide admission and provide an immutable recipe."""
    if type(publish) is not bool:
        raise ValueError('resident preparation publish must be an explicit boolean')
    from .recipe import load_recipe, resolve_stage_inputs, _binding_files, native_recipe_binding
    from .native_state_sft import append_only_rows
    directory = Path(directory).resolve()
    with locked(directory):
        previous = current(directory)
        plan = (previous['payload']['plan'] if previous else
                json.loads((directory.parent / 'recipe-plan.json').read_text()))
        predecessor = previous['generation'] if previous else digest(plan)
    proposed = copy.deepcopy(plan)
    proposed['recipe'] = load_recipe(Path(recipe_path))
    proposed['stage_inputs'] = {stage['id']: ({} if stage.get('admitted') is False else
        resolve_stage_inputs(proposed['recipe'], stage,
                             {'records': None, 'pieces': None, 'text_data': None}))
        for stage in proposed['recipe']['stages']}
    _contract(plan, proposed, stage_id)
    package = Path(runtime_package or Path(__file__).resolve().parents[1])
    if code_inventory(package) != plan['code']:
        raise ValueError('resident preparation runtime differs from the running recipe')
    inputs = {}
    for roles in proposed['stage_inputs'].values():
        for value in roles.values():
            for dep in _binding_files(value):
                inputs[dep['path']] = dep['sha256']
    # Preserve non-role inputs, and remove superseded role dependencies.
    old_paths = {dep['path'] for roles in plan['stage_inputs'].values() for value in roles.values()
                 for dep in _binding_files(value)}
    proposed['inputs'] = {**{p: h for p, h in plan['inputs'].items() if p not in old_paths}, **inputs}
    if delta_sources is not None:
        if geometry_evidence:
            raise ValueError('resident delta preparation compiles new geometry directly, not full-source evidence')
        payload = _delta_payload(directory, previous, plan, proposed, stage_id, recipe_path,
            tokenizer_fingerprint, role_ids, package, delta_sources)
    else:
        from .native_source_preparation import NativeSourcePreparation
        source_preparation = NativeSourcePreparation(tokenizer_fingerprint, role_ids)
        # New manifests expose already validated old subtrees; load those roots once.
        train = source_preparation.load(proposed['stage_inputs'][stage_id]['native_graphs'], split='train')
        held = source_preparation.load(proposed['stage_inputs'][stage_id]['held_native_graphs'], split='test')
        old = source_preparation.load(plan['stage_inputs'][stage_id]['native_graphs'], split='train')
        old_held = source_preparation.load(plan['stage_inputs'][stage_id]['held_native_graphs'], split='test')
        append_only_rows(old.bindings, train.bindings, label='resident training-source')
        append_only_rows(old_held.bindings, held.bindings, label='resident held-source')
        if (train.entries[:len(old.entries)] != old.entries or held.entries[:len(old_held.entries)] != old_held.entries
                or len(train) < len(old) or len(held) < len(old_held)
                or (len(train) == len(old) and len(held) == len(old_held)
                    and plan['stage_inputs'][stage_id].get('pending_native_sources') ==
                        proposed['stage_inputs'][stage_id].get('pending_native_sources'))):
            raise ValueError('resident append must preserve every prior row, order and scheduling priority')
        if train.groups & held.groups:
            raise ValueError('resident train/held groups overlap')
        declarations = pending_declarations(proposed, stage_id, tokenizer_fingerprint)
        # Reject conflicts before publishing, with no earned visits or model action.
        initial = json.loads((directory/'initial.json').read_text())
        validate_source_obligations(proposed, stage_id, train, declarations,
            initial.get('withdrawal_history') or ())
        capacity_binding = initial.get('geometry_qualification')
        if not capacity_binding:
            raise ValueError('resident append needs an explicit current geometry qualification')
        limits = geometry_capacity(capacity_binding, identity=initial['geometry_identity'], code=plan['code'])
        geometry_evidence = geometry_evidence or {}
        if set(geometry_evidence) - {'train', 'test'}:
            raise ValueError('geometry evidence must declare train/test scope explicitly')
        def measured(source, first):
            if len(source) == first:
                return {'forward_tokens': 0, 'kv_slots': 0}
            evidence = geometry_evidence.get(source.required_split)
            return (geometry_from_evidence(source, first, evidence, package=package, source_preparation=source_preparation) if evidence
                    else new_geometry(source, first))
        train_geometry = measured(train, len(old))
        held_geometry = measured(held, len(old_held))
        geometry = {key: max(train_geometry[key], held_geometry[key]) for key in train_geometry}
        if any(geometry[key] > limits[key] for key in limits):
            raise ValueError('resident append exceeds qualified whole-graph geometry; retain pending until memory handling is qualified')
        payload = {'predecessor': predecessor, 'stage_id': stage_id, 'plan': proposed,
            'recipe': native_recipe_binding(recipe_path, plan['recipe_sha256']),
            'train': binding(train.path), 'prior_train': binding(old.path),
            'held': binding(held.path), 'prior_held': binding(old_held.path),
            'prior_recipe': (previous['payload']['prior_recipe'] if previous else
                json.loads((directory/'initial.json').read_text())['recipe']),
            'tokenizer_fingerprint': tokenizer_fingerprint, 'role_ids': list(role_ids),
            'indexes': _source_witness(train, package) + _source_witness(held, package), 'rows': len(train),
            'held_rows': len(held), 'held_entries_sha256': digest(held.entries),
            'entries_sha256': digest(train.entries), 'geometry_qualification': capacity_binding,
            'geometry': geometry, 'geometry_identity': initial['geometry_identity']}
        if geometry_evidence:
            payload['geometry_evidence'] = geometry_evidence
    proposed.setdefault('resident_adoptions', []).append({'predecessor': predecessor,
        'recipe': payload['recipe'], 'train': payload['train'], 'held': payload['held'], 'stage_id': stage_id})
    request_id = digest(payload)
    with locked(directory):
        actual = current(directory)
        actual_predecessor = actual['generation'] if actual else digest(json.loads(
            (directory.parent/'recipe-plan.json').read_text()))
        if actual_predecessor != predecessor:
            raise ValueError('resident generation advanced during CPU preparation; prepare against current data')
        requests = directory / 'requests'
        requests.mkdir(exist_ok=True)
        path = requests / (request_id + '.json')
        if not path.exists():
            write_json(path, {'schema': SCHEMA, 'payload': payload, 'generation': request_id})
        if publish:
            write_json(directory / 'pending.json', binding(path))
        return binding(path)


class ResidentAdoption:
    def __init__(self, directory, *, stage_id, recipe_binding, train, held, package=None, geometry_qualification=None, geometry_identity=None, withdrawal_history=()):
        self.directory = Path(directory)
        self.stage_id, self.recipe_binding = stage_id, recipe_binding
        self.train, self.held = train, held
        self.withdrawal_history = copy.deepcopy(list(withdrawal_history))
        self.package = Path(package or Path(__file__).resolve().parents[1])
        self.state = current(directory)
        self.plan = self.state['payload']['plan'] if self.state else json.loads(
            (self.directory.parent/'recipe-plan.json').read_text())
        self.generation = self.state['generation'] if self.state else digest(self.plan)
        self.rejected = set()
        with locked(self.directory):
            initial = self.directory/'initial.json'
            prior = json.loads(initial.read_text()) if initial.exists() else None
            if prior and prior.get('owner') != process_owner() and owner_alive(prior.get('owner') or {}):
                raise ValueError('resident source state already has a live optimizer owner')
            if prior is None or (self.state is None and prior.get('plan_sha256') != digest(self.plan)) or prior.get('stage_id') != stage_id:
                write_json(initial, {'recipe': recipe_binding, 'stage_id': stage_id,
                    'geometry_qualification': binding(geometry_qualification) if geometry_qualification else None,
                    'geometry_identity': geometry_identity, 'plan_sha256': digest(self.plan),
                    'owner': process_owner(), 'withdrawal_history':self.withdrawal_history,
                    'rows':len(train), 'held_rows':len(held)})
            else:
                if any(item not in self.withdrawal_history for item in prior.get('withdrawal_history') or []):
                    raise ValueError('resident restart lost actual carried withdrawal history')
                write_json(initial, {**prior, 'owner': process_owner(), 'withdrawal_history':self.withdrawal_history})
        self.initial = json.loads((self.directory/'initial.json').read_text())
        if self.initial.get('geometry_identity') != geometry_identity:
            raise ValueError('resident geometry identity changed; use drained runtime handoff')
        actual = self.plan['stage_inputs'][stage_id]
        if (actual['native_graphs']['sha256'] != train.file_sha256
                or actual['held_native_graphs']['sha256'] != held.file_sha256):
            raise ValueError('resident plan differs from actual loaded sources')
        self.pending_declarations = pending_declarations(self.plan, stage_id, train.tokenizer_fingerprint)

    def poll(self, global_step):
        """Return validated new data or None. Rejections do not stop SGD."""
        pointer = self.directory / 'pending.json'
        if not pointer.exists():
            return None
        request = None
        try:
            request = json.loads(pointer.read_text())
            request_id = request['sha256']
            if request_id in self.rejected:
                return None
            value = json.loads(bound(request).read_text())
            payload = value['payload']
            if value.get('schema') != SCHEMA or digest(payload) != value['generation']:
                raise ValueError('resident request is corrupt')
            if value['generation'] == self.generation:
                return None
            if payload['predecessor'] != self.generation or payload['stage_id'] != self.stage_id:
                raise ValueError('resident request targets a different current generation/stage')
            _contract(self.plan, payload['plan'], self.stage_id)
            if binding(payload['train']['path']) != {k: payload['plan']['stage_inputs'][self.stage_id]['native_graphs'][k] for k in ('path','sha256')}:
                raise ValueError('resident payload source differs from its recipe plan')
            if binding(payload['held']['path']) != {k: payload['plan']['stage_inputs'][self.stage_id]['held_native_graphs'][k] for k in ('path','sha256')}:
                raise ValueError('resident held source differs from its recipe plan')
            if code_inventory(self.package) != self.plan['code']:
                raise ValueError('resident running runtime changed')
            if bound(payload['prior_held']) != self.held.path.resolve():
                raise ValueError('resident predecessor differs from loaded held source')
            if bound(payload['prior_train']) != self.train.path.resolve():
                raise ValueError('resident predecessor differs from loaded training source')
            recipe = payload['recipe']
            bound({'path': recipe['path'], 'sha256': recipe['file_sha256']})
            from .recipe import recipe_identity_sha256, load_recipe, resolve_stage_inputs
            if load_recipe(Path(recipe['path'])) != payload['plan']['recipe']:
                raise ValueError('resident recipe differs from prepared plan')
            resolved = {stage['id']: ({} if stage.get('admitted') is False else
                resolve_stage_inputs(payload['plan']['recipe'], stage,
                    {'records': None, 'pieces': None, 'text_data': None}))
                for stage in payload['plan']['recipe']['stages']}
            if resolved != payload['plan']['stage_inputs']:
                raise ValueError('resident resolved recipe inputs changed')
            if recipe_identity_sha256(recipe['path']) != recipe['sha256']:
                raise ValueError('resident recipe inheritance changed')
            if (payload.get('geometry_qualification') != self.initial.get('geometry_qualification')
                    or payload.get('geometry_identity') != self.initial.get('geometry_identity')):
                raise ValueError('resident geometry qualification changed')
            limits = geometry_capacity(payload['geometry_qualification'],
                identity=self.initial['geometry_identity'], code=self.plan['code'])
            if (set(payload['geometry']) != set(limits) or any(type(v) is not int or v < 0 for v in payload['geometry'].values())
                    or any(payload['geometry'][key] > limits[key] for key in limits)):
                raise ValueError('resident append exceeds qualified geometry')
            _check_indexes(payload['indexes'])
            from .native_source_preparation import NativeSourcePreparation
            source_preparation = NativeSourcePreparation(payload['tokenizer_fingerprint'], payload['role_ids'])
            source_preparation.retain(self.train)
            source_preparation.retain(self.held)
            train = source_preparation.load(payload['train'], split='train')
            held = source_preparation.load(payload['held'], split='test')
            full_validation = {'entries_sha256': digest(train.entries),
                'held_entries_sha256': digest(held.entries)}
            delta = payload.get('delta_validation')
            if delta is not None:
                if (delta.get('schema') != 'natlang.native-resident-delta-validation/1'
                        or delta['prior_plan_sha256'] != digest(self.plan)
                        or delta['prior_rows'] != len(self.train)
                        or delta['prior_held_rows'] != len(self.held)
                        or payload['held'] != payload['prior_held']):
                    raise ValueError('resident delta predecessor contract changed')
                suffix = source_preparation.load(delta['source'], split='train')
                suffix_entries = [{**entry, 'training_priority':entry.get('training_priority',0)} for entry in suffix.entries]
                if (len(suffix) != delta['rows'] or digest(suffix_entries) != delta['entries_sha256']
                        or train.entries[len(self.train):] != suffix_entries):
                    raise ValueError('resident delta suffix differs from prepared metadata')
            if (held.entries[:len(self.held.entries)] != self.held.entries or len(held) < len(self.held)
                    or len(held) != payload['held_rows'] or (delta is None and full_validation['held_entries_sha256'] != payload['held_entries_sha256'])):
                raise ValueError('resident held rows are not an exact class-preserving append')
            if (train.tokenizer_fingerprint != self.train.tokenizer_fingerprint
                    or train.role_ids != self.train.role_ids
                    or train.entries[:len(self.train.entries)] != self.train.entries
                    or len(train) < len(self.train) or train.groups & held.groups
                    or (len(train) == len(self.train) and len(held) == len(self.held)
                        and self.plan['stage_inputs'][self.stage_id].get('pending_native_sources') ==
                            payload['plan']['stage_inputs'][self.stage_id].get('pending_native_sources'))
                    or len(train) != payload['rows'] or (delta is None and full_validation['entries_sha256'] != payload['entries_sha256'])):
                raise ValueError('resident rows are not an exact protected append')
            declarations = pending_declarations(payload['plan'], self.stage_id, train.tokenizer_fingerprint)
            validate_source_obligations(payload['plan'], self.stage_id, train, declarations,
                self.withdrawal_history)
            with locked(self.directory):
                actual = current(self.directory)
                generation = actual['generation'] if actual else digest(json.loads(
                    (self.directory.parent/'recipe-plan.json').read_text()))
                if generation != self.generation:
                    raise ValueError('resident generation changed concurrently')
                committed = {**value, 'at_global_step': global_step, 'at': time.time(),
                    'validation': full_validation,
                    'history': [*((actual or {}).get('history') or []),
                                {'generation': value['generation'], 'at_global_step': global_step,
                                 'recipe': recipe, 'train': payload['train'], 'held': payload['held']} ]}
                write_json(self.directory/'current.json', committed)  # sole commit point
                # A crash after this point is recovered from current.json, not rejected.
                repair(self.directory, committed)
                self.state, self.plan, self.generation = committed, payload['plan'], value['generation']
                self.train, self.held, self.recipe_binding = train, held, recipe
                self.pending_declarations = declarations
                return train, held
        except (ValueError, KeyError, TypeError, OSError) as error:
            # A post-commit projection error is fatal/recoverable, never continue
            # SGD under stale source metadata after publishing a new generation.
            committed = current(self.directory)
            if committed and committed['generation'] != self.generation:
                raise RuntimeError('resident commit needs recovery before training continues') from error
            identity = (request or {}).get('sha256', digest(str(error)))
            self.rejected.add(identity)
            errors = self.directory/'rejections'
            errors.mkdir(parents=True, exist_ok=True)
            write_json(errors/(identity+'.json'), {'error': str(error), 'at': time.time(),
                'at_global_step': global_step, 'generation': self.generation, 'request': request})
            return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--retire-checkpoint', type=Path)
    parser.add_argument('--trainer-exit', type=Path)
    parser.add_argument('--stage-id')
    parser.add_argument('--tokenizer-fingerprint')
    parser.add_argument('--role-ids', nargs=4, type=int)
    parser.add_argument('--delta-train-source', type=Path,
                        help='validate only this exact appended source; learner verifies complete prefix and union')
    parser.add_argument('--no-publish', action='store_true',
                        help='validate and write an immutable request without publishing it to the learner')
    args = parser.parse_args(argv)
    if args.retire_checkpoint:
        if not args.trainer_exit:
            parser.error('retirement requires the exact trainer exit receipt')
        retire(args.directory, binding(args.retire_checkpoint), binding(args.trainer_exit))
        return
    if not all((args.recipe, args.stage_id, args.tokenizer_fingerprint, args.role_ids)):
        parser.error('preparation requires recipe, stage, tokenizer fingerprint and four role IDs')
    print(json.dumps(prepare(args.directory, args.recipe, args.stage_id,
                            tokenizer_fingerprint=args.tokenizer_fingerprint, role_ids=args.role_ids,
                            publish=not args.no_publish,
                            delta_sources={'train':binding(args.delta_train_source)} if args.delta_train_source else None)))


if __name__ == '__main__':
    main()
