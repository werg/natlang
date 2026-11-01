"""Actual pending SFT batch geometry and read-only packed stress cases."""
from __future__ import annotations

import hashlib
import json
import time
import contextvars
from pathlib import Path

from ..data.native_sources import NativeGraphSource

_index_geometry = contextvars.ContextVar('native_packed_geometry_index', default=None)


class GeometryIndexSource(NativeGraphSource):
    """Collect geometry while the unchanged source validator indexes cold rows."""
    def _init_manifest(self, parents):
        from ..data.native_source_tree import initialize_manifest
        initialize_manifest(self, parents, load_child=type(self))

    def _parse(self, row):
        example, identity = super()._parse(row)
        collector = _index_geometry.get()
        if collector is not None:
            key = (example.id, identity)
            slots = len(example.plan.slot_sources)
            if key in collector and collector[key] != slots:
                raise ValueError('cold source indexing changed packed geometry')
            collector[key] = slots
        return example, identity


def index_source(path, tokenizer, roles):
    """Build using native validation; collect cold rows without a second parse."""
    observed = {}
    token = _index_geometry.set(observed)
    try:
        source = GeometryIndexSource(path, tokenizer, required_split='train', role_ids=roles)
    finally:
        _index_geometry.reset(token)
    return source, observed


def prepare_source(path, tokenizer, roles, out, *, prior=None, checkpoint_rows=0, progress_callback=None):
    """Validate/index a source and seal geometry, reusing every cold parse."""
    source, observed = index_source(path, tokenizer, roles)
    slots, reference, scan = geometry_for_source(source, out, prior=prior, allow_build=True,
        indexed_geometry=observed, checkpoint_rows=checkpoint_rows, progress_callback=progress_callback)
    return source, slots, reference, scan


def _rows_digest(entries):
    started = time.monotonic()
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(json.dumps([entry['id'], entry['binding'], entry['forward_tokens']],
            ensure_ascii=False, separators=(',', ':')).encode('utf8'))
        digest.update(b'\n')
    result = digest.hexdigest()
    print(json.dumps({'kind': 'native_geometry_rows_digest', 'rows': len(entries),
                      'seconds': time.monotonic() - started, 'sha256': result}), flush=True)
    return result


def geometry_for_source(source, out, *, prior=None, allow_build=False, indexed_geometry=None,
                        checkpoint_rows=0, progress_callback=None):
    """Reuse only an authenticated exact indexed prefix; parse genuinely new rows once."""
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.hashing import sha256_file_hex_cached
    from ..common.jsonio import write_json_durable
    from .native_resident import binding, bound
    package = Path(__file__).resolve().parents[1]
    parser_files = ('data/native_sources.py','data/native_graph_schema.py','data/decision_graph.py','model/native_graph_plan.py',
                    'data/native_pair_schema.py','common/row_index.py','common/jsonio.py','common/hashing.py')
    identity = {'tokenizer': source.tokenizer_fingerprint, 'roles': list(source.role_ids),
        'split': source.required_split, 'pair_model': source.pair_model,
        'parser_policy_sha256': canonical_json_sha256_hex({name:sha256_file_hex_cached(package/name) for name in parser_files})}
    values = []
    if prior is not None:
        saved = json.loads(bound(prior).read_text())
        if saved.get('schema') != 'natlang.native-packed-geometry/1' or saved.get('identity') != identity:
            raise ValueError('packed geometry receipt changed its parser or source identity')
        bound(saved['source'])
        values = saved.get('kv_slots')
        count = len(values) if isinstance(values, list) else -1
        if (count < 0 or count > len(source.entries) or any(type(item) is not int or item < 1 for item in values)
                or saved.get('rows_sha256') != _rows_digest(source.entries[:count])):
            raise ValueError('packed geometry does not bind the unchanged actual source prefix')
    start = len(values)
    if start < len(source.entries) and not allow_build:
        raise ValueError('packed capacity needs an independently prepared source geometry sidecar')
    values = list(values) + [None]*(len(source.entries)-start)
    indexed = indexed_geometry or {}
    for index in range(start,len(source.entries)):
        entry = source.entries[index]
        value = indexed.get((entry['id'],entry['binding']))
        if value is not None: values[index] = value
    directory = Path(out)/'packed-capacity-geometry'; directory.mkdir(parents=True, exist_ok=True)
    def publish(count):
        record = {'schema':'natlang.native-packed-geometry/1','identity':identity,
            'source':binding(source.path),'rows_sha256':_rows_digest(source.entries[:count]),'kv_slots':values[:count]}
        path = directory / (canonical_json_sha256_hex(record)+'.json')
        if path.exists():
            if json.loads(path.read_text()) != record: raise ValueError('immutable packed geometry changed')
        else: write_json_durable(path,record)
        reference = binding(path)
        if progress_callback is not None:
            progress_callback(reference, count, len(values))
        return reference
    missing = [index for index in range(start,len(values)) if values[index] is None]
    prefix = start
    while prefix<len(values) and values[prefix] is not None: prefix+=1
    last_published = start
    try:
        for index, example in zip(missing,source.iter_indices(missing),strict=True):
            entry = source.entries[index]
            if example.id != entry['id'] or example.plan.token_count != entry['forward_tokens']:
                raise ValueError('packed geometry differs from the actual indexed graph')
            slots = len(example.plan.slot_sources)
            if slots < 1: raise ValueError('packed geometry requires actual positive KV slots')
            values[index] = slots
            while prefix<len(values) and values[prefix] is not None: prefix+=1
            if checkpoint_rows and prefix-last_published>=checkpoint_rows:
                publish(prefix);last_published=prefix
    except BaseException:
        # Persist only fully validated contiguous rows. The same bound prefix
        # validation applies when the explicit builder resumes after a stop.
        if prefix>last_published: publish(prefix)
        raise
    if prefix != len(source.entries): raise ValueError('packed geometry omitted source rows')
    reference=publish(len(values))
    return values, reference, {'parsed_rows':len(missing),'indexed_rows':len(values)-start-len(missing),
        'reused_prefix_rows':start}


def maximum_pending_groups(entries, kv_slots, pending_batches):
    """Select actual maxima independently; keep real scheduled whole-graph membership."""
    if len(kv_slots) != len(entries): raise ValueError('packed geometry does not cover the source')
    best = [None, None, None]
    scores = [-1, -1, -1]
    for _,indices,_ in pending_batches:
        if len(indices) <= 1: continue
        group = tuple(indices)
        if len(set(group)) != len(group) or any(type(i) is not int or not 0<=i<len(entries) for i in group):
            raise ValueError('invalid packed pending batch')
        values=(sum(entries[i]['forward_tokens'] for i in group),sum(kv_slots[i] for i in group),len(group))
        for n,value in enumerate(values):
            if value>scores[n]: scores[n]=value; best[n]=group
    return list(dict.fromkeys(group for group in best if group is not None))


def qualify_pending(model, optimizer, train, pending_batches, args, *, backend, checkpoint_sha256,
                    global_step, prior_geometry=None, live_uncommitted=False, indexed_geometry=None,
                    capacity_identity=None):
    """Measure actual packed maxima without updates; not a universal geometry certificate."""
    from ..common.jsonio import write_json_durable
    from ..common.hashing import canonical_json_sha256_hex
    from ..eval.native_memory import qualify_capacity
    from .native_resident import binding, code_inventory
    if not any(len(indices)>1 for _,indices,_ in pending_batches): return None
    cold_complete = prior_geometry is None and indexed_geometry is not None and len(indexed_geometry)==len(train.entries)
    kv, geometry, scan = geometry_for_source(train, args.out, prior=prior_geometry, allow_build=cold_complete,
        indexed_geometry=indexed_geometry if cold_complete else None)
    groups = maximum_pending_groups(train.entries,kv,pending_batches)
    batches = [list(train.iter_indices(list(group))) for group in groups]
    from .native_state_sft import pack_native_graphs, _communication_prefix_allowed
    from .native_prefix_packing import pack_causal_prefixes
    physical=[]
    for batch in batches:
        share = bool(getattr(model,'share_causal_prefixes',False))
        # Match packed_graph_loss: ordinary SFT shares prefixes in BF16 too.
        # A pre-BF16 implementation reported unshared geometry while the
        # actual capacity forward/backward already used the shared layout.
        shared = (pack_causal_prefixes(batch, allow_communication=_communication_prefix_allowed(backend, None))
                  if share else None)
        plan = shared[0] if shared is not None else pack_native_graphs(batch)[0]
        physical.append({'forward_tokens':plan.token_count,'kv_slots':len(plan.slot_sources)})
        del plan, shared
    # A resident boundary follows an optimizer step and may still own its
    # completed gradients. Retain the exact tensors while the read-only probe
    # uses an empty gradient boundary, then restore them even on failure.
    named=getattr(optimizer,'_native_named_parameters',list(model.named_parameters()))
    prior_gradients={id(parameter):(parameter,parameter.grad) for _,parameter in named if parameter.grad is not None}
    for parameter,_ in prior_gradients.values(): parameter.grad=None
    try:
        report = qualify_capacity(model,optimizer,batches,backend=backend,
            checkpoint_layers=args.checkpoint_layers,chunk_tokens=args.chunk_tokens,
            checkpoint_sha256=checkpoint_sha256,source_binding=binding(train.path),production_packing=True,
            retained_gradients=getattr(args,'accumulate_batches',1)>1)
    finally:
        for parameter,gradient in prior_gradients.values():
            parameter.grad=gradient
    record = {'schema':'natlang.native-sft-packed-capacity/1','geometry':geometry,
        'source':binding(train.path),'global_step':global_step,'parent_checkpoint_sha256':checkpoint_sha256,
        'capacity_identity':capacity_identity,
        'runtime_code_sha256':canonical_json_sha256_hex(code_inventory(Path(__file__).resolve().parents[1])),
        'model_boundary':'live-uncommitted' if live_uncommitted else 'restored-checkpoint',
        'scan':scan,'groups':[[{'index':i,'id':train.entries[i]['id'],'binding':train.entries[i]['binding']}
                               for i in group] for group in groups],
        'physical_geometry':physical,
        'capacity':report,'passed':report['passed'],'optimizer_updates':0,'earned_exposure':0,
        'scope':'measured actual pending batches with maximum unshared token/KV aggregates and maximum graph count; not all topologies'}
    path=Path(args.out)/'packed-capacity'/(canonical_json_sha256_hex(record)+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('immutable packed capacity result already differs')
    if not path.exists():write_json_durable(path,record)
    if not report['passed']:raise ValueError('actual pending packed batch exceeded its learner capacity')
    return {'proof':binding(path),'geometry':geometry,'model_boundary':record['model_boundary']}


def oversize_baseline(source, prior_geometry, *, fallback_tokens, current_geometry=None):
    """Derive a source-prefix-bound boundary for newly larger complete graphs."""
    from .native_resident import bound
    if prior_geometry is None:
        return {'forward_tokens':int(fallback_tokens),'kv_slots':int(fallback_tokens),
                'basis':'configured-batch-token-fallback'}
    saved=json.loads(bound(prior_geometry).read_text())
    if current_geometry is not None:
        current=json.loads(bound(current_geometry).read_text())
        if saved.get('identity') != current.get('identity'):
            raise ValueError('oversize baseline parser or tokenizer identity changed')
    values=saved.get('kv_slots')
    if (not isinstance(values,list) or not values or len(values)>len(source.entries)
            or any(type(value) is not int or value<1 for value in values)
            or saved.get('rows_sha256')!=_rows_digest(source.entries[:len(values)])):
        raise ValueError('oversize capacity baseline does not bind the exact source prefix')
    bound(saved['source'])
    return {'forward_tokens':max(row['forward_tokens'] for row in source.entries[:len(values)]),
            'kv_slots':max(values),'basis':'authenticated-prior-source-geometry',
            'geometry':dict(prior_geometry),'prefix_rows':len(values)}


def policy_capacity_baseline(reference, *, checkpoint_sha256, objective_binding, source_binding,
                             cotangent_mode, retained_gradients):
    """Reuse measured geometry; the proof retains its original numerical parent."""
    from ..common.hashing import canonical_json_sha256_hex
    from .native_resident import bound
    proof=reference.get('proof') or {}
    proof_path=Path(reference['path'])
    if (canonical_json_sha256_hex(proof)!=reference.get('proof_sha256')
            or (proof_path.exists() and json.loads(bound(reference).read_text())!=proof)
            or (not proof_path.exists() and reference.get('authenticated_resume_checkpoint_sha256')!=checkpoint_sha256)):
        raise ValueError('policy capacity proof binding changed')
    capacity=proof.get('capacity') or {}; limits=proof.get('packing_limits') or {}
    if (proof.get('schema')!='natlang.native-policy-capacity-proof/1' or proof.get('passed') is not True
            or (proof.get('parent_checkpoint_sha256')!=checkpoint_sha256 and
                reference.get('authenticated_resume_checkpoint_sha256')!=checkpoint_sha256)
            or capacity.get('passed') is not True or capacity.get('objective')!=objective_binding
            or capacity.get('source')!=source_binding or capacity.get('cotangent_mode')!=cotangent_mode
            or capacity.get('production_packing') is not True
            or capacity.get('retained_accumulation_gradients') is not retained_gradients
            or capacity.get('qat_point') is not None
            or any(type(limits.get(k)) is not int or limits[k]<1 for k in ('forward_tokens','kv_slots'))):
        raise ValueError('policy capacity proof does not bind the actual first-update execution')
    return {**limits,'basis':'actual-policy-capacity-proof','proof':{'path':reference['path'],'sha256':reference['sha256']},
            'proof_parent_checkpoint_sha256':proof['parent_checkpoint_sha256'],
            'continuation_checkpoint_sha256':checkpoint_sha256,
            'scope':'measured execution geometry; no fresh numerical certification after an optimizer update'}


def oversized_batches(micro_batches, baseline):
    """Return actual complete microbatches exceeding the authenticated baseline."""
    return [batch for batch in micro_batches
            if sum(example.plan.token_count for example in batch) > baseline['forward_tokens']
            or sum(len(example.plan.slot_sources) for example in batch) > baseline['kv_slots']]


def checkpoint_overlap_barrier(policy, micro_batches, baseline):
    """Finish a detached checkpoint before an unqualified oversized update."""
    oversized = bool(oversized_batches(micro_batches, baseline))
    if oversized:
        policy.drain_async()
    return oversized


def qualify_oversize_batches(model, optimizer, micro_batches, *, args, backend, precision,
                             source_path, checkpoint_sha256, global_step, baseline,
                             objective=None, objective_binding=None):
    """Gate newly larger actual groups on the live model and actual QAT point, before SGD."""
    from ..eval.native_memory import qualify_capacity
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.jsonio import write_json_durable
    from .native_resident import binding
    if (objective is None) != (objective_binding is None):
        raise ValueError('oversize capacity needs the actual objective and its bound identity together')
    batches=oversized_batches(micro_batches, baseline)
    if not batches:return None
    named=getattr(optimizer,'_native_named_parameters',list(model.named_parameters()))
    prior={id(parameter):(parameter,parameter.grad) for _,parameter in named if parameter.grad is not None}
    for parameter,_ in prior.values():parameter.grad=None
    try:
        mode='plain' if getattr(args,'cotangent_accumulator_backend',None)=='plain' else 'vjp'
        report=qualify_capacity(model,optimizer,batches,backend=backend,
            checkpoint_layers=args.checkpoint_layers,chunk_tokens=args.chunk_tokens,
            checkpoint_sha256=checkpoint_sha256,source_binding=binding(source_path),
            production_packing=True,retained_gradients=args.accumulate_batches>1,
            cotangent_mode=mode,precision=precision,objective=objective)
    finally:
        for parameter,gradient in prior.values():parameter.grad=gradient
    record={'schema':'natlang.native-oversize-live-capacity/1','global_step':global_step,
            'parent_checkpoint_sha256':checkpoint_sha256,'source':binding(source_path),
            'baseline':baseline,'model_boundary':'current-live-model-before-update',
            'objective':objective_binding,
            'capacity':report,'passed':report['passed'],'optimizer_updates':0,'earned_exposure':0,
            'scope':'whole-graph forward/backward with retained gradients; excludes optimizer-step workspace'}
    path=Path(args.out)/'oversize-capacity'/(canonical_json_sha256_hex(record)+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():write_json_durable(path,record)
    if not report['passed']:raise ValueError('actual oversize graph failed live-model/QAT capacity qualification: '+str(path))
    return binding(path)
