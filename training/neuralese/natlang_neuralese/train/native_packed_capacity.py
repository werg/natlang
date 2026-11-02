"""Actual pending SFT batch geometry and read-only packed stress cases."""
from __future__ import annotations

import hashlib
import json
import time
import contextvars
import os
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
    from .native_prefix_packing import pack_causal_prefixes, ordinary_sft_physical_sharing_enabled
    physical=[]
    for batch in batches:
        share = ordinary_sft_physical_sharing_enabled(model, getattr(model,'share_causal_prefixes',False))
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
            retained_gradients=getattr(args,'accumulate_batches',1)>1,
            cotangent_mode='plain' if getattr(args,'cotangent_accumulator_backend',None)=='plain' else 'vjp')
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


def _oversize_shape(batches):
    """Describe whole-group geometry without combining maxima from different groups."""
    groups=[]
    for batch in batches:
        graphs=[]
        for example in batch:
            plan=example.plan
            shape={'id':example.id,'forward_tokens':int(plan.token_count),
                   'kv_slots':len(plan.slot_sources)}
            if hasattr(plan,'invocations') and hasattr(plan,'sequences'):
                nodes=[]
                for node, sequence in zip(plan.invocations,plan.sequences,strict=True):
                    nodes.append((len(node.token_ids),len(sequence),len(node.imports),
                                  len(node.producing),len(getattr(node,'logical_producing',()))))
                shape.update({'attention_workspace_rows':int(plan.attention_workspace_rows),
                    'node_shapes':nodes,'target_positions':sum(1 for w in getattr(example,'target_weights',()) if w>0),
                    'decision_count':len(getattr(example,'decisions',()) or ())})
            graphs.append(shape)
        groups.append({'graphs':graphs,'forward_tokens':sum(g['forward_tokens'] for g in graphs),
                       'kv_slots':sum(g['kv_slots'] for g in graphs),'graph_count':len(graphs)})
    return groups


def _one_group_dominates(measured, candidate):
    """Use one actually measured packed group as the envelope, never a cartesian max."""
    if candidate['graph_count'] > measured['graph_count']:
        return False
    for key in ('forward_tokens','kv_slots'):
        if candidate[key] > measured[key]:
            return False
    # Keep per-graph context lengths coupled to KV sizes. Sorting makes the
    # comparison independent of source order while preventing crossed maxima.
    old=sorted(((g['forward_tokens'],g['kv_slots']) for g in measured['graphs']), reverse=True)
    new=sorted(((g['forward_tokens'],g['kv_slots']) for g in candidate['graphs']), reverse=True)
    if not all(t <= old[i][0] and k <= old[i][1] for i,(t,k) in enumerate(new)):
        return False
    measured_graphs=sorted(measured['graphs'],key=lambda g:(g['forward_tokens'],g['kv_slots']),reverse=True)
    candidate_graphs=sorted(candidate['graphs'],key=lambda g:(g['forward_tokens'],g['kv_slots']),reverse=True)
    for actual, bound in zip(candidate_graphs,measured_graphs):
        # Historical proofs lack the richer attention/target layout and must
        # not cover a new graph whose padding and loss allocations are known.
        if 'node_shapes' not in actual:
            continue
        if 'node_shapes' not in bound:
            return False
        if any(actual[key]>bound[key] for key in ('attention_workspace_rows','target_positions','decision_count')):
            return False
        if len(actual['node_shapes'])!=len(bound['node_shapes']):
            return False
        # Preserve each invocation's coupled dimensions; a large total cannot
        # stand in for a larger individually padded attention workspace.
        for node,limit in zip(actual['node_shapes'],bound['node_shapes']):
            if len(node)!=len(limit) or any(a>b for a,b in zip(node,limit)):
                return False
    return True


def _nondominated_group_indices(groups):
    """Measure only actual groups that cover all requested sequential groups."""
    selected=[]
    for i, group in enumerate(groups):
        covered=False
        for j, other in enumerate(groups):
            if i == j or not _one_group_dominates(other, group):
                continue
            # Equal geometry retains the first actual group deterministically.
            if not _one_group_dominates(group, other) or j < i:
                covered=True
                break
        if not covered:
            selected.append(i)
    coverage=[]
    for group in groups:
        matches=[j for j in selected if _one_group_dominates(groups[j], group)]
        if not matches:
            raise ValueError('capacity selection does not cover a requested group')
        coverage.append(min(matches, key=lambda j:groups[j]['forward_tokens']+groups[j]['kv_slots']))
    return selected, coverage


def _capacity_memory_context(device):
    import torch
    from .trajectory_state import available_system_memory_bytes
    host_available=available_system_memory_bytes()
    if device.type != 'cuda' or not torch.cuda.is_initialized():
        return {'device_type':device.type,'free_bytes_at_probe':None,'total_bytes':None,
                'allocated_bytes':None,'reserved_bytes':None,'reusable_cache_bytes':None,
                'host_available_bytes':host_available}
    free,total=torch.cuda.mem_get_info(device)
    allocated=int(torch.cuda.memory_allocated(device));reserved=int(torch.cuda.memory_reserved(device))
    from .memory_policy import ledger_claim,ledger_reusable_bytes
    claim=ledger_claim()
    available=(ledger_reusable_bytes(claim['budget_bytes'],claim['host_bytes'],reserved,allocated,
               margin_bytes=0 if claim['discrete_cuda'] else None) if claim is not None else None)
    return {'device_type':'cuda','free_bytes_at_probe':int(free),'total_bytes':int(total),
            'allocated_bytes':allocated,'reserved_bytes':reserved,
            'reusable_cache_bytes':max(0,reserved-allocated),'host_available_bytes':host_available,
            **({'ledger_available_bytes':available} if available is not None else {})}


def optimizer_shape_identity(model, optimizer):
    """Describe optimizer storage and parameter partitions without reading tensor values."""
    import torch
    from .optim_restore import _children
    from ..common.hashing import canonical_json_sha256_hex
    named = list(getattr(optimizer, '_native_named_parameters', model.named_parameters()))
    names = {id(parameter): name for name, parameter in named}
    def shape(value):
        if torch.is_tensor(value):
            return {'shape':list(value.shape), 'stride':list(value.stride()),
                    'dtype':str(value.dtype), 'device':str(value.device), 'layout':str(value.layout)}
        if isinstance(value, dict):
            return {str(key):shape(item) for key,item in sorted(value.items(), key=lambda item:str(item[0]))}
        if isinstance(value, (tuple,list)):
            return [shape(item) for item in value]
        if value is None or type(value) in (bool,int,float,str):
            return {'type':type(value).__name__}
        raise TypeError('unsupported optimizer state in capacity identity')
    children=[]
    for name, child in _children(optimizer):
        groups=[]
        for group in child.param_groups:
            groups.append({'parameters':[names[id(parameter)] for parameter in group['params']],
                           'options':{key:value for key,value in group.items() if key not in ('params','lr','initial_lr')}})
        state={names[id(parameter)]:shape(value) for parameter,value in child.state.items()}
        children.append({'name':name, 'type':type(child).__module__+'.'+type(child).__qualname__,
                         'groups':groups, 'state':state})
    return canonical_json_sha256_hex(children)


def oversize_execution_identity(model, optimizer, *, args, backend, objective_binding):
    """Bind the memory execution path; parameter values may change, shapes may not."""
    from ..common.hashing import canonical_json_sha256_hex
    from .native_resident import code_inventory
    import torch
    package=Path(__file__).resolve().parents[1]
    code=code_inventory(package)
    params=[(name,tuple(parameter.shape),tuple(parameter.stride()),str(parameter.dtype),str(parameter.device),bool(parameter.requires_grad))
            for name,parameter in getattr(optimizer,'_native_named_parameters',model.named_parameters())]
    module_signature=[]
    roots=(('model',model),('backbone',getattr(model.port,'hf',model.port)))
    for root_name,root in roots:
        for name,module in (root.named_modules() if hasattr(root,'named_modules') else ()):
            forward=module.__dict__.get('forward')
            function=getattr(forward,'__func__',forward)
            module_signature.append((root_name,name,type(module).__module__,type(module).__qualname__,
                None if function is None else (getattr(function,'__module__',None),getattr(function,'__qualname__',None))))
    buffers=[(name,tuple(buffer.shape),tuple(buffer.stride()),str(buffer.dtype),str(buffer.device))
             for name,buffer in model.named_buffers()]
    device=next(model.parameters()).device
    hardware={'device':str(device),'torch':torch.__version__,'cuda':torch.version.cuda}
    if device.type=='cuda':
        hardware['name']=torch.cuda.get_device_name(device)
        hardware['capability']=list(torch.cuda.get_device_capability(device))
        hardware['total_bytes']=int(torch.cuda.get_device_properties(device).total_memory)
    hardware['cuda_budget_gb']=os.environ.get('NATLANG_CUDA_MEMORY_GB')
    from ..common.checkpoint_offload import checkpoint_input_storage_options
    checkpoint_storage=checkpoint_input_storage_options(model.port)
    return {'schema':'natlang.native-oversize-execution-identity/1',
        'sha256':canonical_json_sha256_hex({'code':code,'params':params,'optimizer_shape':optimizer_shape_identity(model,optimizer),'modules':module_signature,'buffers':buffers,'hardware':hardware,
            'backend':backend.configuration() if backend is not None else {'enabled':False},
            'checkpoint_layers':bool(args.checkpoint_layers),'chunk_tokens':int(args.chunk_tokens),
            'checkpoint_layer_group_size':getattr(model.port,'checkpoint_layer_group_size',1),
            'ffn_chunk_tokens':getattr(model.port,'ffn_chunk_tokens',0),
            'checkpoint_storage':checkpoint_storage,
            'cotangent_mode':'plain' if (getattr(args,'cotangent_accumulator_backend',None) or getattr(args,'native_cotangent_accumulator_backend',None))=='plain' else 'vjp',
            'retained_gradients':int(args.accumulate_batches)>1,'production_packing':True,
            'quantization':getattr(args,'quantization',None),
            'fp8_training':getattr(args,'fp8_training',None),
            'fp8_recipe':getattr(args,'fp8_recipe',None),
            'share_causal_prefixes':bool(getattr(model,'share_causal_prefixes',False)),
            'objective':objective_binding}),
        'code':code,'parameter_signature_sha256':canonical_json_sha256_hex(params),
        'checkpoint_layer_group_size':getattr(model.port,'checkpoint_layer_group_size',1),
        'ffn_chunk_tokens':getattr(model.port,'ffn_chunk_tokens',0), 'checkpoint_storage':checkpoint_storage,
        'hardware':hardware,'backend':backend.configuration() if backend is not None else {'enabled':False},
        'objective':objective_binding}


def _profile_shape_compatible(proof, *, execution_identity, precision, effective_layout, current_memory):
    if proof.get('execution_identity') != execution_identity:
        return False
    if proof.get('effective_layout') != effective_layout:
        return False
    capacity=proof.get('capacity') or {}
    qat=capacity.get('qat_point')
    expected=None if precision is None else {'point':precision[1],
        'loss_weight':float(precision[3])}
    # The active mix is recorded on both receipts, but is an interpolation of
    # operands, not an allocation-shape switch. Precision point and objective
    # remain exact identity components.
    if (None if qat is None else {'point':qat.get('point'),'loss_weight':qat.get('loss_weight')}) != expected:
        return False
    memory=proof.get('memory_context') or {}
    now=current_memory
    if memory.get('total_bytes') != now.get('total_bytes') or memory.get('device_type') != now.get('device_type'):
        return False
    return True


def _incremental_headroom_covers(case, current_memory):
    """Require room for measured incremental live allocations plus a safety margin."""
    host_needed=case.get('host_incremental_peak_bytes')
    host_available=current_memory.get('host_available_bytes')
    if any(type(value) is not int or value<0 for value in (host_needed,host_available)):
        return False
    if current_memory.get('device_type')=='cuda':
        needed=max(case.get('cuda_incremental_peak_allocated_bytes') or 0,
                   case.get('cuda_incremental_peak_reserved_bytes') or 0)
        free=current_memory.get('free_bytes_at_probe')
        reusable=current_memory.get('reusable_cache_bytes')
        if any(type(value) is not int or value<0 for value in (needed,free,reusable)):
            return False
        if free+reusable < needed+max(1<<30,(needed+3)//4):
            return False
        if (current_memory.get('ledger_available_bytes') is not None
                and current_memory['ledger_available_bytes']<needed+max(1<<30,(needed+3)//4)):
            return False
    elif current_memory.get('device_type')=='cpu':
        needed=0
    else:
        return False
    incremental=needed+host_needed
    margin=max(1<<30,(incremental+3)//4)
    return host_available>=incremental+margin


def _oversize_frontier_entries(reference):
    """Load valid measured cases once from a bounded durable proof frontier."""
    from .native_resident import bound
    try:
        record=json.loads(bound(reference['proof']).read_text())
        bound(record['source'])
    except (OSError,KeyError,ValueError,TypeError):
        return []
    if (record.get('schema')!='natlang.native-oversize-live-capacity/1' or record.get('passed') is not True
            or record.get('optimizer_updates')!=0 or record.get('earned_exposure')!=0
            or record.get('model_boundary')!='current-live-model-before-update'):
        return []
    capacity=record.get('capacity') or {}
    groups=record.get('candidate_groups') or []
    cases=capacity.get('cases') or []
    if (capacity.get('passed') is not True or capacity.get('optimizer_updates')!=0
            or capacity.get('production_packing') is not True or capacity.get('whole_graphs') is not True
            or capacity.get('cpu_and_all_cuda_rng_restored') is not True
            or capacity.get('task_quality_qualified') is not False or len(groups)!=len(cases)):
        return []
    output=[]
    for index,(case,group) in enumerate(zip(cases,groups,strict=True)):
        if (case.get('status')!='passed' or group.get('forward_tokens')!=case.get('forward_tokens')
                or group.get('kv_slots')!=case.get('kv_slots')
                or group.get('graph_count')!=len(group.get('graphs',()))
                or [graph.get('id') for graph in group.get('graphs',())]!=case.get('example_ids')
                or any(type(case.get(key)) is not int or case[key]<0 for key in (
                    'cuda_incremental_peak_allocated_bytes','cuda_incremental_peak_reserved_bytes',
                    'host_incremental_peak_bytes'))):
            continue
        output.append({'proof':reference['proof'],'record':record,'case_index':index,
            'case':case,'group':group})
    return output


def load_oversize_capacity_frontier(references):
    """Restore only the bounded active frontier; the full proof history stays separate."""
    frontier=[]
    for reference in references or ():
        if not isinstance(reference,dict) or not isinstance(reference.get('proof'),dict):continue
        frontier.extend(_oversize_frontier_entries(reference))
    return frontier


def _frontier_profile(entry):
    record=entry['record'];capacity=record.get('capacity') or {};qat=capacity.get('qat_point')
    return (record.get('execution_identity',{}).get('sha256'),record.get('effective_layout'),
        None if qat is None else (qat.get('point'),qat.get('loss_weight')),
        (record.get('memory_context') or {}).get('device_type'),(record.get('memory_context') or {}).get('total_bytes'))


def _entry_resource_envelope(entry):
    case=entry['case']
    return (max(int(case.get('cuda_incremental_peak_allocated_bytes') or 0),
                int(case.get('cuda_incremental_peak_reserved_bytes') or 0)),
            int(case.get('host_incremental_peak_bytes') or 0))


def _entry_dominates(left,right):
    return (_one_group_dominates(left['group'],right['group'])
        and all(a>=b for a,b in zip(_entry_resource_envelope(left),_entry_resource_envelope(right))))


def remember_oversize_capacity_proof(frontier, reference):
    """Add newly measured cases and retain only their authenticated nondominated envelope."""
    additions=_oversize_frontier_entries({'proof':reference})
    for entry in additions:
        profile=_frontier_profile(entry)
        same=[old for old in frontier if _frontier_profile(old)==profile]
        if any(_entry_dominates(old,entry) for old in same):continue
        frontier[:]=[old for old in frontier if not (_frontier_profile(old)==profile and _entry_dominates(entry,old))]
        frontier.append(entry)
    # Keep the lookup bounded under adversarially diverse geometry. Full proof
    # history remains immutable; truncation can only cause a fresh live gate.
    if len(frontier)>256:
        by_profile={}
        for entry in frontier:
            by_profile.setdefault(_frontier_profile(entry),[]).append(entry)
        frontier[:]=[]
        for entries in by_profile.values():
            entries.sort(key=lambda entry:(sum(entry['group'][k] for k in ('forward_tokens','kv_slots')),
                *_entry_resource_envelope(entry)),reverse=True)
            frontier.extend(entries[:256])
    return frontier


def serialize_oversize_capacity_frontier(frontier):
    unique={}
    for entry in frontier:
        key=(entry['proof']['path'],entry['proof']['sha256'])
        unique.setdefault(key,{'proof':dict(entry['proof'])})
    return list(unique.values())


def _find_oversize_envelope(frontier, candidate_groups, *, execution_identity, precision,
                            effective_layout, current_memory):
    # Microbatches execute sequentially. Each must be covered by one complete
    # measured case; different microbatches may use different measured cases.
    # Never construct an artificial envelope from independent maxima.
    matches=[]
    for candidate in candidate_groups:
        choices=[]
        for entry in frontier or ():
            record=entry['record'];case=entry['case'];measured=entry['group']
            if not _profile_shape_compatible(record,execution_identity=execution_identity,
                    precision=precision,effective_layout=effective_layout,current_memory=current_memory):
                continue
            if not _incremental_headroom_covers(case,current_memory):continue
            if _one_group_dominates(measured,candidate):
                choices.append((measured['forward_tokens']+measured['kv_slots'],entry))
        if not choices:return None
        entry=min(choices,key=lambda item:item[0])[1]
        matches.append((entry['proof'],entry['record'],entry['case'],entry['group']))
    return matches or None



def record_oversize_envelope_reuse(args, *, match, candidate_groups, source_path,
                                   execution_identity, precision, effective_layout,
                                   current_memory, checkpoint_sha256, global_step):
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.jsonio import write_json_durable
    from .native_resident import binding
    source_reference,source_record,source_case,measured=match[0]
    assert len(match)==len(candidate_groups)
    case_matches=[{'candidate_group':candidate,'envelope_source':ref,
        'envelope_source_global_step':record['global_step'],'measured_group':group,
        'measured_case_id':case.get('id')}
        for candidate,(ref,record,case,group) in zip(candidate_groups,match,strict=True)]
    reuse={'schema':'natlang.native-oversize-capacity-envelope-reuse/1',
        'global_step':global_step,'parent_checkpoint_sha256':checkpoint_sha256,
        'source':binding(source_path),'execution_identity':execution_identity,
        'effective_layout':effective_layout,
        'precision':None if precision is None else {'point':precision[1],
            'mix':float(precision[2]),'loss_weight':float(precision[3])},
        'memory_context':current_memory,'candidate_groups':candidate_groups,
        'envelope_source':source_reference,'envelope_source_global_step':source_record['global_step'],
        'measured_group':measured,'measured_case_id':source_case.get('id'),
        'case_matches':case_matches,
        'passed':True,'optimizer_updates':0,'earned_exposure':0,
        'scope':'candidate is bounded componentwise by one complete measured group per sequential microbatch; no independent maxima combined'}
    path=Path(args.out)/'oversize-capacity-reuse'/(canonical_json_sha256_hex(reuse)+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():write_json_durable(path,reuse)
    return binding(path)


def preflight_oversize_update(policy, model, micro_batches, baseline, *, frontier,
                              execution_identity, precision, backend):
    """Skip the async-save drain only when a bound past group covers this update."""
    batches=oversized_batches(micro_batches,baseline)
    if not batches:return False,None,[],None,None
    from .native_state_sft import _communication_prefix_allowed
    layout={'share_causal_prefixes':bool(getattr(model,'share_causal_prefixes',False)),
        'allow_communication_prefixes':bool(_communication_prefix_allowed(backend,precision))}
    groups=_oversize_shape(batches)
    memory=_capacity_memory_context(next(model.parameters()).device)
    match=_find_oversize_envelope(frontier,groups,execution_identity=execution_identity,
        precision=precision,effective_layout=layout,current_memory=memory)
    if match is None:
        policy.drain_async()
    return True,match,groups,layout,memory


def checkpoint_overlap_barrier(policy, micro_batches, baseline):
    """Finish a detached checkpoint before an unqualified oversized update."""
    oversized = bool(oversized_batches(micro_batches, baseline))
    if oversized:
        policy.drain_async()
    return oversized


def capacity_parent_sha(policy, base_sha256):
    """Read a committed parent without accepting a racing async replacement."""
    from ..common.hashing import sha256_file_hex_cached
    for attempt in range(3):
        if not policy.slot.exists():return base_sha256
        try:return sha256_file_hex_cached(policy.slot)
        except RuntimeError as error:
            if not str(error).startswith('file changed while hashing:') or attempt==2:
                raise
            # Keep normal asynchronous overlap. Only a checked read that
            # actually raced waits for the pending commit and retries fresh.
            policy.drain_async()
    raise AssertionError('unreachable capacity parent retry')


def qualify_oversize_batches(model, optimizer, micro_batches, *, args, backend, precision,
                             source_path, checkpoint_sha256, global_step, baseline,
                             objective=None, objective_binding=None, prior_frontier=(),
                             execution_identity=None):
    """Gate newly larger actual groups on the live model and actual QAT point, before SGD."""
    from ..eval.native_memory import qualify_capacity
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.jsonio import write_json_durable
    from .native_resident import binding
    if (objective is None) != (objective_binding is None):
        raise ValueError('oversize capacity needs the actual objective and its bound identity together')
    batches=oversized_batches(micro_batches, baseline)
    if not batches:return None
    candidate_groups=_oversize_shape(batches)
    memory_context=_capacity_memory_context(next(model.parameters()).device)
    from .native_state_sft import _communication_prefix_allowed
    effective_layout={'share_causal_prefixes':bool(getattr(model,'share_causal_prefixes',False)),
        'allow_communication_prefixes':bool(_communication_prefix_allowed(backend,precision))}
    if execution_identity is not None:
        reused=_find_oversize_envelope(prior_frontier,candidate_groups,
            execution_identity=execution_identity,precision=precision,effective_layout=effective_layout,
            current_memory=memory_context)
        if reused is not None:
            return record_oversize_envelope_reuse(args,match=reused,candidate_groups=candidate_groups,
                source_path=source_path,execution_identity=execution_identity,precision=precision,
                effective_layout=effective_layout,current_memory=memory_context,
                checkpoint_sha256=checkpoint_sha256,global_step=global_step)
    selected_indices, coverage_indices = _nondominated_group_indices(candidate_groups)
    measured_batches = [batches[index] for index in selected_indices]
    measured_groups = [candidate_groups[index] for index in selected_indices]
    named=getattr(optimizer,'_native_named_parameters',list(model.named_parameters()))
    prior={id(parameter):(parameter,parameter.grad) for _,parameter in named if parameter.grad is not None}
    for parameter,_ in prior.values():parameter.grad=None
    try:
        mode='plain' if getattr(args,'cotangent_accumulator_backend',None)=='plain' else 'vjp'
        report=qualify_capacity(model,optimizer,measured_batches,backend=backend,
            checkpoint_layers=args.checkpoint_layers,chunk_tokens=args.chunk_tokens,
            checkpoint_sha256=checkpoint_sha256,source_binding=binding(source_path),
            production_packing=True,retained_gradients=args.accumulate_batches>1,
            cotangent_mode=mode,precision=precision,objective=objective)
    finally:
        for parameter,gradient in prior.values():parameter.grad=gradient
    record={'schema':'natlang.native-oversize-live-capacity/1','global_step':global_step,
            'parent_checkpoint_sha256':checkpoint_sha256,'source':binding(source_path),
            'baseline':baseline,'model_boundary':'current-live-model-before-update',
            'candidate_groups':measured_groups,'requested_candidate_groups':candidate_groups,
            'requested_group_coverage':[{'requested_index':i,'measured_index':selected_indices.index(j)}
                                        for i,j in enumerate(coverage_indices)],
            'memory_context':memory_context,
            'execution_identity':execution_identity,'effective_layout':effective_layout,
            'objective':objective_binding,
            'capacity':report,'passed':report['passed'],'optimizer_updates':0,'earned_exposure':0,
            'scope':'whole-graph forward/backward with retained gradients; excludes optimizer-step workspace'}
    path=Path(args.out)/'oversize-capacity'/(canonical_json_sha256_hex(record)+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():write_json_durable(path,record)
    if not report['passed']:raise ValueError('actual oversize graph failed live-model/QAT capacity qualification: '+str(path))
    return binding(path)
