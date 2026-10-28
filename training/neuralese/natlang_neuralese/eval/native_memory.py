"""Actual-parent whole-graph memory evidence without SGD or full gradient copies."""
from __future__ import annotations

import contextlib
import hashlib
import json
import copy
import math
import threading
import time
import traceback
from pathlib import Path

import torch

from ..train.loop import capture_training_rng_state, restore_training_rng_state
from ..train.optim_restore import _children


class CapacityStateChangedError(RuntimeError):
    """The read-only invariant failed; useful SGD must not continue."""


def _tensor_versions(value):
    if torch.is_tensor(value):
        return (id(value), value._version, str(value.dtype), tuple(value.shape))
    if isinstance(value, dict):
        return {str(key): _tensor_versions(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tensor_versions(item) for item in value]
    return value


def _optimizer_versions(optimizer):
    return {name: {'state': {id(parameter): _tensor_versions(state) for parameter,state in child.state.items()},
                   'groups': _tensor_versions([{key:value for key,value in group.items() if key!='params'}
                                              for group in child.param_groups])}
            for name,child in _children(optimizer)}



def _reference_digest(value):
    """Hash small execution/backend/RNG metadata, never model or optimizer values."""
    digest = hashlib.sha256()
    def add(item):
        if torch.is_tensor(item):
            if item.layout != torch.strided:raise TypeError('non-strided reference metadata')
            tensor = item.detach().cpu().contiguous()
            add(('tensor', str(tensor.dtype), tuple(tensor.shape)))
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
        elif isinstance(item, str):
            digest.update(('str:' + json.dumps(str(item)) + ';').encode())
        elif item is None or type(item) in (bool, int):
            digest.update((type(item).__name__ + ':' + json.dumps(item) + ';').encode())
        elif type(item) is float and math.isfinite(item):
            digest.update(('float:' + item.hex() + ';').encode())
        elif isinstance(item, (list, tuple)):
            digest.update(b'(' if isinstance(item,tuple) else b'[')
            for child in item:add(child)
            digest.update(b')' if isinstance(item,tuple) else b']')
        elif isinstance(item, dict) and all(type(key) in (str, int) for key in item):
            digest.update(b'{')
            for key in sorted(item, key=lambda key:(type(key).__name__, str(key))):
                add(key);add(item[key])
            digest.update(b'}')
        else:
            raise TypeError('unsupported numerical-reference metadata')
    add(value)
    return digest.hexdigest()


def _reference_versions(value):
    if torch.is_tensor(value):
        if value.layout != torch.strided:raise TypeError('non-strided reference state')
        return (id(value), value._version, str(value.dtype), str(value.device),
                tuple(value.shape), tuple(value.stride()), value.data_ptr())
    if isinstance(value, dict):
        return {key:_reference_versions(item) for key,item in value.items()}
    if isinstance(value, tuple):
        return tuple(_reference_versions(item) for item in value)
    if isinstance(value, list):
        return [_reference_versions(item) for item in value]
    return value


def _reference_boundary(model, optimizer, backend):
    """Versioned state identities at one unarmed, read-only optimizer boundary."""
    named = list(model.named_parameters())
    named.extend(getattr(optimizer, '_native_named_parameters', ()))
    parameters = {id(parameter):parameter for _,parameter in named}
    execution = ('native_shared_execution', 'native_norm_execution',
                 'native_cotangent_accumulator_backend', 'natlang_precision',
                 'native_linear_precision', 'native_attention_precision',
                 'ffn_chunk_tokens', 'checkpoint_layers', 'checkpoint_direct_token_limit', 'share_causal_prefixes')
    owners = (model, model.port, getattr(model.port, 'hf', None))
    modes = [(id(module), module.training, id(type(module).forward),
              id(module.__dict__.get('forward'))) for module in model.modules()]
    device = next(iter(parameters.values())).device
    state = {
        'model':id(model), 'optimizer':id(optimizer),
        'parameters':[(identity, parameter.requires_grad, _reference_versions(parameter),
                       parameter.grad is None) for identity,parameter in parameters.items()],
        'buffers':[(name,_reference_versions(buffer)) for name,buffer in model.named_buffers()],
        'module_dispatch':modes,
        'execution':[{name:getattr(owner,name) for name in execution if hasattr(owner,name)}
                     for owner in owners if owner is not None],
        'optimizer_children':[(name,id(child),
            {id(parameter):_reference_versions(value) for parameter,value in child.state.items()},
            [_reference_versions(group) for group in child.param_groups])
            for name,child in _children(optimizer)],
        'rng':capture_training_rng_state('cuda' if device.type=='cuda' or torch.cuda.is_initialized() else 'cpu'),
        'backend':None if backend is None else {
            'identity':id(backend), 'configuration':backend.configuration(),
            'extra_state':backend.extra_state(),
            'row_counts':getattr(backend,'row_counts',None),
            'has_executed':getattr(backend,'has_executed',False)},
    }
    # Only backend/RNG tensors are value-hashed; parameter/moment tensors above
    # are replaced by identity/version/device/storage metadata first.
    full = _reference_digest(state)
    # A backend may restore an identical scale value with a fresh buffer
    # version, or initialize a nonparameter execution cache. This prevents
    # reuse without imposing a new buffer-identity law on valid qualification.
    return full, _reference_digest({key:value for key,value in state.items() if key!='buffers'})


class _NumericalReferenceReuse:
    """Successful reference evidence lives only for this serial run_prepared call."""
    def __init__(self):
        self.completed = {}

    def measure(self, invocation, *, model, optimizer, backend, request, operation):
        try:
            key = _reference_digest(invocation)
            before = _reference_boundary(model, optimizer, backend)
        except (TypeError, NotImplementedError):
            # Unknown backend metadata cannot authorize reuse. The existing
            # numerical comparison and its own state guards still run normally.
            return operation(), {'measured':True, 'reuse_supported':False}
        previous = self.completed.get(key)
        if previous is not None and previous['boundary']==before:
            return copy.deepcopy(previous['proof']), {
                'measured':False, 'same_owned_boundary':True,
                'original_request':copy.deepcopy(previous['request']),
                'invocation_sha256':key, 'boundary_sha256':before[0]}
        proof = operation()
        after = _reference_boundary(model, optimizer, backend)
        if after[1] != before[1]:
            raise CapacityStateChangedError('numerical reference changed its owned model/optimizer/backend/RNG boundary')
        if proof.get('passed') is True and after == before:
            self.completed[key] = {'boundary':after, 'proof':copy.deepcopy(proof),
                                   'request':copy.deepcopy(request)}
        return proof, {'measured':True, 'reuse_supported':after==before,
                       'invocation_sha256':key, 'boundary_sha256':before[0]}


def _host_memory():
    result = {}
    try:
        rows = dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        for key in ('VmRSS','VmPin','VmLck'):
            result[key+'_bytes'] = int(rows.get(key,'0').split()[0])*1024
        stat = dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        result['cgroup_anon_bytes'] = int(stat['anon'])
        result['cgroup_current_bytes'] = int(Path('/sys/fs/cgroup/memory.current').read_text())
    except (OSError, ValueError, KeyError):
        pass
    return result


class _HostPeak:
    def __init__(self):
        self.peak = _host_memory()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample,daemon=True)
    def sample(self):
        while not self.stop.wait(.01):
            for key,value in _host_memory().items():self.peak[key]=max(self.peak.get(key,0),value)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*unused):self.stop.set();self.thread.join()


def qualify_capacity(model, optimizer, cases, *, backend=None, checkpoint_layers=True,
                     chunk_tokens=1024, checkpoint_sha256, source_binding, gradient_observer=None, objective=None,
                     production_packing=False):
    """Cases are whole, source-authenticated graphs chosen by the supervisor.

    The existing gated optimizer hooks stay installed but unarmed. Temporary
    observation hooks release each gradient exactly where production consumes
    it; no optimizer method, parameter write or full-model gradient copy runs.
    CUDA lifetime peaks are not reset. This is capacity, not task qualification.
    """
    from ..train.native_state_sft import packed_graph_loss
    if not cases:
        raise ValueError('capacity qualification needs actual complete source graphs')
    # Use the actual optimizer inventory, including functional native HF
    # parameters that a port facade may not register under model.modules().
    named = [(name,p) for name,p in getattr(optimizer,'_native_named_parameters',model.named_parameters()) if p.requires_grad]
    if any(p.grad is not None for _,p in named):
        raise ValueError('capacity qualification requires an empty update boundary')
    for _,child in _children(optimizer):
        if getattr(child,'_armed',None) is not None or (getattr(child,'_hooks',[]) and not getattr(child,'_gated',False)):
            raise ValueError('capacity qualification cannot enter an armed or ungated optimizer')
    device = next(model.parameters()).device
    rng_device = 'cuda' if device.type=='cuda' or torch.cuda.is_initialized() else 'cpu'
    rng = capture_training_rng_state(rng_device)
    training_modes = {module:module.training for module in model.modules()}
    had_layers = hasattr(model.port,'checkpoint_layers')
    layers = getattr(model.port,'checkpoint_layers',None)
    parameters={id(p):p for p in model.parameters()}
    parameters.update({id(p):p for _,p in named})
    parameter_versions = {key:p._version for key,p in parameters.items()}
    optimizer_versions = _optimizer_versions(optimizer)
    extra = copy.deepcopy(backend.extra_state()) if backend is not None else None
    counts = backend.collect_row_counts() if backend is not None else None
    executed = getattr(backend,'has_executed',False)
    rows, hooks, observed = [], [], {}
    def capture(parameter, *, name):
        gradient=parameter.grad
        if gradient is None:return
        # A capacity proof must measure the same widened low-precision
        # fan-in as native_training_update.  The post-accumulate observer runs
        # after carrier anchors have contributed; a low-precision gradient
        # with no carrier contribution would otherwise be cleared here and
        # silently qualify a direct-weight arithmetic bypass.
        from ..model.native_weight_carrier import active_cotangent_scope
        cotangents=active_cotangent_scope()
        if cotangents is not None and parameter.dtype in (torch.float16,torch.bfloat16):
            entry=cotangents._entry_for(parameter)
            if entry is None or entry['count']==0:
                raise RuntimeError('capacity gradient bypassed the shared low-precision cotangent carrier: '+name)
        finite=bool(torch.isfinite(gradient).all())
        norm=float(torch.linalg.vector_norm(gradient,dtype=torch.float32))
        if not finite or not math.isfinite(norm):
            raise FloatingPointError('nonfinite capacity-check gradient: '+name)
        observed[name]={'elements':gradient.numel(),'dtype':str(gradient.dtype),'l2':norm}
        if gradient_observer is not None:gradient_observer(name,gradient)
        parameter.grad=None
    try:
        for name,parameter in named:
            hooks.append(parameter.register_post_accumulate_grad_hook(lambda p,name=name:capture(p,name=name)))
        model.train()
        for example in cases:
            restore_training_rng_state(rng,rng_device)
            observed={}
            loss=None
            batch=list(example) if isinstance(example,(list,tuple)) else [example]
            row={'id':batch[0].id if len(batch)==1 else 'batch:'+','.join(item.id for item in batch),
                 'example_ids':[item.id for item in batch],
                 'forward_tokens':sum(item.plan.token_count for item in batch),
                 'kv_slots':sum(len(item.plan.slot_sources) for item in batch),'status':'running'}
            rows.append(row)
            started=time.monotonic()
            phase='forward'
            try:
                with _HostPeak() as host:
                    from ..model.native_weight_carrier import ParameterCotangentScope
                    with ParameterCotangentScope.for_model(model,mode='vjp') as cotangents:
                        with backend.autocast() if backend is not None else contextlib.nullcontext():
                            packing = ({'share_causal_prefixes':True}
                                if getattr(model,'share_causal_prefixes',False) else {})
                            if production_packing:
                                packing.update(collect_accuracy=False, allow_communication_prefixes=backend is None)
                            if objective is not None:packing['objective']=objective
                            loss,_=packed_graph_loss(model,batch,checkpoint_layers=checkpoint_layers,
                                chunk_tokens=chunk_tokens,**packing)
                        phase='backward'
                        loss.backward()
                        phase='cotangent-finish'
                        cotangents.finish()
                        row['cotangent_accumulators'] = cotangents.allocation_report()
                        if device.type=='cuda':torch.cuda.synchronize(device)
                if not observed or not math.isfinite(float(loss.detach())):
                    raise FloatingPointError('capacity check has no actual finite gradient/loss')
                row.update(status='passed',loss=float(loss.detach()),gradients=dict(observed),
                    host_peak=host.peak)
            except torch.cuda.OutOfMemoryError as error:
                row.update(status='out_of_memory',error=str(error),host_peak=host.peak,
                    failure_phase=phase,failure_traceback=traceback.format_exc())
                hook=getattr(model.port,'activation_checkpoint',None)
                if callable(getattr(hook,'snapshot',None)):
                    row['checkpoint_input_offload']=hook.snapshot()
            finally:
                loss=None
                for _,parameter in named:parameter.grad=None
                row['seconds']=time.monotonic()-started
                row['cuda_peak_allocated_bytes']=(torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0)
                row['cuda_peak_reserved_bytes']=(torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
                row['cuda_peak_includes_prior_restore']=True
                row['host_after']=_host_memory()
                host_stats=getattr(torch.cuda.memory,'host_memory_stats',None)
                row['pinned_host_allocator'] = ({key:value for key,value in host_stats().items()
                    if key in {'allocated_bytes.current','allocated_bytes.peak','active_bytes.current','active_bytes.peak'}}
                    if host_stats is not None and torch.cuda.is_initialized() else {'available':False})
            if row['status']!='passed':break
        after=backend.collect_row_counts() if backend is not None else {}
        return {'schema':'natlang.native-memory-capacity-check/1',
            'passed':len(rows)==len(cases) and all(row['status']=='passed' for row in rows),
            'parent_checkpoint_sha256':checkpoint_sha256,'source':source_binding,'cases':rows,
            **({'objective':objective.binding()} if objective is not None else {}),
            'optimizer_updates':0,'gradients_released_per_parameter':True,'whole_graphs':True,
            'task_quality_qualified':False,'cpu_and_all_cuda_rng_restored':True,
            'checkpoint_layers':checkpoint_layers,'chunk_tokens':chunk_tokens,
            'backend':backend.configuration() if backend is not None else {'enabled':False},
            'executed_projections':{key:after.get(key,0)-counts.get(key,0) for key in after} if counts is not None else {},
            'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,
                'hardware':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
                'capability':list(torch.cuda.get_device_capability(device)) if device.type=='cuda' else None}}
    finally:
        for hook in hooks:hook.remove()
        for _,parameter in named:parameter.grad=None
        for module,mode in training_modes.items():module.training=mode
        if had_layers:model.port.checkpoint_layers=layers
        elif hasattr(model.port,'checkpoint_layers'):delattr(model.port,'checkpoint_layers')
        restore_error=None
        try:
            if backend is not None:
                backend.restore_extra_state(extra)
                if hasattr(backend,'row_counts'):
                    backend.row_counts.clear();backend.row_counts.update(counts)
                backend.has_executed=executed
        except Exception as error:
            restore_error=error
        try:
            restore_training_rng_state(rng,rng_device)
        except Exception as error:
            restore_error=error
        if restore_error is not None:
            raise CapacityStateChangedError('capacity check could not restore backend/RNG state') from restore_error
        if device.type=='cuda':torch.cuda.empty_cache()
        if parameter_versions!={key:p._version for key,p in parameters.items()}:
            raise CapacityStateChangedError('capacity check changed a model parameter')
        if optimizer_versions!=_optimizer_versions(optimizer):
            raise CapacityStateChangedError('capacity check changed optimizer state or rates')


def offload_numerical_parity(model, optimizer, example, *, backend=None, chunk_tokens=1024,
                             checkpoint_sha256, source_binding, checkpoint_layers=True, checkpoint_input_offload=None, checkpoint_input_offload_directory=None, objective=None):
    """Bitwise full named-gradient parity, retaining only per-parameter digests."""
    from ..common.checkpoint_offload import configure_checkpoint_input_offload
    checkpoint_input_offload = checkpoint_layers if checkpoint_input_offload is None else checkpoint_input_offload
    if checkpoint_input_offload and not checkpoint_layers:
        raise ValueError('checkpoint input offload requires activation checkpointing')
    had_hook=hasattr(model.port,'activation_checkpoint')
    original=getattr(model.port,'activation_checkpoint',None)
    gradients={};mismatches=[]
    def reference(name,gradient):
        value=gradient.detach().cpu().contiguous()
        gradients[name]=(str(value.dtype),tuple(value.shape),hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest())
    seen=set()
    def compare(name,gradient):
        value=gradient.detach().cpu().contiguous();seen.add(name)
        actual=(str(value.dtype),tuple(value.shape),hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest())
        if gradients.get(name)!=actual:mismatches.append(name)
    try:
        configure_checkpoint_input_offload(model.port,enabled=False)
        baseline=qualify_capacity(model,optimizer,[example],backend=backend,chunk_tokens=chunk_tokens,
            checkpoint_sha256=checkpoint_sha256,source_binding=source_binding,gradient_observer=reference,objective=objective)
        configure_checkpoint_input_offload(model.port,enabled=checkpoint_input_offload,
            directory=checkpoint_input_offload_directory)
        candidate=qualify_capacity(model,optimizer,[example],backend=backend,chunk_tokens=chunk_tokens,
            checkpoint_layers=checkpoint_layers,checkpoint_sha256=checkpoint_sha256,source_binding=source_binding,gradient_observer=compare,objective=objective)
        return {'passed':baseline['passed'] and candidate['passed'] and not mismatches
                    and seen==gradients.keys() and baseline['cases'][0].get('loss')==candidate['cases'][0].get('loss'),
                'baseline':baseline,'offloaded':candidate,'candidate_checkpoint_layers':checkpoint_layers,
                'candidate_checkpoint_input_offload':checkpoint_input_offload,
                **({'candidate_checkpoint_input_offload_directory':checkpoint_input_offload_directory} if checkpoint_input_offload_directory else {}),
                'gradient_parameters':len(gradients),
                'different_gradient_names':mismatches,'missing_gradient_names':sorted(gradients.keys()-seen),
                'comparison':'bitwise named-gradient digests and loss, no retained full-model copies'}
    finally:
        if had_hook:model.port.activation_checkpoint=original
        elif hasattr(model.port,'activation_checkpoint'):delattr(model.port,'activation_checkpoint')


def memory_runtime(device):
    result={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.device(device).type}
    if torch.device(device).type=='cuda':
        properties=torch.cuda.get_device_properties(device)
        result.update(hardware=properties.name,capability=[properties.major,properties.minor],total_memory_bytes=properties.total_memory)
    return result


def capacity_identity(config,args,backend):
    accumulator_backend = getattr(args, 'cotangent_accumulator_backend', None)
    if accumulator_backend is None:
        accumulator_backend = getattr(args, 'native_cotangent_accumulator_backend', 'cpu_file')
    from ..model.native_weight_carrier import COTANGENT_BACKENDS
    if accumulator_backend not in COTANGENT_BACKENDS:
        raise ValueError('capacity identity needs a declared cotangent accumulator backend')
    objective={}
    if getattr(args,'objective_spec',None):
        from ..train.native_preference import NativePreferenceObjective
        spec=json.loads(Path(args.objective_spec).read_text())
        if spec.get('kind')=='native-policy':
            from ..rl.native_replay import NativePolicyObjective
            value=NativePolicyObjective(args.objective_spec)
        else:
            value=NativePreferenceObjective(args.objective_spec,
                beta=getattr(args,'preference_beta',None),margin=getattr(args,'preference_margin',None))
        objective={'objective':value.binding()}
    return {**objective,
        **({'optimizer_state_storage':args.optimizer_state_storage}
           if getattr(args,'optimizer_state_storage','device')=='cpu' else {}),
        **({'checkpoint_input_offload_directory':args.checkpoint_input_offload_directory}
           if getattr(args,'checkpoint_input_offload_directory',None) else {}),'model_config':config,'batch_graphs':args.batch_graphs,'batch_tokens':args.batch_tokens,
        'checkpoint_layers':args.checkpoint_layers,'chunk_tokens':args.chunk_tokens,
        **({'checkpoint_direct_token_limit':args.checkpoint_direct_token_limit} if getattr(args,'checkpoint_direct_token_limit',0) else {}),
        **({'ffn_chunk_tokens':args.ffn_chunk_tokens}if getattr(args,'ffn_chunk_tokens',0)else {}),
        'checkpoint_input_offload':args.checkpoint_input_offload,
        **({'share_causal_prefixes':True} if getattr(args,'share_causal_prefixes',False) else {}),
        'runtime':memory_runtime(args.device),
        **({'cotangent_accumulator_backend': accumulator_backend}
           if accumulator_backend != 'cpu_file' else {}),
        'fp8':{'kind':args.fp8_training,'recipe':args.fp8_recipe,'amax_history_len':args.fp8_amax_history_len,
               'input_gradient_precision':getattr(backend,'input_gradient_precision',None)}}


def _run_one_prepared(model,optimizer,train,roles,config,args,*,backend,checkpoint_sha256,objective=None,reference_reuse=None,index_validation=None):
    """Called once at the normal restore boundary, before any new optimizer step.

    Failed novel source checks cannot stop useful SGD on the existing source.
    A failed check on a source already made active is an explicit blocking error.
    """
    from ..train.native_resident import bound,binding,code_inventory,_check_indexes
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.jsonio import write_json_durable
    from ..train.native_source_preparation import NativeSourcePreparation
    from ..common.checkpoint_offload import configure_checkpoint_input_offload
    if args.checkpoint_input_offload and not args.checkpoint_layers:
        raise ValueError('checkpoint input offload requires activation checkpointing')
    request_path=bound(json.loads(args.memory_qualification))
    request=json.loads(request_path.read_text())
    package=Path(__file__).resolve().parents[1]
    code_sha=canonical_json_sha256_hex(code_inventory(package))
    identity=capacity_identity(config,args,backend)
    actual_objective=objective.binding() if objective is not None else None
    if request.get('objective')!=actual_objective:
        raise ValueError('memory qualification objective differs from actual paired training objective')
    if (request.get('schema')!='natlang.native-memory-qualification-request/1'
            or request.get('identity')!=identity or request.get('runtime_code_sha256')!=code_sha):
        raise ValueError('memory qualification request differs from actual restored model/runtime/packing')
    (index_validation or _check_indexes)(request['indexes'])
    source_split=request.get('split','train')
    if source_split not in ('train','test'):
        raise ValueError('capacity source split must be train or test')
    preparation=NativeSourcePreparation(config['tokenizer_fingerprint'],roles.ids)
    preparation.retain(train)
    source=preparation.load(request['source'],split=source_split,pair_model=request.get('pair_model'))
    selected=request['cases']
    if not selected or len({row['index'] for row in selected})!=len(selected):
        raise ValueError('memory qualification needs distinct complete graph cases')
    for row in selected:
        entry=source.entries[row['index']]
        if entry['id']!=row['id'] or entry['binding']!=row['binding']:
            raise ValueError('memory qualification case differs from validated source')
    reference=request['reference']
    ref_index=reference['index']
    if (train.entries[ref_index]['id']!=reference['id'] or train.entries[ref_index]['binding']!=reference['binding']):
        raise ValueError('memory qualification numerical reference differs from the current training source')
    result={'schema':'natlang.native-memory-qualification/1','passed':False,'request':binding(request_path),
        'parent_checkpoint_sha256':checkpoint_sha256,'identity':identity,'runtime_code_sha256':code_sha,
        'task_quality_qualified':False,'optimizer_updates':0,'numerical_reference':None,'capacity':None}
    evidence_directory=args.out/'memory-qualifications'/binding(request_path)['sha256']/checkpoint_sha256
    evidence_directory.mkdir(parents=True,exist_ok=True)
    output=evidence_directory/'proof.json'
    certificate=evidence_directory/'capacity.json'
    def publish_capacity(proof):
        limits={key:max(row[key] for row in proof['capacity']['cases']) for key in ('forward_tokens','kv_slots')}
        value={'schema':'natlang.native-resident-geometry-qualification/1',
            'passed':True,'identity':identity,'runtime_code_sha256':code_sha,'limits':limits,'evidence':binding(output),
            'scope':'actual complete admitted-cohort maximum graph cases selected in the bound request'}
        if certificate.exists():
            if json.loads(certificate.read_text())!=value:
                raise ValueError('immutable memory capacity certificate differs from its proof')
        else:
            write_json_durable(certificate,value)
        from ..train.native_resident import geometry_capacity
        geometry_capacity(binding(certificate),identity=identity,code=code_inventory(package))
        args.resident_geometry_qualification=certificate
    if output.exists():
        previous=json.loads(output.read_text())
        if (previous.get('request')!=result['request'] or previous.get('identity')!=identity
                or previous.get('runtime_code_sha256')!=code_sha
                or previous.get('parent_checkpoint_sha256')!=checkpoint_sha256):
            raise ValueError('immutable memory qualification proof differs from its request')
        # Publication may have stopped after the immutable proof and before its
        # certificate. Reconstruct metadata from that proof without another GPU run.
        if previous.get('passed'):
            publish_capacity(previous)
            return previous
        if not (previous.get('numerical_reference') or {}).get('passed'):
            configure_checkpoint_input_offload(model.port,enabled=False)
            args.checkpoint_input_offload=False
            args.checkpoint_input_offload_directory=None
            if not args.checkpoint_layers:
                raise ValueError('noncheckpointed numerical qualification failed; exact-state checkpointed fallback required')
        if any(row['id'] in train.bindings for row in selected):
            raise ValueError('active source still has a failed whole-graph memory qualification')
        return previous
    try:
        def measure_reference():
            example=next(train.iter_indices([ref_index]))
            return offload_numerical_parity(model,optimizer,example,backend=backend,
                chunk_tokens=args.chunk_tokens,checkpoint_sha256=checkpoint_sha256,source_binding=binding(train.path),
                checkpoint_layers=args.checkpoint_layers,checkpoint_input_offload=args.checkpoint_input_offload,
                checkpoint_input_offload_directory=getattr(args,'checkpoint_input_offload_directory',None),objective=objective)
        if reference_reuse is None:
            result['numerical_reference']=measure_reference()
        else:
            invocation={'parent_checkpoint_sha256':checkpoint_sha256,'runtime_code_sha256':code_sha,
                'identity':identity,'reference_source':binding(train.path),'reference':reference,
                'reference_entry':train.entries[ref_index],'roles':list(roles.ids),
                'objective':actual_objective}
            result['numerical_reference'],result['numerical_reference_execution']=reference_reuse.measure(
                invocation,model=model,optimizer=optimizer,backend=backend,
                request=result['request'],operation=measure_reference)
        if result['numerical_reference']['passed']:
            examples=list(source.iter_indices([row['index'] for row in selected]))
            result['capacity']=qualify_capacity(model,optimizer,examples,backend=backend,chunk_tokens=args.chunk_tokens,
                checkpoint_layers=args.checkpoint_layers,checkpoint_sha256=checkpoint_sha256,source_binding=request['source'],objective=objective)
            result['passed']=result['capacity']['passed']
    except (RuntimeError,FloatingPointError,MemoryError) as error:
        if isinstance(error,CapacityStateChangedError):raise
        result['error']=str(error)
    write_json_durable(output,result)
    write_json_durable(args.out/'memory-qualification.json',{'passed':result['passed'],'proof':binding(output),
        'request':result['request'],'parent_checkpoint_sha256':checkpoint_sha256,'optimizer_updates':0})
    if result['numerical_reference'] is None or not result['numerical_reference']['passed']:
        # Retain the qualified arithmetic path when even the small exact-parent
        # offload comparison fails. This is explicit telemetry, never a pass.
        configure_checkpoint_input_offload(model.port,enabled=False)
        args.checkpoint_input_offload=False
        args.checkpoint_input_offload_directory=None
        if not args.checkpoint_layers:
            raise ValueError('noncheckpointed numerical qualification failed; exact-state checkpointed fallback required')
    if not result['passed']:
        if any(row['id'] in train.bindings for row in selected):
            raise ValueError('active source failed whole-graph memory qualification; exact-state fallback required')
        return result
    publish_capacity(result)
    return result


def combine_capacity_proofs(out, proofs, *, identity, code_sha, checkpoint_sha256):
    """Join actual case evidence from one unchanged restored parent, never add sizes."""
    from ..train.native_resident import binding, bound
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.jsonio import write_json_durable
    evidence=[];cases=[]
    for proof in proofs:
        value=json.loads(bound(proof).read_text())
        if (value.get('passed') is not True or value.get('identity')!=identity
                or value.get('runtime_code_sha256')!=code_sha
                or value.get('parent_checkpoint_sha256')!=checkpoint_sha256
                or value.get('optimizer_updates')!=0
                or (value.get('numerical_reference') or {}).get('passed') is not True
                or (value.get('capacity') or {}).get('passed') is not True):
            raise ValueError('combined capacity requires successful unchanged-parent runtime proofs')
        rows=(value.get('capacity') or {}).get('cases') or []
        if not rows or any(row.get('status')!='passed' or any(type(row.get(key)) is not int or row[key]<1
                for key in ('forward_tokens','kv_slots')) for row in rows):
            raise ValueError('combined capacity requires actual successful whole-graph cases')
        evidence.append(proof);cases.extend(rows)
    if not cases:raise ValueError('combined capacity requires actual cases')
    result={'schema':'natlang.native-memory-combined-capacity/1','passed':True,
        'identity':identity,'runtime_code_sha256':code_sha,'parent_checkpoint_sha256':checkpoint_sha256,
        'optimizer_updates':0,'proofs':evidence,'capacity':{'cases':cases}}
    directory=Path(out)/'memory-qualifications'/'combined'/canonical_json_sha256_hex(evidence)/checkpoint_sha256
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'proof.json';certificate=directory/'capacity.json'
    def immutable(target,value):
        if target.exists():
            if json.loads(target.read_text())!=value:raise ValueError('immutable combined capacity changed')
        else:write_json_durable(target,value)
    immutable(path,result)
    immutable(certificate,{'schema':'natlang.native-resident-geometry-qualification/1','passed':True,
        'identity':identity,'runtime_code_sha256':code_sha,
        'limits':{key:max(row[key] for row in cases) for key in ('forward_tokens','kv_slots')},
        'evidence':binding(path),'scope':'union of actually successful whole-graph cases from unchanged parent/runtime; no summed geometry'})
    return certificate


def _materialize_preference_capacity_request(source, objective, descriptor, *, identity,
                                              runtime_code_sha256, request_path):
    """Bind one prepared whole-pair capacity case to the already restored trainer.

    The descriptor selects an exact source pair and records its measured whole
    graph geometry. Runtime/checkpoint identity is deliberately supplied by the
    live trainer, never copied from an earlier ordinary-SFT certificate.
    """
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.jsonio import write_json_durable
    from ..train.native_resident import binding, _source_witness
    reference = descriptor.get('source')
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('preference capacity descriptor needs an exact source-manifest binding')
    if binding(source.path) != reference:
        raise ValueError('preference capacity descriptor source differs from the restored stage input')
    expected_objective = descriptor.get('objective')
    if not isinstance(expected_objective, dict) or expected_objective != objective.binding():
        raise ValueError('preference capacity descriptor objective differs from the restored stage objective')
    case = descriptor.get('case')
    if not isinstance(case, dict) or not isinstance(case.get('id'), str):
        raise ValueError('preference capacity descriptor needs one exact pair id')
    indices = [index for index, entry in enumerate(source.entries) if entry.get('id') == case['id']]
    if len(indices) != 1:
        raise ValueError('preference capacity pair id is absent or duplicated in the restored source')
    index = indices[0]
    entry = source.entries[index]
    objective_row = objective.rows.get(case['id'])
    if not isinstance(case.get('graph_sha256'), str) or not isinstance(objective_row, dict) or \
            objective_row.get('graph_sha256') != case['graph_sha256']:
        raise ValueError('preference capacity pair graph differs from the exact objective row')
    example = next(source.iter_indices([index]))
    actual_geometry = {'forward_tokens': example.plan.token_count,
                       'kv_slots': len(example.plan.slot_sources)}
    if case.get('geometry') != actual_geometry:
        raise ValueError('preference capacity pair geometry differs from the parsed whole graph')
    if example.split != 'train' or case.get('split') != 'train':
        raise ValueError('preference capacity case must be a complete training pair')
    objective.forward_inputs({'examples': [example]})
    if case['id'] not in objective.rows:
        raise ValueError('preference capacity case is absent from the exact objective rows')
    reference_case = descriptor.get('numerical_reference')
    if not isinstance(reference_case, dict) or not isinstance(reference_case.get('id'), str):
        raise ValueError('preference capacity descriptor needs one exact numerical-reference pair')
    reference_indices = [index for index, row in enumerate(source.entries)
                         if row.get('id') == reference_case['id']]
    if len(reference_indices) != 1:
        raise ValueError('preference numerical-reference id is absent or duplicated in the restored source')
    reference_index = reference_indices[0]
    reference_entry = source.entries[reference_index]
    reference_objective_row = objective.rows.get(reference_case['id'])
    if (not isinstance(reference_case.get('graph_sha256'), str)
            or not isinstance(reference_objective_row, dict)
            or reference_objective_row.get('graph_sha256') != reference_case['graph_sha256']):
        raise ValueError('preference numerical-reference graph differs from the exact objective row')
    reference_example = next(source.iter_indices([reference_index]))
    reference_geometry = {'forward_tokens': reference_example.plan.token_count,
                          'kv_slots': len(reference_example.plan.slot_sources)}
    if (reference_example.split != 'train' or reference_case.get('split') != 'train'
            or reference_case.get('geometry') != reference_geometry):
        raise ValueError('preference numerical reference must be an exact complete training pair')
    objective.forward_inputs({'examples': [reference_example]})
    if reference_case['id'] not in objective.rows:
        raise ValueError('preference numerical reference is absent from the exact objective rows')
    request = {
        'schema': 'natlang.native-memory-qualification-request/1',
        'identity': identity,
        'runtime_code_sha256': runtime_code_sha256,
        'objective': objective.binding(),
        'source': binding(source.path),
        'indexes': _source_witness(source),
        'split': 'train',
        'cases': [{'index': index, 'id': entry['id'], 'binding': entry['binding']}],
        'reference': {'index': reference_index, 'id': reference_entry['id'],
                      'binding': reference_entry['binding']},
    }
    request_path = Path(request_path)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('durable preference capacity request differs from this live restored stage')
    else:
        write_json_durable(request_path, request)
    return binding(request_path)


def _materialize_sft_capacity_request(source, descriptor, *, identity, runtime_code_sha256, request_path):
    """Select actual maximum complete SFT graphs from the authenticated current geometry sidecar."""
    from ..train.native_resident import binding, bound, _source_witness
    from ..train.native_packed_capacity import geometry_for_source
    from ..common.jsonio import write_json_durable
    if set(descriptor) != {'schema', 'source', 'geometry'}:
        raise ValueError('SFT capacity descriptor needs exactly source and geometry bindings')
    if descriptor['source'] != binding(source.path):
        raise ValueError('SFT capacity descriptor source differs from the restored training source')
    geometry = json.loads(bound(descriptor['geometry']).read_text())
    if geometry.get('source') != binding(source.path):
        raise ValueError('SFT capacity geometry source differs from the restored training source')
    # This verifier checks the parser policy, tokenizer, role IDs, complete ordered row bindings and every KV slot.
    slots, _, _ = geometry_for_source(source, Path(request_path).parent, prior=descriptor['geometry'])
    if not source.entries:
        raise ValueError('SFT capacity requires a nonempty training source')
    selected = list(dict.fromkeys([max(range(len(slots)), key=lambda i: source.entries[i]['forward_tokens']),
                                  max(range(len(slots)), key=lambda i: slots[i])]))
    reference = min(range(len(slots)), key=lambda i: source.entries[i]['forward_tokens'])
    def row(index):
        entry = source.entries[index]
        return {'index': index, 'id': entry['id'], 'binding': entry['binding']}
    for index, example in zip(selected, source.iter_indices(selected), strict=True):
        if (example.split != 'train' or example.plan.token_count != source.entries[index]['forward_tokens']
                or len(example.plan.slot_sources) != slots[index]):
            raise ValueError('SFT capacity maximum differs from the parsed training graph geometry')
    request = {'schema': 'natlang.native-memory-qualification-request/1', 'identity': identity,
               'runtime_code_sha256': runtime_code_sha256, 'source': binding(source.path),
               'indexes': _source_witness(source), 'split': 'train',
               'cases': [row(index) for index in selected], 'reference': row(reference)}
    request_path = Path(request_path)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError('durable SFT capacity request differs from this live restored stage')
    else:
        write_json_durable(request_path, request)
    return binding(request_path)


def _resolve_memory_qualification(reference, *, model, optimizer, train, roles, config, args,
                                  backend, objective):
    """Resolve a source-bound SFT or pair-case descriptor into the ordinary capacity request.

    All existing prepared requests pass through unchanged. This descriptor
    paths only construct request metadata after the actual objective, tokenizer,
    source index, restored configuration and backend are live.
    """
    from ..train.native_resident import bound, binding, code_inventory
    from ..common.hashing import canonical_json_sha256_hex
    descriptor_path = bound(reference)
    descriptor = json.loads(descriptor_path.read_text())
    schema = descriptor.get('schema')
    if schema not in ('natlang.native-preference-capacity-case/1', 'natlang.native-sft-capacity-cases/1'):
        return reference
    if schema == 'natlang.native-sft-capacity-cases/1' and objective is not None:
        raise ValueError('SFT capacity descriptor requires the ordinary SFT objective')
    if schema == 'natlang.native-preference-capacity-case/1' and (objective is None or getattr(objective, 'kind', None) not in {
            'native-offline-pairwise','native-failure-repair-dpo'}):
        raise ValueError('preference capacity case requires an implemented whole-pair preference objective')
    package = Path(__file__).resolve().parents[1]
    runtime_code_sha = canonical_json_sha256_hex(code_inventory(package))
    identity = capacity_identity(config, args, backend)
    request_key = canonical_json_sha256_hex({
        'descriptor_sha256': binding(descriptor_path)['sha256'],
        'identity': identity,
        'runtime_code_sha256': runtime_code_sha,
    })
    request_path = (Path(args.out) / 'memory-qualification-requests' /
                    descriptor_path.stem / request_key / 'request.json')
    if schema == 'natlang.native-sft-capacity-cases/1':
        return _materialize_sft_capacity_request(train, descriptor, identity=identity,
            runtime_code_sha256=runtime_code_sha, request_path=request_path)
    return _materialize_preference_capacity_request(train, objective, descriptor,
        identity=identity, runtime_code_sha256=runtime_code_sha, request_path=request_path)


def run_prepared(model,optimizer,train,roles,config,args,*,backend,checkpoint_sha256,objective=None):
    """Independent source cohorts share one restored parent and retain each pass.

    A novel larger cohort failing must not discard capacity already established
    for a smaller independently usable cohort. The supervisor can adopt each
    qualified source through the same resident transaction.
    """
    requests=json.loads(args.memory_qualification)
    if isinstance(requests,dict):
        original=args.memory_qualification
        request=_resolve_memory_qualification(requests,model=model,optimizer=optimizer,train=train,
            roles=roles,config=config,args=args,backend=backend,objective=objective)
        try:
            args.memory_qualification=json.dumps(request)
            return _run_one_prepared(model,optimizer,train,roles,config,args,
                backend=backend,checkpoint_sha256=checkpoint_sha256,objective=objective)
        finally:
            args.memory_qualification=original
    if not isinstance(requests,list) or not requests:
        raise ValueError('memory qualification needs a binding or nonempty ordered bindings')
    original=args.memory_qualification;checks=[];reference_reuse=_NumericalReferenceReuse()
    try:
        resolved=[]
        for request in requests:
            resolved.append(_resolve_memory_qualification(request,model=model,optimizer=optimizer,train=train,
                roles=roles,config=config,args=args,backend=backend,objective=objective))
        from ..train.native_resident import bound,binding
        canonical={};aliases=[]
        for requested,request in zip(requests,resolved):
            path=bound(request);actual=binding(path)
            if 'bytes' in request and path.stat().st_size!=request['bytes']:
                raise ValueError('memory qualification request byte count changed')
            key=actual['sha256']
            if key not in canonical:
                # Existing immutable evidence retains its original request path.
                # Reuse that checked identity across reordered/relocated aliases.
                proof_path=args.out/'memory-qualifications'/key/checkpoint_sha256/'proof.json'
                if proof_path.exists():
                    previous=json.loads(proof_path.read_text()).get('request')
                    prior=bound(previous)
                    if previous.get('sha256')!=key or prior.read_bytes()!=path.read_bytes():
                        raise ValueError('immutable memory qualification proof differs from its request')
                    canonical[key]=binding(prior)
                else:canonical[key]=actual
            canonical_binding=canonical[key]
            if bound(canonical_binding).read_bytes()!=path.read_bytes():
                raise ValueError('memory qualification aliases must have identical request bytes')
            aliases.append({'requested_binding':requested,'resolved_binding':request,'canonical_binding':canonical_binding})
        from ..train.native_resident import IndexWitnessValidation,check_index_witness_union
        index_validation=IndexWitnessValidation()
        check_index_witness_union([json.loads(bound(request).read_text())['indexes']
            for request in resolved],index_validation)
        for request,alias in zip(resolved,aliases):
            # Recheck each logical file immediately before using its canonical
            # request; no alias can disappear/change behind a cached proof.
            path=bound(request);canonical_request=alias['canonical_binding']
            if path.read_bytes()!=bound(canonical_request).read_bytes():
                raise ValueError('memory qualification alias changed before its check')
            args.memory_qualification=json.dumps(canonical_request)
            checks.append(_run_one_prepared(model,optimizer,train,roles,config,args,
                backend=backend,checkpoint_sha256=checkpoint_sha256,objective=objective,
                reference_reuse=reference_reuse,index_validation=index_validation))
            if 'numerical_reference' in checks[-1] and not (checks[-1]['numerical_reference'] or {}).get('passed'):
                break  # Failed exact-reference qualification cannot certify a later cohort.
    finally:
        args.memory_qualification=original
    passed=[check for check in checks if check['passed']]
    if len(passed)>1:
        from ..train.native_resident import binding
        proofs=list({check['request']['sha256']:binding(
            args.out/'memory-qualifications'/check['request']['sha256']/checkpoint_sha256/'proof.json')
            for check in passed}.values())
        args.resident_geometry_qualification=combine_capacity_proofs(args.out,proofs,
            identity=passed[0]['identity'],code_sha=passed[0]['runtime_code_sha256'],
            checkpoint_sha256=checkpoint_sha256)
    result={'schema':'natlang.native-memory-cohort-qualification/1',
        'passed':len(checks)==len(requests) and all(check['passed'] for check in checks),
        'request':requests,'request_aliases':aliases,'parent_checkpoint_sha256':checkpoint_sha256,'checks':checks,
        'index_witness_validation':{'strict_runs':index_validation.strict_runs,
            'strict_witnesses':index_validation.strict_witnesses,'unchanged_uses':index_validation.unchanged_uses},
        'optimizer_updates':0,'retained_capacity':str(args.resident_geometry_qualification)
            if args.resident_geometry_qualification else None}
    from ..common.jsonio import write_json_durable
    write_json_durable(args.out/'memory-qualification.json',result)
    return result


def policy_capacity_request(source,objective):
    """Exact whole-example maxima; the source parser remains the authority."""
    from ..train.native_resident import binding
    entries=[]
    for index,example in enumerate(source.iter_from(0)):
        entries.append({'index':index,'id':example.id,'binding':source.bindings[example.id],
            'forward_tokens':example.plan.token_count,'kv_slots':len(example.plan.slot_sources)})
    if not entries or set(objective.rows)!={row['id']for row in entries}:
        raise ValueError('policy capacity requires every admitted objective episode exactly once')
    chosen=sorted({max(range(len(entries)),key=lambda i:entries[i][key]) for key in ('forward_tokens','kv_slots')})
    return {'schema':'natlang.native-policy-capacity-request/1','source':binding(source.path),
        'objective':objective.binding(),'entries':entries,'max_example_indices':chosen}


def qualify_policy_objective(model,optimizer,train,objective,args,config,*,backend,checkpoint_sha256,
                             pending_batches,prior_proof=None):
    """The actual loaded learner runs its policy loss read-only before SGD."""
    from ..common.jsonio import write_json_durable
    from ..common.hashing import canonical_json_sha256_hex as digest
    from ..train.native_resident import binding,bound,code_inventory
    expected=policy_capacity_request(train,objective)
    supplied=getattr(args,'policy_capacity_request',None)
    path=bound(json.loads(supplied)) if supplied else args.out/'policy-capacity-request.json'
    if path.exists():
        if json.loads(path.read_text())!=expected:
            raise ValueError('policy capacity request differs from actual admitted graph/objective source')
    else:write_json_durable(path,expected)
    identity=capacity_identity(config,args,backend)
    code=digest(code_inventory(Path(__file__).resolve().parents[1]))
    # Qualify maximum whole examples AND actual packed pending batches. Packing
    # can use more memory than any individual episode; never split an episode.
    groups=[(i,)for i in expected['max_example_indices']]
    batches=sorted({tuple(batch[1])for batch in pending_batches})
    if batches:
        groups += [max(batches,key=lambda ids:sum(expected['entries'][i][key]for i in ids))
                   for key in ('forward_tokens','kv_slots')]
    groups=list(dict.fromkeys(groups))
    limits={key:max(sum(expected['entries'][i][key]for i in group)for group in groups)
            for key in ('forward_tokens','kv_slots')}
    scope={'request':binding(path),'identity':identity,'runtime_code_sha256':code}
    output=args.out/'policy-capacity'/digest({**scope,'packing_limits':limits})/'proof.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    # A saved passed proof can cover smaller remaining batches after a stop.
    # Larger/new packing receives its own actual check, never an inherited claim.
    candidate=Path(prior_proof['path']) if prior_proof is not None else output
    embedded=prior_proof.get('proof') if prior_proof is not None else None
    if embedded is not None and digest(embedded)!=prior_proof.get('proof_sha256'):
        raise ValueError('saved policy capacity proof payload changed')
    if candidate.exists() or embedded is not None:
        if candidate.exists():
            if prior_proof is not None:bound(prior_proof)
            previous=json.loads(candidate.read_text())
            if embedded is not None and previous!=embedded:
                raise ValueError('saved policy capacity proof differs from durable evidence')
        else:previous=embedded
        if any(previous.get(key)!=value for key,value in scope.items()):
            if candidate==output:raise ValueError('policy capacity proof changed its restored objective identity')
        elif all(previous.get('packing_limits',{}).get(key,-1)>=value for key,value in limits.items()):
            if previous.get('passed') is not True:
                raise ValueError('policy capacity has no passed proof for this exact restored objective')
            if previous['parent_checkpoint_sha256']!=checkpoint_sha256 and prior_proof is None:
                raise ValueError('policy capacity proof lacks its authenticated saved continuation')
            return {**({'path':str(candidate.resolve()),'sha256':prior_proof['sha256']} if not candidate.exists() else binding(candidate)),
                'proof':previous,'proof_sha256':digest(previous)}
    examples=[list(train.iter_indices(list(indices)))for indices in groups]
    result={'schema':'natlang.native-policy-capacity-proof/1',**scope,
        'parent_checkpoint_sha256':checkpoint_sha256,'passed':False,'optimizer_updates':0,
        'earned_exposure':0,'packing_limits':limits,'case_indices':[list(group)for group in groups],
        'scope':'actual admitted whole policy examples and pending production packed-batch maxima; no task-quality claim'}
    try:
        result['capacity']=qualify_capacity(model,optimizer,examples,backend=backend,
            checkpoint_layers=args.checkpoint_layers,chunk_tokens=args.chunk_tokens,
            checkpoint_sha256=checkpoint_sha256,source_binding=expected['source'],objective=objective)
        if not result['capacity']['passed']:
            raise RuntimeError('actual policy objective exceeded its learner capacity')
        if args.fp8_training!='off':
            counts=result['capacity']['executed_projections']
            if not (counts.get('calls',0)>0 and counts.get('rows',0)>0):
                raise ValueError('policy capacity requires actual configured FP8 projection execution')
        result['passed']=True
    except BaseException as error:
        result['error']=str(error);write_json_durable(output,result);raise
    write_json_durable(output,result)
    return {**binding(output),'proof':result,'proof_sha256':digest(result)}
