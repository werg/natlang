"""Actual-parent whole-graph memory evidence without SGD or full gradient copies."""
from __future__ import annotations

import contextlib
import hashlib
import json
import copy
import math
import threading
import time
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
                     chunk_tokens=1024, checkpoint_sha256, source_binding, gradient_observer=None):
    """Cases are whole, source-authenticated graphs chosen by the supervisor.

    The existing gated optimizer hooks stay installed but unarmed. Temporary
    observation hooks release each gradient exactly where production consumes
    it; no optimizer method, parameter write or full-model gradient copy runs.
    CUDA lifetime peaks are not reset. This is capacity, not task qualification.
    """
    from ..train.native_state_sft import packed_graph_loss
    if not cases:
        raise ValueError('capacity qualification needs actual complete source graphs')
    named = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
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
    parameter_versions = {name:p._version for name,p in model.named_parameters()}
    optimizer_versions = _optimizer_versions(optimizer)
    extra = copy.deepcopy(backend.extra_state()) if backend is not None else None
    counts = backend.collect_row_counts() if backend is not None else None
    executed = getattr(backend,'has_executed',False)
    rows, hooks, observed = [], [], {}
    def capture(parameter, *, name):
        gradient=parameter.grad
        if gradient is None:return
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
            row={'id':example.id,'forward_tokens':example.plan.token_count,
                 'kv_slots':len(example.plan.slot_sources),'status':'running'}
            rows.append(row)
            started=time.monotonic()
            try:
                with _HostPeak() as host:
                    with backend.autocast() if backend is not None else contextlib.nullcontext():
                        loss,_=packed_graph_loss(model,[example],checkpoint_layers=checkpoint_layers,
                            chunk_tokens=chunk_tokens,share_causal_prefixes=getattr(model,'share_causal_prefixes',False))
                    loss.backward()
                    if device.type=='cuda':torch.cuda.synchronize(device)
                if not observed or not math.isfinite(float(loss.detach())):
                    raise FloatingPointError('capacity check has no actual finite gradient/loss')
                row.update(status='passed',loss=float(loss.detach()),gradients=dict(observed),
                    host_peak=host.peak)
            except torch.cuda.OutOfMemoryError as error:
                row.update(status='out_of_memory',error=str(error),host_peak=host.peak)
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
        if parameter_versions!={name:p._version for name,p in model.named_parameters()}:
            raise CapacityStateChangedError('capacity check changed a model parameter')
        if optimizer_versions!=_optimizer_versions(optimizer):
            raise CapacityStateChangedError('capacity check changed optimizer state or rates')


def offload_numerical_parity(model, optimizer, example, *, backend=None, chunk_tokens=1024,
                             checkpoint_sha256, source_binding):
    """Bitwise full named-gradient parity, retaining only per-parameter digests."""
    from ..common.checkpoint_offload import configure_checkpoint_input_offload
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
            checkpoint_sha256=checkpoint_sha256,source_binding=source_binding,gradient_observer=reference)
        configure_checkpoint_input_offload(model.port,enabled=True)
        candidate=qualify_capacity(model,optimizer,[example],backend=backend,chunk_tokens=chunk_tokens,
            checkpoint_sha256=checkpoint_sha256,source_binding=source_binding,gradient_observer=compare)
        return {'passed':baseline['passed'] and candidate['passed'] and not mismatches
                    and seen==gradients.keys() and baseline['cases'][0].get('loss')==candidate['cases'][0].get('loss'),
                'baseline':baseline,'offloaded':candidate,'gradient_parameters':len(gradients),
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
    return {'model_config':config,'batch_graphs':args.batch_graphs,'batch_tokens':args.batch_tokens,
        'checkpoint_layers':args.checkpoint_layers,'chunk_tokens':args.chunk_tokens,
        'checkpoint_input_offload':args.checkpoint_input_offload,'runtime':memory_runtime(args.device),
        'fp8':{'kind':args.fp8_training,'recipe':args.fp8_recipe,'amax_history_len':args.fp8_amax_history_len,
               'input_gradient_precision':getattr(backend,'input_gradient_precision',None)}}


def run_prepared(model,optimizer,train,roles,config,args,*,backend,checkpoint_sha256):
    """Called once at the normal restore boundary, before any new optimizer step.

    Failed novel source checks cannot stop useful SGD on the existing source.
    A failed check on a source already made active is an explicit blocking error.
    """
    from ..train.native_resident import bound,binding,code_inventory,_check_indexes
    from ..common.hashing import canonical_json_sha256_hex
    from ..common.jsonio import write_json_durable
    from ..data.native_sources import NativeGraphSource
    from ..common.checkpoint_offload import configure_checkpoint_input_offload
    if not args.checkpoint_input_offload:
        raise ValueError('input-offload capacity request requires the declared offload mode')
    request_path=bound(json.loads(args.memory_qualification))
    request=json.loads(request_path.read_text())
    package=Path(__file__).resolve().parents[1]
    code_sha=canonical_json_sha256_hex(code_inventory(package))
    identity=capacity_identity(config,args,backend)
    if (request.get('schema')!='natlang.native-memory-qualification-request/1'
            or request.get('identity')!=identity or request.get('runtime_code_sha256')!=code_sha):
        raise ValueError('memory qualification request differs from actual restored model/runtime/packing')
    _check_indexes(request['indexes'])
    source=NativeGraphSource(bound(request['source']),config['tokenizer_fingerprint'],
        required_split='train',pair_model=request.get('pair_model'),role_ids=roles.ids)
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
        if any(row['id'] in train.bindings for row in selected):
            raise ValueError('active source still has a failed whole-graph memory qualification')
        return previous
    try:
        example=next(train.iter_indices([ref_index]))
        result['numerical_reference']=offload_numerical_parity(model,optimizer,example,backend=backend,
            chunk_tokens=args.chunk_tokens,checkpoint_sha256=checkpoint_sha256,source_binding=binding(train.path))
        if result['numerical_reference']['passed']:
            examples=list(source.iter_indices([row['index'] for row in selected]))
            result['capacity']=qualify_capacity(model,optimizer,examples,backend=backend,chunk_tokens=args.chunk_tokens,
                checkpoint_layers=args.checkpoint_layers,checkpoint_sha256=checkpoint_sha256,source_binding=request['source'])
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
    if not result['passed']:
        if any(row['id'] in train.bindings for row in selected):
            raise ValueError('active source failed whole-graph memory qualification; exact-state fallback required')
        return result
    publish_capacity(result)
    return result
