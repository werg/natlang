"""Actual whole-pair retained-gradient capacity at semantic phase boundaries."""
from __future__ import annotations
import time
import torch

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from ..eval.native_memory import _HostPeak
from .native_semantic_boundary import ObjectiveBoundary
from .native_semantic import parse_pair, consumer_example, joint_loss
from .native_semantic_gradients import SemanticObjectiveScope, qualify_joint_gradients



def geometry(row, participants):
    """All retained producer, payload, import and consumer rows per model."""
    parsed=parse_pair(row,participants)
    dimensions=[]
    for name in sorted(participants):
        if name not in parsed:
            dimensions.extend((0,)*6);continue
        example,view=parsed[name]
        payload=len(view['export']['offsets'])
        consumer,_=consumer_example(example,view,payload)
        dimensions.extend((example.plan.token_count,len(example.plan.slot_sources),
                           payload+2,payload,consumer.plan.token_count,len(consumer.plan.slot_sources)))
    return tuple(dimensions)


def capacity_frontier(source, participants):
    """Stream once; retain actual Pareto cases and each participant subset.

    This is geometry coverage, not a mathematical memory bound: every frontier
    case still executes the actual full objective and retained-gradient backward.
    """
    frontier=[]; representatives={};count=0
    for row in source.rows():
        if canonical_json_sha256_hex(row)!=source.bindings[row['id']]:
            raise ValueError('semantic capacity source row changed after admission')
        active=tuple(sorted(set(row['models'])&set(participants)))
        if len(active)<2:continue
        dimensions=geometry(row,participants);count+=1
        representatives.setdefault(active,row)
        if any(all(a>=b for a,b in zip(old,dimensions)) for old,_ in frontier):continue
        frontier=[(old,case) for old,case in frontier if not all(a>=b for a,b in zip(dimensions,old))]
        frontier.append((dimensions,row))
    if sha256_file_hex_cached(source.path)!=source.sha256:
        raise ValueError('semantic capacity source changed during geometry scan')
    if not frontier:raise ValueError('semantic capacity source has no ready model pair')
    return frontier,representatives,count


def qualify_capacity(vae, participants, row, *, phase, options, optimizer, fetch_external, device):
    """Match production: keep ALL gradients until whole backward and clipping."""
    boundary=ObjectiveBoundary(participants,vae,optimizer,device)
    started=time.monotonic()
    cuda=torch.device(device).type=='cuda'
    try:
        if cuda:torch.cuda.synchronize(device)
        with _HostPeak() as peak:
            with SemanticObjectiveScope(participants,vae) as scope:
                loss,_=joint_loss(vae,participants,row,phase=phase,options=options,fetch_external=fetch_external)
                loss.backward();scope.finish()
            parameters=[p for _,p in boundary.named if p.requires_grad and p.grad is not None]
            norm=torch.nn.utils.clip_grad_norm_(parameters,options.get('max_grad_norm',float('inf')))
            if cuda:torch.cuda.synchronize(device)
            finite=bool(torch.isfinite(loss.detach()) and torch.isfinite(norm))
            proof={'schema':'natlang.native-semantic-whole-pair-capacity/1','passed':finite,
                'phase':phase,'row_sha256':canonical_json_sha256_hex(row),
                'whole_pair_geometry':list(geometry(row,participants)),
                'loss':float(loss.detach()),'gradient_norm':float(norm),
                'retained_gradient_bytes':sum(p.grad.numel()*p.grad.element_size() for p in parameters),
                'gradient_lifetime':'all participant and VAE gradients retained through whole backward and clipping',
                'host_peak':peak.peak,'seconds':time.monotonic()-started,
                'cuda_lifetime_peak_allocated_bytes':torch.cuda.max_memory_allocated(device) if cuda else 0,
                'cuda_lifetime_peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if cuda else 0,
                'optimizer_updates':0,'exposure_updates':0,'numerical_measurement_performed':True}
        return proof
    finally:boundary.restore()


def ensure_phase_qualification(vae, participants, source, *, phase, options, optimizer,
                               fetch_external, device, training, out, cache):
    """Checkpoint-carried phase evidence; adaptive new profiles qualify anew."""
    from pathlib import Path
    files=[Path(__file__),Path(__file__).with_name('native_semantic_boundary.py'),
           Path(__file__).with_name('native_semantic_stage.py'),Path(__file__).with_name('native_semantic_gradients.py'),
           Path(__file__).with_name('native_semantic.py')]
    from ..model import native_weight_carrier
    from ..eval import native_gradients
    files.extend([Path(native_weight_carrier.__file__),Path(native_gradients.__file__)])
    from .native_resident import code_inventory
    package=Path(__file__).resolve().parents[1]
    runtime_key=('runtime_code',str(package))
    if runtime_key not in cache:cache[runtime_key]=canonical_json_sha256_hex(code_inventory(package))
    execution_names=('native_shared_execution','native_norm_execution','native_linear_precision',
        'native_attention_precision','natlang_precision','native_cotangent_accumulator_backend',
        'ffn_chunk_tokens','checkpoint_layers','share_causal_prefixes')
    execution={name:{'config':entry.config,
        'settings':[{key:getattr(owner,key) for key in execution_names if hasattr(owner,key)}
                    for owner in (entry.model,entry.model.port,entry.model.port.hf)],
        'parameters':[(key,list(p.shape),str(p.dtype),p.requires_grad) for key,p in entry.model.named_parameters()]}
        for name,entry in sorted(participants.items())}
    actual_device=torch.device(device)
    numerical_runtime={'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':actual_device.type,
        'hardware':torch.cuda.get_device_name(actual_device) if actual_device.type=='cuda' else 'cpu',
        'capability':list(torch.cuda.get_device_capability(actual_device)) if actual_device.type=='cuda' else None,
        'device_memory_bytes':torch.cuda.get_device_properties(actual_device).total_memory if actual_device.type=='cuda' else None,
        'float32_matmul':torch.get_float32_matmul_precision(),
        'cuda_matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}
    invocation={'runtime_code_sha256':cache[runtime_key],'numerical_runtime':numerical_runtime,
        'execution':execution,'phase':phase,'options':options,'source':source.binding,
        'participants':{name:entry.identity for name,entry in sorted(participants.items())},
        'trainables':{name:[(key,list(p.shape),str(p.dtype)) for key,p in entry.model.named_parameters() if p.requires_grad]
                      for name,entry in sorted(participants.items())},
        'backends':{name:entry.backend.configuration() if entry.backend is not None else None
                    for name,entry in sorted(participants.items())},
        'vae':vae.specification(),'code':{str(path.name):sha256_file_hex_cached(path) for path in files}}
    key=canonical_json_sha256_hex(invocation)
    completed=training.setdefault('phase_qualifications',{})
    if completed.get(key,{}).get('passed') is True:return completed[key]
    frontier_key=canonical_json_sha256_hex({'source':source.binding,'participants':sorted(participants)})
    if frontier_key not in cache:cache[frontier_key]=capacity_frontier(source,participants)
    frontier,representatives,count=cache[frontier_key]
    numerical=[];capacities=[]
    proof_directory=Path(out)/'phase-qualifications'
    proof_directory.mkdir(parents=True,exist_ok=True)
    progress_path=proof_directory/(key+'.progress.json')
    def progress(status,**extra):
        write_json_durable(progress_path,{'schema':'natlang.native-semantic-qualification-progress/1',
            'profile_sha256':key,'phase':phase,'status':status,'entered_at_step':training['step'],
            'gradient_cases':len(numerical),'capacity_cases':len(capacities),
            'complete_gradients':numerical,'whole_pair_capacity':capacities,'profile':invocation,
            'required_gradient_cases':len(representatives),'required_capacity_cases':len(frontier),
            'optimizer_updates':0,'exposure_updates':0,**extra})
    progress('qualifying')
    boundary=ObjectiveBoundary(participants,vae,optimizer,device)
    try:
        for active,row in sorted(representatives.items()):
            selected={name:participants[name] for name in active}
            numerical.append(qualify_joint_gradients(vae,selected,row,phase=phase,options=options,
                fetch_external=fetch_external,atol=options.get('gradient_atol',1e-5),
                rtol=options.get('gradient_rtol',1e-4),device=device,optimizer=optimizer))
            progress('qualifying',last_gradient=numerical[-1])
            if numerical[-1]['passed'] is not True:raise RuntimeError('complete semantic objective gradient qualification failed')
        for _,row in frontier:
            selected={name:participants[name] for name in sorted(set(row['models'])&set(participants))}
            capacities.append(qualify_capacity(vae,selected,row,phase=phase,options=options,
                optimizer=optimizer,fetch_external=fetch_external,device=device))
            progress('qualifying',last_capacity=capacities[-1])
            if capacities[-1]['passed'] is not True:raise RuntimeError('whole-pair semantic capacity qualification failed')
    except BaseException as error:
        progress('failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:boundary.restore()
        except BaseException as error:
            progress('failed',error=f'{type(error).__name__}: {error}')
            raise
    proof={'schema':'natlang.native-semantic-phase-qualification/1','passed':True,
        'profile':invocation,'profile_sha256':key,'entered_at_step':training['step'],
        'source_rows':count,'frontier_cases':len(frontier),'complete_gradients':numerical,
        'whole_pair_capacity':capacities,'optimizer_updates':0,'exposure_updates':0}
    write_json_durable(proof_directory/(key+'.json'),proof)
    progress('passed')
    completed[key]=proof
    return proof
