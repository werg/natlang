"""Resumable native coefficient updates through the shared graph and optimizer.

All actor adjoints share one current graph. Policy/epoch/evaluation choices are
explicit declarations; this module commits their exact results and credits.
"""
from __future__ import annotations
import copy
from contextlib import nullcontext
import json
from pathlib import Path
import torch

from ..common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from ..model.native_composite import resolve_composite_artifact
from ..model.native_coefficient_policy import policy_core,restore_policy,publish_policy,runtime_binding,actor_policy_sha256,write_training_state
from ..model.native_adapters import optimize_adapter
from ..serve.native_grad import native_coefficient_gradients
from ..train.loop import capture_training_rng_state,restore_training_rng_state
from ..train.trajectory_state import atomic_checkpoint
from .native_replay import NativePolicyObjective

SCHEMA='natlang.native-coefficient-training/1'


def _binding(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex_cached(path)}


def bind_policy(engine,path):
    policy=restore_policy(engine,path)
    engine.native_coefficient_policy_runtime=runtime_binding(policy,path)
    engine.rl_policy_binding={**engine.rl_policy_binding,'coefficient_policy_sha256':actor_policy_sha256(policy),
                              'coefficient_policy':engine.native_coefficient_policy_runtime}
    return policy


def train(engine,**kwargs):
    """One owner of this actual output slot, including direct CLI/recipe runs."""
    import fcntl
    directory=Path(kwargs['out']).resolve();directory.mkdir(parents=True,exist_ok=True)
    with (directory/'learner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _train_locked(engine,**kwargs)


def _prune_previous_private_policy(previous,current,*,out,protected,qualified):
    """Delete only this learner's superseded unpublished policy and sidecar."""
    if qualified:return
    previous,current=Path(previous),Path(current)
    if previous==current or previous in protected or previous.parent.parent!=out/'policies':return
    old=json.loads(previous.read_text())
    frame=resolve_composite_artifact(old['training_state'],previous) if old.get('training_state') else None
    paths=[previous.parent]
    current_frame=resolve_composite_artifact(json.loads(current.read_text())['training_state'],current)
    if frame is not None and frame.parent==out/'training-state' and frame!=current_frame:
        paths.append(frame)
    # Owner storage policy requires listing exact owned paths before deletion.
    write_json_durable(out/'pruned-private-policy.json',{'schema':'natlang.native-coefficient-prune/1',
        'reason':'superseded private intermediate after durable whole-policy checkpoint/report commit',
        'retained_policy':str(current),'paths':[str(path) for path in paths]})
    import shutil
    for path in paths:
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink()


def _train_locked(engine,*,policy,graphs,objective,out,epochs,optimizer,evaluate,stop=None,backend=None,after_commit=None,evaluate_every=None,held_groups=(),held_row_bindings=None,held_receipts=None):
    """Commit whole graph updates; resume every owed row without replaying credit.

    `evaluate(engine, policy_path, final=...)` executes declared actual held/task
    checks. Its result is recorded separately from exact training completion.
    Production callers must supply it; there is no success stand-in.
    """
    if (type(epochs) is not int or epochs<1 or not callable(evaluate)
            or evaluate_every is not None and (type(evaluate_every) is not int or evaluate_every<1)):
        raise ValueError('coefficient training requires positive whole epochs and actual evaluation')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    objective=Path(objective).resolve(strict=True);admitted=NativePolicyObjective(objective)
    if admitted.behavior_policy is None:
        raise ValueError('coefficient training requires its versioned actual invocation behavior policy')
    if admitted.identity!=engine.native_identity:
        raise ValueError('coefficient training objective changed the exact native base')
    slot=out/'checkpoint.pt'
    source_policy=Path(policy).resolve(strict=True)
    source_binding=_binding(source_policy)
    if slot.exists():
        state=torch.load(slot,map_location='cpu',weights_only=False)
        if state.get('schema')!=SCHEMA or state['base_native_identity']!=engine.native_identity:
            raise ValueError('coefficient learner checkpoint belongs to another exact native base')
        if source_binding not in (state['initial_policy'],state['policy']):
            raise ValueError('coefficient learner source policy changed outside its exact continuation')
        resolve_composite_artifact(state['policy'],slot)
        current=restore_policy(engine,state['policy']['path'])
    else:
        current=restore_policy(engine,source_policy)
        if current.get('training_state') is not None:
            frame=resolve_composite_artifact(current['training_state'],source_policy)
            state=torch.load(frame,map_location='cpu',weights_only=False)
            if state.get('schema')!=SCHEMA or state['base_native_identity']!=engine.native_identity or 'policy' in state:
                raise ValueError('portable coefficient progress changed its exact base or contains a cyclic policy binding')
            state['prior_stage_policy_sha256']=state['initial_policy']['sha256']
            state['initial_policy']=source_binding
            state['policy']=source_binding
            state['stage_start_tokens']=state.get('trained_tokens',sum(item['sampled_tokens'] for item in state.get('metrics',[])))
            state['stage_start_step']=state['step']
        else:
            state={'schema':SCHEMA,'base_native_identity':dict(engine.native_identity),'initial_policy':source_binding,
                'policy':source_binding,'step':0,'obligations':[],'cursor':0,'handle_versions':{},'trained_tokens':0,'stage_start_tokens':0,'stage_start_step':0}
            if (getattr(engine,'native_training_state',None) or {}).get('rng') is not None:
                state['rng']=engine.native_training_state['rng']
    state.setdefault('trained_tokens',sum(item['sampled_tokens'] for item in state.get('metrics',[])))
    state.pop('metrics',None)
    if state.get('rng') is not None:restore_training_rng_state(state['rng'],engine.device)
    if backend is not None and state.get('fp8') is not None:backend.restore_extra_state(state['fp8'])
    elif state.get('fp8') is not None:
        raise ValueError('coefficient learner must restore its actual declared FP8 backend')
    if held_row_bindings is not None:
        previous=state.get('held_row_bindings',{})
        if any(held_row_bindings.get(key)!=value for key,value in previous.items()):
            raise ValueError('coefficient held continuation must retain every exact previous protected row')
        state['held_row_bindings']=dict(held_row_bindings)
        state['held_source_receipts']=held_receipts
    core=policy_core(current)
    restored_objectives=set()
    # Bind every version actually admitted at this continuation boundary. Later
    # updates retain these source IDs while replacing current coefficients.
    for handle,entry in core['handles'].items():
        for ident in (handle,entry['current_id']):
            previous=state['handle_versions'].setdefault(ident,handle)
            if previous!=handle:raise ValueError('coefficient source ID aliases two different declared handles')
    from ..data.native_graph import parse_graph
    from .native_replay import graph_identity
    by_key={entry['key']:entry for entry in state['obligations']}
    for graph in graphs:
        example=parse_graph(graph,tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint'])
        row=admitted.rows.get(example.id)
        if (example.split!='train' or row is None
                or row['graph_sha256']!=canonical_json_sha256_hex(graph_identity(example))):
            raise ValueError('coefficient training requires exact admitted train-only graphs')
        key=canonical_json_sha256_hex({'graph':graph,'objective_row':row})
        if key in by_key:
            if by_key[key].get('cancelled'):
                raise ValueError('cancelled old-base behavior cannot be re-admitted; recollect under the actual current policy')
            by_key[key]['required_visits']=max(by_key[key]['required_visits'],epochs)
            by_key[key]['graph']=copy.deepcopy(graph)
            by_key[key]['objective']=_binding(objective)
        else:
            entry={'key':key,'graph':copy.deepcopy(graph),'objective':_binding(objective),
                   'required_visits':epochs,'visits':0,'source_groups':list(graph['source_groups']),'graph_id':graph['id']}
            state['obligations'].append(entry);by_key[key]=entry
    protected=set(held_groups)
    if any(protected.intersection(entry['source_groups']) for entry in state['obligations']):
        raise ValueError('native coefficient train/held source groups overlap, including previous portable exposure')
    if not graphs or not state['obligations']:
        raise ValueError('coefficient training has no admitted whole-trajectory rows')

    def save():
        previous=Path(state['policy']['path'])
        previous_report=out/'report.json'
        published=json.loads(previous_report.read_text()) if previous_report.exists() else {}
        previous_qualified=published.get('qualification_complete') is True and published.get('policy')==state['policy']
        state['rng']=capture_training_rng_state(engine.device)
        state['fp8']=backend.extra_state() if backend is not None else None
        # Completed trajectories compact to exact source/objective identities;
        # owed rows keep their complete graph for autonomous continuation.
        for entry in state['obligations']:
            if entry['visits']>=entry['required_visits']:
                entry.pop('graph',None)
        state['actor_policy_sha256']=actor_policy_sha256(core)
        frame={key:value for key,value in state.items() if key!='policy'}
        core['training_state']=write_training_state(frame,out/'training-state')
        policy_path=publish_policy(engine,core,out/'policies')
        state['policy']=_binding(policy_path)
        atomic_checkpoint(slot,state)
        counts={'episodes':sum(item['visits'] for item in state['obligations']),
                'required_episodes':sum(item['required_visits'] for item in state['obligations'])}
        cancelled=sum(item['required_visits']-item['visits'] for item in state['obligations'] if item.get('cancelled'))
        complete=all(item.get('cancelled') or item['visits']>=item['required_visits'] for item in state['obligations'])
        counts.update(cancelled_uncredited_episodes=cancelled,original_obligations_complete=counts['episodes']==counts['required_episodes'])
        report={'schema':'natlang.native-coefficient-training-report/1','completed':complete,
            'stage_kind':'native-coefficient-policy','base_checkpoint_unchanged':True,'step':state['step'],
            'checkpoint':_binding(slot),'policy':state['policy'],'coefficient_policy_sha256':state['actor_policy_sha256'],'exposure':counts,
            'source_rows':[{'key':item['key'],'visits':item['visits'],'required_visits':item['required_visits'],**({'cancelled':item['cancelled']} if item.get('cancelled') else {})} for item in state['obligations']],
            'evaluation':state.get('evaluation'),'qualification_complete':bool(complete and state.get('evaluation',{}).get('passed') is True
                and state['evaluation'].get('final') is True and state['evaluation'].get('coefficient_policy_sha256')==state['actor_policy_sha256']),
            'train_exposure':{key:state['trained_tokens']-state['stage_start_tokens'] for key in ('supervised_tokens','target_tokens')},
            'cumulative_train_exposure':{key:state['trained_tokens'] for key in ('supervised_tokens','target_tokens')},
            'this_stage_updates':state['step']-state['stage_start_step']}
        write_json_durable(out/'report.json',report)
        _prune_previous_private_policy(previous,policy_path,out=out,
            protected={source_policy,Path(state['initial_policy']['path'])},qualified=previous_qualified)
        return report
    def measure(final):
        saved=capture_training_rng_state(engine.device)
        try:
            path=Path(state['policy']['path']);bind_policy(engine,path)
            with backend.native_modules() if backend is not None else nullcontext():
                observed=evaluate(engine,path,final=final,coverage=state.get('evaluation_coverage'))
            if not isinstance(observed,dict) or type(observed.get('passed')) is not bool:
                raise ValueError('coefficient evaluation must return its actual typed result, not implied success')
            if 'held_coverage' in observed:state['evaluation_coverage']=observed['held_coverage']
            state['evaluation']={**observed,'final':final,'coefficient_policy_sha256':state['actor_policy_sha256']}
        finally:restore_training_rng_state(saved,engine.device)
        return save()
    save()  # source/epoch additions are durable even if stopped before an update
    while any(not item.get('cancelled') and item['visits']<item['required_visits'] for item in state['obligations']):
        if stop is not None and stop.requested:return save()
        size=len(state['obligations']);start=state['cursor']%size
        index=next((start+offset)%size for offset in range(size)
                   if not state['obligations'][(start+offset)%size].get('cancelled') and state['obligations'][(start+offset)%size]['visits']<state['obligations'][(start+offset)%size]['required_visits'])
        entry=state['obligations'][index]
        resolve_composite_artifact(entry['objective'],slot)
        objective_state=NativePolicyObjective(entry['objective']['path'])
        row=objective_state.rows[entry['graph']['id']]
        if entry['objective']['sha256'] not in restored_objectives:
            from .graph_replay import restore_bundle
            restore_bundle(engine,objective_state.bundle[0],policy_id=objective_state.archive_owner)
            restored_objectives.add(entry['objective']['sha256'])
        bindings={}
        active={}
        for name,actor in row['invocation_actors']['invocations'].items():
            values=[]
            for binding in (actor['adapter_binding'] or {}).get('adapters',[]):
                handle=state['handle_versions'].get(binding['id'])
                ident=core['handles'][handle]['current_id'] if handle is not None else binding['id']
                values.append({'id':ident,'scale':float.fromhex(binding['scale_hex'])})
                if handle is not None:active[ident]=handle
            bindings[name]=values
        if not active:
            raise ValueError('admitted graph has no declared trainable coefficient actor; retain it for backbone training instead')
        targets=[{'id':ident,'names':core['handles'][handle]['parameter_names']} for ident,handle in active.items()]
        body={'schema':'natlang.native-gradient-request/1','native_identity':engine.native_identity,
            'graph':entry['graph'],'invocation_adapters':bindings,'adapter_gradient_precision':'fp32',
            'objective':{'kind':'native-policy',**entry['objective']}}
        if core['state_transport']=='same-base-adapter-state':
            body['native_state_modes']={binding['id']:'same-base-adapter-state' for binding in entry['graph'].get('external_states',{}).values()}
        before_rng=capture_training_rng_state(engine.device)
        try:
            with backend.autocast() if backend is not None else nullcontext():
                result=native_coefficient_gradients(engine,body,targets)
            next_core=copy.deepcopy(core)
            for meta in result['gradients']:
                gradient=engine.store.get_native_tensor_artifact(meta['id']).artifact
                handle=active[gradient.source_artifact_id]
                previous=core['handles'][handle]
                update=optimize_adapter(engine,gradient,optimizer,previous['optimizer_state_id'])
                next_core['handles'][handle]['current_id']=update['adapter']['id']
                next_core['handles'][handle]['optimizer_state_id']=update['optimizer_state_id']
        except BaseException:
            restore_training_rng_state(before_rng,engine.device)
            raise
        core=next_core;state['step']+=1
        entry['visits']+=1;state['cursor']=index+1
        state['trained_tokens']+=len(row['actions'])
        state['last_metric']={'step':state['step'],'row':entry['key'],'loss':result['loss'],
                              'sampled_tokens':len(row['actions']),'actor_updates':len(active)}
        state.pop('evaluation',None)
        save()
        if after_commit is not None:after_commit(state)
        if evaluate_every and state['step']%evaluate_every==0 and not (stop is not None and stop.requested):measure(False)
    if stop is not None and stop.requested:return save()
    return measure(True)


def main(argv=None):
    import argparse
    from types import SimpleNamespace
    from ..common.jsonio import iter_jsonl_rows
    from ..serve import load_native_engine
    from ..serve.store import TensorStore
    from ..eval.task_cadence import TaskEval,parse_task_eval,task_gate
    from ..data.native_sources import NativeGraphSource
    from ..train.native_state_sft import _evaluate_cadence
    from ..model.native_adapters import run_native
    from ..train.loop import StopSignal
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coefficient-policy',type=Path,required=True)
    parser.add_argument('--graphs',type=Path,required=True)
    parser.add_argument('--objective',type=Path,required=True)
    parser.add_argument('--continue-from',type=Path,required=True)
    parser.add_argument('--held-native-graphs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--base',type=Path)
    parser.add_argument('--native-participant')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--coefficient-optimizer',type=json.loads,required=True)
    parser.add_argument('--task-eval',required=True)
    parser.add_argument('--task-eval-every',type=int,required=True)
    parser.add_argument('--held-eval-graphs',type=int,default=0)
    parser.add_argument('--held-eval-tokens',type=int,default=0)
    parser.add_argument('--chunk-tokens',type=int,default=128)
    parser.add_argument('--checkpoint-layers',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--fp8-training',dest='fp8_backend',choices=('off','triton-experts'),default='off')
    parser.add_argument('--fp8-recipe',choices=('delayed','current','current-bf16-dx'),default='delayed')
    args=parser.parse_args(argv)
    stop=StopSignal().install()
    engine=None
    try:
        engine=load_native_engine(str(args.continue_from),base=str(args.base) if args.base else None,
            device=args.device,participant=args.native_participant)
        args.out.mkdir(parents=True,exist_ok=True)
        engine.store.close();engine.store=TensorStore(args.out/'actor-store')
        backend=None
        if args.fp8_backend!='off':
            from ..train.fp8_backend import make_backend
            backend=make_backend(args.fp8_backend,engine.shared_model.port.hf,device=args.device,recipe=args.fp8_recipe)
        held=NativeGraphSource(args.held_native_graphs,engine.native_identity['tokenizer_fingerprint'],
                               required_split='test',role_ids=engine.role_tokens.ids)
        task_spec=parse_task_eval(args.task_eval)
        if task_spec is None or not task_spec['neuralese_server']:
            raise ValueError('coefficient production requires actual native task evaluation')
        if task_spec.get('async'):
            raise ValueError('coefficient task evaluation requires synchronous task_eval with exact current actor bindings')
        tasks=TaskEval(task_spec,args.out,engine,prompt_rendering=task_spec.get('prompt_rendering','full'))
        cadence_args=SimpleNamespace(held_eval_tokens=args.held_eval_tokens,held_eval_graphs=args.held_eval_graphs,
                                    checkpoint_layers=args.checkpoint_layers,chunk_tokens=args.chunk_tokens)
        def evaluate(engine,path,*,final,coverage=None):
            policy=restore_policy(engine,path)
            defaults=[{'id':policy['handles'][entry['id']]['current_id'],'scale':float.fromhex(entry['scale_hex'])}
                      for entry in policy['defaults']]
            if not defaults:
                raise ValueError('production held CE needs its declared default actor; explicitly configure default coefficient bindings')
            held_result,held_coverage=run_native(engine,lambda:_evaluate_cadence(engine.shared_model,held,
                args=cadence_args,coverage=coverage,final=final,backend=backend),defaults)
            # TaskEval invokes the same typed runtime binding for BOTH model
            # turns and decisions; it cannot silently evaluate the base actor.
            checkpoint=torch.load(args.out/'checkpoint.pt',map_location='cpu',weights_only=False)
            task=tasks.cadence(checkpoint['step'],final=final)
            gate=task_gate(task,task_spec)
            finite=isinstance(held_result.get('loss'),(int,float)) and __import__('math').isfinite(held_result['loss'])
            passed=bool(finite and task.get('eligible') is True and task.get('coefficient_policy_sha256')==actor_policy_sha256(policy)
                        and (gate is None or gate.get('passed') is True))
            return {'passed':passed,'held':held_result,'held_coverage':held_coverage,'task':task,'task_gate':gate,
                    'held_actor':'declared-default-adapters','base_weights_unchanged':True}
        engine.shared_model.port.checkpoint_layers=args.checkpoint_layers
        report=train(engine,policy=args.coefficient_policy,graphs=[row for _,row in iter_jsonl_rows(args.graphs)],
            objective=args.objective,out=args.out,epochs=args.epochs,optimizer=args.coefficient_optimizer,
            evaluate=evaluate,stop=stop,backend=backend,evaluate_every=args.task_eval_every,
            held_groups=held.groups,held_row_bindings=held.bindings,held_receipts=held.receipt())
        return 0 if report['qualification_complete'] or stop.requested else 1
    finally:
        if engine is not None:engine.store.close()
        stop.restore()


if __name__=='__main__':
    raise SystemExit(main())
