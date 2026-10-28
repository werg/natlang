#!/usr/bin/env python3
"""Full-size CUDA qualification of the production ordinary prefix objective.

This is an exact diagnostic, not a trainer or data-admission policy. Frozen
parameters avoid allocating full-model gradients; leaf token embeddings retain
backpropagation through every actual layer and production loss normalization.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training'/'neuralese'))
from natlang_neuralese.common import line_backbones
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached
from natlang_neuralese.common.jsonio import open_jsonl_text,write_json_durable
from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.model.native_model import load_native_model,load_native_checkpoint
from natlang_neuralese.train.fp8_backend import make_backend


def select_adjacent(path, *, count=2, scan=128, max_tokens=32768):
    """Choose complete adjacent train rows, preferring the smallest useful batch."""
    from natlang_neuralese.train.native_prefix_packing import pack_causal_prefixes
    pending=[];best=None
    with open_jsonl_text(path) as stream:
        for index,line in enumerate(stream):
            if index>=scan:break
            if not line.strip():continue
            row=json.loads(line)
            if row.get('split')!='train':raise ValueError('qualification source must contain only admitted train graphs')
            example=parse_graph(row,tokenizer_fingerprint=row['tokenizer_fingerprint'])
            pending.append((row,example,index,hashlib.sha256(line.rstrip('\r\n').encode()).hexdigest()))
            pending=pending[-count:]
            if len(pending)!=count:continue
            examples=[item[1] for item in pending]
            total=sum(item.plan.token_count for item in examples)
            if total>max_tokens or (best is not None and total>best[0]):continue
            packed=pack_causal_prefixes(examples)
            if packed is None:continue
            plan,mapping=packed
            candidate=(total,-(total-plan.token_count),tuple(pending),plan,mapping)
            if best is None or candidate[:2]<best[:2]:best=candidate
    if best is None:raise ValueError('no complete adjacent train examples have reusable prefixes within the declared batch size')
    _,_,items,plan,mapping=best
    return items,plan,mapping


def gradient_reference(disjoint, maps):
    """Sum all original example uses into their shared physical input row."""
    index=torch.tensor([row for mapping in maps for row in mapping],device=disjoint.device)
    result=torch.zeros((1,max(index.tolist())+1,disjoint.shape[-1]),dtype=torch.float32,device=disjoint.device)
    result.index_add_(1,index,disjoint.float())
    return result


def metrics(actual,expected,*,atol,rtol):
    a,b=actual.detach().float(),expected.detach().float()
    if a.shape!=b.shape:raise ValueError('qualification compared different tensor shapes')
    delta=(a-b).abs();scale=b.abs();bad=delta>atol+rtol*scale
    return {'shape':list(a.shape),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
        'max_abs':float(delta.max()) if delta.numel() else 0.,
        'reference_max':float(scale.max()) if scale.numel() else 0.,
        'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(b).clamp_min(1e-30)),
        'mismatched_elements':int(bad.sum()),'elements':a.numel(),
        'atol':atol,'rtol':rtol,'passed':bool(not bad.any() and torch.isfinite(a).all() and torch.isfinite(b).all())}


def gradient_metrics(actual,expected,*,atol,rtol):
    stats=metrics(actual,expected,atol=atol,rtol=rtol)
    a,b=actual.detach().double().flatten(),expected.detach().double().flatten()
    an,bn=torch.linalg.vector_norm(a),torch.linalg.vector_norm(b)
    cosine=float(torch.dot(a,b)/(an*bn)) if an>0 and bn>0 else float(an==bn)
    stats.update(cosine=cosine,relative_l2_limit=rtol,cosine_min=1-rtol,
                 elementwise_passed=stats['passed'])
    stats['passed'] &= stats['relative_l2']<=rtol and cosine>=1-rtol
    return stats


def snapshot_rng():return (torch.get_rng_state(),torch.cuda.get_rng_state_all())
def restore_rng(state):torch.set_rng_state(state[0]);torch.cuda.set_rng_state_all(state[1])


def qualify_case(model,examples,shared_plan,maps,backend,*,atol,rtol,chunk_tokens):
    from natlang_neuralese.train.native_state_sft import pack_native_graphs,packed_graph_loss
    backend_state=copy.deepcopy(backend.extra_state()) if backend else None
    rng=snapshot_rng();runs={}
    try:
        for shared in (False,True):
            restore_rng(rng)
            if backend:backend.restore_extra_state(copy.deepcopy(backend_state))
            leaves=[];hidden=[]
            def inputs(module,args,kwargs):
                plan=args[0]
                ids=plan.tensor([t for n in plan.invocations for t in n.token_ids],module.input_rows.device)
                leaf=module.embed(ids[None]).detach().requires_grad_(True)
                leaves.append(leaf);return args,{**kwargs,'embeddings':leaf}
            def output(module,args,result):hidden.append(result['h_final'].detach().cpu())
            pre=model.register_forward_pre_hook(inputs,with_kwargs=True)
            post=model.register_forward_hook(output)
            torch.cuda.synchronize();started=time.monotonic()
            try:
                with backend.autocast() if backend else contextlib.nullcontext():
                    loss,details=packed_graph_loss(model,examples,checkpoint_layers=True,
                        chunk_tokens=chunk_tokens,share_causal_prefixes=shared)
                    gradient,=torch.autograd.grad(loss,leaves[0])
                torch.cuda.synchronize()
                if len(leaves)!=1 or len(hidden)!=1:raise AssertionError('production objective replayed a backbone forward')
                runs[shared]={'loss':loss.detach().cpu(),'details':details,'hidden':hidden[0],
                    'gradient':gradient.detach().cpu(),'seconds':time.monotonic()-started}
            finally:pre.remove();post.remove()
            del loss,gradient,leaves,hidden
        original=runs[False];actual=runs[True]
        disjoint_plan=pack_native_graphs(examples)[0]
        flattened=[row for mapping in maps for row in mapping]
        hactual=actual['hidden'][:,flattened]
        comparisons={'h_final':metrics(hactual,original['hidden'],atol=atol,rtol=rtol),
            'loss':metrics(actual['loss'],original['loss'],atol=atol,rtol=rtol),
            'input_embedding_gradient':gradient_metrics(actual['gradient'],gradient_reference(original['gradient'],maps),atol=atol,rtol=rtol)}
        # Compare every vocabulary logit for every original token query without
        # retaining a sequence-by-vocabulary matrix or changing the loss path.
        logit_parts=[]
        with torch.no_grad():
            for start in range(0,hactual.shape[1],chunk_tokens):
                a=model.logits(hactual[:,start:start+chunk_tokens].to(model.input_rows.device))
                b=model.logits(original['hidden'][:,start:start+chunk_tokens].to(model.input_rows.device))
                logit_parts.append(metrics(a,b,atol=atol,rtol=rtol))
        comparisons['logits']={'max_abs':max(x['max_abs'] for x in logit_parts),
            'reference_max':max(x['reference_max'] for x in logit_parts),
            'mismatched_elements':sum(x['mismatched_elements'] for x in logit_parts),
            'elements':sum(x['elements'] for x in logit_parts),
            'finite':all(x['finite'] for x in logit_parts),'atol':atol,'rtol':rtol,
            'passed':all(x['passed'] for x in logit_parts)}
        if [(x['weight'],x['supervised_tokens']) for x in original['details']] != [(x['weight'],x['supervised_tokens']) for x in actual['details']]:
            raise AssertionError('prefix reuse changed original graph loss normalization or target coverage')
        passed=all(x['passed'] for x in comparisons.values())
        if not passed and getattr(model,'_prefix_failure_path',None):
            torch.save({'disjoint':original,'shared':actual,'maps':maps},model._prefix_failure_path)
        return {'passed':passed,'comparisons':comparisons,
            'original_tokens':disjoint_plan.token_count,'physical_tokens':shared_plan.token_count,
            'saved_physical_tokens':disjoint_plan.token_count-shared_plan.token_count,
            'physical_reduction':1-shared_plan.token_count/disjoint_plan.token_count,
            'disjoint_seconds':original['seconds'],'shared_seconds':actual['seconds'],
            'timing_is_isolated':False,'details':actual['details'],
            'gradient_reference':'FP32 index_add of every disjoint original input use; no parameter gradients allocated'}
    finally:
        restore_rng(rng)
        if backend:backend.restore_extra_state(backend_state)



def load_qualification_model(args, *, device='cuda'):
    """Load a real parent without randomizing its learned channel or arithmetic."""
    checkpoint=getattr(args,'checkpoint',None)
    parent=None;backend_options={}
    if checkpoint is None:
        model,tokenizer,roles,config=load_native_model(args.base,device=device,
            dtype=torch.float32 if args.model=='lfm' else torch.bfloat16,
            native_linear_precision=args.linear_precision)
        with torch.no_grad():model.adapter.up.weight.normal_(std=.1)
    else:
        model,tokenizer,roles,config,state=load_native_checkpoint(checkpoint,device=device,base=args.base,
            participant=getattr(args,'native_participant',None))
        training=state.get('training') or state.get('shared_training') or {}
        options=training.get('options') or {}
        backend_options={key:options[key] for key in ('fp8_recipe','fp8_amax_history_len') if key in options}
        backend_options['extra_state']=training.get('fp8')
        parent={'path':str(checkpoint),'sha256':sha256_file_hex_cached(checkpoint),
            'bytes':Path(checkpoint).stat().st_size,'participant':getattr(args,'native_participant',None),
            'learned_parameters_preserved':True}
        del state,training,options
        if config.get('native_linear_precision','ieee')!=args.linear_precision:
            raise ValueError('qualification arithmetic differs from the exact trained parent')
    if config['architecture'] not in line_backbones.architectures(args.model):
        raise ValueError('qualification model family differs from the actual loaded checkpoint')
    for parameter in model.parameters():parameter.requires_grad_(False)
    model.train()
    return model,tokenizer,roles,config,parent,backend_options


def code_revision(explicit):
    if explicit:return explicit
    try:
        result=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=False)
        return result.stdout.strip() if result.returncode==0 else None
    except FileNotFoundError:
        return None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['lfm','mellum'],required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--native-participant')
    p.add_argument('--code-revision',help='exact staged runtime commit when no Git checkout is mounted')
    p.add_argument('--fp8-recipe',choices=['delayed','current','current-bf16-dx'])
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--examples',type=int,choices=[2,3],default=2);p.add_argument('--scan',type=int,default=128)
    p.add_argument('--max-tokens',type=int,default=32768);p.add_argument('--chunk-tokens',type=int,default=128)
    p.add_argument('--linear-precision',choices=['ieee','bf16x6'],default='bf16x6')
    p.add_argument('--atol',type=float,default=1e-5);p.add_argument('--rtol',type=float,default=1e-4)
    args=p.parse_args(argv);report={'schema':'natlang.native-causal-prefix-qualification/1','passed':False,
        'timing_note':'Paired diagnostic timings on a potentially shared GPU are not isolated throughput evidence.',
        'code_revision':code_revision(args.code_revision),
        'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[]};started=time.monotonic()
    try:
        if not torch.cuda.is_available():raise RuntimeError('full-size prefix qualification requires actual CUDA')
        torch.manual_seed(191);torch.cuda.manual_seed_all(191)
        items,plan,maps=select_adjacent(args.source,count=args.examples,scan=args.scan,max_tokens=args.max_tokens)
        report['source']={'path':str(args.source),'sha256':sha256_file_hex_cached(args.source),
            'rows':[{'id':item[1].id,'index':item[2],'line_sha256':item[3],
                'canonical_sha256':canonical_json_sha256_hex(item[0]),'tokens':item[1].plan.token_count} for item in items]}
        model,tokenizer,roles,config,parent,backend_options=load_qualification_model(args)
        report['model']=config;report['parent_checkpoint']=parent
        if any(item[0]['tokenizer_fingerprint']!=config['tokenizer_fingerprint'] for item in items):
            raise ValueError('actual admitted graph tokenizer differs from loaded pinned model')
        report['adapter_up_norm']=float(model.adapter.up.weight.norm())
        report['cuda']={'device':torch.cuda.get_device_name(),'capability':list(torch.cuda.get_device_capability())}
        examples=[item[1] for item in items]
        recipe=args.fp8_recipe or backend_options.get('fp8_recipe','delayed')
        if parent and args.fp8_recipe and backend_options.get('fp8_recipe',args.fp8_recipe)!=args.fp8_recipe:
            raise ValueError('qualification FP8 recipe differs from the trained parent')
        backend=make_backend('triton-experts',model.port.hf,device='cuda',recipe=recipe,
            amax_history_len=backend_options.get('fp8_amax_history_len',16)) if args.model=='mellum' else None
        if backend:
            if backend_options.get('extra_state') is not None:backend.restore_extra_state(backend_options['extra_state'])
            model.port.activation_checkpoint=backend.checkpoint
        report['fp8_recipe']=recipe if backend else None
        # Native BF16 expert execution and production FP8 execution are distinct
        # numerical contracts, both qualified on the same complete examples.
        modes=['native','triton-experts'] if backend else ['native']
        for mode in modes:
            model._prefix_failure_path=args.output.with_name(args.output.stem+'-'+mode+'.failure.pt')
            active=backend if mode=='triton-experts' else None
            scope=backend.native_modules() if backend and active is None else contextlib.nullcontext()
            print(json.dumps({'event':'case_start','model':args.model,'mode':mode,
                'original_tokens':sum(x.plan.token_count for x in examples),'physical_tokens':plan.token_count}),flush=True)
            executed={'fp8_projection_calls':0}
            from natlang_neuralese.maple import expert_gemm
            original_projection=expert_gemm.grouped_fp8
            def projection(*values,**options):
                executed['fp8_projection_calls']+=1
                return original_projection(*values,**options)
            expert_gemm.grouped_fp8=projection
            try:
                with scope:
                    result=qualify_case(model,examples,plan,maps,active,atol=args.atol,rtol=args.rtol,chunk_tokens=args.chunk_tokens)
            finally:
                expert_gemm.grouped_fp8=original_projection
            from natlang_neuralese.maple.model import DENSE_MOE
            result.update(executed=executed,dense_moe_kernel=DENSE_MOE['kernel'])
            if active is not None and executed['fp8_projection_calls']==0:
                result['passed']=False
                result['backend_error']='declared FP8 backend executed no FP8 projection; configuration is not execution proof'
            report['cases'].append({'mode':mode,**result});write_json_durable(args.output,report)
            print(json.dumps({'event':'case_done','mode':mode,**result}),flush=True)
        report['passed']=all(x['passed'] for x in report['cases'])
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error);raise
    finally:
        report.update(seconds=time.monotonic()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
        write_json_durable(args.output,report)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
