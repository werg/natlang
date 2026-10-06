"""Teacherless full-stack neuralese warm-up on causally aligned ordinary text.

Gold IDs anchor next-token CE and raw embedding targets. The input/output token
space is fixed; transformer layers, sketch F and content residual train. Generated
history uses the shared one-stage gradient policy, never recurrent full BPTT.
No task, compression, autonomous stopping or transport certificate is issued.
"""
from __future__ import annotations
import argparse, hashlib, json, random, signal, time
from pathlib import Path
import torch
from torch.nn import functional as F
from .execution import prefill_write_context, replay_local_stages, rollout_sketch_inputs
from .output_embedding_projection import sha
from .trajectory_state import atomic_checkpoint, clip_finite_gradients, gradient_norm


def relative_mse(predicted, target):
    target = target.detach().float()
    return ((predicted.float()-target).square().mean(-1) /
            target.square().mean(-1).clamp_min(1e-6)).mean()


def scheduled_completion(backbone, heads, prefix_ids, span_ids, *, fraction=1., group_size=16,
                         auxiliary_scale=.05):
    """Gold x[j] is predicted by top[j]; its sketch completes top[j+1].

    Detached rollout fixes input history. Each replay row differentiates only
    its own sketch replacement; earlier fixed inputs still carry backbone credit.
    At fraction0 execution consumes exact gold embeddings. No supplied future
    token influences an earlier sketch/state. The last gold token is a target
    only, so no untrained extra completion is performed.
    """
    if not 0 <= fraction <= 1 or not 0 <= auxiliary_scale <= 1 or group_size < 1:
        raise ValueError('invalid scheduled completion controls')
    if span_ids.ndim != 2 or span_ids.shape[1] < 2 or prefix_ids.shape[0] != span_ids.shape[0]:
        raise ValueError('nonempty prefixes and at least two gold target tokens required')
    if not heads.autoregressive:
        raise ValueError('text warm-up requires autoregressive latent-sketch-v2')
    if fraction==0:
        ordinary=backbone.forward_ids(torch.cat([prefix_ids,span_ids[:,:-1]],1),
                                      cutoff=heads.cutoff,logits=False)
        start=prefix_ids.shape[1]-1
        sources=ordinary['h_cut'][:,start:]
        auxiliary=sources.detach()+auxiliary_scale*(sources-sources.detach())
        return {'top':ordinary['h_final'][:,start:],
                'sketches':heads.feedback(auxiliary),
                'replay_delta':torch.zeros((),device=span_ids.device)}
    pre = prefill_write_context(backbone, heads, backbone.embed(prefix_ids))
    gold = backbone.embed(span_ids[:, :-1]).detach()
    k = heads.cutoff
    with torch.no_grad():
        fixed,shallow,predictions,last=rollout_sketch_inputs(
            backbone,heads,pre,gold.shape[1],reference_inputs=gold,fraction=fraction)
        primal,_=backbone.run_layers(shallow,range(k,backbone.num_layers),pre.cache)
        predictions=torch.cat([predictions,heads.feedback(last)[:,None]],1)
    # Held probes use the actual forward path without redundant gradient replay.
    if not torch.is_grad_enabled():
        return {'top':torch.cat([pre.top[:, None], primal], 1), 'sketches':predictions,
                'replay_delta':torch.zeros((), device=gold.device)}
    _, guesses, _, replay = replay_local_stages(
        backbone, heads, pre, fixed, reference_inputs=gold, fraction=fraction,
        group_size=group_size, auxiliary_scale=auxiliary_scale, terminal_guess=True)
    delta=(replay.detach().float()-primal.float()).abs().max()
    final=primal.detach()+(replay-replay.detach())
    return {'top':torch.cat([pre.top[:,None],final],1),
            'sketches':guesses,'replay_delta':delta}


def qualification(report, *, max_ce_delta=.1, max_relative_mse=.25,
                  min_agreement=.9):
    """All held strata must pass; no aggregate can hide a failing stratum."""
    strata=report.get('strata',{})
    return bool(strata) and all(r['tokens']>0 and
        r['ce_delta']<=max_ce_delta and r['embedding_mse_delta']<=max_relative_mse and
        r.get('text_ce_delta_from_initial',0.)<=max_ce_delta and
        r['text_argmax_agreement']>=min_agreement for r in strata.values())


def document_windows(token_ids, *, open_id, close_id, tokens, prefix_tokens):
    """Prime with real open; supervise each text token and the real close once."""
    ids=[open_id]+list(token_ids)+[close_id]
    stride=tokens-prefix_tokens
    windows=[]
    for offset in range(0,len(ids),stride):
        start=max(0,offset-prefix_tokens)
        chunk=ids[start:offset+stride]
        width=1 if offset==0 else offset-start
        if len(chunk)<=width:continue
        windows.append({'ids':chunk,'prefix':width,'offset':offset})
    return windows


def load_text_rows(records, pieces=None, text_data=None):
    """Explicit train/test and factual provenance; exact duplicates stay held out."""
    if text_data:
        rows=list(map(json.loads,Path(text_data).open()))
        if any(r.get('split') not in ('train','test') or not isinstance(r.get('text'),str)
               or not r['text'].strip() or not r.get('source_groups') for r in rows):
            raise ValueError('text JSONL needs nonempty text, train/test split and source_groups')
    else:
        import hashlib
        from ..data.text_corpus import gold_text_rows
        records_path=Path(records)
        record_lines=[line for line in records_path.read_bytes().splitlines() if line]
        record_rows=[json.loads(line) for line in record_lines]
        for row,line in zip(record_rows,record_lines):
            row['_source_record_sha256']=hashlib.sha256(line).hexdigest()
        pieces_path=Path(pieces) if pieces else records_path.parent/'pieces.jsonl'
        piece_rows=list(map(json.loads,pieces_path.open())) if pieces_path.is_file() else []
        rows,_,_,_=gold_text_rows(record_rows,piece_rows)
    groups={s:set(g for r in rows if r['split']==s for g in r['source_groups']) for s in ('train','test')}
    if groups['train']&groups['test']:
        raise ValueError('text warm-up factual source groups cross train/test')
    held={r['text'] for r in rows if r['split']=='test'}
    dedup={}; excluded=0
    for row in rows:
        if row['split']=='train' and row['text'] in held:excluded+=1;continue
        dedup.setdefault((row['split'],row['text']),row)
    rows=list(dedup.values())
    if not all(any(r['split']==s for r in rows) for s in ('train','test')):
        raise ValueError('nonempty independent train and held text required')
    return rows, {'policy':'explicit source-disjoint ordinary text; gold targets, no teacher model',
                  'excluded_train_exact_held_duplicates':excluded,'documents':len(rows)}


def configure_student(engine, policy='full', rank=16):
    backbone,heads=engine.backbone,engine.heads
    for p in backbone.parameters():p.requires_grad_(False)
    for p in heads.parameters():p.requires_grad_(False)
    if policy=='full':
        if getattr(backbone,'ternary',False):
            raise ValueError('ternary full-weight updates require its QAT policy; use adapters')
        for name,p in backbone.hf.named_parameters():
            if name.startswith('model.layers.'):p.requires_grad_(True)
    elif policy=='adapters':
        from .adapters import inject_lora
        inject_lora(backbone,list(range(backbone.num_layers)),rank=rank,alpha=2*rank)
    else:raise ValueError('unknown student backbone training policy')
    for p in heads.feedback.parameters():p.requires_grad_(True)
    for p in heads.content.proj.parameters():p.requires_grad_(True)
    # Keep vocabulary/embedding coordinates and output normalization stable.
    backbone.hf.eval();heads.eval()
    named=[('backbone.'+n,p) for n,p in backbone.hf.named_parameters() if p.requires_grad]
    named += [('heads.'+n,p) for n,p in heads.named_parameters() if p.requires_grad]
    if not any(n.startswith('backbone.') for n,p in named):raise ValueError('student full stack has no trainables')
    return named


def load_initial(heads, checkpoint, device, cutoff):
    if checkpoint:
        from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
        engine,parent=load_recurrence_checkpoint(checkpoint,device=device)
    else:
        from ..serve import load_engine
        engine=load_engine(heads_checkpoint=str(heads),device=device);parent=None
    if engine.heads.profile=='raw-token-v1':
        from .sketch_handoff import install_latent_sketch
        install_latent_sketch(engine,cutoff=cutoff,profile='latent-sketch-v2')
    if engine.heads.profile!='latent-sketch-v2':raise ValueError('requires latent-sketch-v2 or qualified raw foundation')
    return engine,parent


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('heads','records','out'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('pieces','text-data','student-checkpoint','continue-from'):p.add_argument('--'+name,type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--steps',type=int,default=4096)
    p.add_argument('--tokens',type=int,default=1024);p.add_argument('--prefix-tokens',type=int,default=32)
    p.add_argument('--cutoff',type=int,default=4);p.add_argument('--group-size',type=int,default=16)
    p.add_argument('--batch',type=int,default=2,help='same-shape text rows per optimizer update')
    p.add_argument('--backbone-training',choices=['full','adapters'],default='full');p.add_argument('--rank',type=int,default=16)
    p.add_argument('--optimizer',choices=['muon','adamw'],default='muon');p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--sketch-lr',type=float,default=3e-4);p.add_argument('--embedding-weight',type=float,default=1.)
    p.add_argument('--sketch-weight',type=float,default=.1);p.add_argument('--text-weight',type=float,default=.25)
    p.add_argument('--aligned-steps',type=int,default=128);p.add_argument('--ramp-steps',type=int,default=512)
    p.add_argument('--checkpoint-every',type=int,default=128);p.add_argument('--eval-every',type=int,default=128)
    p.add_argument('--held-documents',type=int,default=16);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--checkpoint-layers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--max-ce-delta',type=float,default=.1);p.add_argument('--max-relative-mse',type=float,default=.25)
    p.add_argument('--min-agreement',type=float,default=.9);p.add_argument('--consecutive-gates',type=int,default=2)
    a=p.parse_args(argv)
    if min(a.steps,a.tokens,a.prefix_tokens,a.group_size,a.batch,a.eval_every,a.checkpoint_every,a.held_documents,a.consecutive_gates)<1 or a.tokens<3:
        p.error('positive bounds and at least three tokens required')
    if a.aligned_steps<1 or a.ramp_steps<1 or min(a.lr,a.sketch_lr,a.embedding_weight,a.sketch_weight,a.text_weight)<=0:
        p.error('invalid schedule or optimizer controls')
    if a.prefix_tokens>=a.tokens-1:p.error('prefix must leave at least two target tokens')
    if not 0<=a.min_agreement<=1 or min(a.max_ce_delta,a.max_relative_mse)<0:p.error('invalid gates')
    torch.set_num_threads(2);torch.manual_seed(a.seed);random.seed(a.seed)
    options={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='out'}
    paths=[x for x in (a.heads,a.records,a.pieces,a.text_data,a.student_checkpoint,a.continue_from) if x]
    package=Path(__file__).parents[1]
    identity={'options':options,'inputs':{str(x.resolve()):sha(x) for x in paths},
              'code':{str(x.relative_to(package)):sha(x) for x in package.rglob('*.py')},
              'target':'E(gold next token), fixed raw input table; no teacher; full-stack next-token CE',
              'sketch_gradient':'local_stage','sketch_target_backbone_scale':.05}
    state_path=a.out/'checkpoint.pt'
    resumed=torch.load(state_path,map_location='cpu',weights_only=False,mmap=True) if state_path.exists() else None
    if resumed and (resumed.get('schema')!='natlang.neuralese-text-warmup/1' or resumed['identity']!=identity):raise ValueError('warm-up resume identity changed')
    if a.out.exists() and not resumed:raise ValueError('fresh output or complete checkpoint required')
    continuation=None
    if a.continue_from:
        continuation=torch.load(a.continue_from,map_location='cpu',weights_only=False,mmap=True)
        if continuation.get('schema')!='natlang.neuralese-text-warmup/1':raise ValueError('full text warm-up state required')
        old=continuation['identity']['options']
        if any(old[k]!=options[k] for k in ('optimizer','backbone_training','rank','lr','sketch_lr')):
            raise ValueError('continuation optimizer/parameter policy differs')
    a.out.mkdir(parents=True,exist_ok=True)
    engine,parent=load_initial(a.heads,a.student_checkpoint,a.device,a.cutoff)
    backbone,heads=engine.backbone,engine.heads
    backbone.checkpoint_layers=a.checkpoint_layers;backbone.ffn_chunk_tokens=1024
    named=configure_student(engine,a.backbone_training,a.rank)
    parameters=dict(named)
    backbone_names={n.removeprefix('backbone.') for n,q in named if n.startswith('backbone.')}
    from .optim import PortMuonAdamW
    if a.optimizer=='muon':
        optimizer=PortMuonAdamW(named,lr=a.lr,vocab_size=backbone.embedding_weight.shape[0])
        for child in (optimizer.muon,optimizer.auxiliary):
            if child is None:continue
            original=dict(child.param_groups[0]); buckets={}
            for q in original['params']:buckets.setdefault(a.sketch_lr if any(q is v for n,v in named if n.startswith('heads.')) else a.lr,[]).append(q)
            first,*rest=buckets.items();child.param_groups[0].update(params=first[1],lr=first[0])
            for rate,qs in rest:child.add_param_group({**original,'params':qs,'lr':rate})
        optimizer.param_groups=optimizer._groups()
    else:
        optimizer=torch.optim.AdamW([{'params':[q for n,q in named if n.startswith('backbone.')],'lr':a.lr},
          {'params':[q for n,q in named if n.startswith('heads.')],'lr':a.sketch_lr}],weight_decay=0.)
    rows,receipt=load_text_rows(a.records,a.pieces,a.text_data)
    windows={'train':[],'test':[]}
    for row in rows:
        for window in document_windows(engine._tokens(row['text']),
                open_id=backbone.controls.open_id, close_id=backbone.controls.close_id,
                tokens=a.tokens, prefix_tokens=a.prefix_tokens):
            windows[row['split']].append({**window,
                'document':hashlib.sha256(row['text'].encode()).hexdigest(),
                'groups':row['source_groups']})
    if not all(windows.values()):raise ValueError('no token windows for a split')
    held=[];grouped={};documents={}
    for w in windows['test']:
        documents.setdefault(w['document'],[]).append(w)
        grouped.setdefault(tuple(w['groups']),set()).add(w['document'])
    queues=[sorted(v) for _,v in sorted(grouped.items())]
    selected=[];seen=set()
    while len(selected)<a.held_documents and any(queues):
        for queue in queues:
            if not queue:continue
            doc=queue.pop(0)
            if doc not in seen:selected.append(doc);seen.add(doc)
            if len(selected)>=a.held_documents:break
    for doc in selected:
        # Two positions per factual-round-robin document keep periodic probes
        # bounded while covering both prompt/schema and final target regions.
        values=documents[doc];held.append(values[0])
        if len(values)>1:held.append(values[-1])
    receipt.update(windows={s:len(v) for s,v in windows.items()},held_windows=len(held),
                   boundaries={'policy':'one actual neuralese open/close token per complete document; no synthetic closes at window edges',
                               'open_id':backbone.controls.open_id,'close_id':backbone.controls.close_id},
                   trainable_parameters={s:sum(q.numel() for n,q in named if n.startswith(s)) for s in ('backbone.','heads.')})
    (a.out/'plan.json').write_text(json.dumps({'identity':identity,'receipt':receipt},indent=2)+'\n')
    def ids_for(w):
        rows=[w] if isinstance(w,dict) else w
        ids=torch.tensor([r['ids'] for r in rows],device=a.device)
        return ids[:,:rows[0]['prefix']],ids[:,rows[0]['prefix']:]
    buckets={}
    for window in windows['train']:
        buckets.setdefault((window['prefix'],len(window['ids'])),[]).append(window)
    def objective(w,fraction):
        prefix,span=ids_for(w)
        out=scheduled_completion(backbone,heads,prefix,span,fraction=fraction,group_size=a.group_size)
        top=out['top'];logits=backbone.logits(top).float()
        ce=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),span.reshape(-1))
        target=backbone.embed(span).detach()
        payload=heads.content(torch.zeros_like(top),top)
        embedding=relative_mse(payload,target)
        sketch=relative_mse(out['sketches'],target)
        plain=top if fraction==0 else backbone.forward_ids(torch.cat([prefix,span[:,:-1]],1),logits=False)['h_final'][:,prefix.shape[1]-1:]
        plain_logits=backbone.logits(plain).float()
        plain_ce=F.cross_entropy(plain_logits.reshape(-1,plain_logits.shape[-1]),span.reshape(-1))
        plain_payload=heads.content.reference(plain)
        plain_embedding=relative_mse(plain_payload,target)
        loss=ce+a.embedding_weight*embedding+a.sketch_weight*sketch+a.text_weight*plain_ce
        if step<a.aligned_steps and torch.is_grad_enabled():
            loss=sketch  # explicit projection-first bootstrap against raw gold E
        with torch.no_grad():
            ending=span==backbone.controls.close_id
            close_probability=(logits[...,backbone.controls.close_id]-torch.logsumexp(logits,-1)).exp()
            prediction=logits.argmax(-1)
            close_count=int(ending.sum())
            stop_metrics={'close_targets':close_count,
              'close_probability':float(close_probability[ending].mean()) if close_count else None,
              'close_top1':float((prediction[ending]==backbone.controls.close_id).float().mean()) if close_count else None,
              'premature_close_top1':float((prediction[~ending]==backbone.controls.close_id).float().mean()) if (~ending).any() else 0.}
        return loss,{'ce':float(ce.detach()),'text_ce':float(plain_ce.detach()),'ce_delta':float((ce-plain_ce).detach()),
          'relative_mse':float(embedding.detach()),'sketch_mse':float(sketch.detach()),
          'text_embedding_mse':float(plain_embedding.detach()),
          'embedding_mse_delta':float((embedding-plain_embedding).detach()),
          'text_argmax_agreement':float((logits.argmax(-1)==plain_logits.argmax(-1)).float().mean()),
          'gold_accuracy':float((logits.argmax(-1)==span).float().mean()),'replay_delta':float(out['replay_delta'].detach()),
          'tokens':span.numel(),'positions':span.shape[1],**stop_metrics}
    step=0;streak=0;best=None;updates={'backbone':False,'sketch':False}
    initial_text_ce={}
    restored=resumed or continuation
    if restored:
        with torch.no_grad():
            for n,v in restored['student_parameters'].items():parameters[n].copy_(v.to(parameters[n]))
        heads.load_state_dict(restored['heads']);optimizer.load_state_dict(restored['optimizer'])
        step=restored['step'];updates=restored['updates']
        if resumed:
            streak=resumed['streak'];best=resumed['best'];initial_text_ce=resumed['initial_text_ce']
        random.setstate(restored['python_rng']);torch.set_rng_state(restored['torch_rng'])
        if a.device.startswith('cuda'):torch.cuda.set_rng_state_all(restored['cuda_rng'])
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def log(name,value):
        with (a.out/name).open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    def evaluate():
        strata={};boundaries={'close_targets':0,'close_probability_sum':0.,'close_top1_sum':0.}
        with torch.no_grad():
            for w in held:
                _,m=objective(w,1.)
                boundaries['close_targets']+=m['close_targets']
                if m['close_targets']:
                    boundaries['close_probability_sum']+=m['close_probability']*m['close_targets']
                    boundaries['close_top1_sum']+=m['close_top1']*m['close_targets']
                key='length-'+('short' if m['positions']<=32 else 'medium' if m['positions']<=128 else 'long')+'-'+('start' if w['offset']==0 else 'tail')
                row=strata.setdefault(key,{'tokens':0})
                for n in ('ce','text_ce','ce_delta','relative_mse','sketch_mse','text_embedding_mse','embedding_mse_delta','text_argmax_agreement','gold_accuracy'):
                    row[n]=row.get(n,0.)+m[n]*m['tokens']
                row['tokens']+=m['tokens']
        for row in strata.values():
            for n in row.keys()-{'tokens'}:row[n]/=row['tokens']
        for key,row in strata.items():
            initial_text_ce.setdefault(key,row['text_ce'])
            row['text_ce_delta_from_initial']=row['text_ce']-initial_text_ce[key]
        from .trajectory_state import weights_digest
        report={'step':step,'strata':strata,'runtime_qualified':False,'autonomous_stopping_qualified':False,
                'boundary_supervision':boundaries,
                'weights_digest':weights_digest({n:q for n,q in backbone.hf.named_parameters() if n in backbone_names},heads.state_dict()),
                'updates':dict(updates)}
        report['alignment_gate_passed']=qualification(report,max_ce_delta=a.max_ce_delta,max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement)
        log('eval.jsonl',report);return report
    def save(report=None):
        atomic_checkpoint(state_path,{'schema':'natlang.neuralese-text-warmup/1','identity':identity,'step':step,
          'student_parameters':{n:q.detach().cpu() for n,q in named},'heads':heads.state_dict(),
          'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
          'cuda_rng':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [],
          'streak':streak,'best':best,'updates':updates,'qualification':report,
          'initial_text_ce':initial_text_ce})
        # Shared serving heads carry explicit backbone deltas, never inherited certification.
        from .adapters import lora_state,adapter_layers
        initial=torch.load(a.heads,map_location='cpu',weights_only=False,mmap=True)
        exported={**initial,'heads':heads.state_dict(),'control_rows':backbone.control_rows.detach().cpu(),
          'port_config':{'cutoff':heads.cutoff,'max_length':heads.max_length,**heads.port_config()},
          'backbone_trainables':{n:q.detach().cpu() for n,q in backbone.hf.named_parameters() if n in backbone_names},
          'foundation':{'qualified':False,'runtime_qualified':False,'requires_requalification':True},
          'lora':lora_state(backbone),'lora_layers':adapter_layers(backbone),'lora_rank':a.rank,
          'warmup':{'step':step,'identity':identity,'alignment_qualified':bool(report and report.get('qualified')),
                    'report_path':str((a.out/'report.json').resolve()),
                    'report_sha256':sha(a.out/'report.json') if (a.out/'report.json').is_file() else None}}
        # Parent adapters may have a different rank than the fresh-policy default.
        ranks={v.shape[0] for n,v in exported['lora'].items() if '.lora_A.' in n}
        if len(ranks)==1:exported['lora_rank']=next(iter(ranks))
        atomic_checkpoint(a.out/'heads.pt',exported)
    if not resumed:
        baseline=evaluate();(a.out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n');save()
    for _ in range(step,a.steps):
        if stop[0]:break
        # Optimizer membership/state stays fixed across this declared stage boundary.
        for name,q in named:
            q.requires_grad_(step>=a.aligned_steps or name.startswith('heads.feedback.'))
        fraction=min(1.,max(0.,(step-a.aligned_steps)/a.ramp_steps))
        w=windows['train'][random.randrange(len(windows['train']))]
        pool=buckets[(w['prefix'],len(w['ids']))]
        batch=[w]+[pool[random.randrange(len(pool))] for _ in range(a.batch-1)]
        optimizer.zero_grad(set_to_none=True);started=time.perf_counter()
        loss,m=objective(batch,fraction)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite warm-up loss')
        loss.backward()
        backbone_norm=gradient_norm(q for n,q in named if n.startswith('backbone.'))
        sketch_norm=gradient_norm(q for n,q in named if n.startswith('heads.feedback.'))
        clip_finite_gradients(parameters.values())
        # Positive gradients plus real parameter deltas audit both trainable paths.
        samples={k:next((q for n,q in named if n.startswith(prefix) and q.grad is not None and q.grad.abs().sum()>0),None)
                 for k,prefix in [('backbone','backbone.'),('sketch','heads.feedback.')]}
        before={k:q.detach().clone() for k,q in samples.items() if q is not None}
        optimizer.step();step+=1
        for k,v in before.items():updates[k]|=not torch.equal(v,samples[k].detach())
        m.update(step=step,fraction=fraction,loss=float(loss.detach()),seconds=time.perf_counter()-started,
                 phase='sketch_projection' if step<=a.aligned_steps else 'full_stack',
                 batch=a.batch,
                 backbone_gradient_norm=float(backbone_norm),sketch_gradient_norm=float(sketch_norm),updates=dict(updates))
        log('train.jsonl',m)
        if step%a.eval_every==0:
            report=evaluate()
            streak=streak+1 if fraction==1. and report['alignment_gate_passed'] and all(updates.values()) else 0
            report.update(consecutive_passes=streak,qualified=streak>=a.consecutive_gates,
                          scope='text alignment only; stopping, transport and Natlang tasks unqualified')
            if best is None or sum(r['ce']*r['tokens'] for r in report['strata'].values())<best['score']:
                best={'step':step,'score':sum(r['ce']*r['tokens'] for r in report['strata'].values()),'report':report}
            (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');save(report)
            if report['qualified']:break
        elif step%a.checkpoint_every==0:save()
    report=evaluate();report.update(consecutive_passes=streak,qualified=streak>=a.consecutive_gates,
      updates=updates,status='checkpointed_on_signal' if stop[0] else 'complete',
      scope='text alignment only; stopping, transport and Natlang tasks unqualified')
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');save(report)

if __name__=='__main__':main()
