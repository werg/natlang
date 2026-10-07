"""Teacherless full-stack neuralese warm-up on causally aligned ordinary text.

Gold IDs anchor next-token CE and raw embedding targets. The input/output token
space is fixed; transformer layers, sketch F and content residual train. Gold text history is teacher-forced; individual neuralese completions use the
shared one-stage gradient policy, never unconditional free-running imitation.
No task, compression, autonomous stopping or transport certificate is issued.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, signal, time
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .execution import prefill_write_context, replay_sequence_inputs
from .output_embedding_projection import sha
from .trajectory_state import atomic_checkpoint, clip_finite_gradients, gradient_norm
from .foundation_schedule import ProjectionFirstSchedule


def relative_mse_positions(predicted, target):
    target = target.detach().float()
    return ((predicted.float()-target).square().mean(-1) /
            target.square().mean(-1).clamp_min(1e-6))


def relative_mse(predicted, target):
    return relative_mse_positions(predicted,target).mean()


TEXT_SUPERVISION_POLICY={
    "all_positions_fraction": .5, "observed_suffix_fraction": .5,
    "unannotated_or_no_suffix_window": "uniform-all-positions",
    "objectives": ["full_projection", "sketch_projection", "next_token_ce"],
    "qualification": "unweighted full-history complete-window and last256 strata",
}


def balanced_position_weights(span, suffix_starts):
    """Half all-token supervision, half observed response suffix per document.

    Unlabeled ordinary text and windows without a response retain uniform
    all-token supervision. Context is never removed or detached by this weighting.
    """
    if len(suffix_starts)!=span.shape[0]:raise ValueError('one suffix coordinate per document required')
    starts=torch.tensor([span.shape[1] if start is None else start for start in suffix_starts],device=span.device)
    mask=torch.arange(span.shape[1],device=span.device)[None]>=starts[:,None]
    counts=mask.sum(1,keepdim=True)
    weighted=(TEXT_SUPERVISION_POLICY["all_positions_fraction"]+
              TEXT_SUPERVISION_POLICY["observed_suffix_fraction"]*mask.float()*span.shape[1]/counts.clamp_min(1)
             )
    return torch.where(counts>0,weighted,torch.ones_like(weighted))


def chunked_readout(backbone, states, targets, close_id, *, chunk_size=128,
                    gradients=True, position_weights=None):
    """Exact token-mean CE and readout metrics without retaining T x vocab logits.

    Differentiable chunks are non-reentrantly checkpointed, so backward
    recomputes each readout instead of keeping its vocabulary-sized activation.
    Only scalar CE chunks and small detached token metrics survive the loop.
    """
    if states.ndim != 3 or targets.shape != states.shape[:2]:
        raise ValueError('readout expects [batch,time,width] states and aligned token IDs')
    if chunk_size < 1 or states.shape[1] < 1:
        raise ValueError('positive chunk size and nonempty sequence required')
    if position_weights is None:position_weights=torch.ones_like(targets,dtype=torch.float32)
    if position_weights.shape!=targets.shape or not torch.isfinite(position_weights).all() or (position_weights<=0).any():
        raise ValueError("positive finite aligned position weights required")
    losses=[];weighted_losses=[];predictions=[];close_probabilities=[];token_losses=[]
    needs_grad=gradients and torch.is_grad_enabled() and (
        states.requires_grad or any(p.requires_grad for p in backbone.parameters()))
    for start in range(0,states.shape[1],chunk_size):
        stop=min(start+chunk_size,states.shape[1])
        state_chunk=states[:,start:stop]
        target_chunk=targets[:,start:stop]
        weight_chunk=position_weights[:,start:stop]
        def readout(chunk, gold, weights):
            logits=backbone.logits(chunk).float()
            per_token=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),gold.reshape(-1),reduction='none').reshape_as(gold)
            ce=per_token.sum()
            training_ce=(per_token*weights).sum()
            with torch.no_grad():
                pred=logits.argmax(-1)
                close=(logits[...,close_id]-torch.logsumexp(logits,-1)).exp()
            return ce,training_ce,pred,close,per_token.detach()
        if needs_grad:
            ce,training_ce,pred,close,positions=checkpoint(readout,state_chunk,target_chunk,weight_chunk,use_reentrant=False)
        elif gradients:
            ce,training_ce,pred,close,positions=readout(state_chunk,target_chunk,weight_chunk)
        else:
            with torch.no_grad():
                ce,training_ce,pred,close,positions=readout(state_chunk,target_chunk,weight_chunk)
        losses.append(ce)
        weighted_losses.append(training_ce)
        predictions.append(pred.detach())
        close_probabilities.append(close.detach())
        token_losses.append(positions)
    count=targets.numel()
    return (torch.stack(losses).sum()/count,torch.stack(weighted_losses).sum()/count,
            torch.cat(predictions,dim=1),
            torch.cat(close_probabilities,dim=1),torch.cat(token_losses,dim=1))


def projection_errors(heads, top, sketches, target):
    return (relative_mse_positions(heads.content(torch.zeros_like(top),top),target),
            relative_mse_positions(sketches,target))


def projection_losses(heads, top, sketches, target):
    """Both separate trainable maps see fixed gold embeddings immediately."""
    full,shallow=projection_errors(heads,top,sketches,target)
    return full.mean(),shallow.mean()


@torch.no_grad()
def alignment_region_metrics(losses, prediction, plain_losses, plain_prediction,
                             embedding, sketch, reference, target, gold):
    """Score selected positions after the same full-history forward pass."""
    ce=losses.mean();plain_ce=plain_losses.mean()
    full_error=relative_mse(embedding,target);reference_error=relative_mse(reference,target)
    return {'ce':float(ce),'text_ce':float(plain_ce),'ce_delta':float(ce-plain_ce),
            'relative_mse':float(full_error),'sketch_mse':float(relative_mse(sketch,target)),
            'text_embedding_mse':float(reference_error),
            'embedding_mse_delta':float(full_error-reference_error),
            'text_argmax_agreement':float((prediction==plain_prediction).float().mean()),
            'gold_accuracy':float((prediction==gold).float().mean()),'tokens':gold.numel()}


def gold_completion(backbone, heads, prefix_ids, span_ids, *, auxiliary_scale=.05):
    """Gold history anchors both depth-specific next-token projections."""
    if span_ids.ndim != 2 or span_ids.shape[1] < 1 or prefix_ids.shape[1] < 1:
        raise ValueError('nonempty prefixes and gold target tokens required')
    ordinary=backbone.forward_ids(torch.cat([prefix_ids,span_ids[:,:-1]],1),
                                  cutoff=heads.cutoff,logits=False)
    start=prefix_ids.shape[1]-1
    sources=ordinary['h_cut'][:,start:]
    auxiliary=sources.detach()+auxiliary_scale*(sources-sources.detach())
    return {'top':ordinary['h_final'][:,start:],
            'sketches':heads.feedback(auxiliary)}


def sequence_completions(backbone, heads, prefix_ids, span_ids, *, passes=3,
                         fraction=1., group_size=16, auxiliary_scale=.05):
    """Repeated shared shallow-layer passes, with one-consumer sketch credit.

    Pass zero reads gold text. Every later pass consumes the preceding pass's
    next-token predictions at their corresponding INPUT positions. Each pass
    retains the same prefix and gold next-token targets. Replay recomputes its
    producer from detached older inputs; output at j credits only the incoming
    sketch at j, never a chain of older sketches or later consumer positions.
    Consume/backpropagate each yielded pass before requesting the next to bound
    memory. No optimizer update may occur between these passes.
    """
    if passes < 1 or not 0 <= fraction <= 1:
        raise ValueError('positive sequence passes and bounded fraction required')
    first=gold_completion(backbone,heads,prefix_ids,span_ids,auxiliary_scale=auxiliary_scale)
    yield {**first,'pass_index':0}
    # Never carry an earlier projection's activation graph into a later producer.
    previous=backbone.embed(span_ids[:,:-1]).detach()
    gold=previous
    del first
    for depth in range(1,passes):
        producer= prefill_write_context(backbone,heads,backbone.embed(prefix_ids))
        if previous.shape[1]:
            shallow,_=backbone.run_layers(previous,range(0,heads.cutoff),producer.cache)
            sources=torch.cat([producer.state[:,None],shallow],1)
        else:
            sources=producer.state[:,None]
        predictions=heads.feedback(sources).to(gold.dtype)
        replacements=(1-fraction)*gold+fraction*predictions[:,:-1]
        fixed=replacements.detach()
        consumer=prefill_write_context(backbone,heads,backbone.embed(prefix_ids))
        if fixed.shape[1]:
            history,completed=replay_sequence_inputs(
                backbone,heads,consumer,fixed,replacements,group_size=group_size)
            top=torch.cat([consumer.top[:,None],completed],1)
            sources=torch.cat([consumer.state[:,None],history],1)
        else:
            top=consumer.top[:,None];sources=consumer.state[:,None]
        auxiliary=sources.detach()+auxiliary_scale*(sources-sources.detach())
        guesses=heads.feedback(auxiliary)
        yield {'top':top,'sketches':guesses,'pass_index':depth}
        previous=fixed
        del producer,consumer,predictions,replacements,sources,guesses,top


def qualification(report, *, max_ce_delta=.1, max_relative_mse=.25,
                  min_agreement=.9):
    """All held strata must pass; no aggregate can hide a failing stratum."""
    strata=report.get('strata',{})
    return bool(strata) and all(r['tokens']>0 and
        r['ce_delta']<=max_ce_delta and r['embedding_mse_delta']<=max_relative_mse and
        r.get('text_ce_delta_from_initial',0.)<=max_ce_delta and
        r['text_argmax_agreement']>=min_agreement for r in strata.values())


def alignment_selection_score(report, *, max_ce_delta=.1, max_relative_mse=.25, min_agreement=.9):
    """Prefer certified alignment, then the worst held gate ratio, never prompt volume."""
    ratios=[]
    for row in report.get('strata',{}).values():
        if row.get('tokens',0)<=0:return (1,math.inf)
        for value,limit in ((row['ce_delta'],max_ce_delta),
                            (row['embedding_mse_delta'],max_relative_mse),
                            (row.get('text_ce_delta_from_initial',0.),max_ce_delta),
                            (1-row['text_argmax_agreement'],1-min_agreement)):
            if not math.isfinite(value):return (1,math.inf)
            ratios.append(max(0.,value)/limit if limit>0 else (0. if value<=0 else math.inf))
    return (0 if report.get('qualified') else 1,max(ratios,default=math.inf))


def retain_best_checkpoint(out, report):
    """Keep complete optimizer/model state without another GPU serialization.

    Current checkpoints are atomically replaced, so hard links preserve the old
    inode. The receipt is written last and binds both files; consumers verify it
    before using the serving export. Full-state checkpoint remains atomic alone.
    """
    out=Path(out)
    receipt={'schema':'natlang.neuralese-best-warmup-checkpoint/1','step':report['step'],
             'qualification':report,'files':{}}
    for name in ('checkpoint.pt','heads.pt'):
        source=out/name;destination=out/('best-'+name);pending=destination.with_suffix('.pending-link')
        pending.unlink(missing_ok=True)
        os.link(source,pending);pending.replace(destination)
        receipt['files'][destination.name]={'sha256':sha(destination),'bytes':destination.stat().st_size}
    pending=out/'best-checkpoint.json.pending'
    with pending.open('w') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    pending.replace(out/'best-checkpoint.json')


def document_windows(token_ids, *, open_id, close_id, tokens, prefix_tokens, supervised_suffix_start=None):
    """Prime with real open; supervise each text token and the real close once."""
    ids=[open_id]+list(token_ids)+[close_id]
    stride=tokens-prefix_tokens
    windows=[]
    for offset in range(0,len(ids),stride):
        start=max(0,offset-prefix_tokens)
        chunk=ids[start:offset+stride]
        width=1 if offset==0 else offset-start
        if len(chunk)<=width:continue
        window={'ids':chunk,'prefix':width,'offset':offset}
        if supervised_suffix_start is not None:
            window['supervised_suffix_start']=max(0,supervised_suffix_start+1-(start+width))
        windows.append(window)
    return windows


def evaluation_batches(windows, limit):
    """Batch equal geometry within one held stratum; retain every held window."""
    if limit<1:raise ValueError('positive evaluation batch required')
    buckets={}
    for window in windows:
        key=(window['prefix'],len(window['ids']),window['offset']==0)
        buckets.setdefault(key,[]).append(window)
    for bucket in buckets.values():
        for start in range(0,len(bucket),limit):yield bucket[start:start+limit]


def load_text_rows(records, pieces=None, text_data=None, *, tokenizer=None):
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
        rows,_,_,_=gold_text_rows(record_rows,piece_rows,tokenizer=tokenizer)
    if any('supervised_suffix_start' in r and 'token_ids' not in r for r in rows):
        raise ValueError('supervised suffix requires native token IDs')
    encoded=[r for r in rows if 'token_ids' in r]
    if encoded:
        from ..data.text_corpus import tokenizer_fingerprint
        if tokenizer is None:raise ValueError('native gold token IDs require the student tokenizer')
        fingerprint=tokenizer_fingerprint(tokenizer)
        vocab_size=len(tokenizer)
        for row in encoded:
            ids=row['token_ids']
            if row.get('tokenizer_sha256')!=fingerprint or not isinstance(ids,list) or not ids or any(
                    type(i) is not int or i<0 or i>=vocab_size for i in ids):
                raise ValueError('gold token IDs or tokenizer fingerprint mismatch')
            if 'supervised_suffix_start' in row and (type(row['supervised_suffix_start']) is not int or
                    not 0<=row['supervised_suffix_start']<len(ids)):
                raise ValueError('gold supervised suffix coordinate is invalid')
    groups={s:set(g for r in rows if r['split']==s for g in r['source_groups']) for s in ('train','test')}
    if groups['train']&groups['test']:
        raise ValueError('text warm-up factual source groups cross train/test')
    held={r['text'] for r in rows if r['split']=='test'}
    dedup={}; excluded=0
    for row in rows:
        if row['split']=='train' and row['text'] in held:excluded+=1;continue
        dedup.setdefault((row['split'],row['text'],tuple(row.get('token_ids',[])),row.get('supervised_suffix_start')),row)
    rows=list(dedup.values())
    if not all(any(r['split']==s for r in rows) for s in ('train','test')):
        raise ValueError('nonempty independent train and held text required')
    return rows, {'policy':'explicit source-disjoint ordinary text; gold targets, no teacher model',
                  'excluded_train_exact_held_duplicates':excluded,'documents':len(rows)}


def same_alignment_data(previous, current):
    """A corpus handoff remeasures the crisp baseline; path moves do not."""
    def fingerprints(identity):
        return {name: (identity['inputs'].get(identity['options'][name])
                       if identity['options'].get(name) else None)
                for name in ('records','pieces','text_data')}
    return fingerprints(previous) == fingerprints(current)


def configure_student(engine, policy='full', rank=16):
    backbone,heads=engine.backbone,engine.heads
    from .backbone_policy import configure_backbone_training
    backbone_named=configure_backbone_training(backbone,policy,rank=rank)
    for p in heads.parameters():p.requires_grad_(False)
    for p in heads.feedback.parameters():p.requires_grad_(True)
    for p in heads.content.proj.parameters():p.requires_grad_(True)
    # Keep vocabulary/embedding coordinates and output normalization stable.
    backbone.hf.eval();heads.eval()
    named=[('backbone.'+n,p) for n,p in backbone_named]
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
    p.add_argument('--eval-batch',type=int,default=4,help='same-shape held rows per inference batch')
    p.add_argument('--backbone-training',choices=['auto','full','adapters','qat'],default='auto');p.add_argument('--rank',type=int,default=16)
    p.add_argument('--optimizer',choices=['muon','adamw'],default='muon');p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--sketch-lr',type=float,default=3e-4);p.add_argument('--embedding-weight',type=float,default=1.)
    p.add_argument('--sketch-weight',type=float,default=1.);p.add_argument('--text-weight',type=float,default=.25)
    p.add_argument('--projection-patience',type=int,default=3)
    p.add_argument('--projection-min-evals',type=int,default=2)
    p.add_argument('--projection-min-improvement',type=float,default=.01)
    p.add_argument('--backbone-ramp-evals',type=int,default=4)
    p.add_argument('--pass-ramp-evals',type=int,default=2)
    p.add_argument('--checkpoint-every',type=int,default=128);p.add_argument('--eval-every',type=int,default=128)
    p.add_argument('--held-documents',type=int,default=16);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--checkpoint-layers',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--max-ce-delta',type=float,default=.1);p.add_argument('--max-relative-mse',type=float,default=.25)
    p.add_argument('--min-agreement',type=float,default=.9);p.add_argument('--consecutive-gates',type=int,default=2)
    a=p.parse_args(argv)
    if min(a.steps,a.tokens,a.prefix_tokens,a.group_size,a.batch,a.eval_batch,a.eval_every,a.checkpoint_every,a.held_documents,a.consecutive_gates)<1 or a.tokens<3:
        p.error('positive bounds and at least three tokens required')
    if min(a.lr,a.sketch_lr,a.embedding_weight,a.sketch_weight,a.text_weight)<=0:
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
              'text_history':'gold seed; repeated shared shallow sequence passes with aligned predictions',
              'sketch_gradient':'local_stage','sketch_target_backbone_scale':.05,
              'supervision_policy':TEXT_SUPERVISION_POLICY,
              'checkpoint_selection':'qualified first, then worst held gate ratio; complete best full-state and serving-heads hard links'}
    state_path=a.out/'checkpoint.pt'
    resumed=torch.load(state_path,map_location='cpu',weights_only=False,mmap=True) if state_path.exists() else None
    if resumed and (resumed.get('schema')!='natlang.neuralese-text-warmup/1' or resumed['identity']!=identity):raise ValueError('warm-up resume identity changed')
    if a.out.exists() and not resumed:raise ValueError('fresh output or complete checkpoint required')
    continuation=None
    if a.continue_from:
        continuation=torch.load(a.continue_from,map_location='cpu',weights_only=False,mmap=True)
        if continuation.get('schema')!='natlang.neuralese-text-warmup/1':raise ValueError('full text warm-up state required')
        old=continuation['identity']['options']
        if any(old[k]!=options[k] for k in ('optimizer','rank','lr','sketch_lr')):
            raise ValueError('continuation optimizer/parameter policy differs')
    a.out.mkdir(parents=True,exist_ok=True)
    engine,parent=load_initial(a.heads,a.student_checkpoint,a.device,a.cutoff)
    backbone,heads=engine.backbone,engine.heads
    from .backbone_policy import resolve_backbone_policy
    a.backbone_training=resolve_backbone_policy(backbone,a.backbone_training)
    if continuation and resolve_backbone_policy(backbone,continuation['identity']['options']['backbone_training'])!=a.backbone_training:
        raise ValueError('continuation resolved backbone parameter policy differs')
    backbone.checkpoint_layers=a.checkpoint_layers;backbone.ffn_chunk_tokens=1024
    named=configure_student(engine,a.backbone_training,a.rank)
    codes=None
    if a.backbone_training=='qat':
        from ..maple.ternary import CodeTracker
        codes=CodeTracker(backbone.hf)
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
    rows,receipt=load_text_rows(a.records,a.pieces,a.text_data,tokenizer=engine.tokenizer)
    windows={'train':[],'test':[]}
    for row in rows:
        for window in document_windows(row['token_ids'] if 'token_ids' in row else engine._tokens(row['text']),
                open_id=backbone.controls.open_id, close_id=backbone.controls.close_id,
                tokens=a.tokens, prefix_tokens=a.prefix_tokens,
                supervised_suffix_start=row.get('supervised_suffix_start')):
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
        span=ids[:,rows[0]['prefix']:]
        return ids[:,:rows[0]['prefix']],span,balanced_position_weights(span,[r.get('supervised_suffix_start') for r in rows])
    buckets={}
    for window in windows['train']:
        buckets.setdefault((window['prefix'],len(window['ids'])),[]).append(window)
    def objective_pass(out,span,baseline,bootstrap,weights):
        evaluation=not torch.is_grad_enabled()
        top=out['top']
        ce,training_ce,prediction,close_probability,token_losses=chunked_readout(
            backbone,top,span,backbone.controls.close_id,chunk_size=128,
            gradients=not bootstrap,position_weights=weights)
        target=backbone.embed(span).detach()
        embedding_positions,sketch_positions=projection_errors(heads,top,out['sketches'],target)
        embedding,sketch=embedding_positions.mean(),sketch_positions.mean()
        if out['pass_index']==0:
            with torch.no_grad():
                baseline.update(prediction=prediction,ce=ce.detach(),
                    embedding=relative_mse(heads.content.reference(top.detach()),target))
                if evaluation:
                    baseline.update(token_losses=token_losses,
                                    tail_reference=heads.content.reference(top[:,-256:].detach()))
        plain_prediction=baseline['prediction'];plain_ce=baseline['ce'];plain_embedding=baseline['embedding']
        # Both separate projections receive full-strength gold supervision from
        # the first update. CE joins only when the backbone is gently unfrozen.
        supervised_embedding=(embedding_positions*weights).mean()
        supervised_sketch=(sketch_positions*weights).mean()
        loss=a.embedding_weight*supervised_embedding+a.sketch_weight*supervised_sketch
        if not bootstrap:
            loss=loss+training_ce+(a.text_weight*training_ce if out['pass_index']==0 else 0.)
        with torch.no_grad():
            ending=span==backbone.controls.close_id
            close_count=int(ending.sum())
            stop_metrics={'close_targets':close_count,
              'close_probability':float(close_probability[ending].mean()) if close_count else None,
              'close_top1':float((prediction[ending]==backbone.controls.close_id).float().mean()) if close_count else None,
              'premature_close_top1':float((prediction[~ending]==backbone.controls.close_id).float().mean()) if (~ending).any() else 0.}
        metrics={'ce':float(ce.detach()),'text_ce':float(plain_ce.detach()),'ce_delta':float((ce-plain_ce).detach()),
          'relative_mse':float(embedding.detach()),'sketch_mse':float(sketch.detach()),
          'text_embedding_mse':float(plain_embedding.detach()),
          'embedding_mse_delta':float((embedding-plain_embedding).detach()),
          'text_argmax_agreement':float((prediction==plain_prediction).float().mean()),
          'gold_accuracy':float((prediction==span).float().mean()),
          'tokens':span.numel(),'positions':span.shape[1],**stop_metrics}
        metrics.update(supervised_ce=float(training_ce.detach()),
                       supervised_embedding_mse=float(supervised_embedding.detach()),
                       supervised_sketch_mse=float(supervised_sketch.detach()))
        metrics['pass_index']=out['pass_index']
        if evaluation and span.shape[1]>256:
            with torch.no_grad():
                tail_top=top[:,-256:]
                metrics['regions']={'last256':alignment_region_metrics(
                    token_losses[:,-256:],prediction[:,-256:],baseline['token_losses'][:,-256:],
                    plain_prediction[:,-256:],heads.content(torch.zeros_like(tail_top),tail_top),
                    out['sketches'][:,-256:],baseline['tail_reference'],target[:,-256:],span[:,-256:])}
        return loss,metrics

    def objective(w,passes,bootstrap=False):
        prefix,span,weights=ids_for(w);baseline={}
        for out in sequence_completions(backbone,heads,prefix,span,passes=passes,group_size=a.group_size):
            yield objective_pass(out,span,baseline,bootstrap,weights)

    step=0;streak=0;best=None;updates={'backbone':False,'sketch':False,'full_projection':False}
    initial_text_ce={}
    schedule=ProjectionFirstSchedule(min_evals=a.projection_min_evals,patience=a.projection_patience,
        min_relative_improvement=a.projection_min_improvement,
        backbone_ramp_evals=a.backbone_ramp_evals,pass_ramp_evals=a.pass_ramp_evals)
    last_schedule_step=None
    restored=resumed or continuation
    if restored:
        with torch.no_grad():
            for n,v in restored['student_parameters'].items():parameters[n].copy_(v.to(parameters[n]))
        heads.load_state_dict(restored['heads']);optimizer.load_state_dict(restored['optimizer'])
        step=restored['step'];updates=restored['updates']
        updates.setdefault('full_projection',False)
        if resumed:
            streak=resumed['streak'];best=resumed['best'];initial_text_ce=resumed['initial_text_ce']
            schedule.load_state_dict(resumed['schedule'])
            last_schedule_step=resumed['last_schedule_step']
        elif continuation.get('schedule') and continuation['identity']['text_history']==identity['text_history']:
            # A frozen-code throughput handoff preserves the learned phase, not
            # just its weights/optimizer. Re-measure qualification on new code.
            schedule.load_state_dict(continuation['schedule'])
            last_schedule_step=continuation['last_schedule_step']
            if same_alignment_data(continuation['identity'],identity):
                initial_text_ce=continuation['initial_text_ce']
        random.setstate(restored['python_rng']);torch.set_rng_state(restored['torch_rng'])
        if a.device.startswith('cuda'):torch.cuda.set_rng_state_all(restored['cuda_rng'])
    for group in optimizer.param_groups:
        projection=all(any(q is v for n,v in named if n.startswith('heads.')) for q in group['params'])
        group['foundation_base_lr']=a.sketch_lr if projection else a.lr
        group['foundation_projection']=projection
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def log(name,value):
        with (a.out/name).open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    last_report=None
    def evaluate():
        nonlocal last_schedule_step,last_report
        strata={};boundaries={'close_targets':0,'close_probability_sum':0.,'close_top1_sum':0.}
        with torch.no_grad():
            for batch in evaluation_batches(held,a.eval_batch):
                w=batch[0]
                for _,m in objective(batch,3):
                    if m['pass_index']==2:
                        boundaries['close_targets']+=m['close_targets']
                        if m['close_targets']:
                            boundaries['close_probability_sum']+=m['close_probability']*m['close_targets']
                            boundaries['close_top1_sum']+=m['close_top1']*m['close_targets']
                    key='pass-'+str(m['pass_index'])+'-length-'+('short' if m['positions']<=32 else 'medium' if m['positions']<=128 else 'long')+'-'+('start' if w['offset']==0 else 'tail')
                    row=strata.setdefault(key,{'tokens':0})
                    for n in ('ce','text_ce','ce_delta','relative_mse','sketch_mse','text_embedding_mse','embedding_mse_delta','text_argmax_agreement','gold_accuracy'):
                        row[n]=row.get(n,0.)+m[n]*m['tokens']
                    row['tokens']+=m['tokens']
                    for region,values in m.get('regions',{}).items():
                        regional=strata.setdefault(key+'-'+region,{'tokens':0})
                        for n,value in values.items():
                            if n!='tokens':regional[n]=regional.get(n,0.)+value*values['tokens']
                        regional['tokens']+=values['tokens']
        for row in strata.values():
            for n in row.keys()-{'tokens'}:row[n]/=row['tokens']
        for key,row in strata.items():
            initial_text_ce.setdefault(key,row['text_ce'])
            row['text_ce_delta_from_initial']=row['text_ce']-initial_text_ce[key]
        projection_rows=[r for k,r in strata.items() if k.startswith('pass-0-') and not k.endswith('-last256')]
        total=sum(r['tokens'] for r in projection_rows)
        errors={'shallow':sum(r['sketch_mse']*r['tokens'] for r in projection_rows)/total,
                'full_depth':sum(r['relative_mse']*r['tokens'] for r in projection_rows)/total}
        if last_schedule_step is None or step>last_schedule_step:
            schedule.observe(errors);last_schedule_step=step
        from .trajectory_state import weights_digest
        report={'step':step,'strata':strata,'runtime_qualified':False,'autonomous_stopping_qualified':False,
                'boundary_supervision':boundaries,'text_history_policy':identity['text_history'],
                'weights_digest':weights_digest({n:q for n,q in backbone.hf.named_parameters() if n in backbone_names},heads.state_dict()),
                'updates':dict(updates),'schedule':schedule.controls(),'projection_held_errors':errors}
        report['alignment_gate_passed']=qualification(report,max_ce_delta=a.max_ce_delta,max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement)
        if codes is not None:report['qat_codes']=codes.update()
        log('eval.jsonl',report);last_report=report;return report
    def save(report=None):
        atomic_checkpoint(state_path,{'schema':'natlang.neuralese-text-warmup/1','identity':identity,'step':step,
          'student_parameters':{n:q.detach().cpu() for n,q in named},'heads':heads.state_dict(),
          'optimizer':optimizer.state_dict(),'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),
          'cuda_rng':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [],
          'streak':streak,'best':best,'updates':updates,'qualification':report,
          'initial_text_ce':initial_text_ce,'schedule':schedule.state_dict(),
          'last_schedule_step':last_schedule_step})
        # Shared serving heads carry explicit backbone deltas, never inherited certification.
        from .adapters import lora_state,adapter_layers
        initial=torch.load(a.heads,map_location='cpu',weights_only=False,mmap=True)
        exported={**initial,'heads':heads.state_dict(),'control_rows':backbone.control_rows.detach().cpu(),
          'port_config':{'cutoff':heads.cutoff,'max_length':heads.max_length,**heads.port_config()},
          'backbone_trainables':{n:q.detach().cpu() for n,q in backbone.hf.named_parameters() if n in backbone_names},
          'backbone_training':'lora' if a.backbone_training=='adapters' else a.backbone_training,
          'foundation':{'qualified':False,'runtime_qualified':False,'requires_requalification':True},
          **({'maple_qat':True} if a.backbone_training=='qat' else {}),
          'lora':lora_state(backbone),'lora_layers':adapter_layers(backbone),'lora_rank':a.rank,
          'warmup':{'step':step,'identity':identity,'alignment_qualified':bool(report and report.get('qualified')),
                    'report_path':str((a.out/'report.json').resolve()),
                    'report_sha256':sha(a.out/'report.json') if (a.out/'report.json').is_file() else None}}
        # Parent adapters may have a different rank than the fresh-policy default.
        ranks={v.shape[0] for n,v in exported['lora'].items() if '.lora_A.' in n}
        if len(ranks)==1:exported['lora_rank']=next(iter(ranks))
        atomic_checkpoint(a.out/'heads.pt',exported)
    if not resumed:
        baseline=evaluate();(a.out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
        best={'step':step,'score':alignment_selection_score(baseline,max_ce_delta=a.max_ce_delta,max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement),'report':baseline}
        save(baseline);retain_best_checkpoint(a.out,baseline)
    for _ in range(step,a.steps):
        if stop[0]:break
        controls=schedule.controls();bootstrap=not schedule.plateau_reached
        passes=controls['sequence_passes']
        for name,q in named:
            q.requires_grad_(not bootstrap or name.startswith(('heads.feedback.','heads.content.proj.')))
        for group in optimizer.param_groups:
            group['lr']=group['foundation_base_lr']*(1. if group['foundation_projection'] else controls['backbone_lr_scale'])
        w=windows['train'][random.randrange(len(windows['train']))]
        pool=buckets[(w['prefix'],len(w['ids']))]
        batch=[w]+[pool[random.randrange(len(pool))] for _ in range(a.batch-1)]
        optimizer.zero_grad(set_to_none=True);started=time.perf_counter();pass_metrics=[];total_loss=0.
        if a.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            memory_start=torch.cuda.memory_allocated()
        for loss,m in objective(batch,passes,bootstrap):
            if not torch.isfinite(loss):raise RuntimeError('nonfinite warm-up loss')
            (loss/passes).backward()
            total_loss+=float(loss.detach())/passes;pass_metrics.append(m)
        m=dict(pass_metrics[-1])
        backbone_norm=gradient_norm(q for n,q in named if n.startswith('backbone.'))
        sketch_norm=gradient_norm(q for n,q in named if n.startswith('heads.feedback.'))
        clip_finite_gradients(parameters.values())
        # Positive gradients plus real parameter deltas audit both trainable paths.
        samples={k:next((q for n,q in named if n.startswith(prefix) and q.grad is not None and q.grad.abs().sum()>0),None)
                 for k,prefix in [('backbone','backbone.'),('sketch','heads.feedback.'),
                                  ('full_projection','heads.content.proj.')]}
        before={k:q.detach().clone() for k,q in samples.items() if q is not None}
        optimizer.step();step+=1
        for k,v in before.items():updates[k]|=not torch.equal(v,samples[k].detach())
        m.update(step=step,loss=total_loss,seconds=time.perf_counter()-started,
                 phase=controls['phase'],schedule=controls,pass_metrics=pass_metrics,
                 batch=a.batch,
                 backbone_gradient_norm=float(backbone_norm),sketch_gradient_norm=float(sketch_norm),updates=dict(updates))
        if a.device.startswith('cuda'):
            m['memory']={'start_allocated_bytes':memory_start,
                         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                         'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                         'end_allocated_bytes':torch.cuda.memory_allocated()}
        log('train.jsonl',m)
        if step%a.eval_every==0:
            report=evaluate()
            streak=streak+1 if passes==3 and report['alignment_gate_passed'] and all(updates.values()) else 0
            report.update(consecutive_passes=streak,qualified=streak>=a.consecutive_gates,
                          scope='text alignment only; stopping, transport and Natlang tasks unqualified')
            score=alignment_selection_score(report,max_ce_delta=a.max_ce_delta,max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement)
            improved=best is None or score<best['score']
            if improved:best={'step':step,'score':score,'report':report}
            (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');save(report)
            if improved:retain_best_checkpoint(a.out,report)
            if report['qualified']:break
        elif step%a.checkpoint_every==0:save()
    # A signal during the periodic probe must not repeat the same expensive
    # held evaluation before checkpointing exactly the same weights.
    report=dict(last_report) if last_report is not None and last_report['step']==step else evaluate()
    report.update(consecutive_passes=streak,qualified=streak>=a.consecutive_gates,
      updates=updates,status='checkpointed_on_signal' if stop[0] else 'complete',
      scope='text alignment only; stopping, transport and Natlang tasks unqualified')
    score=alignment_selection_score(report,max_ce_delta=a.max_ce_delta,max_relative_mse=a.max_relative_mse,min_agreement=a.min_agreement)
    improved=best is None or score<best['score']
    if improved:best={'step':step,'score':score,'report':report}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');save(report)
    if improved:retain_best_checkpoint(a.out,report)

if __name__=='__main__':main()
