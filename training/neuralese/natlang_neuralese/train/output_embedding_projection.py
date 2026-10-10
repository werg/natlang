"""Experimental same-position inverse: output state at i -> input embedding of token i.

This is NOT causal next-token feedback initialization or a qualified foundation.
It fits a new isolated projection;
legacy port heads, the backbone and existing checkpoints are not overwritten.
Random ordinary-token contexts supplement admitted train-source texts. Held source
texts are used only for reconstruction evaluation, never optimization.
"""
import argparse, hashlib, json, signal
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ..serve import load_engine
from .trajectories import target_write, handover_notes
from .checkpoint_policy import publish_best
from .trajectory_state import atomic_checkpoint
from ..common.hashing import sha256_file_hex as sha


class OutputEmbeddingProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.hidden = nn.Linear(dim, 2 * dim)
        self.residual = nn.Linear(2 * dim, dim)
        nn.init.zeros_(self.residual.weight); nn.init.zeros_(self.residual.bias)
    def forward(self, states):
        return self.linear(states.float()) + self.residual(F.gelu(self.hidden(states.float())))


def source_texts(records):
    result = {'train':set(), 'test':set()}
    for r in map(json.loads, Path(records).open()):
        name = target_write(r)
        if name and r.get('training_admission',{}).get('approved') is True:
            result[r['split']].add(handover_notes(r)[name])
    return {s:sorted(values,key=lambda x:hashlib.sha256(x.encode()).hexdigest()) for s,values in result.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heads',type=Path,required=True);p.add_argument('--records',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=4096);p.add_argument('--batch',type=int,default=2048);p.add_argument('--lr',type=float,default=0.0003)
    p.add_argument('--train-sources',type=int,default=96);p.add_argument('--held-sources',type=int,default=32)
    p.add_argument('--random-contexts',type=int,default=128);p.add_argument('--context-tokens',type=int,default=256);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--device',default='cuda');p.add_argument('--eval-every',type=int,default=256);p.add_argument('--checkpoint-every',type=int,default=128)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(a.seed)
    if min(a.steps,a.batch,a.context_tokens,a.eval_every,a.checkpoint_every)<1:raise ValueError('positive run bounds required')
    identity={'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items() if k!='out'},'sha256':{str(x):sha(x) for x in [a.heads,a.records,Path(__file__)]}}
    a.out.mkdir(parents=True,exist_ok=True);state_path=a.out/'checkpoint.pt'
    resumed=torch.load(state_path,map_location='cpu',weights_only=False) if state_path.exists() else None
    if resumed and resumed['identity']!=identity:raise ValueError('projection resume identity changed')
    if not resumed and (a.out/'plan.json').exists():raise ValueError('partial run without checkpoint; use a new output')
    (a.out/'plan.json').write_text(json.dumps(identity,indent=2)+'\n')
    engine=load_engine(heads_checkpoint=str(a.heads),device=a.device);backbone=engine.backbone
    for parameter in backbone.parameters():parameter.requires_grad_(False)
    texts=source_texts(a.records);generator=torch.Generator().manual_seed(a.seed)
    special=set(engine.tokenizer.all_special_ids)|set(backbone.controls.__dict__.values())
    ordinary=torch.tensor([i for i in range(backbone.embedding_weight.shape[0]) if i not in special])
    prefix=[engine.tokenizer.bos_token_id] if engine.tokenizer.bos_token_id is not None else []
    def states_and_targets(ids):
        ids=torch.tensor([prefix+ids],device=a.device)
        with torch.no_grad():
            output=backbone.forward_ids(ids,logits=False)['h_final'][0,len(prefix):].float()
            target=backbone.embed(ids)[0,len(prefix):].float()
        return output.detach(),target.detach()
    pairs={}
    for split,bound in [('train',a.train_sources),('test',a.held_sources)]:
        values=[]
        for i,text in enumerate(texts[split][:bound]):
            ids=engine._tokens(text)[:a.context_tokens]
            if ids:values.append(states_and_targets(ids))
        # Random token contexts are synthetic reconstruction examples, not teacher trajectories.
        count=a.random_contexts if split=='train' else max(4,a.random_contexts//8)
        for _ in range(count):
            ids=ordinary[torch.randint(len(ordinary),(a.context_tokens,),generator=generator)].tolist()
            values.append(states_and_targets(ids))
        pairs[split]=tuple(torch.cat([v[k] for v in values]) for k in [0,1])
        print(json.dumps({'prepared':split,'tokens':len(pairs[split][0]),'admitted_source_texts':min(bound,len(texts[split]))}),flush=True)
    x,y=pairs['train'];dim=x.shape[1];projection=OutputEmbeddingProjection(dim).to(a.device)
    # Supervised ridge initialization on train data only, then learn nonlinear residuals.
    with torch.no_grad():
        xm,ym=x.mean(0),y.mean(0);xc,yc=x-xm,y-ym
        covariance=xc.T@xc;ridge=covariance.diag().mean()*0.001
        weights=torch.linalg.solve(covariance+torch.eye(dim,device=a.device)*ridge,xc.T@yc).T
        projection.linear.weight.copy_(weights);projection.linear.bias.copy_(ym-weights@xm)
    optimizer=torch.optim.AdamW(projection.parameters(),lr=a.lr,weight_decay=0.01)
    start=0
    if resumed:
        projection.load_state_dict(resumed['projection']);optimizer.load_state_dict(resumed['optimizer']);generator.set_state(resumed['generator']);start=resumed['step']
    stop=[False]
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    scale=y.square().mean().clamp_min(1e-12);best=resumed.get('best') if resumed else None
    def evaluate(step):
        with torch.no_grad():
            xx,yy=pairs['test'];pred=projection(xx)
            metrics={'step':step,'relative_mse':float((pred-yy).square().mean()/yy.square().mean()),'cosine':float(F.cosine_similarity(pred,yy).mean()),'held_tokens':len(xx)}
        with (a.out/'eval.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True);return metrics
    def save(step,path):
        atomic_checkpoint(path,{'schema':'natlang.output-embedding-projection/1','identity':identity,'step':step,'dim':dim,'projection':projection.state_dict(),'optimizer':optimizer.state_dict(),'generator':generator.get_state(),'best':best,'backbone_heads':str(a.heads),'backbone_heads_sha256':sha(a.heads),'target':'raw input embedding at the same token position; not next-token feedback','read_mode':'transparent; no interface normalization or port markers','qualification':'reconstruction experiment only; requires held task/logit parity before integration'})
    for step in range(start,a.steps):
        indices=torch.randint(len(x),(a.batch,),generator=generator).to(a.device);pred=projection(x[indices]);target=y[indices]
        loss=(pred-target).square().mean()/scale+0.1*(1-F.cosine_similarity(pred,target).mean())
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(projection.parameters(),1.0);optimizer.step()
        if (step+1)%64==0:print(json.dumps({'step':step+1,'loss':float(loss.detach())}),flush=True)
        # Declared step points only (plans/STORAGE_POLICY.md): best = the best among full-state writes, by a hard link.
        metrics=evaluate(step+1) if (step+1)%a.eval_every==0 else None
        if (step+1)%a.checkpoint_every==0 or stop[0] or step+1==a.steps:
            improved=metrics is not None and (best is None or metrics['relative_mse']<best['relative_mse'])
            if improved:best=metrics
            save(step+1,state_path)
            if improved:publish_best(state_path,a.out/'best-checkpoint.pt')
        if stop[0]:break
    print(json.dumps({'status':'checkpointed_on_signal' if stop[0] else 'complete','step':step+1,'best':best}),flush=True)
if __name__=='__main__':main()
