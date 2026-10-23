"""Reference-free offline preference objective for the existing native trainer."""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..common.hashing import canonical_json_sha256_hex
from ..data.native_preference import KIND
from ..rl.native_replay import action_logps,graph_identity

SCHEMA='natlang.native-offline-pairwise-objective/1'
FAILURE_REPAIR_SCHEMA='natlang.native-failure-repair-dpo-objective/1'


class NativePreferenceObjective:
    kind=KIND
    mandatory_progress=True

    def __init__(self,path,*,beta=None,margin=None):
        self.path=Path(path).resolve();self.spec=json.loads(self.path.read_text())
        if self.spec.get('schema')==FAILURE_REPAIR_SCHEMA:
            self._init_failure_repair(beta=beta,margin=margin)
            return
        if self.spec.get('schema')!=SCHEMA or self.spec.get('kind')!=KIND:
            raise ValueError('declared native offline pairwise objective is required')
        self.beta=self.spec.get('beta') if beta is None else beta
        self.margin=self.spec.get('margin') if margin is None else margin
        if (type(self.beta) not in (int,float) or not math.isfinite(self.beta) or self.beta<=0
                or type(self.margin) not in (int,float) or not math.isfinite(self.margin)):
            raise ValueError('offline pairwise beta must be positive and margin finite')
        self.rows=self.spec.get('rows')
        if not isinstance(self.rows,dict) or not self.rows:
            raise ValueError('offline pairwise objective has no jointly admitted pairs')
        chunk=self.spec.get('chunk_tokens',128)
        if type(chunk)is not int or chunk<1:raise ValueError('offline pairwise readout chunk must be positive')
        self.chunk_tokens=chunk

    def _init_failure_repair(self,*,beta=None,margin=None):
        from ..data.native_failure_repair import KIND
        if self.spec.get('kind')!=KIND or self.spec.get('normalization')!='token-mean-v1':
            raise ValueError('genuine failure-repair DPO requires its explicit token-mean reference objective')
        if margin not in (None,0,0.):
            raise ValueError('genuine DPO does not use the reference-free pairwise margin')
        self.mode='failure_repair_dpo';self.kind=KIND
        self.beta=self.spec.get('beta') if beta is None else beta;self.margin=0.
        if type(self.beta) not in (int,float) or not math.isfinite(self.beta) or self.beta<=0:
            raise ValueError('DPO beta must be positive and finite')
        self.rows=self.spec.get('rows')
        if not isinstance(self.rows,dict) or not self.rows:
            raise ValueError('failure-repair DPO objective has no admitted whole-graph pairs')
        identities=set()
        for row in self.rows.values():
            refs=row.get('reference') or {}
            if set(refs)!={'failed','repaired'}:
                raise ValueError('DPO pair must bind both fixed reference-policy alternatives')
            for ref in refs.values():
                if ref.get('normalization')!='token-mean-v1' or not all(ref.get(k) for k in
                    ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256','graph_action_sha256','receipt_sha256')):
                    raise ValueError('DPO fixed reference-policy custody is incomplete')
                receipt=ref.get('receipt')
                if (not isinstance(receipt,dict) or receipt.get('schema')!='natlang.native-fixed-reference-scores/1'
                        or any(receipt.get(k)!=ref.get(k) for k in
                            ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256','graph_action_sha256','normalization'))
                        or canonical_json_sha256_hex(receipt)!=ref.get('receipt_sha256')):
                    raise ValueError('DPO reference score receipt failed its content digest or policy binding')
                identities.add(tuple(ref[k] for k in ('checkpoint_sha256','tokenizer_fingerprint','runtime_sha256')))
            suffixes=row.get('suffixes') or {};counts=row.get('token_counts') or {}
            if set(suffixes)!={'failed','repaired'} or set(counts)!={'failed','repaired'}:
                raise ValueError('DPO pair is missing one complete execution alternative')
            for label in ('failed','repaired'):
                if len(suffixes[label])!=counts[label] or counts[label]<1:
                    raise ValueError('DPO token-mean normalization differs from action inventory')
                receipt=refs[label]['receipt'];actions=receipt.get('actions')
                if (not isinstance(actions,list) or not actions
                        or canonical_json_sha256_hex(actions)!=refs[label]['graph_action_sha256']):
                    raise ValueError('DPO reference action receipt does not match its declared graph/action digest')
                expected=[{'node':label+'/'+action['node'],'position':action['position'],
                    'token_id':action['token_id'],'reference_logp':float(action['reference_logp']),
                    'kind':action['kind']} for action in actions]
                expected.sort(key=lambda action:(action['node'],action['position']))
                if suffixes[label]!=expected:
                    raise ValueError('DPO reference score actions differ from their exact paired graph occurrences')
        if len(identities)!=1:
            raise ValueError('all DPO reference scores must use one fixed checkpoint, tokenizer, and runtime')
        self.reference_identity=next(iter(identities));self.chunk_tokens=self.spec.get('chunk_tokens',128)
        if type(self.chunk_tokens)is not int or self.chunk_tokens<1:raise ValueError('DPO readout chunk must be positive')

    def row_bindings(self):
        return {key:canonical_json_sha256_hex(value) for key,value in self.rows.items()}

    def binding(self):
        from ..common.hashing import sha256_file_hex_cached
        value={'kind':self.kind,'sha256':sha256_file_hex_cached(self.path),
                'beta':self.beta,'margin':self.margin}
        if getattr(self,'mode',None)=='failure_repair_dpo':
            value.update(mode='genuine-reference-dpo',normalization='token-mean-v1',
                         reference_identity=list(self.reference_identity))
        return value

    def forward_inputs(self,context):
        if getattr(self,'mode',None)=='failure_repair_dpo':
            for example in context['examples']:
                row=self.rows.get(example.id)
                if row is None or row.get('graph_sha256')!=canonical_json_sha256_hex(graph_identity(example)):
                    raise ValueError('DPO objective changed its exact paired graph binding')
                if any(example.target_weights) or example.split!='train' or row.get('split')!='train':
                    raise ValueError('DPO pairs require zero CE targets and training-only graph custody')
                if set(row.get('suffixes') or {})!={'failed','repaired'}:
                    raise ValueError('DPO objective requires both complete alternatives in one graph')
                names={node.name for node in example.plan.invocations}
                expected={f'{label}/{action["node"].split("/",1)[-1]}' for label in ('failed','repaired')
                          for action in row['suffixes'][label]}
                if not expected <= names:
                    raise ValueError('DPO action occurrence inventory differs from its complete graph')
            return {}
        for example in context['examples']:
            row=self.rows.get(example.id)
            if row is None or row.get('graph_sha256')!=canonical_json_sha256_hex(graph_identity(example)):
                raise ValueError('offline preference objective changed its exact graph/suffix source binding')
            expected_id='offline-preference:'+canonical_json_sha256_hex({
                'source_record_sha256':row.get('source_record_sha256'),
                'suffixes':row.get('suffixes'),'token_counts':row.get('token_counts')})
            if example.id!=expected_id:
                raise ValueError('offline preference suffix mask differs from its immutable graph identity')
            if any(example.target_weights) or example.external_states:
                raise ValueError('offline preference must have zero ordinary CE targets and no external actor state')
            if row.get('split')!=example.split or example.split not in ('train','test'):
                raise ValueError('offline preference source split differs from graph')
            if example.split!='train' and torch.is_grad_enabled():
                raise ValueError('protected offline preference cannot enter a training update')
            names={node.name for node in example.plan.invocations}
            if set(row.get('suffixes') or {})!=names:
                raise ValueError('offline preference suffix inventory differs from actual invocations')
            counts={'chosen':0,'rejected':0}
            for node in example.plan.invocations:
                suffix=row['suffixes'][node.name];start=suffix.get('suffix_start');stop=suffix.get('suffix_stop')
                if (suffix.get('alternative') not in counts or type(start)is not int or type(stop)is not int
                        or not 0<start<stop==len(node.token_ids)
                        or suffix.get('token_ids_sha256')!=canonical_json_sha256_hex(list(node.token_ids))
                        or node.imports or node.producing or node.logical_producing):
                    raise ValueError('offline preference suffix is not the exact original assistant continuation')
                counts[suffix['alternative']]+=stop-start
            if counts!=row.get('token_counts') or min(counts.values())<1:
                raise ValueError('offline preference alternative token-count normalization differs')
        return {}

    def __call__(self,model,result,context):
        if getattr(self,'mode',None)=='failure_repair_dpo':
            hidden=result['h_final'][0];losses=[];gaps=[];token_total=0
            for example in context['examples']:
                row=self.rows[example.id];offsets={node.name:off for node,off in
                    zip(example.plan.invocations,example.plan.offsets)}
                first=context['row_offsets'][example.id]['physical_start']
                scores={}
                for label in ('failed','repaired'):
                    actions=[];reference=[]
                    for item in row['suffixes'][label]:
                        node=f'{label}/{item["node"].split("/",1)[-1]}';position=item['position']
                        if node not in offsets:
                            # Admission stores prefixed node names; tolerate either bound form.
                            node=item['node']
                        actions.append({'query':first+offsets[node]+position-1,'token_id':item['token_id'],
                            'temperature':1.,'allowed_token_ids':None,'banned_token_ids':None,'behavior_logp':0.})
                        reference.append(float(item['reference_logp']))
                    logps,_=action_logps(model,hidden,actions,chunk_tokens=self.chunk_tokens)
                    scores[label]=logps.mean()-torch.tensor(reference,device=hidden.device,dtype=logps.dtype).mean()
                    token_total+=len(actions)
                # repaired is the preferred completion; beta times the chosen/rejected
                # policy-minus-fixed-reference log-ratio is the genuine DPO margin.
                gap=scores['repaired']-scores['failed'];gaps.append(gap.detach())
                losses.append(-F.logsigmoid(self.beta*gap))
            value=torch.stack(gaps)
            return torch.stack(losses).mean(),{'objective':self.kind,'preference_pairs':len(losses),
                'scored_action_tokens':token_total,'mean_reference_adjusted_logp_gap':float(value.mean()),
                'preference_accuracy':float((value>0).float().mean()),'beta':self.beta,
                'normalization':'token-mean-v1','reference_identity':list(self.reference_identity)}
        hidden=result['h_final'][0];losses=[];scores=[];tokens=0
        for example in context['examples']:
            row=self.rows[example.id];actions=[];owners=[]
            first=context['row_offsets'][example.id]['physical_start']
            for node,offset in zip(example.plan.invocations,example.plan.offsets):
                suffix=row['suffixes'][node.name]
                for position in range(suffix['suffix_start'],suffix['suffix_stop']):
                    actions.append({'query':first+offset+position-1,'token_id':node.token_ids[position],
                        'temperature':1.,'allowed_token_ids':None,'banned_token_ids':None,'behavior_logp':0.})
                    owners.append(suffix['alternative'])
            logps,_=action_logps(model,hidden,actions,chunk_tokens=self.chunk_tokens)
            means={name:logps[torch.tensor([owner==name for owner in owners],device=hidden.device)].sum()/row['token_counts'][name]
                   for name in ('chosen','rejected')}
            difference=means['chosen']-means['rejected']
            losses.append(-F.logsigmoid(self.beta*difference-self.margin));scores.append(difference.detach())
            tokens+=len(actions)
        if not losses:raise ValueError('offline preference batch has no jointly scored pairs')
        gaps=torch.stack(scores)
        return torch.stack(losses).mean(),{'objective':KIND,'preference_pairs':len(losses),
            'scored_assistant_tokens':tokens,'mean_logp_gap':float(gaps.mean()),
            'preference_accuracy':float((gaps>0).float().mean()),'beta':self.beta,'margin':self.margin}

    def evaluate(self,model,example,*,checkpoint_layers,chunk_tokens,boundary_ids):
        """Held pairs use the same one-pass objective; ordinary held CE stays CE."""
        if getattr(self,'mode',None)=='failure_repair_dpo':
            if example.id not in self.rows:
                if not any(example.target_weights):
                    raise ValueError('held failure-repair DPO lacks its exact paired objective row')
                from ..data.native_graph import graph_loss
                return graph_loss(model,example,checkpoint_layers=checkpoint_layers,
                                  chunk_tokens=chunk_tokens,boundary_token_ids=boundary_ids)
            from .native_state_sft import packed_graph_loss
            loss,detail=packed_graph_loss(model,[example],checkpoint_layers=checkpoint_layers,
                chunk_tokens=chunk_tokens,objective=self)
            return loss,{'weight':1.,'accuracy':detail['preference_accuracy'],
                'supervised_tokens':detail['scored_action_tokens'],
                'boundary_sums':{str(token):dict.fromkeys(('count','weight','nll_sum','correct_weight','probability_sum'),0.)
                                 for token in boundary_ids},'metric':self.kind}
        if example.id not in self.rows:
            if not any(example.target_weights):
                raise ValueError('held offline preference lacks its exact jointly bound objective row')
            from ..data.native_graph import graph_loss
            return graph_loss(model,example,checkpoint_layers=checkpoint_layers,
                              chunk_tokens=chunk_tokens,boundary_token_ids=boundary_ids)
        from .native_state_sft import packed_graph_loss
        loss,detail=packed_graph_loss(model,[example],checkpoint_layers=checkpoint_layers,
                                     chunk_tokens=chunk_tokens,objective=self)
        return loss,{'weight':1.,'accuracy':detail['preference_accuracy'],
            'supervised_tokens':detail['scored_assistant_tokens'],
            'boundary_sums':{str(token):dict.fromkeys(('count','weight','nll_sum','correct_weight','probability_sum'),0.)
                             for token in boundary_ids},'metric':KIND}


def main(argv=None):
    """Admission adapter only: the existing native trainer owns every update."""
    import sys
    import argparse
    arguments=list(sys.argv[1:] if argv is None else argv)
    if '--help' not in arguments and '-h' not in arguments:
        parser=argparse.ArgumentParser(add_help=False)
        parser.add_argument('--objective-spec',required=True)
        known,_=parser.parse_known_args(arguments)
        value=json.loads(Path(known.objective_spec).read_text())
        if not ((value.get('schema')==SCHEMA and value.get('kind')==KIND) or
                (value.get('schema')==FAILURE_REPAIR_SCHEMA and value.get('kind')=='native-failure-repair-dpo')):
            raise ValueError('offline preference stage requires its exact jointly bound objective')
        if '--objective-continuation' not in arguments:arguments.append('--objective-continuation')
        if '--stage-id' not in arguments:arguments+=['--stage-id',KIND]
    from .native_state_sft import main as native_main
    return native_main(arguments)


if __name__=='__main__':raise SystemExit(main())
