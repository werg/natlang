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


class NativePreferenceObjective:
    kind=KIND
    mandatory_progress=True

    def __init__(self,path,*,beta=None,margin=None):
        self.path=Path(path).resolve();self.spec=json.loads(self.path.read_text())
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

    def binding(self):
        from ..common.hashing import sha256_file_hex_cached
        return {'kind':self.kind,'sha256':sha256_file_hex_cached(self.path),
                'beta':self.beta,'margin':self.margin}

    def forward_inputs(self,context):
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
        if value.get('schema')!=SCHEMA or value.get('kind')!=KIND:
            raise ValueError('offline preference stage requires its exact jointly bound objective')
        if '--objective-continuation' not in arguments:arguments.append('--objective-continuation')
        if '--stage-id' not in arguments:arguments+=['--stage-id',KIND]
    from .native_state_sft import main as native_main
    return native_main(arguments)


if __name__=='__main__':raise SystemExit(main())
