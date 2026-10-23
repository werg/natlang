"""Hash-bound inline objectives over one authenticated native graph forward."""
from __future__ import annotations
import copy
import math
import torch
from ..common.hashing import canonical_json_sha256_hex
from ..common.sampling import token_mask
from ..rl.native_replay import token_action_logps
from ..common.decision_objective import decision_loss, decision_loss_to_goal

SCHEMA = 'natlang.native-query-objective/1'




class NativeQueryObjective:
    def __init__(self, spec, graph, example, *, vocabulary, resolve_readout=None):
        if (not isinstance(spec, dict) or set(spec) != {'kind','schema','graph_sha256','terms'}
                or spec['kind'] != 'native-query' or spec['schema'] != SCHEMA
                or spec['graph_sha256'] != canonical_json_sha256_hex(graph)):
            raise ValueError('native query objective must bind its exact graph and schema')
        self.spec = copy.deepcopy(spec)
        self.example = example
        next_tokens = {offset+local-1: node.token_ids[local]
            for node, offset in zip(example.plan.invocations, example.plan.offsets)
            for local in range(1, len(node.token_ids))
            if local not in {edge.at for edge in node.imports if not edge.logical}
            and example.plan.own_positions[offset+local]==example.plan.own_positions[offset+local-1]+1}
        def actions(values):
            if not isinstance(values, list) or not values:
                raise ValueError('native query term requires actual next-own-token queries')
            result = []
            for value in values:
                if (not isinstance(value, dict) or set(value)-{'query','token_id','temperature','allowed_token_ids','banned_token_ids'}
                        or type(value.get('query')) is not int or type(value.get('token_id')) is not int
                        or next_tokens.get(value['query']) != value['token_id']):
                    raise ValueError('native query must match its actual next own input token')
                temperature = value.get('temperature', 1.)
                # Reuse the exact serving/RL mask verifier before model forward.
                if type(temperature) not in (int,float) or not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError('native query temperature must be finite and positive')
                allowed,banned=token_mask(vocabulary,allowed=value.get('allowed_token_ids'),banned=value.get('banned_token_ids'))
                if (not 0 <= value['token_id'] < vocabulary or
                        allowed is not None and value['token_id'] not in allowed or
                        banned is not None and value['token_id'] in banned):
                    raise ValueError('native query mask excludes its target token')
                result.append({**value, 'temperature':temperature,
                    'allowed_token_ids':value.get('allowed_token_ids'), 'banned_token_ids':value.get('banned_token_ids')})
            return result
        terms = spec['terms']
        if not isinstance(terms, list) or not terms:
            raise ValueError('native query objective needs finite declared terms')
        self.terms = []
        self.readouts = {}
        def readout(ident):
            if ident is None:return
            from ..model.native_derivative_contract import native_id
            if not native_id(ident,"nnt1_") or resolve_readout is None:
                raise ValueError("native query needs its exact immutable readout resolver")
            if ident not in self.readouts:self.readouts[ident]=resolve_readout(ident)
        for term in terms:
            if not isinstance(term, dict) or type(term.get('weight',1.)) not in (float,int) or not math.isfinite(term.get('weight',1.)):
                raise ValueError('native query term needs a finite signed weight')
            kind = term.get('kind')
            if kind == 'nll' and not set(term)-{'kind','actions','reduction','weight','readout'}:
                if term.get('reduction') not in {'mean','sum'}:
                    raise ValueError('native NLL reduction must be mean or sum')
                readout(term.get('readout'))
                prepared = {**term,'actions':actions(term.get('actions'))}
            elif kind == 'decision' and not set(term)-{'kind','options','rule','probabilities','rewards','weight','readouts'}:
                options = term.get('options')
                if not isinstance(options,list) or not options:
                    raise ValueError('native decision needs nonempty option continuations')
                declared=term.get('readouts',[None]*len(options))
                if not isinstance(declared,list) or len(declared)!=len(options):
                    raise ValueError('native decisions need one declared readout per option')
                for ident in declared:readout(ident)
                prepared = {**term,'options':[actions(option) for option in options],'readouts':declared}
                decision_loss(torch.zeros(len(options)),rule=term.get('rule'),
                              probabilities=term.get('probabilities'),rewards=term.get('rewards'))
            elif kind == 'distill' and not set(term)-{'kind','actions','teacher_actions','weight','readout','teacher_readout'}:
                student,teacher = actions(term.get('actions')),actions(term.get('teacher_actions'))
                if (len(student)!=len(teacher) or any(a['token_id']!=b['token_id'] or
                        a['temperature']!=1 or b['temperature']!=1 or
                        a['allowed_token_ids'] is not None or b['allowed_token_ids'] is not None or
                        a['banned_token_ids'] is not None or b['banned_token_ids'] is not None
                        for a,b in zip(student,teacher))):
                    raise ValueError('native distillation requires aligned ordinary conditional token distributions')
                readout(term.get('readout'));readout(term.get('teacher_readout'))
                prepared = {**term,'actions':student,'teacher_actions':teacher}
            elif kind == 'decision-distill' and not set(term)-{'kind','options','teacher_options','rule','weight','readouts','teacher_readouts'}:
                options, teacher = term.get('options'),term.get('teacher_options')
                if not isinstance(options,list) or not options or not isinstance(teacher,list) or len(teacher)!=len(options):
                    raise ValueError('native decision teacher must have the same finite options')
                selected=[]
                for key in ('readouts','teacher_readouts'):
                    declared=term.get(key,[None]*len(options))
                    if not isinstance(declared,list) or len(declared)!=len(options):
                        raise ValueError('native decision teacher needs one exact readout per option')
                    for ident in declared:readout(ident)
                    selected.append(declared)
                if term.get('rule') not in ('logLoss','brier','rps'):
                    raise ValueError('native decision distillation requires a proper scoring rule')
                prepared={**term,'options':[actions(option) for option in options],
                    'teacher_options':[actions(option) for option in teacher],
                    'readouts':selected[0],'teacher_readouts':selected[1]}
            else:
                raise ValueError('native query kind or fields are unsupported')
            self.terms.append(prepared)

    def __call__(self, model, example, result):
        if example is not self.example or result['plan'] is not example.plan:
            raise ValueError('native query objective changed its bound graph forward')
        hidden = result['h_final'][0]
        values = []
        for term in self.terms:
            kind = term['kind']
            if kind == 'nll':
                logps = token_action_logps(model,hidden,term['actions'],readout=self.readouts.get(term.get('readout')))
                value = -logps.mean() if term['reduction']=='mean' else -logps.sum()
            elif kind == 'decision':
                scores = torch.stack([token_action_logps(model,hidden,option,readout=self.readouts.get(ident)).sum()
                    for option,ident in zip(term['options'],term['readouts'])])
                value = decision_loss(scores,rule=term['rule'],probabilities=term.get('probabilities'),rewards=term.get('rewards'))
            elif kind=='decision-distill':
                scores=torch.stack([token_action_logps(model,hidden,option,readout=self.readouts.get(ident)).sum()
                    for option,ident in zip(term['options'],term['readouts'])])
                with torch.no_grad():
                    teacher=torch.stack([token_action_logps(model,hidden.detach(),option,readout=self.readouts.get(ident)).sum()
                        for option,ident in zip(term['teacher_options'],term['teacher_readouts'])]).softmax(0)
                value=decision_loss_to_goal(scores,teacher,rule=term['rule'])
            else:
                from torch.utils.checkpoint import checkpoint
                student_read=self.readouts.get(term.get('readout'),model.logits)
                teacher_read=self.readouts.get(term.get('teacher_readout'),model.logits)
                def distill(student, teacher,student_read=student_read,teacher_read=teacher_read):
                    with torch.no_grad():
                        goal = teacher_read(teacher).float().log_softmax(-1)
                    predicted = student_read(student).float().log_softmax(-1)
                    return (goal.exp()*(goal-predicted)).sum()
                from ..common.native_higher_order import capture_reference_execution,active_reference_arithmetic
                if active_reference_arithmetic() is not None:
                    from ..model.native_adapters import capture_native_execution
                    distill = capture_reference_execution(capture_native_execution(distill))
                chunks = []
                for start in range(0,len(term['actions']),128):
                    own = [action['query'] for action in term['actions'][start:start+128]]
                    teacher = [action['query'] for action in term['teacher_actions'][start:start+128]]
                    chunks.append(checkpoint(distill,hidden[own],hidden[teacher].detach(),use_reentrant=False))
                value = torch.stack(chunks).sum()/len(term['actions'])
            values.append(value*term.get('weight',1.))
        return torch.stack(values).sum(), {'objective':SCHEMA,'terms':len(values)}
