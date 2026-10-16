"""Hash-bound inline objectives over ONE authenticated native graph forward.

This is the exact internal lowering target for recorded learning, not an
assertion that an arbitrary public GradSession has already been translated.
"""
from __future__ import annotations
import copy
import math
import torch
from ..common.hashing import canonical_json_sha256_hex
from ..common.sampling import token_mask
from ..rl.native_replay import token_action_logps
from ..common.decision_objective import decision_loss

SCHEMA = 'natlang.native-query-objective/1'




class NativeQueryObjective:
    def __init__(self, spec, graph, example, *, vocabulary):
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
        for term in terms:
            if not isinstance(term, dict) or type(term.get('weight',1.)) not in (float,int) or not math.isfinite(term.get('weight',1.)):
                raise ValueError('native query term needs a finite signed weight')
            kind = term.get('kind')
            if kind == 'nll' and not set(term)-{'kind','actions','reduction','weight'}:
                if term.get('reduction') not in {'mean','sum'}:
                    raise ValueError('native NLL reduction must be mean or sum')
                prepared = {**term,'actions':actions(term.get('actions'))}
            elif kind == 'decision' and not set(term)-{'kind','options','rule','probabilities','rewards','weight'}:
                options = term.get('options')
                if not isinstance(options,list) or not options:
                    raise ValueError('native decision needs nonempty option continuations')
                prepared = {**term,'options':[actions(option) for option in options]}
                decision_loss(torch.zeros(len(options)),rule=term.get('rule'),
                              probabilities=term.get('probabilities'),rewards=term.get('rewards'))
            elif kind == 'distill' and not set(term)-{'kind','actions','teacher_actions','weight'}:
                student,teacher = actions(term.get('actions')),actions(term.get('teacher_actions'))
                if (len(student)!=len(teacher) or any(a['token_id']!=b['token_id'] or
                        a['temperature']!=1 or b['temperature']!=1 or
                        a['allowed_token_ids'] is not None or b['allowed_token_ids'] is not None or
                        a['banned_token_ids'] is not None or b['banned_token_ids'] is not None
                        for a,b in zip(student,teacher))):
                    raise ValueError('native distillation requires aligned ordinary conditional token distributions')
                prepared = {**term,'actions':student,'teacher_actions':teacher}
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
                logps = token_action_logps(model,hidden,term['actions'])
                value = -logps.mean() if term['reduction']=='mean' else -logps.sum()
            elif kind == 'decision':
                scores = torch.stack([token_action_logps(model,hidden,option).sum() for option in term['options']])
                value = decision_loss(scores,rule=term['rule'],probabilities=term.get('probabilities'),rewards=term.get('rewards'))
            else:
                from torch.utils.checkpoint import checkpoint
                def distill(student, teacher):
                    with torch.no_grad():
                        goal = model.logits(teacher).float().log_softmax(-1)
                    predicted = model.logits(student).float().log_softmax(-1)
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
