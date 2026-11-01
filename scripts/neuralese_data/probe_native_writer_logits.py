"""Replay a recorded real native writer prefix and measure exact marker/EOS logits.

Run only under an authorized model allocation. This never updates or samples a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def rank_row(logits, ids):
    values = logits.detach().float().flatten()
    ranked = {}
    for label, token in ids.items():
        value = values[token]
        ranked[label] = {'token_id':token, 'logit':float(value),
                         'rank':1+int((values > value).sum()),
                         'log_probability':float(value-torch.logsumexp(values,0))}
    ranked['argmax'] = int(values.argmax())
    ranked['top10'] = [{'token_id':int(i), 'logit':float(values[i])}
                       for i in torch.topk(values,min(10,len(values))).indices]
    return ranked


def replay(engine, capture, *, perturb=False):
    prompt=[]
    for segment in capture['prompt']:
        if segment['kind'] != 'tokens':
            raise ValueError('diagnostic requires a recorded writer prompt without imported blocks')
        prompt.extend(segment['token_ids'])
    if not prompt: raise ValueError('empty recorded prompt')
    if perturb:
        # Deliberately change a single private prefix token, not a production request.
        prompt[0] = (prompt[0]+1) % engine.shared_model.port.embedding_weight.shape[0]
    forced=[x for x in capture['generated_inputs'] if x['sample_index'] is None]
    opened=engine.role_tokens.produce_open
    if not forced or forced[-1]['token_id'] != opened or forced[-1]['input_role'] != 'ordinary':
        raise ValueError('recording does not end forced prefix at a native producer-open token')
    seq=engine._native_prefill_forward(prompt,None)
    cache=seq['cache']
    for item in forced:
        if item['input_role']!='ordinary': raise ValueError('unexpected producer role in forced prefix')
        seq=engine._native_forward([item['token_id']],cache,producing=False);cache=seq['cache']
    selected={'produce_close':engine.role_tokens.produce_close,'eos':capture['protocol_failure']['stop_token_ids'][0]}
    points=[]; samples=capture['samples']
    for sample in samples:
        logits=engine._native_logits(seq['h_final'])
        point=rank_row(logits,selected)
        point.update(index=sample['index'],recorded_token=sample['token_id'],
                     greedy_matches_recording=(point['argmax']==sample['token_id']))
        points.append(point)
        if sample['terminal']:break
        produced=sample['input_role']=='produce'
        seq=engine._native_forward([sample['token_id']],cache,producing=produced);cache=seq['cache']
    return points


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--turns',required=True)
    p.add_argument('--expected-checkpoint',required=True);p.add_argument('--actor-receipt',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--prefill-tokens',type=int,default=8192)
    p.add_argument('--perturb-private-prefix',action='store_true');a=p.parse_args()
    from natlang_neuralese.serve import load_native_engine
    data=Path(a.turns).read_bytes();turns=json.loads(data)
    turn=next(t for t in turns if t.get('native',{}).get('sampling',{}).get('protocol_failure'))
    capture=turn['native']['sampling']
    if capture['policy']['checkpoint_sha256']!=a.expected_checkpoint: raise ValueError('recorded checkpoint mismatch')
    engine=load_native_engine(a.checkpoint,device=a.device,prefill_tokens=a.prefill_tokens)
    if engine.native_identity != capture['policy']: raise ValueError('loaded actor identity differs from recorded policy')
    actor=json.loads(Path(a.actor_receipt).read_text())['info']
    if engine.serving_config != actor.get('serving_config'):
        raise ValueError('loaded serving configuration differs from recorded actor')
    with torch.inference_mode():
        baseline=replay(engine,capture)
        changed=replay(engine,capture,perturb=True) if a.perturb_private_prefix else None
    receipt={'schema':'natlang.native-writer-logit-diagnostic/1','checkpoint':a.expected_checkpoint,
       'recorded_turns_sha256':hashlib.sha256(data).hexdigest(),'actor_identity':engine.native_identity,
       'serving_config':engine.serving_config,'prefill_tokens':a.prefill_tokens,
       'baseline':baseline,'perturbed_first_private_token':changed,
       'scope':'exact recorded writer prefix replay; changed-prefix result is causal diagnosis only, no SGD or task outcome'}
    Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'output':a.output,'greedy_match':all(x['greedy_matches_recording'] for x in baseline)}))
    if not all(x['greedy_matches_recording'] for x in baseline): raise ValueError('replayed greedy choices differ from recorded actor')


if __name__=='__main__':main()
