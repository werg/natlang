#!/usr/bin/env python3
"""Diagnostic-only real native producer/import/decision packed replay; no SGD.

Run under the exact serving source inventory of a bound actor transport. The one
loaded inference model generates real actions, writes its actual returned text as
a typed state, scores and samples a decision consuming that state, then replays
the complete captured graph with the production packed policy objective.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from contextlib import nullcontext
from pathlib import Path
import torch
from natlang_neuralese.common.jsonio import write_json_durable, write_bytes_durable
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.rl.decision_policy import sample_receipt
from natlang_neuralese.rl.episode_graph import whole_episode_turns
from natlang_neuralese.rl.native_replay import compile_replay, NativePolicyObjective, graph_identity
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.serve.native_learning import decide_native
from natlang_neuralese.model.native_graph_plan import invocation_dict
from natlang_neuralese.data.native_graph import SCHEMA, parse_graph
from natlang_neuralese.train.native_state_sft import packed_graph_loss
from natlang_neuralese.model.native_weight_carrier import plain_training_arithmetic

EXPECTED='bf95bec416a1bb062999baa85df3f0d692220ef88575c22099dce008877c8960'

def collect(engine, *, seed=86122, max_tokens=96, options=None, out=None):
    """Collect genuine sampled producer output and genuine sampled decision."""
    if engine.native_identity.get('checkpoint_sha256')!=EXPECTED:raise ValueError('diagnostic actor is not exact BF95')
    if engine.model_config.get('native_shared_execution')!='ffn-stable-v1':raise ValueError('diagnostic actor is not canonical execution')
    message=[{'role':'user','content':'Write one short neutral observation about a blue square. Reply with one sentence.'}]
    generated=engine.generate(GenerationRequest(messages=message,temperature=0.8,seed=seed,
                                                max_tokens=max_tokens,rl_capture=True))
    if out is not None:write_json_durable(out/'generation.json',generated)
    target=generated['choices'][0]['message']['content']
    if not isinstance(target,str) or not target.strip():raise ValueError('actual producer did not return text')
    write=engine.native_write_text(target,value_type='Neuralese<string>');block=write['id']
    if out is not None:
        from natlang_neuralese.serve.store import encode_native_state
        data=encode_native_state(engine.store.get_native_state(block).block)
        write_bytes_durable(out/'producer-state.safetensors',data)
        write_json_durable(out/'producer-state-binding.json',{'id':block,'sha256':hashlib.sha256(data).hexdigest(),
                                                              'bytes':len(data),'write':write})
    messages=[{'role':'user','content':[{'type':'native_state','id':block,'source_occurrence':'writer'},
        {'type':'text','text':'Choose a concise label for the observation.'}]}]
    options=options or ['"blue"','"square"','"other"']
    decision=decide_native(engine,{'messages':messages,'options':options,'native_recording':True})
    if out is not None:write_json_durable(out/'decision-scores.json',decision)
    scores=decision['log_probs'];temperature=0.8
    generator=torch.Generator(device='cpu');generator.manual_seed(seed+1)
    choice=int(torch.multinomial(torch.softmax(torch.tensor(scores,dtype=torch.float64)/temperature,0),1,generator=generator).item())
    capture={**decision['native_decision_capture'],'decision_sample':sample_receipt(scores,chosen=choice,temperature=temperature,seed=seed+1)}
    rollout={'task_id':'diagnostic-native-decision-replay','split':'train','diagnostic_only':True,
        'policy':{'checkpoint_sha256':EXPECTED},
        'turns':[{'invocation_id':'producer','messages':message,'target':target,'sampling':generated['x_natlang_rl_capture']},
                 {'invocation_id':'decision','messages':messages,'target':options[choice],'sampling':capture}],
        'execution_graph':[{'node':'producer#turn1','kind':'model_turn','call_id':'producer'},
            {'node':'decision#turn1','kind':'model_turn','call_id':'decision'},
            {'node':'writer','kind':'block_write','turn':'producer#turn1','block':block,'producer':None,
             'source_kind':'typed-text-result','format':'native-state','source_capture':write['source_capture'],
             'native_identity':engine.native_identity,'text_body_sha256':hashlib.sha256(target.encode()).hexdigest(),
             'inputs':[{'node':'producer#turn1','port':'result-source'}]},
            {'node':'reader','kind':'block_read','turn':'decision#turn1','block':block,
             'inputs':[{'node':'writer','block':block}]}]}
    return rollout, {'generated':generated,'write':write,'decision':decision,'chosen':choice,'options':options}


def prepare(engine,rollout,out):
    turns=whole_episode_turns(rollout)
    replay=compile_replay(turns,identity=engine.native_identity,roles=engine.role_tokens,
                          fetch_block=lambda ident:engine.store.get_native_state(ident).block)
    if replay.external_blocks or not replay.actions or len(replay.decision_actions)!=1:
        raise ValueError('diagnostic lacks complete sampled producer and in-graph decision')
    if not any(node.imports for node in replay.plan.invocations):raise ValueError('decision graph lacks actual producer import')
    ident=canonical_json_sha256_hex(rollout);weights=[0.]*replay.plan.token_count
    for action in replay.actions:weights[action['query']+1]=1.
    nodes=[]
    for node,offset in zip(replay.plan.invocations,replay.plan.offsets,strict=True):
        value=invocation_dict(node);value['token_ids']=list(node.token_ids)
        value['target_weights']=weights[offset:offset+len(node.token_ids)];nodes.append(value)
    graph={'schema':SCHEMA,'id':ident,'split':'train','source_groups':['diagnostic-only-no-training'],
        'tokenizer_fingerprint':engine.native_identity['tokenizer_fingerprint'],'nodes':nodes,'external_states':{}}
    example=parse_graph(graph,tokenizer_fingerprint=engine.native_identity['tokenizer_fingerprint'])
    row={'graph_sha256':canonical_json_sha256_hex(graph_identity(example)),
        'episode_turns':['producer#turn1','decision#turn1'], 'graph_nodes':[n.name for n in replay.plan.invocations],
        'actions':list(replay.actions),'decision_actions':list(replay.decision_actions),'advantage':0.,
        'external_consumption_receipts':{},'task_id':'diagnostic-native-decision-replay'}
    spec={'schema':'natlang.native-policy-objective/1','kind':'native-policy','mode':'grpo',
          'rows':{ident:row},'policy':engine.native_identity,'block_bundle':None}
    write_json_durable(out/'rollout.json',rollout);write_json_durable(out/'graph.json',graph)
    write_json_durable(out/'objective.json',spec)
    return example,replay


def probe(engine,rollout,out):
    example,replay=prepare(engine,rollout,out)
    objective=NativePolicyObjective(out/'objective.json');objective.bind_actor(engine)
    engine.shared_model.port.ffn_chunk_tokens=4096
    engine.shared_model.port.checkpoint_layer_group_size=2
    for name,scope in (('eager',nullcontext),('plain',plain_training_arithmetic)):
        receipt=out/('replay-'+name+'.json');os.environ['NATLANG_POLICY_LOGPROB_RECEIPT']=str(receipt)
        objective._behavior_logprob_probe_done=False
        with torch.enable_grad(),scope():
            packed_graph_loss(engine.shared_model,[example],checkpoint_layers=False,objective=objective,
                              share_causal_prefixes=True,collect_accuracy=False)
        evidence=json.loads(receipt.read_text())
        if evidence.get('action_count')!=len(replay.actions) or evidence.get('decision_count')!=len(replay.decision_actions):
            raise ValueError('diagnostic replay omitted sampled actions')
    os.environ.pop('NATLANG_POLICY_LOGPROB_RECEIPT',None)
    return {'graph_id':example.id,'sampled_tokens':len(replay.actions),'decisions':len(replay.decision_actions),
            'option_counts':[len(d['options']) for d in replay.decision_actions], 'optimizer_updates':0,'backward':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','checkpoint-sha256','transport','transport-sha256','view-receipt','view-receipt-sha256','base','out'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    if os.environ.get('NATLANG_BF16_REDUCTION')!='fp32-accumulate':
        raise ValueError('diagnostic requires explicit FP32 BF16 reduction')
    def ref(path,sha):
        path=Path(path).resolve(strict=True)
        from natlang_neuralese.common.hashing import sha256_file_hex
        if sha256_file_hex(path)!=sha:raise ValueError('diagnostic input digest differs')
        return {'path':str(path),'sha256':sha,'bytes':path.stat().st_size}
    view=ref(a.checkpoint,a.checkpoint_sha256);transport=ref(a.transport,a.transport_sha256);receipt=ref(a.view_receipt,a.view_receipt_sha256)
    from natlang_neuralese.serve import load_native_engine
    engine=load_native_engine(view['path'],base=a.base,device='cuda',prompt_rendering='full',
        transported_actor={'transport':transport,'view_receipt':receipt,'recovery_receipt':None,
                           'source_sha256':EXPECTED,'scope':'diagnostic-local-pin'})
    engine.start()
    try:
        rollout,observed=collect(engine,out=out);write_json_durable(out/'actor-info.json',{'native_identity':engine.native_identity,
            'policy':engine.rl_policy_binding,'view':view,'transport':transport,'view_receipt':receipt,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'bf16_reduction':os.environ['NATLANG_BF16_REDUCTION']})
        write_json_durable(out/'observed.json',observed)
        result=probe(engine,rollout,out)
        write_json_durable(out/'result.json',{'schema':'natlang.native-decision-replay-diagnostic/1',**result,
            'checkpoint_sha256':EXPECTED,'qualification':'diagnostic only; no training or capability claim'})
        print(json.dumps(result),flush=True)
    finally:engine.stop()
if __name__=='__main__':main()
