#!/usr/bin/env python3
"""Full-size CUDA mechanism diagnostic using shared production native objectives.

This does not create foundation readiness, a production checkpoint, an oracle
reward or a model-quality qualification. Run under the memory ledger after
reserving resources for active task evaluations. All model updates stay in RAM.
"""
from __future__ import annotations
import argparse
import contextlib
import copy
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path
import time
import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex_cached
from natlang_neuralese.common.jsonio import write_json_durable,iter_jsonl_rows
from natlang_neuralese.model.native_model import load_native_model, native_identity, upgrade_reasoning_tokens
from natlang_neuralese.model.shared_state import native_step, splice_block, concatenate_captures
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.model.semantic_vae import StateLayout, SemanticStateVAE
from natlang_neuralese.train.native_semantic import SemanticParticipant, parse_pair, producer, joint_loss
from natlang_neuralese.train.native_semantic_stage import _vae_optimizer
from natlang_neuralese.train.native_state_sft import make_native_optimizer, packed_graph_loss
from natlang_neuralese.train.fp8_backend import make_backend


def state_digest(model):
    """Actual RAM tensor content identity, explicitly NOT a checkpoint-file hash."""
    digest=hashlib.sha256(b'natlang.diagnostic-ram-model-state/1\0')
    for name,value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode())
        flat=value.detach().reshape(-1)
        for start in range(0,flat.numel(),4<<20):
            chunk=flat[start:start+(4<<20)].contiguous().cpu().view(torch.uint8).numpy()
            digest.update(memoryview(chunk))
    return digest.hexdigest()


def identity(model,config):
    return native_identity(model,config,checkpoint_sha256=state_digest(model))


def memory():
    torch.cuda.synchronize()
    return {'allocated_bytes':torch.cuda.memory_allocated(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'reserved_bytes':torch.cuda.memory_reserved(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}


def tensor_inventory(value):
    counts={}
    def visit(item):
        if isinstance(item,torch.Tensor):
            key=str(item.dtype);entry=counts.setdefault(key,{'tensors':0,'elements':0,'bytes':0})
            entry['tensors']+=1;entry['elements']+=item.numel();entry['bytes']+=item.numel()*item.element_size()
        elif isinstance(item,dict):
            for child in item.values():visit(child)
        elif isinstance(item,(list,tuple)):
            for child in item:visit(child)
    visit(value)
    return counts


def contexts(participants):
    stack=contextlib.ExitStack()
    for participant in participants.values():
        if participant.backend is not None:stack.enter_context(participant.backend.autocast())
    return stack


def sampled_native_graph(participant,roles,payload,seed,extra_banned=()):
    """Forced native producer; genuinely sampled reader with exact masked law.

    A forced writer has no sampled actions or invented likelihood. All KV and
    hybrid state comes from native_step; consumer imports it using splice_block.
    """
    from natlang_neuralese.serve.engine import Engine,Sequence,GenerationRequest
    from natlang_neuralese.serve.store import native_state_id
    model=participant.model;device=next(model.parameters()).device
    law={'temperature':1.,'seed':seed,'guidance':True,'probabilities':'post-processing-normalized'}
    def capture(prompt):
        return {'schema':'natlang.native-rl-sampling/2','policy':participant.identity,'prompt':prompt,
            'samples':[],'generated_inputs':[],'writes':[],'sample_range_convention':'half-open',
            'finish_reason':'length','sampling_law':dict(law)}
    def step(ids,cache=None,producing=False):
        return native_step(model,torch.tensor([ids],dtype=torch.long,device=device),cache,producing=producing)
    # Use a real vocabulary input, keeping the diagnostic trace small and apart
    # from the complete genuine communication graph used by semantic training.
    prefix=participant.tokenizer.encode('Read',add_special_tokens=False)[:1]
    if not prefix:raise ValueError('diagnostic tokenizer produced no prefix')
    writer=capture([{'kind':'tokens','token_ids':prefix,'producing':[False]}])
    chunks=[]
    with torch.no_grad():
        output=step(prefix)
        forced=[(roles.produce_open,False),*[(token,True) for token in payload],(roles.produce_close,False)]
        for token,producing in forced:
            output=step([token],output['cache'],producing)
            writer['generated_inputs'].append({'token_id':token,'input_role':'produce' if producing else 'ordinary','sample_index':None})
            if producing:chunks.append(output)
        block=NativeStateBlock(text=participant.tokenizer.decode(payload,skip_special_tokens=False),
            **participant.identity,**concatenate_captures(chunks)).validate()
        block_id=native_state_id(block)
        writer['writes']=[{'state_id':block_id,'own_begin':len(prefix),
            'own_end':len(prefix)+len(forced),'payload_own_begin':len(prefix)+1,
            'payload_own_end':len(prefix)+1+len(payload),'native_identity':participant.identity}]
        consumer=capture([{'kind':'tokens','token_ids':[*prefix,roles.consume_open],'producing':[False,False]},
            {'kind':'native_state','id':block_id,'native_identity':participant.identity},
            {'kind':'tokens','token_ids':[roles.consume_close],'producing':[False]}])
        output=step([*prefix,roles.consume_open])
        cache=splice_block(model,output['cache'],block,**{key:value for key,value in participant.identity.items() if key!='architecture'})
        output=step([roles.consume_close],cache)
        banned=sorted(set([roles.produce_open,roles.produce_close,roles.consume_open,roles.consume_close,*extra_banned]))
        sequence=Sequence(GenerationRequest(messages=[],temperature=1.),Future(),
            logits=model.logits(output['h_final'][:,-1]),rng=torch.Generator().manual_seed(seed),
            native_own_position=3,bans={3:set(banned)})
        token,logp=Engine._native_token(None,sequence)
        terminal=token==participant.tokenizer.eos_token_id
        consumer['samples']=[{'index':0,'token_id':token,'logp':logp,'input_role':'ordinary',
            'terminal':terminal,'query_own_position':2,'banned_token_ids':banned}]
        if not terminal:consumer['generated_inputs']=[{'token_id':token,'input_role':'ordinary','sample_index':0}]
    return block_id,block,writer,consumer,token


def run(args):
    from natlang_neuralese.rl.native_replay import prepare_objective,NativePolicyObjective,compile_replay,replay_logps
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.common.attention_backend import configure_sdpa_backend
    configure_sdpa_backend('flash')
    torch.manual_seed(703)
    started=time.time();out=args.out;out.mkdir(parents=True,exist_ok=True)
    proof=json.loads(args.selection_proof.read_text());row=json.loads(args.pair.read_text())
    if row['split']!='train' or canonical_json_sha256_hex(row)!=proof['row_sha256']:
        raise ValueError('diagnostic must retain the complete genuine selected train row')
    if sha256_file_hex_cached(proof['source'])!=proof['source_sha256']:
        raise ValueError('diagnostic admitted pair source changed')
    if type(proof['row_index']) is not int or proof['row_index']<0:raise ValueError('invalid selected source row index')
    for index,(_,source_row) in enumerate(iter_jsonl_rows(proof['source'])):
        if index==proof['row_index']:
            if canonical_json_sha256_hex(source_row)!=proof['row_sha256']:raise ValueError('diagnostic row is not the declared source row')
            break
    else:raise ValueError('diagnostic selected source row does not exist')
    report={'schema':'natlang.native-joint-cuda-fit-diagnostic/1','production_qualification':False,
        'foundation_certificate_created':False,'actual_environment_reward':False,
        'identity_kind':'actual in-memory tensor-content SHA; not a checkpoint-file SHA',
        'selection':proof,'models':{},'phases':[],'rl':{},'device':torch.cuda.get_device_name()}
    def event(kind,**value):
        print(json.dumps({'event':kind,'elapsed_seconds':time.time()-started,**value}),flush=True)
    participants={};roles_by_name={};trainable={}
    for base,dtype in [(args.lfm_base,torch.float32),(args.mellum_base,torch.bfloat16)]:
        model,tokenizer,roles,config=load_native_model(base,device='cuda',dtype=dtype)
        upgrade=upgrade_reasoning_tokens(model,tokenizer,config)
        name=config['architecture']+':'+config['base_inventory_sha256']
        if name not in row['models']:raise ValueError('local base inventory is not an actual selected pair model')
        backend=make_backend('triton-experts',model.port.hf,device='cuda') if config['architecture']=='mellum' else None
        if backend is not None:model.port.activation_checkpoint=backend.checkpoint
        optimizer,named=make_native_optimizer(model,lr=1e-5,in_backward=False)
        model.port.checkpoint_layers=True;model.train()
        participant=SemanticParticipant(model,tokenizer,config,identity(model,config),optimizer,backend)
        participants[name]=participant;roles_by_name[name]=roles;trainable[name]={id(parameter) for _,parameter in named}
        report['models'][name]={'base':str(base),'config':config,'identity':participant.identity,'sparse_upgrade':upgrade}
        event('loaded',model=name,memory=memory())
    parsed=parse_pair(row,participants,required_split='train');layouts={}
    with contexts(participants),torch.no_grad():
        for name,(example,view) in parsed.items():
            block=producer(participants[name],example,view,supervise=False)[0]
            layouts[name]=StateLayout.from_block(block)
            del block
    vae=SemanticStateVAE(layouts,latent_width=128,hidden_width=512).to('cuda')
    vae_optimizer=_vae_optimizer(vae,{'vae_lr':1e-4})
    for phase in ('vae','joint','distill'):
        before={name:value.model.input_rows.detach().clone() for name,value in participants.items()}
        for name,participant in participants.items():
            participant.optimizer.zero_grad(set_to_none=True)
            for parameter in participant.model.parameters():parameter.requires_grad_(phase!='vae' and id(parameter) in trainable[name])
        vae_optimizer.zero_grad(set_to_none=True);vae.requires_grad_(phase!='distill')
        with contexts(participants):
            loss,details=joint_loss(vae,participants,row,phase=phase,options={'chunk_tokens':64,'allow_nearest':True})
            loss.backward()
        parameters=[parameter for participant in participants.values() for parameter in participant.model.parameters()
                    if parameter.requires_grad and parameter.grad is not None]
        parameters += [parameter for parameter in vae.parameters() if parameter.requires_grad and parameter.grad is not None]
        norm=torch.nn.utils.clip_grad_norm_(parameters,float('inf'))
        if not bool(torch.isfinite(norm)) or not float(norm)>0:raise ValueError('missing/nonfinite semantic gradient')
        gradients={name:float(participant.model.input_rows.grad.float().norm()) if participant.model.input_rows.grad is not None else None
                   for name,participant in participants.items()}
        if phase!='vae':
            for participant in participants.values():participant.optimizer.step()
        if phase!='distill':vae_optimizer.step()
        changed={name:bool((participant.model.input_rows.detach()!=before[name]).any()) for name,participant in participants.items()}
        if phase=='vae' and any(changed.values()):raise ValueError('VAE-only changed model rows')
        if phase!='vae' and not all(changed.values()):raise ValueError('joint/distill failed to update actual model rows')
        item={'phase':phase,'loss':float(loss.detach()),'gradient_norm':float(norm),'row_gradient_norms':gradients,
              'role_rows_changed':changed,'details':details,'memory':memory()}
        report['phases'].append(item);write_json_durable(out/'progress.json',report);event('semantic_update',**item)
        del loss,parameters,before
    for participant in participants.values():participant.optimizer.zero_grad(set_to_none=True)
    vae_optimizer.zero_grad(set_to_none=True)
    for name,participant in participants.items():
        participant.optimizer.zero_grad(set_to_none=True);participant.model.eval()
        participant.identity=identity(participant.model,participant.config)
        payload=participant.tokenizer.encode('42',add_special_tokens=False)
        if not payload:raise ValueError('diagnostic payload is empty')
        blocks={};episodes=[];first_token=None;max_gap=0.;ratios=[]
        with contexts({name:participant}):
            for sample in range(2):
                block_id,block,writer,reader,token=sampled_native_graph(participant,roles_by_name[name],payload,901+sample,
                    () if first_token is None else (first_token,))
                first_token=token;blocks[block_id]=block
                turns=[{'graph_turn':'writer','sampling':writer,'written_nodes':{block_id:'write-0'}},
                       {'graph_turn':'reader','sampling':reader,'read_bindings':{block_id:'write-0'}}]
                replay=compile_replay(turns,identity=participant.identity,roles=roles_by_name[name],fetch_block=blocks.__getitem__)
                with torch.no_grad():new,old=replay_logps(participant.model,replay,chunk_tokens=64)
                gap=float((new-old).abs().max());max_gap=max(max_gap,gap);ratios.extend((new-old).exp().cpu().tolist())
                raw=[{'messages':[],'target':{},'invocation_id':turn['graph_turn'],'sampling':turn['sampling'],
                      'replay':{'raw_calls_retained':True,'diagnostic_kind':'exact-native-step-inputs'}} for turn in turns]
                episodes.append({'task_id':'controlled-cuda-fit','sample':sample,'split':'train','group':'controlled',
                    'reward':{'quality':1-sample},'policy':{'checkpoint_sha256':participant.identity['checkpoint_sha256']},
                    'turns':raw,'execution_graph':[{'kind':'block_write','node':'write-0','turn':'writer#turn1','block':block_id},
                    {'kind':'block_read','node':'read-0','turn':'reader#turn1','block':block_id,'inputs':[{'node':'write-0','block':block_id}]}]})
        graphs,spec=prepare_objective(episodes,identity=participant.identity,roles=roles_by_name[name],fetch_block=blocks.__getitem__)
        path=out/(name.split(':')[0]+'-controlled-objective.json');write_json_durable(path,spec)
        examples=[parse_graph(graph,tokenizer_fingerprint=participant.config['tokenizer_fingerprint']) for graph in graphs]
        participant.model.train();before=participant.model.adapter.up.weight.detach().clone()
        participant.optimizer.zero_grad(set_to_none=True)
        with contexts({name:participant}):
            loss,details=packed_graph_loss(participant.model,examples,chunk_tokens=64,objective=NativePolicyObjective(path))
            loss.backward()
        gradient=float(participant.model.adapter.up.weight.grad.float().norm())
        if not gradient>0 or not torch.isfinite(loss):raise ValueError('controlled native policy failed to produce a finite gradient')
        participant.optimizer.step()
        if torch.equal(before,participant.model.adapter.up.weight):raise ValueError('controlled GRPO did not update native interface')
        item={'loss':float(loss.detach()),'adapter_gradient_norm':gradient,'sampled_tokens':details['sampled_tokens'],
            'exact_native_writer_reader':True,'crisp_fallback':False,'sampling_replay_max_logp_abs':max_gap,
            'ratio_min':min(ratios),'ratio_max':max(ratios),
            'initial_ratio_clipped_fraction':sum(value<.8 or value>1.28 for value in ratios)/len(ratios),
            'controlled_outcomes_not_task_rewards':True,'optimizer_state':tensor_inventory(participant.optimizer.state_dict()),'memory':memory()}
        report['rl'][name]=item;write_json_durable(out/'progress.json',report);event('native_policy_update',model=name,**item)
        participant.optimizer.zero_grad(set_to_none=True)
    report.update(completed=True,elapsed_seconds=time.time()-started,memory=memory())
    write_json_durable(out/'proof.json',report);event('completed',memory=memory())
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','selection-proof','lfm-base','mellum-base','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        args.out.mkdir(parents=True,exist_ok=True)
        write_json_durable(args.out/'failure.json',{'error_type':type(error).__name__,'error':str(error),'production_qualification':False,'memory':memory()})
        raise


if __name__=='__main__':main()
