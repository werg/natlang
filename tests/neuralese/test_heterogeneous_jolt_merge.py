"""Actual producer/consumer JOLT gradients participate in a shared outer merge."""
import json
import uuid
import torch
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator,Coordinator,_metadata
from natlang_neuralese.train.diloco.protocol import save_tensors
from natlang_neuralese.train.native_resident import binding as ref

def test_real_tiny_wholegraph_jolt_and_sft_updates_share_outer_merge(tmp_path):
    """Real native producer/consumer graph and JOLT objective; reward is a fixture."""
    import copy
    from test_rl_jolt import load,episode_with_top_k,rollouts,prepare,objective_for,forward
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.train.native_state_sft import packed_graph_loss
    from natlang_neuralese.train.diloco.protocol import TensorSurface,pack_fragment
    model,roles,identity=load('lfm');parent=tmp_path/'actual-parent.pt';torch.save({'model':model.state_dict()},parent);identity={**identity,'checkpoint_sha256':ref(parent)['sha256']}
    episode=episode_with_top_k(model,roles,identity,'write-only');rows=rollouts(episode.turns)
    for row in rows:row['policy']['checkpoint_sha256']=identity['checkpoint_sha256']
    graphs,spec=prepare(rows,episode,roles,identity)
    assert any(node.get('imports') for node in graphs[0]['nodes'])
    graph=tmp_path/'graph.jsonl';graph.write_text(''.join(json.dumps(x)+'\n' for x in graphs));objective=objective_for(tmp_path,spec);objective_path=tmp_path/'objective.json';example=parse_graph(graphs[0],tokenizer_fingerprint=identity['tokenizer_fingerprint'])
    recipe=tmp_path/'recipe';recipe.write_text('tiny recipe');held=tmp_path/'held';held.write_text('held');rollout=tmp_path/'rollouts';rollout.write_text(json.dumps(rows));collector=tmp_path/'collector';collector.write_text(json.dumps({'schema':'natlang.native-rl-admission/1','policy':{'checkpoint_sha256':identity['checkpoint_sha256']},'native_graphs_sha256':ref(graph)['sha256'],'objective_sha256':ref(objective_path)['sha256'],'episodes':len(rows),'rollouts':{str(rollout):ref(rollout)['sha256']},'tasks_selected':[{'task_id':'fixture','split':'train'}]}))
    seed='a'*64;lineage='b'*64;settings={'run_id':'real-mixed','workers':4,'local_steps':8,'wire_dtype':'fp32'};binding={'path':str(recipe),'file_sha256':ref(recipe)['sha256'],'lineage_sha256':lineage}
    def role(kind):
        parent_sha=identity['checkpoint_sha256'] if kind=='jolt' else seed
        argv=['--recipe-binding',json.dumps(binding),'--stage-id',kind,'--native-graphs',str(graph),'--held-native-graphs',str(held)];refs=[ref(recipe),ref(graph),ref(held)]
        x={'kind':kind,'identity':{'schema':'natlang.native-diloco-identity/1','parent_checkpoint_sha256':parent_sha,'diloco':settings,'trainer_argv':argv},'dependencies':refs}
        if kind=='jolt':argv+=['--objective-spec',str(objective_path)];refs += [ref(objective_path),ref(collector),ref(rollout)];x.update(on_policy_parent_sha256=parent_sha,on_policy_parent=ref(parent),collector_receipt=ref(collector))
        return x
    sft,jolt=role('sft'),role('jolt');contract={'schema':'natlang.diloco-heterogeneous-identity/1','baseline_sha256':seed,'recipe_lineage_sha256':lineage,'diloco':settings,'roles':{'0':[sft],'1':[sft],'2':[jolt],'3':[sft]}}
    root=tmp_path/'outer';surface=TensorSurface.from_module(model);initialize_coordinator(root,run_id='real-mixed',masters=surface.masters,identity=contract,workers=4,min_quorum=3,grace_seconds=0,fragment_bytes=1<<28,seed_source={'sha256':seed});c=Coordinator(root);fragment=c.fragments[0];baseline=pack_fragment(surface.masters,fragment)
    for rank in range(3):
        local=copy.deepcopy(model);parameters=[p for p in local.parameters() if p.requires_grad];optimizer=torch.optim.SGD(parameters,lr=.001);optimizer.zero_grad()
        if rank==2:
            _,context,result=forward(local,graphs[0],identity);loss,_=objective(local,result,context)
        else:loss,_=packed_graph_loss(local,[example])
        loss.backward();assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in parameters);optimizer.step()
        delta=baseline-pack_fragment(TensorSurface.from_module(local).masters,fragment);assert delta.abs().sum()>0
        path=tmp_path/f'real-{rank}.st';save_tensors(path,{'delta':delta});session=str(uuid.uuid4());chosen=jolt if rank==2 else sft;c.lease(rank,session,canonical_json_sha256_hex(chosen));c.submit(0,rank,session,1,0,1,10,path)
    meta=_metadata(root/'fragments/00000.safetensors');assert int(meta['version'])==1 and json.loads(meta['contract_ledger'])['2:1']==canonical_json_sha256_hex(jolt)
    restarted=Coordinator(root);assert restarted._version(0)==1 and json.loads(_metadata(restarted._path(0))['contract_ledger'])['2:1']==canonical_json_sha256_hex(jolt)

