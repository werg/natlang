import hashlib,json,uuid,tarfile,shutil
from pathlib import Path
import pytest,torch
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.diloco.role_contract import validate,select
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator,Coordinator,Conflict,_metadata
from natlang_neuralese.train.diloco.protocol import save_tensors

def ref(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def test_exact_role_contracts_bind_source_objective_and_queued_windows(tmp_path):
    paths=[]
    for name in ('recipe','train','held','objective','collector','rollout','parent'):
        p=tmp_path/name;p.write_text(name);paths.append(p)
    recipe,train,held,objective,collector,rollout,parentfile=paths;seed='a'*64;lineage='b'*64;settings={'run_id':'hetero','workers':4,'local_steps':8,'wire_dtype':'fp32'}
    def role(kind,parent):
        binding={'path':str(recipe),'file_sha256':ref(recipe)['sha256'],'lineage_sha256':lineage}
        argv=['--recipe-binding',json.dumps(binding),'--stage-id',kind,'--native-graphs',str(train),'--held-native-graphs',str(held)]
        refs=[ref(recipe),ref(train),ref(held)]
        item={'kind':kind,'identity':{'schema':'natlang.native-diloco-identity/1','parent_checkpoint_sha256':parent,'diloco':settings,'trainer_argv':argv},'dependencies':refs}
        if kind=='jolt':argv.extend(['--objective-spec',str(objective)]);refs.extend([ref(objective),ref(collector)]);item.update(on_policy_parent_sha256=parent,on_policy_parent=ref(parentfile),collector_receipt=ref(collector))
        return item
    parenthash=ref(parentfile)['sha256'];collector.write_text(json.dumps({'schema':'natlang.native-rl-admission/1','policy':{'checkpoint_sha256':parenthash},'native_graphs_sha256':ref(train)['sha256'],'objective_sha256':ref(objective)['sha256'],'episodes':1,'rollouts':{str(rollout):ref(rollout)['sha256']},'tasks_selected':[{'task_id':'tiny','split':'train'}]}))
    sft=role('sft',seed);jolt=role('jolt',parenthash);jolt['dependencies'].append(ref(rollout));contract={'schema':'natlang.diloco-heterogeneous-identity/1','baseline_sha256':seed,'recipe_lineage_sha256':lineage,'diloco':settings,'roles':{'0':[sft],'1':[sft],'2':[sft],'3':[sft]}}
    validate(contract,4,verify_files=True)
    wrong=dict(jolt['identity'],parent_checkpoint_sha256='d'*64)
    with pytest.raises(ValueError,match='not an allowed'):select(contract,2,wrong)
    augmented={**contract,'roles':{**contract['roles'],'2':[sft,jolt]}}
    assert select(augmented,2,jolt['identity'])==canonical_json_sha256_hex(jolt)
    root=tmp_path/'coordinator';initialize_coordinator(root,run_id='hetero',masters={'w':torch.tensor([1.])},identity=contract,workers=4,min_quorum=3,grace_seconds=0,fragment_bytes=1024,seed_source={'sha256':seed})
    replica_root=tmp_path/'replica';shutil.copytree(root,replica_root)
    c=Coordinator(root);sessions=[str(uuid.uuid4()) for _ in range(3)]
    c.lease(2,sessions[2],canonical_json_sha256_hex(sft))
    archive=tmp_path/'role-evidence.tar'
    with tarfile.open(archive,'w') as tar:
        for dependency in {x['sha256']:x for x in jolt['dependencies'] if x not in sft['dependencies']}.values():tar.add(dependency['path'],arcname=dependency['sha256'])
    assert c.register_role_contract(2,sessions[2],jolt,archive)['status']=='registered'
    from natlang_neuralese.train.diloco.replica_state import ReplayCoordinator
    replica=ReplayCoordinator(replica_root,quorum_since={})
    empty=tmp_path/'empty';empty.write_bytes(b'')
    lease_event={'kind':'lease','args':{'worker':2,'session':sessions[2],'contract_sha256':canonical_json_sha256_hex(sft)},'wall':1000.,'monotonic':1000.}
    assert replica.apply(lease_event,empty)['error'] is None
    event={'kind':'register_role_contract','args':{'worker':2,'session':sessions[2],'contract':jolt},'wall':1001.,'monotonic':1001.}
    assert replica.apply(event,archive)['error'] is None
    assert replica.coordinator.role_contracts(2)['roles']['2'][-1]==jolt
    assert Coordinator(root).role_contracts(2)['roles']['2'][-1]==jolt
    for rank in range(3):
        chosen=jolt if rank==2 else sft;c.lease(rank,sessions[rank],canonical_json_sha256_hex(chosen));p=tmp_path/f'update-{rank}';save_tensors(p,{'delta':torch.tensor([0.1*(rank+1)])});c.submit(0,rank,sessions[rank],1,0,1,2,p)
    meta=_metadata(root/'fragments/00000.safetensors');windows=json.loads(meta['windows']);assert windows['2']['contract_sha256']==canonical_json_sha256_hex(jolt);assert int(meta['version'])==1;assert json.loads(meta['contract_ledger'])['2:1']==canonical_json_sha256_hex(jolt)
    with pytest.raises(ValueError,match='not authorized'):c.lease(3,str(uuid.uuid4()),canonical_json_sha256_hex(jolt))
    objective.write_text('tampered')
    with pytest.raises(ValueError,match='changed'):select(augmented,2,jolt['identity'])

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

def test_real_http_registration_carries_drained_state_into_jolt_and_back(tmp_path,monkeypatch):
    import argparse,socket,threading,time
    from natlang_neuralese.train.diloco.worker import DilocoLearner,DilocoSettings,Transport
    from natlang_neuralese.train.diloco.protocol import TensorSurface
    from natlang_neuralese.train.diloco.coordinator import serve_coordinator
    from natlang_neuralese.train.native_diloco import identity
    from natlang_neuralese.rl import diloco_contract as dc
    model=torch.nn.Linear(2,1,bias=False);seed=tmp_path/'seed';torch.save({'model':model.state_dict()},seed);seedsha=ref(seed)['sha256']
    recipe=tmp_path/'recipe';recipe.write_text('recipe');source=tmp_path/'source.jsonl';source.write_text('source');held=tmp_path/'held.jsonl';held.write_text('held');objective=tmp_path/'objective.json';objective.write_text('{}');rollout=tmp_path/'rollout.jsonl';rollout.write_text('rollout')
    binding={'path':str(recipe),'sha256':ref(recipe)['sha256'],'file_sha256':ref(recipe)['sha256'],'lineage_sha256':'b'*64}
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    url=f'http://127.0.0.1:{port}';settings=DilocoSettings(run_id='switch',workers=4,worker_id=2,coordinator_url=url,allow_private_http=True,local_steps=1,wire_dtype='fp32',heartbeat_seconds=.1)
    common=['--recipe-binding',json.dumps(binding),'--stage-id','sft','--native-graphs',str(source),'--held-native-graphs',str(held),'--diloco',json.dumps({**settings.__dict__})]
    sft_identity=identity(seed,common,settings);sft={'kind':'sft','identity':sft_identity,'dependencies':[ref(recipe),ref(source),ref(held)]};authority={'schema':'natlang.diloco-heterogeneous-identity/1','baseline_sha256':seedsha,'recipe_lineage_sha256':'b'*64,'diloco':settings.identity(),'roles':{str(i):[sft] for i in range(4)}}
    root=tmp_path/'outer';initialize_coordinator(root,run_id='switch',masters=TensorSurface.from_module(model).masters,identity=authority,workers=4,min_quorum=1,grace_seconds=0,fragment_bytes=1024,seed_source={'sha256':seedsha})
    stop=threading.Event();thread=threading.Thread(target=serve_coordinator,args=(root,'127.0.0.1',port,'x'*32),kwargs={'stop':stop},daemon=True);thread.start();transport=Transport(url,'x'*32,10,allow_http=True)
    for _ in range(100):
        try:transport.json('GET','/v1/manifest');break
        except OSError:time.sleep(.02)
    monkeypatch.setattr(dc,'transport_from_env',lambda _:transport)
    first=DilocoLearner(settings,surface=TensorSurface.from_module(model),identity=sft_identity,output=tmp_path/'first',transport=transport);second=None
    try:
        first.start();saved=first.state_dict();first.close();parent=tmp_path/'parent.pt';torch.save({'training':{'diloco':saved}},parent);parent_sha=ref(parent)['sha256']
        out=tmp_path/'rl';out.mkdir();receipt=out/'rl-data-receipt.json';receipt.write_text(json.dumps({'schema':'natlang.native-rl-admission/1','policy':{'checkpoint_sha256':parent_sha},'native_graphs_sha256':ref(source)['sha256'],'objective_sha256':ref(objective)['sha256'],'episodes':1,'rollouts':{str(rollout):ref(rollout)['sha256']},'tasks_selected':[{'task_id':'x','split':'train'}]}))
        rl_argv=['--recipe-binding',json.dumps(binding),'--stage-id','rl','--native-graphs',str(source),'--held-native-graphs',str(held),'--objective-spec',str(objective),'--continue-from',str(parent),'--out',str(out),'--diloco',json.dumps(settings.__dict__)]
        actual=dc.register_admitted_stage(argparse.Namespace(continue_from=parent,out=out,rollouts=[rollout]),rl_argv);assert actual['status']=='registered'
        rl_identity=identity(parent,rl_argv,settings);second=DilocoLearner(settings,surface=TensorSurface.from_module(model),identity=rl_identity,output=tmp_path/'second',transport=transport);second.resume(saved)
        assert second.contract_history[-1]['from']==saved['contract_sha256'] and second.contract_history[-1]['to']==actual['sha256']
        bad={**saved,'steps_since':[1]*len(saved['steps_since'])}
        with pytest.raises(ValueError,match='uncommitted windows'):
            second.resume(bad)
        with torch.no_grad():model.weight.add_(0.1)
        second.after_step(tokens=10);second.declare_completion(drained=False,generation='c'*64)
        assert second.drain_for_contract();second.declare_completion(drained=True,generation='c'*64)
        rl_saved=second.state_dict();second.close();second=None
        parent2=tmp_path/'rl-parent.pt';torch.save({'training':{'diloco':rl_saved}},parent2)
        sftout=tmp_path/'sft-again';sftout.mkdir()
        sft_argv=['--recipe-binding',json.dumps(binding),'--stage-id','sft','--native-graphs',str(source),'--held-native-graphs',str(held),'--continue-from',str(parent2),'--out',str(sftout),'--diloco',json.dumps(settings.__dict__)]
        result=dc.register_supervised_stage(argparse.Namespace(continue_from=parent2,out=sftout),sft_argv,rl_saved);assert result['status']=='registered'
        sft2=DilocoLearner(settings,surface=TensorSurface.from_module(model),identity=identity(parent2,sft_argv,settings),output=tmp_path/'third',transport=transport)
        try:
            sft2.resume(rl_saved);sft2.reopen(previous='c'*64,generation='d'*64,sequences=rl_saved['sequences'])
            assert sft2.contract_history[-1]['to']==result['sha256']
            with torch.no_grad():model.weight.add_(0.1)
            sft2.after_step(tokens=10);assert sft2.drain_for_contract()
            assert transport.json('GET','/v1/status')['versions'][0]>=2
            extra=tmp_path/'extra.jsonl';extra.write_text('extra')
            successor=tmp_path/'successor.json';successor.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs',**ref(source)},{'kind':'native-graphs',**ref(extra)}]}))
            content=tmp_path/'content-audit';content.write_text(json.dumps({'schema':'natlang.big2-multi-content-audit/1','base':{'sha256':ref(source)['sha256']},'candidates':[{'sha256':ref(extra)['sha256']}],'duplicates':[]}))
            protection=tmp_path/'protection-audit';protection.write_text(json.dumps({'schema':'natlang.fresh-restart-protection-union/1','sources':{'base':{'source':ref(source),'conflicts':[]},'candidate-001':{'source':ref(extra),'conflicts':[]}}}))
            newrecipe=tmp_path/'new-recipe';newrecipe.write_text('new-recipe');newbinding={'path':str(newrecipe),'file_sha256':ref(newrecipe)['sha256'],'lineage_sha256':'b'*64,'external_provenances':[{'content_audit':ref(content),'clean_closure_audit':ref(protection)}]}
            current=sft2.state_dict();slot=sftout/'checkpoint.pt';torch.save({'training':{'diloco':current}},slot)
            append_argv=['--recipe-binding',json.dumps(newbinding),'--stage-id','sft','--native-graphs',str(successor),'--continued-native-graphs',str(source),'--held-native-graphs',str(held),'--continue-from',str(parent2),'--out',str(sftout),'--diloco',json.dumps(settings.__dict__)]
            transition={'schema':'natlang.diloco-source-contract-transition/1','kind':'append','verified':True,'checkpoint_sha256':ref(slot)['sha256'],'prior_source_sha256':ref(source)['sha256'],'source_sha256':ref(successor)['sha256']}
            result,newidentity=dc.register_source_append(argparse.Namespace(continue_from=parent2,out=sftout),append_argv,current,transition)
            assert result['status']=='registered';assert sft2.switch_role_contract(newidentity)==result['sha256']
            with torch.no_grad():model.weight.add_(0.1)
            sft2.after_step(tokens=10);assert sft2.drain_for_contract()
            assert transport.json('GET','/v1/status')['versions'][0]>=3
        finally:sft2.close()
    finally:
        if second:second.close()
        stop.set();thread.join(timeout=5)

def test_resident_argv_rebinds_only_declared_source_and_keeps_actual_runtime_controls(tmp_path):
    from natlang_neuralese.train.diloco.role_contract import resident_argv
    old=['--native-graphs','/work/old','--held-native-graphs','/work/held','--foundation-native-graphs','/work/old','--device','cuda','--resident-data-directory','/work/resident','--learning-rate','0.000001','--recipe-binding','old','--stage-id','sft','--continue-from','/work/seed','--out','/work/out']
    inputs={'native_graphs':{'path':'/work/new'},'held_native_graphs':{'path':'/work/held'},'foundation_native_graphs':{'path':'/work/new'},'continued_native_graphs':{'path':'/work/old'}}
    payload={'plan':{'recipe':{'stages':[{'id':'sft','kind':'native_state_sft','parameters':{}}]},'stage_inputs':{'sft':inputs}},'recipe':{'path':'/work/recipe','sha256':'a'*64}}
    result=resident_argv(old,payload,'sft');assert result[result.index('--native-graphs')+1]=='/work/new';assert result[result.index('--continued-native-graphs')+1]=='/work/old';assert result[result.index('--foundation-native-graphs')+1]=='/work/new'
    assert result[result.index('--resident-data-directory')+1]=='/work/resident' and result[result.index('--learning-rate')+1]=='0.000001' and result[result.index('--continue-from')+1]=='/work/seed'
    assert json.loads(result[result.index('--recipe-binding')+1])==payload['recipe']

def test_logical_source_paths_preserve_sandbox_alias(tmp_path):
    from natlang_neuralese.rl.diloco_contract import _logical_source_paths
    real=tmp_path/'mirror';real.mkdir();logical=tmp_path/'logical';logical.symlink_to(real,target_is_directory=True)
    child=real/'rows.jsonl';child.write_text('rows');manifest=real/'source.json';manifest.write_text(json.dumps({'schema':'natlang.native-graph-sources/1','sources':[{'kind':'native-graphs','path':str(logical/'rows.jsonl'),'sha256':ref(child)['sha256']}]}))
    assert _logical_source_paths(logical/'source.json')==[logical/'rows.jsonl']
    child.write_text('changed')
    with pytest.raises(ValueError,match='changed'):_logical_source_paths(logical/'source.json')
