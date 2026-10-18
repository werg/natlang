"""Real resumed optimizer state and sealed raw bridge boundary; no quality claim."""
import copy,json
from pathlib import Path
import pytest
import torch
from test_native_completed_supervised import fixture
from test_native_state_sft import graph
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.rl import native_completed_supervised as adapt,native_bridge as bridge
from natlang_neuralese.train.native_continuation import collect_sft_progress,progress_obligations
from natlang_neuralese.train.native_operation_report import compact_obligations
from natlang_neuralese.train.loop import capture_training_rng_state,restore_training_rng_state


def raw_fixture(tmp_path):
    root,config,request,result,controller,plan,recipe=fixture(tmp_path)
    _,pair=adapt.consume(config,bridge.binding(request),bridge.binding(result),tmp_path/'prior-consumed.json',controller=controller)
    parent=copy.deepcopy(pair[1]);parent_ref=pair[0]
    write(root/'cycle.json',{'phase':'rl','cycle':4,'next_stage':0,'pending_rl_round':3,'current_checkpoint':parent_ref})
    production=root/'production';state=production/'state.json'
    write(state,{'phase':'collect','checkpoint':parent_ref['path'],'round':2,'trained_rounds':2,'production_complete':False})
    raw_request=bridge.write_request(root/'cycles/000004/rl-bridge',kind='round',out=production,config=config,state=state)
    learner=production/'round-000002/learner';learner.mkdir(parents=True)
    graphs=learner/'rl-native-graphs.jsonl';graphs.write_text(json.dumps(graph('policy'))+'\n')
    objective=learner/'rl-native-objective.json';write(objective,{'policy':{'checkpoint_sha256':parent_ref['sha256']},'rows':{'policy':{'task_id':'train-protocol'}}})
    weight=torch.nn.Parameter(parent['model']['weight'].clone());optimizer=torch.optim.AdamW([weight],lr=.01)
    optimizer.load_state_dict(parent['training']['optimizer']);restore_training_rng_state(parent['training']['rng'],'cpu')
    for _ in range(3):optimizer.zero_grad();weight.square().sum().backward();optimizer.step()
    inherited=collect_sft_progress(parent['training'])
    training={**copy.deepcopy(parent['training']),'global_step':7,'step':3,'optimizer':optimizer.state_dict(),
        'rng':capture_training_rng_state('cpu'),'identity':{'objective':{'policy':parent_ref['sha256']}},
        'options':{**parent['training']['options'],'stage_id':'actual-raw-policy','continuation':{'checkpoint_sha256':parent_ref['sha256']}},
        'source_receipts':[{'path':str(graphs),'sha256':bridge.binding(graphs)['sha256']}],
        'source_row_bindings':{'policy':'policy-binding'},'source_row_order':['policy'],'exposure_counts':{'policy':3},'sft_progress':inherited}
    saved={**parent,'model':{'weight':weight.detach()},'training':training};checkpoint=learner/'checkpoint.pt';torch.save(saved,checkpoint);cp=bridge.binding(checkpoint)
    write(learner/'rl-data-receipt.json',{'schema':'natlang.native-rl-admission/1','policy':{'checkpoint_sha256':parent_ref['sha256']},
        'tasks_selected':[{'task_id':'train-protocol','split':'train'}],'native_graphs_sha256':bridge.binding(graphs)['sha256'],
        'objective_sha256':bridge.binding(objective)['sha256']})
    write(learner/'native-training-receipt.json',{'schema':'natlang.native-training-readiness/1','stage_id':'actual-raw-policy',
        'policy_operation_completed':True,'completed':False,'checkpoint':cp,'continuation':{'checkpoint_sha256':parent_ref['sha256']},
        'continuation_boundary':{'new_optimizer_updates':3,'checkpoint_sha256':cp['sha256']},
        'mandatory_sft':compact_obligations(progress_obligations(inherited),cp)})
    write(state,{'phase':'collect','checkpoint':str(checkpoint),'round':3,'trained_rounds':3,'production_complete':False})
    req=bridge.read_request(raw_request);raw_result=raw_request.with_suffix('.result.json')
    write(raw_result,{'schema':bridge.RESULT_SCHEMA,'kind':'round','request_id':req['request_id'],
        'request_sha256':bridge.binding(raw_request)['sha256'],'status':'completed','state':bridge.binding(state),'checkpoint':cp,
        'artifacts':bridge.inventory(production)})
    return root,config,raw_request,raw_result,controller,plan,recipe,saved,inherited


def same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b


@pytest.mark.parametrize('lr',[1e-5,1e-6])
def test_real_raw_consumption_preserves_exact_state_pending_stages_and_single_decode(tmp_path,monkeypatch,lr):
    root,config,request,result,controller,plan,recipe,saved,inherited=raw_fixture(tmp_path)
    original=torch.load;loads=[]
    def load(*args,**kwargs):loads.append(args[0]);return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',load)
    before=(config.read_bytes(),(root/'production/state.json').read_bytes());out=tmp_path/'consumed-raw.json'
    receipt,pair=adapt.consume_raw(config,bridge.binding(request),bridge.binding(result),out,controller=controller)
    assert len(loads)==1 and receipt['schema']==adapt.CONSUMED_RAW and receipt['supervised_stage_credits_added']==0
    assert receipt['stage_id']is None and receipt['new_optimizer_updates']==3 and receipt['global_step']==7
    cursor=json.loads((root/'cycle.json').read_text());assert cursor['phase']=='sft'and cursor['next_stage']==0 and 'pending_rl_round'not in cursor
    assert before==(config.read_bytes(),(root/'production/state.json').read_bytes())
    same(pair[1],saved);same(pair[1]['training']['sft_progress'],inherited)
    package=tmp_path/'package';package.mkdir();(package/'identity.py').write_text('VALUE=1\n')
    future=json.loads(recipe.read_text());future['stages'][1]['parameters']['learning_rate']=lr;path=tmp_path/'future.json';write(path,future)
    proof=adapt.prepare(config,plan,'lfm',adapt.binding(path),adapt.binding(out),pair,tmp_path/'adaptation/proof.json',package=package)
    verified=adapt.verify(proof);assert verified['schema']==adapt.SCHEMA_RAW and verified['new_sources_earned_visits']==0
    assert len(loads)==1 and set(verified['started_stage_ids'])=={'dpo','full'}
    assert json.loads((root/'cycle.json').read_text())['next_stage']==0
    assert len(loads)==1
    adapt.consume_raw(config,bridge.binding(request),bridge.binding(result),out,controller=controller)
    assert json.loads((root/'cycle.json').read_text())==cursor


@pytest.mark.parametrize('mutation',['wrong-parent','wrong-round','skipped-raw-credit','already-supervised','wrong-result','fake-supervised-receipt','lost-carried-frame','changed-admission','wrong-stage'])
def test_raw_authority_mutations_reject_without_advancing_cursor(tmp_path,mutation):
    root,config,request,result,controller,plan,recipe,saved,inherited=raw_fixture(tmp_path)
    if mutation in('wrong-parent','wrong-round','already-supervised'):
        p=root/'cycle.json';v=json.loads(p.read_text())
        if mutation=='wrong-parent':v['current_checkpoint']['sha256']='f'*64
        if mutation=='wrong-round':v['pending_rl_round']=4
        if mutation=='already-supervised':v['phase']='sft'
        write(p,v)
    elif mutation=='wrong-result':
        v=json.loads(result.read_text());v['request_id']='f'*64;write(result,v)
    else:
        r=json.loads(result.read_text());cp=Path(r['checkpoint']['path']);learner=cp.parent
        if mutation=='skipped-raw-credit':
            p=root/'production/state.json';v=json.loads(p.read_text());v['trained_rounds']=2;write(p,v)
        if mutation=='fake-supervised-receipt':
            p=learner/'native-training-receipt.json';v=json.loads(p.read_text());v['policy_operation_completed']=False;v['completed']=True;write(p,v)
        if mutation=='lost-carried-frame':
            saved['training']['sft_progress'].pop('full');torch.save(saved,cp);r['checkpoint']=bridge.binding(cp)
        if mutation=='wrong-stage':
            p=learner/'native-training-receipt.json';v=json.loads(p.read_text());v['stage_id']='relabelled';write(p,v)
        if mutation=='changed-admission':
            p=learner/'rl-data-receipt.json';v=json.loads(p.read_text());v['policy']['checkpoint_sha256']='f'*64;write(p,v)
        r['state']=bridge.binding(root/'production/state.json');r['artifacts']=bridge.inventory(root/'production');write(result,r)
    before=[p.read_bytes()for p in(config,root/'cycle.json',root/'production/state.json')]
    with pytest.raises(ValueError):adapt.consume_raw(config,bridge.binding(request),bridge.binding(result),tmp_path/'raw-consumed.json',controller=controller)
    assert before==[p.read_bytes()for p in(config,root/'cycle.json',root/'production/state.json')]


def test_raw_consume_recovers_after_cursor_commit_without_final_receipt(tmp_path,monkeypatch):
    root,config,request,result,controller,*_=raw_fixture(tmp_path)
    out=tmp_path/'raw-consumed.json';original=adapt.write_json_durable
    def crash(path,value):
        if Path(path)==out:raise RuntimeError('simulated crash after cursor publication')
        return original(path,value)
    monkeypatch.setattr(adapt,'write_json_durable',crash)
    with pytest.raises(RuntimeError,match='simulated crash'):
        adapt.consume_raw(config,bridge.binding(request),bridge.binding(result),out,controller=controller)
    cursor=(root/'cycle.json').read_bytes();assert json.loads(cursor)['next_stage']==0
    assert not out.exists()and out.with_name('raw-consumed-intent.json').exists()
    monkeypatch.setattr(adapt,'write_json_durable',original)
    receipt,_=adapt.consume_raw(config,bridge.binding(request),bridge.binding(result),out,controller=controller)
    assert (root/'cycle.json').read_bytes()==cursor and receipt['supervised_stage_credits_added']==0


@pytest.mark.parametrize('field',['admission','objective','graphs','mandatory_sft_sha256'])
def test_raw_proof_revalidates_exact_admission_evidence(tmp_path,field):
    root,config,request,result,controller,plan,recipe,*_=raw_fixture(tmp_path)
    consumed=tmp_path/'raw-consumed.json';_,pair=adapt.consume_raw(config,bridge.binding(request),bridge.binding(result),consumed,controller=controller)
    package=tmp_path/'package';package.mkdir();(package/'identity.py').write_text('VALUE=1\n')
    future=json.loads(recipe.read_text());future['stages'][1]['parameters']['learning_rate']=1e-5;path=tmp_path/'future.json';write(path,future)
    proof=adapt.prepare(config,plan,'lfm',adapt.binding(path),adapt.binding(consumed),pair,tmp_path/'adaptation/proof.json',package=package)
    value=json.loads(consumed.read_text())
    if field=='mandatory_sft_sha256':value['raw_admission'][field]='f'*64
    else:value['raw_admission'][field]['sha256']='f'*64
    write(consumed,value);payload=json.loads(Path(proof['path']).read_text());payload['consumption']=adapt.binding(consumed);write(Path(proof['path']),payload)
    with pytest.raises(ValueError):adapt.verify(adapt.binding(proof['path']))
