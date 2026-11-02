import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train import native_zero_sgd_adoption as adoption
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.own_output_supervision import identity
from test_static_native_context import row, objective, FP
from test_native_source_correction import _receipt, _write_proof, _row
from natlang_neuralese.data.static_native_context import graphify
from natlang_neuralese.data.native_graph_schema import indexed_graph_metadata
from natlang_neuralese.data.native_sources import _row_bindings


def fixture(tmp_path, monkeypatch, *, carried):
    (tmp_path/'prior').write_text('{}');(tmp_path/'next').write_text('{}')
    original=row();child,_=graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    ident=original['id']
    old=_row_bindings([indexed_graph_metadata(original,tokenizer_fingerprint=FP)])[ident]
    new=_row_bindings([indexed_graph_metadata(child,tokenizer_fingerprint=FP)])[ident]
    prior=_receipt('/old/source','old',[ident],['old-leaf']);current=_receipt('/new/source','new',[ident],['new-leaf'])
    policy=tmp_path/'policy.json';policy.write_text(json.dumps({'objective':objective(mass='token')}))
    pairs=tmp_path/'pairs.jsonl';pairs.write_text(json.dumps({'parent':original,'child':child})+'\n')
    ref=lambda p:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    proof_path=_write_proof(tmp_path,prior,current,[_row(ident,old,new,'old-leaf','new-leaf','corrected',old_communication=False)])
    proof=json.loads(proof_path.read_text());proof['static_native_context_correction']={
        'schema':'natlang.static-native-context-source-correction/1','graph_pairs':ref(pairs),'ce_policy':ref(policy),'rows':1}
    proof_path.write_text(json.dumps(proof))
    opts={'stage_id':'native_full_sft','epochs':1,'_communication_ids':[],
          'communication_quota_policy':{'stage_id':'native_full_sft','visits':7},
          'supervision_objective':identity(objective(mass='token')),'ce_supervision_policy_binding':ref(policy)}
    frame={'schema':'natlang.neuralese-native-sft-training/1','identity':{'objective':None},
       'source_receipts':[prior,{'path':'/held','sha256':'held'}], 'source_row_bindings':{ident:old},
       'source_row_order':[ident], 'exposure_counts':{ident:0},'required_visits':{ident:1},'options':opts,
       'step':4,'global_step':9,'optimizer':{'state':{'a':torch.tensor([1.,2.])}},'rng':{'cpu':torch.tensor([1,2],dtype=torch.uint8)}}
    frame['sft_progress']={'native_full_sft':{k:copy.deepcopy(v) for k,v in frame.items() if k not in ('optimizer','rng','sft_progress')}}
    training=frame
    if carried:
        training={'schema':frame['schema'],'identity':{'objective':'policy'},'options':{'stage_id':'policy'},
            'optimizer':frame['optimizer'],'rng':frame['rng'],'step':2,'global_step':9,
            'exposure_counts':{'policy-episode':3},'sft_progress':frame['sft_progress']}
    parent={'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'lfm2','base_inventory_sha256':'inventory',
        'tokenizer_fingerprint':FP,'marker_ids':[1,2,3,4]},'model_state':{'weight':torch.tensor([3.,4.])},'training':training}
    roles={'native_graphs':{'path':'/new','sha256':'new'},'held_native_graphs':{'path':'/held','sha256':'held'},
        'source_correction':{'path':str(proof_path)},'ce_supervision_policy':ref(policy)}
    monkeypatch.setattr(adoption,'_roles',lambda *args: roles)
    monkeypatch.setattr(adoption,'_stage_parameters',lambda *args: {'epochs':1.,'communication_min_visits':None})
    import natlang_neuralese.train.recipe as recipe
    import natlang_neuralese.data.native_sources as sources
    import natlang_neuralese.train.native_operation_report as reports
    import natlang_neuralese.train.native_supervision as supervision
    import natlang_neuralese.train.own_output_supervision as own
    monkeypatch.setattr(recipe,'_source_correction_continuation',lambda *args:True)
    monkeypatch.setattr(sources,'NativeGraphSource',lambda *a,**k:SimpleNamespace(bindings={ident:new}, file_sha256='new',
        entries=[{'id':ident,'communication':True}],ordered_ids=lambda:[ident]))
    monkeypatch.setattr(reports,'source_receipt',lambda source:current)
    monkeypatch.setattr(supervision,'load_policy',lambda ref:json.loads(policy.read_text()))
    monkeypatch.setattr(own,'prepare_policy',lambda policy,source:policy)
    import natlang_neuralese.train.native_continuation as continuation
    monkeypatch.setattr(continuation,'apply_communication_quota_options',lambda options,*a,**kw:options)
    return parent,ident,ref


@pytest.mark.parametrize('carried',[False,True])
def test_real_correction_full_checkpoint_roundtrip_and_forgery(tmp_path,monkeypatch,carried):
    parent,ident,ref=fixture(tmp_path,monkeypatch,carried=carried)
    child,summary=adoption.admitted_checkpoint(parent,prior_recipe=tmp_path/'prior',successor_recipe=tmp_path/'next',stage_id='native_full_sft')
    assert summary['corrected_rows']==1
    frame=child['training']['sft_progress']['native_full_sft'] if carried else child['training']
    assert frame['required_visits'][ident]==7 and frame['exposure_counts'][ident]==0
    assert child['training']['global_step']==9
    assert torch.equal(child['training']['optimizer']['state']['a'],torch.tensor([1.,2.]))
    p=tmp_path/'parent.pt';c=tmp_path/'child.pt';torch.save(parent,p);torch.save(child,c)
    verified=adoption.verify(ref(p),ref(c),prior_recipe=tmp_path/'prior',successor_recipe=tmp_path/'next',stage_id='native_full_sft')
    assert verified['optimizer_updates']==0 and verified['stage_completed'] is False
    child['model_state']['weight']=torch.tensor([9.,4.]);torch.save(child,c)
    with pytest.raises(ValueError,match='tensor bytes'):adoption.verify(ref(p),ref(c),prior_recipe=tmp_path/'prior',successor_recipe=tmp_path/'next',stage_id='native_full_sft')


def test_publication_rederives_checkpoint_and_refuses_forged_full_state(tmp_path,monkeypatch):
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_zero_sgd_adoption, _authenticated_refresh_publication
    parent,ident,ref=fixture(tmp_path,monkeypatch,carried=True)
    p=tmp_path/'parent.pt';torch.save(parent,p)
    prior=tmp_path/'old-recipe.json';prior.write_text('{}')
    successor=tmp_path/'new-recipe.json';successor.write_text('{}')
    from natlang_neuralese.train import checkpoint_safety
    monkeypatch.setattr(checkpoint_safety.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    output=tmp_path/'staged';proof=adoption.prepare(ref(p),prior_recipe=prior,successor_recipe=successor,
        stage_id='native_full_sft',output=output)
    state=tmp_path/'state.json';state.write_text(json.dumps({'checkpoint':str(p),'phase':'awaiting_semantic_refresh','round':4}))
    (tmp_path/'controller.lock').touch()
    publication_path=tmp_path/'publication.json'
    published=publish_zero_sgd_adoption(state,proof_ref=ref(output/'proof.json'),out=publication_path)
    assert published['optimizer_updates']==0 and published['graph_visits']==0 and published['stage_completed'] is False
    committed=json.loads(state.read_text())
    assert committed['checkpoint']==str(output/'checkpoint.pt')
    assert committed['phase']=='awaiting_semantic_refresh' and committed['round']==4
    checked=_authenticated_refresh_publication(ref(publication_path))
    assert checked['publication']==published and checked['authenticated_ancestors']=={ref(p)['sha256'],proof['checkpoint']['sha256']}
    assert committed['direct_sft_refresh_history'][-1]['sha256']==ref(publication_path)['sha256']
    child=torch.load(output/'checkpoint.pt',weights_only=False)
    (output/'checkpoint.pt').chmod(0o600)
    child['training']['optimizer']['state']['a'][0]=99
    torch.save(child,output/'checkpoint.pt')
    # Even a consistent rebind of both JSON documents cannot hide changed weights.
    proof['checkpoint']=ref(output/'checkpoint.pt')
    (output/'proof.json').write_text(json.dumps(proof))
    published['checkpoint']=proof['checkpoint'];published['proof']=ref(output/'proof.json')
    publication_path.write_text(json.dumps(published))
    with pytest.raises(ValueError,match='tensor bytes'):_authenticated_refresh_publication(ref(publication_path))


@pytest.mark.parametrize('tamper',['model','optimizer','rng','global_step','quota','exposure','ce_policy'])
def test_independent_verification_rejects_full_state_or_metadata_forgery(tmp_path,monkeypatch,tamper):
    parent,ident,ref=fixture(tmp_path,monkeypatch,carried=True)
    child,_=adoption.admitted_checkpoint(parent,prior_recipe=tmp_path/'prior',successor_recipe=tmp_path/'next',stage_id='native_full_sft')
    child=copy.deepcopy(child)
    frame=child['training']['sft_progress']['native_full_sft']
    if tamper=='model':child['model_state']['weight'][0]=99
    elif tamper=='optimizer':child['training']['optimizer']['state']['a'][0]=99
    elif tamper=='rng':child['training']['rng']['cpu'][0]=99
    elif tamper=='global_step':child['training']['global_step']+=1
    elif tamper=='quota':frame['required_visits'][ident]=1
    elif tamper=='exposure':frame['exposure_counts'][ident]=1
    elif tamper=='ce_policy':frame['options']['ce_supervision_policy_binding']['sha256']='f'*64
    p=tmp_path/'p.pt';c=tmp_path/'c.pt';torch.save(parent,p);torch.save(child,c)
    with pytest.raises(ValueError):adoption.verify(ref(p),ref(c),prior_recipe=tmp_path/'prior',successor_recipe=tmp_path/'next',stage_id='native_full_sft')


def test_real_source_and_recipe_admission_including_family_rebinding(tmp_path,monkeypatch):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.native_operation_report import source_receipt
    from natlang_neuralese.train.native_supervision import SCHEMA as POLICY_SCHEMA
    from natlang_neuralese.train.own_output_supervision import policy_objective
    original=row();target,_=graphify(original,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    def put(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value)+'\n');return path
    def ref(path):
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    old_path=put('old.jsonl',original);new_path=put('new.jsonl',target)
    old=NativeGraphSource(old_path,FP,required_split='train',role_ids=(58,59,60,61))
    new=NativeGraphSource(new_path,FP,required_split='train',role_ids=(58,59,60,61))
    old_receipt,new_receipt=source_receipt(old),source_receipt(new);ident=original['id']
    old_binding,new_binding=old.bindings[ident],new.bindings[ident]
    old_map=put('old-map.json',{'schema':'natlang.logical-family-map/1','source_sha256':old.file_sha256,
        'rows':[{'id':ident,'binding':old_binding,'family':'general-agent'}]})
    new_map=put('new-map.json',{'schema':'natlang.logical-family-map/1','source_sha256':new.file_sha256,
        'rows':[{'id':ident,'binding':new_binding,'family':'general-agent'}]})
    base=objective(mass='token');balance={'key':'logical-family','max_scale':8,'normalization':'declared/1',
                                         'scales':{'general-agent':1.0}}
    before={**base,'family_balance':{**balance,'map':{k:v for k,v in ref(old_map).items() if k!='bytes'}}}
    after={**base,'family_balance':{**balance,'map':{k:v for k,v in ref(new_map).items() if k!='bytes'}}}
    old_policy={'schema':POLICY_SCHEMA,'sources':[],'objective':before}
    new_policy={'schema':POLICY_SCHEMA,'sources':[],'objective':after,'inventory':[ref(new_path)],
                'objective_transition':{'from':identity(policy_objective(old_policy))['sha256']}}
    old_ce=put('old-ce.json',old_policy);new_ce=put('new-ce.json',new_policy)
    proof_path=_write_proof(tmp_path,old_receipt,new_receipt,[_row(ident,old_binding,new_binding,
        old.file_sha256,new.file_sha256,'corrected',old_communication=False)])
    pairs=put('pairs.jsonl',{'parent':original,'child':target})
    proof=json.loads(proof_path.read_text());proof['static_native_context_correction']={
        'schema':'natlang.static-native-context-source-correction/1','graph_pairs':ref(pairs),'ce_policy':ref(old_ce),'rows':1}
    proof_path.write_text(json.dumps(proof))
    held={'path':'/held','sha256':'held','binding':'held'}
    old_roles={'native_graphs':{**ref(old_path),'binding':'train'},'held_native_graphs':held,
               'ce_supervision_policy':{**ref(old_ce),'binding':'ce'}}
    new_roles={'native_graphs':{**ref(new_path),'binding':'train'},'continued_native_graphs':{**ref(old_path),'binding':'previous'},
        'held_native_graphs':held,'ce_supervision_policy':{**ref(new_ce),'binding':'ce'},
        'source_correction':{**ref(proof_path),'binding':'correction'}}
    monkeypatch.setattr(adoption,'_roles',lambda path,stage:old_roles if str(path).endswith('old-recipe') else new_roles)
    opts={'stage_id':'native_full_sft','epochs':1,'_communication_ids':[],
        'communication_quota_policy':{'stage_id':'native_full_sft','visits':7},
        'supervision_objective':identity(policy_objective(old_policy)),'ce_supervision_policy_binding':ref(old_ce)}
    training={'schema':'natlang.neuralese-native-sft-training/1','identity':{'objective':None},'options':opts,
        'source_receipts':[old_receipt,{'path':'/held','sha256':'held'}],'source_row_bindings':old.bindings,
        'source_row_order':[ident],'exposure_counts':{ident:0},'required_visits':{ident:1},
        'step':4,'global_step':9,'optimizer':{'state':{'x':torch.tensor([1.])}},'rng':{'cpu':torch.tensor([1])}}
    parent={'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'lfm2','base_inventory_sha256':'inventory',
        'tokenizer_fingerprint':FP,'marker_ids':[58,59,60,61]},'model_state':{'weight':torch.tensor([3.])},'training':training}
    from natlang_neuralese.train.native_quota_aware_ce import make_basis
    new_policy['quota_aware_basis']=make_basis(training,stage_id='native_full_sft',source_sha256=old.file_sha256,
        candidate_sha256=new.file_sha256,epochs=1.,communication_min_visits=7)
    new_ce.write_text(json.dumps(new_policy)+'\n')
    new_roles['ce_supervision_policy']={**ref(new_ce),'binding':'ce'}
    monkeypatch.setattr(adoption,'_stage_parameters',lambda *args: {'epochs':1.,'communication_min_visits':7})
    import natlang_neuralese.train.native_continuation as continuation
    monkeypatch.setattr(continuation,'apply_communication_quota_options',lambda options,*a,**kw:options)
    result,_=adoption.admitted_checkpoint(parent,prior_recipe='old-recipe',successor_recipe='new-recipe',stage_id='native_full_sft')
    assert result['training']['required_visits'][ident]==7
    assert result['training']['options']['ce_supervision_policy_binding']=={'path':str(new_ce),'sha256':ref(new_ce)['sha256']}
    assert result['training']['exposure_counts'][ident]==0


def test_publication_cannot_interrupt_active_controller(tmp_path):
    import fcntl
    from natlang_neuralese.rl.native_direct_sft_recovery import publish_zero_sgd_adoption
    state=tmp_path/'state.json';state.write_text('{}')
    lock_path=tmp_path/'controller.lock';lock_path.touch()
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            publish_zero_sgd_adoption(state,proof_ref={'path':'missing','sha256':'f'*64},out=tmp_path/'publication.json')
    assert state.read_text()=='{}'


@pytest.mark.parametrize('which',['recipe','parent','child'])
def test_verifier_rejects_identity_changes_during_derivation(tmp_path,monkeypatch,which):
    parent,_,ref=fixture(tmp_path,monkeypatch,carried=False)
    prior=tmp_path/'prior';successor=tmp_path/'next'
    child,_=adoption.admitted_checkpoint(parent,prior_recipe=prior,successor_recipe=successor,stage_id='native_full_sft')
    p=tmp_path/'p.pt';c=tmp_path/'c.pt';torch.save(parent,p);torch.save(child,c)
    actual=adoption.admitted_checkpoint
    def mutate(*args,**kwargs):
        result=actual(*args,**kwargs)
        if which=='recipe': successor.write_text('{"different":true}')
        else:
            import os
            path=p if which=='parent' else c
            st=path.stat();os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns+1000000000))
        return result
    monkeypatch.setattr(adoption,'admitted_checkpoint',mutate)
    with pytest.raises(ValueError,match='changed while verifying'):
        adoption.verify(ref(p),ref(c),prior_recipe=prior,successor_recipe=successor,stage_id='native_full_sft')


@pytest.mark.parametrize('carried',[False,True])
@pytest.mark.parametrize('static_change',[False,True])
@pytest.mark.parametrize('deferred_comm',[False,True])
def test_mixed_static_correction_quality_deferral_and_append_preserves_earned_state(tmp_path,monkeypatch,carried,static_change,deferred_comm):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.native_operation_report import source_receipt
    from natlang_neuralese.train.native_supervision import SCHEMA as POLICY_SCHEMA
    from natlang_neuralese.train.own_output_supervision import policy_objective
    from natlang_neuralese.train.native_quota_aware_ce import make_basis
    def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def put(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value)+'\n');return path
    def original(name):
        value=copy.deepcopy(row());value['id']=name;return value
    keep,deferred,static,old_comm,added,added_comm=(original(name) for name in ('keep','deferred','static','old-comm','added','added-comm'))
    fixed,_=graphify(static,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    if not static_change: fixed=static
    if deferred_comm: deferred,_=graphify(deferred,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    old_comm,_=graphify(old_comm,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    added_comm,_=graphify(added_comm,objective(mass='token'),max_tokens=32,consumer_context=6,state_tokens=2)
    old_path=tmp_path/'old.jsonl';new_path=tmp_path/'new.jsonl'
    old_path.write_text(''.join(json.dumps(value)+'\n' for value in (keep,deferred,static,old_comm)))
    new_path.write_text(''.join(json.dumps(value)+'\n' for value in ((keep,old_comm,fixed,added,added_comm) if static_change else (keep,fixed,old_comm,added,added_comm))))
    old=NativeGraphSource(old_path,FP,required_split='train',role_ids=(58,59,60,61))
    new=NativeGraphSource(new_path,FP,required_split='train',role_ids=(58,59,60,61))
    old_receipt,new_receipt=source_receipt(old),source_receipt(new)
    def family_rows(source):
        return [{'id':entry['id'],'binding':entry['binding'],
                 'family':'native-communication' if entry['id'] in (('old-comm','added-comm','deferred') if deferred_comm else ('old-comm','added-comm')) else 'general-agent'}
                for entry in source.entries]
    old_map=put('old-map.json',{'schema':'natlang.logical-family-map/1','source_sha256':old.file_sha256,'rows':family_rows(old)})
    new_map=put('new-map.json',{'schema':'natlang.logical-family-map/1','source_sha256':new.file_sha256,'rows':family_rows(new)})
    balance={'key':'logical-family','max_scale':8,'normalization':'declared/1','scales':{'general-agent':1.,'native-communication':1.}}
    old_policy={'schema':POLICY_SCHEMA,'sources':[],'objective':{**objective(mass='token'),'family_balance':{**balance,'map':ref(old_map)}}}
    old_ce=put('old-ce.json',old_policy)
    opts={'stage_id':'native_full_sft','epochs':1.,'_communication_ids':['old-comm']+(['deferred'] if deferred_comm else []),
          'communication_quota_policy':{'stage_id':'native_full_sft','visits':7},
          '_objective_required_visits':{'old-comm':7},
          'supervision_objective':identity(policy_objective(old_policy)),'ce_supervision_policy_binding':ref(old_ce)}
    frame={'schema':'natlang.neuralese-native-sft-training/1','identity':{'objective':None},'options':opts,
        'source_receipts':[old_receipt,{'path':'/held','sha256':'held'}], 'source_row_bindings':old.bindings,
        'source_row_order':[entry['id'] for entry in old.entries],
        'exposure_counts':{'keep':1,'deferred':1,'static':0,'old-comm':2},
        'required_visits':{'keep':1,'deferred':1,'static':1,'old-comm':7},'step':4,'global_step':9,
        'optimizer':{'state':{'a':torch.tensor([1.,2.])}},'rng':{'cpu':torch.tensor([1,2],dtype=torch.uint8)}}
    next_balance={**balance,'map':ref(new_map),'scales':{'general-agent':.7,'native-communication':1.2}}
    new_policy={'schema':POLICY_SCHEMA,'sources':[],'objective':{**objective(mass='token'),'family_balance':next_balance},
                'inventory':[ref(new_path)],'objective_transition':{'from':identity(policy_objective(old_policy))['sha256']},
                'quota_aware_basis':make_basis(frame,stage_id='native_full_sft',source_sha256=old.file_sha256,
                    candidate_sha256=new.file_sha256,epochs=1.5,communication_min_visits=7)}
    new_ce=put('new-ce.json',new_policy)
    correction_rows=[_row('static',old.bindings['static'],new.bindings['static'],old.file_sha256,new.file_sha256,'corrected',old_communication=False),
                     _row('deferred',old.bindings['deferred'],None,old.file_sha256,None,'deferred','quality exclusion; source retained',old_communication=deferred_comm),
                     *[_row(key,None,new.bindings[key],None,new.file_sha256,'added') for key in ('added','added-comm')]]
    if not static_change: correction_rows=correction_rows[1:]
    proof_path=_write_proof(tmp_path,old_receipt,new_receipt,correction_rows)
    pairs=put('pairs.jsonl',{'parent':static,'child':fixed})
    proof=json.loads(proof_path.read_text());proof['static_native_context_correction']={
        'schema':'natlang.static-native-context-source-correction/1','graph_pairs':ref(pairs),'ce_policy':ref(old_ce),'rows':1}
    if not static_change: del proof['static_native_context_correction']
    proof_path.write_text(json.dumps(proof))
    held={'path':'/held','sha256':'held','binding':'held'}
    old_roles={'native_graphs':{**ref(old_path),'binding':'train'},'held_native_graphs':held,
               'ce_supervision_policy':{**ref(old_ce),'binding':'ce'}}
    new_roles={'native_graphs':{**ref(new_path),'binding':'train'},'continued_native_graphs':{**ref(old_path),'binding':'previous'},
               'held_native_graphs':held,'ce_supervision_policy':{**ref(new_ce),'binding':'ce'},
               'source_correction':{**ref(proof_path),'binding':'correction'}}
    monkeypatch.setattr(adoption,'_roles',lambda path,stage:old_roles if str(path).endswith('old-recipe') else new_roles)
    monkeypatch.setattr(adoption,'_stage_parameters',lambda *args:{'epochs':1.5,'communication_min_visits':7})
    frame['sft_progress']={'native_full_sft':{k:copy.deepcopy(v) for k,v in frame.items() if k not in ('optimizer','rng','sft_progress')}}
    training=frame if not carried else {'identity':{'objective':'policy'},'options':{'stage_id':'policy'},
        'optimizer':frame['optimizer'],'rng':frame['rng'],'step':2,'global_step':9,
        'exposure_counts':{'policy-episode':3},'sft_progress':frame['sft_progress']}
    parent={'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'lfm2','base_inventory_sha256':'inventory',
        'tokenizer_fingerprint':FP,'marker_ids':[58,59,60,61]},'model_state':{'weight':torch.tensor([3.,4.])},'training':training}
    before=copy.deepcopy(parent)
    child,summary=adoption.admitted_checkpoint(parent,prior_recipe='old-recipe',successor_recipe='new-recipe',stage_id='native_full_sft')
    assert (summary['corrected_rows'],summary['deferred_rows'],summary['added_rows'])==(int(static_change),1,2)
    active=child['training']['sft_progress']['native_full_sft'] if carried else child['training']
    assert active['exposure_counts']=={'keep':1,'deferred':1,'static':0,'old-comm':2,'added':0,'added-comm':0}
    assert active['required_visits']['added-comm']==7 and active['required_visits']['added']==1
    assert active['required_visits']['old-comm']==7
    if static_change:
        assert active['required_visits']['static']==7
        assert active['options']['_objective_required_visits']['static']==7
    assert active['pending_source_rows']['deferred']['retired_exposure']['visits']==1
    assert active['options']['quota_aware_ce_history'][-1]['basis']==new_policy['quota_aware_basis']
    adoption._same(before,parent)
    for key in ('optimizer','rng','global_step','step'):adoption._same(parent['training'][key],child['training'][key])
    adoption._same(parent['model_state'],child['model_state'])
    old_recipe=put('old-recipe',{}); new_recipe=put('new-recipe',{})
    parent_path=tmp_path/'parent.pt';child_path=tmp_path/'child.pt'
    torch.save(parent,parent_path);torch.save(child,child_path)
    checked=adoption.verify(ref(parent_path),ref(child_path),prior_recipe=old_recipe,successor_recipe=new_recipe,stage_id='native_full_sft')
    assert checked['optimizer_updates']==0 and checked['graph_visits']==0
    forged=copy.deepcopy(child)
    forged_frame=forged['training']['sft_progress']['native_full_sft'] if carried else forged['training']
    forged_frame['pending_source_rows']['deferred']['retired_exposure']['visits']=0
    torch.save(forged,child_path)
    with pytest.raises(ValueError):
        adoption.verify(ref(parent_path),ref(child_path),prior_recipe=old_recipe,successor_recipe=new_recipe,stage_id='native_full_sft')
    # A measured CE basis is bound to actual old credits and exact requested quotas.
    changed=copy.deepcopy(parent)
    target=changed['training']['sft_progress']['native_full_sft'] if carried else changed['training']
    target['exposure_counts']['keep']=0
    with pytest.raises(ValueError,match='quotas or exposure changed'):
        adoption.admitted_checkpoint(changed,prior_recipe='old-recipe',successor_recipe='new-recipe',stage_id='native_full_sft')
    monkeypatch.setattr(adoption,'_stage_parameters',lambda *args:{'epochs':1.5,'communication_min_visits':6})
    with pytest.raises(ValueError,match='quota settings'):
        adoption.admitted_checkpoint(parent,prior_recipe='old-recipe',successor_recipe='new-recipe',stage_id='native_full_sft')


def test_adoption_reserves_checkpoint_and_preserves_failed_serialization(tmp_path,monkeypatch):
    from natlang_neuralese.train import checkpoint_safety, trajectory_state
    parent,_,ref=fixture(tmp_path,monkeypatch,carried=True)
    parent_path=tmp_path/'parent.pt';torch.save(parent,parent_path)
    prior=tmp_path/'prior-recipe.json';prior.write_text('{}')
    successor=tmp_path/'next-recipe.json';successor.write_text('{}')
    monkeypatch.setattr(checkpoint_safety.shutil,'disk_usage',lambda _:SimpleNamespace(free=40*1024**3))
    low=tmp_path/'low'
    with pytest.raises(checkpoint_safety.CheckpointSpaceUnavailable):
        adoption.prepare(ref(parent_path),prior_recipe=prior,successor_recipe=successor,
                         stage_id='native_full_sft',output=low)
    assert not (low/'checkpoint.pt').exists()
    monkeypatch.setattr(checkpoint_safety.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**3))
    def fail_save(state, stream):
        assert stream.stream.fileno() >= 0
        stream.write(b'partial checkpoint')
        raise RuntimeError('serialization failed')
    monkeypatch.setattr(trajectory_state.torch,'save',fail_save)
    failed=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='serialization failed'):
        adoption.prepare(ref(parent_path),prior_recipe=prior,successor_recipe=successor,
                         stage_id='native_full_sft',output=failed)
    pending=failed/'checkpoint.pt.reserved-pending'
    assert pending.read_bytes()[:18]==b'partial checkpoint'
    assert pending.stat().st_blocks*512 >= pending.stat().st_size
    assert (failed/'checkpoint.pt.reservation.json').is_file()
    assert not (failed/'checkpoint.pt').exists()
