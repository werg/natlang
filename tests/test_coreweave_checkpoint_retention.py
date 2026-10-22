import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'checkpoint_retention', Path(__file__).resolve().parents[1] / 'scripts/coreweave/checkpoint_retention.py')
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)


def setup_export(tmp_path, role):
    name = lambda step: f'{role}/{"a" * 64}/{role}-native-checkpoint-step-{step}.pt'
    state = {'last_verified_step': 3, 'receipt': {'private': True, 'run': 'owner/project/run',
             'identity': {'sha256': 'a' * 64, 'bytes': 40},
             'independent_verification': {'sha256': 'a' * 64, 'bytes': 40}, 'name': name(3)}}
    files = {n: SimpleNamespace(name=n, size=40, md5='md5') for n in
             [name(1), name(2), name(3), name(4), f'{role}/source.tar', f'{role}/inference.pt', 'metadata.json']}
    same_step = f'{role}/{"b" * 64}/{role}-native-checkpoint-step-3.pt'
    files[same_step] = SimpleNamespace(name=same_step, size=40, md5='other')
    deleted = []
    for n, f in files.items():
        def remove(n=n):
            assert list((tmp_path / 'journal').glob('*-intent-*.json'))
            deleted.append(n)
            del files[n]
        f.delete = remove
    run = SimpleNamespace(files=lambda **kwargs: list(files.values()))
    api = SimpleNamespace(run=lambda path: run)
    state_path = tmp_path / 'state.json'
    state_path.write_text(json.dumps(state))
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text(json.dumps({'models': {role: {'run': 'owner/project/run',
        'state': str(state_path), 'protected_files': [name(2)]}}}))
    return name, state, files, deleted, api, state_path, policy_path


@pytest.mark.parametrize('role', ['lfm', 'mellum'])
def test_verified_replacement_prunes_only_superseded_unprotected_full_state(tmp_path, role):
    name, state, files, deleted, api, _, policy = setup_export(tmp_path, role)
    invoke = lambda **kwargs: retention.tick(policy, tmp_path / 'journal', api=api,
                                            ownership_directory=tmp_path / 'owners', **kwargs)
    assert invoke()[0]['candidates'] == 1
    assert not deleted
    intent = json.loads(next((tmp_path / 'journal').glob('*-intent-*.json')).read_text())
    assert intent['verified_export_state'] == state
    assert intent['protected_files'] == [name(2)]
    assert intent['policy']['path'] == str(policy)
    invoke(apply=True)
    assert deleted == [name(1)]
    assert {name(2), name(3), name(4), f'{role}/source.tar', f'{role}/inference.pt', 'metadata.json',
            f'{role}/{"b" * 64}/{role}-native-checkpoint-step-3.pt'} == set(files)
    # A resumed sweep does not delete again or reinterpret the live latest.
    assert invoke(apply=True)[0]['candidates'] == 0


@pytest.mark.parametrize('damage', ['readback', 'private', 'step', 'missing', 'size', 'run'])
def test_invalid_recovery_authority_never_deletes(tmp_path, damage):
    name, state, files, deleted, api, state_path, policy = setup_export(tmp_path, 'mellum')
    if damage == 'readback': state['receipt']['independent_verification']['sha256'] = 'b' * 64
    if damage == 'private': state['receipt']['private'] = False
    if damage == 'step': state['last_verified_step'] = 4
    if damage == 'missing': del files[name(3)]
    if damage == 'size': files[name(3)].size += 1
    if damage == 'run': state['receipt']['run'] = 'owner/other/run'
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        retention.tick(policy, tmp_path / 'journal', apply=True, api=api, ownership_directory=tmp_path / 'owners')
    assert not deleted


def test_conflicting_policy_cannot_take_over_same_run_through_another_journal(tmp_path):
    name, _, _, deleted, api, _, policy = setup_export(tmp_path, 'mellum')
    owners = tmp_path / 'owners'
    retention.tick(policy, tmp_path / 'journal', apply=True, api=api, ownership_directory=owners)
    other = tmp_path / 'other-policy.json'
    data = json.loads(policy.read_text())
    data['models']['mellum']['protected_files'] = []
    other.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='another explicit retention policy'):
        retention.tick(other, tmp_path / 'another-journal', apply=True, api=api, ownership_directory=owners)
    assert deleted == [name(1)]


def with_inference_views(tmp_path):
    name,state,files,deleted,api,state_path,policy=setup_export(tmp_path,'mellum')
    def add(step,digest):
        receipt={'private':True,'run':'owner/project/run','role':'mellum',
            'name':f'mellum/{digest}/mellum-native-inference-step-{step}.pt',
            'identity':{'sha256':digest,'bytes':20},'independent_verification':{'sha256':digest,'bytes':20}}
        file=SimpleNamespace(name=receipt['name'],size=20,md5='view')
        def remove():
            assert list((tmp_path/'journal').glob('*-intent-*.json'))
            deleted.append(file.name);del files[file.name]
        file.delete=remove;files[file.name]=file
        return receipt
    receipts={i:add(i,hex(10+i)[2:]*64)for i in (1,2,3,4)}
    parent=lambda step:{'schema':'natlang.native-inference-source/1',
        'training_checkpoint_sha256':'a'*64,'training_checkpoint_bytes':40,'step':step}
    view=tmp_path/'view-state.json';view.write_text(json.dumps({'last_verified_step':3,
        'receipt':receipts[3],'inference_source':parent(3)}))
    declaration=json.loads(policy.read_text());declaration['models']['mellum']['inference_state']=str(view)
    policy.write_text(json.dumps(declaration))
    return name,files,deleted,api,policy,view,receipts,parent


def test_inference_retention_protects_active_and_incoming_consumers_and_their_full_parents(tmp_path):
    name,files,deleted,api,policy,view,receipts,parent=with_inference_views(tmp_path)
    consumer=tmp_path/'consumer';request=consumer/'request';request.mkdir(parents=True)
    (request/'request.json').write_text(json.dumps({'evaluation_provenance':{
        'source_checkpoint_sha256':receipts[1]['identity']['sha256'],
        'source_export':receipts[1],'inference_source':parent(1)}}))
    (consumer/'native-async-state.json').write_text(json.dumps({'active':{
        'request':str(request),'step':1,'checkpoint_sha256':receipts[1]['identity']['sha256']}}))
    incoming=consumer/'incoming';incoming.mkdir();(incoming/'fetch.receipt.json').write_text(json.dumps(receipts[2]))
    declaration=json.loads(policy.read_text());declaration['models']['mellum']['inference_ingesters']=[str(consumer)]
    policy.write_text(json.dumps(declaration))
    invoke=lambda:retention.tick(policy,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert invoke()[0]['candidates']==0 and not deleted
    (incoming/'fetch.receipt.json').unlink()
    assert invoke()[0]['candidates']==1 and deleted==[receipts[2]['name']]
    assert name(1)in files and receipts[1]['name']in files and receipts[3]['name']in files and receipts[4]['name']in files


@pytest.mark.parametrize('damage',['get','parent_step','future_step','run'])
def test_invalid_inference_authority_prevents_all_deletions(tmp_path,damage):
    _,_,deleted,api,policy,view,_,_=with_inference_views(tmp_path)
    state=json.loads(view.read_text())
    if damage=='get':state['receipt']['independent_verification']['sha256']='0'*64
    if damage=='parent_step':state['inference_source']['step']=2
    if damage=='future_step':state['last_verified_step']=state['inference_source']['step']=4
    if damage=='run':state['receipt']['run']='foreign/run'
    view.write_text(json.dumps(state))
    with pytest.raises(ValueError):retention.tick(policy,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert not deleted


def test_pending_inference_export_retains_full_parent_and_physical_view(tmp_path):
    name,_,deleted,api,policy,view,receipts,_=with_inference_views(tmp_path)
    fullstate=json.loads((tmp_path/'state.json').read_text());fullreceipt=fullstate['receipt'].copy()
    fullreceipt['name']=name(1)
    pending=tmp_path/'pending.json';pending.write_text(json.dumps({'step':1,
        'identity':receipts[1]['identity'],'full_private_receipt':fullreceipt,
        'full_training_checkpoint':{'step':1,'sha256':'a'*64,'bytes':40}}))
    declaration=json.loads(policy.read_text());declaration['models']['mellum']['inference_pending']=[str(pending)]
    policy.write_text(json.dumps(declaration))
    retention.tick(policy,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert deleted==[receipts[2]['name']]


def test_incoming_view_keeps_its_full_parent_before_request_provenance_exists(tmp_path):
    name,files,deleted,api,policy,_,receipts,_=with_inference_views(tmp_path)
    consumer=tmp_path/'consumer';incoming=consumer/'incoming';incoming.mkdir(parents=True)
    (incoming/'fetch.receipt.json').write_text(json.dumps(receipts[1]))
    declaration=json.loads(policy.read_text());declaration['models']['mellum']['inference_ingesters']=[str(consumer)]
    policy.write_text(json.dumps(declaration))
    retention.tick(policy,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert deleted==[receipts[2]['name']] and name(1)in files


@pytest.mark.parametrize('damage',['active_receipt','active_step','pending_parent','pending_step'])
def test_mismatched_valid_dependency_receipts_fail_closed(tmp_path,damage):
    _,_,deleted,api,policy,_,receipts,parent=with_inference_views(tmp_path)
    declaration=json.loads(policy.read_text())
    if damage.startswith('active'):
        consumer=tmp_path/'consumer';request=consumer/'request';request.mkdir(parents=True)
        (request/'request.json').write_text(json.dumps({'evaluation_provenance':{
            'source_checkpoint_sha256':receipts[1]['identity']['sha256'],
            'source_export':receipts[2]if damage=='active_receipt'else receipts[1],
            'inference_source':parent(1)}}))
        (consumer/'native-async-state.json').write_text(json.dumps({'active':{'request':str(request),
            'step':2 if damage=='active_step'else 1,'checkpoint_sha256':receipts[1]['identity']['sha256']}}))
        declaration['models']['mellum']['inference_ingesters']=[str(consumer)]
    else:
        receipt=json.loads((tmp_path/'state.json').read_text())['receipt'];pending=tmp_path/'pending.json'
        pending.write_text(json.dumps({'step':2 if damage=='pending_step'else 3,
            'identity':receipts[1]['identity'],'full_private_receipt':receipt,
            'full_training_checkpoint':{'step':3,'sha256':'f'*64 if damage=='pending_parent'else'a'*64,'bytes':40}}))
        declaration['models']['mellum']['inference_pending']=[str(pending)]
    policy.write_text(json.dumps(declaration))
    with pytest.raises(ValueError):retention.tick(policy,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert not deleted


@pytest.mark.parametrize('damage',[None,'missing_receipt','readback','run','identity','name','missing_object','size'])
def test_ahead_inference_requires_exact_private_parent_without_advancing_full_frontier(tmp_path,damage):
    name,files,deleted,api,policy,view,receipts,parent=with_inference_views(tmp_path)
    state_path=tmp_path/'state.json';before=state_path.read_bytes()
    full=json.loads(before)['receipt'];full['name']=name(4)
    advanced={'last_verified_step':4,'receipt':receipts[4],'inference_source':parent(4),
              'derivation':{'full_private_receipt':full}}
    if damage=='missing_receipt':advanced.pop('derivation')
    if damage=='readback':full['independent_verification']['sha256']='f'*64
    if damage=='run':full['run']='foreign/run'
    if damage=='identity':full['identity']={**full['identity'],'sha256':'f'*64}
    if damage=='name':full['name']=name(3)
    if damage=='missing_object':del files[name(4)]
    if damage=='size':files[name(4)].size+=1
    view.write_text(json.dumps(advanced))
    invoke=lambda:retention.tick(policy,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    if damage:
        with pytest.raises(ValueError):invoke()
        assert not deleted
    else:
        assert invoke()[0]['verified_latest_step']==3
        assert set(deleted)=={name(1),receipts[1]['name'],receipts[2]['name'],receipts[3]['name']}
        assert name(3)in files and name(4)in files and receipts[4]['name']in files
        intent=json.loads(next((tmp_path/'journal').glob('*-intent-*.json')).read_text())
        assert intent['verified_export_state']['last_verified_step']==3
        assert intent['verified_inference_export_state']==advanced
        assert name(4)in intent['protected_files']
    assert state_path.read_bytes()==before


def owned_fixture(tmp_path):
    _, _, files, deleted, api, _, path = setup_export(tmp_path, 'lfm')
    name = lambda digest: f'lfm/{digest}/lfm-{digest}.pt'
    for digest in ('c'*64, 'd'*64, 'e'*64):
        file = SimpleNamespace(name=name(digest), size=50, md5='owned')
        def remove(file=file):
            assert list((tmp_path/'journal').glob('*-intent-*.json'))
            deleted.append(file.name); del files[file.name]
        file.delete = remove; files[file.name] = file
    state = tmp_path/'owned-state.json'
    state.write_text(json.dumps({'receipt': {'role':'lfm','run':'owner/project/run',
        'private':True,'name':name('e'*64),'identity':{'sha256':'e'*64,'bytes':50},
        'independent_verification':{'sha256':'e'*64,'bytes':50}}}))
    policy = json.loads(path.read_text()); declaration = policy['models']['lfm']
    declaration['owned_checkpoint_retention'] = {'recovery_states':[str(state)],
        'protected_sha256':[], 'pending_exports':[], 'discarded_files':[
            {'name':name('c'*64),'sha256':'c'*64,'bytes':50,'reason':'superseded; no active consumer'}]}
    path.write_text(json.dumps(policy))
    return name,files,deleted,api,path,policy,state


def test_owned_retention_deletes_only_explicit_discard_and_keeps_unknown_publications(tmp_path):
    name,files,deleted,api,path,_,_ = owned_fixture(tmp_path)
    result=retention.tick(path,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert result[0]['candidates']==1  # primary recovery aliases remain protected too
    assert name('c'*64) in deleted and name('d'*64) in files and name('e'*64) in files
    intents=[json.loads(p.read_text()) for p in (tmp_path/'journal').glob('*-intent-*.json')]
    assert intents[0]['owned_checkpoint_retention']['protected_sha256']==['a'*64,'e'*64]
    assert retention.tick(path,tmp_path/'journal',apply=True,api=api,
        ownership_directory=tmp_path/'owners')[0]['candidates']==0


@pytest.mark.parametrize('damage',['protected','pending','parent','bad_size','bad_readback','bad_run',
    'unknown_name','empty_reason','duplicate','missing_recovery'])
def test_owned_retention_refuses_invalid_or_current_custody_before_deletion(tmp_path,damage):
    name,files,deleted,api,path,policy,state = owned_fixture(tmp_path)
    declaration=policy['models']['lfm']['owned_checkpoint_retention']; item=declaration['discarded_files'][0]
    if damage=='protected':declaration['protected_sha256']=['c'*64]
    if damage in ('pending','parent'):
        pending=tmp_path/'pending.json';value={'sha256':'c'*64 if damage=='pending' else 'f'*64,'bytes':50}
        if damage=='parent':value['owned_parent']={'sha256':'c'*64,'bytes':50}
        pending.write_text(json.dumps(value));declaration['pending_exports']=[str(pending)]
    if damage=='bad_size':item['bytes']=51
    if damage in ('bad_readback','bad_run'):
        v=json.loads(state.read_text())
        if damage=='bad_readback':v['receipt']['independent_verification']['sha256']='a'*64
        else:v['receipt']['run']='other/project/run'
        state.write_text(json.dumps(v))
    if damage=='unknown_name':item['name']='lfm/source.tar'
    if damage=='empty_reason':item['reason']=' '
    if damage=='duplicate':declaration['discarded_files'].append(dict(item))
    if damage=='missing_recovery':del files[name('e'*64)]
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        retention.tick(path,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert not deleted


@pytest.mark.parametrize('custody',['canonical_recovery','explicit_owned','inference_parent'])
def test_owned_discard_preserves_existing_custody_across_aliases(tmp_path,custody):
    name,files,deleted,api,path,policy,_=owned_fixture(tmp_path)
    declaration=policy['models']['lfm'];item=declaration['owned_checkpoint_retention']['discarded_files'][0]
    if custody=='canonical_recovery':
        item.update(name=name('a'*64),sha256='a'*64,bytes=40)
        files[item['name']]=SimpleNamespace(name=item['name'],size=40,md5='alias')
    elif custody=='explicit_owned':declaration['protected_files'].append(name('c'*64))
    else:
        # This exact parent spelling is also supplied by inference_dependencies
        # for current views/active requests. Its alternate owned alias is kept.
        declaration['protected_files'].append(f'lfm/{"c"*64}/lfm-native-checkpoint-step-2.pt')
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError,match='protected'):
        retention.tick(path,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert not deleted


@pytest.mark.parametrize('custody',['recovery','pending'])
def test_owned_custody_also_protects_older_canonical_aliases(tmp_path,custody):
    _,files,deleted,api,path,policy,_=owned_fixture(tmp_path)
    digest='e'*64
    if custody=='pending':
        digest='f'*64;pending=tmp_path/'owned-pending.json'
        pending.write_text(json.dumps({'sha256':digest,'bytes':50}))
        policy['models']['lfm']['owned_checkpoint_retention']['pending_exports']=[str(pending)]
    alias=f'lfm/{digest}/lfm-native-checkpoint-step-1.pt'
    file=SimpleNamespace(name=alias,size=50,md5='alias')
    def remove():deleted.append(alias);del files[alias]
    file.delete=remove;files[alias]=file;path.write_text(json.dumps(policy))
    retention.tick(path,tmp_path/'journal',apply=True,api=api,ownership_directory=tmp_path/'owners')
    assert alias in files and alias not in deleted
    proof=json.loads(next((tmp_path/'journal').glob('*-intent-*.json')).read_text())
    assert alias in proof['protected_files']


@pytest.mark.parametrize('role',['lfm','mellum'])
@pytest.mark.parametrize('damage',[None,'missing','size','digest','role','readback','step'])
def test_current_owned_recovery_retains_real_object_and_only_prunes_old_aliases(tmp_path,role,damage):
    name,state,files,deleted,api,state_path,policy=setup_export(tmp_path,role)
    actual=f'{role}/{"a"*64}/{role}-{"a"*64}.pt'
    files[actual]=SimpleNamespace(name=actual,size=40,md5='owned-current')
    state['receipt'].update(name=actual,role=role)
    if damage=='missing':del files[actual]
    if damage=='size':files[actual].size+=1
    if damage=='digest':state['receipt']['identity']['sha256']='b'*64
    if damage=='role':state['receipt']['role']='mellum'if role=='lfm'else'lfm'
    if damage=='readback':state['receipt']['independent_verification']['bytes']+=1
    if damage=='step':state['last_verified_step']=True
    state_path.write_text(json.dumps(state))
    invoke=lambda:retention.tick(policy,tmp_path/'journal',apply=True,api=api,
                                ownership_directory=tmp_path/'owners')
    if damage:
        with pytest.raises(ValueError):invoke()
        assert not deleted
    else:
        assert invoke()[0]['candidates']==1
        assert deleted==[name(1)]and actual in files and name(4)in files
        intent=json.loads(next((tmp_path/'journal').glob('*-intent-*.json')).read_text())
        assert intent['verified_latest']==actual


@pytest.mark.parametrize('relative_step',[1,3,4])
@pytest.mark.parametrize('damage',[None,'missing','size','digest','role','readback','step','run'])
def test_owned_inference_parent_uses_actual_private_object_at_every_frontier(tmp_path,relative_step,damage):
    name,files,deleted,api,policy,view,receipts,parent=with_inference_views(tmp_path)
    digest='e'*64;actual=f'mellum/{digest}/mellum-{digest}.pt'
    full={'private':True,'run':'owner/project/run','role':'mellum','name':actual,
          'identity':{'sha256':digest,'bytes':40},
          'independent_verification':{'sha256':digest,'bytes':40}}
    files[actual]=SimpleNamespace(name=actual,size=40,md5='owned-full')
    if relative_step==3:
        current=json.loads((tmp_path/'state.json').read_text())
        current['receipt']=json.loads(json.dumps(full))
        (tmp_path/'state.json').write_text(json.dumps(current))
    source={**parent(relative_step),'training_checkpoint_sha256':digest}
    v={'last_verified_step':relative_step,'receipt':receipts[relative_step],
       'inference_source':source,'derivation':{'full_private_receipt':full}}
    if damage=='missing':del files[actual]
    if damage=='size':files[actual].size+=1
    if damage=='digest':source['training_checkpoint_sha256']='f'*64
    if damage=='role':full['role']='lfm'
    if damage=='readback':full['independent_verification']['sha256']='f'*64
    if damage=='step':source['step']=relative_step+1
    if damage=='run':full['run']='other/project/run'
    view.write_text(json.dumps(v))
    invoke=lambda:retention.tick(policy,tmp_path/'journal',apply=True,api=api,
                                ownership_directory=tmp_path/'owners')
    if damage:
        with pytest.raises(ValueError):invoke()
        assert not deleted
    else:
        assert invoke()[0]['verified_latest_step']==3
        assert actual in files and receipts[relative_step]['name']in files
        # No invented native spelling is needed to establish object presence.
        assert f'mellum/{digest}/mellum-native-checkpoint-step-{relative_step}.pt'not in files
        if relative_step>1:assert receipts[1]['name']in deleted


@pytest.mark.parametrize('damage',[None,'role','step','identity','readback'])
def test_pending_owned_full_parent_remains_protected_without_step_filename(tmp_path,damage):
    _,files,deleted,api,policy,_,receipts,_=with_inference_views(tmp_path)
    digest='e'*64;actual=f'mellum/{digest}/mellum-{digest}.pt'
    receipt={'private':True,'run':'owner/project/run','role':'mellum','name':actual,
             'identity':{'sha256':digest,'bytes':40},
             'independent_verification':{'sha256':digest,'bytes':40}}
    pending={'step':1,'identity':receipts[1]['identity'],'full_private_receipt':receipt,
             'full_training_checkpoint':{'step':1,'sha256':digest,'bytes':40}}
    if damage=='role':receipt['role']='lfm'
    if damage=='step':pending['step']=2
    if damage=='identity':pending['full_training_checkpoint']['sha256']='f'*64
    if damage=='readback':receipt['independent_verification']['bytes']+=1
    path=tmp_path/'pending-owned.json';path.write_text(json.dumps(pending))
    declaration=json.loads(policy.read_text());declaration['models']['mellum']['inference_pending']=[str(path)]
    policy.write_text(json.dumps(declaration))
    invoke=lambda:retention.tick(policy,tmp_path/'journal',apply=True,api=api,
                                ownership_directory=tmp_path/'owners')
    if damage:
        with pytest.raises(ValueError):invoke()
        assert not deleted
    else:
        invoke();assert receipts[1]['name']not in deleted
        intent=json.loads(next((tmp_path/'journal').glob('*-intent-*.json')).read_text())
        assert actual in intent['protected_files']
