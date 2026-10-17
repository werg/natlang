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
