import copy
import pytest
from test_native_source_preparation import row, manifest
from natlang_neuralese.train.native_resident import binding
from natlang_neuralese.train.native_source_preparation import NativeSourcePreparation
from natlang_neuralese.train.native_capacity_cases import resolve_capacity_case_requests


def setup_source(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    a=row(tmp_path/'a.jsonl','a'); b=row(tmp_path/'b.jsonl','b')
    root=manifest(tmp_path/'root.json',[b,a])
    train=NativeSourcePreparation('tok',()).load(root,split='train')
    case=lambda index: {k:train.entries[index][k] for k in ('id','binding')}
    request=dict(schema='natlang.native-memory-qualification-request/1',source=root,
                 cases=[dict(index=0,**case(1))],reference=dict(index=1,**case(0)),
                 indexes=[{'untouched':'complete original authority'}])
    return train,request,a


def test_actual_manifest_order_replaces_only_explicit_indices(tmp_path,monkeypatch):
    train,request,_=setup_source(tmp_path,monkeypatch); before=copy.deepcopy(request)
    with pytest.raises(ValueError,match='index differs'):
        resolve_capacity_case_requests([request],train)
    result=resolve_capacity_case_requests([request],train,relocate=True)
    assert result['changed_indices']==2 and result['case_checks']==result['reference_checks']==1
    output=result['requests'][0]
    assert output['cases'][0]['index']==1 and output['reference']['index']==0
    output['cases'][0]['index']=0;output['reference']['index']=1
    assert output==before==request
    corrected=resolve_capacity_case_requests([request],train,relocate=True)['requests']
    assert resolve_capacity_case_requests(corrected,train)['changed_indices']==0


@pytest.mark.parametrize('kind',['cases','reference'])
@pytest.mark.parametrize('field',['id','binding'])
def test_missing_or_changed_identity_is_never_relabelled(tmp_path,monkeypatch,kind,field):
    train,request,_=setup_source(tmp_path,monkeypatch)
    item=request[kind][0] if kind=='cases' else request[kind]
    item[field]='unseen'
    with pytest.raises(ValueError,match='absent'):
        resolve_capacity_case_requests([request],train,relocate=True)


def test_pending_leaf_cases_use_pending_order_but_reference_uses_active_train(tmp_path,monkeypatch):
    train,request,a=setup_source(tmp_path,monkeypatch)
    request['source']=a; request['cases'][0]['index']=0
    result=resolve_capacity_case_requests([request],train,relocate=True)
    assert result['requests'][0]['cases'][0]['index']==0
    assert result['requests'][0]['reference']['index']==0
    assert result['source_groups']==2


def test_geometry_bytes_and_non_cache_pool_are_rejected(tmp_path,monkeypatch):
    train,request,_=setup_source(tmp_path,monkeypatch)
    request['cases'][0]['forward_tokens']=train.entries[1]['forward_tokens']+1
    with pytest.raises(ValueError,match='geometry'):
        resolve_capacity_case_requests([request],train,relocate=True)
    request['cases'][0].pop('forward_tokens');request['source']['bytes']=1
    with pytest.raises(ValueError,match='byte binding'):
        resolve_capacity_case_requests([request],train,relocate=True)
    request['source'].pop('bytes')
    with pytest.raises(ValueError,match='cache-only'):
        resolve_capacity_case_requests([request],train,preparation=NativeSourcePreparation('tok',()))


def test_cache_only_pending_load_cannot_parse_a_new_leaf(tmp_path,monkeypatch):
    train,request,_=setup_source(tmp_path,monkeypatch)
    request['source']=row(tmp_path/'cold.jsonl','cold')
    with pytest.raises(ValueError,match='exact completed source index'):
        resolve_capacity_case_requests([request],train,relocate=True)


def test_duplicate_selected_indices_are_rejected_before_model_restore(tmp_path,monkeypatch):
    train,request,_=setup_source(tmp_path,monkeypatch)
    request['cases'].append(copy.deepcopy(request['cases'][0]))
    with pytest.raises(ValueError,match='repeats a selected case index'):
        resolve_capacity_case_requests([request],train,relocate=True)
