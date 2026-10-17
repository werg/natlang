import copy,json
import pytest
from scripts.coreweave.teacher_reference_append import registry_append
from scripts.coreweave.teacher_generation_guardian import binding


def sealed(tmp_path,name,value):
    p=tmp_path/name;p.write_text(json.dumps(value));return binding(p)


def registry():return {'version':2,'corpora':[{'id':'old','training_admission':True,'include':['turns.jsonl'],'sha256':'old-bytes'}]}


def test_registry_append_keeps_exact_existing_policy_and_never_admits(tmp_path):
    old=registry();new=copy.deepcopy(old);new['corpora'].append({'id':'new','training_admission':False})
    a,b=sealed(tmp_path,'old.json',old),sealed(tmp_path,'new.json',new)
    result=registry_append(a,b)
    assert result['before']==a and result['after']==b
    assert result['original_corpora']==1 and result['current_corpora']==2
    assert result['appended_corpus_ids']==['new']and result['training_admission']is False
    assert old==registry()
    assert registry_append(b,b)['appended_corpus_ids']==[]


@pytest.mark.parametrize('change',['source','admission','order','remove','duplicate','field','identity','numeric_type'])
def test_registry_append_rejects_changes_to_old_decisions_and_identities(tmp_path,change):
    old=registry();new=copy.deepcopy(old);new['corpora'].append({'id':'new'})
    if change=='source':new['corpora'][0]['sha256']='other'
    elif change=='admission':new['corpora'][0]['training_admission']=False
    elif change=='order':new['corpora'].reverse()
    elif change=='remove':new['corpora']=[]
    elif change=='duplicate':new['corpora'].append({'id':'old'})
    elif change=='field':new['version']=3
    elif change=='numeric_type':new['version']=2.0
    else:new['corpora'].append({'id':False})
    with pytest.raises(ValueError):registry_append(sealed(tmp_path,'old.json',old),sealed(tmp_path,'new.json',new))


def test_registry_append_authenticates_both_actual_files(tmp_path):
    a,b=sealed(tmp_path,'old.json',registry()),sealed(tmp_path,'new.json',registry())
    (tmp_path/'old.json').write_text('{}')
    with pytest.raises(ValueError,match='input changed'):registry_append(a,b)
    a=sealed(tmp_path,'old.json',registry());(tmp_path/'new.json').write_text('{}')
    with pytest.raises(ValueError,match='input changed'):registry_append(a,b)


from scripts.coreweave.teacher_reference_append import registry_cache_cleanup


def cleanup(tmp_path):
    digest='a'*64
    receipt=sealed(tmp_path,'retained.json',{'schema':'natlang.verified-teacher-decision-materialization/1','output_sha256':digest,'identity':{'original':True}})
    prior={'receipt_path':str(tmp_path/'deleted/receipt.json'),'receipt_sha256':receipt['sha256'],'output_path':str(tmp_path/'deleted/decisions.jsonl'),'output_sha256':digest}
    item={'output_path':prior['output_path'],'output_sha256':digest,'output_bytes':100,'receipt_sha256':receipt['sha256'],'retained_receipt':receipt['path'],'key_dir':str(tmp_path/'deleted')}
    plan=sealed(tmp_path,'plan.json',{'schema':'natlang.rebuildable-decision-cache-retention/1','items':[item]})
    declaration={'path':plan['path'],'sha256':plan['sha256'],'status':'cache-key-removed-after-output-hash-verification'}
    old={'version':2,'corpora':[{'id':'source','source_binding':{'sha256':'source-bytes','decision_admission':prior},'training_admission':True,'split':'train'}]}
    new=copy.deepcopy(old);new['corpora'][0]['source_binding']['decision_admission']={key:value for key,value in prior.items()if key!='output_path'}
    new['corpora'][0]['source_binding']['decision_admission'].update(receipt_path=receipt['path'],output_original_path=prior['output_path'],output_bytes=100,output_storage_status='deleted_rebuildable_generated_cache',cache_cleanup_plan=declaration)
    return old,new,receipt,plan


def test_cleanup_authenticates_same_receipt_and_all_other_source_fields(tmp_path):
    old,new,receipt,plan=cleanup(tmp_path)
    a,b=sealed(tmp_path,'old.json',old),sealed(tmp_path,'new.json',new)
    result=registry_cache_cleanup(a,b)
    assert result['changes']==[{'corpus_id':'source','retained_receipt':receipt,'cleanup_plan':plan}]
    assert result['training_admission']is False
    with pytest.raises(ValueError):registry_append(a,b)
    assert registry_cache_cleanup(b,b)['changes']==[]


@pytest.mark.parametrize('change',['source','admission','split','typed','output','receipt','plan','extra','duplicate','add','size','status','plan_output','plan_receipt','plan_numeric'])
def test_cleanup_cannot_authorize_source_or_policy_changes(tmp_path,change):
    old,new,receipt,plan=cleanup(tmp_path);row=new['corpora'][0];admission=row['source_binding']['decision_admission']
    if change=='source':row['source_binding']['sha256']='changed'
    elif change=='admission':row['training_admission']=False
    elif change=='split':row['split']='held'
    elif change=='typed':new['version']=2.0
    elif change=='output':admission['output_sha256']='b'*64
    elif change=='receipt':(tmp_path/'retained.json').write_text('{}')
    elif change=='plan':(tmp_path/'plan.json').write_text('{}')
    elif change=='extra':admission['cache_cleanup_plan']['allow_source_changes']=True
    elif change=='duplicate':
        body=json.loads((tmp_path/'plan.json').read_text());body['items']*=2;ref=sealed(tmp_path,'plan.json',body);admission['cache_cleanup_plan']['sha256']=ref['sha256']
    elif change=='add':new['corpora'].append({'id':'new'})
    elif change=='size':admission['output_bytes']=100.0
    elif change=='status':admission['output_storage_status']='retained'
    elif change.startswith('plan_'):
        body=json.loads((tmp_path/'plan.json').read_text())
        if change=='plan_output':body['items'][0]['output_sha256']='c'*64
        elif change=='plan_receipt':body['items'][0]['retained_receipt']=str(tmp_path/'other')
        else:body['items'][0]['output_bytes']=100.0
        ref=sealed(tmp_path,'plan.json',body);admission['cache_cleanup_plan']['sha256']=ref['sha256']
    with pytest.raises(ValueError):registry_cache_cleanup(sealed(tmp_path,'old.json',old),sealed(tmp_path,'new.json',new))
