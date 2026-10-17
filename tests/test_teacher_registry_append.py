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
