import copy
import json
from pathlib import Path
import pickle

import pytest

from scripts.neuralese_data import continue_native_source_indexes as c
from natlang_neuralese.common import row_index
from natlang_neuralese.train import native_source_classification as m


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    old, new = tmp_path/'old', tmp_path/'new'
    fixture=Path(__file__).parent/'fixtures/reviewed-ce-index-packages-20261020.json'
    reviewed=json.loads(fixture.read_text())
    after=reviewed['new_native_sources']
    before=reviewed['old_files']['data/native_sources.py']
    assert before != after
    for root, source in ((old, before), (new, after)):
        for name in c.POLICY_FILES:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(source.encode() if name=='data/native_sources.py' else reviewed['old_files'][name].encode())
        p=root/'train/native_source_classification.py';p.parent.mkdir(exist_ok=True);p.write_text(reviewed['classification_producer'])
    source=tmp_path/'source.jsonl';source.write_text('NOT JSON: helper must not parse graph content\n')
    def bind(p):return {'path':str(p),'sha256':c.sha256_file_hex(p),'bytes':p.stat().st_size}
    q=c.prove_ce_objective_extension(old,new)
    key={'source_sha256':bind(source)['sha256'],'tokenizer':'tokenizer', 'split':'train',
         'policy_sha256':q['old_policy_sha256'],'kind':'native-graphs','pair_model':None,'role_ids':[1,2,3,4]}
    entries=[{'id':'ordinary','binding':'a'*64,'groups':['g1'],'forward_tokens':9,'supervised_tokens':8,
              'communication':False,'producer_consumer':False,'external_state_consumption':False},
             {'id':'genuine','binding':'b'*64,'groups':['g2'],'forward_tokens':12,'supervised_tokens':10,
              'communication':True,'producer_consumer':True,'external_state_consumption':False},
             {'id':'external','binding':'c'*64,'groups':['g3'],'forward_tokens':3,'supervised_tokens':2,
              'communication':True,'producer_consumer':False,'external_state_consumption':True}]
    value={'schema':'natlang.row-index/1','key':key,'entries':entries,'metadata_sha256':c.digest(entries)}
    index=tmp_path/'prior.pkl'
    def write():
        with index.open('wb')as f:pickle.dump(value,f)
        return bind(index)
    return old,new,source,bind,entries,value,write,index


def run(case):
    old,new,source,bind,entries,value,write,index=case
    return c.continue_classified_objective_index(bind(source),write(),old_package=old,new_package=new)


def test_preserves_all_classes_and_real_normal_verifier(case, monkeypatch, tmp_path):
    part=run(case);old,new,source,bind,entries,value,write,index=case
    result=row_index.load(Path(part['new_index']['path']))
    assert result['entries']==entries and result['metadata_sha256']==value['metadata_sha256']
    assert {k:v for k,v in result['key'].items()if k!='policy_sha256'}=={k:v for k,v in value['key'].items()if k!='policy_sha256'}
    assert result['key']['policy_sha256']!=value['key']['policy_sha256']
    monkeypatch.setattr(m,'__file__',str(new/'train/native_source_classification.py'))
    monkeypatch.setattr(m,'_jsonl',lambda *_:pytest.fail('raw graph parse'))
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'schema':m.SCHEMA,'train_sha256':bind(source)['sha256'],'indexes':[part]}))
    class Source:
        def receipt(self):return bind(source)
    current=Source();current.entries=entries
    assert m.migration_classes(proof,current)=={'external':{'binding':'c'*64,'communication':True,'producer_consumer':False,'external_state_consumption':True}}
    assert run(case)==part
    assert pickle.loads(index.read_bytes())==value


@pytest.mark.parametrize('mutation',['parser','setter','yield','sixth','classifier'])
def test_refuses_nonobjective_package_changes(case,mutation):
    old,new,*_=case
    p=new/('common/jsonio.py'if mutation=='sixth'else 'train/native_source_classification.py'if mutation=='classifier'else 'data/native_sources.py')
    s=p.read_text()
    if mutation=='parser':s+='\n# unreviewed parser edit\n'
    elif mutation=='setter':s=s.replace('self.ce_policy = policy','self.ce_policy = None')
    elif mutation=='yield':s=s.replace('source_sha256=self.file_sha256','source_sha256="wrong"')
    else:s+='changed\n'
    p.write_text(s)
    with pytest.raises(ValueError):run(case)


@pytest.mark.parametrize('mutation',['source','policy','kind','pair','digest','class','duplicate'])
def test_refuses_corrupt_witness_or_classification(case,mutation):
    old,new,source,bind,entries,value,write,index=case
    if mutation=='source':value['key']['source_sha256']='0'*64
    elif mutation=='policy':value['key']['policy_sha256']='0'*64
    elif mutation=='kind':value['key']['kind']='text'
    elif mutation=='pair':value['key']['pair_model']='wrong'
    elif mutation=='digest':value['metadata_sha256']='0'*64
    elif mutation=='class':entries[1]['communication']=False;value['metadata_sha256']=c.digest(entries)
    else:entries.append(copy.deepcopy(entries[0]));value['metadata_sha256']=c.digest(entries)
    with pytest.raises(ValueError):run(case)


def test_witness_hash_and_existing_destination_conflict(case):
    old,new,source,bind,entries,value,write,index=case
    ref=write();ref['sha256']='0'*64
    with pytest.raises(ValueError):c.continue_classified_objective_index(bind(source),ref,old_package=old,new_package=new)
    part=run(case);target=Path(part['new_index']['path']);prior=target.read_bytes();v=row_index.load(target);v['entries']=[];row_index.save(target,v)
    with pytest.raises(ValueError):run(case)
    assert target.read_bytes()!=prior


def test_semantic_pair_and_held_keep_exact_model_split(case):
    old,new,source,bind,entries,value,write,index=case
    value['key'].update(kind='semantic-pairs',pair_model='mellum21',split='held')
    part=run(case);result=row_index.load(Path(part['new_index']['path']))
    assert result['key']['pair_model']=='mellum21'and result['key']['split']=='held'
    assert result['entries']==entries
