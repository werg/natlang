import hashlib
import json
from pathlib import Path

from natlang_neuralese.train.recipe import _data_continuation


def test_standalone_rebind_requires_exact_retirement_and_unchanged_custody(tmp_path):
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    evidence=tmp_path/'evidence.json';evidence.write_text('{}')
    proofs=[]
    for i in range(2):
        path=tmp_path/f'withdrawal-{i}.json';path.write_text(json.dumps({'schema':'natlang.native-graph-withdrawal/1','evidence':{'path':str(evidence),'sha256':sha(evidence)},'id':i}));proofs.append({'path':str(path),'sha256':sha(path),'binding':f'proof{i}'})
    manifest=tmp_path/'proofs.json';manifest.write_text(json.dumps({'schema':'natlang.native-retired-withdrawal-set/1','proofs':[{k:v for k,v in item.items() if k!='binding'} for item in proofs]}))
    role=lambda path,digest,binding:{'path':path,'sha256':digest,'binding':binding}
    train=role('/train','t','train');held=role('/held','h','held')
    old={'native_graphs':train,'foundation_native_graphs':train,'held_native_graphs':held,'retired_native_withdrawal':proofs[0],'withdrawn_native_graphs':proofs[1],'standalone_diloco_parent':role('/old-pin','a','parent')}
    new={'native_graphs':train,'foundation_native_graphs':train,'held_native_graphs':held,'continued_native_graphs':train,'retired_native_withdrawal_set':role(str(manifest),sha(manifest),'retired'),'standalone_diloco_parent':role('/new-pin','b','parent')}
    assert _data_continuation(old,new)
    for field in ('standalone_diloco_parent','retired_native_withdrawal_set'):
        altered=dict(new);altered.pop(field);assert not _data_continuation(old,altered)
    altered=dict(new);altered['held_native_graphs']=role('/other','x','held');assert not _data_continuation(old,altered)
    manifest.write_text(json.dumps({'schema':'natlang.native-retired-withdrawal-set/1','proofs':[{k:v for k,v in proofs[1].items() if k!='binding'}]}));assert not _data_continuation(old,new)


def test_standalone_rebind_with_new_correction_retains_prior_proof():
    role=lambda path,digest,binding:{'path':path,'sha256':digest,'binding':binding}
    oldtrain=role('/old','old','train');newtrain=role('/new','new','train');held=role('/held','h','held');proof=role('/prior','p','prior')
    old={'native_graphs':oldtrain,'foundation_native_graphs':oldtrain,'held_native_graphs':held,'retired_native_withdrawal':proof,'standalone_diloco_parent':role('/parent1','a','parent')}
    new={'native_graphs':newtrain,'foundation_native_graphs':newtrain,'continued_native_graphs':oldtrain,'held_native_graphs':held,'retired_native_withdrawal':proof,'withdrawn_native_graphs':role('/correction','c','correction'),'standalone_diloco_parent':role('/parent2','b','parent')}
    assert _data_continuation(old,new)
    altered=dict(new);altered['retired_native_withdrawal']=role('/other','x','prior');assert not _data_continuation(old,altered)
    altered=dict(new);altered['standalone_diloco_parent']=role('/parent1','b','parent');assert not _data_continuation(old,altered)
