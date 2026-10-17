"""Sealed collection input bytes survive durable snapshot key normalization."""
import json
from pathlib import Path
import pytest
from natlang_neuralese.rl import production
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable


def setup(tmp_path,monkeypatch):
    cases=[{'split':'train','id':'first','nested':{'z':1,'a':'text'}},
           {'split':'train','id':'second','nested':{'q':2,'b':'other'}}]
    source=tmp_path/'source.jsonl';source.write_text(''.join(json.dumps(row)+'\n' for row in cases))
    parent=tmp_path/'parent';parent.write_bytes(b'exact actor checkpoint')
    config=tmp_path/'config.json';out=tmp_path/'production'
    write_json_durable(config,{'out':str(out),'cases':str(source),'checkpoint':str(parent),
        'after_update':['unused'],'endpoints':['unused'],'neuralese_server':False})
    monkeypatch.setattr(production,'select_tasks',lambda *args,**kw:[{'row':i,'samples':4}for i in range(2)])
    calls=[]
    def collect(config,values,**kw):
        path=kw['cases'];calls.append(path.read_bytes());binding=kw['out']/'collection-binding.json'
        checksum=sha256_file_hex(path)
        if binding.exists():assert json.loads(binding.read_text())['cases_sha256']==checksum
        else:write_json_durable(binding,{'cases_sha256':checksum})
        raise InterruptedError('durable pending collection; no completion')
    monkeypatch.setattr(production,'collect_with_actor',collect)
    return config,out,cases,calls


def test_exact_selected_bytes_survive_interrupted_collection_resume(tmp_path,monkeypatch):
    config,out,cases,calls=setup(tmp_path,monkeypatch)
    first=production.run(config,rounds=1)
    assert first['phase']=='requested_stop'
    selected=out/'round-000000/selected-cases.jsonl';before=selected.read_bytes()
    snapshot=json.loads((out/'round-000000-selection.json').read_text())
    assert snapshot['cases']==cases
    assert before!=(''.join(json.dumps(row)+'\n'for row in snapshot['cases'])).encode()
    second=production.run(config,rounds=1)
    assert second['phase']=='requested_stop' and second['round']==0
    assert selected.read_bytes()==before and calls==[before,before]
    assert second['trained_rounds']==0 and not second['trained_task_ids']


@pytest.mark.parametrize('change',['objects','order','numeric_type'])
def test_existing_selected_objects_order_and_numeric_types_fail_closed(tmp_path,monkeypatch,change):
    config,out,cases,calls=setup(tmp_path,monkeypatch)
    production.run(config,rounds=1)
    selected=out/'round-000000/selected-cases.jsonl'
    if change=='objects':cases[0]['nested']['z']=99
    elif change=='order':cases.reverse()
    else:cases[0]['nested']['z']=1.0
    selected.write_text(''.join(json.dumps(row)+'\n'for row in cases));altered=selected.read_bytes()
    with pytest.raises(ValueError,match='selected case objects or order changed'):
        production.run(config,rounds=1)
    assert len(calls)==1 and selected.read_bytes()==altered
