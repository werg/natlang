import hashlib
import json
from types import SimpleNamespace
import pytest
from scripts.neuralese_data import prepare_combined_acquired_delta as tool


def write(path,value):
    path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}


def test_combined_manifest_reuses_only_all_accepted_disjoint_original_files(tmp_path,monkeypatch):
    monkeypatch.setattr(tool.ProtectedWorlds,'load',lambda:SimpleNamespace(digest='protected'))
    monkeypatch.setattr(tool,'current_protection',lambda path:{'proof':{'sha256':'quality'}})
    monkeypatch.setattr(tool,'load_decision_holds',lambda paths:(set(),[]))
    predecessor=write(tmp_path/'pred.json',{'source':{'sha256':'source'}})
    protection=tmp_path/'protection.json';protection.write_text('{}')
    screens=[];proofs=[]
    for i in range(2):
        leaf=tmp_path/f'leaf{i}.zst';leaf.write_bytes(b'original'+bytes([i]));leaf_ref={'path':str(leaf),'sha256':hashlib.sha256(leaf.read_bytes()).hexdigest(),'bytes':leaf.stat().st_size,'rows':1}
        builder=tmp_path/f'builder{i}.py';builder.write_text('screen code');builder_ref={'path':str(builder),'sha256':hashlib.sha256(builder.read_bytes()).hexdigest()}
        decisions=write(tmp_path/f'decisions{i}.json',[])
        screen=write(tmp_path/f'screen{i}.json',{'schema':'natlang.new-acquired-source-screen/1','training_admission':False,
            'prepared_source_sha256':'source','protected_worlds_sha256':'protected','quality_protection':{'sha256':'quality'},'decision_holds':[],
            'active':{'rows':1},'pending':{'rows':0},'counts':{'raw':1,'active':1},'decisions':decisions,'output_inventory':[leaf_ref],
            'tokenizer_fingerprint':'fp','builder':builder_ref})
        proof=write(tmp_path/f'proof{i}.json',{'schema':'natlang.acquired-source-occurrence-overlap/1','training_admission':False,
            'screen':screen,'predecessor':predecessor,'active_rows':1,'source_occurrence_suffix_missing':0,
            'duplicate_source_occurrences_within_active':0,'predecessor_source_occurrence_overlaps':[],
            'reserved_source_occurrence_overlaps':{},'reserved_screens':screens[:]})
        screens.append(screen);proofs.append(proof)
    ledger=tmp_path/'pending.jsonl';ledger.write_text(json.dumps({'original_omission':{'id':'held-a'}})+'\n')
    ledger_ref={'path':str(ledger),'sha256':hashlib.sha256(ledger.read_bytes()).hexdigest()}
    pending=write(tmp_path/'pending.json',{'schema':'natlang.acquired-unresolved-render-obligations/1',
        'training_admission':False,'rows':1,'ledger':ledger_ref})
    post=write(tmp_path/'post.json',{'schema':'natlang.acquired-repair-still-pending/1','training_admission':False,
        'original_omission':{'id':'held-b'},'derived_omission':{'id':'held-b'}})
    result=tool.build(predecessor,screens,proofs,protection,tmp_path/'out',pending_refs=[pending,post])
    manifest=json.loads((tmp_path/'out'/'train-sources.json').read_text())
    assert result['rows']==2 and result['training_admission'] is False
    assert result['pending_render_obligations']['rows']==2
    assert [x['path'] for x in manifest['sources']]==[str(tmp_path/'leaf0.zst'),str(tmp_path/'leaf1.zst')]
    bad=json.loads((tmp_path/'proof1.json').read_text());bad['reserved_screens']=[]
    proofs[1]=write(tmp_path/'proof1.json',bad)
    with pytest.raises(ValueError,match='preceding'):tool.build(predecessor,screens,proofs,protection,tmp_path/'out2')
