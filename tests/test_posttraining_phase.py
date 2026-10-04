"""Post-training phase admission and checkpoint identity checks."""
import json
from pathlib import Path
import pytest
from scripts.corpus import digest, split_programs, index_pairs
from scripts.posttraining_phase import prepare_phase, verify_phase


def test_completed_exclusion_lineage_can_start_positive_lr_phase(tmp_path):
    base=tmp_path/'base.jsonl';data=tmp_path/'phase.jsonl';parent=tmp_path/'checkpoint';parent.mkdir()
    rows=[{'id':'h','program_id':'held','split':'test','prompt':'x','completion':'a'},
          {'id':'t','program_id':'train','split':'train','prompt':'x','completion':'b'}]
    base.write_text(''.join(json.dumps(r)+'\n' for r in rows));data.write_bytes(base.read_bytes())
    _,_,split=split_programs(index_pairs(base),1,42)
    sp=tmp_path/'split.json';sp.write_text(json.dumps(split))
    state={'cursor':1,'trained_examples':1,'step':1,'corpus':{'target_examples':1,'steps':1,'exclusion_manifest_sha256':'old'},
           'args':{'seed':'42','holdout':'1','accum':'1','data':str(base)}}
    (parent/'state.json').write_text(json.dumps(state));(parent/'optimizer.pt').write_text('momentum');(parent/'rng.pt').write_text('rng')
    from scripts.corpus import file_digest
    gp=tmp_path/'gates.json';gp.write_text(json.dumps({'schema':'natlang.posttraining_data_gates/1','data_sha256':file_digest(data),'native_admission':True,'source_closure':True}))
    manifest=tmp_path/'phase.json';m=prepare_phase(data=data,parent=parent,split_path=sp,gate_path=gp,output=manifest)
    m['status']='approved';manifest.write_text(json.dumps(m))
    actual,first=verify_phase(manifest,data,parent,seed=42,holdout=1,lr=2e-5,accum=1)
    assert first and actual['target_examples']==2 and actual['end_step']==2
    assert (parent/'optimizer.pt').read_text()=='momentum'
    with pytest.raises(ValueError,match='controls'):verify_phase(manifest,data,parent,seed=42,holdout=1,lr=0,accum=1)
    (parent/'optimizer.pt').write_text('changed')
    with pytest.raises(ValueError,match='exact copied'):verify_phase(manifest,data,parent,seed=42,holdout=1,lr=2e-5,accum=1)
