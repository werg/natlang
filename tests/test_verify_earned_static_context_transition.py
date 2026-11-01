import json
from pathlib import Path
import pytest
from scripts.neuralese_data import verify_earned_static_context_transition as tool
from scripts.neuralese_data.partition_acquired_context_pending import bind


def test_verification_derives_endpoint_hashes_from_snapshotter(tmp_path,monkeypatch):
    stage='stage';ledger=tmp_path/'ledger.jsonl';ledger.write_text(json.dumps({'id':'x','disposition':'corrected','old_binding':'a','new_binding':'b'})+'\n')
    proof_path=tmp_path/'proof.json';proof_path.write_text(json.dumps({'stage_id':stage,'static_native_context_correction':{'schema':'fixture'},
        'prior_source':{'sha256':'old-source'},'current_source':{'sha256':'new-source'},'row_bindings_path':str(ledger)}))
    parent=tmp_path/'parent.pt';parent.write_bytes(b'parent');child=tmp_path/'child.pt';child.write_bytes(b'child')
    parent_ref={**bind(parent),'global_step':1};child_ref={**bind(child),'global_step':2};proof_ref=bind(proof_path)
    before={'source_row_bindings':{'x':'a'},'required_visits':{'x':1},'exposure_counts':{'x':0}}
    after={'source_row_bindings':{'x':'b'},'required_visits':{'x':7},'exposure_counts':{'x':1}}
    called=[]
    def snapshot(path,sha,out):
        called.append((path,sha));is_parent=path==str(parent)
        target=Path(out)/('before.json' if is_parent else 'after.json')
        target.write_text(json.dumps({'global_step':1 if is_parent else 2,'sft_frames':{stage:before if is_parent else after}}))
        return bind(target)
    monkeypatch.setattr(tool,'_rolling_parent_custody_snapshot',snapshot)
    monkeypatch.setattr(tool,'source_correction_dependencies',lambda ref:[])
    monkeypatch.setattr(tool,'_rolling_endpoint_snapshots',lambda decl,*args:{'verified_correction':decl['source_correction']})
    result=tool.verify(parent_ref,child_ref,proof_ref,tmp_path/'out')
    assert len(called)==2 and result['endpoint_custody']['verified_correction']==proof_ref
    value=json.loads(Path(result['verification']['path']).read_text())
    assert value['changed_bindings']==[{'id':'x','binding_before':'a','binding_after':'b'}]
    assert value['old_quotas_preserved'] is True and value['old_exposure_preserved'] is True
    with pytest.raises(ValueError,match='fresh'):tool.verify(parent_ref,child_ref,proof_ref,tmp_path/'out')
