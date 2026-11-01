import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from scripts.neuralese_data import audit_saved_native_positions as tool
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_packed_capacity import _rows_digest


def test_position_audit_binds_saved_exposure_and_checks_actual_plan(tmp_path,monkeypatch):
    import natlang_neuralese.rl.native_direct_sft_recovery as recovery
    def write(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return {'path':str(p),'sha256':sha256_file_hex(p),'bytes':p.stat().st_size}
    manifest=write('manifest.json',{});entries=[{'id':'row','binding':'binding','forward_tokens':12,'communication':False}]
    geometry=write('geometry.json',{'identity':{'tokenizer':'f'*64,'roles':[1,2,3,4]},'source':manifest,
        'rows_sha256':_rows_digest(entries),'kv_slots':[12]})
    source=write('source.json',{'source':manifest,'geometry':geometry})
    checkpoint={'sha256':'a'*64};snapshot=write('snapshot.json',{'schema':'natlang.rolling-sft-parent-custody-snapshot/1',
        'checkpoint':checkpoint,'optimizer_present':True,'rng_present':True,'sft_frames':{'stage':{'source_receipts':[manifest],
        'source_row_bindings':{'row':'binding'},'exposure_counts':{'row':1},'required_visits':{'row':2}}}})
    publication=write('publication.json',{})
    monkeypatch.setattr(recovery,'_authenticated_refresh_publication',lambda _:{'publication':{'checkpoint':checkpoint,
        'rolling_endpoint_custody_proof':{'successor':snapshot}}})
    node=SimpleNamespace(token_ids=list(range(12)),imports=(),producing=(),logical_producing=())
    example=SimpleNamespace(id='row',plan=SimpleNamespace(invocations=[node],slot_positions=list(range(12))),external_states={})
    fake=SimpleNamespace(entries=entries,file_sha256=manifest['sha256'],iter_indices=lambda indices:iter([example]))
    monkeypatch.setattr(tool,'NativeGraphSource',lambda *a,**kw:fake)
    result=tool.audit(source,publication,tmp_path/'audit.json',stage='stage',limit=10)
    assert result['counts']['ordinary_over_limit']==1 and result['counts']['exposed_ordinary_over_limit']==1
    assert result['rows'][0]['max_position']==11 and result['rows'][0]['saved_visits']==1
    with pytest.raises(ValueError,match='fresh'):tool.audit(source,publication,tmp_path/'audit.json',stage='stage',limit=10)
