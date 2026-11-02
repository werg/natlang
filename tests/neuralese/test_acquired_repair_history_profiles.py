import hashlib
import json

import pytest
from scripts.neuralese_data.bind_acquired_repair_history_profiles import build


def write(path,value):
    path.write_text(json.dumps(value));return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize('profile',[None,'keep'])
def test_repair_export_inherits_exact_original_history_profile(tmp_path,profile):
    source={'path':'/original/source.zst','sha256':'a'*64}
    converter=write(tmp_path/'converter.json',{'source':source,'explicit_history_reasoning_policy':profile})
    audit=write(tmp_path/'audit.json',{'source':source,'converter_receipt':converter})
    census=write(tmp_path/'census.json',{'schema':'natlang.acquired-render-pending-census/1','receipts':[audit|{'source':source}]})
    view=write(tmp_path/'export.json',{'contract':{'source_syntax_repair':{'parent_source':source}}})
    batch=write(tmp_path/'batch.json',{'schema':'natlang.acquired-syntax-repair-batch/1','training_admission':False,
                                     'census':census,'views':[view]})
    result=build(batch,census,tmp_path/'out');exports=json.loads((tmp_path/'out'/'exports.json').read_text())
    assert exports['history_reasoning_policy']==({str(tmp_path/'export.json'):'keep'} if profile else {})
    assert result['counts']=={profile or 'default':1}


def test_repair_history_binding_rejects_changed_original_source(tmp_path):
    source={'path':'/original/source.zst','sha256':'a'*64}
    converter=write(tmp_path/'converter.json',{'source':{**source,'sha256':'b'*64},'explicit_history_reasoning_policy':'keep'})
    audit=write(tmp_path/'audit.json',{'source':source,'converter_receipt':converter})
    census=write(tmp_path/'census.json',{'schema':'natlang.acquired-render-pending-census/1','receipts':[audit|{'source':source}]})
    batch=write(tmp_path/'batch.json',{'schema':'natlang.acquired-syntax-repair-batch/1','training_admission':False,'census':census,'views':[]})
    with pytest.raises(ValueError,match='profile'):
        build(batch,census,tmp_path/'out')
