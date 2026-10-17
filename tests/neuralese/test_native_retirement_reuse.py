"""Retire a real drained resident using its single verified full-state decode."""
import json,shutil,time
from pathlib import Path
import pytest
import torch
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.rl.native_raw_handoff import complete_drain,verify_handoff,binding
from natlang_neuralese.common.jsonio import write_json_durable
from test_native_resident import setup
from test_native_raw_handoff import actual_drain


@pytest.mark.parametrize('change',[None,'checkpoint','terminal','directory','current','initial','loaded-record','pin','missing-reference','missing-decode'])
def test_retire_reuses_real_verified_drain_without_second_tensor_load(tmp_path,monkeypatch,actual_drain,change):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'cache'))
    source=tmp_path/'source';source.mkdir()
    _,directory,manager,recipe=setup(source)
    resident.prepare(directory,recipe,'sft',tokenizer_fingerprint='tokenizer-sha',role_ids=())
    manager.poll(7);current=resident.current(directory)
    process,intent,terminal,slot,pin=actual_drain
    tree=json.loads(Path(intent['path']).read_text())
    initial=json.loads((directory/'initial.json').read_text());initial['owner']=tree['children'][0]
    write_json_durable(directory/'initial.json',initial)
    saved=torch.load(slot,weights_only=False)
    saved['training'].update(global_step=7,source_receipts=[current['payload']['train']],
        options={'resident_adoption':{'generation':current['generation'],'history':current['history']}})
    torch.save(saved,slot);shutil.copyfile(slot,pin)
    process.communicate();write_json_durable(terminal,{'at':time.time(),'returncode':process.returncode})
    handoff=complete_drain(intent,binding(terminal),binding(pin),tmp_path/'resident-handoff.json',
        original_slot=binding(slot),resident_directory=directory)
    loaded=verify_handoff(handoff);checkpoint=binding(pin);exit_ref=binding(terminal)
    if change=='checkpoint':checkpoint=binding(slot)
    elif change=='terminal':
        other=tmp_path/'other-exit.json';write_json_durable(other,{'at':time.time(),'returncode':0});exit_ref=binding(other)
    elif change=='directory':loaded[0]['resident']['directory']=str(tmp_path/'other')
    elif change=='current':
        current={**current,'at':current['at']+1};write_json_durable(directory/'current.json',current)
    elif change=='initial':
        initial={**initial,'unused':'different-owner-snapshot'};write_json_durable(directory/'initial.json',initial)
    elif change=='loaded-record':loaded[0]['at']+=1
    elif change=='pin':pin.write_bytes(b'changed immutable checkpoint')
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('duplicate full tensor decode'))
    kwargs={'handoff':handoff,'verified_handoff':loaded}
    if change=='missing-reference':kwargs.pop('handoff')
    if change=='missing-decode':kwargs.pop('verified_handoff')
    if change:
        with pytest.raises(ValueError):resident.retire(directory,checkpoint,exit_ref,**kwargs)
        assert resident.current(directory)==current
        assert not(directory/'retired'/f"{current['generation']}.json").exists()
    else:
        resident.retire(directory,checkpoint,exit_ref,**kwargs)
        assert resident.current(directory)is None
        result=json.loads((directory/'retired'/f"{current['generation']}.json").read_text())
        assert result=={**current,'checkpoint':checkpoint,'terminal':exit_ref}
