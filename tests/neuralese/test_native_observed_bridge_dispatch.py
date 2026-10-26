"""Live bridge observations preserve actual OS ancestry without fabricated startup files."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from natlang_neuralese.rl import native_owned_bridge_dispatch as bridge, native_raw_handoff as handoff, native_runtime_handoff as runtime
from natlang_neuralese.rl.native_bridge import REQUEST_SCHEMA
from natlang_neuralese.train.native_resident import binding, code_inventory, digest
from natlang_neuralese.common.jsonio import write_json_durable as write


@pytest.fixture
def live(tmp_path):
    fake=tmp_path/'fake'; module=fake/'natlang_neuralese/train'; module.mkdir(parents=True)
    (module.parent/'__init__.py').write_text(''); (module/'__init__.py').write_text('')
    (module/'native_state_sft.py').write_text('import sys;sys.stdin.buffer.read()\n')
    resident=tmp_path/'resident'; resident.mkdir()
    write(resident/'current.json',{'payload':{'plan':{'code':code_inventory(module.parent)}}})
    parent=tmp_path/'parent.pt'; parent.write_bytes(b'full-state-protocol-fixture')
    policy=tmp_path/'recipe.json'; write(policy,{'id':'fixture'})
    rb=binding(policy);out=tmp_path/'operation';out.mkdir()
    config=tmp_path/'config.json';write(config,{'trainer_parameters':{'stage_id':'sft','recipe_binding':rb}})
    request=tmp_path/'request.json';value={'schema':REQUEST_SCHEMA,'kind':'round','native_actor_mode':'raw-state',
        'config':binding(config),'parent_checkpoint':binding(parent),'out':str(out)}
    value['request_id']=digest(value);write(request,value)
    argv=[sys.executable,'-m','natlang_neuralese.train.native_state_sft','--out',str(out/'learner'),
          '--continue-from',str(parent),'--stage-id','sft','--recipe-binding',json.dumps(rb),
          '--resident-data-directory',str(resident)]
    entry=tmp_path/'supervisor.py';entry.write_text('import sys,json,subprocess;'
        'p=subprocess.Popen(json.loads(sys.argv[1]),stdin=subprocess.PIPE);'
        'print(p.pid,flush=True);sys.stdin.buffer.read();p.communicate()\n')
    child_repo=tmp_path/'actual-child-repo';child_repo.mkdir()
    expected_env={'PYTHONPATH':str(fake),'NATLANG_REPO':str(child_repo)}
    p=subprocess.Popen([sys.executable,str(entry),json.dumps(argv)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                       text=True,env={**os.environ,**expected_env})
    try:
        child=int(p.stdout.readline());owner=handoff.observe(p.pid);learner=handoff.observe(child)
        launch=tmp_path/'original-launch.json';write(launch,{**owner,'source':str(entry)})
        observed=bridge.capture(owner,learner,binding(request),binding(launch),binding(entry),tmp_path/'observed',
                                historical_package=module.parent)
        intent=handoff.capture_drain_intent(observed,tmp_path/'drain-intent.json')
        yield dict(captured=json.loads(Path(intent['path']).read_text()),observed=json.loads(Path(observed['path']).read_text()),
                   intent=intent,owner=owner,child=learner,entry=entry,argv=argv,tmp=tmp_path,expected_env=expected_env)
    finally:
        p.communicate(timeout=10)


def test_actual_live_bridge_follows_current_os_request_and_import_environment(live):
    value,argv,refs=runtime._owned_launch_chain({'intent':live['intent']})
    assert argv==live['argv'] and value['startup_capture_claimed'] is False
    assert value['owner']['pid']==live['owner']['pid'] and value['child']['pid']==live['child']['pid']
    assert value['env']==live['expected_env']
    assert refs['supervisor']==binding(live['entry'])


@pytest.mark.parametrize('change',['owner','child','argv','parent','config','historical-code'])
def test_observed_bridge_rejects_changed_control_or_process_custody(live,change):
    observed=live['observed'];ref=observed['observed_bridge_dispatch'];value=json.loads(Path(ref['path']).read_text())
    if change=='owner':value['owner']['start_ticks']='0'
    elif change=='child':value['child']['pid']+=100000
    elif change=='argv':value['argv'][2]='unrelated.module'
    elif change=='parent':value['parent']['sha256']='1'*64
    elif change=='config':value['config']['sha256']='2'*64
    elif change=='historical-code':value['historical_code']={}
    path=live['tmp']/'forged-dispatch.json';write(path,value);observed['observed_bridge_dispatch']=binding(path)
    with pytest.raises((ValueError,FileNotFoundError)):
        bridge.verify(live['captured'],observed)
