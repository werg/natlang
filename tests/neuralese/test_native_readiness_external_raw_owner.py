"""External ownership suppresses duplicate RAW launch, never readiness checks."""
import json
from types import SimpleNamespace
import pytest
from natlang_neuralese.train import native_readiness as m

@pytest.mark.parametrize('value',[None,0,1,'false'])
def test_external_owner_declaration_requires_boolean(value):
    with pytest.raises(ValueError,match='explicit boolean'):
        m._manage_raw_bootstrap({'manage_raw_bootstrap':value})

def test_external_owner_requires_bound_release_plan():
    with pytest.raises(ValueError,match='bound release plan'):
        m._manage_raw_bootstrap({'manage_raw_bootstrap':False})

def test_default_still_manages_raw():
    assert m._manage_raw_bootstrap({}) is True

@pytest.mark.parametrize('manage',[False,True])
@pytest.mark.parametrize('ready_count',[0,2])
def test_controller_checks_readiness_but_only_owned_mode_launches_raw(tmp_path,monkeypatch,manage,ready_count):
    curriculum=tmp_path/'curriculum.json';curriculum.write_text('{}')
    config={'schema':'natlang.native-shared-stage-controller/1','out':str(tmp_path/'out'),
            'curriculum':str(curriculum),'command':['semantic'],
            'raw_rl_bootstrap':{'path':'release-plan','sha256':'0'*64},
            'manage_raw_bootstrap':manage}
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    stop=SimpleNamespace(requested=False);checks=[];launches=[]
    def readiness(cfg):
        checks.append(cfg['raw_rl_bootstrap']);return [{'model_id':str(i)}for i in range(ready_count)],[{'reason':'actual owner has not released'}]if not ready_count else []
    monkeypatch.setattr(m,'ready_participants',readiness)
    monkeypatch.setattr(m,'_role_local_released',lambda c:False)
    monkeypatch.setattr(m,'raw_bootstrap_command',lambda c:['real-raw'])
    def launch(argv,kind,*args,**kwargs):
        launches.append((argv,kind));raise OSError('fixture launch failure')
    monkeypatch.setattr(m,'_launch_child',launch)
    monkeypatch.setattr(m.time,'sleep',lambda seconds:setattr(stop,'requested',True))
    result=m._run(path,stop=stop)
    assert len(checks)==2 and result['ready_models']==ready_count
    assert result['manage_raw_bootstrap'] is manage
    if ready_count:
        assert len(launches)==1 and launches[0][1]=='native-semantic'
        assert json.loads((tmp_path/'out'/'participants.json').read_text())['models']==[{'model_id':'0'},{'model_id':'1'}]
    else:
        assert launches==([(['real-raw'],'native-raw-rl')] if manage else [])
        assert not (tmp_path/'out'/'participants.json').exists()
