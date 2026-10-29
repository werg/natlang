import json
import pytest
from scripts.coreweave.retention_transaction import retire_pair


def test_pair_recovers_after_control_delete_without_duplicate_credit(tmp_path):
    objects={'full':'a'*64,'control':'b'*64};candidate={'role':'full','sha256':'a'*64,'control_role':'control','control_sha256':'b'*64};removed=[];fail=[True];phases=[]
    def inspect(c,phase):phases.append(phase)
    def present(role,sha):return objects.get(role)==sha
    def remove(role,sha):
        if role=='full' and fail[0]:raise RuntimeError('interrupted')
        assert objects.pop(role)==sha;removed.append(role)
    def write(path,state):path.write_text(json.dumps(state))
    args={'lock_path':tmp_path/'lock','state_path':tmp_path/'state','candidate':candidate,'inspect':inspect,'remove':remove,'present':present,'write_durable':write}
    with pytest.raises(RuntimeError,match='interrupted'):retire_pair(**args)
    assert json.loads((tmp_path/'state').read_text())['operations'][0]['phase']=='control_deleted'
    fail[0]=False;assert retire_pair(**args)['phase']=='deleted';assert removed==['control','full']
    assert retire_pair(**args)['phase']=='deleted' and removed==['control','full']
    assert set(json.loads((tmp_path/'state').read_text())['retired_roles'])=={'full','control'}


def test_inspection_failure_never_persists_or_deletes(tmp_path):
    def reject(*a):raise ValueError('protected owner')
    state=tmp_path/'state'
    with pytest.raises(ValueError,match='protected owner'):
        retire_pair(lock_path=tmp_path/'lock',state_path=state,candidate={'role':'f','sha256':'a'*64,'control_role':'c','control_sha256':'b'*64},inspect=reject,remove=lambda *a:pytest.fail('removed'),present=lambda *a:True,write_durable=lambda p,v:p.write_text(json.dumps(v)))
    assert not state.exists()
