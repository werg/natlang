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


def test_crash_after_control_delete_before_phase_write_is_resumable(tmp_path):
    candidate={'role':'full','sha256':'a'*64,'control_role':'control','control_sha256':'b'*64};objects={'full':'a'*64,'control':'b'*64};crash=[True];removed=[]
    def write(path,state):
        if state['operations'][-1]['phase']=='control_deleted' and crash[0]:raise RuntimeError('crash before phase commit')
        path.write_text(json.dumps(state))
    def remove(role,sha):assert objects.pop(role)==sha;removed.append(role)
    args={'lock_path':tmp_path/'lock','state_path':tmp_path/'state','candidate':candidate,'inspect':lambda *a:None,'remove':remove,'present':lambda r,h:objects.get(r)==h,'write_durable':write}
    with pytest.raises(RuntimeError,match='before phase commit'):retire_pair(**args)
    assert json.loads((tmp_path/'state').read_text())['operations'][0]['phase']=='intent'
    crash[0]=False;assert retire_pair(**args)['phase']=='deleted';assert removed==['control','full']


def test_full_retirement_resume_rejects_changed_successor(tmp_path):
    from scripts.coreweave.retention_transaction import retire_full
    state=tmp_path/'state';state.write_text(json.dumps({'retired_roles':[],'operations':[{'role':'old','sha256':'a'*64,'successor_role':'new','successor_sha256':'b'*64,'fallback_role':'backup','phase':'intent'}]}))
    with pytest.raises(ValueError,match='identity changed'):
        retire_full(lock_path=tmp_path/'lock',state_path=state,candidate={'role':'old','sha256':'a'*64,'successor_role':'new','successor_sha256':'c'*64,'fallback_role':'backup'},inspect=lambda *a:None,remove=lambda *a:None,present=lambda *a:False,write_durable=lambda p,v:None)
