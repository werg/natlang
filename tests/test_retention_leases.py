import json
import pytest
from scripts.coreweave.retention_leases import change_lease,leased_roles
from scripts.coreweave.retention_transaction import retire_full


def test_registration_and_retirement_are_mutually_fenced(tmp_path):
    lock=tmp_path/'standalone-cloud-retention.lock';state=tmp_path/'state';leases=tmp_path/'standalone-retention-consumer-leases.json';write=lambda p,v:p.write_text(json.dumps(v));identity={'boot':'b','pid':3,'start':'4'}
    kwargs={'lock_path':lock,'state_path':leases,'retirement_state_path':state,'role':'old','consumer_id':'consumer','identity':identity,'write_durable':write}
    with pytest.raises(ValueError,match='shared retention namespace'):change_lease(**(kwargs|{'lock_path':tmp_path/'other'}),release=False)
    change_lease(**kwargs,release=False);assert leased_roles(leases)=={'old'}
    c={'role':'old','sha256':'a'*64}
    with pytest.raises(ValueError,match='lease'):retire_full(lock_path=lock,state_path=state,candidate=c,inspect=lambda *a:None,remove=lambda *a:None,present=lambda *a:False,write_durable=write)
    with pytest.raises(ValueError,match='identity'):change_lease(**(kwargs|{'identity':{'pid':9}}),release=True)
    change_lease(**kwargs,release=True);retire_full(lock_path=lock,state_path=state,candidate=c,inspect=lambda *a:None,remove=lambda *a:None,present=lambda *a:False,write_durable=write)
    with pytest.raises(ValueError,match='retirement'):change_lease(**kwargs,release=False)
