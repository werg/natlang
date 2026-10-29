from copy import deepcopy
from scripts.coreweave.native_publication_state import full_readback_verified


def evidence():
    pin={'path':'/work/checkpoint.pt','sha256':'a'*64,'bytes':123}
    entry={'role':'peer2-step8925','full_backup_verified':False}
    receipt={'role':entry['role'],'source':pin['path'],'private':True,
             'identity':{'sha256':pin['sha256'],'bytes':123},'run':'owner/project/run',
             'name':entry['role']+'/'+pin['sha256']+'/checkpoint.pt',
             'independent_verification':{'readback_location':'sandbox','sha256':pin['sha256'],'bytes':123}}
    return entry,pin,receipt


def test_complete_cloud_get_survives_owner_readback_phase_reset():
    entry,pin,receipt=evidence()
    assert full_readback_verified(entry,pin,receipt)
    assert receipt['independent_verification']['readback_location']=='sandbox'


def test_only_exact_completed_private_readback_qualifies():
    entry,pin,receipt=evidence()
    for field,value in [('role','another-role'),('source','/work/other.pt'),('name','other/object'),('private',False)]:
        altered=deepcopy(receipt);altered[field]=value
        assert not full_readback_verified(entry,pin,altered)
    for field,value in [('sha256','b'*64),('bytes',122),('status','awaiting_target_readback')]:
        altered=deepcopy(receipt);altered['independent_verification'][field]=value
        assert not full_readback_verified(entry,pin,altered)
    assert not full_readback_verified(entry,pin,None)
