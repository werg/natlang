import json
import pytest
from scripts.coreweave.diloco_replica_controller import run_promotion


def test_controller_never_releases_before_exact_ready(tmp_path):
    intent={'run_id':'r','generation_nonce':'n'*48,'sequence':4,'head':'h'*64,'new_url':'https://new','new_token_sha256':'t'*64}
    fence=lambda i:['fenced']
    verify=lambda i:{'run_id':'r','primary_fenced':True,'all_workers_quiesced':True,'proof_sha256':'p'*64}
    called=[]
    def release(i,f):called.append(True);return f
    with pytest.raises(ValueError):
        run_promotion(tmp_path,intent=intent,fence_workers=fence,verify_fence=verify,start_primary=lambda *_:{'ready':False},release_workers=release)
    assert not called and json.loads((tmp_path/'promotion.json').read_text())['phase']=='provider-fenced'
    with pytest.raises(ValueError):
        run_promotion(tmp_path,intent=intent,fence_workers=fence,verify_fence=verify,start_primary=lambda *_:{},release_workers=release)
    good=tmp_path/'good'
    ready={'ready':True,'run_id':'r','sequence':4,'head':'h'*64,'url':'https://new','token_sha256':'t'*64}
    result=run_promotion(good,intent=intent,fence_workers=fence,verify_fence=verify,start_primary=lambda *_:ready,release_workers=release)
    assert result['phase']=='released' and called==[True]
