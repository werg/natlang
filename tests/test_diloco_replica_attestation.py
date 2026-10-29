import hashlib,json,os,subprocess,sys,time
import pytest
from natlang_neuralese.train.diloco.replica_attestation import load_owned_config,verify_attestation
from scripts.coreweave.diloco_replica_controller import prepare_owned_activation


def test_owned_activation_requires_exact_fresh_evidence(tmp_path):
    now=time.time();primary={'sandbox_id':'s','boot_id':'b','owner_id':'o','pid':4,'start_ticks':'5'}
    config={'run_id':'r','baseline_sha256':'a'*64,'sequence':2,'head':'b'*64,'activation_nonce':'x'*48,'expected_primary':primary,'expected_workers':[0],
            'session_transition':{'generation_nonce':'n'*48,'new_url':'https://new','new_token_sha256':'c'*64}}
    evidence={'run_id':'r','primary_fenced':True,'all_workers_quiesced':True,'proof_sha256':'d'*64,'owner_observed_at':now-1,'primary':primary,'quiesced_workers':[0]}
    stop={'identity':primary,'nonce':'x'*48,'process_absent':True,'port_closed':True,'observed_at':now-1}
    raw,digest=prepare_owned_activation(config,mode='restart',supervisor_evidence=evidence,now=now,owner_stop=stop)
    path=tmp_path/'owned.json';path.write_bytes(raw);owned=load_owned_config(path,digest)
    assert verify_attestation(owned,mode='restart',now=now)==evidence
    with pytest.raises(ValueError):verify_attestation(owned,mode='restart',now=now+301)
    owned['head']='e'*64
    with pytest.raises(ValueError):verify_attestation(owned,mode='restart',now=now)
    with pytest.raises(ValueError):load_owned_config(path,'0'*64)
    with pytest.raises(ValueError):prepare_owned_activation(config,mode='restart',supervisor_evidence=evidence,now=now,owner_stop=dict(stop,port_closed=False))


def test_cli_help_without_provider_credentials():
    env={k:v for k,v in os.environ.items() if 'COREWEAVE' not in k.upper() and 'CW_' not in k.upper()}
    result=subprocess.run([sys.executable,'-m','natlang_neuralese.train.diloco.replica_cli','--help'],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0 and 'restart' in result.stdout
