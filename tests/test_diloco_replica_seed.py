import hashlib,json,shutil
import pytest,torch
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.diloco.coordinator import initialize_coordinator
from natlang_neuralese.train.diloco.replica_baseline import capture
from natlang_neuralese.train.diloco.replica_generation import load_active
from natlang_neuralese.train.diloco.replica_journal import ReplicaJournal,_identity
from natlang_neuralese.train.diloco.replica_seed import publish_seed,verify_seed,install_seed,release_seed_lease,serve_seed,download_seed
from natlang_neuralese.train.diloco.replica_state import ReplayCoordinator


def test_publish_and_install_exact_witnessed_replacement_peer(tmp_path):
    base=tmp_path/'base';initialize_coordinator(base,run_id='r',masters={'weight':torch.ones(4)},identity={'parent':'p'},workers=1,min_quorum=1,fragment_bytes=1024)
    machine=ReplayCoordinator(base,quorum_since={})
    class Capture:
        def __init__(self):self.machine=machine
        def read(self,fn):return fn()
    baseline=tmp_path/'baseline';receipt=capture(Capture(),baseline);store=tmp_path/'oldstore';store.mkdir()
    j=ReplicaJournal(store/'journal0',run_identity='r',baseline_sha256=receipt['sha256'])
    empty=tmp_path/'empty';empty.write_bytes(b'');event={'kind':'lease','args':{'worker':0,'session':'00000000-0000-0000-0000-000000000001'},'wall':100,'monotonic':20}
    j.append(event,empty,expected_sequence=1,expected_previous=j.head);state=machine.apply(event,empty)
    j.append({'kind':'seal','operation_sequence':1,'state':state},empty,expected_sequence=2,expected_previous=j.head)
    head=j.head;write_json_durable(store/'replica-active.json',{'id':'initial','baseline':str(baseline),'baseline_sha256':receipt['sha256'],'journal':str(j.root)})
    uploads=store/'replica-uploads';uploads.mkdir();(uploads/'source.bin').write_bytes(b'source-file');(uploads/'source.bin.sha256').write_text(hashlib.sha256(b'source-file').hexdigest()+'\n')
    write_json_durable(store/'replica-files-frozen.json',{'schema':'natlang.diloco-replica-file-freeze/1','nonce':'n'*48})
    with pytest.raises(ValueError):publish_seed(store,tmp_path/'bad',expected_sequence=2,expected_head='f'*64,lease_id='transfer',freeze_nonce='n'*48)
    record=publish_seed(store,tmp_path/'seed',expected_sequence=2,expected_head=head,lease_id='transfer',freeze_nonce='n'*48);j.close()
    digest=hashlib.sha256((tmp_path/'seed'/'seed-receipt.json').read_bytes()).hexdigest()
    assert verify_seed(tmp_path/'seed',expected_receipt_sha256=digest)['head']==head
    import multiprocessing as mp,socket,time
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    ctx=mp.get_context('fork');stop=ctx.Event();server=ctx.Process(target=serve_seed,args=(tmp_path/'seed','127.0.0.1',port,'s'*32),kwargs={'stop':stop});server.start()
    try:
        for _ in range(100):
            try:
                downloaded=download_seed(f'http://127.0.0.1:{port}','s'*32,tmp_path/'received',expected_receipt_sha256=digest,allow_http=True);break
            except OSError:time.sleep(.02)
        assert downloaded['head']==head
    finally:stop.set();server.join(8)
    assert server.exitcode==0
    install_seed(tmp_path/'received',tmp_path/'replacement',expected_receipt_sha256=digest)
    assert (tmp_path/'replacement'/'replica-uploads'/'source.bin').read_bytes()==b'source-file'
    new,recovered,_=load_active(tmp_path/'replacement',run_identity='r',destination=tmp_path/'recovered')
    assert new.sequence==2 and new.head==head and recovered.coordinator.progress==machine.coordinator.progress
    release_seed_lease(store,record,witnessed={'run_id':'r','baseline_sha256':receipt['sha256'],'sequence':2,'head':head});new.close()
