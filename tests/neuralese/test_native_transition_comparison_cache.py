"""Cross-process comparison memoization keeps fresh checkpoint authentication."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_native_execution_transition import parent
from natlang_neuralese.train.native_execution_transition import derive


def _ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size}


@pytest.fixture
def process_fixture(tmp_path):
    source=parent();old=tmp_path/'old.pt';new=tmp_path/'new.pt';torch.save(source,old)
    old_ref=_ref(old);torch.save(derive(source,parent_sha256=old_ref['sha256']),new)
    refs=tmp_path/'refs.json';refs.write_text(json.dumps({'old':old_ref,'new':_ref(new)}))
    script=tmp_path/'verify.py'
    script.write_text('''import json, os, sys\nfrom pathlib import Path\nfrom natlang_neuralese.train import native_execution_transition as tx\nfrom natlang_neuralese.rl import native_raw_policy, native_direct_sft_recovery as recovery\nroot=Path(sys.argv[1]); refs=json.loads((root/'refs.json').read_text()); counter=root/'comparisons'; hashlog=root/'hashes'\nnative_raw_policy.saved_native_identity=lambda state,checkpoint_sha256: {'checkpoint_sha256':checkpoint_sha256,'profile':state['model_config'].get('native_shared_execution','source-v1')}\nreal=tx.verify_states\ndef counted(*args, **kwargs):\n with counter.open('a') as f:f.write('1\\n')\n return real(*args, **kwargs)\ntx.verify_states=counted\nstrong=recovery._strong_binding\ndef observed(path):\n with hashlog.open('a') as f:f.write(str(path)+'\\n')\n return strong(path)\nrecovery._strong_binding=observed\nif os.environ.get('CHANGE_CONTROL'): tx.TARGET='other-profile'\nproof=root/'proof.json'\nif not proof.exists():\n result=tx.verify_files(refs['old'],refs['new']);proof.write_text(json.dumps(result,sort_keys=True))\nelse:\n import hashlib\n raw=proof.read_bytes();binding={'path':str(proof),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}\n result=tx.verify_files(refs['old'],refs['new'],proof_ref=binding)\nprint(json.dumps(result,sort_keys=True))\n''')
    env={**os.environ, 'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'training/neuralese'),
         'NATLANG_EXECUTION_TRANSITION_CACHE':str(tmp_path/'cache'),
         'NATLANG_HASH_CACHE':str(tmp_path/'hash-cache')}
    def run(*, control=False, expect=True):
        result=subprocess.run([sys.executable,str(script),str(tmp_path)],
            env={**env, **({'CHANGE_CONTROL':'1'} if control else {})},capture_output=True,text=True)
        if expect: assert result.returncode==0,result.stderr
        else: assert result.returncode!=0
        return result
    return tmp_path, old, new, run


def test_cross_process_reuse_needs_proof_and_fresh_bytes(process_fixture):
    root,old,new,run=process_fixture
    run();run()  # First full comparison makes the proof; second authenticates and caches it.
    assert (root/'comparisons').read_text().splitlines()==['1','1']
    first_hashes=len((root/'hashes').read_text().splitlines())
    run()
    assert (root/'comparisons').read_text().splitlines()==['1','1']
    later=(root/'hashes').read_text().splitlines()[first_hashes:]
    assert str(old) in later and str(new) in later and str(root/'proof.json') in later
    # Same bytes, changed physical identity: recompute, never reuse the prior memo.
    s=new.stat();os.utime(new,ns=(s.st_atime_ns,s.st_mtime_ns+1))
    run();assert len((root/'comparisons').read_text().splitlines())==3
    proof=root/'proof.json';s=proof.stat();os.utime(proof,ns=(s.st_atime_ns,s.st_mtime_ns+1))
    run();assert len((root/'comparisons').read_text().splitlines())==4
    # A changed verifier control forces the actual comparator and fails closed.
    run(control=True,expect=False)
    assert len((root/'comparisons').read_text().splitlines())==5


def test_modified_cache_or_independently_bound_proof_never_authorizes_hit(process_fixture):
    root,old,new,run=process_fixture
    run();run();run()
    cache=root/'cache';records=list(cache.glob('*.json'));assert len(records)==1
    record=json.loads(records[0].read_text());record['body']['result']['global_step']=999
    records[0].write_text(json.dumps(record))
    run();assert len((root/'comparisons').read_text().splitlines())==3
    # An externally rebound but false proof is compared, rejected and never cached.
    proof=root/'proof.json';value=json.loads(proof.read_text());value['global_step']=float(value['global_step']);proof.write_text(json.dumps(value))
    run(expect=False);assert len((root/'comparisons').read_text().splitlines())==4


def test_corrupt_checkpoint_is_rejected_even_with_existing_cross_process_memo(process_fixture):
    root,old,new,run=process_fixture
    run();run();run();old.write_bytes(old.read_bytes()+b'corrupt')
    run(expect=False)
    assert len((root/'comparisons').read_text().splitlines())==2


def test_live_verifier_code_change_invalidates_in_process_result(tmp_path, monkeypatch):
    from natlang_neuralese.rl import native_raw_policy
    from natlang_neuralese.train import native_execution_transition as tx
    source=parent();old=tmp_path/'old.pt';new=tmp_path/'new.pt';torch.save(source,old)
    old_ref=_ref(old);torch.save(derive(source,parent_sha256=old_ref['sha256']),new)
    monkeypatch.setenv('NATLANG_HASH_CACHE',str(tmp_path/'hash-cache'))
    monkeypatch.setenv('NATLANG_EXECUTION_TRANSITION_CACHE',str(tmp_path/'cache'))
    monkeypatch.setattr(native_raw_policy,'saved_native_identity',lambda state,checkpoint_sha256:
        {'checkpoint_sha256':checkpoint_sha256,'profile':state['model_config'].get('native_shared_execution','source-v1')})
    tx.verify_files(old_ref,_ref(new))
    def changed(*args,**kwargs):
        raise ValueError('new comparator ran')
    monkeypatch.setattr(tx,'verify_states',changed)
    with pytest.raises(ValueError,match='new comparator ran'):
        tx.verify_files(old_ref,_ref(new))
