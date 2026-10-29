import json
import pytest
from natlang_neuralese.train.native_predecessor_retirement import load_retirement,verify_retirement
from natlang_neuralese.train.recipe import _source_predecessor_retirement
from natlang_neuralese.common.hashing import sha256_file_hex


def test_exact_earned_source_and_history_are_required(tmp_path):
    proof={'schema':'natlang.native-source-predecessor-retirement/1','checkpoint_sha256':'a'*64,'train_sha256':'b'*64,'held_sha256':'c'*64,'predecessor_sha256':'d'*64}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof));proof=load_retirement(path)
    training={'source_receipts':[{'sha256':'b'*64},{'sha256':'c'*64}],'options':{'source_predecessors':{'continued_native_graphs':'d'*64}}}
    assert verify_retirement(training,proof,checkpoint_sha256='a'*64,train_sha256='b'*64,held_sha256='c'*64)==proof
    with pytest.raises(ValueError,match='checkpoint or source'):verify_retirement(training,proof,checkpoint_sha256='e'*64,train_sha256='b'*64,held_sha256='c'*64)
    with pytest.raises(ValueError,match='historical predecessor'):verify_retirement({**training,'options':{}},proof,checkpoint_sha256='a'*64,train_sha256='b'*64,held_sha256='c'*64)
    old={'native_graphs':{'sha256':'b'*64},'held_native_graphs':{'sha256':'c'*64},'continued_native_graphs':{'sha256':'d'*64},'foundation_native_graphs':{'sha256':'b'*64}}
    new={k:v for k,v in old.items() if k!='continued_native_graphs'};new['source_predecessor_retirement']={'path':str(path),'sha256':sha256_file_hex(path)}
    assert _source_predecessor_retirement(old,new)
    assert not _source_predecessor_retirement(old,{**new,'held_native_graphs':{'sha256':'e'*64}})
    path.write_text('{}');assert not _source_predecessor_retirement(old,new)


def test_retired_source_proof_is_provider_dependency_and_tampering_rejected(tmp_path):
    from scripts.coreweave.provider_dependencies import dependency_closure
    proof={'schema':'natlang.native-source-predecessor-retirement/1','checkpoint_sha256':'a'*64,'train_sha256':'b'*64,'held_sha256':'c'*64,'predecessor_sha256':'d'*64}
    path=tmp_path/'proof.json';path.write_text(json.dumps(proof))
    record=dict(proof,proof_path=str(path),proof_sha256=sha256_file_hex(path))
    payload={'training':{'options':{'source_predecessor_retirement_history':[record]}}}
    closure=dependency_closure({},namespace=str(tmp_path),_checkpoint_payload=payload)
    assert str(path) in {item['path'] for item in closure['files']}
    path.write_text('{}')
    with pytest.raises(ValueError):dependency_closure({},namespace=str(tmp_path),_checkpoint_payload=payload)
