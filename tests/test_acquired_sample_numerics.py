"""Source checks use actual per-shard reduction settings without resetting history."""
import json

import pytest
import torch

from scripts.neuralese_data.qualify_acquired_session_samples import conversion_threads,recorded_runtime
from scripts.neuralese_data.build_native_graph_text_corpus import numerical_runtime
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.text_supervision import segmented_position_weights


def test_recorded_threads_restore_exact_weights_and_caller_after_failure():
    previous=torch.get_num_threads()
    span=torch.ones((1,80013),dtype=torch.bool)
    try:
        torch.set_num_threads(2)
        reference=segmented_position_weights(span,[(79900,80013)],context_weight=.05)
        settings=numerical_runtime()
        torch.set_num_threads(1)
        with conversion_threads(settings) as actual:
            assert actual['torch_threads']==2
            assert torch.equal(reference,segmented_position_weights(span,[(79900,80013)],context_weight=.05))
        assert torch.get_num_threads()==1
        with pytest.raises(RuntimeError,match='reference error'):
            with conversion_threads(settings):raise RuntimeError('reference error')
        assert torch.get_num_threads()==1
    finally:torch.set_num_threads(previous)


def test_legacy_override_is_current_reference_not_invented_history():
    previous=torch.get_num_threads()
    with conversion_threads(None,2) as actual:assert actual['torch_threads']==2
    assert torch.get_num_threads()==previous
    with pytest.raises(ValueError,match='positive integer'):
        with conversion_threads(None,0):pass


def test_each_shard_keeps_its_actual_resume_setting(tmp_path):
    shards=[]
    for start,stop,threads in ((0,2,2),(2,5,4)):
        path=tmp_path/f'{start}.json';path.write_text(json.dumps({'runtime':{'numerical_runtime':{'torch_threads':threads}}}))
        shards.append({'path':str(path),'sha256':sha256_file_hex(path),'start':start,'stop':stop})
    receipt={'shards':shards,'runtime':{'numerical_runtime':{'torch_threads':8}}}
    assert recorded_runtime(receipt,0)=={'torch_threads':2}
    assert recorded_runtime(receipt,4)=={'torch_threads':4}
    assert recorded_runtime({},0) is None
