import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

spec=importlib.util.spec_from_file_location('writer_logit_probe',Path(__file__).resolve().parents[2]/'scripts/neuralese_data/probe_native_writer_logits.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class Engine:
    role_tokens=SimpleNamespace(produce_open=8,produce_close=9)
    shared_model=SimpleNamespace(port=SimpleNamespace(embedding_weight=torch.empty(20,2)))
    def _native_prefill_forward(self,ids,cache):return {'cache':list(ids),'h_final':torch.tensor([[float(sum(ids))]])}
    def _native_forward(self,ids,cache,producing=False):
        assert producing is (ids==[3])
        return self._native_prefill_forward(ids+cache,None)
    def _native_logits(self,h):
        result=torch.zeros(1,20);result[0,9]=h[0,0];return result

def test_recorded_prefix_replay_measures_each_actual_sample_and_private_context():
    capture={'prompt':[{'kind':'tokens','token_ids':[1,2]}],
       'generated_inputs':[{'token_id':8,'input_role':'ordinary','sample_index':None},
                           {'token_id':3,'input_role':'produce','sample_index':0}],
       'samples':[{'index':0,'token_id':3,'input_role':'produce','terminal':False},
                  {'index':1,'token_id':2,'input_role':'produce','terminal':True}],
       'protocol_failure':{'stop_token_ids':[2]}}
    result=probe.replay(Engine(),capture)
    assert len(result)==2 and result[0]['produce_close']['token_id']==9
    assert result[1]['produce_close']['logit']==result[0]['produce_close']['logit']+3
    changed=probe.replay(Engine(),capture,perturb=True)
    assert changed[0]['produce_close']['logit']==result[0]['produce_close']['logit']+1
