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


def test_open_writer_sampler_keeps_padded_vocabulary_close_token_sampleable():
    from natlang_neuralese.serve.engine import Engine, GenerationRequest, Sequence
    request=GenerationRequest(messages=[],temperature=0)
    seq=Sequence(request,None,native_capture_open=True,native_own_position=11)
    seq.logits=torch.zeros(1,128000)
    seq.logits[0,125034]=10  # produce close; 125036 is consume close
    seq.bans[10]={125034}  # an unrelated guidance position is not a global ban
    assert Engine._native_token(None,seq)[0]==125034
    seq.bans[11]={125034}  # a specific guidance rejection is the only mask here
    assert Engine._native_token(None,seq)[0]!=125034
