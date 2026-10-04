import pytest
import torch
from types import SimpleNamespace
from scripts.projection_scoring import validate_completion, score_completion


def test_eos_and_support_are_not_silently_truncated():
    validate_completion([2,3,1],1,4)
    for ids in ([2,3],[1,2,1],[True,1],[-1,1],[],[9,1]):
        with pytest.raises(ValueError):validate_completion(ids,1,4)


def test_canonical_initialization_stops_at_eos_not_template_newline(monkeypatch):
    import scripts.projection_scoring as scoring
    monkeypatch.setattr(scoring,'_call_template',lambda t,m,tools,g: 'prompt' if g else 'promptanswer<EOS>\n')
    class Tokenizer:
        def __len__(self):return 4
        def encode(self,text,add_special_tokens):
            assert text=='answer<EOS>' and not add_special_tokens
            return [2,1]
    assert scoring.completion_ids(Tokenizer(),[],[],{'content':'answer'},1)==[2,1]


def test_scoring_matches_temperature_distribution_and_first_token(monkeypatch):
    import scripts.projection_scoring as scoring
    monkeypatch.setattr(scoring,'_call_template',lambda *args:'prompt')
    class Tokenizer:
        def __len__(self):return 4
        def encode(self,text,add_special_tokens):
            assert not add_special_tokens
            return [0,2]
    class Model:
        device='cpu'
        def __call__(self,input_ids,use_cache):
            assert input_ids.tolist()==[[0,2,3,1]] and not use_cache
            # Positions 1 and 2 predict target tokens 3 and EOS(1).
            # A large padded output-head logit must have zero proposal support,
            # exactly as it does in projection generation.
            return SimpleNamespace(logits=torch.tensor([[[0.,0.,0.,0.,100.],[0.,1.,2.,3.,100.],[3.,2.,1.,0.,100.],[0.,0.,0.,0.,100.]]]))
    for temp in (1.,.6):
        d=score_completion(Model(),Tokenizer(),[],[],[3,1],temp,4,1)
        want=torch.log_softmax(torch.tensor([[0.,1.,2.,3.],[3.,2.,1.,0.]])/temp,-1)[torch.arange(2),torch.tensor([3,1])].sum().item()
        assert d['sum_logprob']==pytest.approx(want)
        assert d['token_count']==2 and d['includes_assistant_terminator']
    with pytest.raises(ValueError,match='context'):score_completion(Model(),Tokenizer(),[],[],[3,1],1,3,1)


def test_hidden_generation_processors_cannot_masquerade_as_temperature_only():
    from scripts.projection_scoring import validate_sampling_config
    validate_sampling_config(SimpleNamespace(num_beams=1,min_length=0,forced_eos_token_id=None))
    for config in (SimpleNamespace(num_beams=2),SimpleNamespace(forced_eos_token_id=1),SimpleNamespace(min_new_tokens=1)):
        with pytest.raises(ValueError,match='constraint'):validate_sampling_config(config)
