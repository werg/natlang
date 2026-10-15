"""Actual decoding captures equal exact-token replay on both backbone families."""
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.rl.sampling import capture_items, captured_logps


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('terminal', [True, False])
def test_actual_engine_capture_matches_replay(family, terminal, tmp_path):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.model.lfm2_port import ControlTokens
    from natlang_neuralese.serve.engine import Engine, GenerationRequest
    from natlang_neuralese.serve.grad import GradSession
    from natlang_neuralese.serve.store import TensorStore
    from natlang_neuralese.serve.chat import declared_history_reasoning
    if family == 'lfm':
        from test_backbone_policy import tiny_lfm
        backbone = tiny_lfm()
    else:
        from test_mellum_port import tiny_mellum
        from natlang_neuralese.maple.model import load_maple
        from natlang_neuralese.maple.maple_port import MaplePortBackbone
        tiny_mellum(tmp_path)
        model = load_maple(tmp_path, device='cpu', dtype=torch.float32, ternary_attention=False)
        backbone = MaplePortBackbone(model, ControlTokens(126, 127), fast=False)
    heads = PortHeads(backbone, cutoff=backbone.num_layers, max_length=8, profile='raw-token-v1').eval()
    class Tokens:
        eos_token_id = 2
        natlang_history_reasoning = {'policy': declared_history_reasoning(backbone.config.model_type)}
        def convert_tokens_to_ids(self, name):
            return 2 if name == '<|im_end|>' else None
        def decode(self, ids, **kwargs):
            return ' '.join(f't{i}' for i in ids)
    engine = Engine(backbone, heads, Tokens(), TensorStore(), 'test', context=64)
    engine._prompt_plan = lambda *_: [('tokens', [1, 4, 9])]
    if terminal:
        engine.stop_ids = set(range(backbone.config.vocab_size))
    # Actual generation, not a fabricated probability record.
    response = engine.generate(GenerationRequest(messages=[{'role': 'user', 'content': 'input'}],
                                                 max_tokens=4, temperature=1., seed=71, rl_capture=True))
    capture = response['x_natlang_rl_capture']
    prompt, target, events = capture_items(capture)
    assert prompt == [('tok', 1), ('tok', 4), ('tok', 9)]
    if terminal:
        assert len(capture['items']) == 0 and len(target) == 1
    with torch.no_grad():
        current, sampled = captured_logps(GradSession(engine), capture)
    assert current.shape == sampled.shape and sampled.numel() >= 1
    assert torch.allclose(current, sampled, atol=2e-5, rtol=2e-5)


def test_capture_refuses_guided_or_missing_receipts():
    with pytest.raises(ValueError, match='RFT-only'):
        capture_items({})
    with pytest.raises(ValueError, match='guided'):
        capture_items({'schema': 'natlang.rl-sampling/1', 'guided': True})
