"""Exercise the text-mode control boundary through the real serving sampler."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.serve.engine import Engine, GenerationRequest, Sequence


@pytest.mark.parametrize('temperature', [0.0, 0.6])
@pytest.mark.parametrize('token', [6, 7, 18])
def test_text_sampling_stops_at_close_without_emitting_or_forwarding_it(temperature, token):
    # The real QAD divergence gave the close marker almost all probability.
    # Use a deterministic margin to test the same branch without model weights.
    engine = Engine.__new__(Engine)
    engine.backbone = SimpleNamespace(controls=SimpleNamespace(open_id=17, close_id=18))
    engine.stop_ids = {6}
    engine._response = lambda seq: {'items': list(seq.items), 'reason': seq.finish_reason}
    events = []
    request = GenerationRequest(messages=[], temperature=temperature, on_delta=events.append)
    logits = torch.full((1, 32), -1000.0)
    logits[0, token] = 1000.0
    seq = Sequence(request, Future(), phase='text', items=[7], generated_positions=1,
                   logits=logits, rng=torch.Generator().manual_seed(42))

    chosen = engine._choose(seq)
    if token in {6, 18}:
        assert chosen is None and seq.phase == 'done'
        assert seq.future.result() == {'items': [7], 'reason': 'stop'}
    else:
        assert chosen == 7 and seq.phase == 'text'
        assert not seq.future.done()
    assert events == [] and seq.items == [7] and seq.generated_positions == 1
