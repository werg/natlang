"""Exercise terminal and producer-boundary handling in the current native sampler."""
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.serve.engine import Engine, GenerationRequest, Sequence
from natlang_neuralese.serve.chat import RequestError


@pytest.mark.parametrize('temperature', [0.0, 0.6])
@pytest.mark.parametrize('token', [7, 5, 18])
def test_native_sampling_stops_or_rejects_only_at_native_boundaries(temperature, token):
    # Stop/eos terminates text. Ordinary tokens enter the native cache. A
    # produce-close marker is structural and must fail when no producer span is
    # open; it is not the old two-marker text-mode stop control.
    engine = Engine.__new__(Engine)
    engine.native_state_enabled = True
    engine.stop_ids = {7}
    engine.role_tokens = SimpleNamespace(produce_open=17, produce_close=18,
                                         consume_open=19, consume_close=20)
    engine._native_decoded_tokens = 0
    class Tokenizer:
        def decode(self, tokens, skip_special_tokens=False):
            return ''.join(map(str, tokens))
    engine.tokenizer = Tokenizer()
    engine._native_response = lambda seq: {"items": list(seq.items), "reason": seq.finish_reason}
    events = []
    engine._native_forward = lambda ids, cache, **kwargs: {
        'cache': cache, 'h_final': torch.zeros((1, 1, 4))}
    engine._native_logits = lambda hidden: torch.zeros((1, 32))
    engine._native_record_input = lambda *args, **kwargs: None
    engine._native_guide = lambda seq: False
    engine._emit = lambda seq, token: events.append(token)

    request = GenerationRequest(messages=[], temperature=temperature, on_delta=events.append)
    logits = torch.full((1, 32), -1000.0)
    logits[0, token] = 1000.0
    seq = Sequence(request, Future(), phase='text', items=[5], generated_positions=1,
                   logits=logits, rng=torch.Generator().manual_seed(42))

    engine._native_round_impl([seq])
    if token == 7:
        assert seq.phase == 'done'
        assert seq.future.result() == {'items': [5], 'reason': 'stop'}
        assert seq.items == [5] and seq.generated_positions == 1
        assert events == []
    elif token == 18:
        with pytest.raises(RequestError, match='produce end marker without an open span'):
            seq.future.result()
        assert seq.phase == 'done'
        assert seq.items == [5] and seq.generated_positions == 1
        assert events == []
    else:
        assert seq.phase == 'text' and not seq.future.done()
        assert seq.items == [5, 5] and seq.generated_positions == 2
        assert events == [5]
