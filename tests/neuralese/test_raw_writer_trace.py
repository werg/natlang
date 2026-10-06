from types import SimpleNamespace

import torch

from natlang_neuralese.eval.raw_writer_trace import RawWriterTrace


def test_exact_bfloat16_tokens_survive_float32_wire_transport():
    table = torch.tensor([[.1, .2], [.3, .4], [.5, .6]], dtype=torch.bfloat16)
    tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: ','.join(map(str, ids)))
    trace = RawWriterTrace(table)
    report = trace.decode(table[[2, 0]].float(), tokenizer)
    assert report['decoded'] == '2,0'
    assert report['exact_unique_embeddings'] == 2
    assert report['unknown_positions'] == report['ambiguous_positions'] == []
    assert report['approximate_decoding'] is False


def test_nonexact_and_duplicate_vectors_are_reported_without_guessing():
    table = torch.tensor([[1., 2.], [1., 2.], [3., 4.]])
    tokenizer = SimpleNamespace(decode=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('must not decode')))
    trace = RawWriterTrace(table)
    report = trace.decode(torch.tensor([[1., 2.], [3., 4.00001]]), tokenizer)
    assert report['decoded'] is None
    assert report['token_ids'] == [None, None]
    assert report['ambiguous_positions'] == [0]
    assert report['unknown_positions'] == [1]
