"""Empty Neuralese is a model-bound zero-slot identity, never a fake token/vector."""
import pytest
import torch
from test_shared_state import tiny
from natlang_neuralese.model.shared_state import (SharedStateModel, native_step, splice_block,
    concatenate_captures, compile_graph, Invocation, ImportSpan, capture_block)
from natlang_neuralese.model.state_block import NativeStateBlock
from natlang_neuralese.serve.store import encode_native_state, decode_native_state


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_empty_capture_roundtrip_splice_and_graph(family, dtype):
    model = SharedStateModel(tiny(family), (60, 61, 62, 63), rank=4).to(dtype=dtype).eval()
    identity = dict(checkpoint_sha256='1'*64, tokenizer_fingerprint='2'*64, embedding_weight_sha256='3'*64)
    architecture = model.port.config.model_type
    with torch.inference_mode():
        prefix = native_step(model, torch.tensor([[7, 8]]))
        empty = native_step(model, torch.empty((1, 0), dtype=torch.int64), prefix['cache'], producing=True)
        assert empty['cache'] is prefix['cache']
        assert all(value.dtype == torch.float32 for layer in empty['layers'] for key, value in layer.items() if key != 'kind')
        fields = concatenate_captures([empty])
        block = NativeStateBlock(text='', architecture=architecture, **identity, **fields).validate()
        wire = encode_native_state(block)
        restored, state_id = decode_native_state(wire)
        assert state_id.startswith('nzs1_') and len(restored.token_ids) == 0
        assert restored.embeddings.shape == (0, model.port.embedding_weight.shape[1])
        assert splice_block(model, prefix['cache'], restored, **identity) is prefix['cache']
        actual = native_step(model, torch.tensor([[9]]), prefix['cache'])
        # Compare against an independent cache to catch hidden state mutation.
        expected = native_step(model, torch.tensor([[9]]), native_step(model, torch.tensor([[7, 8]]))['cache'])
        assert torch.equal(actual['h_final'], expected['h_final'])
        plan = compile_graph([Invocation('source', (60, 61), producing=((1, 1),)),
            Invocation('target', (62, 63, 9), imports=(ImportSpan(1, 'source', 1, 1), ImportSpan(1, 'empty', 0, 0)))], external_sizes={'empty': 0})
        result = model(plan, external_blocks={'empty': restored}, external_identity=dict(identity, architecture=architecture))
        captured = capture_block(result, 'source', 1, 1, text='', architecture=architecture, **identity)
        assert len(captured.token_ids) == 0
        plain = model(compile_graph([Invocation('target', (62, 63, 9))]))
        assert torch.equal(result['h_final'][:, 2:], plain['h_final'])
