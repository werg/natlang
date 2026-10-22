"""Complete trained state replaces the base weight read, without numeric drift."""
import pytest
import torch

from natlang_neuralese.model.native_model import load_native_checkpoint, model_state
from natlang_neuralese.model.shared_state import Invocation, compile_graph
from test_native_serving_adapters import engine_fixture


def exact(actual, expected):
    if torch.is_tensor(expected):
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            exact(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_checkpoint_restore_uses_saved_backbone_and_keeps_exact_buffers_and_logits(tmp_path, monkeypatch, family):
    engine = engine_fixture(tmp_path, family)
    original = engine.shared_model
    plan = compile_graph([Invocation('answer', (4, 5, 6, 7))])
    expected = original.logits(original(plan)['h_final']).detach()
    if family == 'mellum':
        from natlang_neuralese.maple import student
        monkeypatch.setattr(student, 'load_student',
                            lambda *a, **kw: pytest.fail('base weights loaded during checkpoint restore'))
    else:
        from transformers import AutoModelForCausalLM
        model_type = type(original.port.hf)
        loader = model_type.from_pretrained
        def from_saved(path, *args, **options):
            assert path is None
            assert options['state_dict']
            return loader(path, *args, **options)
        monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained',
                            lambda *a, **kw: pytest.fail('Auto loader attempted a base weight source'))
        monkeypatch.setattr(model_type, 'from_pretrained', from_saved)
    restored, _, _, _, state = load_native_checkpoint(
        tmp_path / 'native.pt', device=original.port.embedding_weight.device)
    exact(model_state(restored), model_state(original))
    assert dict(original.named_buffers()).keys() == dict(restored.named_buffers()).keys()
    for (name, before), (other_name, after) in zip(original.named_buffers(), restored.named_buffers()):
        assert name == other_name and not after.is_meta
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert restored.tied == original.tied
    torch.testing.assert_close(restored.logits(restored(plan)['h_final']), expected, rtol=0, atol=0)
    # A CPU model is mutable independently of the decoded parent state.
    first_name, first = next(restored.port.hf.named_parameters())
    saved = state['backbone_state'][first_name].clone()
    with torch.no_grad():
        first.add_(1)
    torch.testing.assert_close(state['backbone_state'][first_name], saved, rtol=0, atol=0)
