"""Run in the training image; the lightweight harness environment has no torch."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('peft')
import torch.nn.functional as F
from transformers import Lfm2Config, Lfm2ForCausalLM
from scripts.train_dpo import dpo_backward, sequence_logp


def tiny_model():
    torch.manual_seed(7)
    config = Lfm2Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, full_attn_idxs=[0, 1], use_cache=False)
    return Lfm2ForCausalLM(config).eval()


def test_sequence_logp_scores_only_the_completion():
    model = tiny_model()
    x, y = [3, 5, 7, 9], [11, 13, 2]
    logits = model(input_ids=torch.tensor([x + y])).logits[0].float()
    expected = sum(F.log_softmax(logits[len(x) - 1 + i], -1)[token] for i, token in enumerate(y))
    torch.testing.assert_close(sequence_logp(model, (x, y)), expected)


@pytest.mark.parametrize('nll_weight', [0.0, 0.5])
def test_side_by_side_gradients_equal_the_joint_dpo_loss(nll_weight):
    model = tiny_model()
    pair = {'chosen': ([3, 5, 7, 9], [11, 13, 2]), 'rejected': ([3, 5, 7, 9], [17, 4])}
    reference, beta = [-7.5, -5.0], 0.3
    chosen, rejected = sequence_logp(model, pair['chosen']), sequence_logp(model, pair['rejected'])
    margin = beta * ((chosen - reference[0]) - (rejected - reference[1]))
    joint = -F.logsigmoid(margin) - nll_weight * chosen / len(pair['chosen'][1])
    joint.backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    loss, got_margin = dpo_backward(model, pair, reference, beta, nll_weight)
    assert got_margin == pytest.approx(margin.item(), rel=1e-5)
    assert loss == pytest.approx(-F.logsigmoid(margin).item(), rel=1e-5)
    for n, p in model.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n], atol=1e-6, rtol=1e-5)
