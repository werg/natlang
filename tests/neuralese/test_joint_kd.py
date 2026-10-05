import json

import pytest
import torch

from natlang_neuralese.train.joint_kd import (
    IGNORE, chunked_ce_kl, kl_ramp, require_shared_tokenizer, tokenizer_fingerprint,
)


def _reference(hidden, weight, labels, teacher_hidden, teacher_weight, ce_weight, kl_weight, temperature):
    mask = labels != IGNORE
    logits = hidden[mask].float() @ weight.float().T
    ce = torch.nn.functional.cross_entropy(logits, labels[mask], reduction="sum")
    teacher = (teacher_hidden[mask].float() @ teacher_weight.float().T).detach()
    log_q = torch.log_softmax(teacher / temperature, -1)
    log_p = torch.log_softmax(logits / temperature, -1)
    kl = (log_q.exp() * (log_q - log_p)).sum()
    return (ce_weight * ce + kl_weight * temperature ** 2 * kl) / mask.sum()


@pytest.mark.parametrize("temperature", [1.0, 2.0])
def test_matches_autograd_reference(temperature):
    torch.manual_seed(0)
    vocab, d, dt = 50, 8, 12
    hidden = torch.randn(2, 7, d, requires_grad=True)
    weight = torch.randn(vocab, d, requires_grad=True)
    teacher_hidden, teacher_weight = torch.randn(2, 7, dt), torch.randn(vocab, dt)
    labels = torch.randint(0, vocab, (2, 7))
    labels[0, :3] = IGNORE
    loss, parts = chunked_ce_kl(hidden, weight, labels, teacher_hidden, teacher_weight, ce_weight=0.7,
                                kl_weight=0.5, temperature=temperature, chunk=3)
    (loss * 2.0).backward()
    gh, gw = hidden.grad.clone(), weight.grad.clone()
    hidden.grad = weight.grad = None
    reference = _reference(hidden, weight, labels, teacher_hidden, teacher_weight, 0.7, 0.5, temperature)
    (reference * 2.0).backward()
    assert torch.allclose(loss, reference, atol=1e-5)
    assert torch.allclose(gh, hidden.grad, atol=1e-5)
    assert torch.allclose(gw, weight.grad, atol=1e-5)
    assert parts.tokens == 11


def test_ce_only_and_normaliser():
    torch.manual_seed(1)
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    weight = torch.randn(9, 4)
    labels = torch.randint(0, 9, (1, 5))
    loss, parts = chunked_ce_kl(hidden, weight, labels, normaliser=10)
    reference = torch.nn.functional.cross_entropy(hidden[0] @ weight.T, labels[0], reduction="sum") / 10
    assert torch.allclose(loss, reference, atol=1e-5) and parts.kl == 0.0


def test_kl_ramp():
    assert kl_ramp(0, 100, 1.0) == 0.0 and kl_ramp(50, 100, 1.0) == 0.5 and kl_ramp(500, 100, 1.0) == 1.0


def _tokenizer(path, merges):
    path.mkdir()
    (path / "tokenizer.json").write_text(json.dumps({
        "model": {"vocab": {"a": 0, "b": 1}, "merges": merges}, "pre_tokenizer": None, "normalizer": None,
        "added_tokens": [{"id": 2, "content": "<x>"}]}))
    return path


def test_tokenizer_identity(tmp_path):
    a = _tokenizer(tmp_path / "a", ["a b"])
    b = _tokenizer(tmp_path / "b", ["a b"])
    c = _tokenizer(tmp_path / "c", [])
    assert require_shared_tokenizer(a, b) == tokenizer_fingerprint(a)
    with pytest.raises(ValueError):
        require_shared_tokenizer(a, c)
