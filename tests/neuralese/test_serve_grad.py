"""Reference server: gradient replay sessions and optimiser steps."""

import math

import pytest
import torch

from natlang_neuralese.serve.store import TensorStore, make_block

DIALECT = "nd:natlang@1"


@pytest.fixture(scope="module")
def engine(loaded):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine

    _, tokenizer, backbone = loaded
    torch.manual_seed(1)
    heads = PortHeads(backbone, cutoff=6, max_length=8).eval()
    for p in heads.parameters():
        p.requires_grad_(False)
    return Engine(backbone, heads, tokenizer, TensorStore(), DIALECT, max_block=4)


def term(kind, hint, target_text, **extra):
    return dict(kind=kind, messages=[{"role": "user", "content": [
        {"type": "text", "text": "Hint: "}, {"type": "neuralese", "id": hint},
        {"type": "text", "text": "\nWhat is the capital of France? Answer in one word."}]}],
        target={"role": "assistant", "content": target_text}, **extra)


def test_grad_and_adam_tune_a_soft_argument(engine):
    from natlang_neuralese.serve.grad import GradSession, embed_text, grad_dialect, optim_step

    hint = embed_text(engine, "geography quiz", type="Neuralese<string>")
    session = GradSession(engine)
    first = session.run({"arguments": [hint.id], "terms": [term("crossEntropy", hint.id, "Paris")]})
    grad = engine.store.get(first["gradients"][hint.id])
    assert grad.dialect == grad_dialect(DIALECT) and grad.payload.shape == hint.payload.shape
    assert float(grad.payload.abs().sum()) > 0
    # A constant (stopGradient): not an argument, so no gradient is returned for it.
    other = engine.store.put(make_block(torch.randn(2, engine.width), DIALECT))
    constant = session.run({"arguments": [], "terms": [term("crossEntropy", other.id, "Paris")]})
    assert constant["gradients"] == {}
    params, state, losses = [hint.id], None, [first["loss"]]
    for _ in range(4):
        result = session.run({"arguments": params, "terms": [term("crossEntropy", params[0], "Paris")]})
        step = optim_step(engine, {"optimizer": "adam", "hyper": {"lr": 0.05}, "params": params,
                                   "grads": [result["gradients"][params[0]]], "state": state})
        params, state = step["params"], step["state"]
        losses.append(result["loss"])
    final = session.run({"arguments": [], "terms": [term("crossEntropy", params[0], "Paris")]})["loss"]
    assert final < losses[0], (losses, final)
    assert engine.store.get(hint.id) is not None  # immutable: the original is unchanged
    assert state["step"] == 4 and len(state["m"]) == 1


def test_log_likelihood_self_distill_and_kl_terms(engine):
    from natlang_neuralese.serve.engine import GenerationRequest
    from natlang_neuralese.serve.grad import GradSession, embed_text

    hint = embed_text(engine, "a note", type="Neuralese<string>")
    forced = ["Sure: ", {"neuralese": "write"}, " done"]
    messages = [{"role": "user", "content": [{"type": "neuralese", "id": hint.id}, {"type": "text", "text": " go"}]}]
    response = engine.generate(GenerationRequest(messages=messages, forced=forced, neuralese_temperature=0.5, seed=2))
    target = response["choices"][0]["message"]
    written = response["neuralese"]["blocks"][0]
    session = GradSession(engine)
    ll = session.run({"arguments": [hint.id], "terms": [{"kind": "logLikelihood", "messages": messages,
                                                          "target": target}]})
    # Continuous densities can exceed 1, so the negative log-likelihood may be negative.
    assert math.isfinite(ll["loss"]) and ll["gradients"][hint.id]
    sd = session.run({"arguments": [hint.id], "terms": [{
        "kind": "selfDistill", "messages": messages, "target": {"role": "assistant", "content": "Paris"},
        "teacher_messages": [{"role": "user", "content": "a note go"}]}]})
    assert sd["loss"] >= 0
    producer = written["producer"]
    kl = session.run({"arguments": [producer["mean"]], "terms": [{"kind": "klPrior", "blocks": [
        {"mean": producer["mean"], "log_sigma": producer["log_sigma"]}]}]})
    assert kl["loss"] > 0
    with pytest.raises(Exception):
        session.run({"order": 2, "arguments": [hint.id], "terms": [{"kind": "klPrior", "blocks": []}]})
