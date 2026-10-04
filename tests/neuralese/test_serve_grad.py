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


def test_decision_readout_scores_options_once_and_trains_with_a_proper_rule(engine):
    from natlang_neuralese.serve.grad import GradSession, decide, embed_text

    hint = embed_text(engine, "geography quiz", type="Neuralese<string>")
    messages = term("crossEntropy", hint.id, "Paris")["messages"]
    options = ['"Paris"', '"Lyon"', '"Rome"']
    scored = decide(engine, {"messages": messages, "options": options})
    assert len(scored["log_probs"]) == 3 and all(v < 0 for v in scored["log_probs"])
    # Each option's score is its teacher-forced reply log-probability past the tokens all options share.
    session = GradSession(engine)
    for option, value, count in zip(options, scored["log_probs"], scored["tokens"]):
        loss = session.run({"arguments": [], "terms": [term("crossEntropy", hint.id, option)]})["loss"]
        prompt, rest = session._target_items(messages, None, {"role": "assistant", "content": option})
        with torch.no_grad():
            full = session._score(prompt, rest, {}, write_terms=False)["token_logp"]
        assert abs(float(full[len(rest) - count:].sum()) - value) < 1e-3
        assert abs(-float(full.mean()) - loss) < 1e-3
    # A decision term moves the soft argument toward the target distribution.
    goal = {"probabilities": [0.0, 1.0, 0.0]}
    before = session.run({"arguments": [hint.id], "terms": [dict(kind="decision", messages=messages, options=options,
                                                                  target=goal)]})
    grad = engine.store.get(before["gradients"][hint.id]).payload
    stepped = engine.store.put(make_block(engine.lookup(hint.id).payload - 0.5 * grad / grad.norm(), DIALECT))
    moved = [dict(m) for m in messages]
    moved[0] = {"role": "user", "content": [p if p.get("type") != "neuralese" else {"type": "neuralese", "id": stepped.id}
                                            for p in messages[0]["content"]]}
    after = session.run({"arguments": [], "terms": [dict(kind="decision", messages=moved, options=options, target=goal)]})
    assert after["loss"] < before["loss"]
    for rule in ("brier", "rps"):
        value = session.run({"arguments": [], "terms": [dict(kind="decision", messages=messages, options=options,
                                                              target=goal, rule=rule)]})["loss"]
        assert 0 <= value <= 2
