"""Reference server: gradient replay sessions and optimiser steps."""

import json
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


def test_projection_anchor_aligns_full_and_shallow_ports_to_gold_target_tokens(engine):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.serve.grad import GradSession

    heads = PortHeads(engine.backbone, cutoff=6, max_length=8, profile="latent-sketch-v2")
    anchored_engine = Engine(engine.backbone, heads, engine.tokenizer, TensorStore(), DIALECT,
                             max_block=4, device="cpu")
    session = GradSession(anchored_engine)
    term_ = {"messages": [{"role": "user", "content": "Reply with one city name."}],
             "target": {"role": "assistant", "content": "Paris"}}

    loss = session.supervised_text_loss(term_, {}, projection_anchor_weight=0.5,
                                        projection_anchor_backbone_scale=0.05)
    assert torch.isfinite(loss)
    assert torch.isfinite(session.last_projection_anchor_loss)
    assert session.last_projection_anchor_loss > 0
    loss.backward()
    assert heads.content.proj.weight.grad is not None
    assert heads.feedback.correction.weight.grad is not None


def test_projection_anchor_applies_to_native_value_producer_replay(engine):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine
    from natlang_neuralese.serve.grad import GradSession

    heads = PortHeads(engine.backbone, cutoff=6, max_length=8, profile="latent-sketch-v2")
    anchored_engine = Engine(engine.backbone, heads, engine.tokenizer, TensorStore(), DIALECT,
                             max_block=4, device="cpu")
    session = GradSession(anchored_engine)

    loss = session.supervised_continuation_loss(
        [{"role": "user", "content": "Return one city name."}], None,
        "", "Paris", {}, text_weight=1., projection_anchor_weight=1.,
        projection_anchor_backbone_scale=0.05)
    assert torch.isfinite(loss)
    assert torch.isfinite(session.last_projection_anchor_loss)
    assert session.last_projection_anchor_loss > 0
    loss.backward()
    assert heads.content.proj.weight.grad is not None
    assert heads.feedback.correction.weight.grad is not None


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
        session.run({"order": 3, "arguments": [hint.id], "terms": [{"kind": "klPrior", "blocks": []}]})


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


def test_per_option_backward_matches_the_whole_decision_loss(engine):
    from natlang_neuralese.serve.grad import GradSession

    session = GradSession(engine)
    messages = [{"role": "user", "content": "Which city is the capital of France? Answer with a JSON string."}]
    options = ['"Paris"', '"Lyon"', '"Rome"', '"Nice"']
    goal = torch.tensor([0.7, 0.2, 0.1, 0.0])
    weight = next(p for n, p in engine.backbone.hf.named_parameters() if n.endswith("embedding_norm.weight"))
    weight.requires_grad_(True)
    try:
        with torch.enable_grad():
            logp = torch.log_softmax(session.decision_logprobs(messages, None, options, {})[0], 0)
            whole = -(goal * logp).sum()
            (expected,) = torch.autograd.grad(whole, [weight])
            weight.grad = None
            loss = session.decision_backward(messages, None, options, goal)
        assert abs(loss - float(whole)) < 1e-4
        assert torch.allclose(weight.grad, expected, atol=1e-5, rtol=1e-3)
    finally:
        weight.requires_grad_(False)
        weight.grad = None


def test_encode_is_one_pass_through_the_port_one_vector_per_token(engine):
    from natlang_neuralese.serve.grad import encode_text

    text = "Refunds are allowed within 30 days of delivery."
    ids = engine.tokenizer(text, add_special_tokens=False)["input_ids"]
    block = encode_text(engine, text, "Neuralese<string>")
    assert block.payload.shape == (len(ids), engine.width) and block.producer["kind"] == "text-encode"
    heads, backbone = engine.heads, engine.backbone
    with torch.no_grad():
        supplied = heads.interface(backbone.embed(torch.tensor([ids], device=engine.device)))[0].float().cpu()
    content = heads.content.proj
    if float(content.weight.abs().max()) == 0:
        # Untrained content projection: the payload is the supplied sketch itself.
        assert torch.allclose(block.payload, supplied, atol=1e-4)
    saved = content.weight.detach().clone()
    try:
        with torch.no_grad():
            content.weight.normal_(0, 0.02)
        plain = encode_text(engine, text).payload
        framed = encode_text(engine, text, context=[{"role": "user", "content": "Keep only what matters for refund decisions."}]).payload
        assert not torch.allclose(plain, supplied, atol=1e-4), "the output port shapes the payload"
        assert not torch.allclose(plain, framed), "the write site's context reaches the payload"
    finally:
        with torch.no_grad():
            content.weight.copy_(saved)


def test_a_written_block_read_by_another_turn_carries_gradient_to_its_producers_context(engine):
    """Replay through writes (spec/NEURALESE_GRAPH.md, step 4): a call writes a block from a context holding the
    argument; a later turn reads the block and is scored. With the producing turn given, the gradient reaches the
    argument through the write; the block keeps its recorded value, so the loss is the same either way."""
    from natlang_neuralese.serve.engine import GenerationRequest
    from natlang_neuralese.serve.grad import GradSession, embed_text

    instructions = embed_text(engine, "Summarise the city in one word.", type="Neuralese<string>")
    child = [{"role": "user", "content": [{"type": "neuralese", "id": instructions.id}, {"type": "text", "text": " Paris, France"}]}]
    response = engine.generate(GenerationRequest(messages=child, forced=["Result: ", {"neuralese": "write"}], seed=3))
    reply = response["choices"][0]["message"]
    written = response["neuralese"]["blocks"][0]["id"]
    caller = {"kind": "crossEntropy", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "The helper said: "}, {"type": "neuralese", "id": written},
        {"type": "text", "text": "\nWhich city? One word."}]}], "target": {"role": "assistant", "content": "Paris"}}
    session = GradSession(engine)
    alone = session.run({"arguments": [instructions.id], "terms": [caller]})
    chained = session.run({"arguments": [instructions.id], "terms": [caller],
                           "producers": [{"messages": child, "reply": reply}]})
    assert float(engine.store.get(alone["gradients"][instructions.id]).payload.abs().sum()) == 0
    assert float(engine.store.get(chained["gradients"][instructions.id]).payload.abs().sum()) > 0
    assert chained["loss"] == pytest.approx(alone["loss"], abs=1e-5)
    # A producer whose context holds no argument leaves the block a constant.
    unrelated = embed_text(engine, "unrelated", type="Neuralese<string>")
    constant = session.run({"arguments": [unrelated.id], "terms": [caller], "producers": [{"messages": child, "reply": reply}]})
    assert float(engine.store.get(constant["gradients"][unrelated.id]).payload.abs().sum()) == 0


def test_a_reply_that_returns_a_written_value_renders_it_as_a_block(engine):
    """Replies return a written call argument as a part list inside the arguments JSON; rendering it back (a recorded
    turn's reply as a grad target or producer) must keep it a block."""
    from natlang_neuralese.serve.chat import render_messages

    block = "nz1_" + "a" * 52
    reply = {"role": "assistant", "content": None, "tool_calls": [{"id": "c", "type": "function", "function": {
        "name": "return_result", "arguments": json.dumps({"status": "success", "value": [{"type": "neuralese", "id": block}]})}}]}
    rendered = render_messages([{"role": "user", "content": "x"}, reply], None,
                               lambda m, t: engine.tokenizer.apply_chat_template(m, tokenize=False), engine.specials)
    assert rendered.blocks == [block]


def test_second_order_differentiates_through_an_inner_update(engine):
    """order 2: the outer loss of h' = h − lr·∇L_in(h) is differentiated through the inner gradient (its Hessian
    term), matching a finite difference of F(h) = L_out(h'), where the first-order gradient ∇L_out(h') does not."""
    from natlang_neuralese.serve.grad import GradSession, embed_text, optim_step

    session = GradSession(engine)
    hyper = {"lr": 2.0}

    def put(payload):
        return engine.store.put(make_block(payload, DIALECT, type="Neuralese<string>")).id

    def inner_and_step(h):
        inner = session.run({"arguments": [h], "terms": [term("crossEntropy", h, "Paris")]})
        step = optim_step(engine, {"optimizer": "sgd", "hyper": hyper, "params": [h], "grads": [inner["gradients"][h]]})
        derived = [{"kind": "grad", "arguments": [h], "terms": [term("crossEntropy", h, "Paris")], "gradients": inner["gradients"]},
                   {"kind": "optim", "optimizer": "sgd", "hyper": hyper, "params": [h], "grads": [inner["gradients"][h]],
                    "results": step}]
        return step["params"][0], derived

    def outer(h_new):
        return term("crossEntropy", h_new, "France")

    h = embed_text(engine, "geography quiz", type="Neuralese<string>").id
    h_new, derived = inner_and_step(h)
    first = session.run({"arguments": [h], "terms": [outer(h_new)]})
    assert first["gradients"][h] and float(engine.store.get(first["gradients"][h]).payload.abs().sum()) == 0
    second = session.run({"arguments": [h], "order": 2, "derived": derived, "terms": [outer(h_new)]})
    assert abs(second["loss"] - session.run({"arguments": [], "terms": [outer(h_new)]})["loss"]) < 1e-4
    g2 = engine.store.get(second["gradients"][h]).payload.float()
    # First order at h' (what the first-order approximation would pass back to h).
    g1 = engine.store.get(session.run({"arguments": [h_new], "terms": [outer(h_new)]})["gradients"][h_new]).payload.float()
    direction = g2 - g1
    direction = direction / direction.norm()

    def F(payload):
        h_new, _ = inner_and_step(put(payload))
        return session.run({"arguments": [], "terms": [outer(h_new)]})["loss"]

    base = engine.store.get(h).payload.float()
    eps = 1e-2 * float(base.norm())
    numeric = (F(base + eps * direction) - F(base - eps * direction)) / (2 * eps)
    exact, approx = float((g2 * direction).sum()), float((g1 * direction).sum())
    assert abs(numeric - exact) < 0.25 * abs(exact - approx), (numeric, exact, approx)


def test_context_loss_scores_new_prompt_text_like_a_full_forward(engine):
    """The whole trajectory is a target: instructions/inputs at full weight, tool results and other mechanical feedback
    after a reply at the lower feedback weight, never the earlier reply itself (its own record supervises it) or a
    block payload; scored exactly as a plain forward over the prompt."""
    from natlang_neuralese.serve.grad import GradSession, embed_text

    hint = embed_text(engine, "geography quiz", type="Neuralese<string>")
    first = term("crossEntropy", hint.id, "Paris")["messages"]
    messages = first + [{"role": "assistant", "content": "Let me check."},
                        {"role": "tool", "content": "Paris is the capital."},
                        {"role": "user", "content": "There is no result yet. Answer in one word."}]
    target = {"role": "assistant", "content": "Paris"}
    session = GradSession(engine)
    prompt, rest = session._target_items(messages, None, target)
    weights = session._context_weights(prompt, .25)
    decode = lambda w: engine.tokenizer.decode([v for (k, v), x in zip(prompt, weights) if k == "tok" and x == w])
    assert "Let me check" in decode(0.) and "capital of France" in decode(0.)
    assert "Paris is the capital" in decode(.25) and "There is no result yet" in decode(.25)
    assert "assistant" in decode(1.) and "Paris" not in decode(1.)
    opening = session._target_items(first, None, target)[0]
    assert set(session._context_weights(opening, .25)) == {1.0}
    with torch.no_grad():
        scored = session._score(prompt, rest, {}, write_terms=False, context_weights=weights)
        assert torch.equal(scored["token_logp"], session._score(prompt, rest, {}, write_terms=False)["token_logp"])
        starts = []
        h = engine.backbone.forward_embeds(session._embed_items(prompt, {}, starts), logits=False)["h_final"]
        logp = torch.log_softmax(engine.backbone.logits(h).float()[0], -1)
        index = [j for j in range(1, len(prompt)) if weights[j] > 0]
        assert all(prompt[j][0] == "tok" for j in index)
        expected = torch.stack([logp[starts[j] - 1, prompt[j][1]] for j in index])
        w = torch.tensor([weights[j] for j in index])
    torch.testing.assert_close(scored["context_logp"], expected, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(scored["context_weight"], w)
    term_ = {"messages": messages, "target": target}
    target_only = session.supervised_text_loss(term_, {})
    whole = session.supervised_text_loss(term_, {}, context_weight=.5, feedback_weight=.25)
    torch.testing.assert_close(whole, target_only - .5 * (w * expected).mean(), atol=1e-4, rtol=1e-4)


def test_context_coverage_supervises_turns_no_earlier_record_covers(engine):
    """A corpus with only some turns of a trajectory as records (harness_bench) passes the number of leading replies an
    earlier record supervises: later assistant replies become targets at weight 1, tool results after the first reply
    stay at the feedback weight, and the covered prefix weighs 0. The default (None) keeps last-reply coverage."""
    from natlang_neuralese.serve.grad import GradSession
    from natlang_neuralese.serve.chat import RequestError

    messages = [{"role": "user", "content": "Fix the bug in parser.py."},
                {"role": "assistant", "content": "Reading the file."},
                {"role": "tool", "content": "def parse(): pass"},
                {"role": "assistant", "content": "Running the tests."},
                {"role": "tool", "content": "3 failed"}]
    target = {"role": "assistant", "content": "Patching."}
    session = GradSession(engine)
    prompt, _ = session._target_items(messages, None, target)
    weighted = lambda weights, w: engine.tokenizer.decode(
        [v for (k, v), x in zip(prompt, weights) if k == "tok" and x == w])
    legacy = session._context_weights(prompt, .25)
    assert session._context_weights(prompt, .25, None) == legacy
    assert "Running the tests" in weighted(legacy, 0.) and "def parse" in weighted(legacy, 0.)
    nothing = session._context_weights(prompt, .25, 0)
    assert "Fix the bug" in weighted(nothing, 1.) and "Reading the file" in weighted(nothing, 1.)
    assert "Running the tests" in weighted(nothing, 1.)
    assert "def parse" in weighted(nothing, .25) and "3 failed" in weighted(nothing, .25)
    first = session._context_weights(prompt, .25, 1)
    assert "Fix the bug" in weighted(first, 0.) and "Reading the file" in weighted(first, 0.)
    assert "def parse" in weighted(first, .25) and "Running the tests" in weighted(first, 1.)
    assert session._context_weights(prompt, .25, 2) == legacy
    for bad in (3, -1, 1.0):
        with pytest.raises(RequestError):
            session._context_weights(prompt, .25, bad)
