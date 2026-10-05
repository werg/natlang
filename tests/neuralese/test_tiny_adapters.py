"""Tiny weight adapters as values (model/tiny_adapters.py): spec round trip, zero identity, LoRA equivalence,
per-row application in mixed batches, gradients through grad sessions, and a tuned adapter that lowers loss."""

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


def _put(engine, spec, coefficients):
    return engine.store.put(make_block(coefficients, spec.dialect(), type="Adapter"))


def _random(engine, kind="xs", scale=0.05, seed=0, **extra):
    from natlang_neuralese.serve.grad import new_adapter
    from natlang_neuralese.model.tiny_adapters import AdapterSpec

    zero = new_adapter(engine, {"kind": kind, **extra})
    spec = AdapterSpec.parse(zero.dialect)
    generator = torch.Generator().manual_seed(seed)
    return spec, _put(engine, spec, scale * torch.randn(zero.payload.shape, generator=generator))


def _ask(text="What is the capital of France? Answer in one word."):
    return [{"role": "user", "content": text}]


def test_spec_round_trips_and_validates():
    from natlang_neuralese.model.tiny_adapters import AdapterSpec, is_adapter_dialect

    spec = AdapterSpec(base="abc123", kind="tiny", rank=4, dim=3, layers=(6, 7, 8), targets=("out",), seed=2)
    assert AdapterSpec.parse(spec.dialect()) == spec
    assert AdapterSpec.parse(spec.dialect() + "#grad") == spec and is_adapter_dialect(spec.dialect() + "#opt")
    odd = AdapterSpec(base="abc123", layers=(2, 9), targets=("ffn_up",))
    assert AdapterSpec.parse(odd.dialect()) == odd
    with pytest.raises(ValueError):
        AdapterSpec(base="x", kind="tiny", dim=0, layers=(1,))
    with pytest.raises(ValueError):
        AdapterSpec(base="x", layers=(1,), targets=("nope",))


def test_zero_adapter_is_the_base_and_a_nonzero_one_is_its_lora(engine):
    from natlang_neuralese.model.tiny_adapters import active
    from natlang_neuralese.serve.grad import new_adapter

    backbone = engine.backbone
    ids = torch.tensor([engine._tokens("The capital of France is")])
    with torch.no_grad():
        base = backbone.forward_ids(ids)["logits"]
        zero = new_adapter(engine, {"kind": "xs", "rank": 4})
        with active([engine.resolve_adapters([zero.id])]):
            same = backbone.forward_ids(ids)["logits"]
        assert torch.equal(base, same)
        spec, block = _random(engine, rank=4, seed=3)
        with active([engine.resolve_adapters([block.id])]):
            adapted = backbone.forward_ids(ids)["logits"]
        assert (adapted - base).abs().max() > 1e-3
        # The same change as merging the decoded rank-r LoRA into the weights.
        lora = engine.adapter_bank.lora(spec, block.payload)
        saved = {}
        for name, (a, b) in lora.items():
            module = _module(backbone, name)
            saved[name] = module.weight.detach().clone()
            module.weight.data += (b @ a).to(module.weight.dtype)
        try:
            merged = backbone.forward_ids(ids)["logits"]
        finally:
            for name, weight in saved.items():
                _module(backbone, name).weight.data.copy_(weight)
        assert torch.allclose(merged, adapted, atol=2e-3), (merged - adapted).abs().max()


def _module(backbone, name):
    layer, rest = name.removeprefix("model.layers.").split(".", 1)
    return backbone.layers[int(layer)].get_submodule(rest)


def test_tiny_kind_and_a_mismatched_base_is_refused(engine):
    from natlang_neuralese.model.tiny_adapters import AdapterSpec
    from natlang_neuralese.serve.chat import RequestError

    spec, block = _random(engine, kind="tiny", u=3, rank=4, seed=4)
    assert block.payload.shape[1] == 3
    other = AdapterSpec(base="000000000000", kind="xs", rank=4, layers=spec.layers, targets=spec.targets)
    foreign = _put(engine, other, torch.zeros(block.payload.shape[0], 16))
    with pytest.raises(RequestError):
        engine.resolve_adapters([foreign.id])


def test_mixed_adapter_batch_matches_each_request_alone(engine):
    from natlang_neuralese.serve.engine import GenerationRequest

    _, one = _random(engine, rank=4, scale=0.3, seed=5)
    _, two = _random(engine, kind="tiny", u=4, rank=4, scale=2.0, seed=6)
    requests = [dict(adapters=None), dict(adapters=[{"id": one.id}]), dict(adapters=[{"id": two.id, "scale": 1.5}])]
    alone = [engine.generate(GenerationRequest(messages=_ask(), max_tokens=6, **r))["choices"][0]["message"]["content"]
             for r in requests]
    engine.start()
    try:
        futures = [engine.submit(GenerationRequest(messages=_ask(), max_tokens=6, **r)) for r in requests]
        together = [f.result(timeout=300)["choices"][0]["message"]["content"] for f in futures]
    finally:
        engine.stop()
        engine._stopping, engine._thread = False, None
    assert together == alone
    assert len(set(alone)) > 1, alone  # the adapters change the output


def test_gradient_reaches_coefficients_and_adam_lowers_the_loss(engine):
    from natlang_neuralese.serve.grad import GradSession, grad_dialect, new_adapter, optim_step

    adapter = new_adapter(engine, {"kind": "xs", "rank": 4})
    target = {"role": "assistant", "content": "Lyon"}
    term = {"kind": "crossEntropy", "messages": _ask(), "target": target}
    session = GradSession(engine)
    params, state, losses = [adapter.id], None, []
    for _ in range(5):
        result = session.run({"arguments": params, "adapters": [{"id": params[0]}], "terms": [term]})
        grad = engine.store.get(result["gradients"][params[0]])
        assert grad.dialect == grad_dialect(adapter.dialect) and float(grad.payload.abs().sum()) > 0
        step = optim_step(engine, {"optimizer": "adam", "hyper": {"lr": 0.02}, "params": params,
                                   "grads": [grad.id], "state": state})
        params, state = step["params"], step["state"]
        losses.append(result["loss"])
    final = session.run({"arguments": [], "adapters": [{"id": params[0]}], "terms": [term]})["loss"]
    base = session.run({"arguments": [], "terms": [term]})["loss"]
    assert abs(base - losses[0]) < 1e-4  # a zero adapter scores as the base
    assert final < losses[0] - 0.1, (losses, final)
    # The tuned adapter serves through the decision readout too.
    from natlang_neuralese.serve.grad import decide

    options = ["Paris", "Lyon"]
    plain = decide(engine, {"messages": _ask(), "options": options})["log_probs"]
    tuned = decide(engine, {"messages": _ask(), "options": options, "adapters": [{"id": params[0]}]})["log_probs"]
    assert tuned[1] - tuned[0] > plain[1] - plain[0]


def test_crossing_measures_how_a_block_written_under_an_adapter_reads_outside_it(engine):
    from natlang_neuralese.serve.crossing import crossing
    from natlang_neuralese.serve.grad import new_adapter

    write = [{"role": "user", "content": "Remember that the meeting moved to Lyon."}]
    read = [{"role": "user", "content": [{"type": "text", "text": "Note: "}, {"type": "neuralese", "id": "$block"},
                                         {"type": "text", "text": " Where is the meeting? Reply with a JSON value."}]}]
    options = ['"Paris"', '"Lyon"', '"Rome"']
    zero = new_adapter(engine, {"kind": "xs", "rank": 4})
    same = crossing(engine, [{"id": zero.id}], write, read, options)
    assert same["blocks"]["adapted"] == same["blocks"]["plain"] and same["crossing_kl"] < 1e-9
    _, strong = _random(engine, rank=4, scale=0.5, seed=8)
    # Untrained heads: the payload mean is the sketch (the content projection starts at zero), and the sketch runs
    # below the cutoff where the adapter does not act, so the block is the plain one and crossing = reference.
    measured = crossing(engine, [{"id": strong.id}], write, read, options)
    assert measured["blocks"]["adapted"] == measured["blocks"]["plain"]
    assert measured["crossing_kl"] > 0 and abs(measured["crossing_kl"] - measured["reference_kl"]) < 1e-9
    # With a content projection that reads the final states (as trained heads do), the adapter changes the block.
    proj = engine.heads.content.proj
    saved = proj.weight.detach().clone()
    try:
        with torch.no_grad():
            proj.weight.copy_(0.05 * torch.randn(proj.weight.shape, generator=torch.Generator().manual_seed(3)))
        written = crossing(engine, [{"id": strong.id}], write, read, options)
    finally:
        with torch.no_grad():
            proj.weight.copy_(saved)
    assert written["blocks"]["adapted"] != written["blocks"]["plain"]
    assert math.isfinite(written["crossing_kl"]) and math.isfinite(written["reference_kl"])


def test_adapter2_bases_have_fixed_signs_so_an_adapter_means_the_same_on_every_device(engine):
    """Singular vectors come back with arbitrary signs (CPU and CUDA disagree); adapter/2 fixes them, so ΔW from the
    same coefficients is the same whichever SVD produced the bases. adapter/1 keeps the legacy bases."""
    from natlang_neuralese.model.tiny_adapters import AdapterBank, AdapterSpec

    bank = AdapterBank.of(engine.backbone)
    assert AdapterBank.of(engine.backbone) is bank and engine.adapter_bank is bank
    spec = bank.spec(kind="xs", rank=4, cutoff=6)
    assert spec.dialect().startswith("adapter/2;") and AdapterSpec.parse(spec.dialect()) == spec
    legacy = AdapterSpec.parse(spec.dialect().replace("adapter/2;", "adapter/1;", 1))
    assert legacy.version == 1 and legacy.dialect().startswith("adapter/1;")
    layer, name, module = bank.matrices(spec)[0]
    u_r, v_r = bank.bases(layer, name, module, 4)
    assert bool((u_r.gather(0, u_r.abs().argmax(0, keepdim=True)) > 0).all())
    # The same matrix with its singular vectors flipped (as another device's SVD may return them): same bases.
    weight = module.weight.detach().float()
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    flip = torch.tensor([1.0, -1.0, -1.0, 1.0])
    core = torch.randn(4, 4)
    flipped_u, flipped_v = u[:, :4] * flip, vh[:4].t() * flip
    signs = torch.sign(flipped_u.gather(0, flipped_u.abs().argmax(0, keepdim=True)))
    assert torch.allclose(flipped_u * signs, u_r, atol=1e-5) and torch.allclose(flipped_v * signs, v_r, atol=1e-5)
    assert torch.allclose((flipped_u * signs) @ core @ (flipped_v * signs).t(), u_r @ core @ v_r.t(), atol=1e-5)
