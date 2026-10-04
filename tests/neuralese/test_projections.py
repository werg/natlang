"""Projections from Neuralese into adapters (model/projections.py; LEARNING_CONTINUUM §6.4, §13a)."""

import pytest
import torch

from natlang_neuralese.model.projections import AdapterProjection, BlockProjection
from natlang_neuralese.serve.store import TensorStore, make_block

DIALECT = "nd:natlang@1"


def test_block_projection_shapes_zero_block_and_untrained_identity():
    torch.manual_seed(0)
    p = BlockProjection(dim=16, rows=5, width=3, hidden=8)
    for length in (1, 4, 9):
        assert p(torch.randn(length, 16)).shape == (5, 3)
    assert p(torch.randn(2, 4, 16)).shape == (2, 5, 3)
    assert torch.equal(p(torch.randn(4, 16)), torch.zeros(5, 3)), "an untrained projection decodes to zero"
    with torch.no_grad():
        p.out.weight.normal_()
    assert torch.equal(p(torch.zeros(4, 16)), torch.zeros(5, 3)), "a zero block decodes to zero"
    assert torch.equal(p(torch.zeros(0, 16)), torch.zeros(5, 3))
    assert p(torch.randn(4, 16)).abs().sum() > 0


def test_save_load_checks_identity(tmp_path):
    p = AdapterProjection(DIALECT, "adapter/1;base=x;kind=xs;r=2;u=0;layers=6-6;targets=out;seed=0", 16, 1, 4, hidden=8)
    p.save(tmp_path / "p.pt")
    assert AdapterProjection.load(tmp_path / "p.pt").identity() == p.identity()
    saved = torch.load(tmp_path / "p.pt", weights_only=False)
    saved["state"]["project.out.weight"] += 1
    torch.save(saved, tmp_path / "tampered.pt")
    with pytest.raises(ValueError):
        AdapterProjection.load(tmp_path / "tampered.pt")


@pytest.fixture(scope="module")
def engine(loaded):
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.serve.engine import Engine

    _, tokenizer, backbone = loaded
    torch.manual_seed(1)
    heads = PortHeads(backbone, cutoff=6, max_length=8).eval()
    for q in heads.parameters():
        q.requires_grad_(False)
    return Engine(backbone, heads, tokenizer, TensorStore(), DIALECT, max_block=4)


def test_codes_decode_into_adapters_and_train_through_the_projection(engine):
    from natlang_neuralese.serve.chat import RequestError
    from natlang_neuralese.serve.grad import GradSession, decide, embed_text

    bank = engine.adapter_bank
    spec = bank.spec(kind="xs", rank=4, cutoff=engine.heads.cutoff)
    rows = len(bank.matrices(spec))
    projection = AdapterProjection(DIALECT, spec.dialect(), engine.width, rows, spec.width, hidden=32)
    engine.projections["p"] = projection
    code = embed_text(engine, "prefer Lyon", type="Neuralese<string>")
    ask = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    options = ["Paris", "Lyon"]
    bound = [{"code": code.id, "projection": "p"}]
    plain = decide(engine, {"messages": ask, "options": options})["log_probs"]
    assert decide(engine, {"messages": ask, "options": options, "adapters": bound})["log_probs"] == plain, \
        "an untrained projection decodes the identity update"
    with torch.no_grad():
        projection.project.out.weight.normal_(0, 0.05)
    changed = decide(engine, {"messages": ask, "options": options, "adapters": bound})["log_probs"]
    assert changed != plain
    # The code is a soft value: a gradient session reaches it through the projection.
    term = {"kind": "crossEntropy", "messages": ask, "target": {"role": "assistant", "content": "Lyon"}}
    result = GradSession(engine).run({"arguments": [code.id], "adapters": bound, "terms": [term]})
    grad = engine.store.get(result["gradients"][code.id])
    assert float(grad.payload.abs().sum()) > 0
    with pytest.raises(RequestError):
        decide(engine, {"messages": ask, "options": options, "adapters": [{"code": code.id, "projection": "missing"}]})
    other = engine.store.put(make_block(torch.randn(2, engine.width), "other@1"))
    with pytest.raises(RequestError):
        decide(engine, {"messages": ask, "options": options, "adapters": [{"code": other.id, "projection": "p"}]})
    del engine.projections["p"]
