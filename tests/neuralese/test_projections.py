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


def test_stage_one_fit_reconstructs_and_reports_the_random_projection_control(tmp_path, monkeypatch):
    import json

    from natlang_neuralese.train import projection as fit

    torch.manual_seed(0)
    basis = torch.randn(3, 6, 4)  # adapters in a 3-dimensional family: what a shared code space should capture
    adapters = {f"nz1_{i}": torch.einsum("k,krw->rw", torch.randn(3), basis) for i in range(20)}
    adapters["nz1_zero"] = torch.zeros(6, 4)
    spec = "adapter/1;base=x;kind=xs;r=2;u=0;layers=6-11;targets=out;seed=0"
    monkeypatch.setattr(fit, "load_adapters", lambda paths: (spec, {k: v for k, v in adapters.items() if v.abs().max() > 0}))
    fit.main(["--adapters", "unused.nz", "--out", str(tmp_path / "fit"), "--width", "16", "--hidden", "32",
              "--steps", "400", "--code-steps", "300", "--code-length", "4"])
    summary = json.loads((tmp_path / "fit" / "summary.json").read_text())
    assert summary["adapters"] == 20 and summary["heldout"] == 4
    assert summary["train_relative_error"] < 0.05
    assert summary["heldout_relative_error"] < summary["heldout_relative_error_random_projection"]
    assert AdapterProjection.load(tmp_path / "fit" / "projection.pt").target == spec


def test_delta_projection_shapes_and_identity():
    from natlang_neuralese.model.projections import DeltaProjection

    torch.manual_seed(0)
    d = DeltaProjection(DIALECT, dim=16, hidden=8)
    base = torch.randn(5, 16)
    assert torch.equal(d(torch.randn(3, 16), base), torch.zeros(5, 16)), "an untrained D is the identity update"
    with torch.no_grad():
        d.out.weight.normal_()
    assert d(torch.randn(7, 16), base).shape == (5, 16)
    assert torch.equal(d(torch.zeros(3, 16), base), torch.zeros(5, 16)), "a zero written block is a zero delta"
    assert torch.equal(d(torch.zeros(0, 16), base), torch.zeros(5, 16))
    written = torch.randn(3, 16)
    assert torch.allclose(d(written, 2 * base), 2 * d(written, base), atol=1e-5), "deltas scale with the base"


def test_delta_fit_reads_in_place_steps_and_beats_the_random_control(tmp_path, monkeypatch):
    import json

    from natlang_neuralese.nz import NzExport
    from natlang_neuralese.train import delta_projection as fit

    torch.manual_seed(0)
    base = torch.randn(5, 16)
    basis = torch.randn(3, 5, 16)  # deltas in a 3-dimensional family: what a shared code space should capture
    blocks = {"nz1_base": base, **{f"nz1_{i}": base + torch.einsum("k,kld->ld", torch.randn(3), basis) for i in range(20)},
              "nz1_short": torch.randn(4, 16), "nz1_embedded": torch.randn(5, 16)}
    exports = {k: NzExport(k, "Neuralese<string>", k, DIALECT, v) for k, v in blocks.items()}
    monkeypatch.setattr(fit, "read_nz", lambda path: ({}, exports))
    skill = lambda i: [{"kind": "soft-skill", "id": i, "role": "skill"}]
    steps = [{"id": f"s{i}", "operator": {"kind": "refine"}, "before": skill("nz1_base"), "after": skill(f"nz1_{i}")} for i in range(20)]
    steps += [{"id": "short", "operator": {"kind": "refine"}, "before": skill("nz1_base"), "after": skill("nz1_short")},
              {"id": "embed", "operator": {"kind": "memetic:embed"}, "before": skill("nz1_base"), "after": skill("nz1_embedded")},
              {"id": "same", "operator": {"kind": "propose"}, "before": skill("nz1_base"), "after": skill("nz1_base")}]
    (tmp_path / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    assert len(fit.load_deltas([f"{tmp_path / 'steps.jsonl'}=unused.nz"])[1]) == 20, "only in-place changes are deltas"
    fit.main(["--runs", f"{tmp_path / 'steps.jsonl'}=unused.nz", "--out", str(tmp_path / "fit"), "--hidden", "32",
              "--steps", "400", "--code-steps", "300", "--code-length", "4"])
    summary = json.loads((tmp_path / "fit" / "summary.json").read_text())
    assert summary["deltas"] == 20 and summary["heldout"] == 4
    assert summary["train_relative_error"] < 0.1
    assert summary["heldout_relative_error"] < summary["heldout_relative_error_random_projection"]
