"""Historical learned delta projection tests; projected adapter decoders are retired."""

import torch

DIALECT = "nd:natlang@1"


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
