"""The recurrence of calling a function, retrieving its value and splicing it into the caller's trajectory, trained
across one run's chunks (train.trajectories, child results): a caller's loss reaches the child's write and, one level
further, the grandchild's."""

import json

import pytest
import torch

from natlang_neuralese.train.trajectories import main, write_site

GRAND = "The fee on line 2 matches clause 4.2 of the supply contract."
CHILD = "Line 2 is a contractual fee, not a disputed charge."
RETURN_TOOLS = [{"type": "function", "function": {"name": "return_result", "parameters": {
    "type": "object", "properties": {"status": {"type": "string"}, "value": {"type": "string"}},
    "required": ["status", "value"]}}}]


def opening(name, text, body_name=None):
    return {"role": "user", "content": [{"type": "text", "text": f"You are inside this call: {name}(): string\n\nInstructions:\n"},
                                        {"type": "soft", "name": f"instructions@{body_name or name}"},
                                        {"type": "text", "text": "\n\nIn eval you can use nothing."}]}


def result(name, value):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"r-{name}", "type": "function", "function": {
        "name": "return_result", "arguments": json.dumps({"status": "success", "value": {"$write": {
            "name": f"result:{name}", "type": "Neuralese<string>", "source": value}}})}}]}


def output(call_id, name, value):
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {
        "name": "eval", "arguments": json.dumps({"code": "console.log(await nl`Look it up.`())"})}}]},
            {"role": "tool", "tool_call_id": call_id, "content": [{"type": "text", "text": "console:\n"},
                                                                  {"type": "read", "name": f"result:{name}", "source": value}]}]


RECORDS = [
    # Grandchild and child are producers only (held out); the caller is the one training record.
    {"id": "grand", "split": "test", "source_groups": ["grand-facts"],
     "messages": [{"role": "system", "content": "You run one call."}, opening("grand", "")],
     "tools": RETURN_TOOLS, "target": result("grand", GRAND)},
    {"id": "child", "split": "test", "source_groups": ["grand-facts", "child-facts"],
     "messages": [{"role": "system", "content": "You run one call."}, opening("child", ""),
                  *output("e1", "grand", GRAND)], "tools": RETURN_TOOLS,
     "target": result("child", CHILD)},
    {"id": "caller", "split": "train", "source_groups": ["grand-facts", "child-facts", "caller-facts"],
     "messages": [{"role": "system", "content": "You run one call."}, opening("caller", ""),
                  *output("e2", "child", CHILD)],
     "target": {"role": "assistant", "content": "The charge stands: it is a contractual fee."}},
]
PIECES = [{"name": f"instructions@{n}", "kind": "function-body", "text": t} for n, t in
          [("grand", "Find the clause that covers line 2."), ("child", "Classify the line item."), ("caller", "Decide whether the charge stands.")]]


@pytest.fixture
def latent_sketch_heads(tmp_path, loaded):
    """Unqualified test fixture for raw/top-state gradient mechanics, not a foundation certificate."""
    from natlang_neuralese.model.heads import PortHeads

    backbone = loaded[2]
    heads = PortHeads(backbone, cutoff=2, max_length=64, profile="latent-sketch-v2").eval()
    path = tmp_path / "latent-sketch-heads.pt"
    torch.save({
        "port_config": {"cutoff": heads.cutoff, "max_length": heads.max_length,
                        **heads.port_config()},
        "heads": heads.state_dict(),
        "control_rows": backbone.control_rows.detach().cpu(),
    }, path)
    return path


def test_write_site_is_the_template_readout_cut():
    assert write_site(RECORDS[1]) == ("return_result", {"status": "success"}, "value", "result:child")


@pytest.mark.parametrize("depth,grand_moves", [(1, False), (2, True)])
def test_a_callers_loss_trains_its_childs_write_and_nested_writes_to_the_depth(loaded, latent_sketch_heads, tmp_path, depth, grand_moves):
    records, pieces = tmp_path / "records.jsonl", tmp_path / "pieces.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    pieces.write_text("".join(json.dumps(p) + "\n" for p in PIECES))
    out = tmp_path / "out"
    assert main(["--records", str(records), "--pieces", str(pieces), "--out", str(out), "--heads", str(latent_sketch_heads), "--device", "cpu", "--steps", "1",
                 "--batch", "1", "--eval", "0", "--handover", "written", "--write-depth", str(depth), "--tokens-per-vector", "1",
                 "--writer-supervision", "native-value", "--writer-length-policy", "native-value",
                 "--stop-supervision", "generated-length", "--writer-text-weight", "0",
                 "--distill", "0", "--lr", "1e-2"]) == 0
    moved = json.loads((out / "summary.json").read_text())["relative_change"]
    assert moved["instructions@caller"] > 0, "the caller's own soft instructions train"
    assert moved["instructions@child"] > 0, "the caller's loss reaches the child's write context"
    assert (moved["instructions@grand"] > 0) == grand_moves, "nested writes train to --write-depth levels"


def test_held_out_readers_compare_their_written_values_with_another_readers(loaded, latent_sketch_heads, tmp_path):
    other = "Line 7 is a late fee under clause 9."
    records = [
        {**RECORDS[1], "split": "train"},
        {"id": "child2", "split": "train", "source_groups": ["child2-facts"],
         "messages": [{"role": "system", "content": "You run one call."}, opening("child2", "", body_name="child")],
         "tools": RETURN_TOOLS, "target": result("child2", other)},
        {**RECORDS[2], "split": "test"},
        {"id": "caller2", "split": "test", "source_groups": ["child2-facts", "caller2-facts"],
         "messages": [{"role": "system", "content": "You run one call."},
                      opening("caller", ""), *output("e3", "child2", other)],
         "target": {"role": "assistant", "content": "The late fee stands under clause 9."}},
    ]
    path, pieces = tmp_path / "records.jsonl", tmp_path / "pieces.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    pieces.write_text("".join(json.dumps(p) + "\n" for p in PIECES))
    out = tmp_path / "out"
    assert main(["--records", str(path), "--pieces", str(pieces), "--out", str(out), "--heads", str(latent_sketch_heads), "--device", "cpu", "--steps", "1",
                 "--batch", "1", "--eval", "4", "--handover", "written", "--write-depth", "1", "--tokens-per-vector", "1",
                 "--writer-supervision", "native-value", "--writer-length-policy", "native-value",
                 "--stop-supervision", "generated-length", "--writer-text-weight", "0",
                 "--distill", "0"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    for label in ("written-init", "written-trained"):
        got = summary[label]
        assert got["n"] == 2 and got["written"] > 0 and got["shuffled"] > 0 and 0 <= got["written_better"] <= 1
        assert got["written"] != got["shuffled"]


def _direct_stand_in(loaded, tmp_path, monkeypatch):
    """The shared load_direct_artifact replaced by a stand-in over raw-token heads; returns (records, pieces,
    projections) for a direct-writer end-to-end run."""
    from types import SimpleNamespace
    from natlang_neuralese.model.heads import PortHeads
    from natlang_neuralese.model.neuralese_projection import NeuraleseProjection
    from natlang_neuralese.serve import load_engine
    from natlang_neuralese.train import direct_export

    backbone = loaded[2]
    raw = PortHeads(backbone, cutoff=backbone.num_layers, max_length=64, stop_source="final", stop_position=False,
                    profile="raw-token-v1").eval()
    anchor = tmp_path / "raw-token-heads.pt"
    torch.save({"port_config": {"cutoff": raw.cutoff, "max_length": raw.max_length, **raw.port_config()},
                "heads": raw.state_dict(), "control_rows": backbone.control_rows.detach().cpu()}, anchor)
    projections = {}

    def restore(artifact, *, base_snapshot=None, device="cpu", dtype=None):
        engine = load_engine(heads_checkpoint=str(anchor), device=device)
        torch.manual_seed(0)
        projection = NeuraleseProjection(int(engine.backbone.hf.config.hidden_size), hidden=16)
        projections["neuralese"] = projection
        return SimpleNamespace(engine=engine, neuralese_projection=projection, token_choice_projection=None,
                               manifest={"schema": "test-stand-in"})
    monkeypatch.setattr(direct_export, "load_direct_artifact", restore)
    # The stand-in has no exported manifest; bind the run to a fixed stand-in parent document.
    monkeypatch.setattr(direct_export, "direct_parent_binding",
                        lambda artifact: {"schema": "test-stand-in-parent", "artifact_path": str(artifact),
                                                  "manifest": {"schema": "test-stand-in"}})
    records, pieces = tmp_path / "records.jsonl", tmp_path / "pieces.jsonl"
    records.write_text("".join(json.dumps(r) + "\n" for r in RECORDS))
    pieces.write_text("".join(json.dumps(p) + "\n" for p in PIECES))
    return records, pieces, projections


def _direct_arguments(records, pieces, out, tmp_path):
    return ["--records", str(records), "--pieces", str(pieces), "--out", str(out), "--direct-artifact",
            str(tmp_path), "--direct-base-snapshot", str(tmp_path), "--device", "cpu", "--steps", "1", "--batch", "1",
            "--eval", "0", "--handover", "written", "--write-depth", "1", "--tokens-per-vector", "1",
            "--writer-supervision", "native-value", "--writer-length-policy", "native-value",
            "--stop-supervision", "generated-length", "--writer-text-weight", "0", "--distill", "0", "--lr", "1e-2",
            "--heads-lr", "1e-2"]


def test_direct_writer_end_to_end_stage_trains_the_child_write_through_the_neuralese_projection(loaded, tmp_path,
                                                                                                monkeypatch):
    """Neuralese direct v1's end-to-end stage (plans/neuralese/NEURALESE_DIRECT_V1.md): with --direct-artifact the
    child's result is written by the shared direct writer, and the caller's loss reaches the child's write context and
    the NeuraleseProjection. The restore is the shared load_direct_artifact; here a stand-in over raw-token heads."""
    records, pieces, projections = _direct_stand_in(loaded, tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert main(_direct_arguments(records, pieces, out, tmp_path)) == 0
    moved = json.loads((out / "summary.json").read_text())["relative_change"]
    assert moved["instructions@caller"] > 0 and moved["instructions@child"] > 0
    down = projections["neuralese"].stack.blocks[-1].down.weight
    assert down.abs().sum() > 0, "the caller's loss trains the NeuraleseProjection (zero-initialised output)"
    saved = torch.load(out / "heads.pt", weights_only=False)
    assert torch.equal(saved["neuralese_projection"]["stack.blocks.1.down.weight"], down.detach().cpu())
    # Full state at the end only: never before the first update (a fresh start has nothing to protect).
    phases = [json.loads(line) for line in (out / "host-phases.jsonl").read_text().splitlines()]
    assert [p["completed_updates"] for p in phases if p.get("event") == "checkpoint_saved"] == [1]


def test_a_smoke_without_saved_state_writes_logs_and_report_only(loaded, tmp_path, monkeypatch):
    records, pieces, _ = _direct_stand_in(loaded, tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert main(_direct_arguments(records, pieces, out, tmp_path) + ["--no-save-state"]) == 0
    assert (out / "summary.json").exists() and (out / "train.jsonl").exists()
    for written in ("checkpoint.pt", "best-checkpoint.pt", "heads.pt", "adapter.pt"):
        assert not (out / written).exists(), written


def test_direct_end_to_end_folds_crisp_sft_into_the_routed_backward(loaded, tmp_path, monkeypatch):
    """The crisp SFT term is one more term of the update's routed backward (the joint attempt, or the staged path
    after a joint miss), not a separate backward after the gradients exist; under the auto policy its memory is part
    of the routed estimate."""
    records, pieces, _ = _direct_stand_in(loaded, tmp_path, monkeypatch)
    out = tmp_path / "out"
    assert main(_direct_arguments(records, pieces, out, tmp_path) + [
        "--no-save-state", "--crisp-weight", "1", "--backward-policy", "auto"]) == 0
    row = json.loads((out / "train.jsonl").read_text().splitlines()[-1])
    assert row["crisp_sft_loss"] > 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["relative_change"]["instructions@child"] > 0


def test_direct_end_to_end_trains_the_child_context_under_the_in_backward_latent_step(loaded, tmp_path, monkeypatch):
    """--latent-step-in-backward on the direct E2E stage: the caller reads the child's value (made from the writer's
    top layer) at its first layer. Before the layer staging resolved such cross-stream edges, the child context got
    no gradient (relative change 0.0 against 0.065 with the separate backward); now it moves like it."""
    if not hasattr(torch.optim, "Muon"):
        pytest.skip("needs torch.optim.Muon")
    moved = {}
    for label, extra in (("separate", []), ("in-backward", ["--latent-step-in-backward"])):
        (tmp_path / label).mkdir()
        records, pieces, _ = _direct_stand_in(loaded, tmp_path / label, monkeypatch)
        out = tmp_path / label / "out"
        assert main(_direct_arguments(records, pieces, out, tmp_path / label) + [
            "--no-save-state", "--backbone-training", "latent", "--optimizer", "muon", "--latent-optimizer", "muonsr",
            "--latent-lr", "1e-3", "--backward-policy", "joint", *extra]) == 0
        moved[label] = json.loads((out / "summary.json").read_text())["relative_change"]
    assert moved["in-backward"]["instructions@child"] > 0
    assert moved["in-backward"]["instructions@child"] == pytest.approx(moved["separate"]["instructions@child"], rel=0.05)
    assert moved["in-backward"]["instructions@caller"] == pytest.approx(moved["separate"]["instructions@caller"], rel=0.05)


@pytest.mark.parametrize("feedback", ["token", "value"])
def test_direct_writer_feeds_back_tokens_so_a_consumer_reaches_each_value_at_depth_one(loaded, feedback):
    """Neuralese direct v1: inside a write the backbone reads token embeddings, never the emitted values, so the
    gradient from a later value to an earlier one is zero (depth 1 per producer->consumer boundary). With
    feedback="value" (the recurrence stage) it is not."""
    from types import SimpleNamespace
    from natlang_neuralese.model.neuralese_projection import NeuraleseProjection
    from natlang_neuralese.serve.direct_write import write_direct
    from natlang_neuralese.train.execution import prefill_write_context

    model, tokenizer, backbone = loaded
    heads = SimpleNamespace(cutoff=backbone.num_layers, read_markers=False)
    torch.manual_seed(0)
    projection = NeuraleseProjection(int(model.config.hidden_size), hidden=8)
    with torch.no_grad():
        for block in projection.stack.blocks:
            block.down.weight.normal_(0, 0.05)
    context = backbone.embed(torch.tensor([tokenizer.encode("The parcel is ready.", add_special_tokens=False)]))
    pre = prefill_write_context(backbone, heads, context)
    from natlang_neuralese.write import Opened
    opened = Opened(pre.cache, pre.state, backbone.logits(pre.top[:, None])[:, -1], pre.top)
    stop = {tokenizer.eos_token_id}
    forced = tokenizer.encode(" It is", add_special_tokens=False)[:3]
    writer = write_direct(backbone, projection, opened, 3, stop, length=3, feedback=feedback,
                          tokens=forced if feedback == "token" else None)
    first, last = writer.payloads[0], writer.payloads[-1]
    gradient, = torch.autograd.grad(last.float().sum(), first, allow_unused=True)
    if feedback == "token":
        assert gradient is None or float(gradient.abs().sum()) == 0
        # The consumer still reaches the producer's projection through every emitted value.
        down = projection.stack.blocks[-1].down.weight
        assert float(torch.autograd.grad(last.float().sum(), down)[0].abs().sum()) > 0
    else:
        assert gradient is not None and float(gradient.abs().sum()) > 0


def test_forced_sequence_write_matches_the_step_by_step_write(loaded):
    """The E2E stage writes known-token values in one causal sequence pass (DirectWriter.forced_sequence); in FP32 it
    is the step-by-step token-feedback write, values and gradients."""
    from natlang_neuralese.model.neuralese_projection import NeuraleseProjection
    from natlang_neuralese.serve.direct_write import DirectWriter, write_direct
    from natlang_neuralese.train.execution import prefill_write_context
    from natlang_neuralese.write import Opened
    from types import SimpleNamespace

    model, tokenizer, backbone = loaded
    heads = SimpleNamespace(cutoff=backbone.num_layers, read_markers=False)
    torch.manual_seed(0)
    projection = NeuraleseProjection(int(model.config.hidden_size), hidden=8)
    with torch.no_grad():
        for block in projection.stack.blocks:
            block.down.weight.normal_(0, 0.05)
            block.taps.normal_(0, 0.5)
    context = backbone.embed(torch.tensor([tokenizer.encode("The parcel is ready.", add_special_tokens=False)]))
    forced = tokenizer.encode(" It arrived on Monday at noon.", add_special_tokens=False)
    stop = {tokenizer.eos_token_id}

    def opened():
        pre = prefill_write_context(backbone, heads, context)
        return Opened(pre.cache, pre.state, backbone.logits(pre.top[:, None])[:, -1], pre.top)
    stepwise = write_direct(backbone, projection, opened(), len(forced), stop, length=len(forced), tokens=forced)
    sequence = DirectWriter(backbone, projection, opened(), len(forced), stop, length=len(forced),
                            tokens=forced).forced_sequence()
    a, b = stepwise.complete()[0], sequence.complete()[0]
    assert a.shape == b.shape and torch.allclose(a, b, atol=1e-4, rtol=1e-4)
    down = projection.stack.blocks[-1].down.weight
    ga, = torch.autograd.grad(a.float().square().sum(), down)
    gb, = torch.autograd.grad(b.float().square().sum(), down)
    assert torch.allclose(ga, gb, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("forced", [True, False])
def test_direct_write_context_gets_the_same_gradient_under_the_layer_staged_backward(loaded, forced):
    """--latent-step-in-backward runs every layer on staged leaves (model/layer_staging.py). The direct writer's
    producer context must receive the same gradient as with the plain backward, through the prefill cache the writer
    attends to and through the opened top state, for the sequence (forced) and step-by-step writes."""
    from types import SimpleNamespace
    from natlang_neuralese.model import layer_staging
    from natlang_neuralese.model.neuralese_projection import NeuraleseProjection
    from natlang_neuralese.serve.direct_write import DirectWriter
    from natlang_neuralese.train.execution import prefill_write_context
    from natlang_neuralese.write import Opened

    model, tokenizer, backbone = loaded
    heads = SimpleNamespace(cutoff=backbone.num_layers, read_markers=False)
    torch.manual_seed(0)
    projection = NeuraleseProjection(int(model.config.hidden_size), hidden=8)
    with torch.no_grad():
        for block in projection.stack.blocks:
            block.down.weight.normal_(0, 0.05)
    ids = torch.tensor([tokenizer.encode("The parcel is ready for collection.", add_special_tokens=False)])
    soft = torch.nn.Parameter(backbone.embed(ids).detach().clone())
    tokens = tokenizer.encode(" It is ready", add_special_tokens=False)[:3]

    def gradient(staged):
        soft.grad = None
        tape = layer_staging.begin() if staged else None
        try:
            pre = prefill_write_context(backbone, heads, soft)
            opened = Opened(pre.cache, pre.state, backbone.logits(pre.top[:, None])[:, -1], pre.top)
            writer = DirectWriter(backbone, projection, opened, len(tokens), {tokenizer.eos_token_id},
                                  length=len(tokens), feedback="token", tokens=tokens)
            if forced:
                writer.forced_sequence()
            else:
                while not writer.step():
                    pass
            loss = torch.stack([value.float().pow(2).sum() for value in writer.payloads]).sum()
            loss.backward()
            if tape is not None:
                tape.backward()
        finally:
            if tape is not None:
                layer_staging.end()
        return soft.grad.detach().clone()

    plain, staged = gradient(False), gradient(True)
    assert float(plain.abs().sum()) > 0
    torch.testing.assert_close(staged, plain, rtol=1e-4, atol=1e-6)
