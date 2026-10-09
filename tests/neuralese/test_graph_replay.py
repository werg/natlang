"""Whole-program replay (train/graph_replay.py) and the Python standard library (stdlib.py) on LFM2.5-350M with
untrained heads: a downstream loss reaches an upstream value through the producer chain, teacher forcing keeps the
recorded observations, controls propagate changed values, the bank trains, and libraries round-trip."""

import json

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


def _text_block(engine, text):
    from natlang_neuralese.serve.grad import embed_text

    return embed_text(engine, text, "Neuralese<string>").id


def _program(engine, body, written, answer="Paris", rid="r0", split=None):
    """Call A reads the soft body and writes `written`; call B reads `written` and answers."""
    producer = {"messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Instructions: "}, {"type": "neuralese", "id": body},
                    {"type": "text", "text": " Topic: the capital of France."}]}],
                "reply": {"role": "assistant", "content": [{"type": "text", "text": "Value: "},
                                                           {"type": "neuralese", "id": written}]}}
    term = {"kind": "crossEntropy", "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Read this and answer with the city: "}, {"type": "neuralese", "id": written}]}],
            "target": {"role": "assistant", "content": answer}}
    return {"schema": "natlang.replay-record/1", "id": rid, "arguments": [body], "terms": [term],
            "producers": [producer], **({"split": split} if split else {})}


def test_downstream_loss_reaches_the_upstream_value_through_the_producer_chain(engine):
    from natlang_neuralese.train.graph_replay import LeafBank, ProgramReplay

    body = _text_block(engine, "Name the city the topic asks about.")
    written = engine.store.put(make_block(0.02 * torch.randn(3, engine.backbone.config.hidden_size), DIALECT)).id
    record = _program(engine, body, written)
    replay = ProgramReplay(engine)
    bank = LeafBank(engine, [body])
    terms = replay.record_terms(record, bank.leaves())
    terms[0].backward()
    grad = bank.params[body].grad
    assert grad is not None and float(grad.abs().sum()) > 0, "the body sits only in call A: its gradient came via B"
    # Teacher forcing keeps the recorded observation: the loss equals the replay without any leaf.
    with torch.no_grad():
        plain = float(replay.record_terms(record, {})[0])
    assert abs(float(terms[0]) - plain) < 1e-4
    # Controls propagate: a zeroed body changes what call B reads, so the whole-program loss changes.
    replay.propagate = True
    with torch.no_grad():
        trained = float(replay.record_terms(record, bank.leaves("trained"))[0])
        zeroed = float(replay.record_terms(record, bank.leaves("zeroed"))[0])
    replay.propagate = False
    assert abs(trained - zeroed) > 1e-4


def test_the_bank_trains_and_controls_and_gate_report(engine):
    from natlang_neuralese.train.graph_replay import LeafBank, ProgramReplay, evaluate, gate, train

    bodies = [_text_block(engine, f"Name the city the topic asks about, variant {i}.") for i in range(2)]
    written = engine.store.put(make_block(0.02 * torch.randn(3, engine.backbone.config.hidden_size), DIALECT)).id
    records = [_program(engine, bodies[i % 2], written, rid=f"r{i}") for i in range(2)]
    replay = ProgramReplay(engine)
    bank = LeafBank(engine, bodies)

    before_eval = evaluate(replay, records, bank)
    assert set(before_eval) >= {"trained", "initial", "zeroed", "shuffled", "by_term"}
    assert abs(before_eval["trained"] - before_eval["initial"]) < 1e-6
    # The propagated whole-program loss is what training follows: a small step against its gradient lowers it.
    replay.propagate = True
    total = sum(replay.record_terms(r, bank.leaves())[0] for r in records)
    grads = torch.autograd.grad(total, list(bank.params.values()))
    with torch.no_grad():
        stepped = {b: p - 1e-3 * p.pow(2).mean().sqrt() * g / g.norm() for (b, p), g in zip(bank.params.items(), grads)}
        lowered = sum(float(replay.record_terms(r, stepped)[0]) for r in records)
    replay.propagate = False
    assert lowered < float(total), (lowered, float(total))
    rows = []
    train(replay, records, records, bank, steps=4, lr=1e-2, batch=2, eval_every=4, log=rows.append)
    after_eval = evaluate(replay, records, bank)
    assert after_eval["trained"] < before_eval["trained"], (before_eval, after_eval)
    assert abs(after_eval["initial"] - before_eval["initial"]) < 1e-5
    assert max(bank.drift().values()) > 0 and "held" in rows[-1]
    verdict = gate(evaluate(replay, records, bank))
    assert set(verdict["beats"]) == {"initial", "zeroed", "shuffled"} and isinstance(verdict["passed"], bool)
    assert bank.leaves("shuffled")[bodies[0]].shape == bank.params[bodies[1]].shape


def test_records_load_with_their_blocks_and_split_by_group(engine, tmp_path):
    from natlang_neuralese.serve.store import encode_block
    from natlang_neuralese.train.graph_replay import load_blocks, load_records, split_records

    block = make_block(torch.randn(2, engine.backbone.config.hidden_size), DIALECT)
    blocks = tmp_path / "records.blocks"
    blocks.mkdir()
    (blocks / f"{block.id}.safetensors").write_bytes(encode_block(block))
    rows = [{"schema": "natlang.replay-record/1", "id": f"r{i}", "group": f"g{i // 2}", "arguments": [], "terms": [],
             "producers": []} for i in range(10)]
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    records = load_records([tmp_path / "records.jsonl"])
    assert records[0]["blocks_dir"] == str(blocks)
    assert load_blocks(engine, [blocks]) == 1 and engine.store.get(block.id) is not None
    train, held = split_records(records, 0.4, seed=0)
    assert len(train) + len(held) == 10 and held
    assert not {r["group"] for r in train} & {r["group"] for r in held}


def test_text_library_builds_loads_and_rewrites(engine, tmp_path):
    from natlang_neuralese.stdlib import (NAMES, build_text_library, combinator_texts, library_bodies, load_into_store,
                                          read_blocks, rewrite_nz)

    texts = combinator_texts()
    assert list(texts) == list(NAMES) and all(t["text"] and t["type"] for t in texts.values())
    path = tmp_path / "stdlib.nz"
    bodies = build_text_library(engine, path, method="embed")
    assert library_bodies(path) == bodies
    assert all(old == new for old, new in load_into_store(engine, path).items())
    header, tensors = read_blocks(path)
    assert header["provenance"]["initialisation"] == "text-embeddings"
    assert header["exports"]["read"]["value"]["$neuralese-fn"]["body"] == bodies["read"]
    new_read = tensors[bodies["read"]] + 0.1
    mapping = rewrite_nz(path, tmp_path / "trained.nz", {bodies["read"]: new_read}, provenance={"trained_by": "test"})
    trained = library_bodies(tmp_path / "trained.nz")
    assert trained["read"] == mapping[bodies["read"]] != bodies["read"]
    assert trained["map"] == bodies["map"]
    header2, tensors2 = read_blocks(tmp_path / "trained.nz")
    assert torch.allclose(tensors2[trained["read"]], new_read)
    assert header2["blocks"][trained["read"]]["producer"]["from"] == bodies["read"]
    assert header2["provenance"]["trained_by"] == "test"
