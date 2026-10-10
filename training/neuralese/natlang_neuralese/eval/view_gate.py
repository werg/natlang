"""The view operator gate (DECISIONS 2026-10-09, one summarizer family; TRAINING_RECIPE.md "The view operator stage").

Measures, for exact weights (a heads checkpoint and its backbone), what a reader gets from blocks written at view's
template write site (`view.py`; the servers' `POST /v1/neuralese/view` path, run in process):

1. **Reconstruction fidelity per artifact type against compression ratio.** `reconstruct` records: the value is
   viewed faithfully (`view(x)`, no instructions) and read back; the value's cross-entropy per token given the
   block is compared with no source (the reader's prior) and with the full text (copying). `recovery` is
   (CE_none - CE_block) / (CE_none - CE_full); `compression` is value tokens per block vector.
2. **QA with block vs full text vs none.** `consume`/`compare` records: the target's CE given the purpose-written
   block, the full value, and nothing (`ask(view(x), q)` against its law: answering from the block as from the text).
3. **Correct vs shuffled block.** The same targets given a block written (for its own purpose) over another value
   of the same artifact type.
4. **Purpose sensitivity.** `compare` records: the target's CE given the block written for its own purpose, for a
   partner purpose over the same value, and for the purpose-free general instructions.
5. **pi's next-action loss.** Harness-bench records (held repositories): the assistant target's CE with each view
   part as a written view block (its own instructions), as the crisp preview, and as the full output.

Every check reports its numbers and a pass/fail against the thresholds below (declared in the recipe's view_gate
stage; proposals until the owner confirms them). `view_gate_passed` is true only if every check passes. The records
are the corpus's held split (`--split test`); the gate never trains.

Run (under the memory ledger), e.g. a smoke on the untrained port of LFM2.5-350M:
    python -m natlang_neuralese.eval.view_gate --corpus <view-ask corpus> --out <dir> --per-artifact 2 --harness-records 2
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

GATE_SCHEMA = "natlang.view-gate/1"
# Proposed thresholds (VIEW_CORPUS.md §6 / recipe view_gate parameters); every one is a parameter.
DEFAULTS = {
    "min_reconstruction_recovery": 0.8,   # per artifact type, at the compression the stop head chose
    "min_qa_recovery": 0.7,               # (CE_none - CE_block) / (CE_none - CE_full), per artifact type
    "min_shuffle_margin": 0.05,           # nats/token: CE(shuffled block) - CE(own block), per artifact type
    "min_purpose_margin": 0.0,            # nats/token: CE(partner-purpose block) - CE(own-purpose block), overall
    "max_next_action_delta": 0.05,        # nats/token: CE(view blocks) - CE(crisp previews) on harness-bench
}
ARTIFACTS = ("prose", "code", "html", "data", "table", "log")


def _records(corpus: Path, split: str) -> list[dict]:
    return [json.loads(line) for path in sorted(Path(corpus).glob("*.port-records.jsonl"))
            for line in path.open(encoding="utf-8") if line.strip() and json.loads(line)["split"] == split]


def select(records: list[dict], per_artifact: int, seed: int) -> dict:
    """Per artifact: reconstruct records, purposeful records (consume/compare) and compare records with a partner
    purpose over the same value; seeded."""
    rng = random.Random(seed)
    by_id = {r["id"]: r for r in records}
    out = {"reconstruct": defaultdict(list), "purposeful": defaultdict(list), "compare": defaultdict(list)}
    order = sorted(records, key=lambda r: r["id"])
    rng.shuffle(order)
    for r in order:
        artifact = r["lineage"]["notes"]["artifact"]
        if r["task"] == "reconstruct":
            if len(out["reconstruct"][artifact]) < per_artifact:
                out["reconstruct"][artifact].append(r)
            continue
        if len(out["purposeful"][artifact]) < per_artifact:
            out["purposeful"][artifact].append(r)
        partners = [by_id[i] for i in r["contrasts"]["purpose_pairs"] if i in by_id and
                    by_id[i]["writer"]["instructions"] != r["writer"]["instructions"]]
        if r["task"] == "compare" and partners and len(out["compare"][artifact]) < per_artifact:
            out["compare"][artifact].append((r, partners[0]))
    return out


class Reader:
    """In-process view writes and reader scores on one engine (serve.load_engine)."""

    def __init__(self, engine, window: int | None = None):
        from ..serve.grad import GradSession

        self.engine = engine
        self.session = GradSession(engine)
        self.window = window
        self.writes = 0

    def view(self, value: str, instructions: str | None) -> tuple[str, int]:
        """A view block of `value` (view.write_view at view's template site) and its length in vectors."""
        from ..serve.engine import GenerationRequest
        from ..view import INSTRUCTIONS, TEMPLATE, TOOLS, window_of, write_view

        engine = self.engine

        def write(messages):
            request = GenerationRequest(messages=messages, tools=TOOLS, template=dict(TEMPLATE),
                                        max_tokens=engine.max_block + 64)
            blocks = (engine.generate(request).get("neuralese") or {}).get("blocks") or []
            if not blocks:
                raise RuntimeError("a view write produced no block")
            self.writes += 1
            return blocks[0]["id"]

        window = window_of(engine, instructions, self.window)
        block, _parts = write_view(write, INSTRUCTIONS, value, instructions, engine.tokenizer, window)
        return block, int(engine.lookup(block).length)

    def score(self, messages: list[dict], target: dict, tools=None) -> tuple[float, int]:
        """(total log-probability, tokens) of the assistant `target` message after `messages`."""
        import torch

        with torch.no_grad():
            prompt, rest = self.session._target_items(messages, tools, target)
            own = [value for kind, value in rest if kind == "tok"]
            if len(own) != len(rest):
                raise ValueError("the target renders blocks; only text targets are scored")
            cache, last = self.session.decision_prefill(prompt, {})
            return float(self.session.option_logprob(cache, last, own)), len(own)

    def ce(self, messages, target, tools=None) -> float:
        logp, n = self.score(messages, target, tools)
        return -logp / max(1, n)


def _target(port: dict) -> dict:
    value = port["target"]["value"]
    return {"role": "assistant", "content": value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}


def _mean(values) -> float | None:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return sum(values) / len(values) if values else None


def _recovery(none, block, full) -> float | None:
    if None in (none, block, full) or none - full <= 1e-6:
        return None
    return (none - block) / (none - full)


def run_corpus_checks(reader: Reader, chosen: dict, log=print) -> dict:
    from ..data.view_records import consumer_messages

    report = {"reconstruction": {}, "qa": {}, "shuffle": {}, "purpose": {}}
    blocks: dict[str, tuple[str, int]] = {}

    def block_for(port, instructions, key):
        if key not in blocks:
            blocks[key] = reader.view(port["sources"][0]["text"], instructions)
        return blocks[key]

    for artifact, rows in sorted(chosen["reconstruct"].items()):
        stats = defaultdict(list)
        for port in rows:
            block, length = block_for(port, None, ("faithful", port["sources"][0]["text"]))
            target = _target(port)
            ce_block = reader.ce(consumer_messages(port, {"block": block}), target)
            ce_none = reader.ce(consumer_messages(port, {}), target)
            ce_full = reader.ce(consumer_messages(port, {"text": port["sources"][0]["text"]}), target)
            tokens = reader.score(consumer_messages(port, {}), target)[1]
            stats["ce_block"].append(ce_block)
            stats["ce_none"].append(ce_none)
            stats["ce_full"].append(ce_full)
            stats["recovery"].append(_recovery(ce_none, ce_block, ce_full))
            stats["compression"].append(tokens / max(1, length))
        report["reconstruction"][artifact] = {k: _mean(v) for k, v in stats.items()} | {"records": len(rows)}
        log(f"view gate: reconstruction {artifact} {report['reconstruction'][artifact]}")
    for artifact, rows in sorted(chosen["purposeful"].items()):
        stats = defaultdict(list)
        others = [r for a, rs in chosen["purposeful"].items() if a == artifact for r in rs]
        for k, port in enumerate(rows):
            instructions = port["writer"]["instructions"]
            block, _ = block_for(port, instructions, ("purpose", port["sources"][0]["text"], instructions))
            target = _target(port)
            ce_block = reader.ce(consumer_messages(port, {"block": block}), target)
            ce_none = reader.ce(consumer_messages(port, {}), target)
            ce_full = reader.ce(consumer_messages(port, {"text": port["sources"][0]["text"]}), target)
            stats["ce_block"].append(ce_block)
            stats["ce_none"].append(ce_none)
            stats["ce_full"].append(ce_full)
            stats["recovery"].append(_recovery(ce_none, ce_block, ce_full))
            other = next((o for o in others[k + 1:] + others[:k] if o["sources"][0]["text"] != port["sources"][0]["text"]), None)
            if other is not None:
                oi = other["writer"]["instructions"]
                shuffled, _ = block_for(other, oi, ("purpose", other["sources"][0]["text"], oi))
                stats["ce_shuffled"].append(reader.ce(consumer_messages(port, {"block": shuffled}), target))
        report["qa"][artifact] = {k: _mean(v) for k, v in stats.items() if k != "ce_shuffled"} | {"records": len(rows)}
        report["shuffle"][artifact] = {"ce_block": _mean(stats["ce_block"]), "ce_shuffled": _mean(stats["ce_shuffled"]),
                                       "margin": (None if _mean(stats["ce_shuffled"]) is None else
                                                  _mean(stats["ce_shuffled"]) - _mean(stats["ce_block"])),
                                       "records": len(stats["ce_shuffled"])}
        log(f"view gate: qa {artifact} {report['qa'][artifact]} shuffle {report['shuffle'][artifact]}")
    for artifact, pairs in sorted(chosen["compare"].items()):
        stats = defaultdict(list)
        for port, partner in pairs:
            text = port["sources"][0]["text"]
            own_i, partner_i = port["writer"]["instructions"], partner["writer"]["instructions"]
            general = port["writer"].get("instructions_general")
            target = _target(port)
            own, _ = block_for(port, own_i, ("purpose", text, own_i))
            other, _ = block_for(port, partner_i, ("purpose", text, partner_i))
            ce_own = reader.ce(consumer_messages(port, {"block": own}), target)
            ce_partner = reader.ce(consumer_messages(port, {"block": other}), target)
            stats["ce_own"].append(ce_own)
            stats["ce_partner"].append(ce_partner)
            stats["margin"].append(ce_partner - ce_own)
            if general:
                gen, _ = block_for(port, general, ("purpose", text, general))
                stats["ce_general"].append(reader.ce(consumer_messages(port, {"block": gen}), target))
        report["purpose"][artifact] = {k: _mean(v) for k, v in stats.items()} | {"records": len(pairs)}
        log(f"view gate: purpose {artifact} {report['purpose'][artifact]}")
    return report


def run_harness_check(reader: Reader, records_path: Path, pieces_path: Path, limit: int, seed: int, log=print) -> dict:
    """pi's next-action CE on held harness-bench records: view parts as written blocks, crisp previews, full outputs."""
    from ..train.trajectories import crisp_messages, handover_notes, render

    texts = {}
    with open(pieces_path) as stream:
        for line in stream:
            piece = json.loads(line)
            texts[piece["name"]] = piece["text"]
    held = []
    with open(records_path) as stream:
        for line in stream:
            r = json.loads(line)
            if r.get("split") == "test" and r.get("target") and any(
                    isinstance(m.get("content"), list) and any(p.get("type") == "view" for p in m["content"])
                    for m in r["messages"]):
                held.append(r)
    random.Random(seed).shuffle(held)
    stats = defaultdict(list)
    for r in held[:limit]:
        views, full = {}, []
        for m in r["messages"]:
            for part in m.get("content") if isinstance(m.get("content"), list) else []:
                if part.get("type") == "view":
                    views[part["name"]], _ = reader.view(part["source"], part.get("instructions") or None)
        soft = lambda name: {"type": "text", "text": texts[name]}  # noqa: E731 - crisp soft prompts
        notes = handover_notes(r)
        with_views = render(r["messages"], soft, notes, views=views)
        previews = crisp_messages(r["messages"], texts, notes)
        for m in r["messages"]:
            m2 = dict(m)
            if isinstance(m.get("content"), list):
                m2["content"] = [dict(p, preview=p["source"]) if p.get("type") == "view" else p for p in m["content"]]
            full.append(m2)
        full_text = crisp_messages(full, texts, notes)
        target = render([r["target"]], soft, notes)[0]
        stats["ce_views"].append(reader.ce(with_views, target, r.get("tools")))
        stats["ce_preview"].append(reader.ce(previews, target, r.get("tools")))
        stats["ce_full"].append(reader.ce(full_text, target, r.get("tools")))
    out = {k: _mean(v) for k, v in stats.items()} | {"records": len(stats["ce_views"])}
    if out["records"]:
        out["delta_views_minus_preview"] = out["ce_views"] - out["ce_preview"]
    log(f"view gate: harness {out}")
    return out


def verdict(report: dict, thresholds: dict) -> dict:
    checks = {}
    for artifact, row in report["reconstruction"].items():
        checks[f"reconstruction:{artifact}"] = row.get("recovery") is not None and \
            row["recovery"] >= thresholds["min_reconstruction_recovery"]
    for artifact, row in report["qa"].items():
        checks[f"qa:{artifact}"] = row.get("recovery") is not None and row["recovery"] >= thresholds["min_qa_recovery"]
    for artifact, row in report["shuffle"].items():
        checks[f"shuffle:{artifact}"] = row.get("margin") is not None and row["margin"] >= thresholds["min_shuffle_margin"]
    margins = [row["margin"] for row in report["purpose"].values() if row.get("margin") is not None]
    checks["purpose"] = bool(margins) and _mean(margins) > thresholds["min_purpose_margin"]
    harness = report.get("harness") or {}
    checks["harness_next_action"] = harness.get("records", 0) > 0 and \
        harness["delta_views_minus_preview"] <= thresholds["max_next_action_delta"]
    missing = [a for a in ARTIFACTS if a not in report["reconstruction"] or a not in report["qa"]]
    return {"checks": checks, "missing_artifacts": missing,
            "view_gate_passed": bool(checks) and all(checks.values()) and not missing}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="view-ask corpus directory (port records)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-artifact", type=int, default=32)
    parser.add_argument("--harness-records", type=int, default=32)
    parser.add_argument("--harness", type=Path, default=None, help="harness-bench records directory (records.jsonl, pieces.jsonl)")
    parser.add_argument("--heads", default=None, help="port heads checkpoint (exact weights); none: untrained heads (smoke only)")
    parser.add_argument("--base", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-block", type=int, default=None)
    parser.add_argument("--view-window", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    for name, value in DEFAULTS.items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=value)
    args = parser.parse_args(argv)
    from ..serve import load_engine

    started = time.time()
    engine = load_engine(args.base, heads_checkpoint=args.heads, device=args.device, max_block=args.max_block)
    reader = Reader(engine, args.view_window)
    records = _records(args.corpus, args.split)
    chosen = select(records, args.per_artifact, args.seed)
    report = run_corpus_checks(reader, chosen, log=lambda m: print(m, flush=True))
    if args.harness and args.harness_records:
        report["harness"] = run_harness_check(reader, args.harness / "records.jsonl", args.harness / "pieces.jsonl",
                                              args.harness_records, args.seed, log=lambda m: print(m, flush=True))
    thresholds = {name: getattr(args, name) for name in DEFAULTS}
    result = {"schema": GATE_SCHEMA, "corpus": str(args.corpus), "split": args.split, "heads": args.heads,
              "base": args.base, "trained": args.heads is not None, "thresholds": thresholds,
              "per_artifact": args.per_artifact, "view_writes": reader.writes,
              "seconds": round(time.time() - started), **report, **verdict(report, thresholds)}
    if args.heads is None:
        result["view_gate_passed"] = False
        result["note"] = "untrained heads: a smoke run, never a qualification"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "view-gate.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("view_gate_passed", "checks", "missing_artifacts", "view_writes", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
