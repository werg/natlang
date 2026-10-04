"""Trajectory training on Neuralese-converted natlang records (DECISIONS.md 40, 41; S5 §3, first step).

Input: records written by ts-host/scripts/neuralese-convert-trajectories.mjs and its pieces file. Every `soft` part
(the runtime's prompt pieces, older system-prompt versions, program guidance) is a trainable soft parameter,
initialised from its text: from a system-prompt bank where one is given and the piece is in it (`prompt:<id>`),
otherwise encoded from the text in one pass through the port (`encode_text`). Each training step teacher-forces the record's target turn (cross-entropy on
its tokens) with the soft parameters as gradient leaves, optionally with a LoRA on the backbone. The trained
parameters are saved as a bank of the current runtime's pieces (`system-prompts.nz`) and as all soft parameters by
name (`soft-params.pt`).

Handover notes (`--handover`): `crisp` renders a note's `$write` and the pinned note's `read` as the crisp note (the
write's source). `written` has the model write them: each note's block is written once by the model from its
producing record (the record whose target is the `compact_history` call; crisp context, the block written at the
note argument's position, `<|tool_call_start|>[compact_history(note="` then the write), and every record that reads
or shows the note gets that block. Records that read a written note add a self-distillation term (weight
`--distill`) from the same model given the crisp note. The writer is not trained through its readers here; that is
the graph-level replay of S5 §3.2.

Evaluation on held-out records (split `test`): mean target cross-entropy with the system prompt as crisp text, with the
soft parameters as initialised, and as trained.

Usage: python -m natlang_neuralese.train.trajectories --records converted.jsonl --pieces pieces.jsonl --out DIR
         [--heads CKPT --bank system-prompts.nz --steps 500 --batch 4 --lr 1e-3 --rank 0 --lora-lr 2e-4
          --max-tokens 6144 --train 2000 --eval 64]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch


def crisp_messages(messages: list[dict], texts: dict[str, str], notes: dict[str, str]) -> list[dict]:
    """Messages with every soft part as its text and every handover as its note: the crisp rendering."""
    return render(messages, lambda name: {"type": "text", "text": texts[name]}, notes)


def render(messages: list[dict], soft_part, notes: dict[str, str], blocks: dict[str, str] | None = None) -> list[dict]:
    """Converted messages → engine messages: `soft` parts via `soft_part(name)`; handover reads and writes as the
    written block where `blocks` has one (name → block ID), else as the crisp note; parts merged into text where no
    block remains."""
    blocks = blocks or {}
    out = []
    for message in messages:
        message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "soft":
                    parts.append(soft_part(part["name"]))
                elif part["type"] == "digest":
                    # Until the digest operator writes them (decision 43), a digest shows the listing's crisp preview.
                    parts.append({"type": "text", "text": part["preview"]})
                elif part["type"] == "read":
                    name = part["name"]
                    parts.append({"type": "neuralese", "id": blocks[name]} if name in blocks else {"type": "text", "text": notes[name]})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            message["content"] = ("".join(p["text"] for p in parts) if all(p["type"] == "text" for p in parts) else parts)
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                args = call["function"]["arguments"]
                if '"$write"' in args:
                    value = json.loads(args)
                    written = {k: v["$write"]["name"] for k, v in value.items() if isinstance(v, dict) and "$write" in v and v["$write"]["name"] in blocks}
                    value = {k: (v["$write"]["source"] if isinstance(v, dict) and "$write" in v else v) for k, v in value.items()}
                    if written:
                        # The block inside the argument's quoted string (decision 25): JSON text around a block part.
                        (key, name), = written.items()
                        rest = {k: v for k, v in value.items() if k != key}
                        tail = json.dumps(rest, ensure_ascii=False)[1:-1]
                        arguments = [{"type": "text", "text": "{" + json.dumps(key) + ': "'}, {"type": "neuralese", "id": blocks[name]},
                                     {"type": "text", "text": '"' + (", " + tail if tail else "") + "}"}]
                    else:
                        arguments = json.dumps(value, ensure_ascii=False)
                    call = {**call, "function": {**call["function"], "arguments": arguments}}
                calls.append(call)
            message["tool_calls"] = calls
        out.append(message)
    return out


def reads(record: dict) -> set[str]:
    return {part["name"] for m in record["messages"] if isinstance(m.get("content"), list)
            for part in m["content"] if part["type"] == "read"}


def target_write(record: dict) -> str | None:
    """The handover name a record's target writes, if its target is a compaction call."""
    for call in (record.get("target") or {}).get("tool_calls") or []:
        if '"$write"' in call["function"]["arguments"]:
            for value in json.loads(call["function"]["arguments"]).values():
                if isinstance(value, dict) and "$write" in value:
                    return value["$write"]["name"]
    return None


def handover_notes(record: dict) -> dict[str, str]:
    notes = {}
    for message in record["messages"] + ([record["target"]] if record.get("target") else []):
        for call in message.get("tool_calls") or []:
            if '"$write"' in call["function"]["arguments"]:
                for value in json.loads(call["function"]["arguments"]).values():
                    if isinstance(value, dict) and "$write" in value:
                        notes[value["$write"]["name"]] = value["$write"]["source"]
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", required=True)
    parser.add_argument("--pieces", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--heads", default=None, help="port heads checkpoint (soft parameters are read through them)")
    parser.add_argument("--base", default=None)
    parser.add_argument("--bank", default=None, help="system-prompt bank to initialise current pieces from")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3, help="soft parameters")
    parser.add_argument("--rank", type=int, default=0, help="LoRA rank (0: soft parameters only)")
    parser.add_argument("--lora-lr", type=float, default=2e-4)
    parser.add_argument("--max-tokens", type=int, default=6144, help="skip records whose crisp prompt is longer")
    parser.add_argument("--train", type=int, default=2000, help="training records to read")
    parser.add_argument("--eval", type=int, default=64, help="held-out records")
    parser.add_argument("--handover", choices=["crisp", "written"], default="crisp")
    parser.add_argument("--only-handover", action="store_true", help="only records that read or write a note")
    parser.add_argument("--distill", type=float, default=1.0, help="weight of the self-distillation term on written notes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=8)
    args = parser.parse_args(argv)

    from ..prompt_bank import load_bank, save_bank
    from ..serve import load_engine
    from ..serve.chat import RequestError, render_messages
    from ..serve.engine import GenerationRequest
    from ..serve.grad import GradSession, encode_text
    from ..serve.store import make_block
    from .adapters import inject_lora, lora_state

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    engine = load_engine(args.base, heads_checkpoint=args.heads, device=args.device)
    for p in engine.backbone.parameters():
        p.requires_grad_(False)
    session = GradSession(engine)

    texts = {}
    with open(args.pieces) as stream:
        for line in stream:
            piece = json.loads(line)
            texts[piece["name"]] = piece["text"]
    bank = load_bank(args.bank) if args.bank else None
    params, leaf_ids, from_bank = {}, {}, []
    for name, text in texts.items():
        piece = name.removeprefix("prompt:")
        if bank and name.startswith("prompt:") and piece in bank.rows:
            rows = bank.rows[piece]
            from_bank.append(name)
        else:
            rows = encode_text(engine, text).payload
        block = engine.store.put(make_block(rows, engine.dialect, type="Neuralese<SystemPrompt>",
                                            producer={"kind": "text-init", "text": text}))
        params[name] = torch.nn.Parameter(block.payload.clone().float().to(engine.device))
        leaf_ids[name] = block.id
    init = {name: p.detach().clone() for name, p in params.items()}

    producers, written = {}, {}
    if args.handover == "written":
        # Every record whose target writes a note, by note name: the producer of that note's block.
        with open(args.records) as stream:
            for line in stream:
                if '"$write"' in line:
                    record = json.loads(line)
                    name = target_write(record)
                    if name:
                        producers.setdefault(name, record)

    def note_block(name):
        """The note's block, written once by the model from its producer's crisp context."""
        if name not in written and name in producers:
            producer = producers[name]
            messages = crisp_messages(producer["messages"], texts, handover_notes(producer))
            response = engine.generate(GenerationRequest(messages=messages, tools=producer.get("tools"),
                                                         forced=['<|tool_call_start|>[compact_history(note="', {"neuralese": "write"}],
                                                         max_tokens=engine.max_block + 8))
            written[name] = response["neuralese"]["blocks"][0]["id"]
        return written.get(name)

    def blocks_of(record):
        if args.handover != "written":
            return {}
        names = reads(record) | set(handover_notes(record))
        return {name: block for name in names if (block := note_block(name))}

    def soft_messages(record):
        return render(record["messages"], lambda name: {"type": "neuralese", "id": leaf_ids[name]}, handover_notes(record),
                      blocks_of(record))

    def target_of(record):
        return render([record["target"]], lambda name: {"type": "text", "text": texts[name]}, handover_notes(record),
                      blocks_of(record))[0]

    def prompt_tokens(record):
        crisp = crisp_messages(record["messages"], texts, handover_notes(record))
        rendered = render_messages(crisp, record.get("tools"), engine._template, engine.specials)
        return len(session._items(rendered.segments, rendered.blocks))

    train, held, skipped = [], [], {"long": 0, "no-target": 0}
    with open(args.records) as stream:
        for line in stream:
            if len(train) >= args.train and len(held) >= args.eval:
                break
            record = json.loads(line)
            if not record.get("target"):
                skipped["no-target"] += 1
                continue
            if args.only_handover and not (reads(record) or handover_notes(record)):
                continue
            bucket = held if record.get("split") == "test" else train
            if len(bucket) >= (args.eval if bucket is held else args.train):
                continue
            if prompt_tokens(record) > args.max_tokens:
                skipped["long"] += 1
                continue
            bucket.append(record)
    handovers = sum(1 for r in train if handover_notes(r))
    if args.handover == "written":
        # Training records that read or show a note come first in the mix's accounting; their producers may lie
        # outside the read window, which is why every producer was indexed.
        handovers = sum(1 for r in train if reads(r) or handover_notes(r))
    print(json.dumps({"train": len(train), "heldout": len(held), "skipped": skipped, "soft_params": len(params),
                      "from_bank": len(from_bank), "records_with_handover": handovers,
                      "handover": args.handover, "note_producers": len(producers)}), flush=True)

    lora = []
    if args.rank:
        groups = inject_lora(engine.backbone, list(range(engine.backbone.num_layers)), rank=args.rank, alpha=2 * args.rank)
        lora = [p for ps in groups.values() for p in ps]
    optimizer = torch.optim.AdamW([{"params": list(params.values()), "lr": args.lr}] +
                                  ([{"params": lora, "lr": args.lora_lr}] if lora else []), weight_decay=0.0)

    def loss_of(record, leaves, soft=True):
        messages = soft_messages(record) if soft else crisp_messages(record["messages"], texts, handover_notes(record))
        target = target_of(record) if soft else render([record["target"]], lambda name: {"type": "text", "text": texts[name]},
                                                      handover_notes(record))[0]
        loss = session._term({"kind": "crossEntropy", "messages": messages, "tools": record.get("tools"), "target": target}, leaves)
        if soft and args.distill and blocks_of(record) and reads(record) and not target_write(record):
            # The teacher sees the crisp note where the student reads the written block.
            loss = loss + args.distill * session._term({"kind": "selfDistill", "messages": messages, "tools": record.get("tools"),
                                                        "target": target, "teacher_messages": crisp_messages(
                                                            record["messages"], texts, handover_notes(record))}, leaves)
        return loss

    def evaluate(label, leaves, soft=True):
        values = []
        with torch.no_grad():
            for record in held:
                try:
                    values.append(float(loss_of(record, leaves, soft)))
                except RequestError:
                    pass
        return {"label": label, "cross_entropy": sum(values) / max(1, len(values)), "n": len(values)}

    leaves = {leaf_ids[name]: p for name, p in params.items()}
    report = {"crisp": evaluate("crisp", {}, soft=False), "soft-init": evaluate("soft-init", leaves)}
    print(json.dumps(report), flush=True)
    log = open(out / "train.jsonl", "w")
    started, cursor, errors, used = time.time(), 0, 0, set()
    with torch.enable_grad():
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(args.batch):
                record = train[cursor % len(train)]
                cursor += 1
                try:
                    loss = loss_of(record, leaves) / args.batch
                except RequestError:
                    errors += 1
                    continue
                loss.backward()
                losses.append(float(loss.detach()) * args.batch)
                used.update(part["name"] for m in record["messages"] if isinstance(m.get("content"), list)
                            for part in m["content"] if part["type"] == "soft")
            torch.nn.utils.clip_grad_norm_(list(params.values()) + lora, 1.0)
            optimizer.step()
            if step % 10 == 0 or step == args.steps - 1:
                entry = {"step": step, "loss": sum(losses) / max(1, len(losses)), "seconds": round(time.time() - started),
                         "errors": errors}
                if args.device.startswith("cuda"):
                    entry["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps(entry), flush=True)
    report["soft-trained"] = evaluate("soft-trained", leaves)
    report["notes_written"] = len(written)
    moved = {name: float((params[name].detach() - init[name]).norm() / init[name].norm().clamp_min(1e-9)) for name in used}
    report["relative_change"] = moved
    torch.save({"params": {k: v.detach().cpu() for k, v in params.items()}, "texts": texts}, out / "soft-params.pt")
    if lora:
        torch.save({"lora": lora_state(engine.backbone), "rank": args.rank}, out / "adapter.pt")
    if bank:
        current = {name.removeprefix("prompt:"): params[name] for name in used if name.removeprefix("prompt:") in bank.texts}
        save_bank(out / "system-prompts.nz", bank, current, {"kind": "system-prompt-bank", "init": args.bank,
                  "trained_by": "natlang_neuralese.train.trajectories", "records": args.records, "steps": args.steps})
    (out / "summary.json").write_text(json.dumps({"options": vars(args), **report}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "relative_change"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
