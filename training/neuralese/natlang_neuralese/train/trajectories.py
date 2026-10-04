"""Trajectory training on Neuralese-converted natlang records (DECISIONS.md 40, 41; S5 §3, first step).

Input: records written by ts-host/scripts/neuralese-convert-trajectories.mjs and its pieces file. Every `soft` part
(the runtime's prompt pieces, older system-prompt versions, program guidance) is a trainable soft parameter,
initialised from its text: from a system-prompt bank where one is given and the piece is in it (`prompt:<id>`),
otherwise encoded from the text in one pass through the port (`encode_text`). Each training step teacher-forces the record's target turn (cross-entropy on
its tokens) with the soft parameters as gradient leaves, optionally with a LoRA on the backbone. The trained
parameters are saved as a bank of the current runtime's pieces (`system-prompts.nz`) and as all soft parameters by
name (`soft-params.pt`).

Written values (`--handover written`, `--digest written`): handover notes and listing digests are written by the
model through the port's differentiable write procedure (S3 `unroll_write`), afresh at every step, and enter their
readers as gradient leaves. The readers' losses therefore train the writer: the payload carries gradients into the
content projection, the sketch recurrence, the LoRA and every soft parameter of the write site (the digest
instructions, the producing record's prompts), so a note or digest learns to hold what its readers need. A note is
written from its producing record (the record whose target is the `compact_history` call), soft-rendered, at the note
argument (the reply forced to the model's own rendering of `compact_history(note='`, chat.call_reply, then the
write: the same cut the server's template readout makes); a digest by the digest operator's plan
(digest.py), chunked when the value exceeds `--digest-window` tokens, every chunk write and the combining write
differentiable. Stop decisions are sampled and trained by a policy gradient with reward −(reader loss + λ·length)
against a running baseline (`--stop-pg λ`); without it they are detached and the length is the stop head's choice. With the crisp modes, notes are rendered as the crisp note and
digests as the listing's preview. Records that read written values add a self-distillation term (weight `--distill`)
from the same model given the crisp note and preview.

Evaluation on held-out records (split `test`): mean target cross-entropy with the system prompt as crisp text, with the
soft parameters as initialised, and as trained.

Usage: python -m natlang_neuralese.train.trajectories --records converted.jsonl --pieces pieces.jsonl --out DIR
         [--heads CKPT --bank system-prompts.nz --steps 500 --batch 4 --lr 1e-3 --rank 0 --lora-lr 2e-4
          --max-tokens 6144 --train 2000 --eval 64]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import re
import time
from pathlib import Path

import torch

from ..digest import PREFIX as DIGEST_PREFIX, digest_note, write_digest

INSTRUCTIONS = re.compile(r"Instructions:\n([\s\S]*?)\n\n(?:In eval|Eval also|$)")


def crisp_messages(messages: list[dict], texts: dict[str, str], notes: dict[str, str]) -> list[dict]:
    """Messages with every soft part as its text and every handover as its note: the crisp rendering."""
    return render(messages, lambda name: {"type": "text", "text": texts[name]}, notes)


def render(messages: list[dict], soft_part, notes: dict[str, str], blocks: dict[str, str] | None = None,
           digests: dict[str, str] | None = None) -> list[dict]:
    """Converted messages → engine messages: `soft` parts via `soft_part(name)`; handover reads and writes as the
    written block where `blocks` has one (name → block ID), else as the crisp note; parts merged into text where no
    block remains."""
    blocks, digests = blocks or {}, digests or {}
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
                    # Written digests (decision 43) show as the runtime lists them; otherwise the crisp cut-off preview.
                    if part["name"] in digests:
                        parts.append({"type": "neuralese", "id": digests[part["name"]]})
                        parts.append({"type": "text", "text": digest_note(part["holder"])})
                    else:
                        parts.append({"type": "text", "text": part["preview"]})
                elif part["type"] == "read":
                    name = part["name"]
                    parts.append({"type": "neuralese", "id": blocks[name]} if name in blocks else
                                 {"type": "text", "text": part.get("source", notes.get(name, ""))})
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
    """Note texts by name: from the record's compaction calls and from its reads (which carry their note)."""
    notes = {part["name"]: part["source"] for m in record["messages"] if isinstance(m.get("content"), list)
             for part in m["content"] if part["type"] == "read" and "source" in part}
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
    parser.add_argument("--digest", choices=["preview", "written"], default="preview",
                        help="digest sites: the crisp preview, or a digest the model writes at the operator's write site")
    parser.add_argument("--digest-window", type=int, default=4096,
                        help="value tokens per digest write site in training (longer values are digested in chunks)")
    parser.add_argument("--stop-pg", type=float, default=0.0,
                        help="train the stop head on written values by policy gradient with this length cost per vector (0: off)")
    parser.add_argument("--heads-lr", type=float, default=1e-4, help="the writer's port heads, when notes or digests are written")
    parser.add_argument("--detach-write-context", action="store_true",
                        help="no gradient into the write sites' context (saves memory; soft prompts there then do not learn from writing)")
    parser.add_argument("--distill", type=float, default=1.0, help="weight of the self-distillation term on written notes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=8)
    args = parser.parse_args(argv)

    from ..prompt_bank import load_bank, save_bank
    from ..serve import load_engine
    from ..serve.chat import RequestError, call_reply, render_messages
    from ..serve.grad import GradSession, encode_text
    from .execution import Prefilled, unroll_write
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

    producers = {}
    if args.handover == "written":
        # Every record whose target writes a note, by note name: the producer of that note's block.
        with open(args.records) as stream:
            for line in stream:
                if "$write" in line:  # escaped inside the arguments string in the raw line
                    record = json.loads(line)
                    name = target_write(record)
                    if name:
                        producers.setdefault(name, record)

    backbone, heads = engine.backbone, engine.heads

    def placeholder(name: str) -> str:
        """A block ID that stands for a value written afresh each step; its rows always come from the leaves."""
        digest = hashlib.sha256(name.encode()).digest()
        return "nz1_" + base64.b32encode(digest).decode().rstrip("=").lower()

    def write(messages, tools, prefix, leaves):
        """The write procedure with gradients (S3 `unroll_write`): the site's prompt (soft parts from `leaves`), the
        forced prefix and the open marker, then the sketch recurrence until the stop head stops. The stop decisions
        are detached; the payload carries gradients into the writer (feedback, content projection, LoRA) and into
        every soft parameter of the site."""
        prompt = render_messages(messages, tools, engine._template, engine.specials)
        items = session._items(prompt.segments, prompt.blocks) + [("tok", t) for t in engine._tokens(prefix)]
        items.append(("tok", backbone.controls.open_id))
        context = session._embed_items(items, leaves)
        if args.detach_write_context:
            context = context.detach()
        out = backbone.forward_embeds(context, cutoff=heads.cutoff, logits=False)
        written = unroll_write(backbone, heads, Prefilled(out["cache"], out["h_cut"][:, -1], out["h_cut"], None),
                               sample=bool(args.stop_pg), generator=stop_generator)
        n = int(written.lengths[0])
        lengths.append(n)
        if args.stop_pg and torch.is_grad_enabled():
            # Log-probability of the sampled stop decisions under the stop head (continue after 1..n-1, stop after n
            # unless the write ran to the maximum), differentiable in the stop head.
            logits = written.stop_logits[0].float()
            logp = torch.nn.functional.logsigmoid(-logits[:n - 1]).sum()
            if not bool(written.truncated[0]):
                logp = logp + torch.nn.functional.logsigmoid(logits[n - 1])
            stop_terms.append((logp, n))
        return written.payload[0, :n]

    lengths: list[int] = []
    stop_terms: list = []  # (log-probability of the stop decisions, length) of this record's writes
    stop_generator = torch.Generator().manual_seed(args.seed)
    baseline = {"value": None}

    note_prefix, _ = call_reply(lambda m, g: engine.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=g),
                                "compact_history", {}, "note")

    def note_payload(name, leaves):
        """The note written by the model from its producing record (soft-rendered), at the note argument."""
        producer = producers[name]
        messages = render(producer["messages"], lambda n: {"type": "neuralese", "id": leaf_ids[n]}, handover_notes(producer))
        return write(messages, producer.get("tools"), note_prefix, leaves)

    def digest_payload(record, part, leaves):
        """The digest of a listing value by the operator's plan (digest.py), every write differentiable."""
        crisp = crisp_messages(record["messages"], texts, handover_notes(record))
        opening = next((m["content"] for m in crisp if m["role"] == "user" and isinstance(m["content"], str)), "")
        found = INSTRUCTIONS.search(opening)
        system = [{"type": "neuralese", "id": leaf_ids["prompt:digest"]}]
        written = {}

        def site_write(messages):
            payload = write(messages, None, DIGEST_PREFIX, {**leaves, **written})
            block = placeholder(f"{part['name']}#{len(written)}")
            written[block] = payload
            return block

        block, _ = write_digest(site_write, system, part["holder"], part["value_type"], part["source"],
                                found.group(1) if found else "", engine.tokenizer, args.digest_window)
        return written[block]

    def written_values(record, leaves):
        """Blocks written for this record this step: handover notes it reads or shows, digests in its listing.
        Returns (name → placeholder ID, placeholder ID → payload)."""
        names, payloads = {}, {}
        if args.handover == "written":
            for name in reads(record) | set(handover_notes(record)):
                if name in producers:
                    names[name] = placeholder(name)
                    payloads[names[name]] = note_payload(name, leaves)
        if args.digest == "written":
            for message in record["messages"]:
                for part in message.get("content") if isinstance(message.get("content"), list) else []:
                    if part["type"] == "digest":
                        names[part["name"]] = placeholder(part["name"])
                        payloads[names[part["name"]]] = digest_payload(record, part, leaves)
        return names, payloads

    def soft_messages(record, names):
        return render(record["messages"], lambda name: {"type": "neuralese", "id": leaf_ids[name]}, handover_notes(record),
                      names, names)

    def target_of(record, names):
        return render([record["target"]], lambda name: {"type": "text", "text": texts[name]}, handover_notes(record), names)[0]

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
    # Soft parameters for the names the selected records use (a corpus has thousands of instructions texts).
    used_names = {part["name"] for record in train + held for message in record["messages"]
                  if isinstance(message.get("content"), list) for part in message["content"] if part["type"] == "soft"}
    if args.digest == "written":
        used_names.add("prompt:digest")
    for name in sorted(used_names):
        text = texts[name]
        piece = name.removeprefix("prompt:")
        if bank and name.startswith("prompt:") and piece in bank.rows:
            rows = bank.rows[piece]
            from_bank.append(name)
        else:
            rows = encode_text(engine, text).payload
        block = engine.store.put(make_block(rows, engine.dialect, type="Neuralese<SystemPrompt>",
                                            producer={"kind": "bank" if name in from_bank else "text-encode", "text": text}))
        params[name] = torch.nn.Parameter(block.payload.clone().float().to(engine.device))
        leaf_ids[name] = block.id
    init = {name: p.detach().clone() for name, p in params.items()}
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
    # The writer's own modules (feedback, content projection) learn from the readers of what they write.
    head_params = [p for p in heads.parameters()] if args.heads_lr and (args.handover == "written" or args.digest == "written") else []
    for p in head_params:
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params": list(params.values()), "lr": args.lr}] +
                                  ([{"params": lora, "lr": args.lora_lr}] if lora else []) +
                                  ([{"params": head_params, "lr": args.heads_lr}] if head_params else []), weight_decay=0.0)

    def loss_of(record, leaves, soft=True):
        if not soft:
            crisp = crisp_messages(record["messages"], texts, handover_notes(record))
            target = render([record["target"]], lambda name: {"type": "text", "text": texts[name]}, handover_notes(record))[0]
            return session._term({"kind": "crossEntropy", "messages": crisp, "tools": record.get("tools"), "target": target}, leaves)
        # Notes and digests are written afresh by the current writer; their payloads are leaves of this loss.
        stop_terms.clear()
        names, payloads = written_values(record, leaves)
        leaves = {**leaves, **payloads}
        messages, target = soft_messages(record, names), target_of(record, names)
        loss = session._term({"kind": "crossEntropy", "messages": messages, "tools": record.get("tools"), "target": target}, leaves)
        if args.distill and payloads and not target_write(record):
            # The teacher (no gradient) sees the crisp note and the listing's crisp preview where the student reads blocks.
            loss = loss + args.distill * session._term({"kind": "selfDistill", "messages": messages, "tools": record.get("tools"),
                                                        "target": target, "teacher_messages": crisp_messages(
                                                            record["messages"], texts, handover_notes(record))}, leaves)
        if stop_terms:
            # Stop policy (phase E's objective on real readers): reward = -(reader loss + λ·length), against a running
            # baseline; the stop head learns how long a note or digest must be for what its readers need.
            reward = -(float(loss.detach()) + args.stop_pg * sum(n for _, n in stop_terms))
            advantage = reward - (baseline["value"] if baseline["value"] is not None else reward)
            baseline["value"] = reward if baseline["value"] is None else 0.9 * baseline["value"] + 0.1 * reward
            loss = loss - advantage * sum(logp for logp, _ in stop_terms)
            stop_terms.clear()
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
            # The writer's gradient from its readers: zero would mean written values do not train the writer.
            writer_grad = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in head_params if p.grad is not None)
                                           or torch.zeros(()))) if head_params else None
            torch.nn.utils.clip_grad_norm_(list(params.values()) + lora + head_params, 1.0)
            optimizer.step()
            if step % 10 == 0 or step == args.steps - 1:
                entry = {"step": step, "loss": sum(losses) / max(1, len(losses)), "seconds": round(time.time() - started),
                         "errors": errors, **({"writer_grad_norm": writer_grad} if head_params else {}),
                         **({"write_lengths": lengths[-8:]} if lengths else {})}
                if args.device.startswith("cuda"):
                    entry["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                log.write(json.dumps(entry) + "\n")
                log.flush()
                print(json.dumps(entry), flush=True)
    report["soft-trained"] = evaluate("soft-trained", leaves)
    if lengths:
        report["writes"] = {"count": len(lengths), "mean_length": sum(lengths) / len(lengths), "max_length": max(lengths)}
    if head_params:
        torch.save({"heads": heads.state_dict(), "port_config": heads.port_config()}, out / "heads.pt")
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
