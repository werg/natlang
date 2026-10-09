"""Learned updater v0 (S6_META_LEARNING.md §5; LEARNING_CONTINUUM.md §5, §9.1 step 1: imitation of improvement).

The updater is the model itself, conditioned on a trained updater block U, the soft skill to update (B) and the
evidence it may see (the family's support cases with their gold answers, as text). It writes an ordinary block W
through the write port (fixed length, differentiable unroll), and the delta projection D turns it into the update:
B' = B + D(W, B). The model never emits a delta (owner, 2026-10-04).

Training targets are recorded improvement steps: in-place soft-skill deltas `after − before` of the gradient arm
(`method-arm:soft-gold`, run-method-arms.mjs), so the updater learns in one write what gradient tuning found in its
steps. U and D train; the backbone and port heads stay frozen. Families are split: held-out families are never seen.

Evaluation on held-out families: the relative error of D(W, B) against the recorded delta (against the zero update and
the mean training delta), and the updated skills B' are uploaded to the model server (`--server`) so their query
quality is measured by `ts-host/scripts/skills/evaluate-soft-skills.mjs` beside the gradient arms at matched compute.

    python -m natlang_neuralese.train.learned_updater --arms DIR --cases decision-cases.jsonl --out DIR
        [--heldout-families a,b] [--server URL] [--steps 300] [--code-length 8] [--updater-length 8] [--device cuda]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch

from ..model.projections import DeltaProjection
from ..nz import read_nz
from .projection import relative_error

SCHEMA = "natlang.learned-updater/1"
UPDATER_SYSTEM = ("You update a decision skill. Read the updater instructions, the current skill and the examples with "
                  "their correct answers, then write the update that makes the skill give those answers.")


def server_block(server: str):
    """A fetch for blocks a run did not save (the arms' shared soft-init skill): the model server's stored copy."""
    def fetch(block_id: str):
        from ..serve.store import decode_block

        try:
            with urllib.request.urlopen(f"{server.rstrip('/')}/v1/neuralese/blocks/{block_id}", timeout=600) as response:
                block = decode_block(response.read())
        except urllib.error.HTTPError:
            return None
        return block.payload.float(), block.dialect
    return fetch


def load_examples(arms_dir: str | Path, operator: str = "method-arm:soft-gold", fetch=None) -> tuple[str, list[dict]]:
    """[{family, step, base [L, d], delta [L, d]}] from a run-method-arms output directory; blocks the run did not
    save come from `fetch(block_id)` when given."""
    arms_dir = Path(arms_dir)
    # One bundle (older runs) or one file per family (artifacts/<family>.nz).
    paths = [arms_dir / "artifacts.nz"] if (arms_dir / "artifacts.nz").exists() else sorted((arms_dir / "artifacts").glob("*.nz"))
    blocks = {}
    for path in paths:
        _, exports = read_nz(path)
        blocks.update({e.block_id: (e.payload, e.dialect) for e in exports.values() if e.payload is not None})
    examples, dialects = [], set()
    for line in open(arms_dir / "improvement-steps.jsonl"):
        step = json.loads(line)
        if step["operator"]["kind"] != operator:
            continue
        before = [r["id"] for r in step["before"] if r["kind"] == "soft-skill"]
        after = [r["id"] for r in step["after"] if r["kind"] == "soft-skill"]
        if fetch is not None:
            for block_id in before + after:
                if block_id not in blocks and (found := fetch(block_id)) is not None:
                    blocks[block_id] = found
        if len(before) != 1 or len(after) != 1 or before[0] not in blocks or after[0] not in blocks:
            continue
        (base, dialect), (changed, _) = blocks[before[0]], blocks[after[0]]
        if base.shape != changed.shape:
            continue
        family = step["episode"].get("family") or step["episode"]["id"].split(":", 1)[-1]
        examples.append({"family": family, "step": step["id"], "base": base.float(), "delta": (changed - base).float(),
                         "outcome": step.get("outcome")})
        dialects.add(dialect)
    if len(dialects) != 1:
        raise SystemExit(f"examples must share one dialect; found {sorted(dialects)}")
    return dialects.pop(), examples


def support_view(cases_path: str | Path, family: str, count: int = 16, max_chars: int = 400) -> str:
    """The evidence the updater sees: `count` of the family's train-role cases (answers interleaved by stratum,
    deterministic), as question / input / answer text."""
    rows = [json.loads(line) for line in open(cases_path)]
    rows = sorted((r for r in rows if r["family"] == family and r.get("role") == "train"), key=lambda r: r["id"])
    strata: dict[str, list] = {}
    for row in rows:
        strata.setdefault(json.dumps(row["answer"], sort_keys=True), []).append(row)
    chosen, queues = [], list(strata.values())
    while len(chosen) < count and any(queues):
        for queue in queues:
            if queue and len(chosen) < count:
                chosen.append(queue.pop(0))
    question = chosen[0]["question"] if chosen else family
    lines = [f"Question: {question}"]
    for row in chosen:
        lines.append(f"Input: {str(row['state'])[:max_chars]}\nAnswer: {row['answer']}")
    return "\n\n".join(lines)


class Updater:
    """U and D around a frozen engine (serve.load_engine)."""

    def __init__(self, engine, dialect: str, updater_length: int, code_length: int, hidden: int, seed: int):
        from ..serve.grad import GradSession

        self.engine, self.session, self.code_length = engine, GradSession(engine), code_length
        for parameter in list(engine.backbone.parameters()) + list(engine.heads.parameters()):
            parameter.requires_grad_(False)
        width = engine.backbone.embedding_weight.shape[1]
        generator = torch.Generator().manual_seed(seed)
        self.U = (0.02 * torch.randn(updater_length, width, generator=generator)).to(engine.device).requires_grad_(True)
        self.D = DeltaProjection(dialect, width, hidden).to(engine.device)
        with torch.no_grad():
            self.D.out.weight.normal_(0, 1e-3, generator=None)
        tokenizer = engine.tokenizer
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": UPDATER_SYSTEM},
             {"role": "user", "content": "Updater instructions: \x00U\x00\nCurrent skill: \x00B\x00\n\n\x00V\x00"}],
            tokenize=False, add_generation_prompt=True)
        self.template = rendered.split("\x00")
        if [self.template[i] for i in (1, 3, 5)] != ["U", "B", "V"]:
            raise ValueError("the chat template moved the updater placeholders")

    def items(self, view: str) -> list:
        t, tok = self.template, self.engine.tokenizer
        ids = lambda text: [("tok", i) for i in tok(text, add_special_tokens=False).input_ids]
        return ids(t[0]) + [("block", "U")] + ids(t[2]) + [("block", "B")] + ids(t[4]) + ids(view) + ids(t[6])

    def delta(self, base: torch.Tensor, view: str) -> torch.Tensor:
        from .execution import prefill_write_context, unroll_write

        engine = self.engine
        base = base.to(engine.device)
        context = self.session._embed_items(self.items(view), {"U": self.U, "B": base})
        pre = prefill_write_context(engine.backbone, engine.heads, context)
        written = unroll_write(engine.backbone, engine.heads, pre, length=self.code_length)
        return self.D(written.payload[0, :self.code_length].float(), base.float())

    def parameters(self):
        return [self.U] + list(self.D.parameters())


def upload(server: str, payload: torch.Tensor, dialect: str, type_: str, producer: dict) -> str:
    from ..serve.store import encode_block, make_block

    block = make_block(payload.detach().cpu(), dialect, type=type_, producer=producer)
    request = urllib.request.Request(f"{server.rstrip('/')}/v1/neuralese/blocks/{block.id}", data=encode_block(block),
                                     headers={"content-type": "application/octet-stream"}, method="PUT")
    with urllib.request.urlopen(request, timeout=600) as response:
        response.read()
    return block.id


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arms", required=True, help="run-method-arms output directory with soft-gold steps")
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--heldout-families", default="", help="comma-separated; default: a seeded quarter of the families")
    p.add_argument("--server", default=None, help="model server to upload the held-out updated skills to")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--code-length", type=int, default=8)
    p.add_argument("--updater-length", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--support", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    from ..serve import load_engine

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    dialect, examples = load_examples(a.arms, fetch=server_block(a.server) if a.server else None)
    families = sorted({e["family"] for e in examples})
    if a.heldout_families:
        held_families = set(a.heldout_families.split(","))
    else:
        shuffled = families[:]
        random.Random(a.seed).shuffle(shuffled)
        held_families = set(shuffled[:max(1, len(families) // 4)])
    train = [e for e in examples if e["family"] not in held_families]
    held = [e for e in examples if e["family"] in held_families]
    if not train or not held:
        raise SystemExit(f"need train and held-out families; have {families}")
    views = {f: support_view(a.cases, f, a.support) for f in families}
    torch.manual_seed(a.seed)
    engine = load_engine(None, None, None, None, None, a.device, None)
    if engine.dialect != dialect:
        raise SystemExit(f"recorded deltas are {dialect}; the engine speaks {engine.dialect}")
    updater = Updater(engine, dialect, a.updater_length, a.code_length, a.hidden, a.seed)
    optimizer = torch.optim.Adam(updater.parameters(), lr=a.lr)
    log = (out / "train.jsonl").open("w")
    started = time.time()
    for step in range(a.steps):
        example = train[step % len(train)]
        optimizer.zero_grad(set_to_none=True)
        loss = relative_error(updater.delta(example["base"], views[example["family"]])[None],
                              example["delta"].to(engine.device)[None])
        loss.backward()
        optimizer.step()
        if step % 10 == 0 or step == a.steps - 1:
            log.write(json.dumps({"step": step, "family": example["family"], "relative_error": float(loss)}) + "\n")
            log.flush()
    mean_delta = torch.stack([e["delta"] for e in train]).mean(0) if len({tuple(e["delta"].shape) for e in train}) == 1 else None
    report = {"schema": SCHEMA, "dialect": dialect, "train_families": sorted({e["family"] for e in train}),
              "heldout_families": sorted(held_families), "train_seconds": time.time() - started, "options": vars(a),
              "heldout": [], "train": []}
    with torch.no_grad():
        for split, items in (("train", train), ("heldout", held)):
            for e in items:
                predicted = updater.delta(e["base"], views[e["family"]]).cpu()
                row = {"family": e["family"], "relative_error": float(relative_error(predicted[None], e["delta"][None])),
                       "cosine": float(torch.nn.functional.cosine_similarity(predicted.flatten(), e["delta"].flatten(), 0)),
                       "zero_update_error": 1.0,
                       "mean_delta_error": float(relative_error(mean_delta[None], e["delta"][None])) if mean_delta is not None
                       and mean_delta.shape == e["delta"].shape else None}
                if split == "heldout" and a.server:
                    row["updated_skill"] = upload(a.server, e["base"] + predicted, dialect, "Neuralese<string>",
                                                  {"kind": "learned-updater", "updater": SCHEMA, "family": e["family"]})
                    row["base_skill"] = upload(a.server, e["base"], dialect, "Neuralese<string>", {"kind": "base"})
                report[split].append(row)
    torch.save({"schema": SCHEMA, "dialect": dialect, "U": updater.U.detach().cpu(), "D": updater.D.state_dict(),
                "template": updater.template, "options": vars(a)}, out / "updater.pt")
    digest = hashlib.sha256((out / "updater.pt").read_bytes()).hexdigest()
    report["updater_sha256"] = digest
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)
    print(json.dumps({"train_relative_error": mean(report["train"], "relative_error"),
                      "heldout_relative_error": mean(report["heldout"], "relative_error"),
                      "heldout_cosine": mean(report["heldout"], "cosine"), "updater_sha256": digest[:16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
