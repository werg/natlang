"""Whole-program replay training (plans/neuralese/S5_PROGRAM_TRAINING.md §3; spec/NEURALESE_GRAPH.md, "Replay").

A **replay record** (`natlang.replay-record/1`, one JSON line) is what an in-program `valueAndGrad` sends to a
`grad` session, written by the runtime's replay sink (`natlang:learning`, `replayRecordSink`): the argument blocks,
the loss terms (each over a recorded model turn: `crossEntropy` of a checked output, `law` terms, `selfDistill`,
`decision`, ...), and the **producers**, every recorded turn of the program that wrote blocks. Its blocks travel in a
sidecar directory of one-block files named by block ID. A record is a whole multi-call program: a loss on a downstream
call reaches upstream values through the producer chain (a block a later call reads is re-written from the turn that
wrote it, with gradient, `GradSession._rewritten`), with discrete choices and effect results held at their recorded
values.

This module trains **values** over many records: a bank of leaf parameters keyed by block ID (operator bodies of the
standard library, prompt pieces, data blocks) shared by every record that reads them, with AdamW over minibatches of
records. It also measures whole-program **controls** (S5 risks: correct soft values must beat shuffled and zeroed ones
on whole-program results, not only on local losses). For controls a re-written block takes the value its writer
produces from the changed context (`propagate`), so a zeroed upstream value changes what downstream calls read;
training does the same by default (`--recorded-values`: S4 semantics, recorded blocks kept). The **gate** (S5 G2, "Do trained operators beat their
text-initialised versions?") compares trained and initial values on held-out records under the same replay.

    python -m natlang_neuralese.train.graph_replay --records R.jsonl --blocks DIR --library stdlib.nz --out OUT \
        [--steps 200 --lr 1e-3 --register-artifact ID]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import torch

from ..serve.grad import GradSession

SCHEMA = "natlang.replay-record/1"
CONTROLS = ("trained", "initial", "zeroed", "shuffled")


# Records ------------------------------------------------------------------------------------------------------------
def load_records(paths) -> list[dict]:
    """Replay records from JSONL files; each keeps the directory its blocks live in (`blocks`, default
    `<file stem>.blocks/` beside it)."""
    out = []
    for path in paths:
        path = Path(path)
        default_blocks = path.with_suffix("").with_suffix(".blocks") if path.suffix == ".jsonl" else path.parent
        with open(path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema") != SCHEMA:
                    raise ValueError(f"{path}: not a {SCHEMA} record")
                record.setdefault("blocks_dir", str(Path(record.get("blocks_dir") or default_blocks)))
                out.append(record)
    return out


def load_blocks(engine, directories) -> int:
    """Put every one-block file (`<id>.safetensors`, the server's block wire format) of the directories into the
    engine's store, verifying each ID against its content. Returns the number of blocks."""
    from ..serve.store import decode_block

    count = 0
    for directory in dict.fromkeys(str(d) for d in directories):
        for file in sorted(Path(directory).glob("nz1_*.safetensors")):
            block = decode_block(file.read_bytes())
            if block.id != file.name.removesuffix(".safetensors"):
                raise ValueError(f"{file}: content hashes to {block.id}")
            engine.store.put(block)
            count += 1
    return count


def split_records(records, held_share: float = 0.2, seed: int = 0) -> tuple[list, list]:
    """(train, held): an explicit `split` field wins; otherwise a seeded split by record `group` (or ID), so cases of
    one source never sit on both sides."""
    train, held, undecided = [], [], {}
    for record in records:
        split = record.get("split")
        if split in ("test", "held", "heldout"):
            held.append(record)
        elif split == "train":
            train.append(record)
        else:
            undecided.setdefault(str(record.get("group") or record["id"]), []).append(record)
    groups = sorted(undecided)
    random.Random(seed).shuffle(groups)
    cut = int(round(len(groups) * held_share))
    for index, group in enumerate(groups):
        (held if index < cut else train).extend(undecided[group])
    return train, held


# Replay -------------------------------------------------------------------------------------------------------------
class ProgramReplay(GradSession):
    """A grad session over whole recorded programs, with trainer-owned leaves. `propagate` (controls): a re-written
    block takes its new value instead of keeping the recorded one, so changed upstream values reach their consumers."""

    propagate = False

    def _rewritten(self, block_id, leaves, produced, memo, visiting=()):
        if not self.propagate:
            return super()._rewritten(block_id, leaves, produced, memo, visiting)
        # As GradSession._rewritten, but the block becomes what its writer writes from the (changed) context instead
        # of the recorded value; every block upstream of a leaf is re-written, in dependency order.
        if block_id in memo:
            return memo[block_id]
        from .execution import prefill_write_context, unroll_write

        producer, prompt, reply, at = produced[block_id]
        context = prompt + reply[:at]
        local = {}
        for inner in dict.fromkeys(self._block_ids(context)):
            if inner in produced and inner not in leaves and inner not in visiting:
                value = self._rewritten(inner, leaves, produced, memo, visiting + (block_id,))
                if value is not None:
                    local[inner] = value
        adapters = producer.get("adapters") or []
        depends = bool(local) or any(b in leaves for b in self._block_ids(context)) or \
            any(a.get("id") in leaves for a in adapters if isinstance(a, dict))
        if not depends:
            memo[block_id] = None
            return None
        stored = self.engine.lookup(block_id)
        with self._adapted(adapters, leaves):
            pre = prefill_write_context(self.backbone, self.heads, self._embed_items(context, {**leaves, **local}))
            written = unroll_write(self.backbone, self.heads, pre, length=max(1, stored.length))
        memo[block_id] = written.payload[0, :stored.length].float()
        return memo[block_id]

    def record_terms(self, record: dict, leaves: dict) -> list[torch.Tensor]:
        """The weighted loss terms of one record (each with a graph to `leaves`)."""
        if int(record.get("order") or 1) != 1:
            raise ValueError(f"record {record['id']}: second-order records are replayed by a grad session, not here")
        produced = self._index_producers(record.get("producers")) if leaves else {}
        return [self._term_value(term, leaves, produced, record.get("adapters")) for term in record["terms"]]


# Leaves -------------------------------------------------------------------------------------------------------------
class LeafBank:
    """Trainable values keyed by block ID: one parameter per block, shared by every record that reads it."""

    def __init__(self, engine, block_ids):
        self.initial = {b: engine.lookup(b).payload.detach().float().to(engine.device) for b in dict.fromkeys(block_ids)}
        self.params = {b: torch.nn.Parameter(v.clone()) for b, v in self.initial.items()}

    def leaves(self, mode: str = "trained", seed: int = 0) -> dict:
        """Leaf tensors for a replay: the trained values, the initial ones, zeros, or a derangement of the bank
        (each block replaced by another block's trained value; with one block, by a seeded permutation of its rows)."""
        if mode == "trained":
            return dict(self.params)
        if mode == "initial":
            return {b: v.clone() for b, v in self.initial.items()}
        if mode == "zeroed":
            return {b: torch.zeros_like(v) for b, v in self.params.items()}
        if mode == "shuffled":
            ids = sorted(self.params)
            if len(ids) == 1:
                value = self.params[ids[0]].detach()
                order = torch.randperm(value.shape[0], generator=torch.Generator().manual_seed(seed)).to(value.device)
                return {ids[0]: value[order]}
            shift = 1 + seed % (len(ids) - 1)
            return {b: self.params[ids[(i + shift) % len(ids)]].detach().clone() for i, b in enumerate(ids)}
        raise ValueError(f"unknown leaf mode {mode}")

    def drift(self) -> dict:
        """Relative change of each value from its initialisation."""
        return {b: float((p.detach() - self.initial[b]).norm() / self.initial[b].norm().clamp_min(1e-8))
                for b, p in self.params.items()}


def trainable_ids(records, only: set[str] | None = None) -> list[str]:
    """The argument blocks of the records (restricted to `only`, e.g. a library's bodies)."""
    ids = [b for record in records for b in record.get("arguments") or []]
    return [b for b in dict.fromkeys(ids) if only is None or b in only]


# Training and evaluation --------------------------------------------------------------------------------------------
def evaluate(replay: ProgramReplay, records, bank: LeafBank, modes=CONTROLS, seed: int = 0) -> dict:
    """Mean whole-program loss of the records under each leaf mode, with propagation (controls change what
    downstream calls read). Also per-term-kind means for the trained values."""
    out = {}
    replay.propagate = True
    try:
        for mode in modes:
            leaves = bank.leaves(mode, seed)
            totals, kinds = [], {}
            with torch.no_grad():
                for record in records:
                    terms = replay.record_terms(record, leaves)
                    totals.append(float(sum(t.detach() for t in terms)))
                    if mode == "trained":
                        for term, value in zip(record["terms"], terms):
                            key = term.get("law") or term.get("operator") or term["kind"]
                            kinds.setdefault(key, []).append(float(value))
            out[mode] = sum(totals) / max(1, len(totals))
            if mode == "trained":
                out["by_term"] = {k: sum(v) / len(v) for k, v in sorted(kinds.items())}
    finally:
        replay.propagate = False
    return out


def gate(held: dict, *, margin: float = 0.0) -> dict:
    """S5 G2 operator question and channel use on held-out programs: trained values beat their initialisation, and
    beat zeroed and shuffled values, each by `margin` (relative). Registration stays unqualified unless it passes."""
    trained = held["trained"]
    beats = {mode: (held[mode] - trained) / max(abs(held[mode]), 1e-8) > margin for mode in ("initial", "zeroed", "shuffled")
             if mode in held}
    return {"passed": all(beats.values()), "beats": beats, "trained": trained,
            **{mode: held[mode] for mode in ("initial", "zeroed", "shuffled") if mode in held}, "margin": margin}


def train(replay: ProgramReplay, train_records, held_records, bank: LeafBank, *, steps: int, lr: float, batch: int,
          eval_every: int, log, seed: int = 0, weight_decay: float = 0.0, propagate: bool = True) -> dict:
    """AdamW on the bank over minibatches of records (gradients summed per record, so the peak memory is one
    program's graph). Discrete choices stay fixed (text tokens are teacher-forced, writes unroll at their recorded
    lengths); with `propagate` (default) every written block is recomputed from the current values (S5 §3.2 step 5),
    so the loss is the program's under the values being trained. Without it a re-written block keeps its recorded
    value (the S4 grad-session semantics): the loss then cannot change when a value sits only upstream, and only its
    first-order signal at the recorded point trains it."""
    # Adam's early steps move every element by about lr whatever its gradient, so the step is set relative to each
    # value's RMS: port-encoded bodies sit near 0.5, token-embedding ones near 0.025.
    optimizer = torch.optim.AdamW([{"params": [p], "lr": lr * float(bank.initial[b].pow(2).mean().sqrt().clamp_min(1e-6))}
                                   for b, p in bank.params.items()], lr=lr, weight_decay=weight_decay)
    replay.propagate = propagate
    rng = random.Random(seed)
    order = list(range(len(train_records)))
    cursor, history = len(order), []
    for step in range(1, steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(batch):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            record = train_records[order[cursor]]
            cursor += 1
            terms = replay.record_terms(record, bank.leaves("trained"))
            loss = torch.stack([t.reshape(()) for t in terms]).sum() / batch
            if loss.requires_grad:
                loss.backward()
            losses.append(float(loss.detach()) * batch)
            del terms, loss  # one program's graph at a time
        norm = float(torch.nn.utils.clip_grad_norm_(list(bank.params.values()), 1.0))
        optimizer.step()
        row = {"step": step, "loss": sum(losses) / len(losses), "grad_norm": norm,
               "seconds": time.perf_counter() - started}
        if eval_every and (step % eval_every == 0 or step == steps):
            row["held"] = evaluate(replay, held_records, bank, ("trained",), seed)
            replay.propagate = propagate
            row["drift"] = max(bank.drift().values(), default=0.0)
        history.append(row)
        log(row)
    replay.propagate = False
    return {"history": history}


def save_bank(bank: LeafBank, source_nz, target, provenance: dict) -> dict[str, str]:
    """Write the trained values into a copy of the `.nz` they came from (a standard library, a prompt bank)."""
    from ..stdlib import rewrite_nz

    return rewrite_nz(source_nz, target, {b: p.detach() for b, p in bank.params.items()}, provenance=provenance)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", action="append", required=True, help="replay-record JSONL (repeatable)")
    parser.add_argument("--blocks", action="append", default=[], help="extra block directories")
    parser.add_argument("--library", required=True,
                        help=".nz whose blocks are the trained values (a standard library or a prompt bank)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default=None)
    parser.add_argument("--heads", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2, help="Adam step relative to each value's RMS")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--held-share", type=float, default=0.2)
    parser.add_argument("--gate-margin", type=float, default=0.0)
    parser.add_argument("--checkpoint-layers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recorded-values", action="store_true",
                        help="keep re-written blocks at their recorded values while training (S4 grad semantics)")
    parser.add_argument("--register-artifact", default=None, metavar="ID")
    parser.add_argument("--artifact-corpus", action="append", default=[])
    parser.add_argument("--artifact-kind", default="standard-library")
    parser.add_argument("--backbone-revision", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory-gb", type=float, default=float(os.environ.get("NATLANG_CUDA_MEMORY_GB", 16)))
    args = parser.parse_args(argv)

    from ..serve import load_engine
    from ..stdlib import load_into_store, read_blocks

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    if args.device.startswith("cuda"):
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_gb * 2**30 / total))
    torch.manual_seed(args.seed)
    log_file = (out / "train.jsonl").open("a")

    def log(row):
        print(json.dumps(row), flush=True)
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()

    engine = load_engine(args.base, heads_checkpoint=args.heads, device=args.device)
    # A program is several full prompts (each call's opening carries the runtime prompt) with graphs at once:
    # recompute layers in backward instead of keeping their activations.
    engine.backbone.checkpoint_layers = args.checkpoint_layers
    records = load_records(args.records)
    blocks = load_blocks(engine, [r["blocks_dir"] for r in records] + args.blocks)
    stored = load_into_store(engine, args.library)
    if any(old != new for old, new in stored.items()):
        raise SystemExit(f"{args.library}: block IDs do not match their content under the f32 rule")
    header, _ = read_blocks(args.library)
    if header["dialect"] != engine.dialect:
        raise SystemExit(f"{args.library} is {header['dialect']}, the model speaks {engine.dialect}")
    library_ids = set(header["blocks"])
    ids = trainable_ids(records, library_ids)
    if not ids:
        raise SystemExit("no record takes a block of the library as an argument")
    train_records, held_records = split_records(records, args.held_share, args.seed)
    replay = ProgramReplay(engine)
    bank = LeafBank(engine, ids)
    log({"event": "start", "records": len(records), "train": len(train_records), "held": len(held_records),
         "blocks": blocks, "trainable": len(ids), "dialect": engine.dialect})
    before = evaluate(replay, held_records, bank, CONTROLS, args.seed)
    log({"event": "held-before", **before})
    result = train(replay, train_records, held_records, bank, steps=args.steps, lr=args.lr, batch=args.batch,
                   eval_every=args.eval_every, log=log, seed=args.seed, propagate=not args.recorded_values)
    after = evaluate(replay, held_records, bank, CONTROLS, args.seed)
    verdict = gate(after, margin=args.gate_margin)
    log({"event": "held-after", **after, "gate": verdict, "drift": bank.drift()})
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4],
                            capture_output=True, text=True).stdout.strip()
    trained_path = out / "trained.nz"
    mapping = save_bank(bank, args.library, trained_path, {
        "trained_by": "natlang_neuralese.train.graph_replay", "commit": commit, "records": len(records),
        "steps": args.steps, "gate": verdict})
    (out / "result.json").write_text(json.dumps({"before": before, "after": after, "gate": verdict,
                                                 "mapping": mapping, "history": result["history"][-1:]}, indent=1) + "\n")
    if args.register_artifact:
        from .. import artifacts
        from ..model.lfm2_port import DEFAULT_REVISION, resolve_base

        parent = artifacts.find_by_sha(artifacts.digest(Path(args.library)))
        revision = args.backbone_revision or (DEFAULT_REVISION if args.base is None else None)
        manifest = artifacts.register_output(
            trained_path, identity=args.register_artifact, kind=args.artifact_kind, dialect=engine.dialect,
            backbone=artifacts.backbone_identity(args.base or resolve_base(None), revision),
            trainer="natlang_neuralese.train.graph_replay", commit=commit, corpora=args.artifact_corpus,
            parent=parent[0] if parent else None, run=str(out.resolve()),
            notes=f"held gate {'passed' if verdict['passed'] else 'failed'}: trained {verdict['trained']:.4f}, "
                  f"initial {verdict.get('initial', math.nan):.4f}, zeroed {verdict.get('zeroed', math.nan):.4f}, "
                  f"shuffled {verdict.get('shuffled', math.nan):.4f} (registration does not qualify)")
        log({"event": "registered", "artifact": args.register_artifact, "bytes": manifest["bytes"]})


if __name__ == "__main__":
    main()
