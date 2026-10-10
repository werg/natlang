"""View-stage records: `view`/`ask` port records rendered for the trajectory trainer and the view gate.

The view operator stage (TRAINING_RECIPE.md "The view operator stage"; DECISIONS 2026-10-09, one summarizer family)
trains the Neuralese instance of the builtin `view(value, instructions?)` through its readers. Its data is the
`view-ask` corpus (plans/neuralese/VIEW_CORPUS.md): port records whose source is the value (several sources: the
list of documents, rendered by `value_text`), whose
`writer.instructions` is the purpose and whose consumer request and target are what a reader of the view must
produce. This module turns each record into one `natlang.teacher_training_turn.native/1` record of the trajectory
trainer (`train.trajectories`), so the stage reuses its written-view path unchanged:

- The user turn holds one `view` part (the value as `source`; the full value again as `preview`, so the crisp
  rendering is the full-text reader that `--distill` distils from) followed by the request. The source itself is
  not in the prompt: with `--view written` the reader sees only the block written at view's template write site
  (`view.py`, the same site the servers' `POST /v1/neuralese/view` uses).
- `reconstruct` records ask for the value back with a **faithful** view (`"faithful": true`: written without
  instructions, `view(x)`); `consume` and `compare` records carry the purpose as the part's `instructions`.
- The target is the record's gold answer as the assistant reply (reconstruction: the value itself).
- `compare` records keep their partners' instructions (`view_stage.contrast_instructions`) and the purpose-free
  form (`view_stage.instructions_general`) for the purpose-sensitivity contrast and the gate.

The gate (`eval.view_gate`) renders its consumers with `consumer_messages`, the same rendering as training.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from ..view import INSTRUCTIONS

VERSION = "natlang.teacher_training_turn.native/1"
CONVERTER = "natlang_neuralese.data.view_records@2"
COHORT = "view"
NOTE = "  // view of the value; the value itself is not shown"
PIECES = [{"name": "prompt:view", "kind": "system-prompt", "text": INSTRUCTIONS}]
# The trajectory trainer has train and held (test) records only; validation records stay with the corpus (model
# selection, the gate's development set) and are not converted.
SPLITS = {"train": "train", "test": "test"}


def _target_text(target: dict) -> str:
    value = target["value"]
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def value_text(port: dict) -> str:
    """The viewed value as text. A multi-source record (several documents, one purpose: Multi-News, VIEW_CORPUS.md
    §1) is `view(xs, instructions)` over the list; its value is rendered as the documents in order, each under a
    `Document k of n` line."""
    sources = port["sources"]
    if len(sources) == 1:
        return sources[0]["text"]
    return "\n\n".join(f"Document {k} of {len(sources)}:\n{s['text']}" for k, s in enumerate(sources, 1))


def view_part(port: dict) -> dict:
    """The record's value as a view part: faithful for reconstruction, written for the purpose otherwise."""
    source = value_text(port)
    part = {"type": "view", "name": "view:" + hashlib.sha256(port["id"].encode("utf-8")).hexdigest()[:16],
            "holder": "the value", "value_type": "string", "source": source, "preview": source, "note": NOTE}
    if port["task"] == "reconstruct":
        part["faithful"] = True
    else:
        part["instructions"] = port["writer"]["instructions"]
    return part


def request_of(port: dict) -> str:
    return next(m["content"] for m in port["consumer"]["context"] if m["role"] == "user")


def consumer_messages(port: dict, shown: dict) -> list[dict]:
    """The reader's prompt with `shown` in the view's place: {"block": id} (a written view, with the note),
    {"text": …} (the full value: the crisp reader) or {} (nothing: the no-source baseline)."""
    request = request_of(port)
    if "block" in shown:
        content = [{"type": "neuralese", "id": shown["block"]}, {"type": "text", "text": NOTE + "\n\n" + request}]
    elif "text" in shown:
        content = shown["text"] + "\n\n" + request
    else:
        content = request
    return [{"role": "user", "content": content}]


def stage_record(port: dict, *, corpus: str, by_id: dict | None = None) -> dict | None:
    """One trajectory-trainer record for a view port record (None for splits the trainer does not take)."""
    split = SPLITS.get(port["split"])
    if split is None or not port["sources"] or any("text" not in s for s in port["sources"]):
        return None
    if len(port["sources"]) > 1 and port["task"] == "reconstruct":
        return None  # one value per reconstruction
    notes = port["lineage"].get("notes") or {}
    pairs = [(by_id or {}).get(i) for i in (port.get("contrasts") or {}).get("purpose_pairs") or []]
    contrast = [p["writer"]["instructions"] for p in pairs if p and p["writer"]["instructions"] != port["writer"]["instructions"]]
    return {
        "version": VERSION,
        "id": "view-stage:" + port["id"],
        "source_ref": {"port_record": port["id"], "port_record_sha256": port["lineage"].get("sha256"), "corpus": corpus},
        "provenance": {"builder": CONVERTER, "corpus": corpus, "teacher": None,
                       "target_origin": notes.get("target_origin"), "view_call": notes.get("view_call")},
        "task": request_of(port),
        "family": "view_stage",
        "task_family": port["family"],
        "task_kind": port["task"],
        "task_modality": notes.get("artifact"),
        "license": port["license"]["spdx"],
        "license_provenance": notes.get("license_provenance"),
        "split": split,
        "source_groups": list(port["split_groups"]),
        "source_ids": [port["id"]],
        "outcome": {"label": port["outcome"]["label"], "checked": port["outcome"].get("checked")},
        "training_admission": {"kind": "view-stage/1", "approved": False,
                               "reason": "the view operator stage awaits its qualification (TRAINING_RECIPE.md)"},
        "cohort": COHORT,
        "messages": [{"role": "user", "content": [view_part(port), {"type": "text", "text": "\n\n" + request_of(port)}]}],
        "target": {"role": "assistant", "content": _target_text(port["target"])},
        "tools": None,
        "view_stage": {"artifact": notes.get("artifact"), "task": port["task"],
                       "instructions_general": port["writer"].get("instructions_general"),
                       "contrast_instructions": contrast[:4],
                       "purpose_pairs": ["view-stage:" + i for i in (port.get("contrasts") or {}).get("purpose_pairs") or []]},
    }


def convert(corpus_dir: Path, out: Path, *, corpus_id: str | None = None) -> dict:
    """Write `records.jsonl`, `pieces.jsonl` and `conversion.json` for the trajectory trainer."""
    corpus_dir = Path(corpus_dir)
    corpus_id = corpus_id or corpus_dir.name
    ports = [json.loads(line) for path in sorted(corpus_dir.glob("*.port-records.jsonl"))
             for line in path.open(encoding="utf-8") if line.strip()]
    by_id = {p["id"]: p for p in ports}
    out.mkdir(parents=True, exist_ok=True)
    counts, skipped = Counter(), Counter()
    with open(out / "records.jsonl", "w", encoding="utf-8") as stream:
        for port in ports:
            record = stage_record(port, corpus=corpus_id, by_id=by_id)
            if record is None:
                skipped[port["split"]] += 1
                continue
            counts[f"{record['split']}:{record['task_kind']}"] += 1
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(out / "pieces.jsonl", "w", encoding="utf-8") as stream:
        for piece in PIECES:
            stream.write(json.dumps(piece, ensure_ascii=False) + "\n")
    summary = {"schema": "natlang.view-stage-conversion/1", "converter": CONVERTER, "corpus": corpus_id,
               "records": sum(counts.values()), "by_split_task": dict(sorted(counts.items())),
               "not_converted": dict(skipped), "pieces": [p["name"] for p in PIECES],
               "sha256": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                          for name in ("records.jsonl", "pieces.jsonl")},
               "admission": {"training_admission": False, "reason": "inherits the corpus hold"}}
    (out / "conversion.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Convert a view-ask corpus into view-stage trajectory records.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus-id", default=None)
    args = parser.parse_args(argv)
    print(json.dumps(convert(args.corpus, args.out, corpus_id=args.corpus_id), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
