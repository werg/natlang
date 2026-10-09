"""refine-calibration (plans/REFINEMENT_TYPES.md section 5.4): human-reviewed verdicts that measure the judge's calibration.

Files (JSONL, one record per line):

  queue.jsonl     items to review, `natlang.refine-calibration-item/1`:
                  {id, predicate, predicate_id, family, value, value_sha256, source, stratum}
                  Built by `build_queue` from refine-judge rows. Model probabilities are never stored in the queue, so a
                  reviewer cannot be anchored by them.
  verdicts.jsonl `natlang.refine-calibration/1`, append-only:
                  {id (item id), predicate, predicate_id, family, value, value_sha256, verdict: true|false|"unclear",
                   reviewer, reviewed_at, note, review_round}
                  The latest verdict for an item wins; earlier ones stay in the file as history.

Verdicts are for evaluation only: they are held-out by construction (role `heldout`, split `calibration`) and never enter a
training corpus. `unclear` verdicts are kept for predicate diagnosis and excluded from the metrics. The metrics are the
Brier score and the expected calibration error (equal-width bins) of the judge's P(true) against the human verdict, per
predicate family and overall; the gate that uses them is in plans/REFINEMENT_DATA.md.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from .common import judge_row, read_jsonl, sha256_hex, utc_now_iso

ITEM_SCHEMA = "natlang.refine-calibration-item/1"
VERDICT_SCHEMA = "natlang.refine-calibration/1"
VERDICTS = (True, False, "unclear")
FAMILY = "refine-judge/calibration"


def build_queue(rows: list[dict], per_predicate: int = 4, seed: str = "refine-calibration/1", sources: tuple[str, ...] | None = None) -> list[dict]:
    """A review queue from judge rows: up to `per_predicate` items per predicate, balanced across the rows' gold labels.

    The queue draws from rows of any split (calibration measures the judge, not training); it records the source so that
    reports can separate synthetic near-misses from mined values.
    """
    by_predicate: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if sources and row.get("source") not in sources:
            continue
        by_predicate[row["predicate_id"]].append(row)
    queue = []
    for pid in sorted(by_predicate):
        rng = random.Random(sha256_hex(f"{seed}|{pid}".encode())[:16])
        group = sorted(by_predicate[pid], key=lambda row: row["id"])
        positives = [row for row in group if row["gold"][0] >= 0.5]
        negatives = [row for row in group if row["gold"][0] < 0.5]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        picked = []
        while len(picked) < per_predicate and (positives or negatives):
            for pool in (positives, negatives):
                if pool and len(picked) < per_predicate:
                    picked.append(pool.pop())
        for row in picked:
            value = _value_of(row)
            queue.append({"schema": ITEM_SCHEMA, "id": "calibration:" + row["id"], "predicate": row["predicate"], "predicate_id": pid,
                          "family": row["family"], "value": value, "value_sha256": row["value_sha256"], "source": row.get("source"),
                          "stratum": "positive" if row["gold"][0] >= 0.5 else "negative"})
    return queue


def _value_of(row: dict):
    """The value as the judge saw it: the text between the markers of the row's prompt."""
    content = row["messages"][1]["content"]
    return content.split("<<<value\n", 1)[1].rsplit("\nvalue>>>", 1)[0]


def latest_verdicts(verdicts: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for record in verdicts:
        out[record["id"]] = record
    return out


def make_verdict(item: dict, verdict, reviewer: str, note: str = "", review_round: int = 1) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be true, false or 'unclear', got {verdict!r}")
    return {"schema": VERDICT_SCHEMA, "id": item["id"], "predicate": item["predicate"], "predicate_id": item["predicate_id"],
            "family": item["family"], "value": item["value"], "value_sha256": item["value_sha256"], "verdict": verdict,
            "reviewer": reviewer, "reviewed_at": utc_now_iso(), "note": note, "review_round": review_round}


def pending(queue: list[dict], verdicts: list[dict]) -> list[dict]:
    done = latest_verdicts(verdicts)
    return [item for item in queue if item["id"] not in done]


PROMPT = "[t]rue / [f]alse / [u]nclear / [s]kip / [q]uit > "
ANSWERS = {"t": True, "true": True, "y": True, "f": False, "false": False, "n": False, "u": "unclear", "unclear": "unclear"}


def review(queue: list[dict], verdict_path: Path, reviewer: str, ask: Callable[[str], str] = input,
           show: Callable[[str], None] = print, limit: int | None = None) -> dict:
    """Interactive review: show each pending item, append the verdict. `quit` stops; `skip` leaves the item pending."""
    existing = read_jsonl(verdict_path) if verdict_path.exists() else []
    todo = pending(queue, existing)
    stats = {"pending_at_start": len(todo), "recorded": 0, "skipped": 0}
    with open(verdict_path, "a", encoding="utf-8") as stream:
        for number, item in enumerate(todo, 1):
            if limit is not None and stats["recorded"] >= limit:
                break
            show(f"\n--- {number}/{len(todo)}  {item['family']}\nProperty: the value is {item['predicate']}\n<<<value\n{item['value']}\nvalue>>>")
            while True:
                answer = ask(PROMPT).strip().lower()
                if answer in ("q", "quit"):
                    return stats
                if answer in ("s", "skip", ""):
                    stats["skipped"] += 1
                    break
                if answer in ANSWERS:
                    note = ask("note (optional) > ").strip() if answer in ("u", "unclear") else ""
                    stream.write(json.dumps(make_verdict(item, ANSWERS[answer], reviewer, note), ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    stats["recorded"] += 1
                    break
                show("answer t, f, u, s or q")
    return stats


def status(queue: list[dict], verdicts: list[dict]) -> dict:
    done = latest_verdicts(verdicts)
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"queued": 0, "true": 0, "false": 0, "unclear": 0})
    for item in queue:
        by_family[item["family"]]["queued"] += 1
        record = done.get(item["id"])
        if record:
            by_family[item["family"]][str(record["verdict"]).lower()] += 1
    return {"queued": len(queue), "reviewed": sum(1 for item in queue if item["id"] in done), "by_family": dict(sorted(by_family.items()))}


def to_rows(verdicts: list[dict]) -> list[dict]:
    """Evaluation rows (decision-prompt format, role heldout, split `calibration`) for the clear verdicts."""
    rows = []
    for record in sorted(latest_verdicts(verdicts).values(), key=lambda r: r["id"]):
        if record["verdict"] == "unclear":
            continue
        rows.append(judge_row(row_id=record["id"], family=FAMILY, split="calibration", value=record["value"], predicate=record["predicate"],
                              gold_true=1.0 if record["verdict"] is True else 0.0, source="human-review", label_source=f"human:{record['reviewer']}",
                              review_round=record["review_round"]))
    return rows


def brier(pairs: list[tuple[float, bool]]) -> float:
    return sum((p - float(label)) ** 2 for p, label in pairs) / len(pairs)


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    total, error = len(pairs), 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [(p, label) for p, label in pairs if low <= p < high or (index == bins - 1 and p == 1.0)]
        if members:
            confidence = sum(p for p, _ in members) / len(members)
            accuracy = sum(label for _, label in members) / len(members)
            error += len(members) / total * abs(confidence - accuracy)
    return error


def calibration_report(verdicts: list[dict], predictions: dict[str, float]) -> dict:
    """Brier and ECE of `predictions` (item id -> P(true)) against the clear human verdicts, overall and per family."""
    groups: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for record in latest_verdicts(verdicts).values():
        if record["verdict"] == "unclear" or record["id"] not in predictions:
            continue
        pair = (float(predictions[record["id"]]), record["verdict"] is True)
        groups["all"].append(pair)
        groups[record["family"]].append(pair)
    return {name: {"n": len(pairs), "brier": round(brier(pairs), 6), "ece": round(ece(pairs), 6)} for name, pairs in sorted(groups.items())}
