#!/usr/bin/env python3
"""Convert labeled Jev-style datasets into natlang's external task JSONL.

Examples (inputs are files already downloaded from their source repositories):

    python scripts/prepare_direct_pairs.py nanojev stage2/train.jsonl -o tasks.jsonl
    python scripts/prepare_direct_pairs.py jeff jeff/bench/data -o jeff-test.jsonl
    python scripts/prepare_direct_pairs.py typed-decisions all/train.parquet -o typed.jsonl

JSONL is supported for every source. Typed Decisions also accepts its original
Parquet shards if ``pyarrow`` is installed. jeff ships only benchmark records;
its adapter always marks them test, regardless of the output filename.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _state(value: Any) -> str:
    return value if isinstance(value, str) else _json(value)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value: Any, field: str) -> dict:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _kind(raw: str) -> str:
    # Both jeff and Typed Decisions use the Jev wire spelling "noul".
    result = {"noul": "boolean", "bool": "boolean"}.get(raw, raw)
    if result not in {"choice", "boolean", "score"}:
        raise ValueError(f"unsupported question type: {raw!r}")
    return result


def _description(value: Any, label: str) -> str:
    """Some source criteria use null for a self-describing option."""
    return value.strip() if isinstance(value, str) and value.strip() else label


def _decision(question: dict, answer: Any) -> tuple[str, list[str], str, dict[str, str] | None]:
    kind = _kind(question["type"])
    criteria = question.get("criteria")
    if kind == "boolean":
        labels = ["false", "true"]
        descriptions = {str(k).lower(): _description(v, str(k).lower()) for k, v in criteria.items()} if isinstance(criteria, dict) else None
        if isinstance(answer, bool):
            gold = str(answer).lower()
        else:
            gold = str(answer).lower()
    elif kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score questions require a criteria list with at least two levels")
        labels = [str(i) for i in range(len(criteria))]
        descriptions = {label: _description(value, label) for label, value in zip(labels, criteria)}
        gold = str(answer)
    else:
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice questions require a criteria object with at least two options")
        labels = list(criteria)
        descriptions = {str(k): _description(v, str(k)) for k, v in criteria.items()}
        gold = str(answer)
    if gold not in labels:
        raise ValueError(f"gold {gold!r} is not in labels {labels!r}")
    return kind, labels, gold, descriptions


def _task(*, source: str, row_id: str, group_id: str, split: str, state: Any,
          key: str, question: dict, answer: Any, gold_source: str,
          source_revision: str | None, license: str | None,
          source_meta: dict | None = None) -> dict:
    kind, labels, gold, criteria = _decision(question, answer)
    instruction = question.get("instructions")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"{row_id}/{key}: missing instructions")
    result = {
        "id": f"{source}/{split}/{row_id}/{key}", "source": source,
        "group_id": f"{source}/{group_id}", "split": split,
        "state": _state(state), "instruction": instruction,
        "kind": kind, "labels": labels, "gold": gold,
        "gold_source": gold_source,
    }
    if criteria:
        result["criteria"] = criteria
    if license:
        result["license"] = license
    if source_revision:
        result["source_revision"] = source_revision
    if source_meta:
        result["source_meta"] = source_meta
    return result


def nanojev(row: dict, *, source_revision: str | None = None) -> Iterator[dict]:
    """Expand each NanoJev state to one task per independently checked question.

    An action with several optimal answers has no single gold value and is
    skipped. Other questions from the same state remain usable.
    """
    meta = row.get("metadata") or {}
    state = row["state"]
    split = row.get("split")
    if split not in {"train", "dev", "validation", "test", "calibration", "ood"}:
        raise ValueError(f"unknown NanoJev split: {split!r}")
    row_id = str(row["id"])
    # source_group_id groups counterfactuals and symmetries; state_id catches
    # stage-1 replay in stage 2 when source_group_id is absent.
    group = str(meta.get("source_group_id") or row.get("state_id") or _digest(_state(state)))
    kinds = row.get("gold_label_kind") or {}
    optimal = row.get("optimal_actions") or {}
    for key, question in row["questions"].items():
        if key not in row["gold"]:
            continue
        if len(optimal.get(key, [])) > 1:
            continue
        label_kind = kinds.get(key) if isinstance(kinds, dict) else kinds
        # Some navigation questions encode a canonical argmax for a tied policy.
        # A unique argmax is fine; an actual multi-answer tie was skipped above.
        yield _task(
            source="nanojev", row_id=row_id, group_id=group, split=split,
            state=state, key=key, question=question, answer=row["gold"][key],
            gold_source="programmatic" if meta.get("source") == "self_authored_programmatic" else "dataset",
            source_revision=source_revision, license=meta.get("license"),
            source_meta={"family_id": row.get("family_id"), "state_id": row.get("state_id"),
                         "question_key": key, "gold_label_kind": label_kind,
                         "gold_probs_kind": (row.get("gold_probs_kind") or {}).get(key)
                         if isinstance(row.get("gold_probs_kind"), dict) else row.get("gold_probs_kind"),
                         "gold_probs": (row.get("gold_probs") or {}).get(key),
                         "teacher_model": (row.get("teacher") or {}).get("model"),
                         "teacher_probs": ((row.get("teacher") or {}).get("native_probs") or {}).get(key)},
        )


def jeff(row: dict, *, source_revision: str | None = None, filename: str | None = None) -> Iterator[dict]:
    """Convert jeff's committed benchmark rows; do not promote them to train."""
    row_id = str(row["id"])
    state = row["state"]
    # Hash of the full state also joins accidental duplicate IDs and repeated
    # passages into one group in downstream split audits.
    group = _digest(_state(state))
    questions = row["questions"]
    gold = row["gold"]
    if not isinstance(gold, dict):
        if len(questions) != 1:
            raise ValueError(f"{row_id}: scalar gold with multiple questions")
        gold = {next(iter(questions)): gold}
    for key, question in questions.items():
        if key not in gold:
            continue
        yield _task(
            source="jeff", row_id=row_id, group_id=group, split="test",
            state=state, key=key, question=question, answer=gold[key],
            gold_source="dataset", source_revision=source_revision,
            license=None, source_meta={"dataset": filename or row_id.split("-")[0],
                                       "original_id": row_id, "question_key": key,
                                       "reuse_terms": "check upstream dataset"},
        )


def typed_decisions(row: dict, *, source_revision: str | None = None) -> Iterator[dict]:
    """Convert LocalLLaMA/typed-decisions JSONL or Parquet row."""
    split = row.get("split")
    if split not in {"train", "test"}:
        raise ValueError(f"unknown Typed Decisions split: {split!r}")
    row_id = str(row["id"])
    questions = _object(row["questions"], "questions")
    gold = _object(row["gold"], "gold")
    state = row["state"]
    # The dataset's "all" config repeats rows in workflow configs. The ID and
    # group are config independent so a merged import can de-duplicate them.
    group = row_id
    for key, question in questions.items():
        if key not in gold:
            continue
        answer = gold[key]
        yield _task(
            source="typed-decisions", row_id=row_id, group_id=group,
            split=split, state=state, key=key, question=question,
            answer=answer["label"] if isinstance(answer, dict) else answer,
            gold_source="synthetic", source_revision=source_revision,
            license="Apache-2.0",
            source_meta={"workflow": row.get("workflow"), "question_key": key,
                         "teacher_confidence": answer.get("confidence") if isinstance(answer, dict) else None,
                         "teacher_probabilities": answer.get("probabilities") if isinstance(answer, dict) else None,
                         "teacher_agreement": (_object(row["label_agreement"], "label_agreement").get(key)
                                               if row.get("label_agreement") else None)},
        )


ADAPTERS = {"nanojev": nanojev, "jeff": jeff, "typed-decisions": typed_decisions}


def _rows(path: Path) -> Iterator[dict]:
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise SystemExit("Parquet input needs pyarrow; use a JSONL export or install pyarrow") from exc
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256):
            yield from batch.to_pylist()
        return
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc


def _inputs(paths: list[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix in {".jsonl", ".parquet"})
        else:
            yield path


def convert(source: str, paths: Iterable[Path], output: Path,
            *, source_revision: str | None = None, limit: int | None = None,
            nanojev_families: set[str] | None = None) -> Counter:
    adapter = ADAPTERS[source]
    counts: Counter = Counter()
    seen: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as out:
        for path in _inputs(list(paths)):
            for row in _rows(path):
                if source == "nanojev" and nanojev_families and row.get("family_id") not in nanojev_families:
                    counts["excluded_family"] += 1
                    continue
                kw = {"source_revision": source_revision}
                if source == "jeff":
                    kw["filename"] = path.stem
                for task in adapter(row, **kw):
                    if task["id"] in seen:
                        counts["duplicate_tasks"] += 1
                        continue
                    seen.add(task["id"])
                    out.write(_json(task) + "\n")
                    counts[task["split"]] += 1
                    if limit is not None and sum(v for k, v in counts.items()
                                                 if k not in {"duplicate_tasks", "excluded_family"}) >= limit:
                        return counts
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", choices=ADAPTERS)
    parser.add_argument("inputs", type=Path, nargs="+", help="JSONL files, directories, or Typed Decisions Parquet files")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--source-revision", help="Git commit or HF dataset revision for reproducibility")
    parser.add_argument("--limit", type=int, help="Maximum emitted tasks for a pilot")
    parser.add_argument("--nanojev-families", nargs="+",
                        help="include only these NanoJev family_id values")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.nanojev_families and args.source != "nanojev":
        parser.error("--nanojev-families applies only to nanojev")
    counts = convert(args.source, args.inputs, args.output,
                     source_revision=args.source_revision, limit=args.limit,
                     nanojev_families=set(args.nanojev_families) if args.nanojev_families else None)
    manifest = {"source": args.source, "source_revision": args.source_revision,
                "inputs": {str(path): _file_hash(path) for path in _inputs(args.inputs)},
                "output_sha256": _file_hash(args.output), "counts": dict(counts),
                "limit": args.limit, "nanojev_families": args.nanojev_families}
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(_json(dict(counts)), file=sys.stderr)


if __name__ == "__main__":
    main()
