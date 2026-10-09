#!/usr/bin/env python3
"""Source-review registry manifests, reviewer receipts and explicit decisions.

The registry (`training/source-reviews/`) holds the verdicts that keep disputed source items out of generation and
training. The natlang reviewers (`reviewSourceItem`, `reviewSourceRow`) only recommend. This tool records which
reviewer said what (by the hash of its exact definition), and the explicit decision that followed:

  receipt  <review-output.json[l]>   write one receipt per reviewer output (decision null, training_admission false)
  decide   <receipt id> --decision hold|clear|defer --by owner|agent:<session> [--note ...]
  manifest --id source-reviews-<date>-v<n>   write the immutable SHA-256 manifest of the registry
  show     <receipt id>

A `hold` decision on an item receipt appends the proposed entry to holds.jsonl as a pending record and writes a new
manifest. `clear` and `defer` never change the registry. No receipt grants training admission: admission, split and
quality decisions stay explicit in training/neuralese_corpora.json.

The same module maps row receipts to the review vocabulary of the S1 semantic reviews (v7 categories) and to the
dispositions the packet builders require, so build_*_review_packet.py and bind_*_receipts.py read reviewSourceRow
output where they used to read an outside annotation file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_hex  # noqa: E402
from natlang_neuralese.common.jsonio import canonical_json_str  # noqa: E402

REGISTRY = ROOT / "training" / "source-reviews"
MANIFESTS = ROOT / "training" / "corpus-manifests"
RECEIPT_SCHEMA = "natlang.source-review-receipt/1"
MANIFEST_SCHEMA = "natlang.source-review-registry-manifest/1"
REGISTRY_FILES = ("holds.jsonl", "family-holds.json", "datasets.json")
REVIEW_KEYS = ("dataset", "id", "aliases", "text", "annotatedLabel", "status", "reason",
               "sourceRevision", "sourceSnapshotSha256", "sourcePrompt")
ITEM_CONCERNS = ("none", "label-disagrees-with-source", "unstated-premise", "ambiguous-question",
                 "contract-mismatch", "needs-context")
ROW_STATUSES = ("equivalent", "normalization-candidate", "mismatch", "ambiguous")
CONFIDENCES = ("high", "medium", "low")
DECISIONS = ("hold", "clear", "defer")
BY_PATTERN = re.compile(r"^(owner|agent:\S.*)$")
HASH_PATTERN = re.compile(r"^natlang@[0-9a-f]{16}$")

# The categories of plans/neuralese/S1_QUESTION_COLLECTION_V7_SEMANTIC_REVIEW_*.json, by row status.
V7_CATEGORY = {
    "equivalent": "receipts",
    "normalization-candidate": "semantic_normalization_candidate",
    "mismatch": "strict_semantic_mismatches",
    "ambiguous": "semantic_ambiguities",
}
# What a decision means to a packet builder: hold stays held; clear makes the row a candidate (never an admission).
DISPOSITION = {"hold": "hold", "clear": "candidate"}


class ReviewError(ValueError):
    """A receipt, decision or registry that does not hold; the message says what to write instead."""


def canonical_sha256(value: Any) -> str:
    return canonical_json_sha256_hex(value)


def reviewer_hash(definition_source_sha256: str, compiler_version: str, executor: str) -> str:
    """`natlang@` and the first 16 hex of sha256(definition source hash, compiler version, executor identity)."""
    digest = sha256_hex(f"{definition_source_sha256}\0{compiler_version}\0{executor}".encode("utf-8"))
    return "natlang@" + digest[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReviewError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


# ---------------------------------------------------------------------------------------------------------- manifests

def registry_sha256(registry: Path = REGISTRY) -> str:
    return sha256_hex((registry / "holds.jsonl").read_bytes())


def build_manifest(manifest_id: str, registry: Path = REGISTRY, previous: str | None = None) -> dict[str, Any]:
    files = []
    for name in REGISTRY_FILES:
        raw = (registry / name).read_bytes()
        entry: dict[str, Any] = {"path": f"training/source-reviews/{name}", "bytes": len(raw), "sha256": sha256_hex(raw)}
        if name.endswith(".jsonl"):
            entry["lines"] = sum(1 for line in raw.splitlines() if line.strip())
        files.append(entry)
    reviews = read_jsonl(registry / "holds.jsonl")
    per_dataset: dict[str, int] = {}
    for review in reviews:
        per_dataset[review["dataset"]] = per_dataset.get(review["dataset"], 0) + 1
    family = json.loads((registry / "family-holds.json").read_text(encoding="utf-8"))
    return {
        "schema": MANIFEST_SCHEMA,
        "id": manifest_id,
        "previous": previous,
        "files": files,
        "record_count": len(reviews),
        "pending_count": sum(review["status"] == "pending" for review in reviews),
        "identity_count": sum(1 + len(review["aliases"]) for review in reviews),
        "per_dataset": dict(sorted(per_dataset.items())),
        "family_hold_count": len(family["holds"]),
        "training_admission": False,
        "note": "A registry of source-review holds, not a training corpus; no entry grants admission.",
    }


def write_manifest(manifest: dict[str, Any], manifests: Path = MANIFESTS) -> Path:
    path = manifests / f"{manifest['id']}.json"
    if path.exists():
        raise ReviewError(f"{path} exists; manifests are immutable, so choose a new id")
    manifests.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path


def next_manifest_id(day: str, manifests: Path = MANIFESTS) -> tuple[str, str | None]:
    """The next free `source-reviews-<day>-v<n>` id and the id of the newest manifest before it."""
    existing = sorted(int(m.group(1)) for p in manifests.glob(f"source-reviews-{day}-v*.json")
                      if (m := re.fullmatch(rf"source-reviews-{day}-v(\d+)\.json", p.name)))
    number = (existing[-1] + 1) if existing else 1
    older = sorted(p.stem for p in manifests.glob("source-reviews-*-v*.json"))
    return f"source-reviews-{day}-v{number}", (older[-1] if older else None)


# ------------------------------------------------------------------------------------------------------------ receipts

def agreement(kind: str, first: dict[str, Any] | None, second: dict[str, Any] | None) -> str:
    if second is None:
        return "single"
    key = "recommendation" if kind == "item" else "status"
    return "agree" if (first or {}).get(key) == second.get(key) else "disagree"


def _check_reviewer(reviewer: dict[str, Any], where: str) -> None:
    if reviewer.get("kind") == "crisp":
        return  # an exact precheck: no natlang definition to hash
    if reviewer.get("kind") != "natlang":
        raise ReviewError(f"{where}.kind is \"natlang\" or \"crisp\"")
    inputs = reviewer.get("hash_inputs") or {}
    expected = reviewer_hash(str(inputs.get("definition_source_sha256")), str(inputs.get("compiler_version")),
                             str(inputs.get("executor")))
    if reviewer.get("reviewer_hash") != expected or not HASH_PATTERN.match(str(reviewer.get("reviewer_hash"))):
        raise ReviewError(f"{where}.reviewer_hash must equal {expected}, the hash of its definition source, compiler "
                          "version and executor identity (hash_inputs)")


def check_recommendation(kind: str, recommendation: dict[str, Any], subject_input: dict[str, Any]) -> None:
    """The crisp checks that the typed contract of each function states; failures say what to write instead."""
    if kind == "item":
        verdict, concern = recommendation.get("recommendation"), recommendation.get("concern")
        if verdict not in ("admit", "hold") or concern not in ITEM_CONCERNS:
            raise ReviewError("an item recommendation is admit or hold, with one of the concerns " + ", ".join(ITEM_CONCERNS))
        if (concern == "none") != (verdict == "admit"):
            raise ReviewError("the concern is none exactly when the recommendation is admit")
        proposed = recommendation.get("proposed_entry")
        if (verdict == "hold") != (isinstance(proposed, dict) and isinstance(proposed.get("reason"), str) and bool(proposed["reason"].strip())):
            raise ReviewError("proposed_entry with a reason is present exactly when the recommendation is hold")
        source = (subject_input.get("item") or {}).get("visible", "")
    else:
        if recommendation.get("status") not in ROW_STATUSES:
            raise ReviewError("a row verdict status is one of " + ", ".join(ROW_STATUSES))
        precheck = subject_input.get("precheck") or {}
        if precheck.get("exact_match") is True and recommendation["status"] != "equivalent":
            raise ReviewError("the status is equivalent whenever precheck.exact_match is true")
        source = (subject_input.get("row") or {}).get("evidence", "")
    if recommendation.get("confidence") not in CONFIDENCES:
        raise ReviewError("confidence is high, medium or low")
    for number, entry in enumerate(recommendation.get("evidence") or []):
        quote = entry.get("quote") if isinstance(entry, dict) else None
        if not isinstance(quote, str) or quote not in source:
            raise ReviewError(f"evidence[{number}].quote must occur in the " + ("item's visible text" if kind == "item" else "row evidence"))


def receipt_id(receipt: dict[str, Any]) -> str:
    """Identity of what was reviewed and recommended; the decision and registry fields do not change it."""
    basis = {key: receipt.get(key) for key in ("schema", "kind", "subject", "reviewer", "second_reviewer", "recommendation", "excluded")}
    return canonical_sha256(basis)[:24]


def build_receipt(output: dict[str, Any], registry: Path = REGISTRY) -> dict[str, Any]:
    """A receipt from one reviewer output: {kind, input, reviewer, recommendation, second_reviewer?, second_recommendation?}."""
    kind = output.get("kind")
    if kind not in ("item", "row"):
        raise ReviewError("a reviewer output has kind \"item\" or \"row\"")
    subject_input = output.get("input")
    if not isinstance(subject_input, dict):
        raise ReviewError("a reviewer output carries the reviewed input as `input`")
    identity = subject_input.get("item" if kind == "item" else "row")
    if not isinstance(identity, dict):
        raise ReviewError("the reviewed input carries `item` (kind item) or `row` (kind row)")
    flat = subject_input
    reviewer, second = output.get("reviewer"), output.get("second_reviewer")
    recommendation, excluded = output.get("recommendation"), output.get("excluded")
    if not isinstance(reviewer, dict):
        raise ReviewError("a reviewer output names its reviewer")
    _check_reviewer(reviewer, "reviewer")
    if second is not None:
        _check_reviewer(second, "second_reviewer")
    if recommendation is None:
        if excluded != "transport_error":
            raise ReviewError("a reviewer output without a recommendation says excluded: \"transport_error\"")
    else:
        check_recommendation(kind, recommendation, flat)
    second_recommendation = output.get("second_recommendation")
    if second is not None and second_recommendation is None:
        raise ReviewError("a second reviewer comes with its second_recommendation")
    if second_recommendation is not None:
        check_recommendation(kind, second_recommendation, flat)
    dataset = identity.get("dataset")
    subject_id = identity.get("id")
    if not isinstance(subject_id, str) or not subject_id:
        raise ReviewError("the reviewed input names an id (item.id, or row.id for a row)")
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "kind": kind,
        "subject": {"dataset": dataset, "id": subject_id, "subject_sha256": canonical_sha256(subject_input),
                    "input": subject_input},
        "reviewer": reviewer,
        "second_reviewer": ({**second, "recommendation": second_recommendation} if second else None),
        "recommendation": recommendation,
        "agreement": agreement(kind, recommendation, second_recommendation),
        "decision": None,
        "registry": {"before_sha256": registry_sha256(registry), "after_sha256": registry_sha256(registry)},
        "training_admission": False,
    }
    if excluded:
        receipt["excluded"] = excluded
    receipt["id"] = receipt_id(receipt)
    return receipt


def validate_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ReviewError(f"receipt schema is {RECEIPT_SCHEMA}")
    if receipt.get("training_admission") is not False:
        raise ReviewError("a source-review receipt never grants training admission (training_admission false)")
    subject = receipt.get("subject") or {}
    if subject.get("subject_sha256") != canonical_sha256(subject.get("input")):
        raise ReviewError("subject_sha256 differs from the canonical JSON of the reviewed input")
    if receipt.get("id") != receipt_id(receipt):
        raise ReviewError("the receipt id differs from the hash of what was reviewed and recommended")
    _check_reviewer(receipt.get("reviewer") or {}, "reviewer")
    decision = receipt.get("decision")
    if decision is not None:
        if decision.get("value") not in DECISIONS or not BY_PATTERN.match(str(decision.get("by"))):
            raise ReviewError("a decision is hold, clear or defer, signed by owner or agent:<session>")


def receipt_path(identifier: str, receipts: Path) -> Path:
    return receipts / f"{identifier}.json"


def save_receipt(receipt: dict[str, Any], receipts: Path, replace: bool = False) -> Path:
    receipts.mkdir(parents=True, exist_ok=True)
    path = receipt_path(receipt["id"], receipts)
    if path.exists() and not replace:
        raise ReviewError(f"receipt {receipt['id']} exists; decide it with `decide` rather than writing it again")
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def load_receipt(identifier: str, receipts: Path) -> dict[str, Any]:
    path = receipt_path(identifier, receipts)
    if not path.is_file():
        raise ReviewError(f"no receipt {identifier} under {receipts}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt)
    return receipt


# ------------------------------------------------------------------------------------------------------------ decisions

def decide(identifier: str, value: str, by: str, note: str = "", *, registry: Path = REGISTRY, receipts: Path | None = None,
           manifests: Path = MANIFESTS, now: datetime | None = None) -> dict[str, Any]:
    """Record the explicit decision. Only a `hold` on an item receipt changes the registry (a new pending record and manifest)."""
    receipts = receipts or registry / "receipts"
    now = now or datetime.now(timezone.utc)
    if value not in DECISIONS:
        raise ReviewError("the decision is hold, clear or defer")
    if not BY_PATTERN.match(by):
        raise ReviewError("sign the decision as owner or agent:<session name>")
    receipt = load_receipt(identifier, receipts)
    previous = receipt.get("decision")
    if previous and previous["value"] != "defer":
        raise ReviewError(f"receipt {identifier} was decided ({previous['value']} by {previous['by']}); write a new review to change it")
    if value == "hold" and receipt.get("recommendation") is None:
        raise ReviewError("a hold decision needs a recommendation; this receipt has none (transport error)")
    before = registry_sha256(registry)
    if value == "hold" and receipt["kind"] == "item":
        item = receipt["subject"]["input"]["item"]
        proposed = (receipt.get("recommendation") or {}).get("proposed_entry")
        if not isinstance(proposed, dict) or not proposed.get("reason"):
            raise ReviewError("a hold decision on an item needs the recommendation's proposed_entry; the reviewer recommended admit")
        entry = {"dataset": item["dataset"], "id": item["id"], "aliases": [], "text": item["visible"],
                 "annotatedLabel": item["annotated_label"], "status": "pending", "reason": proposed["reason"]}
        existing = read_jsonl(registry / "holds.jsonl")
        taken = {(row["dataset"], identity) for row in existing for identity in [row["id"], *row["aliases"]]}
        if (entry["dataset"], entry["id"]) in taken:
            raise ReviewError(f"{entry['dataset']} {entry['id']} already has a registry entry; the entry is the standing decision")
        with open(registry / "holds.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({key: entry[key] for key in REVIEW_KEYS if key in entry},
                                    ensure_ascii=False, separators=(",", ":")) + "\n")
        manifest_id, older = next_manifest_id(now.strftime("%Y%m%d"), manifests)
        write_manifest(build_manifest(manifest_id, registry, older), manifests)
        receipt["registry"] = {"before_sha256": before, "after_sha256": registry_sha256(registry), "manifest": manifest_id}
    else:
        receipt["registry"] = {"before_sha256": before, "after_sha256": before}
    receipt["decision"] = {"value": value, "by": by, "at": now.isoformat(), "note": note}
    save_receipt(receipt, receipts, replace=True)
    return receipt


# ------------------------------------------------------------------------- reading row receipts (packet scripts)

def v7_category(receipt: dict[str, Any]) -> str:
    """The category of the S1 semantic review that this row receipt falls in."""
    if receipt.get("recommendation") is None:
        return "transport_error"
    precheck = (receipt["subject"]["input"].get("precheck") or {})
    if precheck.get("numerically_equal") is True and precheck.get("exact_match") is not True:
        return "held_numeric_representation"
    return V7_CATEGORY[receipt["recommendation"]["status"]]


def load_row_receipts(path: Path) -> dict[str, dict[str, Any]]:
    """Row receipts by subject id, from a receipt file, a JSONL of receipts or a folder of receipt files."""
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        values = [json.loads(file.read_text(encoding="utf-8")) for file in files]
    elif path.suffix == ".jsonl":
        values = read_jsonl(path)
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        values = loaded if isinstance(loaded, list) else [loaded]
    by_id: dict[str, dict[str, Any]] = {}
    for value in values:
        validate_receipt(value)
        if value["kind"] != "row":
            raise ReviewError(f"{path}: receipt {value['id']} is an item receipt; the packets read row receipts")
        subject_id = value["subject"]["id"]
        if subject_id in by_id:
            raise ReviewError(f"{path}: two row receipts for {subject_id}")
        by_id[subject_id] = value
    return by_id


def row_assessment(receipt: dict[str, Any]) -> dict[str, Any]:
    """The assessment a packet builder records for a row: the receipt's decision as disposition, never an admission."""
    decision = receipt.get("decision")
    if not decision or decision["value"] not in DISPOSITION:
        raise ReviewError(f"row {receipt['subject']['id']} has no explicit hold or clear decision; record one with "
                          f"`source_review.py decide {receipt['id']} --decision hold|clear --by owner`")
    recommendation = receipt.get("recommendation") or {}
    return {
        "disposition": DISPOSITION[decision["value"]],
        "v7_category": v7_category(receipt),
        "review_status": recommendation.get("status"),
        "rationale": recommendation.get("rationale"),
        "reviewer": receipt["reviewer"].get("reviewer_hash"),
        "receipt_id": receipt["id"],
        "decision_by": decision["by"],
        "training_admission": False,
    }


def annotations_from_row_receipts(path: Path) -> dict[str, Any]:
    """The `{"rows": {id: assessment}}` document that the packet builders used to read from an annotation file."""
    return {"schema": "natlang.source-review-row-annotations/1",
            "rows": {subject_id: row_assessment(receipt) for subject_id, receipt in load_row_receipts(path).items()}}


# ------------------------------------------------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--receipts", type=Path, default=None, help="receipt folder (default: <registry>/receipts)")
    parser.add_argument("--manifests", type=Path, default=MANIFESTS)
    sub = parser.add_subparsers(dest="command", required=True)
    receipt = sub.add_parser("receipt")
    receipt.add_argument("outputs", type=Path)
    decision = sub.add_parser("decide")
    decision.add_argument("receipt_id")
    decision.add_argument("--decision", required=True, choices=DECISIONS)
    decision.add_argument("--by", required=True)
    decision.add_argument("--note", default="")
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--id", required=True)
    manifest.add_argument("--previous", default=None)
    show = sub.add_parser("show")
    show.add_argument("receipt_id")
    args = parser.parse_args(argv)
    receipts = args.receipts or args.registry / "receipts"
    try:
        if args.command == "receipt":
            text = args.outputs.read_text(encoding="utf-8")
            outputs = read_jsonl(args.outputs) if args.outputs.suffix == ".jsonl" else (
                json.loads(text) if text.lstrip().startswith("[") else [json.loads(text)])
            for output in outputs:
                built = build_receipt(output, args.registry)
                save_receipt(built, receipts)
                print(json.dumps({"receipt": built["id"], "kind": built["kind"], "agreement": built["agreement"],
                                  "decision": None, "training_admission": False}))
        elif args.command == "decide":
            built = decide(args.receipt_id, args.decision, args.by, args.note, registry=args.registry,
                           receipts=receipts, manifests=args.manifests)
            print(json.dumps({"receipt": built["id"], "decision": built["decision"], "registry": built["registry"]}))
        elif args.command == "manifest":
            path = write_manifest(build_manifest(args.id, args.registry, args.previous), args.manifests)
            print(json.dumps({"manifest": str(path), "sha256": sha256_hex(path.read_bytes())}))
        else:
            print(json.dumps(load_receipt(args.receipt_id, receipts), ensure_ascii=False, indent=2))
    except ReviewError as error:
        print(f"source_review: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
