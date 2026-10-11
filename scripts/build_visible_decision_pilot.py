#!/usr/bin/env python3
"""Build a stratified, train-only 48-sample pilot per visible task variant."""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "neuralese"))
from natlang_neuralese.common.hashing import sha256_hex


def rank(prefix: str, row: dict) -> str:
    return sha256_hex(f"{prefix}:{row['id']}".encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases_bytes, bindings_bytes, parent_bytes = (args.cases.read_bytes(), args.bindings.read_bytes(),
        args.parent_manifest.read_bytes())
    cases = [json.loads(line) for line in cases_bytes.decode().splitlines() if line.strip()]
    bindings_doc = json.loads(bindings_bytes)
    if bindings_doc.get("schema") != "natlang.visible-decision-task-bindings/1":
        raise SystemExit("unsupported bindings schema")
    by_id = {row["variant_id"]: row for row in bindings_doc["bindings"]}
    if len(by_id) != len(bindings_doc["bindings"]) or set(by_id) != {row["id"] for row in cases}:
        raise SystemExit("case and source-binding IDs do not agree exactly")
    if any(row.get("role") != "train" for row in cases) or any(row.get("source_role") != "train" for row in by_id.values()):
        raise SystemExit("pilot includes a non-train source")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.out}")

    toxicity = [row for row in cases if row["family"] == "decision:toxicity-binary-threshold-v1"]
    satisfaction = [row for row in cases if row["family"] == "decision:review-satisfaction-estimate-v1"]
    positives = [row for row in toxicity if row["answer"] is True]
    negatives = [row for row in toxicity if row["answer"] is False]
    if len(positives) < 24 or len(negatives) < 24 or len(satisfaction) < 48:
        raise SystemExit("not enough source cases for the 48-per-variant pilot")

    positive_bins = {"boundary": [], "moderate": [], "clear": []}
    for row in positives:
        score = float(by_id[row["id"]]["source_annotation"])
        stratum = "boundary" if score < 0.6 else "moderate" if score <= 0.8 else "clear"
        positive_bins[stratum].append(row)
    selected_positive = []
    for stratum in ("boundary", "moderate", "clear"):
        ranked = sorted(positive_bins[stratum], key=lambda row: rank(f"pilot-tox-{stratum}-v1", row))
        if len(ranked) < 8:
            raise SystemExit(f"toxicity positive stratum {stratum} has {len(ranked)} cases; 8 required")
        selected_positive.extend(ranked[:8])

    negative_bins = {"low": [], "near": []}
    for row in negatives:
        score = float(by_id[row["id"]]["source_annotation"])
        negative_bins["low" if score < 0.1 else "near"].append(row)
    selected_negative = []
    # Round-robin over available negative source-score bands. When one band is
    # short, the other fills the sample without discarding useful train rows.
    ranked_negative = {key: sorted(value, key=lambda row: rank(f"pilot-tox-negative-{key}-v1", row))
        for key, value in negative_bins.items()}
    while len(selected_negative) < 24:
        progressed = False
        for key in ("low", "near"):
            if ranked_negative[key] and len(selected_negative) < 24:
                selected_negative.append(ranked_negative[key].pop(0))
                progressed = True
        if not progressed:
            raise SystemExit("failed to select 24 negative toxicity examples")

    satisfaction_bins = {}
    for row in satisfaction:
        satisfaction_bins.setdefault(str(row["answer"]), []).append(row)
    selected_satisfaction = []
    while any(satisfaction_bins.values()):
        for label in sorted(satisfaction_bins):
            if satisfaction_bins[label]:
                selected_satisfaction.append(satisfaction_bins[label].pop(0))
    selected_satisfaction = selected_satisfaction[:48]
    selected = sorted(selected_positive + selected_negative + selected_satisfaction,
        key=lambda row: (row["family"], row["id"]))
    if len(selected) != 96 or len({row["id"] for row in selected}) != 96:
        raise SystemExit("pilot selection must contain 96 unique rows")

    args.out.mkdir(parents=True, exist_ok=True)
    selected_bytes = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in selected) + "\n").encode()
    selected_bindings = {"schema": "natlang.visible-decision-task-bindings/1",
        "bindings": [by_id[row["id"]] for row in selected]}
    selected_bindings_bytes = (json.dumps(selected_bindings, ensure_ascii=False, indent=2) + "\n").encode()
    (args.out / "cases.jsonl").write_bytes(selected_bytes)
    (args.out / "source-bindings.json").write_bytes(selected_bindings_bytes)

    toxicity_strata = {"positive_boundary": 0, "positive_moderate": 0, "positive_clear": 0,
        "negative_low": 0, "negative_near": 0}
    for row in selected_positive:
        score = float(by_id[row["id"]]["source_annotation"])
        key = "positive_boundary" if score < 0.6 else "positive_moderate" if score <= 0.8 else "positive_clear"
        toxicity_strata[key] += 1
    for row in selected_negative:
        score = float(by_id[row["id"]]["source_annotation"])
        toxicity_strata["negative_low" if score < 0.1 else "negative_near"] += 1
    labels = {family: {str(value): sum(row["family"] == family and str(row["answer"]) == str(value)
        for row in selected) for value in sorted({str(row["answer"]) for row in selected if row["family"] == family})}
        for family in ("decision:toxicity-binary-threshold-v1", "decision:review-satisfaction-estimate-v1")}
    manifest = {"schema": "natlang.visible-decision-task-pilot-manifest/1", "training_admission": False,
        "review_only": True, "cases_sha256": sha256_hex(selected_bytes), "source_bindings_sha256": sha256_hex(selected_bindings_bytes),
        "parent_cases_sha256": sha256_hex(cases_bytes), "parent_source_bindings_sha256": sha256_hex(bindings_bytes),
        "parent_source_manifest_sha256": sha256_hex(parent_bytes), "count": len(selected),
        "families": {"decision:toxicity-binary-threshold-v1": 48,
            "decision:review-satisfaction-estimate-v1": 48}, "answer_counts": labels,
        "toxicity_source_annotation_strata": toxicity_strata,
        "selection": "48 train examples per variant; 24 toxicity positives across boundary/moderate/clear source-label strata plus 24 negatives, and 48 satisfaction cases balanced over the five source-rating bins; deterministic hash rank within strata",
        "live_prompt_sampling": {"required": True, "planned_samples_per_variant": 48, "launched": False,
            "report_per_class_and_source_annotation_strata": True,
            "raw_accuracy_alone_sufficient": False}, "independent_new_worlds": 0}
    (args.out / "pilot-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"count": len(selected), "families": manifest["families"], "answer_counts": labels,
        "toxicity_source_annotation_strata": toxicity_strata, "cases_sha256": manifest["cases_sha256"],
        "bindings_sha256": manifest["source_bindings_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
