#!/usr/bin/env python3
"""Build versioned, source-bound tasks whose requested answer is visible-text based."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


TOXICITY_QUESTION = (
    "Based on the visible comment, is it more likely than not toxic? Treat insults, threats, hateful or demeaning language, and attacks on someone for their identity as toxic. Mere disagreement or criticism is not enough. Return true for toxic and false otherwise."
)
SATISFACTION_QUESTION = (
    "Estimate the reviewer's overall satisfaction from the visible review using the supplied 1–5 criteria. "
    "Choose the level that best matches the overall sentiment expressed in the text."
)
SATISFACTION_CRITERIA = {
    "1 star": "Very dissatisfied overall; the review is strongly negative.",
    "2 star": "Dissatisfied overall; negative impressions outweigh positive ones.",
    "3 stars": "Mixed or neutral overall; positive and negative impressions are balanced or muted.",
    "4 stars": "Satisfied overall; positive impressions outweigh some limitations.",
    "5 stars": "Very satisfied overall; the review is strongly positive.",
    "1": "Very dissatisfied overall; the review is strongly negative.",
    "2": "Dissatisfied overall; negative impressions outweigh positive ones.",
    "3": "Mixed or neutral overall; positive and negative impressions are balanced or muted.",
    "4": "Satisfied overall; positive impressions outweigh some limitations.",
    "5": "Very satisfied overall; the review is strongly positive.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--quality-holds", type=Path, default=Path("training/decision_source_quality_holds.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_bytes = args.cases.read_bytes()
    source_sha = sha(source_bytes)
    source_manifest_bytes = args.source_manifest.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    if source_manifest.get("selected_sha256", source_manifest.get("cases_sha256")) != source_sha:
        raise SystemExit("source manifest does not bind input cases")
    holds_bytes = args.quality_holds.read_bytes()
    holds = json.loads(holds_bytes)
    if holds.get("schema") != "natlang.decision-source-quality-holds/1":
        raise SystemExit("unsupported source-quality hold policy")
    held_ids = {entry["item_id"] for entry in holds.get("excluded_items", [])}
    held_groups = {entry["original_source_group"] for entry in holds.get("excluded_items", [])}

    rows = [json.loads(line) for line in source_bytes.decode().splitlines() if line.strip()]
    roles_by_group = {}
    for row in rows:
        roles_by_group.setdefault(row["group"], set()).add(row["role"])
    variants, bindings = [], []
    for row in rows:
        if row.get("role") != "train" or row["id"] in held_ids or row["group"] in held_groups:
            continue
        family = row.get("family")
        if family not in ("decision:toxicity", "decision:yelp-stars", "decision:app-stars"):
            continue
        if roles_by_group[row["group"]] != {"train"}:
            continue
        original_hash = sha(canonical(row))
        variant = dict(row)
        variant_id = "dv-" + sha(f"visible-task-v1:{family}:{row['id']}".encode())[:20]
        variant["id"] = variant_id
        if family == "decision:toxicity":
            if row.get("kind") != "noul" or not isinstance(row.get("answer"), (int, float)):
                raise SystemExit(f"unexpected toxicity source schema: {row['id']}")
            variant["family"] = "decision:toxicity-binary-threshold-v1"
            variant["question"] = TOXICITY_QUESTION
            variant["answer"] = row["answer"] >= 0.5
            variant["criteria"] = {
                "false": "The visible comment is not more likely than not to be toxic under the stated criteria.",
                "true": "The visible comment is more likely than not to be toxic under the stated criteria.",
            }
            adaptation = "binary-source-threshold-0.5"
        else:
            if row.get("kind") != "score" or not isinstance(row.get("answer"), (int, float)):
                raise SystemExit(f"unexpected star-rating source schema: {row['id']}")
            variant["family"] = "decision:review-satisfaction-estimate-v1"
            variant["question"] = SATISFACTION_QUESTION
            variant["criteria"] = {label: SATISFACTION_CRITERIA[label] for label in row["levels"]}
            adaptation = "visible-review-satisfaction-ordinal-tolerance-1"
        variants.append(variant)
        bindings.append({"variant_id": variant_id, "source_id": row["id"], "source_group": row["group"],
            "source_role": row["role"], "source_record_sha256": original_hash,
            "source_annotation": row["answer"], "adaptation": adaptation})

    if not variants:
        raise SystemExit("no eligible source cases")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"output directory is non-empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    case_bytes = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                             for row in variants) + "\n").encode()
    (args.out / "cases.jsonl").write_bytes(case_bytes)
    binding_bytes = (json.dumps({"schema": "natlang.visible-decision-task-bindings/1", "bindings": bindings},
                                ensure_ascii=False, indent=2) + "\n").encode()
    (args.out / "source-bindings.json").write_bytes(binding_bytes)
    new_manifest = {"schema": "natlang.decision-case-source-manifest/1", "source": str(args.cases),
        "source_sha256": source_sha, "selected_sha256": sha(case_bytes), "cases_sha256": sha(case_bytes),
        "count": len(variants), "families": {}, "training_admission": False, "new_world_credit": 0,
        "selection": "train-only source-bound task revisions; canonical item/group holds excluded",
        "adaptations": ["visible-text toxicity judgment with binary label boundary; source threshold retained only in source annotations", "visible review satisfaction estimate with one-star annotation tolerance retained only in oracle metadata"],
        "oracle_tolerances": {"decision:review-satisfaction-estimate-v1": {"absolute_ordinal_distance_max": 1, "model_visible": False}},
        "source_manifest_sha256": sha(source_manifest_bytes), "quality_hold_policy_sha256": sha(holds_bytes),
        "source_bindings_sha256": sha(binding_bytes), "live_prompt_sampling": "required before provider launch or task adoption"}
    for row in variants:
        new_manifest["families"][row["family"]] = new_manifest["families"].get(row["family"], 0) + 1
    (args.out / "source-manifest.json").write_text(json.dumps(new_manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(variants), "families": new_manifest["families"],
        "cases_sha256": sha(case_bytes), "training_admission": False,
        "live_prompt_sampling_required": True}, indent=2))


if __name__ == "__main__":
    main()
