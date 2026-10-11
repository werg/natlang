#!/usr/bin/env python3
"""Add source-pinned Civil Comments positives to the visible-toxicity task pool.

This reads only the pinned train parquet shards already present in the source
cache. The source annotation is stored in hidden case metadata and the
source-bindings receipt; the model-visible question is a comment judgment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: str) -> str:
    return sha(value.encode())


def normalize_state(text: str) -> str:
    return " ".join(str(text).split())


def identity_state(text: str, limit: int = 1500) -> str:
    text = normalize_state(text)
    return text if len(text) <= limit else text[:limit] + " ..."


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="visible-decision-task-variants-v1 directory")
    parser.add_argument("--source-manifest", type=Path, required=True, help="pinned source-bound manifest")
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--jeff-cases", type=Path, required=True)
    parser.add_argument("--quality-holds", type=Path, required=True)
    parser.add_argument("--prior-cases", type=Path, action="append", default=[],
        help="case JSONL files whose source ids/groups must not be repeated")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra-positive-count", type=int, default=44)
    args = parser.parse_args()

    base_cases_bytes = (args.base / "cases.jsonl").read_bytes()
    base_bindings_bytes = (args.base / "source-bindings.json").read_bytes()
    base_cases = [json.loads(line) for line in base_cases_bytes.decode().splitlines() if line.strip()]
    base_bindings_doc = json.loads(base_bindings_bytes)
    base_bindings = list(base_bindings_doc["bindings"])
    case_ids = {item["id"] for item in base_cases}
    source_ids = {binding["source_id"] for binding in base_bindings}
    source_groups = {binding["source_group"] for binding in base_bindings}
    seen_identity = set()

    source_manifest_bytes = args.source_manifest.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    dataset = next(item for item in source_manifest["dataset_revisions"] if item["repo"] == "google/civil_comments")
    download_manifest_bytes = args.download_manifest.read_bytes()
    download_manifest = json.loads(download_manifest_bytes)
    pins = {item["path"]: item for item in download_manifest["entries"] if item["repo"] == "google/civil_comments"}
    if dataset["revision"] != "fd8a8ed30121358918ca5557f52fd3073321f9ec" or dataset["source_split"] != "train":
        raise SystemExit("unexpected Civil Comments source revision or split")

    # Exact protected test texts are checked over the complete comment, using
    # the same reservation normalization used by the reviewed child selector.
    reservation = [json.loads(line) for line in args.jeff_cases.read_text().splitlines() if line.strip()]
    needles = []
    for row in reservation:
        text = row.get("state")
        if isinstance(text, str):
            normalized = " ".join(text.replace("\\", " ").split())
            if len(normalized) >= 40:
                needles.append((row.get("id"), normalized[:60]))

    holds_doc = json.loads(args.quality_holds.read_text())
    if holds_doc.get("schema") != "natlang.decision-source-quality-holds/1":
        raise SystemExit("unsupported source-quality policy")
    held_ids = {item["item_id"] for item in holds_doc.get("excluded_items", [])}
    held_groups = {item["original_source_group"] for item in holds_doc.get("excluded_items", [])}

    for prior_path in args.prior_cases:
        for line in prior_path.read_text().splitlines():
            if not line.strip():
                continue
            prior = json.loads(line)
            prior_id = prior.get("id")
            prior_group = prior.get("group")
            if isinstance(prior_id, str):
                source_ids.add(prior_id)
            if isinstance(prior_group, str):
                source_groups.add(prior_group)
            if prior.get("family") == "decision:toxicity" and isinstance(prior.get("state"), str):
                seen_identity.add(identity_state(prior["state"]))

    bins = {"boundary": [], "moderate": [], "clear": []}
    for base in base_cases:
        if base["family"] == "decision:toxicity-binary-threshold-v1":
            ident = identity_state(base["state"])
            seen_identity.add(ident)

    for shard in sorted(dataset["shards"], key=lambda item: item["upstream_path"]):
        local = Path(shard["local_path"])
        raw_path = args.raw_dir / local.name
        raw_bytes_hash = hashlib.sha256()
        with raw_path.open("rb") as source:
            for block in iter(lambda: source.read(1 << 20), b""):
                raw_bytes_hash.update(block)
        expected_pin = pins.get(shard["upstream_path"])
        if not expected_pin or raw_bytes_hash.hexdigest() != shard["raw_sha256"] or shard["raw_sha256"] != expected_pin["sha256"]:
            raise SystemExit(f"source shard hash/pin mismatch: {raw_path}")

        parquet = pq.ParquetFile(raw_path)
        physical_start = 0
        for group_index in range(parquet.metadata.num_row_groups):
            row_group = parquet.read_row_group(group_index, columns=["text", "toxicity"]).to_pylist()
            for row_index, row in enumerate(row_group):
                text, score = row.get("text"), row.get("toxicity")
                if not isinstance(text, str) or not text.strip() or not isinstance(score, (float, int)) or score < 0.5:
                    continue
                ident = identity_state(text)
                if not ident or ident in seen_identity:
                    continue
                case_id = "dc-" + digest(f"toxicity:{ident}")[:20]
                group_id = "g-" + digest(f"decision:{ident}")[:24]
                if case_id in source_ids or group_id in source_groups or case_id in held_ids or group_id in held_groups:
                    continue
                normalized_comment = " ".join(text.replace("\\", " ").split())
                if any(normalized_comment.find(needle) >= 0 for _, needle in needles):
                    continue

                record = {"version": "natlang.decision-case/1", "id": case_id,
                    "source": "google/civil_comments", "family": "decision:toxicity", "role": "train",
                    "kind": "noul", "question": "Would a typical annotator consider this comment toxic?",
                    "state": text, "group": group_id, "license": dataset["license"],
                    "identity_state_sha256": digest(ident), "full_state_sha256": digest(text),
                    "answer": float(score), "source_refs": [{"dataset": "google/civil_comments", "split": "train",
                        "local_path": shard["local_path"], "local_file_sha256": shard["raw_sha256"],
                        "row_group": group_index, "row_in_group": row_index,
                        "physical_row": physical_start + row_index,
                        "upstream": {"revision": dataset["revision"], "path": shard["upstream_path"],
                            "lfs_sha256": shard["hub_lfs_sha256"]},
                        "upstream_metadata_status": "pinned_and_byte_bound"}]}
                record_sha = sha(canonical(record))
                variant_id = "dv-" + sha(f"visible-task-v1:decision:toxicity:{case_id}".encode())[:20]
                variant = {"version": "natlang.decision-case/1", "id": variant_id,
                    "source": "google/civil_comments", "family": "decision:toxicity-binary-threshold-v1",
                    "role": "train", "kind": "noul",
                    "question": ("Based on the visible comment, is it more likely than not toxic? Treat insults, threats, hateful or demeaning language, and attacks on someone for their identity as toxic. Mere disagreement or criticism is not enough. Return true for toxic and false otherwise."),
                    "state": text, "group": group_id, "license": dataset["license"],
                    "criteria": {"false": "The visible comment is not more likely than not to be toxic under the stated criteria.",
                        "true": "The visible comment is more likely than not to be toxic under the stated criteria."},
                    "answer": score >= 0.5}
                binding = {"variant_id": variant_id, "source_id": case_id, "source_group": group_id,
                    "source_role": "train", "source_record_sha256": record_sha,
                    "source_annotation": float(score), "adaptation": "binary-visible-toxicity-judgment-source-threshold-0.5",
                    "source_revision": dataset["revision"], "source_shard": shard["upstream_path"],
                    "source_row_group": group_index, "source_row_in_group": row_index,
                    "source_physical_row": physical_start + row_index}
                bucket = "boundary" if score < 0.6 else "moderate" if score <= 0.8 else "clear"
                rank = sha(f"toxicity-positive-pool-v1:{case_id}".encode())
                bins[bucket].append((rank, variant, binding, record))
                seen_identity.add(ident)
            physical_start += parquet.metadata.row_group(group_index).num_rows

    quotas = {"boundary": args.extra_positive_count * 15 // 44,
        "moderate": args.extra_positive_count * 15 // 44,
        "clear": args.extra_positive_count - 2 * (args.extra_positive_count * 15 // 44)}
    extras, extra_bindings, extra_sources = [], [], []
    for bucket in ("boundary", "moderate", "clear"):
        selected = sorted(bins[bucket], key=lambda item: item[0])[:quotas[bucket]]
        if len(selected) < quotas[bucket]:
            raise SystemExit(f"insufficient source rows in toxicity stratum {bucket}: {len(selected)} < {quotas[bucket]}")
        for _, variant, binding, record in selected:
            extras.append(variant); extra_bindings.append(binding); extra_sources.append(record)
    if len({row["id"] for row in extras}) != len(extras):
        raise SystemExit("new source case ids are not unique")

    all_cases = base_cases + extras
    all_bindings = base_bindings + extra_bindings
    if len({row["id"] for row in all_cases}) != len(all_cases):
        raise SystemExit("combined variant IDs are not unique")
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is non-empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    case_bytes = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in all_cases) + "\n").encode()
    binding_bytes = (json.dumps({"schema": "natlang.visible-decision-task-bindings/1", "bindings": all_bindings},
        ensure_ascii=False, indent=2) + "\n").encode()
    source_bytes = ("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in extra_sources) + "\n").encode()
    (out / "cases.jsonl").write_bytes(case_bytes)
    (out / "source-bindings.json").write_bytes(binding_bytes)
    (out / "additional-source-cases.jsonl").write_bytes(source_bytes)
    summary = {"schema": "natlang.visible-toxicity-positive-pool/1", "training_admission": False,
        "count": len(extras), "answer_counts": {"true": len(extras)},
        "strata": {name: len(values) for name, values in bins.items()}, "stratum_quotas": quotas,
        "cases_sha256": sha(case_bytes), "source_bindings_sha256": sha(binding_bytes),
        "additional_source_cases_sha256": sha(source_bytes),
        "base_cases_sha256": sha(base_cases_bytes), "base_bindings_sha256": sha(base_bindings_bytes),
        "source_manifest_sha256": sha(source_manifest_bytes),
        "download_manifest_sha256": sha(download_manifest_bytes),
        "quality_hold_policy_sha256": sha(args.quality_holds.read_bytes()),
        "jeff_test_cases_sha256": sha(args.jeff_cases.read_bytes()),
        "dataset_revision": dataset["revision"], "source_split": "train",
        "source_license": dataset["license"], "independent_new_worlds": 0,
        "live_prompt_sampling_required": True}
    (out / "source-manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"additional_positive_cases": len(extras), "strata": summary["strata"],
        "quotas": quotas, "total_cases": len(all_cases), "cases_sha256": summary["cases_sha256"],
        "bindings_sha256": summary["source_bindings_sha256"], "source_cases_sha256": summary["additional_source_cases_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
