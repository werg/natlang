#!/usr/bin/env python3
"""Prepare an immutable combined corpus and receipt for append-aware resume."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.corpus import digest, index_pairs, split_programs
from scripts.training_append import ids_digest
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def verify_audited_file(data_path: Path, audit_path: Path, label: str) -> dict:
    audit = _read_json(audit_path, f"{label} audit manifest")
    expected = audit.get("sha256")
    actual = sha256_file(data_path)
    if expected != actual:
        raise ValueError(f"{label} ready data hash differs from its audit manifest")
    if audit.get("audit", {}).get("ready") is not True:
        raise ValueError(f"{label} ready data is not token audited")
    return audit


def directory_file_hashes(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def write_immutable_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_bytes(payload)


def prepare_append_intake(*, base_path: Path, base_audit_path: Path,
                          base_split_path: Path, checkpoint_path: Path,
                          candidate_path: Path, candidate_audit_path: Path,
                          gate_receipt_path: Path, combined_path: Path,
                          output_path: Path) -> dict:
    """Check dataset/checkpoint identity and ensure candidate rows remain train-only."""
    base_path, candidate_path = Path(base_path), Path(candidate_path)
    base_audit = verify_audited_file(base_path, Path(base_audit_path), "base")
    candidate_audit = verify_audited_file(candidate_path, Path(candidate_audit_path), "candidate")
    if base_audit.get("renderer") != candidate_audit.get("renderer"):
        raise ValueError("candidate renderer/tokenizer differs from the base ready corpus")
    if base_audit.get("audit", {}).get("max_len") != candidate_audit.get("audit", {}).get("max_len"):
        raise ValueError("candidate max_len differs from the base ready corpus")

    checkpoint = _read_json(Path(checkpoint_path), "checkpoint state")
    corpus = checkpoint.get("corpus")
    if not isinstance(corpus, dict):
        raise ValueError("checkpoint has no corpus identity")
    base_sha = sha256_file(base_path)
    if corpus.get("data_sha256") != base_sha:
        raise ValueError("checkpoint is not bound to the supplied base corpus")

    split_saved = _read_json(Path(base_split_path), "saved program split")
    seed = split_saved.get("seed")
    holdout_target = split_saved.get("holdout_target")
    if not isinstance(seed, int) or not isinstance(holdout_target, int):
        raise ValueError("saved split is missing integer seed/holdout_target")
    base_pairs = index_pairs(base_path)
    if base_audit.get("rows") != len(base_pairs):
        raise ValueError("base audit row count differs from the ready data")
    base_held, base_train, recomputed_split = split_programs(base_pairs, holdout_target, seed)
    base_split_sha = digest(recomputed_split)
    if base_split_sha != corpus.get("split_sha256"):
        raise ValueError("recomputed base split differs from the checkpoint split hash")
    if recomputed_split != split_saved:
        raise ValueError("saved split.json is not the split derived from the pinned base rows")
    if len(base_train) != int(corpus.get("target_examples", -1)):
        raise ValueError("checkpoint target_examples does not match the base training split")

    cursor = checkpoint.get("cursor")
    if not isinstance(cursor, int) or cursor < 0 or cursor >= len(base_train):
        raise ValueError("append is only supported before the old training order is exhausted")
    if checkpoint.get("step", 0) >= int(corpus.get("steps", 0)):
        raise ValueError("append is not allowed after the scheduled run has completed")

    candidate_pairs = index_pairs(candidate_path)
    if not candidate_pairs:
        raise ValueError("candidate ready corpus has no rows")
    if candidate_audit.get("rows") != len(candidate_pairs):
        raise ValueError("candidate audit row count differs from the ready data")
    base_ids = {row["id"] for row in base_pairs}
    candidate_ids = [row["id"] for row in candidate_pairs]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate contains duplicate row IDs")
    if base_ids.intersection(candidate_ids):
        raise ValueError("candidate row IDs overlap the base corpus")
    if any(row.get("split") != "train" for row in candidate_pairs):
        raise ValueError("every candidate row must have an explicit train split")

    # Recompute split assignment over the combined identity graph. This catches
    # direct and transitive source_group links into protected test components.
    combined_held, combined_train, combined_split = split_programs(
        base_pairs + candidate_pairs, holdout_target, seed)
    if {row["id"] for row in combined_held} != {row["id"] for row in base_held}:
        raise ValueError("append would change the protected held-out row set")
    combined_train_ids = {row["id"] for row in combined_train}
    unsafe = [row["id"] for row in candidate_pairs if row["id"] not in combined_train_ids]
    if unsafe:
        raise ValueError(f"candidate rows join held-out/reserved components: {unsafe[:20]}")

    gate = _read_json(Path(gate_receipt_path), "current native gate receipt")
    if gate.get("schema") != "lfm-training-append-gates/1":
        raise ValueError("unsupported append gate receipt schema")
    required_true = ("current_admission_passed", "source_conversion_passed",
                     "source_review_passed", "native_materialization_passed")
    if any(gate.get(key) is not True for key in required_true):
        raise ValueError("append gate receipt has missing/failed current-policy gates")
    policy_sha = gate.get("current_policy_sha256")
    if not isinstance(policy_sha, str) or len(policy_sha) != 64 or any(ch not in "0123456789abcdef" for ch in policy_sha):
        raise ValueError("append gate receipt lacks a valid current-policy SHA-256")
    if gate.get("candidate_sha256") != sha256_file(candidate_path):
        raise ValueError("gate receipt is bound to different candidate bytes")
    if gate.get("candidate_rows") != len(candidate_pairs):
        raise ValueError("gate receipt row count does not match candidate data")
    ids_sha = digest(candidate_ids)
    if gate.get("candidate_row_ids_sha256") != ids_sha:
        raise ValueError("gate receipt is bound to a different ordered candidate ID list")

    # Write an immutable combined input; the pinned base and candidate remain
    # untouched. Preserve their JSONL bytes and source order.
    combined_path = Path(combined_path)
    if combined_path.exists():
        raise FileExistsError(f"refusing to overwrite combined corpus {combined_path}")
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("xb") as out:
        for source_path in (base_path, candidate_path):
            with source_path.open("rb") as source:
                last = b""
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    out.write(block)
                    last = block[-1:]
                if last and last != b"\n":
                    out.write(b"\n")
    combined_pairs = index_pairs(combined_path)
    if len(combined_pairs) != len(base_pairs) + len(candidate_pairs):
        raise ValueError("combined corpus row count differs from its inputs")
    expected_physical_ids = [row["id"] for row in base_pairs + candidate_pairs]
    if [row["id"] for row in combined_pairs] != expected_physical_ids:
        raise ValueError("combined corpus physical row order differs from base plus candidate inputs")

    # Both inputs have passed the same tokenizer audit and are copied byte for
    # byte. Record this exact composition as a derived audit, then regenerate
    # the reducer mix report for the combined corpus.
    audit_derivation_path = combined_path.with_name(combined_path.name + ".audit-derivation.json")
    combined_audit_path = combined_path.with_name(combined_path.name + ".manifest.json")
    mix_path = combined_path.with_name(combined_path.name + ".mix.json")
    derivation = {
        "schema": "lfm-audited-jsonl-concatenation/1",
        "method": "byte-preserving concatenation of two ready corpora with exact renderer/max_len identity",
        "base_sha256": base_sha, "base_audit_sha256": sha256_file(Path(base_audit_path)),
        "candidate_sha256": sha256_file(candidate_path),
        "candidate_audit_sha256": sha256_file(Path(candidate_audit_path)),
        "combined_sha256": sha256_file(combined_path),
        "combined_rows": len(combined_pairs),
        "row_order": "base physical rows followed by candidate physical rows",
        "tokenization_recomputed": False,
        "justification": "every input row was already token-audited under the identical renderer, tokenizer fingerprint, and max_len; this operation does not change row bytes",
    }
    write_immutable_json(audit_derivation_path, derivation)
    renderer = base_audit["renderer"]
    combined_audit = {
        "version": base_audit.get("version", "natlang.sft.native/1"),
        "source": [base_audit.get("source"), candidate_audit.get("source")],
        "source_sha256": [base_audit.get("source_sha256"), candidate_audit.get("source_sha256")],
        "rows": len(combined_pairs), "renderer": renderer,
        "identity_sha256": digest({"base": base_sha, "candidate": sha256_file(candidate_path),
                                    "renderer": renderer, "max_len": base_audit["audit"]["max_len"]}),
        "sha256": sha256_file(combined_path), "rejections": {},
        "rejections_sha256": hashlib.sha256(b"").hexdigest(),
        "audit": {"max_len": base_audit["audit"]["max_len"], "ready": True,
                  "report_sha256": sha256_file(audit_derivation_path),
                  "rejections_sha256": hashlib.sha256(b"").hexdigest(),
                  "derived_from_audited_inputs": True,
                  "derivation_sha256": sha256_file(audit_derivation_path)},
    }
    write_immutable_json(combined_audit_path, combined_audit)
    from scripts.audit_training_mix import build_report
    mix_report = build_report([combined_path], 0.25)
    write_immutable_json(mix_path, mix_report)

    # Hash the base train permutation actually used by split_programs. This is
    # needed to preserve cursor semantics in a future append-aware trainer.
    ordered_ids = [row["id"] for row in base_train]
    prefix_ids = ordered_ids[:cursor]
    result = {
        "schema": "natlang.training_append_intake/2",
        "status": "prepared_for_append_aware_resume",
        "base": {
            "path": str(base_path.resolve()), "sha256": base_sha,
            "audit_path": str(Path(base_audit_path).resolve()),
            "audit_sha256": sha256_file(Path(base_audit_path)),
            "split_path": str(Path(base_split_path).resolve()),
            "split_sha256": base_split_sha,
            "train_rows": len(base_train), "heldout_rows": len(base_held),
            "train_order_ids_sha256": digest(ordered_ids),
            "consumed_prefix_ids_sha256": digest(prefix_ids),
        },
        "checkpoint": {
            "path": str(Path(checkpoint_path).resolve()),
            "sha256": sha256_file(Path(checkpoint_path)),
            "files": directory_file_hashes(Path(checkpoint_path).parent),
            "step": checkpoint.get("step"), "cursor": cursor,
            "trained_examples": checkpoint.get("trained_examples"),
            "optimizer": corpus.get("optimizer", "adamw"),
            "corpus_identity": corpus,
            "optimizer_state_required": True,
        },
        "candidate": {
            "path": str(candidate_path.resolve()), "sha256": sha256_file(candidate_path),
            "audit_path": str(Path(candidate_audit_path).resolve()),
            "audit_sha256": sha256_file(Path(candidate_audit_path)),
            "rows": len(candidate_pairs), "ordered_row_ids_sha256": ids_sha,
            "append_order": "candidate file physical row order, after reconstructed base train permutation",
        },
        "combined": {
            "path": str(combined_path.resolve()), "sha256": sha256_file(combined_path),
            "rows": len(combined_pairs),
            "audit_manifest": str(combined_audit_path.resolve()),
            "audit_manifest_sha256": sha256_file(combined_audit_path),
            "audit_derivation": str(audit_derivation_path.resolve()),
            "audit_derivation_sha256": sha256_file(audit_derivation_path),
            "mix_report": str(mix_path.resolve()), "mix_report_sha256": sha256_file(mix_path),
            "mix_target_met": mix_report.get("target_met"),
            "mix_reducer_share": mix_report.get("reducer_share"),
            "train_order": "base split_programs permutation, then candidate physical order",
        },
        "append_order": {
            "base_train_order_ids_sha256": ids_digest(ordered_ids),
            "candidate_order_ids_sha256": ids_digest(candidate_ids),
            "base_train_rows": len(ordered_ids),
            "candidate_rows": len(candidate_ids),
            "cursor_at_transition": cursor,
        },
        "current_gate_receipt": {
            "path": str(Path(gate_receipt_path).resolve()),
            "sha256": sha256_file(Path(gate_receipt_path)),
            "current_policy_sha256": gate.get("current_policy_sha256"),
        },
        "combined_split": {
            "train_rows": len(combined_train), "heldout_rows": len(combined_held),
            "split_sha256": digest(combined_split),
            "candidate_rows_in_train": len(candidate_ids) - len(unsafe),
            "candidate_rows_joining_heldout": len(unsafe),
        },
        "resume_requirements": {
            "preserve_base_train_order_as_prefix": True,
            "preserve_optimizer_state": True,
            "preserve_rng_state": True,
            "preserve_cursor": True,
            "scheduler_extension_must_be_explicit": True,
            "current_trainer_consumes_append_manifest": True,
            "do_not_edit_or_resume_active_checkpoint_with_changed_data_identity": True,
        },
        "split_implementation_sha256": sha256_file(REPO_ROOT / "scripts/corpus.py"),
        "output_path": str(Path(output_path).resolve()),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite append intake manifest {output_path}")
    output_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--combined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_append_intake(
            base_path=args.base, base_audit_path=args.base_audit,
            base_split_path=args.base_split, checkpoint_path=args.checkpoint,
            candidate_path=args.candidate, candidate_audit_path=args.candidate_audit,
            gate_receipt_path=args.gate_receipt, combined_path=args.combined,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest_path": str(args.output.resolve()),
                      "manifest_sha256": sha256_file(args.output), "manifest": result},
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
