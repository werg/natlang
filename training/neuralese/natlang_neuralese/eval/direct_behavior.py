"""Paired ordinary-HF behavior comparison for a direct serving artifact.

This is a thin runner over ``plain_behavior``'s renderer, model-independent
checks, output budgets, and raw generation loop. It reports measurements only:
it does not qualify a runtime or change training/data admission.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from . import plain_behavior as common


def evaluate(args):
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite behavior output: {args.out}")
    if not args.base_snapshot.is_dir():
        raise FileNotFoundError("an explicit local immutable HF base snapshot is required")
    if sha256_file_hex(args.probes) != args.probes_sha256:
        raise ValueError("probe file changed from its declared SHA-256")
    probes = [json.loads(line) for line in args.probes.read_text().splitlines() if line.strip()]
    if len(probes) != 48 or len({row.get("id") for row in probes}) != 48:
        raise ValueError("paired direct behavior evaluation requires the pinned 48-row probe set")
    signals = {row.get("signal") for row in probes}
    if signals != {"ordinary_competence", "native_diagnostic"}:
        raise ValueError("the probe file must separate ordinary competence and native diagnostics")
    artifact_dir = args.direct_artifact.resolve(strict=True)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256_file_hex(manifest_path)
    if (manifest.get("schema") != "natlang.neuralese-direct-serving/1" or
            manifest.get("status") != "direct-writer-pending" or
            manifest.get("qualification", {}).get("runtime") is not False):
        raise ValueError("candidate is not an explicitly pending direct serving artifact")
    if sha256_file_hex(artifact_dir / manifest["payload"]["path"]) != manifest["payload"]["sha256"]:
        raise ValueError("direct artifact tensor payload hash mismatch")
    if args.base_snapshot.resolve() != Path(manifest["base_binding"]["snapshot_path_at_export"]).resolve():
        raise ValueError("evaluation base snapshot differs from the artifact source binding")
    if common._execution_components() != args.execution_components:
        raise RuntimeError("evaluation source bytes changed after launch inputs were frozen")

    args.out.mkdir(parents=True, exist_ok=False)
    journal = args.out / "execution-journal.jsonl"
    status_path = args.out / "run-status.json"
    status = {"schema": "natlang.direct-plain-behavior-status/1", "status": "running",
              "selected_probe_count": len(probes), "completed_roles": [], "completed_role_probes": 0}
    common._write_status(status_path, status)
    invocation = {
        "schema": "natlang.direct-plain-behavior-invocation/1",
        "probe_path": str(args.probes), "probe_sha256": args.probes_sha256,
        "probe_ids": [row["id"] for row in probes],
        "direct_artifact": str(artifact_dir), "direct_manifest_sha256": manifest_sha,
        "direct_payload_sha256": manifest["payload"]["sha256"],
        "training_checkpoint": manifest["training_checkpoint"],
        "base_snapshot": str(args.base_snapshot.resolve()),
        "base_revision": manifest["base_binding"]["revision"],
        "runtime_image": args.runtime_image, "device": args.device,
        "execution_components": args.execution_components,
        "interpretation": "paired descriptive ordinary HF generation only; no qualification or admission",
    }
    invocation["invocation_sha256"] = canonical_json_sha256_hex(invocation)
    (args.out / "invocation.json").write_text(json.dumps(invocation, indent=2, ensure_ascii=False) + "\n")
    progress = {"completed": 0, "completed_roles": [], "current_role_completed": 0}
    role_results = {}
    roles = (("unmodified_pinned_base", None), ("selected_direct_candidate", artifact_dir))
    try:
        common._append_jsonl(journal, {"event": "run_started", "invocation_sha256": invocation["invocation_sha256"]})
        for role, checkpoint in roles:
            common._append_jsonl(journal, {"event": "role_started", "role": role,
                                           "artifact_manifest_sha256": manifest_sha if checkpoint else None})
            progress["current_role_completed"] = 0
            rows = common._run_role(role, checkpoint, probes, device=args.device,
                                    base=str(args.base_snapshot.resolve()), max_new_tokens=1024,
                                    journal=journal, status_path=status_path, progress=progress)
            role_results[role] = rows
            result_path = args.out / f"{role}.jsonl"
            with result_path.open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            progress["completed_roles"].append(role)
            status.update(completed_roles=list(progress["completed_roles"]))
            common._write_status(status_path, status)
            common._append_jsonl(journal, {"event": "role_completed", "role": role,
                                           "probe_count": len(rows),
                                           "results_sha256": sha256_file_hex(result_path)})
    except BaseException as error:
        common._append_jsonl(journal, {"event": "run_error", "error_type": type(error).__name__,
                                       "error": str(error)[:1000]})
        common._write_status(status_path, {**status, "status": "failed",
                                           "error_type": type(error).__name__, "error": str(error)[:1000]})
        raise
    if common._execution_components() != args.execution_components:
        common._append_jsonl(journal, {"event": "run_error", "error_type": "ExecutionCodeDrift"})
        common._write_status(status_path, {**status, "status": "failed", "error_type": "ExecutionCodeDrift"})
        raise RuntimeError("evaluation package changed during paired behavior capture")

    paired = []
    for index, probe in enumerate(probes):
        pair = {name: role_results[name][index] for name, _ in roles}
        if (len({row["prompt_sha256"] for row in pair.values()}) != 1 or
                len({tuple(row["stop_token_ids"]) for row in pair.values()}) != 1):
            raise ValueError(f"paired prompt rendering differs for {probe['id']}")
        paired.append({"id": probe["id"], "kind": probe.get("kind"), "signal": probe["signal"],
                       "prompt_sha256": pair["unmodified_pinned_base"]["prompt_sha256"], "roles": pair})
    paired_path = args.out / "paired-results.jsonl"
    with paired_path.open("x", encoding="utf-8") as handle:
        for row in paired:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    report = {
        "schema": "natlang.direct-plain-behavior-comparison/1",
        "interpretation": "descriptive paired ordinary-HF generation; direct writer remains pending; no qualification or admission",
        "invocation_sha256": invocation["invocation_sha256"],
        "inputs": {"probes_sha256": args.probes_sha256, "direct_manifest_sha256": manifest_sha,
                   "direct_payload_sha256": manifest["payload"]["sha256"],
                   "base_snapshot": str(args.base_snapshot.resolve()),
                   "base_revision": manifest["base_binding"]["revision"]},
        "signal_counts": {signal: sum(row["signal"] == signal for row in probes)
                          for signal in sorted(signals)},
        "budgets": {"plain_exact_math": 512, "tool": 1024, "open_answer": 1024,
                    "native_diagnostic": 2048, "thinking_override": 4096},
        "roles": {name: common._role_summary(rows) for name, rows in role_results.items()},
        "completion_outputs": {name: {"path": f"{name}.jsonl", "sha256": sha256_file_hex(args.out / f"{name}.jsonl")}
                               for name, _ in roles},
        "paired_results": {"path": paired_path.name, "sha256": sha256_file_hex(paired_path),
                           "all_same_prompt_hash": True},
        "execution_components": args.execution_components,
    }
    report_path = args.out / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    common._write_status(status_path, {**status, "status": "complete", "completed_role_probes": sum(map(len, role_results.values())),
                                       "summary_sha256": sha256_file_hex(report_path)})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--probes-sha256", required=True)
    parser.add_argument("--direct-artifact", type=Path, required=True)
    parser.add_argument("--base-snapshot", type=Path, required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.execution_components = common._execution_components()
    result = evaluate(args)
    print(json.dumps({"out": str(args.out), "signals": result["signal_counts"],
                      "roles": {key: value["probe_count"] for key, value in result["roles"].items()}}))


if __name__ == "__main__":
    main()
