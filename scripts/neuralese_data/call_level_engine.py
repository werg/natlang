#!/usr/bin/env python3
"""Oracle-Free Call-Level & Judge-Site Extraction Engine (`plans/NATLANG_COMPILER.md` §8.1, W3.1).

Extracts single NL call sites and combinator judge sites (`holds`, `beats`, `settled`,
`relates`, `meets`, `decide`) from collected episode journals and authored program runs,
enforces `held_eval.exclusion_ids()` closure isolation, deduplicates by `(site_id, header_sha,
input_sha)`, checks multi-sample teacher agreement, and emits unified JSONL rows feeding:
  1. Compiler T2 Context Distillation + Judge Compilation
  2. Upgraded `expert_iteration` `sft_sources` call-level tier (teacher reasoning zero-weighted)
  3. Initial Experience KB records
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

JUDGE_KINDS = frozenset({"holds", "beats", "settled", "relates", "meets", "decide", "decision"})


def sha256_canonical(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class CallSiteRow:
    row_id: str
    site_id: str
    site_kind: str                     # "nl" | "judge" | "decision" | "literal"
    header_text: str
    header_sha: str
    input_payload: dict
    input_sha: str
    target_output: object
    target_text: str
    teacher_reasoning: str | None      # always zero-weighted context (owner 2026-11-24)
    agreement: float                   # fraction of agreeing teacher samples in [0, 1]
    type_valid: bool
    ensures_valid: bool
    source_task_id: str
    source_family: str
    combinator_role: str | None = None
    decision_labels: tuple[str, ...] = ()
    soft_label_probs: dict[str, float] | None = None


def classify_site_kind(call: dict) -> str:
    readout = str(call.get("readout") or "").lower()
    role = str(call.get("combinator_role") or call.get("role") or "").lower()
    name = str(call.get("name") or call.get("site_id") or "").lower()
    if readout == "decision" or role in {"decide", "decision"}:
        return "decision"
    if role in JUDGE_KINDS or any(name.endswith(f":{k}") or name == k for k in JUDGE_KINDS):
        return "judge"
    if call.get("synthesized_literal") or call.get("is_literal"):
        return "literal"
    return "nl"


def compute_sample_agreement(samples: list[object]) -> tuple[object, float]:
    """Return (majority_answer, agreement_fraction) over independent teacher samples."""
    if not samples:
        raise ValueError("samples must be nonempty")
    counts: dict[str, int] = {}
    exemplars: dict[str, object] = {}
    for sample in samples:
        key = sha256_canonical(sample)
        counts[key] = counts.get(key, 0) + 1
        exemplars.setdefault(key, sample)
    best_key = max(counts.keys(), key=lambda k: (counts[k], k))
    return exemplars[best_key], float(counts[best_key]) / float(len(samples))


def extract_call_sites(
    episodes: Iterable[dict],
    *,
    excluded_task_ids: set[str] | frozenset[str] = frozenset(),
    min_agreement: float = 0.66,
    require_type_valid: bool = True,
    require_ensures_valid: bool = True,
    max_per_site_header: int = 256,
) -> tuple[list[CallSiteRow], dict[str, int]]:
    """Extract verified, deduplicated, held-safe single call sites from episode dicts."""
    rows: list[CallSiteRow] = []
    seen_ids: set[str] = set()
    per_site_counts: dict[str, int] = {}
    stats = {
        "episodes_seen": 0,
        "held_excluded": 0,
        "calls_seen": 0,
        "dropped_invalid": 0,
        "dropped_low_agreement": 0,
        "dropped_duplicate": 0,
        "dropped_site_cap": 0,
        "admitted": 0,
        "admitted_judges": 0,
    }

    for ep in episodes:
        stats["episodes_seen"] += 1
        task_id = str(ep.get("task_id") or ep.get("id") or "")
        family = str(ep.get("family") or ep.get("source_family") or "unknown")
        if task_id and task_id in excluded_task_ids:
            stats["held_excluded"] += 1
            continue
        for call in ep.get("calls") or ():
            stats["calls_seen"] += 1
            type_ok = bool(call.get("type_valid", True))
            ensures_ok = bool(call.get("ensures_valid", True))
            if (require_type_valid and not type_ok) or (require_ensures_valid and not ensures_ok):
                stats["dropped_invalid"] += 1
                continue

            samples = call.get("teacher_samples")
            if isinstance(samples, list) and samples:
                target_val, agreement = compute_sample_agreement(samples)
            else:
                target_val = call.get("target_output")
                agreement = float(call.get("agreement", 1.0))
            if target_val is None or agreement < float(min_agreement):
                stats["dropped_low_agreement"] += 1
                continue

            site_id = str(call.get("site_id") or call.get("binding") or call.get("name") or "anonymous")
            header_text = str(call.get("header") or call.get("instructions") or "").strip()
            if not header_text:
                stats["dropped_invalid"] += 1
                continue
            input_payload = call.get("inputs") if isinstance(call.get("inputs"), dict) else {"value": call.get("inputs")}
            header_sha = sha256_canonical(header_text)
            input_sha = sha256_canonical(input_payload)
            row_id = sha256_canonical({"site_id": site_id, "header_sha": header_sha, "input_sha": input_sha})
            if row_id in seen_ids:
                stats["dropped_duplicate"] += 1
                continue
            site_bucket = f"{site_id}:{header_sha[:16]}"
            if per_site_counts.get(site_bucket, 0) >= max_per_site_header:
                stats["dropped_site_cap"] += 1
                continue

            seen_ids.add(row_id)
            per_site_counts[site_bucket] = per_site_counts.get(site_bucket, 0) + 1
            kind = classify_site_kind(call)
            target_text = (
                target_val if isinstance(target_val, str) else json.dumps(target_val, ensure_ascii=False, sort_keys=True)
            )
            labels = tuple(str(x) for x in (call.get("decision_labels") or ()))
            soft_probs = call.get("soft_label_probs") if isinstance(call.get("soft_label_probs"), dict) else None
            row = CallSiteRow(
                row_id=row_id,
                site_id=site_id,
                site_kind=kind,
                header_text=header_text,
                header_sha=header_sha,
                input_payload=input_payload,
                input_sha=input_sha,
                target_output=target_val,
                target_text=target_text,
                teacher_reasoning=(str(call["teacher_reasoning"]) if call.get("teacher_reasoning") else None),
                agreement=agreement,
                type_valid=type_ok,
                ensures_valid=ensures_ok,
                source_task_id=task_id,
                source_family=family,
                combinator_role=(str(call["combinator_role"]) if call.get("combinator_role") else None),
                decision_labels=labels,
                soft_label_probs=soft_probs,
            )
            rows.append(row)
            stats["admitted"] += 1
            if kind in {"judge", "decision"}:
                stats["admitted_judges"] += 1

    return rows, stats


def write_call_sites_jsonl(rows: Iterable[CallSiteRow], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            payload = asdict(row)
            payload["decision_labels"] = list(row.decision_labels)
            # Explicitly mark zero-weight reasoning policy for downstream sft_sources consumers.
            payload["teacher_reasoning_weight"] = 0.0
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(out_path)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract verified oracle-free NL call sites and judge sites.")
    parser.add_argument("--input", required=True, nargs="+", help="Episode JSONL path(s)")
    parser.add_argument("--out", required=True, help="Output call-sites JSONL path")
    parser.add_argument("--min-agreement", type=float, default=0.66)
    parser.add_argument("--max-per-site", type=int, default=256)
    parser.add_argument("--no-held-exclude", action="store_true", help="Skip held_eval.exclusion_ids() (tests only)")
    args = parser.parse_args(argv)

    excluded: set[str] = set()
    if not args.no_held_exclude:
        try:
            from natlang_neuralese.rl import held_eval

            excluded = set(held_eval.exclusion_ids())
        except Exception as exc:
            raise RuntimeError(f"Failed to load held_eval.exclusion_ids(): {exc}") from exc

    def _iter_episodes():
        for path_str in args.input:
            with Path(path_str).open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)

    rows, stats = extract_call_sites(
        _iter_episodes(),
        excluded_task_ids=excluded,
        min_agreement=args.min_agreement,
        max_per_site_header=args.max_per_site,
    )
    written = write_call_sites_jsonl(rows, Path(args.out))
    print(json.dumps({"written": written, **stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
