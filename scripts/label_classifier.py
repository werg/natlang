#!/usr/bin/env python3
"""Label normalized task JSONL with classifier.dev's fast Jev endpoint.

Example:
  python scripts/label_classifier.py tasks.jsonl labels.jsonl --max-items 20 --max-requests 4

The output keeps the source record and adds a `jev` audit object. Only an
approved, scored answer from a Jev model fills missing gold. Existing gold is
never replaced. A dry run prints request bodies without contacting the API.
Re-running skips IDs already written with the same task hash. Error rows may be
retried with --retry-errors; the latest row for an ID then supersedes earlier
error rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

ENDPOINT = "https://classifier.dev/v1/classify"
MAX_BATCH = 1000


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_hash(row: dict[str, Any]) -> str:
    source = {k: v for k, v in row.items() if k != "jev"}
    return hashlib.sha256(canonical_json(source).encode("utf-8")).hexdigest()


def input_text(row: dict[str, Any]) -> str:
    state = row.get("state", row.get("args", row.get("input")))
    if isinstance(state, str):
        result = state
    elif state is not None:
        result = canonical_json(state)
    else:
        raise ValueError("missing state, args, or input")
    if not result.strip():
        raise ValueError("empty input text")
    if len(result) > 32000:
        raise ValueError("input exceeds API limit of 32000 characters")
    return result


def instructions(row: dict[str, Any]) -> str:
    instruction = row.get("instruction", row.get("question"))
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("missing nonempty instruction")
    criteria = row.get("criteria")
    if criteria is None:
        return instruction.strip()
    if not isinstance(criteria, dict) or set(criteria) != set(row["labels"]):
        raise ValueError("criteria must describe exactly the canonical labels")
    lines = [instruction.strip(), "Choose one label using these definitions:"]
    for label in row["labels"]:
        desc = criteria[label]
        if not isinstance(desc, str) or not desc.strip():
            raise ValueError("criteria descriptions must be nonempty strings")
        lines.append(f"{label}: {desc.strip()}")
    return "\n".join(lines)


def validate(row: Any, line: int) -> tuple[str, tuple[str, ...], str, str]:
    if not isinstance(row, dict):
        raise ValueError(f"line {line}: expected JSON object")
    rid = row.get("id")
    if not isinstance(rid, str) or not rid:
        raise ValueError(f"line {line}: id must be a nonempty string")
    labels = row.get("labels")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 100:
        raise ValueError(f"line {line}: labels must contain 2 to 100 values")
    if any(not isinstance(x, str) or not x.strip() or len(x) > 200 for x in labels):
        raise ValueError(f"line {line}: labels must be nonempty strings of at most 200 characters")
    if len(set(labels)) != len(labels):
        raise ValueError(f"line {line}: duplicate labels")
    kind = row.get("kind", "choice")
    if kind not in ("choice", "boolean", "score"):
        raise ValueError(f"line {line}: invalid kind {kind!r}")
    if kind == "boolean" and labels != ["false", "true"]:
        raise ValueError(f"line {line}: boolean labels must be ['false', 'true']")
    if row.get("gold") is not None and row["gold"] not in labels:
        raise ValueError(f"line {line}: gold must be an allowed label")
    try:
        return rid, tuple(labels), instructions(row), input_text(row)
    except ValueError as exc:
        raise ValueError(f"line {line}: {exc}") from exc


def read_tasks(path: Path, existing: dict[str, dict[str, Any]], retry_errors: bool, max_items: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            rid, labels, rubric, text = validate(row, lineno)
            if rid in seen:
                raise ValueError(f"line {lineno}: duplicate id {rid!r}")
            seen.add(rid)
            digest = task_hash(row)
            old = existing.get(rid)
            if old is not None:
                if old["jev"].get("input_hash") != digest:
                    raise ValueError(f"line {lineno}: task {rid!r} changed since output was written")
                if not retry_errors or old["jev"].get("status") != "error":
                    continue
            tasks.append({"row": row, "id": rid, "labels": labels, "instructions": rubric,
                          "text": text, "hash": digest})
            if len(tasks) >= max_items:
                break
    return tasks


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("jev"), dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"output line {lineno}: invalid previous result")
            existing[row["id"]] = row
    return existing


def groups(tasks: list[dict[str, Any]], batch_size: int):
    grouped: OrderedDict[tuple[tuple[str, ...], str], list[dict[str, Any]]] = OrderedDict()
    for task in tasks:
        grouped.setdefault((task["labels"], task["instructions"]), []).append(task)
    for (labels, rubric), members in grouped.items():
        for offset in range(0, len(members), batch_size):
            chunk = members[offset:offset + batch_size]
            yield chunk, {"inputs": [task["text"] for task in chunk], "labels": list(labels),
                          "instructions": rubric, "tier": "fast"}


def retry_delay(headers: dict[str, str], attempt: int, ceiling: float) -> float | None:
    value = headers.get("retry-after")
    if value:
        try:
            parsed = float(value)
        except ValueError:
            try:
                parsed = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                parsed = 0
        if math.isfinite(parsed) and parsed > 0:
            # A day-limit 429 must not be retried early just because our
            # allowed wait is shorter than the server's Retry-After.
            return parsed if parsed <= ceiling else None
    return min(2 ** attempt + random.uniform(0, 0.25), ceiling)


def post(payload: dict[str, Any], timeout: float, retries: int, max_wait: float, url: str) -> tuple[int, dict[str, str], Any, list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "application/json",
                                              "User-Agent": "natlang-label-classifier/1"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                headers = {k.lower(): v for k, v in response.headers.items()}
                raw = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            status = exc.code
            headers = {k.lower(): v for k, v in exc.headers.items()}
            raw = exc.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            status, headers, raw = 0, {}, str(exc)
        try:
            response_body: Any = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            response_body = raw
        entry = {"attempt": attempt + 1, "http_status": status, "response": response_body,
                 "headers": {k: v for k, v in headers.items() if k in
                             ("x-api-version", "retry-after", "ratelimit-limit",
                              "ratelimit-remaining", "ratelimit-policy")}}
        history.append(entry)
        if status == 200:
            return status, headers, response_body, history
        if status not in (0, 429, 502) or attempt == retries:
            return status, headers, response_body, history
        delay = retry_delay(headers, attempt, max_wait)
        if delay is None:
            return status, headers, response_body, history
        entry["sleep_seconds"] = delay
        time.sleep(delay)
    raise AssertionError("unreachable")


def make_result(task: dict[str, Any], request: dict[str, Any], response: Any,
                index: int, status: int, headers: dict[str, str], history: list[dict[str, Any]],
                valid_response: bool = True, batch_id: str | None = None) -> dict[str, Any]:
    row = dict(task["row"])
    result = None
    if status == 200 and valid_response and isinstance(response, dict):
        results = response.get("results")
        if isinstance(results, list) and index < len(results):
            result = results[index]
    model = result.get("model") if isinstance(result, dict) else None
    # A batch-level model is sufficient only when the server says one model
    # answered every item. A mixed result without item attribution is rejected.
    if model is None and isinstance(response, dict) and response.get("model") != "mixed":
        model = response.get("model")
    label = result.get("label") if isinstance(result, dict) else None
    scores = result.get("scores") if isinstance(result, dict) else None
    valid_scores = (isinstance(scores, dict) and set(scores) == set(task["labels"])
                    and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) and 0 <= value <= 1 for value in scores.values()))
    accepted = (status == 200 and isinstance(result, dict) and isinstance(model, str)
                and model.lower().startswith("jev") and label in task["labels"]
                and valid_scores and not result.get("escalated") and not result.get("unscored"))
    if status != 200:
        reason = "http_error"
    elif result is None:
        reason = "missing_result"
    elif not isinstance(model, str) or not model.lower().startswith("jev"):
        reason = "non_jev_model"
    elif label not in task["labels"]:
        reason = "invalid_label"
    elif not valid_scores:
        reason = "invalid_scores"
    elif result.get("escalated") or result.get("unscored"):
        reason = "unscored_or_escalated"
    else:
        reason = None
    row["jev"] = {"status": "accepted" if accepted else ("error" if status != 200 else "rejected"),
                  "reason": reason, "input_hash": task["hash"],
                  "text_sha256": hashlib.sha256(task["text"].encode("utf-8")).hexdigest(),
                  "request": request, "response": response, "response_index": index,
                  "result": result, "model": model, "label": label,
                  "scores": scores,
                  "http_status": status, "api_version": headers.get("x-api-version"),
                  "response_headers": {k: v for k, v in headers.items() if k in
                                       ("x-api-version", "ratelimit-limit", "ratelimit-remaining",
                                        "ratelimit-policy", "retry-after")},
                  "retry_history": history}
    if batch_id is not None:
        # Large batches are logged once in a sidecar. Repeating the entire
        # request and response for every item makes output quadratic in batch
        # size (particularly when inputs are long conversations).
        row["jev"].pop("request")
        row["jev"].pop("response")
        row["jev"].pop("retry_history")
        row["jev"]["batch_id"] = batch_id
    if accepted and row.get("gold") is None:
        row["gold"] = label
        row["gold_source"] = "jev"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="normalized task JSONL")
    parser.add_argument("output", type=Path, help="resumable labeled JSONL")
    parser.add_argument("--batch-size", type=int, default=20, help="items per same-rubric request, at most 1000")
    parser.add_argument("--max-items", type=int, default=100, help="maximum new classifications this run")
    parser.add_argument("--max-requests", type=int, default=10, help="maximum request batches this run")
    parser.add_argument("--retries", type=int, default=3, help="retries for 429/502/network failures")
    parser.add_argument("--max-wait", type=float, default=60, help="maximum retry sleep in seconds")
    parser.add_argument("--timeout", type=float, default=30, help="HTTP timeout in seconds")
    parser.add_argument("--retry-errors", action="store_true", help="retry previous HTTP error rows")
    parser.add_argument("--dry-run", action="store_true", help="print request bodies; do not write or call API")
    parser.add_argument("--compact-audit", action="store_true",
                        help="write each complete request/response once to OUTPUT.batches.jsonl")
    parser.add_argument("--url", default=ENDPOINT, help="API URL, useful for local contract tests")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= MAX_BATCH or args.max_items < 1 or args.max_requests < 1:
        parser.error("batch-size must be 1..1000 and both caps must be positive")
    if args.retries < 0 or args.max_wait <= 0 or args.timeout <= 0:
        parser.error("retries must be nonnegative; waits and timeout must be positive")
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output paths must differ")
    previous = read_existing(args.output)
    tasks = read_tasks(args.input, previous, args.retry_errors, args.max_items)
    batches = list(groups(tasks, args.batch_size))
    if args.dry_run:
        for chunk, payload in batches[:args.max_requests]:
            print(canonical_json({"ids": [task["id"] for task in chunk], "request": payload}))
        visible = batches[:args.max_requests]
        print(f"dry run: {sum(len(chunk) for chunk, _ in visible)} items, {len(visible)} requests", file=sys.stderr)
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    accepted = 0
    remaining_calls = args.max_requests
    batch_log_path = args.output.with_suffix(args.output.suffix + ".batches.jsonl")
    with args.output.open("a", encoding="utf-8") as out:
        for chunk, payload in batches:
            if remaining_calls == 0:
                break
            status, headers, response, history = post(payload, args.timeout,
                                                      min(args.retries, remaining_calls - 1),
                                                      args.max_wait, args.url)
            remaining_calls -= len(history)
            # Malformed successful responses are evidence, but never become gold.
            valid_response = (isinstance(response, dict) and isinstance(response.get("results"), list)
                              and len(response["results"]) == len(chunk))
            if status == 200 and not valid_response:
                print("warning: malformed or incomplete API response; no labels accepted", file=sys.stderr)
            batch_id = None
            if args.compact_audit:
                batch_id = hashlib.sha256(canonical_json({"ids": [t["id"] for t in chunk],
                                                          "request": payload}).encode()).hexdigest()
                with batch_log_path.open("a", encoding="utf-8") as batch_log:
                    batch_log.write(json.dumps({"batch_id": batch_id,
                                                "ids": [t["id"] for t in chunk],
                                                "request": payload, "response": response,
                                                "http_status": status, "headers": headers,
                                                "retry_history": history}, ensure_ascii=False) + "\n")
                    batch_log.flush()
            for index, task in enumerate(chunk):
                row = make_result(task, payload, response, index, status, headers, history,
                                  valid_response, batch_id)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += row["jev"]["status"] == "accepted"
                count += 1
            out.flush()
            print(f"request {len(chunk)} items: HTTP {status}; {accepted}/{count} Jev answers accepted", file=sys.stderr)
            if status != 200:
                print("stopping after HTTP error; rerun with --retry-errors after the issue clears", file=sys.stderr)
                return 1
    print(f"wrote {count} rows; accepted {accepted}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
