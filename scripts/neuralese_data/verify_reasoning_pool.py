#!/usr/bin/env python3
"""Independently solve and verify the generated reasoning-gym and SynLogic task pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def grid_shape(value: object) -> bool:
    return (isinstance(value, list) and bool(value) and
            all(isinstance(row, list) and row and all(type(cell) is int for cell in row) for row in value) and
            len({len(row) for row in value}) == 1)


def rewrite_problem(question: str) -> tuple[dict[tuple[str, str], tuple[str, ...]], list[str]]:
    token_re = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z][&@#$+]|[&@#$+][A-Za-z])(?![A-Za-z0-9])")
    token_names = list(dict.fromkeys(re.findall(r"`([^`\s]+)`", question)))
    tokens = {token for token in token_names if re.fullmatch(r"(?:[A-Za-z][&@#$+]|[&@#$+][A-Za-z])", token)}
    if len(tokens) != 4:
        raise ValueError("rewrite task must define exactly four tokens")
    rules: dict[tuple[str, str], tuple[str, ...]] = {}
    for line in question.splitlines():
        line = line.strip().lstrip("-* ")
        matches = token_re.findall(line)
        if len(matches) < 2:
            continue
        left, right = matches[:2]
        if left not in tokens or right not in tokens or left[-1] != right[0]:
            continue
        if re.search(r"\b(?:is removed|removed|nothing|empty)\b", line, re.I):
            result: tuple[str, ...] = ()
        elif re.search(r"\b(?:is replaced by|becomes)\b", line, re.I):
            rest = re.split(r"\b(?:is replaced by|becomes)\b", line, maxsplit=1, flags=re.I)[1]
            result_tokens = [token for token in token_re.findall(rest) if token in tokens]
            if len(result_tokens) != 2:
                continue
            result = tuple(result_tokens)
        else:
            continue
        key = (left, right)
        if key in rules and rules[key] != result:
            raise ValueError(f"conflicting rules for {key}")
        rules[key] = result
    if len(rules) != 4:
        raise ValueError(f"expected four directed rewrite rules, found {len(rules)}")
    marker = list(re.finditer(r"(?:following program:|program is:|program:)([^\n]*)", question, re.I))
    if not marker:
        raise ValueError("task program is missing")
    rest = marker[-1].group(1)
    if not rest.strip():
        rest = next((line for line in question[marker[-1].end():].splitlines() if line.strip()), "")
    program = rest.strip().replace("`", "").rstrip(". ").split()
    if not program or any(token not in tokens for token in program):
        raise ValueError("task program contains unsupported tokens")
    return rules, program


def solve_rewrite(question: str) -> str:
    rules, state = rewrite_problem(question)
    seen: set[tuple[str, ...]] = set()
    while True:
        fingerprint = tuple(state)
        if fingerprint in seen:
            raise ValueError("rewrite system entered a repeated state before reaching a terminal sequence")
        seen.add(fingerprint)
        for index in range(len(state) - 1):
            pair = (state[index], state[index + 1])
            if pair in rules:
                state[index:index + 2] = rules[pair]
                break
        else:
            return " ".join(state) if state else "empty"


def extract_rewrite_gold(target: str) -> str:
    labels = list(re.finditer(r"(?:final state|final sequence|final program)(?:\s+is)?\s*:?\s*", target, re.I))
    for label in reversed(labels):
        text = target[label.end():].strip().removeprefix("**").strip()
        quoted = re.match(r"\s*`([^`]+)`", text)
        result = quoted.group(1).strip() if quoted else text.splitlines()[0].strip().rstrip(".")
        if result.lower() == "empty":
            return result
        parts = result.split()
        if parts and all(re.fullmatch(r"(?:[A-Za-z][&@#$+]|[&@#$+][A-Za-z])", part) for part in parts):
            return " ".join(parts)
    raise ValueError("source target has no parseable final sequence")


def parse_grids(question: str, label: str) -> list[list[list[int]]]:
    found = []
    pattern = re.compile(rf"^\s*{label}\s*:", re.I | re.M)
    for match in pattern.finditer(question):
        start = match.end()
        while start < len(question) and question[start].isspace():
            start += 1
        if start >= len(question) or question[start] != "[":
            continue
        depth = 0
        quoted = escaped = False
        for index in range(start, len(question)):
            char = question[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    try:
                        grid = json.loads(question[start:index + 1])
                    except json.JSONDecodeError:
                        break
                    if grid_shape(grid):
                        found.append(grid)
                    break
    return found


def d4(grid: list[list[int]]) -> list[tuple[str, list[list[int]]]]:
    options = []
    rotated = grid
    for turns in range(1, 5):
        rotated = [list(row) for row in zip(*rotated[::-1])]
        options.append((f"rotate-{turns}", rotated))
        options.append((f"rotate-{turns}-mirror-horizontal", [row[::-1] for row in rotated]))
    return options


def solve_grid(question: str) -> tuple[str, list[str]]:
    inputs = parse_grids(question, "Input")
    outputs = parse_grids(question, "Output")
    if len(inputs) != len(outputs) + 1 or len(outputs) < 2:
        raise ValueError("unsupported demonstration/test layout")
    predictions = []
    for name, _ in d4(inputs[0]):
        transform = lambda grid: dict(d4(grid))[name]
        color_map: dict[int, int] = {}
        valid = True
        for source, target in zip(inputs[:-1], outputs):
            source = transform(source)
            if len(source) != len(target) or any(len(a) != len(b) for a, b in zip(source, target)):
                valid = False
                break
            for row_a, row_b in zip(source, target):
                for before, after in zip(row_a, row_b):
                    if before in color_map and color_map[before] != after:
                        valid = False
                        break
                    color_map[before] = after
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            continue
        test = transform(inputs[-1])
        if any(cell not in color_map for row in test for cell in row):
            continue
        predictions.append((name, [[color_map[cell] for cell in row] for row in test]))
    distinct = {json.dumps(grid, separators=(",", ":")) for _, grid in predictions}
    if len(distinct) != 1:
        raise ValueError(f"not uniquely determined by D4+global-color-map checker ({len(distinct)} outputs)")
    return next(iter(distinct)), [name for name, _ in predictions]


def source_rows(corpus: Path, family: str, ids: set[str]) -> dict[str, dict]:
    rows = {}
    with (corpus / f"{family}.port-records.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in ids:
                rows[row["id"]] = row
    missing = ids - rows.keys()
    if missing:
        raise ValueError(f"source rows missing: {sorted(missing)}")
    return rows


def verify(args: argparse.Namespace) -> dict:
    corpus = Path(args.corpus)
    source_path = Path(args.source)
    case_lines = [line for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [json.loads(line) for line in case_lines]
    all_ids = {source["id"] for case in cases for source in case["dataset_records"]}
    rows_by_family = {}
    for family in ("reasoning_gym", "reasoning_synlogic"):
        ids = {source["id"] for case in cases if case["dataset"] == family for source in case["dataset_records"]}
        if ids:
            rows_by_family[family] = source_rows(corpus, family, ids)
    verified = []
    for case in cases:
        family = case["dataset"]
        records = rows_by_family[family]
        files = case["semantics"]["folder_files"]
        expected = case["semantics"]["expected"]
        expected_files = case["semantics"]["expected_files"]
        expected_items = [path for path in files if path.startswith("items/") and path.endswith(".json")]
        if len(expected_items) != len(expected) or case["split"] != "train":
            raise ValueError(f"{case['id']}: item count/split mismatch")
        original_groups = sorted({group for entry in case["dataset_records"] for group in entry["group"]})
        if case["source_groups"] != original_groups or case["generation"].get("original_source_groups") != original_groups:
            raise ValueError(f"{case['id']}: top-level source groups do not preserve the original source-group union")
        if case["generation"].get("source_bundle_group") != case["curriculum"]["split_group"]:
            raise ValueError(f"{case['id']}: task-variant bundle group is not recorded separately")
        computed = {}
        for path in expected_items:
            item = json.loads(files[path])
            source = records[item["source_record_id"]]
            prompt = source["consumer"]["context"][0]["content"]
            if source["split"] != "train" or source["license"]["spdx"] not in ("Apache-2.0", "MIT"):
                raise ValueError(f"{source['id']}: split/license changed")
            if item["question"] != "Solve the task described in the complete evidence." or item["evidence"] != prompt:
                raise ValueError(f"{source['id']}: source evidence or task instruction differs from the original consumer question")
            source_groups = source["split_groups"]
            source_meta = next(entry for entry in case["dataset_records"] if entry["id"] == source["id"])
            if source_meta["group"] != source_groups or source_meta["split"] != source["split"]:
                raise ValueError(f"{source['id']}: original group/split provenance changed")
            if family == "reasoning_gym":
                predicted = solve_rewrite(prompt)
                gold = extract_rewrite_gold(source["target"]["value"])
                if predicted != gold:
                    raise ValueError(f"{source['id']}: independent rewrite solver disagrees with source label")
            else:
                predicted, _ = solve_grid(prompt)
                answer_tag = re.search(r"<answer>\s*(\[.*?\])\s*</answer>", source["target"]["value"], re.S)
                if not answer_tag or json.loads(predicted) != json.loads(answer_tag.group(1)):
                    raise ValueError(f"{source['id']}: independent grid solver disagrees with source label")
            key = Path(path).stem
            if expected.get(key) != predicted:
                raise ValueError(f"{source['id']}: emitted expected answer differs from independent solver")
            computed[key] = predicted
        saved_expected = json.loads(expected_files["answers.json"])
        if saved_expected != computed or expected != computed:
            raise ValueError(f"{case['id']}: answer map/readback differs from independent solver")
        visible_text = "\n".join(files.values()) + case["semantics"]["files"]["collect_answers.nl"]
        for source in records.values():
            if source["id"] in {entry["id"] for entry in case["dataset_records"]}:
                # The raw task question is visible, but the annotation and target are not copied into task inputs.
                item_text = "\n".join(body for path, body in files.items() if path.startswith("items/"))
                if source["target"]["value"] in item_text:
                    raise ValueError(f"{source['id']}: source target leaked into an item input")
        verified.append({"case_id": case["id"], "family": family, "answer_mode": case["generation"]["answer_payload_mode"],
            "source_records": len(computed), "independent_answers": computed})
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    return {"schema": "verified-reasoning-pool-independent-check/1", "source_sha256": digest(source_path.read_bytes()),
        "case_count": len(cases), "source_record_count": len(all_ids), "cases": verified,
        "families": ["reasoning_gym", "reasoning_synlogic"], "original_corpus_manifest_sha256": digest((corpus / "manifest.json").read_bytes()),
        "solver": {"reasoning_gym": "parse visible directed rewrite rules, apply leftmost rule until stable, compare terminal sequence to original labeled final state",
            "reasoning_synlogic": "within the declared D4-orientation plus demonstration-derived global-color-function hypothesis class, require known test colors and one predicted grid, then compare to the original labeled grid; this does not establish global ARC-rule uniqueness"},
        "model_calls": 0, "provider_calls": 0, "training_admission": False, "generation_admission": "pending root review"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = verify(args)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="")
    print(json.dumps({"out": str(output), "source_sha256": report["source_sha256"],
        "cases": report["case_count"], "source_records": report["source_record_count"], "provider_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
