#!/usr/bin/env python3
"""Prepare sales-conversation prompts or SQL-injection pairs for natlang.

Sales input is the original Hugging Face CSV (read as a stream). SQL input is
Modified_SQL_Dataset.csv from the specified Kaggle dataset. This script does
not download data or contact a labeling service.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SALES_LICENSE = "Apache-2.0"
SQL_LICENSE = "Unknown"
SALES_REVISION = "714f4544cdbc3f192e7f8ea93053815c8e5479cf"
OBJECTION = {
    "price": "The customer is concerned about price, budget, or return on investment.",
    "integration": "The customer is concerned about connecting the product to existing systems or processes.",
    "training": "The customer is concerned about learning, onboarding, or team adoption.",
    "security": "The customer is concerned about security, privacy, or compliance.",
    "capability": "The customer asks whether the product has a needed feature or performance.",
    "timing": "The customer is concerned about timing, implementation schedule, or readiness.",
    "none": "The customer states no objection in the latest turn.",
}
INTENT = {
    "ask_question": "The customer is asking for information or clarification.",
    "evaluate": "The customer is weighing options or expressing tentative interest.",
    "request_demo": "The customer explicitly requests a demonstration or trial.",
    "commit": "The customer explicitly agrees to purchase or proceed.",
    "decline": "The customer explicitly declines or ends the sales discussion.",
    "unclear": "The latest customer turn does not establish another listed intent.",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _split(source, group):
    n = int(hashlib.sha256(f"{source}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if n < 80 else "dev" if n < 90 else "test"


def sales_tasks(path, max_conversations=None, max_tasks=None, max_prefixes=3,
                source_revision=SALES_REVISION):
    emitted = 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), 1):
            if max_conversations is not None and row_number > max_conversations:
                break
            conversation_id = row.get("conversation_id") or f"row-{row_number}"
            company = row.get("company_id") or row.get("company_name") or conversation_id
            group = f"sales/company/{company}"
            split = _split("sales", group)
            try:
                messages = json.loads(row.get("conversation") or "[]")
            except json.JSONDecodeError:
                continue
            if not isinstance(messages, list):
                continue
            customer_indices = [i for i, m in enumerate(messages)
                                if isinstance(m, dict) and m.get("speaker") == "customer"
                                and isinstance(m.get("message"), str) and m["message"].strip()]
            # Spread selected prefixes across the conversation rather than
            # always choosing the opening turns.
            if max_prefixes > 0 and len(customer_indices) > max_prefixes:
                positions = [(j * (len(customer_indices) - 1)) // (max_prefixes - 1)
                             for j in range(max_prefixes)] if max_prefixes > 1 else [len(customer_indices) - 1]
                customer_indices = [customer_indices[j] for j in sorted(set(positions))]
            for turn_index in customer_indices:
                prefix = messages[max(0, turn_index - 9):turn_index + 1]
                state = "\n".join(f"{m.get('speaker', 'unknown')}: {m.get('message', '')}"
                                  for m in prefix if isinstance(m, dict))[-12000:]
                if not state.strip():
                    continue
                for name, instruction, criteria in (
                    ("objection", "What is the customer's main objection in the latest customer turn?", OBJECTION),
                    ("intent", "What is the customer's intent in the latest customer turn?", INTENT),
                ):
                    yield {"id": f"sales/{conversation_id}/{turn_index}/{name}", "source": "saas-sales-conversations",
                           "source_revision": source_revision,
                           "group_id": group, "split": split, "state": state,
                           "instruction": instruction, "kind": "choice",
                           "labels": list(criteria), "criteria": criteria,
                           "license": SALES_LICENSE,
                           "source_meta": {"conversation_id": conversation_id, "turn_index": turn_index,
                                           "company_id": company, "product_type": row.get("product_type")}}
                    emitted += 1
                    if max_tasks is not None and emitted >= max_tasks:
                        return


def _sql_group(query):
    # Group variants, but retain the original query unchanged as model input.
    text = query.lower()
    text = re.sub(r"/\*.*?\*/|--[^\n]*|#[^\n]*", " ", text, flags=re.S)
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "<string>", text)
    text = re.sub(r"\b(?:0x[0-9a-f]+|\d+)\b", "<number>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def sql_tasks(path, max_tasks=None, source_revision="kaggle-version-1"):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    labels_by_query = defaultdict(set)
    for row in rows:
        query, label = row.get("Query", ""), row.get("Label", "")
        labels_by_query[query].add(label)
    seen = set()
    candidates = []
    for row_number, row in enumerate(rows, 1):
        query, label = row.get("Query", ""), row.get("Label", "")
        if not query.strip() or label not in {"0", "1"} or len(labels_by_query[query]) != 1 or query in seen:
            continue
        seen.add(query)
        candidates.append((row_number, query, label))
    # The CSV is label-ordered. A capped pilot must exercise both answers.
    order = lambda item: hashlib.sha256(item[1].encode()).hexdigest()
    if max_tasks is not None:
        negatives = sorted((item for item in candidates if item[2] == "0"), key=order)
        positives = sorted((item for item in candidates if item[2] == "1"), key=order)
        chosen = []
        for index in range(max_tasks):
            pool = negatives if index % 2 == 0 else positives
            position = index // 2
            if position < len(pool):
                chosen.append(pool[position])
        candidates = chosen
    else:
        candidates.sort(key=order)
    for row_number, query, label in candidates:
        group = f"sql/template/{_sql_group(query)}"
        yield {"id": f"sql-injection/v1/{row_number}", "source": "sajid576-sql-injection",
               "source_revision": source_revision, "group_id": group,
               "split": _split("sql", group), "state": query,
               "instruction": "Does this input attempt to alter SQL query structure or execution through injection?",
               "kind": "boolean", "labels": ["false", "true"], "gold": "true" if label == "1" else "false",
               "gold_source": "dataset", "license": SQL_LICENSE,
               "source_meta": {"original_row": row_number, "raw_label": label}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["sales", "sql"])
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument("--max-prefixes", type=int, default=3)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--source-revision", help="source dataset revision recorded with every task")
    args = parser.parse_args()
    if args.max_tasks is not None and args.max_tasks < 1 or args.max_prefixes < 1:
        parser.error("max-tasks and max-prefixes must be positive")
    if args.source == "sql" and args.max_conversations is not None:
        parser.error("max-conversations applies only to sales")
    tasks = (sales_tasks(args.input, args.max_conversations, args.max_tasks, args.max_prefixes,
                         args.source_revision or SALES_REVISION)
             if args.source == "sales" else sql_tasks(args.input, args.max_tasks,
                                                       args.source_revision or "kaggle-version-1"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with args.out.open("w", encoding="utf-8") as stream:
        for task in tasks:
            stream.write(_json(task) + "\n")
            counts[task["split"]] += 1
    digest = hashlib.sha256()
    with args.out.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    input_sha256 = None
    if args.source == "sql":
        source_digest = hashlib.sha256()
        with args.input.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                source_digest.update(block)
        input_sha256 = source_digest.hexdigest()
    manifest = {"source": args.source, "source_revision": args.source_revision or
                (SALES_REVISION if args.source == "sales" else "kaggle-version-1"),
                "input": str(args.input), "input_bytes": args.input.stat().st_size,
                "input_sha256": input_sha256,
                "output_sha256": digest.hexdigest(), "counts": dict(counts),
                "max_conversations": args.max_conversations, "max_prefixes": args.max_prefixes,
                "max_tasks": args.max_tasks}
    args.out.with_suffix(args.out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(_json(dict(counts)), file=sys.stderr)


if __name__ == "__main__":
    main()
