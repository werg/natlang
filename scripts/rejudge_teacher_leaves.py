#!/usr/bin/env python3
"""Recheck rejected teacher values after a check revision; leave admission to review."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from natlang.checks import make_judge, run_checks
from natlang.decoder import LlamaServerDecoder
from natlang.gen import codebases as C
from scripts.teacher_leaves import CHECKS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8081")
    a = ap.parse_args()
    if a.out.exists():
        ap.error(f"refusing overwrite: {a.out}")
    judge = make_judge(LlamaServerDecoder(a.server, timeout=120,
                       chat_extra={"chat_template_kwargs": {"enable_thinking": False}}))
    checked = 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.audit.open() as source, a.out.open("w") as output:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["accepted"] or row["key"] in C.REFERENCES or not row.get("value"):
                continue
            checked += 1
            results = run_checks(CHECKS[row["function"]](row["args"]), row["value"], judge)
            passed = all(result is True for _, result in results)
            audit = {"source": str(a.audit), "line": line_number, "key": row["key"],
                     "function": row["function"], "value": row["value"],
                     "original_checks": row["checks"], "revised_checks": results,
                     "judge_passed": passed}
            output.write(json.dumps(audit, ensure_ascii=False) + "\n")
            output.flush()
            print(f"{row['key']}: {'judge pass' if passed else 'judge fail'}", flush=True)
    print(f"checked {checked}; audit {a.out}")


if __name__ == "__main__":
    main()
