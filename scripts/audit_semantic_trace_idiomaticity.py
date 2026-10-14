#!/usr/bin/env python3
"""Count review signals in a teacher trace; never assign a quality verdict."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


TYPED_ARRAY = re.compile(r"\b(?:const|let)\s+\w+\s*:\s*[\w.<>| ]+\[\]\s*=\s*\[")
ARRAY_LITERAL = re.compile(r"\b(?:const|let)\s+\w+\s*=\s*\[")
NL_CALL = re.compile(r"\bnl(?:\.with)?\s*(?:<|`|\()")


def scan(path: Path, large_code_chars: int) -> dict:
    trace = json.loads(path.read_text())
    groups: dict[str, dict] = defaultdict(lambda: {
        "eval_calls": 0, "eval_code_chars": 0, "nl_call_sites": 0,
        "large_eval_turns": [], "array_literal_turns": [],
        "typed_array_turns": [], "return_result_calls": 0,
    })
    for turn_number, turn in enumerate(trace.get("turns", [])):
        invocation = turn.get("invocation_id", "unknown")
        row = groups[invocation]
        for call in turn.get("response", {}).get("calls", []):
            if not isinstance(call, list) or len(call) != 2 or not isinstance(call[1], dict):
                continue
            tool, args = call
            if tool == "return_result":
                row["return_result_calls"] += 1
            if tool != "eval":
                continue
            code = args.get("code", "")
            if not isinstance(code, str):
                continue
            row["eval_calls"] += 1
            row["eval_code_chars"] += len(code)
            row["nl_call_sites"] += len(NL_CALL.findall(code))
            if len(code) >= large_code_chars:
                row["large_eval_turns"].append(turn_number)
            if ARRAY_LITERAL.search(code):
                row["array_literal_turns"].append(turn_number)
            if TYPED_ARRAY.search(code):
                row["typed_array_turns"].append(turn_number)
    return {
        "schema": "natlang.semantic-trace-idiomaticity-signals/1",
        "source_trace": str(path.resolve()),
        "turn_count": len(trace.get("turns", [])),
        "large_code_chars": large_code_chars,
        "invocations": dict(sorted(groups.items())),
        "decision": "none; manual semantic and downstream-consumption review required",
        "training_admission": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Collector job .partial.json or completed job JSON")
    parser.add_argument("--large-code-chars", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(scan(args.trace, args.large_code_chars), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
