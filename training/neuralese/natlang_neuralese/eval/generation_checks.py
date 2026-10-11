"""Model-independent checks shared by plain text and port generation gates."""
from __future__ import annotations

import json
import re


def check(probe: dict, text: str, tokens: list[int], eos: int | list[int]) -> dict:
    """Apply structural checks; legacy `answer` means only digit presence, while
    `expected_exact_text` is an explicit exact final-answer check.
    """
    end_ids = {eos} if isinstance(eos, int) else set(eos)
    checks = {"end": bool(tokens) and tokens[-1] in end_ids, "no_loop": not _loops(tokens)}
    convention = probe.get("output_convention", "maple")
    if convention != "lfm":
        if probe.get("enable_thinking"):
            open_at, close_at = text.find("<think>"), text.find("</think>")
            checks["think"] = open_at >= 0 and close_at > open_at
        else:
            checks["think"] = "<think>" not in text
    final = text.split("</think>")[-1]
    if probe.get("tools"):
        names = {t["function"]["name"]: t["function"]["parameters"].get("required", []) for t in probe["tools"]}
        ok = False
        if convention == "lfm":
            from ..serve.chat import parse_pythonic_calls
            calls = re.findall(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", final, flags=re.S)
            for body in calls:
                try:
                    parsed_calls = parse_pythonic_calls(body)
                except (ValueError, SyntaxError):
                    continue
                for name, arguments in parsed_calls:
                    expected = probe.get("expected_tool")
                    if name in names and isinstance(arguments, dict) and all(k in arguments for k in names[name]) and \
                            (expected is None or name == expected):
                        ok = True
        else:
            calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", final, flags=re.S)
            for call in calls:
                try:
                    parsed = json.loads(call)
                except json.JSONDecodeError:
                    continue
                name, arguments = parsed.get("name"), parsed.get("arguments")
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = None
                expected = probe.get("expected_tool")
                if name in names and isinstance(arguments, dict) and all(k in arguments for k in names[name]) and \
                        (expected is None or name == expected):
                    ok = True
        if probe.get("kind") in ("tool", "natlang"):
            checks["tool"] = ok
    if probe.get("answer") is not None:
        checks["exact"] = probe["answer"] in re.findall(r"-?\d+", final.replace(",", ""))
    if probe.get("expected_exact_text") is not None:
        exact_final = re.sub(r"(?:<\|im_end\|>|</s>|<\|end_of_text\|>)\s*$", "", final).strip()
        checks["exact_text"] = exact_final == str(probe["expected_exact_text"])
    return checks


def _loops(tokens: list[int], n: int = 8, repeats: int = 4) -> bool:
    counts: dict[tuple[int, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i + n])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= repeats:
            return True
    return False


def summarize(results: list[dict]) -> dict:
    """Summarize completed checks with the same per-check/per-kind denominator policy."""
    if not results:
        raise ValueError("generation gate requires at least one completed probe")
    names = sorted({key for row in results for key in row["checks"]})
    report = {"gate_pass": sum(row["pass"] for row in results) / len(results),
              "gate_checks": {key: round(sum(row["checks"][key] for row in results if key in row["checks"]) /
                                          max(1, sum(key in row["checks"] for row in results)), 3)
                              for key in names},
              "gate_kinds": {}}
    for kind in sorted({row.get("kind") for row in results}):
        rows = [row for row in results if row.get("kind") == kind]
        report["gate_kinds"][kind] = round(sum(row["pass"] for row in rows) / len(rows), 3)
    return report
