"""Model-independent checks shared by plain text and port generation gates."""
from __future__ import annotations

import json
import re


def output_budget(probe: dict, *, ordinary: int = 256, thinking: int = 512) -> int:
    """A declared probe budget, shared by gate generation and reference capture."""
    value = probe.get("max_new_tokens", thinking if probe.get("enable_thinking") else ordinary)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("probe max_new_tokens must be a positive integer")
    return value


def _argument_type(value, schema):
    """Check the argument shapes declared by a tool, without coercing its values."""
    kind = schema.get("type")
    if isinstance(kind, list):
        return any(_argument_type(value, {**schema, "type": item}) for item in kind)
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "string": isinstance(value, str), "boolean": isinstance(value, bool),
               "integer": isinstance(value, int) and not isinstance(value, bool),
               "number": isinstance(value, (int, float)) and not isinstance(value, bool),
               "null": value is None}
    if kind in matches and not matches[kind]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if isinstance(value, dict):
        if any(key not in value for key in schema.get("required", [])):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and value.keys() - properties.keys():
            return False
        return all(_argument_type(value[key], spec) for key, spec in properties.items() if key in value)
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return all(_argument_type(item, schema["items"]) for item in value)
    return True


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
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in probe["tools"]}
        expected_names = probe.get("expected_tools")
        if expected_names is not None and (not isinstance(expected_names, list) or not expected_names or
                                           any(name not in names for name in expected_names)):
            raise ValueError("expected_tools must name offered tools")
        def acceptable(name, arguments):
            expected = probe.get("expected_tool")
            return (name in names and isinstance(arguments, dict) and
                    all(key in arguments for key in names[name]) and
                    (expected is None or name == expected) and
                    (expected_names is None or name in expected_names) and
                    (not probe.get("check_argument_types") or _argument_type(arguments, schemas[name])))
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
                    if acceptable(name, arguments):
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
                if acceptable(name, arguments):
                    ok = True
        if probe.get("kind") in ("tool", "natlang"):
            checks["tool"] = ok
        if probe.get("require_tool_call"):
            checks["native_tool" if probe.get("signal") == "native_diagnostic" else "tool"] = ok
    elif probe.get("require_tool_call"):
        raise ValueError("require_tool_call needs offered tools")
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
    if any("signal" in row for row in results):
        if any(row.get("signal") not in {"ordinary_competence", "native_diagnostic"} for row in results):
            raise ValueError("every probe must declare a signal when separating competence and native diagnostics")
        ordinary = [row for row in results if row["signal"] == "ordinary_competence"]
        native = [row for row in results if row["signal"] == "native_diagnostic"]
        if not ordinary:
            raise ValueError("a competence gate needs ordinary competence probes")
        report["gate_pass_all"] = report["gate_pass"]
        report["gate_pass"] = sum(row["pass"] for row in ordinary) / len(ordinary)
        report["gate_signal"] = "ordinary_competence"
        report["ordinary_probe_count"] = len(ordinary)
        report["native_diagnostic"] = {"probe_count": len(native), "pass_rate":
                                       sum(row["pass"] for row in native) / len(native) if native else None}
    return report
