"""Admit reviewed browser cases into the shared program IR when their trace is exactly representable.

This adapter accepts natural-language leaf functions with successful root actions. It records
rejections for cases requiring richer graph/effect semantics instead of silently losing them.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from natlang.codebase import load_function
from program_ir import VERSION, validate


def _safe_path(path: str) -> PurePosixPath:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts or candidate.suffix not in {".nl", ".ts"}:
        raise ValueError(f"unsafe or unsupported source path: {path}")
    return candidate


def _root(case: dict, folder: Path) -> dict:
    source = case.get("source") or {}
    files = source.get("files") or {}
    root_path = _safe_path(source.get("root", ""))
    if root_path.suffix != ".nl" or root_path.as_posix() not in files:
        raise ValueError("only natural-language root functions can become lambda scenarios")
    for name, body in files.items():
        path = _safe_path(name)
        if not isinstance(body, str):
            raise ValueError("source file must contain text")
        target = folder.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    function = load_function(folder.joinpath(*root_path.parts))
    if function.kind != "instructions" or function.codebase:
        raise ValueError("linked child functions need a graph adapter")
    return {"$lambda": {**function.to_lambda_doc(), "args": case.get("inputs") or {}}}


def _operations(events: list[dict]) -> list[dict]:
    actions = [event for event in events if event.get("kind") == "action"]
    if not actions:
        raise ValueError("no recorded actions")
    operations = []
    for action in actions:
        if action.get("call_id") != "$root@1" or action.get("outcome") != "ok":
            raise ValueError("nested or rejected actions need a richer adapter")
        args = action.get("arguments") or {}
        name = action.get("name")
        if name == "write" and isinstance(args.get("path"), str) and isinstance(args.get("type"), str):
            op = {"op": "assign", "target": args["path"], "value_type": args["type"]}
            if "source" in args:
                op["from"] = args["source"]
            elif "value" in args:
                op["value"] = args["value"]
            else:
                raise ValueError("write action has neither source nor value")
            if "done" in args:
                op["completion"] = args["done"]
            operations.append(op)
        elif name == "report_error" and isinstance(args.get("message"), str):
            operations.append({"op": "fail", "message": args["message"]})
        elif name == "report_blocker" and isinstance(args.get("missing"), str):
            operations.append({"op": "block", "missing": args["missing"]})
        else:
            raise ValueError(f"action {name!r} needs a richer adapter")
    return operations


def convert(case: dict, folder: Path) -> dict:
    if case.get("schema") != "natlang.playground.case/1" or case.get("reviewStatus") != "accepted" or case.get("admission", {}).get("admitted") is not True:
        raise ValueError("case is not reviewed and exactly admitted")
    events = case.get("trace") or []
    if not events or events[0].get("kind") != "manifest" or any(
        event.get("seq") != index or event.get("version") != "reduction-trace/1"
        for index, event in enumerate(events)
    ):
        raise ValueError("missing or discontinuous reduction trace")
    final = [event for event in events if event.get("kind") == "state" and event.get("phase") == "final"]
    if len(final) != 1:
        raise ValueError("trace needs one final state")
    expected = case.get("expected") or {}
    if expected.get("kind") != final[0].get("outcome") or (
        expected.get("kind") == "done" and expected.get("value") != final[0].get("value")
    ):
        raise ValueError("expected result differs from recorded outcome")
    if any(event.get("kind") == "effect" for event in events):
        raise ValueError("host effects need a richer adapter")
    root = _root(case, folder)
    operations = _operations(events)
    semantic_actions = [event for event in events if event.get("kind") == "action"]
    contract = {"kind": expected["kind"], "value": expected.get("value"),
                "effects": case.get("effects") or [],
                "required_actions": [{"tool": event["name"], "arguments": event["arguments"]}
                                     for event in semantic_actions],
                "instruction_obligations": case.get("requiredActions") or [],
                "constrained_calls": case.get("constrainedCalls") or []}
    if expected["kind"] in {"error", "blocked"}:
        last = semantic_actions[-1]
        contract["explanation"] = next(iter((last.get("arguments") or {}).values()), "")
    record = {"version": VERSION, "id": str(case["id"]), "kind": "lambda_scenario",
              "source": "browser_playground", "split": case.get("split", "train"),
              "source_ids": [str(case["id"])],
              "source_groups": [str(case.get("groupId") or case.get("projectId") or case["id"])],
              "source_revisions": [str(case["revision"])], "license": "user-authored",
              "gold_sources": ["reviewed-browser-case"],
              "semantics": {"root": root, "operations": operations, "contract": contract}}
    return validate(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--rejects", type=Path)
    args = parser.parse_args()
    if args.src.resolve() == args.dst.resolve():
        parser.error("input and output must differ")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = args.rejects or args.dst.with_name(args.dst.stem + "-rejects.jsonl")
    accepted = rejected = 0
    with args.src.open() as source, args.dst.open("w") as output, rejects_path.open("w") as rejects:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            case = None
            try:
                case = json.loads(line)
                with tempfile.TemporaryDirectory(prefix="natlang-case-") as directory:
                    record = convert(case, Path(directory))
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                accepted += 1
            except Exception as error:
                rejects.write(json.dumps({"line": line_number, "id": case.get("id") if isinstance(
                    case, dict) else None, "reason": str(error)}, ensure_ascii=False) + "\n")
                rejected += 1
    print(json.dumps({"accepted": accepted, "rejected": rejected,
                      "program_ir": str(args.dst), "rejects": str(rejects_path)}))


if __name__ == "__main__":
    main()
