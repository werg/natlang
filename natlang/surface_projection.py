"""Explicit projection of historical engine-less tool calls onto tools-v3."""
from __future__ import annotations

import copy
import json


def project_legacy_turn(row: dict, *, engine: str = "quickjs-isolated") -> dict:
    """Return a new row; retain the original alongside it in corpus storage.

    Only known historical run_code calls are changed. Tool schemas and model
    messages need rebuilding by the materializer for the selected surface.
    """
    projected = copy.deepcopy(row)

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if value.get("name") == "run_code" and isinstance(value.get("args"), dict):
                value["args"].setdefault("engine", engine)
            function = value.get("function")
            if isinstance(function, dict) and function.get("name") == "run_code" and "arguments" in function:
                args = json.loads(function["arguments"])
                if "engine" not in args:
                    args["engine"] = engine
                    function["arguments"] = json.dumps(args, ensure_ascii=False)
            for item in value.values():
                visit(item)

    visit(projected)
    projected["surface_projection"] = {"from": "tools-v2", "to": "tools-v3",
                                       "historical_engine": engine}
    return projected
