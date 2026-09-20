"""Command line: python -m natlang run PROGRAM --in name=path --server URL"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .native import NativeCallDecoder
from .host import export, load
from .runtime import Runtime
from .tool_agent import ToolAgent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="natlang")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="reduce a program with a model behind a llama.cpp server")
    run.add_argument("program", type=Path)
    run.add_argument("--in", dest="inputs", action="append", default=[], metavar="NAME=PATH")
    run.add_argument("--server", default="http://127.0.0.1:8080", help="llama-server base URL")
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--format", choices=("yaml", "json"), default="yaml")
    run.add_argument("--trace", type=Path, help="write the action trace as JSON lines")
    a = ap.parse_args(argv)

    inputs = dict(kv.split("=", 1) for kv in a.inputs)
    doc_inputs = {}
    import yaml
    doc = None if a.program.suffix in (".nl", ".ts") else yaml.safe_load(a.program.read_text())
    if isinstance(doc, dict) and "inputs" in doc:     # conformance files carry their own inputs
        doc_inputs = doc["inputs"]
    root = load(a.program, {**doc_inputs, **inputs})
    decoder = NativeCallDecoder(a.server)
    rt = Runtime(lambda lam: ToolAgent(decoder, temperature=a.temperature))
    out, value = rt.run_root(root)
    if a.trace:
        a.trace.write_text("".join(json.dumps(t, default=str) + "\n" for t in rt.trace))
    print(f"# {out.kind}" + (f": {out.detail}" if out.kind != "done" else ""), file=sys.stderr)
    print(export(value, a.format))
    return 0 if out.kind == "done" else 2


if __name__ == "__main__":
    sys.exit(main())
