"""Command line: python -m natlang run PROGRAM --in name=path --server URL"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .decoder import WRAPPERS, LlamaServerDecoder
from .host import export, load
from .model_agent import SMALL_PROMPT, SYSTEM_PROMPT, ModelAgent
from .runtime import Runtime


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="natlang")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="reduce a program with a model behind a llama.cpp server")
    run.add_argument("program", type=Path)
    run.add_argument("--in", dest="inputs", action="append", default=[], metavar="NAME=PATH")
    run.add_argument("--server", default="http://127.0.0.1:8080", help="llama-server base URL")
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--prompt", choices=("small", "full"), default="small",
                     help="small: for an untuned small model; full: for a capable teacher")
    run.add_argument("--wrapper", choices=tuple(WRAPPERS), default="generic",
                     help="generic: template-agnostic; lfm: native tool-call token; reasoning: think first")
    run.add_argument("--format", choices=("yaml", "json"), default="yaml")
    run.add_argument("--trace", type=Path, help="write the action trace as JSON lines")
    a = ap.parse_args(argv)

    inputs = dict(kv.split("=", 1) for kv in a.inputs)
    doc_inputs = {}
    import yaml
    doc = yaml.safe_load(a.program.read_text())
    if isinstance(doc, dict) and "inputs" in doc:     # conformance files carry their own inputs
        doc_inputs = doc["inputs"]
    root = load(a.program, {**doc_inputs, **inputs})
    decoder = LlamaServerDecoder(a.server)
    prompt = SMALL_PROMPT if a.prompt == "small" else SYSTEM_PROMPT
    rt = Runtime(lambda lam: ModelAgent(decoder, wrapper=WRAPPERS[a.wrapper], temperature=a.temperature,
                                           system_prompt=prompt))
    out, value = rt.run_root(root)
    if a.trace:
        a.trace.write_text("".join(json.dumps(t, default=str) + "\n" for t in rt.trace))
    print(f"# {out.kind}" + (f": {out.detail}" if out.kind != "done" else ""), file=sys.stderr)
    print(export(value, a.format))
    return 0 if out.kind == "done" else 2


if __name__ == "__main__":
    sys.exit(main())
