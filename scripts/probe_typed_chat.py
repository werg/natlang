#!/usr/bin/env python3
"""Probe whether the current chat tool parser preserves an exact nested record."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from natlang.decoder import LlamaServerDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8081")
    args = parser.parse_args()
    value = {"type": "object", "properties": {
        "id": {"type": "number"}, "label": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["id", "label", "tags"], "additionalProperties": False}
    tool = {"type": "function", "function": {"name": "write",
        "description": "Write the exact requested record.",
        "parameters": {"type": "object", "properties": {
            "path": {"const": "return"}, "type": {"const": "Probe"}, "value": value},
            "required": ["path", "type", "value"], "additionalProperties": False}}}
    decoder = LlamaServerDecoder(args.server, chat_extra={
        "thinking_budget_tokens": 256, "top_p": 0.95, "top_k": 20,
        "chat_template_kwargs": {"reasoning_effort": "low"}})
    prompt = [{"role": "user", "content":
        'Call write once with path "return", type "Probe", and value '
        '{"id":7,"label":"sample","tags":["a","b"]}. Do not reply in prose.'}]
    expected = [("write", {"path": "return", "type": "Probe",
                           "value": {"id": 7, "label": "sample", "tags": ["a", "b"]}})]
    exact = decoder.chat(prompt, [tool], temperature=0, seed=11)
    one_of = {"type": "function", "function": {**tool["function"],
        "parameters": {"oneOf": [tool["function"]["parameters"],
                                 {"type": "object", "properties": {
                                     "path": {"const": "let/note"},
                                     "type": {"const": "Text"},
                                     "value": {"type": "string"}},
                                  "required": ["path", "type", "value"],
                                  "additionalProperties": False}]}}}
    variant = decoder.chat(prompt, [one_of], temperature=0, seed=11)
    separate = [{"type": "function", "function": {**tool["function"], "name": "write_return"}},
                {"type": "function", "function": {**tool["function"], "name": "write_note",
                 "parameters": one_of["function"]["parameters"]["oneOf"][1]}}]
    split = decoder.chat(prompt, separate, temperature=0, seed=11)
    record = {"exact": {"calls": exact.calls, "text": exact.text,
                         "raw_response": exact.raw_response,
                         "parser_preserved": exact.calls == expected},
              "one_of": {"calls": variant.calls, "text": variant.text,
                         "raw_response": variant.raw_response,
                         "parser_preserved": variant.calls == expected},
              "separate": {"calls": split.calls, "text": split.text,
                           "raw_response": split.raw_response,
                           "parser_preserved": split.calls == [("write_return", expected[0][1])]}}
    with args.out.open("x") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({kind: {"parser_preserved": item["parser_preserved"],
                             "calls": item["calls"], "text": item["text"]}
                      for kind, item in record.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
