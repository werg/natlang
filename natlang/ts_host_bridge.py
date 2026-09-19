"""Private JSONL bridge for the TypeScript application host.

The Python runtime remains the semantic authority. The Node host owns model
turns and, when selected, the live TypeScript eval environment. One process
handles one run, so an RPC failure cannot leave a reusable half-run behind.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from . import js
from .decoder import ChatTurn
from .execution import CrispRequest, ExecutionError, QuickJSExecutor
from .host import load_definitions
from .invocation import ModelSettings, RunOptions, SeedPolicy
from .runtime import Runtime
from .streams import Poll, StreamBuffer
from .tool_agent import ToolAgent
from .types import TypeEnv
from .values import coerce, dump_state, load_program

PROTOCOL = "natlang-ts-host/1"


class Channel:
    def __init__(self):
        self.serial = 0

    def send(self, data):
        sys.stdout.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        sys.stdout.flush()

    def read(self):
        line = sys.stdin.readline()
        if not line:
            raise ConnectionError("TypeScript host disconnected")
        value = json.loads(line)
        if not isinstance(value, dict) or value.get("protocol") != PROTOCOL:
            raise ValueError("invalid TypeScript host protocol message")
        return value

    def request(self, kind, **payload):
        self.serial += 1
        serial = self.serial
        self.send({"protocol": PROTOCOL, "id": serial, "kind": kind, **payload})
        response = self.read()
        if response.get("id") != serial or response.get("kind") != "reply":
            raise ValueError("mismatched TypeScript host reply")
        if "error" in response:
            error = ExecutionError(str(response["error"]))
            error.events = response.get("events") or []
            raise error
        return response.get("value")


class RemoteDecoder:
    def __init__(self, channel):
        self.channel = channel
        self.deadline = None

    def chat(self, messages, tools, *, temperature, seed=None, max_tokens=700):
        value = self.channel.request("model_turn", messages=messages, tools=tools,
                                     temperature=temperature, seed=seed, max_tokens=max_tokens)
        if not isinstance(value, dict):
            raise ValueError("model turn must be a record")
        calls = value.get("calls", [])
        if not isinstance(calls, list) or any(not isinstance(c, list) or len(c) != 2 or
                                               not isinstance(c[0], str) or not isinstance(c[1], dict)
                                               for c in calls):
            raise ValueError("model calls must be [name, arguments] pairs")
        return ChatTurn(calls=calls, text=value.get("text", ""),
                        raw_calls=value.get("raw_calls", []),
                        completion_tokens=value.get("completion_tokens"),
                        value_confidence=value.get("value_confidence", []),
                        raw_response=value.get("raw_response"))


class RemoteTypeScriptExecutor:
    name = "typescript-host"
    parallel_safe = False

    def __init__(self, channel):
        self.channel = channel
        self.events = []

    def run(self, request: CrispRequest, effect):
        try:
            value = self.channel.request("eval", code=request.code,
                                         scope=js.to_js(request.scope), body=request.body,
                                         path=request.path, effectful=request.effectful)
        except ExecutionError as error:
            self.events.extend(getattr(error, "events", []))
            raise
        if not isinstance(value, dict) or "result" not in value:
            raise ExecutionError("TypeScript host returned an invalid eval response")
        self.events.extend(value.get("events", []))
        return value["result"]

    def drain_events(self):
        events, self.events = self.events, []
        return events


class RemoteSource:
    def __init__(self, channel, part):
        self.channel, self.part = channel, part

    def poll(self):
        try:
            value = self.channel.request("stream_poll", part=self.part)
            if not isinstance(value, dict) or value.get("kind") not in ("item", "closed", "failed"):
                raise ValueError("stream poll must yield item, closed or failed")
            return Poll(value["kind"], value.get("value"), value.get("detail", ""))
        except Exception as error:
            return Poll("failed", detail=str(error))


def _root(start, channel):
    source = start.get("source")
    if not isinstance(source, dict):
        raise ValueError("source must be a record")
    if source.get("kind") == "definitions":
        _, root = load_definitions(source["entries"], source["root"], start.get("inputs") or {})
    elif source.get("kind") == "program":
        root = load_program(source["program"])
        for name, value in (start.get("inputs") or {}).items():
            if name not in root.type.params:
                raise ValueError(f"{name} is not a program parameter")
            root.in_[name] = coerce(value, root.type.params[name][0], root.env(TypeEnv()),
                                    yaml=False, path=f"args/{name}")
    else:
        raise ValueError("source.kind must be program or definitions")
    for part in start.get("stream_parts") or []:
        if part != "over" or not hasattr(root, part):
            raise ValueError(f"unsupported root stream part {part!r}")
        setattr(root, part, StreamBuffer(RemoteSource(channel, part)))
    return root


def _options(start):
    raw = start.get("options") or {}
    seed = raw.get("seed") or {}
    model = raw.get("model")
    return RunOptions(seed=SeedPolicy(**seed), model=ModelSettings(**model) if model else None,
                      world_seed=raw.get("world_seed"),
                      max_episodes=raw.get("max_episodes", 256),
                      max_depth=raw.get("max_depth", 8),
                      **({"run_id": raw["run_id"]} if "run_id" in raw else {}))


def main():
    channel = Channel()
    try:
        start = channel.read()
        if start.get("kind") != "start":
            raise ValueError("first message must start a run")
        root = _root(start, channel)
        options = _options(start)
        decoder = RemoteDecoder(channel)
        agent_factory = lambda _: ToolAgent(decoder)
        engines = {"quickjs-isolated": QuickJSExecutor()}
        if start.get("typescript", True):
            engines["typescript-host"] = RemoteTypeScriptExecutor(channel)
        trace_path = Path(start["trace_path"]) if start.get("trace_path") else None
        runtime = Runtime(agent_factory, executors=engines, engine_selection=True,
                          options=options, trace_path=trace_path,
                          map_workers=start.get("map_workers", 1))
        outcome, value = runtime.run_root(root)
        channel.send({"protocol": PROTOCOL, "kind": "complete",
                      "outcome": {"kind": outcome.kind, "path": outcome.path, "detail": outcome.detail},
                      "value": dump_state(value), "emitted": runtime.emitted,
                      "trace": runtime.trace_sink.events if runtime.trace_sink else None,
                      "run_id": options.run_id})
    except Exception as exc:
        channel.send({"protocol": PROTOCOL, "kind": "fatal", "error": str(exc),
                      "type": type(exc).__name__, "stack": traceback.format_exc(limit=8)})


if __name__ == "__main__":
    main()
