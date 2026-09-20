"""Explicit source/type/run operations that an embedding may expose inside eval."""
from __future__ import annotations

import copy
from dataclasses import dataclass

from .codebase import CheckedGraph, from_definitions
from .host import load_definitions
from .invocation import RunOptions
from .runtime import Runtime
from .trace import TraceReader, TraceRecorder
from .types import TypeEnv, fits, format_type, parse_type
from .values import dump


@dataclass(frozen=True)
class ChildResult:
    source_revision: str
    parent_call_id: str | None
    outcome: str
    value: object
    trace: list[dict]


class SourceWorkspace:
    """A versioned in-memory source container; active graphs are snapshots."""

    def __init__(self, definitions: dict, root: str):
        self.definitions = copy.deepcopy(definitions)
        self.root = root
        self.graph = from_definitions(self.definitions, root)

    def edited(self, name: str, doc: dict) -> "SourceWorkspace":
        definitions = copy.deepcopy(self.definitions)
        definitions[name] = copy.deepcopy(doc)
        return SourceWorkspace(definitions, self.root)

    def describe(self, name: str | None = None) -> dict:
        fn = self.graph.get(name or self.root)
        return {"name": fn.name, "signature": fn.signature, "kind": fn.kind,
                "engine": fn.engine if fn.kind == "code" else None,
                "functions": sorted(fn.codebase), "revision": self.graph.revision}

    def type_check(self, actual: str, expected: str) -> dict:
        a, b = parse_type(actual), parse_type(expected)
        env = TypeEnv()
        return {"actual": format_type(a), "expected": format_type(b), "fits": fits(a, b, env)}

    def invoke(self, name: str, inputs: dict, *, agent_factory, options: RunOptions,
               executors: dict | None = None, capabilities: dict | None = None,
               parent_call_id: str | None = None, max_episodes: int | None = None,
               parent_runtime: Runtime | None = None) -> ChildResult:
        max_episodes = options.max_episodes if max_episodes is None else max_episodes
        if ((max_episodes is not None and max_episodes < 1) or
                (options.max_episodes is not None and max_episodes is not None and
                 max_episodes > options.max_episodes)):
            raise ValueError("child episode budget must be positive and bounded by explicit parent budget")
        graph, root = load_definitions(self.definitions, name, inputs)
        child_options = RunOptions(seed=options.seed, model=options.model,
                                   world_seed=options.world_seed, max_episodes=max_episodes,
                                   max_depth=options.max_depth, max_actions=options.max_actions,
                                   max_tool_calls=options.max_tool_calls)
        recorder = TraceRecorder({"run_id": child_options.run_id, "source_sha256": graph.revision,
                                  "parent_call_id": parent_call_id, "seed_policy": vars(options.seed),
                                  "capture": "reduction"})
        runtime = Runtime(agent_factory, options=child_options, executors=executors,
                          capabilities=capabilities, trace_sink=recorder,
                          _budget=parent_runtime._budget if parent_runtime else None,
                          _parent_path=parent_call_id,
                          _call_prefix=f"{parent_call_id}/child" if parent_call_id else "")
        out, value = runtime.run_root(root)
        if parent_runtime is not None:
            parent_runtime.episodes_started = parent_runtime._budget.used
        return ChildResult(graph.revision, parent_call_id, out.kind,
                           dump(value) if out.kind == "done" else None, recorder.events)


def meta_capabilities(workspace: SourceWorkspace, *, agent_factory, options: RunOptions,
                      executors: dict | None = None, parent_runtime: Runtime | None = None) -> dict:
    """Choose individual wrappers to put in a lambda's declared effects.

    This does not pass through the caller's locals, capabilities, or native host
    authority. The embedding must explicitly supply any child capability.
    """
    def describe(args):
        return workspace.describe(args[0] if args else None)

    def check(args):
        return workspace.type_check(args[0], args[1])

    def invoke(args):
        request = args[0]
        result = workspace.invoke(request["name"], request.get("inputs") or {},
                                  agent_factory=agent_factory, options=options, executors=executors,
                                  parent_call_id=request.get("parent_call_id"),
                                  max_episodes=request.get("max_episodes"),
                                  parent_runtime=parent_runtime)
        return {"source_revision": result.source_revision, "parent_call_id": result.parent_call_id,
                "outcome": result.outcome, "value": result.value, "trace": result.trace}

    return {"meta.describe": describe, "meta.check": check, "meta.invoke": invoke}
