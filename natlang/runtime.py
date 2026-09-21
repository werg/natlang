"""The runtime: sessions that apply actions, and triggering of pending nodes."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path as FilePath
from typing import Any, Callable, Optional

import yaml

from . import js
from .execution import CrispRequest, ExecutionError, QuickJSExecutor, portable
from .invocation import Invocation, RunOptions
from .trace import TraceRecorder, _view, changes
from .streams import StreamBuffer
from .actions import Action
from .diag import BLOCKS, Diagnostic, Refuse, Reject, reject
from .nodes import (DONE, MISSING, QUIESCED, WAITING, RUNNING, UNREDUCED, FoldNode, IterateNode, Lambda,
                    MapNode, Pending, is_pending)
from .paths import Path, parse_path
from .refs import Ref, pending_refs_under, resolve
from .render import opening, pending_line, render, scalar
from .types import (TEXT, LambdaT, ListT, Record, TypeEnv, TypeSyntaxError, PENDING_TYPES, fits,
                    format_type, is_pending_type, parse_type, FoldT, IterateT, MapT)
from .values import (body_lambda_fits, build_pending, coerce, dump, problems, unbound_parts)
from .values import dump_state

# Reference values for explicit legacy budgets; RunOptions defaults to None.
MAX_ACTIONS = 40
MAX_TOOL_CALLS = 128
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
_WRAPPER_FOR = {LambdaT: "$lambda", MapT: "$map", FoldT: "$fold", IterateT: "$iterate"}


class OpenList(list):
    """A list an external source keeps appending to (SPEC 4.4)."""

    is_open = True

    def __init__(self, source):
        super().__init__()
        self._source = iter(source)

    def pull(self) -> bool:
        if not self.is_open:
            return False
        try:
            item = next(self._source)
        except StopIteration:
            self.is_open = False
            return False
        if item == "$close":
            self.is_open = False
            return False
        self.append(item)
        return True

    def __deepcopy__(self, memo):
        return self


@dataclass
class Result:
    kind: str  # ok | rejected | refused | completed | done | quiesced | replaced | error | budget | blocked
    text: str = ""
    diags: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    value: Any = None

    @property
    def codes(self):
        return [d.code for d in self.diags]


@dataclass
class Outcome:
    path: str
    kind: str  # done | quiesced | replaced
    detail: str = ""
    value: Any = None


class EpisodeBudget:
    def __init__(self, limit: Optional[int]):
        self.limit, self.used = limit, 0
        self.lock = threading.Lock()

    def reserve(self) -> bool:
        with self.lock:
            if self.limit is not None and self.used >= self.limit:
                return False
            self.used += 1
            return True


class Runtime:
    def __init__(self, agent_factory: Callable[[Lambda], Any], capabilities: Optional[dict] = None,
                 max_episodes: Optional[int] = None, max_depth: Optional[int] = None,
                 options: Optional[RunOptions] = None,
                 executor=None, trace_sink: Optional[TraceRecorder] = None,
                 trace_path: Optional[FilePath] = None, executors: Optional[dict] = None,
                 engine_selection: bool = False, map_workers: int = 1,
                 parallel_model_safe: bool = False, _budget: Optional[EpisodeBudget] = None,
                 _parent_path: Optional[str] = None, _call_prefix: str = ""):
        self.agent_factory = agent_factory
        self.options = options or RunOptions.compatibility(max_episodes=max_episodes, max_depth=max_depth)
        self.max_episodes, self.max_depth = self.options.max_episodes, self.options.max_depth
        self.executor = executor or QuickJSExecutor()
        self.executors = dict(executors or {getattr(self.executor, "name", "quickjs-isolated"): self.executor})
        self.engine_selection = engine_selection
        if map_workers < 1:
            raise ValueError("map_workers must be positive")
        self.map_workers, self.parallel_model_safe = map_workers, parallel_model_safe
        self._budget = _budget or EpisodeBudget(self.max_episodes)
        self._parent_path = _parent_path
        self._call_prefix = _call_prefix
        self.trace_sink = trace_sink
        self.trace_path = trace_path
        self.deadline = None
        self._depth = 0
        self._fn_stack: list = []   # names of the code-base functions currently being reduced
        self._stack: list = []   # hashes of (body, args) of the lambdas currently being reduced
        self.emitted: list = []
        self.capabilities = {"out.emit": lambda args: self.emitted.append(args[0] if args else None)}
        self.capabilities.update(capabilities or {})
        self.origins: dict = {}
        self.trace: list = []
        self.episodes_started = 0
        self._ids = 0
        self.invocations: list[Invocation] = []

    # ------------------------------------------------------------------ running programs
    def run_root(self, root: Pending, env: Optional[TypeEnv] = None):
        """Reduce a root pending node. Returns (outcome, value-or-node)."""
        holder = _Box(root)
        self._root_holder = holder
        if self.trace_path is not None and self.trace_sink is None:
            initial = _view(dump_state(root))
            digest = hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()
            self.trace_sink = TraceRecorder({"run_id": self.options.run_id, "source_sha256": digest,
                                             "seed_policy": vars(self.options.seed),
                                             "backend_seed_range": self.options.seed.backend_range,
                                             "world_seed": self.options.world_seed,
                                             "model_settings": vars(self.options.model) if self.options.model else None,
                                             "tool_schema": "tools-v3" if self.engine_selection else "tools-v2",
                                             "engines": sorted(self.executors),
                                             "engine_selection": self.engine_selection,
                                             "coverage": "natlang-state-and-declared-effects"}, self.trace_path)
        elif self.trace_path is not None:
            self.trace_sink.reopen()
        initial_state = _view(dump_state(root))
        self._trace_last_state = initial_state
        self._observe("state", phase="initial", value=initial_state)
        ref = Ref(type=None, env=env or TypeEnv(), path="", holder=holder, attr="value")
        try:
            out = self.trigger(ref, acting_effects=None)
            self._observe_state("final", outcome=out.kind)
            return out, holder.value
        finally:
            if self.trace_path is not None:
                self.trace_sink.close()

    def _observe(self, kind: str, **data):
        if self.trace_sink is not None:
            self.trace_sink.emit(kind, **data)

    def _observe_state(self, phase: str, outcome: Optional[str] = None):
        if self.trace_sink is not None and hasattr(self, "_root_holder"):
            state = _view(dump_state(self._root_holder.value))
            delta = changes(self._trace_last_state, state)
            if delta:
                self._observe("reduction", phase=phase, changes=delta)
            self._observe("state", phase=phase, value=state, **({"outcome": outcome} if outcome else {}))
            self._trace_last_state = state

    def _executor(self, engine: str):
        selected = self.executors.get(engine)
        if selected is None:
            raise ExecutionError(f"engine {engine!r} is unavailable; available: {', '.join(sorted(self.executors))}")
        return selected

    def _drain_executor_events(self, executor):
        if self.trace_sink is not None and hasattr(executor, "drain_events"):
            for event in executor.drain_events():
                self._observe("host", engine=getattr(executor, "name", type(executor).__name__), **event)

    # ------------------------------------------------------------------ trigger
    def trigger(self, ref: Ref, acting_effects) -> Outcome:
        node = ref.get()
        if not is_pending(node):
            raise reject(ref.path, "no-such-path", "a pending node")
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return self._quiesce(node, ref, "run wall-clock budget exhausted")
        if node.status == RUNNING:
            raise reject(ref.path, "frozen", "a node that is not running")
        inner = node.env(ref.env)
        for dep in self._dependencies(node, ref, inner):
            out = self.trigger(dep, acting_effects)
            if out.kind != "done":
                raise Refuse(Diagnostic(dep.path, "stuck-dependency", BLOCKS, got=out.detail))
        unbound = unbound_parts(node, ref.env, ref.path)
        if unbound:
            raise Refuse(*unbound)
        if isinstance(node, Lambda):
            return self._run_crisp(node, ref, inner) if node.is_crisp else self._run_episode(node, ref)
        if isinstance(node, MapNode):
            return self._run_map(node, ref, inner)
        if isinstance(node, FoldNode):
            return self._run_fold(node, ref, inner)
        return self._run_iterate(node, ref, inner)

    def _dependencies(self, node, ref, inner):
        def attr_ref(attr, t):
            return Ref(type=t, env=inner, path=f"{ref.path}/{attr}".lstrip("/"), holder=node, attr=attr)

        if isinstance(node, Lambda):
            in_ref = Ref(type=node.type.params, env=inner, path=f"{ref.path}/args".lstrip("/"),
                         holder=node, attr="in_")
            for name, ft, _ in node.type.params.fields:
                if name in node.in_ and not is_pending_type(ft, inner):
                    yield from pending_refs_under(in_ref.child(name))
            return
        from .values import part_type
        parts = {MapNode: ("over",), FoldNode: ("over", "init"), IterateNode: ("init",)}[type(node)]
        for part in parts:
            yield from pending_refs_under(attr_ref(part, part_type(node, part)))

    def _swap_out(self, node: Pending, ref: Ref, value: Any) -> Outcome:
        node.status = DONE
        ref.set(value)
        self.origins[ref.slot_key()] = node
        self._observe("node", path=ref.path, transition="done", node_type=type(node).__name__)
        return Outcome(ref.path, "done", _one_line(value), value)

    def _quiesce(self, node: Pending, ref: Ref, note: str) -> Outcome:
        node.status, node.note = QUIESCED, note
        self._observe("node", path=ref.path, transition="quiesced", node_type=type(node).__name__, detail=note)
        return Outcome(ref.path, "quiesced", note)

    # -- crisp lambda
    def _run_crisp(self, node: Lambda, ref: Ref, inner: TypeEnv) -> Outcome:
        node.status = RUNNING
        if node.original_body is None:
            node.original_body = node.body
        scope = {"args": node.in_, "return": node.ret}
        try:
            executor = self._executor(node.engine)
        except ExecutionError as e:
            return self._quiesce(node, ref, f"code error: {e}")
        self._observe("eval", phase="start", path=ref.path, mode="body",
                      engine=node.engine,
                      code=node.body, effectful=bool(node.effects))
        try:
            try:
                raw = portable(executor.run(CrispRequest(node.body, scope, True, ref.path,
                                                              bool(node.effects)), self._fx(node)))
            finally:
                self._drain_executor_events(executor)
            value = coerce(raw, node.type.returns, inner, yaml=False, path=ref.path)
        except ExecutionError as e:
            self._observe("eval", phase="failed", path=ref.path, error=str(e))
            return self._quiesce(node, ref, f"code error: {e}")
        except Reject as e:
            self._observe("eval", phase="rejected", path=ref.path, error=str(e))
            return self._quiesce(node, ref, f"rejected: {e}")
        self._observe("eval", phase="completed", path=ref.path, value=dump(value))
        if is_pending(value):
            node.status = DONE
            ref.set(value)
            return Outcome(ref.path, "replaced", f"{format_type(value.type)} unreduced")
        holes, _ = problems(value, node.type.returns, inner, ref.path)
        if holes:
            return self._quiesce(node, ref, "returned value is incomplete: " + "; ".join(map(str, holes)))
        return self._swap_out(node, ref, value)

    # -- natural-language lambda
    def _run_episode(self, node: Lambda, ref: Ref) -> Outcome:
        if self.max_depth is not None and self._depth >= self.max_depth:
            return self._quiesce(node, ref, f"run budget: episodes nested deeper than {self.max_depth}")
        if ((self.max_episodes is not None and self.episodes_started >= self.max_episodes) or
                (self._budget.limit is not None and self._budget.used >= self._budget.limit)):
            return self._quiesce(node, ref, f"run budget: more than {self.max_episodes} episodes")
        key = _hash({"body": node.body, "args": node.in_, "type": format_type(node.type)})
        if key in self._stack:
            return self._quiesce(node, ref, "identical to a lambda already being reduced above it: "
                                            "delegating the same task to a child cannot make progress")
        if node.fn_name and node.fn_name in self._fn_stack:     # the loader refuses cycles; an edited copy could
            return self._quiesce(node, ref, f"recursion: {node.fn_name} is already running above this call")
        self._depth += 1
        self._stack.append(key)
        self._fn_stack.append(node.fn_name)
        try:
            return self._episode(node, ref)
        finally:
            self._depth -= 1
            self._stack.pop()
            self._fn_stack.pop()

    def _episode(self, node: Lambda, ref: Ref) -> Outcome:
        cold = node.status == QUIESCED
        node.status, node.note = RUNNING, ""
        node.attempts += 1
        parent = self.invocations[-1].path if self.invocations else self._parent_path
        logical_path = f"{self._call_prefix}/{ref.path}".rstrip("/") if self._call_prefix else ref.path
        invocation = Invocation(self.options.run_id, logical_path, node.attempts, parent)
        if node.original_body is None:
            node.original_body = node.body
        if not self._budget.reserve():
            return self._quiesce(node, ref, f"run budget: more than {self.max_episodes} episodes")
        self.episodes_started = self._budget.used
        session = Session(self, node, ref.env, cold=cold)
        session.invocation = invocation
        self._observe("invocation", phase="start", call_id=invocation.call_id,
                      path=invocation.path, attempt=invocation.attempt, parent_path=parent)
        self.invocations.append(invocation)
        try:
            note = self.agent_factory(node).run(session)
        finally:
            self.invocations.pop()
            self._observe("invocation", phase="end", call_id=invocation.call_id)
        if session.completed:
            return self._swap_out(node, ref, node.ret)
        return self._quiesce(node, ref, note or "budget exhausted")

    # -- Map
    def _run_map(self, node: MapNode, ref: Ref, inner: TypeEnv) -> Outcome:
        if node.slots is None:
            wants_index = node.fn.type.params.get("index") is not None
            node.slots = []
            for i, item in enumerate(node.over):
                inst = _instantiate(node.fn)
                inst.in_[node.item_name] = copy.deepcopy(item)
                if wants_index:
                    inst.in_["index"] = i
                node.slots.append(inst)
        node.status = RUNNING
        rt = ref.env.resolve(ref.type) if ref.type is not None else None
        elem = rt.elem if isinstance(rt, ListT) else node.type.b
        if self._can_parallel_map(node):
            return self._run_map_parallel(node, ref, inner, elem)
        stuck = []
        for i, slot in enumerate(node.slots):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                return self._quiesce(node, ref, "run wall-clock budget exhausted")
            if not is_pending(slot):
                continue
            sref = Ref(type=elem, env=inner, path=f"{ref.path}/{i}", container=node.slots, key=i)
            out = self.trigger(sref, None)
            if out.kind != "done":
                stuck.append(out)
        if not stuck:
            return self._swap_out(node, ref, list(node.slots))
        done = len(node.slots) - len(stuck)
        detail = f"{done} of {len(node.slots)} reduced\n" + "\n".join(
            f"{o.path}: {o.kind} \"{o.detail}\"" for o in stuck)
        return self._quiesce(node, ref, detail)

    def _can_parallel_map(self, node: MapNode) -> bool:
        if self.map_workers <= 1 or not self.parallel_model_safe or not isinstance(node.over, list):
            return False
        if any(not getattr(engine, "parallel_safe", False) for engine in self.executors.values()):
            return False
        def pure(fn, seen):
            if id(fn) in seen:
                return True
            seen.add(id(fn))
            return not fn.effects and all(pure(child, seen) for child in fn.codebase.values())
        return isinstance(node.fn, Lambda) and pure(node.fn, set())

    def _run_map_parallel(self, node: MapNode, ref: Ref, inner: TypeEnv, elem) -> Outcome:
        pending = [i for i, slot in enumerate(node.slots) if is_pending(slot)]
        results = {}
        def reduce_slot(i):
            child = Runtime(self.agent_factory, capabilities={}, options=self.options,
                            executors=self.executors, engine_selection=self.engine_selection,
                            trace_sink=self.trace_sink, _budget=self._budget,
                            _parent_path=ref.path, _call_prefix=self._call_prefix, map_workers=1)
            child.deadline = self.deadline
            child._depth = self._depth
            child._stack = list(self._stack)
            child._fn_stack = list(self._fn_stack)
            slot_ref = Ref(type=elem, env=inner, path=f"{ref.path}/{i}", container=node.slots, key=i)
            result = child.trigger(slot_ref, None)
            return result, child.origins

        with ThreadPoolExecutor(max_workers=self.map_workers) as pool:
            cursor = iter(pending)
            active = {}
            for _ in range(min(self.map_workers, len(pending))):
                i = next(cursor)
                active[pool.submit(reduce_slot, i)] = i
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    i = active.pop(future)
                    outcome, origins = future.result()
                    results[i] = outcome
                    self.origins.update(origins)
                    self._observe("map_slot", path=f"{ref.path}/{i}", slot=i,
                                  outcome=outcome.kind, value=dump(node.slots[i]))
                    try:
                        following = next(cursor)
                    except StopIteration:
                        continue
                    active[pool.submit(reduce_slot, following)] = following
        self.episodes_started = self._budget.used
        stuck = [results[i] for i in sorted(results) if results[i].kind != "done"]
        if not stuck:
            return self._swap_out(node, ref, list(node.slots))
        done = len(node.slots) - len(stuck)
        detail = f"{done} of {len(node.slots)} reduced\n" + "\n".join(
            f"{o.path}: {o.kind} \"{o.detail}\"" for o in stuck)
        return self._quiesce(node, ref, detail)

    # -- Fold
    def _run_fold(self, node: FoldNode, ref: Ref, inner: TypeEnv) -> Outcome:
        node.status = RUNNING
        if node.acc is MISSING:
            node.acc, node.at = copy.deepcopy(node.init), 0
        if isinstance(node.over, StreamBuffer):
            return self._run_stream_fold(node, ref, inner)
        while True:
            if node.at >= len(node.over):
                if isinstance(node.over, OpenList) and node.over.pull():
                    continue
                break
            if node.current is None:
                inst = _instantiate(node.step)
                inst.in_["acc"] = copy.deepcopy(node.acc)
                inst.in_["item"] = copy.deepcopy(node.over[node.at])
                node.current = inst
            cref = Ref(type=node.type.s, env=inner, path=f"{ref.path}/step/{node.at}", holder=node, attr="current")
            out = self.trigger(cref, None)
            if out.kind != "done":
                return self._quiesce(node, ref, f"step {node.at} {out.kind}: {out.detail}")
            node.acc, node.current, node.at = node.current, None, node.at + 1
        return self._swap_out(node, ref, node.acc)

    def _run_stream_fold(self, node: FoldNode, ref: Ref, inner: TypeEnv) -> Outcome:
        source = node.over
        while True:
            polled = source.peek()
            if polled.kind == "empty":
                node.status, node.note = WAITING, "waiting for stream input"
                self._observe("stream", path=ref.path, phase="waiting", position=source.position)
                return Outcome(ref.path, "waiting", node.note)
            if polled.kind == "closed":
                self._observe("stream", path=ref.path, phase="closed", position=source.position)
                return self._swap_out(node, ref, node.acc)
            if polled.kind == "failed":
                self._observe("stream", path=ref.path, phase="failed", position=source.position,
                              detail=polled.detail)
                return self._quiesce(node, ref, "stream failed: " + polled.detail)
            if node.current is None:
                self._observe("stream", path=ref.path, phase="admitted", position=source.position,
                              value=polled.value)
                inst = _instantiate(node.step)
                inst.in_["acc"] = copy.deepcopy(node.acc)
                inst.in_["item"] = copy.deepcopy(polled.value)
                node.current = inst
            cref = Ref(type=node.type.s, env=inner, path=f"{ref.path}/step/{node.at}", holder=node, attr="current")
            out = self.trigger(cref, None)
            if out.kind != "done":
                return self._quiesce(node, ref, f"step {source.position} {out.kind}: {out.detail}")
            node.acc, node.current, node.at = node.current, None, node.at + 1
            source.ack()
            self._observe("stream", path=ref.path, phase="consumed", position=source.position)

    # -- Iterate
    def _run_iterate(self, node: IterateNode, ref: Ref, inner: TypeEnv) -> Outcome:
        node.status = RUNNING
        if node.state is MISSING:
            node.state, node.iteration = copy.deepcopy(node.init), 0
            node.recent, node.seen_hashes = [copy.deepcopy(node.init)], [_hash(node.init)]
        while node.iteration < node.max:
            if node.current is None:
                inst = _instantiate(node.step)
                inst.in_[node.state_name] = copy.deepcopy(node.state)
                node.current = inst
            cref = Ref(type=node.type.s, env=inner, path=f"{ref.path}/step/{node.iteration}", holder=node, attr="current")
            out = self.trigger(cref, None)
            if out.kind != "done":
                return self._quiesce(node, ref, f"step {node.iteration} {out.kind}: {out.detail}")
            node.state, node.current = node.current, None
            node.iteration += 1
            h = _hash(node.state)
            if h in node.seen_hashes:
                return self._quiesce(node, ref, "degenerate: the state repeated an earlier state")
            node.seen_hashes.append(h)
            node.recent = (node.recent + [copy.deepcopy(node.state)])[-3:]
            chk = _instantiate(node.check)
            if node.check_name:                      # check(state) -> Bool, a function of the code base
                chk.in_[node.check_name] = copy.deepcopy(node.state)
            else:
                chk.in_["recent"], chk.in_["iteration"] = copy.deepcopy(node.recent), node.iteration
            box = _Box(chk)
            kref = Ref(type=chk.type.returns, env=inner, path=f"{ref.path}/check/{node.iteration}", holder=box, attr="value")
            out = self.trigger(kref, None)
            if out.kind != "done":
                return self._quiesce(node, ref, f"check {out.kind}: {out.detail}")
            verdict = box.value
            if node.check_name:
                verdict = {"verdict": "done" if verdict is True else "continue", "reason": ""}
            if verdict["verdict"] == "done":
                return self._swap_out(node, ref, node.state)
            if verdict["verdict"] == "degenerate":
                return self._quiesce(node, ref, f"degenerate: {verdict['reason']}")
        return self._quiesce(node, ref, f"max iterations reached ({node.max})")

    # ------------------------------------------------------------------ effects
    def _fx(self, lam: Lambda):
        def call(cap, fn, args):
            name = f"{cap}.{fn}"
            if name not in lam.effects:
                raise js.EffectError("effect-undeclared")
            if name not in self.capabilities:
                raise js.EffectError("effect-unavailable")
            entry = {"seq": len(lam.journal) + 1, "capability": name, "function": fn,
                     "args_preview": json.dumps(args)[:80], "status": "pending"}
            lam.journal.append(entry)
            call_id = self.invocations[-1].call_id if self.invocations else None
            self._observe("effect", phase="requested", call_id=call_id,
                          capability=name, sequence=entry["seq"], args=args)
            try:
                out = self.capabilities[name](args)
                entry["status"] = "ok"
                self._observe("effect", phase="completed", call_id=call_id,
                              capability=name, sequence=entry["seq"], result=out)
                return out
            except Exception:
                entry["status"] = "error"
                self._observe("effect", phase="failed", call_id=call_id,
                              capability=name, sequence=entry["seq"])
                raise

        return call


class _Box:
    def __init__(self, value):
        self.value = value


def _instantiate(template: Lambda) -> Lambda:
    inst = copy.deepcopy(template)
    inst.status, inst.note, inst.steps, inst.attempts = UNREDUCED, "", 0, 0
    return inst


def _hash(x) -> str:
    return hashlib.sha256(json.dumps(dump(x), sort_keys=True, default=str).encode()).hexdigest()


def _fn_of(node) -> str:
    lam = node if isinstance(node, Lambda) else getattr(node, "fn", None) or getattr(node, "step", None)
    return lam.fn_name if isinstance(lam, Lambda) else ""


def _known(env, name: str) -> bool:
    try:
        env.check_names(parse_type(name))
        return True
    except TypeSyntaxError:
        return False


def _split2(inner: str):
    """Split `A, B` at the top-level comma."""
    depth = 0
    for i, ch in enumerate(inner):
        depth += ch in "<{([" ; depth -= ch in ">})]"
        if ch == "," and depth == 0:
            return inner[:i].strip(), inner[i + 1:].strip()
    raise reject("type", "type-mismatch", "two type arguments")


def _one_line(value) -> str:
    if isinstance(value, list):
        return f"{len(value)} items"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k}: {scalar(v) if not isinstance(v, (list, dict)) else '…'}"
                                for k, v in list(value.items())[:4]) + " }"
    return scalar(value)


class Session:
    """One episode on one natural-language lambda (SPEC 6)."""

    def __init__(self, runtime: Runtime, lam: Lambda, outer_env: TypeEnv, *, cold: bool = False):
        self.rt, self.lam, self.outer_env = runtime, lam, outer_env
        self.env = lam.env(outer_env)
        self.cold = cold
        self.completed = False
        self.actions = 0
        self.tool_calls = 0

    # -- observations
    def observation(self) -> str:
        holes, _ = self._draft_problems()
        return opening(self.lam, self.outer_env, holes=len(holes), cold=self.cold)

    def _draft_problems(self):
        if self.lam.ret is MISSING:
            return [], []
        holes, pend = problems(self.lam.ret, self.lam.type.returns, self.env, "return")
        for q in pend:  # unbound parts of pending nodes in the draft are holes too
            ref = resolve(self.lam, self.outer_env, parse_path(q))
            holes = holes + self._unbound_deep(ref)
        return holes, pend

    def _unbound_deep(self, ref: Ref):
        node = ref.get()
        out = list(unbound_parts(node, ref.env, ref.path))
        inner = node.env(ref.env)
        if isinstance(node, Lambda):
            in_ref = Ref(type=node.type.params, env=inner, path=ref.path + "/args", holder=node, attr="in_")
            for sub in pending_refs_under(in_ref):
                out += self._unbound_deep(sub)
        return out

    def _summary(self) -> str:
        holes, _ = self._draft_problems()
        return f"problems: 0 blocking · {len(holes)} holes"

    # ------------------------------------------------------------------ tool surface (natlang/surface.py)
    def apply(self, name: str, args: dict) -> Result:
        """Apply one native tool call. Same bookkeeping as `act`, plus a hint on failure."""
        if self.completed:
            return Result("error", "the task has already finished")
        if ((self.rt.options.max_actions is not None and self.actions >= self.rt.options.max_actions) or
                (self.rt.options.max_tool_calls is not None and
                 self.tool_calls >= self.rt.options.max_tool_calls)):
            return Result("budget", "action or tool-call budget exhausted")
        self.tool_calls += 1
        if name != "mark_done":                 # bookkeeping does not spend the budget of work
            self.actions += 1
            self.lam.steps += 1
        args = dict(args) if isinstance(args, dict) else args
        submitted = copy.deepcopy(args)
        op = getattr(self, "_op_" + name, None)
        try:
            if op is None:
                raise reject(name, "bad-action", "a known tool")
            done = args.pop("done", None) if name in ("write", "call") and isinstance(args, dict) else None
            if done is not None:                # validate the en-passant mark before the work, apply it after success
                rng_ = done if isinstance(done, list) else [done]
                if not (1 <= len(rng_) <= 2) or not all(isinstance(v, int) and not isinstance(v, bool) for v in rng_):
                    raise reject("done", "bad-range", "a line number, or [first, last]")
                self._validate_mark({"start": rng_[0], "end": rng_[-1]})
            result = op(args)
            if done is not None and result.kind in ("ok", "done"):
                marked = self._op_mark_done({"start": rng_[0], "end": rng_[-1]})
                result.text = result.text.rstrip() + "\n" + marked.text.split("\n", 1)[1]
        except Reject as e:
            result = Result("rejected", "rejected\n" + "\n".join(map(str, e.diags)) + _hint(e.diags), e.diags)
        except Refuse as e:
            result = Result("refused", "refused\n" + "\n".join(map(str, e.diags)) + _hint(e.diags), e.diags)
        except (js.JsError, ExecutionError) as e:
            result = Result("error", f"error: {e}")
        except (KeyError, TypeError, AttributeError) as e:
            result = Result("rejected", f"rejected\n{name}: bad arguments ({e})")
        if name in ("write", "call", "edit") and result.kind in ("ok", "done", "quiesced"):
            result.text = result.text.rstrip() + "\n" + self._progress()      # where the program stands, at no extra turn
        self.rt.trace.append({"lambda": id(self.lam), "n": self.actions, "action": f"{name} {json.dumps(args, default=str)}",
                              "kind": result.kind, "result": result.text})
        self.rt._observe("action", call_id=getattr(getattr(self, "invocation", None), "call_id", None),
                         surface="tools-v2", name=name, arguments=submitted,
                         outcome=result.kind, result_text=result.text,
                         diagnostics=result.codes)
        self.rt._observe_state("after-action")
        return result

    def _progress(self) -> str:
        """One or two lines after every change: which locals exist, and what `return` still lacks. A long program is
        followed by comparing this with its text; the harness keeps the memory, the model keeps no count."""
        def size(v):
            if is_pending(v):
                return "not finished"
            if isinstance(v, list):
                return f"{len(v)} items"
            if isinstance(v, str):
                return f"{len(v.split())} words"
            return json.dumps(v, ensure_ascii=False, default=str)[:40] if not isinstance(v, dict) else "record"
        lam = self.lam
        locals_ = ", ".join(f"{n} ({size(lam.let[n])})" for n in lam.let_types if n in lam.let) or "none"
        rt = self.env.resolve(lam.type.returns)
        if lam.ret is MISSING:
            ret = "not written yet"
        elif isinstance(rt, Record) and isinstance(lam.ret, dict):
            missing = [n for n, _, opt in rt.fields if not opt and lam.ret.get(n, MISSING) is MISSING]
            ret = "complete" if not missing else "has " + (", ".join(k for k in lam.ret) or "nothing") + "; still missing " + ", ".join(missing)
        else:
            ret = "not finished" if is_pending(lam.ret) else "written"
        return f"locals: {locals_}\nreturn: {ret}"

    def _op_read(self, args):
        path = args["path"]
        if path == "codebase" or path.startswith("codebase/"):      # the code base is read-only text
            from .codebase import listing
            name = path[9:]
            if not name:
                return Result("ok", listing(self.lam.codebase) or "(no functions)")
            fn = self.lam.codebase.get(name)
            if fn is None:
                raise reject(path, "no-such-path", "one of: " + ", ".join(self.lam.codebase))
            return Result("ok", f"{fn.signature}\n{fn.description}\n\n{fn.body}")
        args = {**args, **{k: args[a] for a, k in (("start", "from"), ("end", "to")) if a in args}}
        if "from" not in args and "to" not in args:          # a read shows the whole value, not a preview
            p, ref = self.resolve(path)
            v = ref.get()
            if not p.meta and v is MISSING:
                return Result("ok", f"{path}: not supplied (missing value; not empty text)")
            if not p.meta and not is_pending(v) and v is not MISSING:
                text = v if isinstance(v, str) else json.dumps(dump(v), ensure_ascii=False, indent=1)
                if len(text) <= 6000:
                    self.reads_done = getattr(self, "reads_done", set()) | {path}
                    return Result("ok", text, value=v)
                raise reject(path, "too-large", "a range: pass `from` and `to`", f"{len(text)} characters")
        if "from" in args or "to" in args:
            path += f"[{int(args.get('from', args.get('to')))}..{int(args.get('to', args.get('from')))}]"
        p, ref = self.resolve(path)
        if p.rng and not p.meta:                              # a range is read in full too, never previewed
            part, _ = _slice(ref.get(), p, ref)
            if isinstance(part, list):
                lo = p.rng[0]
                text = "\n".join(f"{lo + i}: {x if isinstance(x, str) else json.dumps(dump(x), ensure_ascii=False)}"
                                 for i, x in enumerate(part))
            else:
                text = part
            if len(text) > 6000:
                raise reject(path, "too-large", "a smaller range", f"{len(text)} characters")
            return Result("ok", text, value=part)
        r = self._do_read(Action("read", path=path))
        self.reads_done = getattr(self, "reads_done", set()) | {args["path"]}
        return r

    def _op_write(self, args):
        """write(path, type, value). A sub-task type (Task, Code, Map, Fold, Iterate) puts a pending
        node at the path; the harness derives every type the model did not have to choose."""
        ty = str(args.get("type") or "")
        if args.get("source") is not None and args.get("value") is None:      # copy, so that data is not re-emitted
            path = str(args.get("path") or "")
            _, sref = self.resolve(str(args["source"]))
            if sref.get() is MISSING or sref.type is None:
                raise reject(str(args["source"]), "no-such-path", "an existing value")
            undo = self._local_type(path, format_type(sref.type), {}) if path.startswith("let/") else None
            try:
                return self._do_copy(Action("copy", path=str(args["source"]), dst=path))
            except (Reject, Refuse):
                if undo:
                    undo()
                raise
        m = re.match(r"^Function<\s*([A-Za-z_]\w*)\s*>$", ty.strip())
        if m:
            return self._copy_function(str(args.get("path") or ""), m.group(1))
        if re.match(r"^(Task|Code|Call|Map|Fold|Iterate|Lambda)<", ty.strip()):
            raise reject("type", "anonymous-lambda", "`call` with a function of the code base: " +
                         (", ".join(self.lam.codebase) or "(none; do the task yourself)"))
        undo = self._local_type(str(args.get("path") or ""), ty, {}) if str(args.get("path") or "").startswith("let/") else None
        try:
            return self._write_plain(args)
        except (Reject, Refuse):
            if undo:
                undo()
            raise

    def _write_plain(self, args):
        value = args["value"]
        try:
            return self._set_value(args["path"], None, value, yaml=False)
        except Reject as first:
            if isinstance(value, str):
                # A single JSON-text layer (e.g. XML tool parameters / excess quotes).
                try:
                    parsed = json.loads(value)
                except (ValueError, TypeError):
                    raise first
                return self._set_value(args["path"], None, parsed, yaml=False)
            _, ref = self.resolve(args["path"], create=True)
            if ref.env.resolve(ref.type) == TEXT and (value is None or isinstance(value, (bool, int, float))):
                # Missing quotes around a scalar; never stringify/unwrap objects.
                try:
                    text = json.dumps(value, allow_nan=False)
                except (ValueError, TypeError):
                    raise first
                return self._set_value(args["path"], None, text, yaml=False)
            raise

    def _op_edit(self, args):
        p, ref = self.resolve(args["path"])
        self._writable(ref)
        text, old, new = ref.get(), args["old"], args.get("new", "")
        if not isinstance(text, str):
            raise reject(ref.path, "type-mismatch", "a text")
        n = text.count(old) if old else 0
        if n != 1:
            raise reject(ref.path, "old-not-unique" if n else "old-not-found",
                         "`old` copied exactly from the text, occurring once", f"{n} occurrences")
        updated = text.replace(old, new, 1)
        if new == "":                                  # deleting a whole line should not leave it blank
            updated = "".join(l for l in updated.splitlines(keepends=True) if l.strip() or not old.strip())
        return self._write_text(ref, updated)

    def _op_copy(self, args):
        return self._do_copy(Action("copy", path=args["from"], dst=args["to"]))

    def _op_delete(self, args):
        return self._do_unset(Action("unset", path=args["path"]))

    def _op_run(self, args):
        paths = args["paths"] if isinstance(args["paths"], list) else [args["paths"]]
        return self._do_reduce(Action("reduce", paths=paths))

    def _op_retry(self, args):
        return self._do_reopen(Action("reopen", path=args["path"], body=args.get("feedback", "")))

    def _op_run_code(self, args):
        engine = args.get("engine")
        if self.rt.engine_selection and engine is None:
            raise reject("engine", "bad-action", "an explicit available engine")
        if engine is None:
            engine = "quickjs-isolated"
        return self._do_eval(Action("eval", body=args["code"]), engine=engine)

    def finish(self) -> bool:
        """The agent replied instead of calling a tool. Complete the lambda if `return` is valid."""
        if self.completed:
            return True
        if self.lam.ret is MISSING:
            return False
        try:
            self._commit_check()
        except Refuse:
            return False
        self.lam.body = ""
        self.completed = True
        return True

    def _validate_mark(self, args):
        from .render import program_lines
        lines = program_lines(self.lam.original_body or self.lam.body)
        start = args.get("start")
        end = args.get("end", start)
        status = "skipped" if args.get("skipped") is True else "done"
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (start, end)) or not (1 <= start <= end <= len(lines)):
            raise reject("start", "bad-range", f"line numbers between 1 and {len(lines)}, start <= end", f"{start}..{end}")
        return lines, start, end, status

    def _op_mark_done(self, args):
        """Mark only after the complete range has been validated."""
        from .render import listing, pending_lines
        lines, start, end, status = self._validate_mark(args)
        for n, _, markable in lines[start - 1:end]:
            if markable:
                self.lam.marks[n] = status
        text = "ok\n" + listing(self.lam.original_body or self.lam.body,
                                self.lam.marks, compact=True, window=3)
        if not pending_lines(self.lam.original_body or self.lam.body, self.lam.marks):
            holes, pending = problems(self.lam.ret, self.lam.type.returns, self.env, "return")
            if not holes and not pending:
                text += "\nAll numbered work is closed and return is complete. Reply normally to finish."
        return Result("ok", text)

    def _op_report_blocker(self, args):
        """The inputs do not determine the result. Ends the episode; the lambda quiesces with the note."""
        missing = str(args.get("missing") or "").strip()
        if len(missing) < 8:
            raise reject("missing", "bad-action", "a sentence saying what is missing")
        self.blocker = missing
        return Result("blocked", "blocked: " + missing)

    def _op_report_error(self, args):
        """Explicit program failure; same quiescence as a blocker, distinct diagnostic."""
        message = str(args.get("message") or "").strip()
        if len(message) < 8:
            raise reject("message", "bad-action", "a sentence explaining the error")
        self.blocker = message
        return Result("blocked", "error: " + message)

    # -- code-base calls (spec/CODEBASES.md 4) ---------------------------------------------------------
    def _local_type(self, path: str, type_text: str, extra_types: dict):
        """Create the local `let/<name>` with this type if `path` names a local that does not exist yet.
        Returns an undo function (or None)."""
        segs = path.split("/")
        if segs[0] != "let" or len(segs) < 2 or segs[1] in self.lam.let_types:
            return None
        if len(segs) != 2 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", segs[1]):
            raise reject(path, "no-such-path", "let/<name> with an identifier as name")
        try:
            t = parse_type(type_text)
        except TypeSyntaxError as e:
            raise reject(path, "type-mismatch", "a type for the new local, written like Bool[], Text[], Num, Text, "
                                                "{ name: Text, count: Num }, or a type name of this task", type_text)
        added = []
        for n, text in (extra_types or {}).items():       # named types the callee brings along
            if n not in self.lam.types and not _known(self.env, n):
                self.lam.types[n] = parse_type(text)
                self.lam.types_src[n] = text
                added.append(n)
        self.env = self.lam.env(self.outer_env)
        try:
            self.env.check_names(t)
        except TypeSyntaxError as e:
            for n in added:
                self.lam.types.pop(n, None); self.lam.types_src.pop(n, None)
            self.env = self.lam.env(self.outer_env)
            raise reject(path, "type-mismatch", "a type written like Bool[], Text[], Num, Text, { name: Text, count: Num }, "
                                                "or a type name of this task", type_text)
        name = segs[1]
        self.lam.let_types[name] = t

        def undo():
            if name not in self.lam.let:
                self.lam.let_types.pop(name, None)
        return undo

    def _function(self, ref_text: str):
        """A function to call: a name of the (immutable) code base, or `let/<name>` holding an edited copy."""
        from .codebase import FunctionDef
        cb = self.lam.codebase
        if ref_text in cb:
            return cb[ref_text]
        if ref_text.startswith("let/"):
            tpl = self.lam.let.get(ref_text[4:])
            if isinstance(tpl, Lambda) and tpl.fn_name:
                base = self.lam.fn_copies.get(ref_text[4:])
                return FunctionDef(name=base.name, kind=tpl.kind, body=tpl.body, args=base.args, returns=base.returns,
                                   types=base.types, description=base.description, effects=base.effects,
                                   codebase=base.codebase, source=ref_text)
        raise reject("function", "no-such-function", "one of: " + (", ".join(cb) or "(this task has no functions)"), ref_text)

    def _copy_function(self, path: str, fn_name: str):
        """write(path="let/x", type="Function<f>"): an editable copy of a code-base function, as a local."""
        fn = self.lam.codebase.get(fn_name)
        if fn is None:
            raise reject("type", "no-such-function", "one of: " + ", ".join(self.lam.codebase), fn_name)
        if not path.startswith("let/"):
            raise reject(path, "not-writable", "a local: let/<name>")
        undo = self._local_type(path, fn.type_text, fn.types)
        try:
            self._op_define({"path": path, "type": fn.type_text, fn.kind: fn.body, "types": fn.types or None,
                             "function": fn.name})
        except (Reject, Refuse):
            if undo:
                undo()
            raise
        self.lam.fn_copies[path[4:]] = fn
        return Result("ok", f"ok   {path} is a copy of {fn.signature}. Edit {path}/{fn.kind}, then call it.")

    def _init(self, init, state_type, types=None):
        """`init` is a path when it names an existing value, otherwise the value itself."""
        if isinstance(init, str):
            try:
                _, r = self.resolve(init)
                if r.get() is not MISSING:
                    return {"init_from": init}
            except Reject:
                pass
        env = self.env.child({n: parse_type(t) for n, t in (types or {}).items()})
        expected = parse_type(state_type)
        try:
            coerce(init, expected, env, yaml=False, path="init")
        except Reject:
            if isinstance(init, str):
                try:
                    parsed = json.loads(init)
                except ValueError:
                    return {"init": init}
                coerce(parsed, expected, env, yaml=False, path="init")
                init = parsed
            elif env.resolve(expected) == TEXT and (init is None or isinstance(init, (bool, int, float))):
                init = json.dumps(init, allow_nan=False)
        return {"init": init}

    def _place_call(self, path: str, fn_name: str, v: dict):
        """Put an instance of a code-base function (plain, or under Map / Fold / Iterate) at `path`.
        Everything is derived from the function's signature; nothing is parsed from instructions."""
        cb = self.lam.codebase
        fn = self._function(fn_name)
        v = v or {}
        inputs = dict(v.get("inputs") or {})
        values = dict(v.get("values") or {})
        overlap = sorted(set(inputs) & set(values))
        if overlap:
            raise reject("values", "bad-call", "parameters bound once", ", ".join(overlap))
        later = []
        names = {n.rstrip("?"): t for n, t in fn.args.items()}
        for n in list(inputs) + list(values) + later:
            if n not in names:
                raise reject(f"inputs/{n}", "unknown-field", fn.signature)
        unbound = [n.rstrip("?") for n in fn.required() if n not in inputs and n not in values and n not in later]
        over, init, until = v.get("over"), v.get("init"), v.get("until")
        lam_spec = {"type": fn.type_text, fn.kind: fn.body, "args_from": inputs, "args": values or None,
                    "types": fn.types or None, "effects": fn.effects or None, "function": fn.name}
        d = {"path": path, "types": fn.types or None}
        if until is not None or v.get("max") is not None:                       # Iterate
            if len(unbound) != 1:
                raise reject("inputs", "bad-call", f"exactly one parameter left for the state: {fn.signature}", ", ".join(unbound))
            st = names[unbound[0]]
            chk = cb.get(str(until))
            if chk is None or len(chk.required()) != 1:
                raise reject("until", "no-such-function", "a function of one parameter returning Bool", str(until))
            if not isinstance(v.get("max"), int) or isinstance(v.get("max"), bool):
                raise reject("max", "type-mismatch", "a whole number: the most rounds allowed")
            if init is None:
                raise reject("init", "type-mismatch", "`init`: the path of the starting state")
            d.update(type=f"Iterate<{st}>", **self._init(init, st, fn.types), max=v["max"], state_name=unbound[0],
                     check_name=chk.required()[0].rstrip("?"), step=lam_spec,
                     check={"type": chk.type_text, chk.kind: chk.body, "types": chk.types or None, "function": chk.name})
            slot_type = st
        elif over is not None and ("acc" in names and "item" in names) and init is not None:   # Fold
            rest = [n for n in unbound if n not in ("acc", "item")]
            if rest:
                raise reject("inputs", "bad-call", f"inputs for: {', '.join(rest)}", fn.signature)
            d.update(type=f"Fold<{names['item']}, {names['acc']}>", over_from=over, **self._init(init, names['acc'], fn.types), step=lam_spec)
            slot_type = names["acc"]
        elif over is not None:                                                  # Map
            if len(unbound) != 1:
                raise reject("inputs", "bad-call", f"exactly one parameter left for the item: {fn.signature}", ", ".join(unbound) or "none")
            d.update(type=f"Map<{names[unbound[0]]}, {fn.returns}>", over_from=over, item_name=unbound[0], fn=lam_spec)
            slot_type = f"({fn.returns})[]"
        else:                                                                   # plain call
            if unbound:
                raise reject("inputs", "bad-call", f"inputs for: {', '.join(unbound)}", fn.signature)
            d.update(lam_spec)
            d["types"] = fn.types or None
            slot_type = fn.returns
        undo = self._local_type(path, slot_type, fn.types)
        try:
            result = self._op_define(d)
        except (Reject, Refuse):
            if undo:
                undo()
            raise
        _, ref = self.resolve(path)
        node = ref.get()
        for lam, f in ((node, fn),) if isinstance(node, Lambda) else \
                ((getattr(node, "fn", None), fn), (getattr(node, "step", None), fn),
                 (getattr(node, "check", None), cb.get(str(until)))):
            if isinstance(lam, Lambda) and f is not None:
                lam.codebase, lam.fn_name = f.codebase, f.name
        return result

    def _op_call(self, args):
        """call(function, to, inputs, ...): place an instance of a function and run it, in one action."""
        path = str(args.get("to") or "")
        fn_ref = str(args.get("function") or "")
        try:                                   # the same function, unfinished, already at `to`: resume it, so
            _, ref = self.resolve(path)        # that only what failed runs again (there is no separate `run`)
            node = ref.get()
        except Reject:
            node = None
        if is_pending(node) and node.status in (UNREDUCED, QUIESCED) and _fn_of(node) == self._function(fn_ref).name \
                and not any(args.get(k) is not None for k in ("inputs", "values", "over", "init", "until", "max")):
            return self._do_reduce(Action("reduce", paths=[path]))
        v = {k: args[k] for k in ("inputs", "values", "over", "init", "until", "max") if args.get(k) is not None}
        self._place_call(path, str(args.get("function") or ""), v)
        return self._do_reduce(Action("reduce", paths=[path]))

    def _op_define(self, args):
        try:
            stated = parse_type(args["type"])
        except TypeSyntaxError as e:
            raise reject(args.get("path", ""), "type-mismatch", "a type such as Lambda<{ item: Text }, Bool>", str(e))
        path = args["path"]
        copies = []

        def sub(spec, where):
            body = {k: spec[k] for k in ("type", "instructions", "code", "args", "types", "effects", "function") if spec.get(k) is not None}
            copies.extend((src, f"{where}/args/{n}") for n, src in (spec.get("args_from") or {}).items())
            return {"$lambda": body}

        kind = type(stated).__name__
        keys = {"LambdaT": ("instructions", "code", "args", "types", "effects", "function"),
                "MapT": ("types", "item_name"), "FoldT": ("init", "types"),
                "IterateT": ("init", "max", "types", "state_name", "check_name")}.get(kind)
        if keys is None:
            raise reject(path, "type-mismatch", "a Lambda, Map, Fold or Iterate type", args["type"])
        body = {k: args[k] for k in keys if args.get(k) is not None}
        for part in {"MapT": ("fn",), "FoldT": ("step",), "IterateT": ("step", "check")}.get(kind, ()):
            if args.get(part):
                body[part] = sub(args[part], f"{path}/{part}")
        copies.extend((src, f"{path}/args/{n}") for n, src in (args.get("args_from") or {}).items())
        if args.get("over_from"):
            copies.append((args["over_from"], f"{path}/over"))
        if args.get("init_from"):
            copies.append((args["init_from"], f"{path}/init"))

        # A call may update its own destination, as in state = step(state).
        # Capture every source before replacing the destination tree.
        snapshots = [(self._snapshot_copy(src), dst) for src, dst in copies]

        _, ref = self.resolve(path, create=True)
        before = ref.get()
        result = self._set_value(path, stated, body, yaml=False)
        try:
            for snapshot, dst in snapshots:
                self._apply_copy_snapshot(snapshot, dst)
        except (Reject, Refuse):
            ref.set(before) if before is not MISSING else ref.delete()   # a define is all or nothing
            raise
        return Result("ok", "ok   " + self._summary())

    def resolve(self, text: str, *, create=False) -> tuple:
        p = parse_path(text)
        return p, resolve(self.lam, self.outer_env, p, create=create)

    def _writable(self, ref: Ref):
        if ref.deny:
            raise reject(ref.path, ref.deny)

    # -- read
    def _do_read(self, a: Action) -> Result:
        p, ref = self.resolve(a.path)
        if p.meta:
            return Result("ok", self._meta(p, ref))
        value = ref.get()
        if p.rng:
            value, _ = _slice(value, p, ref)
        lines = render(value, ref.type, ref.env, 0, 0)
        return Result("ok", "\n".join(lines), value=value)

    def _meta(self, p: Path, ref: Ref) -> str:
        v = ref.get()
        if p.meta == "status":
            return v.status if is_pending(v) else "done"
        if p.meta == "note":
            return v.note if is_pending(v) else ""
        if p.meta == "effects":
            lam = v if isinstance(v, Lambda) else self.lam
            return yaml.safe_dump(lam.journal, sort_keys=False) if lam.journal else "(none)"
        if p.meta == "problems":
            if is_pending(v) or ref.type is None:
                return "(not a value)"
            holes, pend = problems(v, ref.type, ref.env, ref.path)
            return "\n".join([str(h) for h in holes] + [f"{q}: pending" for q in pend]) or "no problems"
        if p.meta == "origin":
            o = self.rt.origins.get(ref.slot_key())
            return yaml.safe_dump(dump(o), sort_keys=False) if o is not None else "(no lambda origin)"
        return "(not recorded)"

    # -- edit
    def _do_edit(self, a: Action) -> Result:
        p, ref = self.resolve(a.path)
        self._writable(ref)
        old = ref.get()
        if not isinstance(old, str):
            raise reject(ref.path, "type-mismatch", "a Text node", format_type(ref.type) if ref.type else "")
        lines = old.splitlines(keepends=True)
        body = a.body
        if body and not body.endswith("\n"):
            body += "\n"
        if p.rng:
            lo, hi = p.rng
            if lo < 1 or hi > len(lines):
                raise reject(p.text, "bad-range", f"lines 1..{len(lines)}")
            new = "".join(lines[: lo - 1]) + body + "".join(lines[hi:])
        else:
            new = body
        return self._write_text(ref, new)

    def _write_text(self, ref: Ref, new: str) -> Result:
        is_own_body = ref.holder is self.lam and ref.attr == "body"
        if is_own_body and new.strip() == "":
            self._commit_check()
            ref.set("")
            self.completed = True
            return Result("completed", "completed", value=self.lam.ret)
        ref.set(new)
        return Result("ok", "ok   " + self._summary())

    def _commit_check(self):
        holes, pend = problems(self.lam.ret, self.lam.type.returns, self.env, "return")
        diags = [Diagnostic(h.path, "commit-holes", BLOCKS, h.expected) for h in holes]
        diags += [Diagnostic(q, "commit-pending", BLOCKS) for q in pend]
        if diags:
            raise Refuse(*diags)

    # -- set / unset
    def _do_set(self, a: Action) -> Result:
        try:
            stated = parse_type(a.type_text)
        except TypeSyntaxError as e:
            raise reject(a.path, "type-mismatch", "a type", str(e))
        body = a.body[:-1] if a.body.endswith("\n") else a.body
        return self._set_value(a.path, stated, body, yaml=True)

    def _set_value(self, path_text: str, stated, raw, *, yaml: bool) -> Result:
        """Create or replace the node at a path. `stated` None means: the slot's declared type."""
        p, ref = self.resolve(path_text, create=True)
        self._writable(ref)
        if p.rng or p.meta:
            raise reject(path_text, "not-writable")
        if stated is None:
            if ref.type is None:
                raise reject(path_text, "not-writable")
            stated = ref.type
        try:
            ref.env.check_names(stated)
        except TypeSyntaxError as e:
            raise reject(path_text, "type-mismatch", "declared type names", str(e))
        self._check_fit(stated, ref)
        rs = ref.env.resolve(stated)
        if isinstance(rs, PENDING_TYPES):
            body = _load_yaml(raw, ref.path) if yaml else raw
            value = build_pending(_WRAPPER_FOR[type(rs)], body or {}, ref.env, yaml=yaml, path=ref.path,
                                  header_type=rs)
        elif yaml and rs == TEXT:
            value = raw
        else:
            value = coerce(_load_yaml(raw, ref.path) if yaml else raw, stated, ref.env, yaml=yaml, path=ref.path)
        self._check_effects(value, ref.path)
        if ref.holder is self.lam and ref.attr == "body":
            return self._write_text(ref, value + ("\n" if value and not value.endswith("\n") else ""))
        self._set_checked(ref, value)
        self._collapse()
        return Result("ok", "ok   " + self._summary())

    def _check_fit(self, stated, ref: Ref):
        if ref.type is None:
            return
        want = ref.env.resolve(ref.type)
        got = ref.env.resolve(stated)
        ok = (body_lambda_fits(got, want, ref.env)
              if isinstance(want, LambdaT) and isinstance(got, LambdaT)
              else fits(stated, ref.type, ref.env))
        if not ok:
            raise reject(ref.path, "type-does-not-fit-slot", format_type(ref.type), format_type(stated))

    def _check_effects(self, value, path):
        allowed = set(self.lam.effects)

        def walk(x, where):
            if isinstance(x, Lambda):
                extra = set(x.effects) - allowed
                if extra:
                    raise reject(where, "effect-wider-than-parent", ", ".join(sorted(allowed)) or "none",
                                 ", ".join(sorted(extra)))
                for k, v in x.in_.items():
                    walk(v, f"{where}/args/{k}")
                walk(x.ret, f"{where}/return")
            elif is_pending(x):
                for part in ("over", "fn", "init", "step", "check"):
                    if hasattr(x, part):
                        walk(getattr(x, part), f"{where}/{part}")
            elif isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{where}/{k}")
            elif isinstance(x, list):
                for i, v in enumerate(x):
                    walk(v, f"{where}/{i}")

        walk(value, path)

    def _do_unset(self, a: Action) -> Result:
        p, ref = self.resolve(a.path)
        self._writable(ref)
        if ref.holder is not None and ref.attr in ("body", "in_"):
            raise reject(ref.path, "not-writable")
        ref.delete()
        return Result("ok", "ok   " + self._summary())

    # -- copy
    def _snapshot_copy(self, path: str):
        sp, src = self.resolve(path)
        value = src.get()
        if value is MISSING:
            raise reject(path, "no-such-path")
        stype = src.type
        if sp.rng:
            value, stype = _slice(value, sp, src)
        if is_pending(value):
            if value.status == RUNNING:
                raise reject(path, "frozen", "a node that is not running")
            value = _instantiate(value) if isinstance(value, Lambda) else copy.deepcopy(value)
            value.status = UNREDUCED
            stype = value.type
        else:
            value = copy.deepcopy(value)
        return value, stype

    def _apply_copy_snapshot(self, snapshot, destination: str) -> Result:
        value, stype = snapshot
        dp, dst = self.resolve(destination, create=True)
        self._writable(dst)
        if dp.rng or dp.meta:
            raise reject(destination, "not-writable")
        if stype is not None:
            self._check_fit(stype, dst)
        self._check_effects(value, dst.path)
        if dst.holder is self.lam and dst.attr == "body":
            return self._write_text(dst, value)
        self._set_checked(dst, value)
        self._collapse()
        return Result("ok", "ok   " + self._summary())

    def _do_copy(self, a: Action) -> Result:
        return self._apply_copy_snapshot(self._snapshot_copy(a.path), a.dst)

    def _set_checked(self, ref: Ref, value):
        """Write the checked value; call recursion is guarded separately."""
        ref.set(value)

    # -- reduce / reopen
    def _do_reduce(self, a: Action) -> Result:
        outcomes = []
        for text in a.paths:
            p, ref = self.resolve(text)
            self._writable(ref)
            outcomes.append(self.rt.trigger(ref, self.lam.effects))
        self._collapse()
        lines = [f"{o.path}: {o.kind}  {o.detail}".rstrip() for o in outcomes]
        kind = outcomes[0].kind if len(outcomes) == 1 else "ok"
        return Result(kind, "\n".join(lines), outcomes=outcomes,
                      value=outcomes[0].value if len(outcomes) == 1 else None)

    def _do_reopen(self, a: Action) -> Result:
        p, ref = self.resolve(a.path)
        self._writable(ref)
        value = ref.get()
        if value is MISSING or is_pending(value) or ref.type is None:
            raise reject(a.path, "no-such-path", "a value")
        origin = self.rt.origins.get(ref.slot_key())
        text = a.body.strip("\n")
        if not isinstance(origin, Lambda) or origin.is_crisp:
            # A value the agent wrote itself can simply be set again. Reopening it would only
            # delegate the agent's own task to a child (observed as a runaway loop).
            raise reject(a.path, "no-origin", "a value that a natural-language lambda produced")
        lam = copy.deepcopy(origin)
        lam.body = (text or origin.original_body or "").rstrip("\n") + "\n"
        lam.ret, lam.status, lam.note = copy.deepcopy(value), UNREDUCED, ""
        self._set_checked(ref, lam)
        return Result("ok", f"ok   {ref.path}: {pending_line(lam)}, draft return prefilled")

    # -- eval
    def _do_eval(self, a: Action, engine: str = "quickjs-isolated") -> Result:
        scope = {"instructions": self.lam.body, "args": self.lam.in_, "return": self.lam.ret,
                 "let": {k: v for k, v in self.lam.let.items() if not is_pending(v)}}
        executor = self.rt._executor(engine)
        self.rt._observe("eval", phase="start", path="eval", mode="expression", engine=engine,
                         code=a.body, effectful=bool(self.lam.effects))
        try:
            try:
                out = portable(executor.run(CrispRequest(a.body, scope, False, "eval",
                                                                bool(self.lam.effects)), self.rt._fx(self.lam)))
            finally:
                self.rt._drain_executor_events(executor)
        except (ExecutionError, Reject) as exc:
            self.rt._observe("eval", phase="failed", path="eval", error=str(exc))
            raise
        self.rt._observe("eval", phase="completed", path="eval", value=out)
        text = json.dumps(out, ensure_ascii=False)
        return Result("ok", text if len(text) <= 400 else text[:400] + f" … ({len(text)} chars)", value=out)

    # -- housekeeping
    def _collapse(self):
        """An expanded Map whose slots are all values is the list (SPEC 4.1)."""
        def visit(ref: Ref):
            for pref in list(pending_refs_under(ref)):
                node = pref.get()
                if isinstance(node, MapNode) and node.slots is not None and \
                        not any(is_pending(s) for s in node.slots):
                    self.rt._swap_out(node, pref, list(node.slots))
                elif isinstance(node, Lambda):
                    inner = node.env(pref.env)
                    visit(Ref(type=node.type.params, env=inner, path=pref.path + "/args", holder=node, attr="in_"))
                    visit(Ref(type=node.type.returns, env=inner, path=pref.path + "/return", holder=node,
                              attr="ret"))

        visit(Ref(type=self.lam.type.returns, env=self.env, path="return", holder=self.lam, attr="ret"))


def _slice(value, p: Path, ref: Ref):
    lo, hi = p.rng
    if isinstance(value, str):
        lines = value.splitlines(keepends=True)
        if lo < 1 or hi > len(lines):
            raise reject(p.text, "bad-range", f"lines 1..{len(lines)}")
        return "".join(lines[lo - 1: hi]), TEXT
    if isinstance(value, list):
        if hi >= len(value):
            raise reject(p.text, "bad-range", f"items 0..{len(value) - 1}")
        return value[lo: hi + 1], ref.type
    raise reject(p.text, "bad-range", "Text or a list")


_HINTS = {
    "commit-holes": "Fill what is still missing with write, then finish your turn.",
    "commit-pending": "A sub-task has not produced its result yet: call run on it, then finish your turn.",
    "not-writable": "args are read-only. Write into return, or into the args of a sub-task you defined.",
    "frozen": "That sub-task is running; its args cannot change now.",
    "type-mismatch": "Pass the value itself with the type shown as expected, not wrapped in another object: for Bool `true`, for Num `42.5`, for Text a string, for a record an object with exactly its fields.",
    "type-does-not-fit-slot": "That slot needs the type shown as expected.",
    "unknown-field": "Use one of the fields listed as expected.",
    "unbound-param": "Give the sub-task its inputs first: copy a value into the path shown, or define it with args_from.",
    "unbound-part": "The sub-task is missing the part shown: for a Map or Fold, pass the list with over_from or copy it to .../over.",
    "no-such-path": "Use a path that appears in the state.",
    "old-not-found": "Copy `old` exactly from the text, including punctuation.",
    "old-not-unique": "Make `old` longer so that it occurs only once.",
    "no-origin": "Only a result that a sub-task produced can be retried. Write the value again instead.",
    "too-large": "Read a part of it with `from` and `to`, or define a Map over it so that each sub-task sees one item.",
    "stuck-dependency": "An input of this sub-task could not be produced: read its note, fix it, run it again.",
}


def _hint(diags) -> str:
    for d in diags:
        if d.code in _HINTS:
            return "\nhint: " + _HINTS[d.code]
    return ""


class _StrictLoader(yaml.BaseLoader):
    """BaseLoader keeps every scalar a string, which is what type-directed parsing needs."""


def _load_yaml(text: str, path: str):
    if text.strip() == "":
        return None
    try:
        events = list(yaml.parse(text, Loader=_StrictLoader))
        for ev in events:
            if getattr(ev, "anchor", None) or isinstance(ev, yaml.AliasEvent):
                raise reject(path, "bad-yaml", "no anchors or aliases")
            tag = getattr(ev, "tag", None)
            if tag and not getattr(ev, "implicit", (True, True))[0] and not str(tag).startswith("tag:yaml.org"):
                raise reject(path, "bad-yaml", "no tags")
        docs = [e for e in events if isinstance(e, yaml.DocumentStartEvent)]
        if len(docs) > 1:
            raise reject(path, "bad-yaml", "a single document")
        return yaml.load(text, Loader=_StrictLoader)
    except yaml.YAMLError as e:
        raise reject(path, "bad-yaml", got=str(e).split("\n")[0])
