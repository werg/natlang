"""Read-only type analysis over the single natlang TS-style type algebra."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from natlang.codebase import CheckedGraph, FunctionDef, load_function
from natlang.host import load
from natlang.invocation import RunOptions, SeedPolicy
from natlang.runtime import Runtime
from natlang.trace import TraceRecorder
from natlang.types import TypeEnv, TypeSyntaxError, fits, parse_type
from natlang.values import dump


ROOT = Path(__file__).resolve().parent.parent


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _diagnostic(level: str, source: str, message: str) -> dict:
    return {"level": level, "source": source, "message": message}


class TypeStudio:
    def __init__(self, target: dict, context: dict):
        self.target = copy.deepcopy(target)
        self.context = copy.deepcopy(context)
        if not self.target.get("revision"):
            raise ValueError("type analysis needs a source revision")
        self.snapshot_sha256 = _digest([self.target, self.context])

    @classmethod
    def from_graph(cls, graph: CheckedGraph, target_name: str, *,
                   obligations: list[dict] | None = None, witnesses: list[dict] | None = None,
                   required_effects: list[str] | None = None) -> "TypeStudio":
        fn = graph.get(target_name)
        target = {"name": fn.name, "body": fn.body,
                  "parameters": [name.rstrip("?") for name in fn.args], "revision": graph.revision}
        context = {"named_types": dict(fn.types),
                   "signatures": [{"name": other.name, "args": dict(other.args), "returns": other.returns}
                                  for other in graph.definitions.values() if other.name != fn.name],
                   "obligations": obligations or [], "required_effects": required_effects or [],
                   "witnesses": witnesses or []}
        return cls(target, context)

    @classmethod
    def from_file(cls, path: Path, target_name: str, **evidence) -> "TypeStudio":
        root = load_function(path)
        found: dict[str, FunctionDef] = {}
        def visit(fn: FunctionDef) -> None:
            if fn.name in found and found[fn.name] is not fn:
                raise ValueError(f"ambiguous function name {fn.name}; use an in-memory graph")
            if fn.name in found:
                return
            found[fn.name] = fn
            for child in fn.codebase.values():
                visit(child)
        visit(root)
        if target_name not in found:
            raise ValueError(f"unknown function {target_name}")
        revision = _digest({name: fn.to_inline() for name, fn in sorted(found.items())})
        return cls.from_graph(CheckedGraph(root, found, revision), target_name, **evidence)

    def _unchanged(self, target: dict | None, context: dict) -> None:
        if context != self.context or (target is not None and target != self.target):
            raise ValueError("type evidence differs from the frozen source snapshot")

    def _env(self) -> TypeEnv:
        names = {name: parse_type(text) for name, text in self.context["named_types"].items()}
        env = TypeEnv(names)
        for declared in names.values():
            env.check_names(declared)
        return env

    def _type(self, text: str, env: TypeEnv):
        parsed = parse_type(text)
        env.check_names(parsed)
        return parsed

    def check_candidate(self, target: dict, context: dict, candidate: dict) -> dict:
        self._unchanged(target, context)
        diagnostics = []
        checked = 0
        try:
            env = self._env()
            declared = set(candidate["args"])
            parameters = set(self.target["parameters"])
            if declared != parameters:
                diagnostics.append(_diagnostic("exact", "signature",
                                               "parameter names differ from the source definition"))
            args = {name: self._type(text, env) for name, text in candidate["args"].items()}
            returns = self._type(candidate["returns"], env)
            parseable = True
        except (KeyError, TypeSyntaxError, ValueError, TypeError) as exc:
            diagnostics.append(_diagnostic("exact", "signature", f"invalid type syntax or name: {exc}"))
            return {"parseable": False, "obligations_ok": False, "checked": 0,
                    "diagnostics": diagnostics}
        required = set(self.context["required_effects"])
        effects = candidate.get("effects", [])
        if len(effects) != len(set(effects)) or not required.issubset(effects):
            diagnostics.append(_diagnostic("exact", "effects", "required effects are missing or duplicated"))
        for obligation in self.context["obligations"]:
            checked += 1
            try:
                evidence_type = self._type(obligation["type"], env)
                if obligation["kind"] == "argument":
                    parameter = obligation["parameter"]
                    okay = parameter in args and fits(evidence_type, args[parameter], env)
                    description = f"caller value does not fit parameter {parameter}"
                elif obligation["kind"] == "return":
                    okay = fits(returns, evidence_type, env)
                    description = "proposed return does not fit consumer expectation"
                else:
                    raise ValueError("unknown obligation kind")
                if not okay:
                    diagnostics.append(_diagnostic("exact", obligation["source"], description))
            except (TypeSyntaxError, ValueError) as exc:
                diagnostics.append(_diagnostic("unknown", obligation["source"],
                                               f"invalid supplied obligation: {exc}"))
        exact_failure = any(d["level"] == "exact" for d in diagnostics)
        unknown_evidence = any(d["level"] == "unknown" for d in diagnostics)
        return {"parseable": parseable, "obligations_ok": not exact_failure and not unknown_evidence,
                "checked": checked, "diagnostics": diagnostics}

    def check_calls(self, context: dict, claims: list[dict]) -> list[dict]:
        self._unchanged(None, context)
        if len(claims) > 8:
            return [_diagnostic("unknown", "call search", "more than eight call claims")]
        try:
            env = self._env()
        except TypeSyntaxError as exc:
            return [_diagnostic("unknown", "type context", str(exc))]
        signatures = {item["name"]: item for item in self.context["signatures"]}
        witnesses = {item["id"]: item for item in self.context["witnesses"]}
        diagnostics = []
        for claim in claims:
            signature = signatures.get(claim["callee"])
            witness_id = claim["evidence_id"]
            witness = witnesses.get(witness_id) if witness_id else None
            if witness_id and (not witness or witness["callee"] != claim["callee"] or
                               witness["arg_types"] != claim["arg_types"]):
                diagnostics.append(_diagnostic("unknown", witness_id, "call claim does not match a frozen witness"))
                continue
            level = "exact" if witness else "hypothesis"
            source = witness["source"] if witness else self.target["revision"]
            if signature is None:
                diagnostics.append(_diagnostic("unknown", source, f"unknown callee {claim['callee']}"))
                continue
            expected_names = {name.rstrip("?") for name in signature["args"]}
            for name, expected in signature["args"].items():
                try:
                    optional = name.endswith("?")
                    actual = claim["arg_types"].get(name.rstrip("?"))
                    okay = (actual is None and optional) or (actual is not None and
                            fits(self._type(actual, env), self._type(expected, env), env))
                    if not okay:
                        diagnostics.append(_diagnostic(level, source,
                                                       f"argument {name} does not fit {claim['callee']}"))
                except TypeSyntaxError as exc:
                    diagnostics.append(_diagnostic("unknown", source, f"unparseable call type: {exc}"))
            extras = set(claim["arg_types"]) - expected_names
            if extras:
                diagnostics.append(_diagnostic(level, source,
                                               f"unexpected arguments for {claim['callee']}: {', '.join(sorted(extras))}"))
        return diagnostics

    def capabilities(self) -> dict:
        return {"types.check": lambda args: self.check_candidate(*args),
                "types.calls": lambda args: self.check_calls(*args)}

    def run(self, operation: str, *, agent_factory, model_id: str, root_seed: int,
            trace_path: Path | None = None) -> dict:
        if operation not in ("infer", "check"):
            raise ValueError("operation must be infer or check")
        entry = ROOT / "codebases/type_studio" / f"{operation}.nl"
        options = RunOptions(seed=SeedPolicy("derived", root_seed))
        recorder = TraceRecorder({"run_id": options.run_id, "source_sha256": self.snapshot_sha256,
                                  "model": model_id, "seed_policy": vars(options.seed),
                                  "operation": operation}, trace_path)
        try:
            outcome, value = Runtime(agent_factory, options=options, capabilities=self.capabilities(),
                                     trace_sink=recorder).run_root(load(entry, {"target": self.target,
                                                                               "context": self.context}))
        finally:
            recorder.close()
        return {"operation": operation, "source_revision": self.target["revision"],
                "snapshot_sha256": self.snapshot_sha256, "outcome": outcome.kind,
                "detail": outcome.detail, "value": dump(value) if outcome.kind == "done" else None,
                "trace_sha256": _digest(recorder.events),
                "trace_path": str(trace_path) if trace_path else None}
