"""The runtime: sessions that apply actions, and triggering of pending nodes."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import yaml

from . import js
from .actions import Action, parse_action
from .diag import BLOCKS, Diagnostic, Refuse, Reject, reject
from .nodes import (DONE, MISSING, QUIESCED, RUNNING, UNREDUCED, FoldNode, IterateNode, Lambda,
                    MapNode, Pending, is_pending)
from .paths import Path, parse_path
from .refs import Ref, pending_refs_under, resolve
from .render import opening, pending_line, render, scalar
from .types import (TEXT, LambdaT, ListT, TypeEnv, TypeSyntaxError, PENDING_TYPES, fits,
                    format_type, is_pending_type, parse_type, FoldT, IterateT, MapT)
from .values import (body_lambda_fits, build_pending, coerce, dump, problems, unbound_parts)

MAX_ACTIONS = 24
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
    kind: str  # ok | rejected | refused | completed | done | quiesced | replaced | error | budget
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


class Runtime:
    def __init__(self, agent_factory: Callable[[Lambda], Any], capabilities: Optional[dict] = None):
        self.agent_factory = agent_factory
        self.emitted: list = []
        self.capabilities = {"out.emit": lambda args: self.emitted.append(args[0] if args else None)}
        self.capabilities.update(capabilities or {})
        self.origins: dict = {}
        self.trace: list = []
        self.episodes_started = 0
        self._ids = 0

    # ------------------------------------------------------------------ running programs
    def run_root(self, root: Pending, env: Optional[TypeEnv] = None):
        """Reduce a root pending node. Returns (outcome, value-or-node)."""
        holder = _Box(root)
        ref = Ref(type=None, env=env or TypeEnv(), path="", holder=holder, attr="value")
        out = self.trigger(ref, acting_effects=None)
        return out, holder.value

    # ------------------------------------------------------------------ trigger
    def trigger(self, ref: Ref, acting_effects) -> Outcome:
        node = ref.get()
        if not is_pending(node):
            raise reject(ref.path, "no-such-path", "a pending node")
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
        return Outcome(ref.path, "done", _one_line(value), value)

    def _quiesce(self, node: Pending, ref: Ref, note: str) -> Outcome:
        node.status, node.note = QUIESCED, note
        return Outcome(ref.path, "quiesced", note)

    # -- crisp lambda
    def _run_crisp(self, node: Lambda, ref: Ref, inner: TypeEnv) -> Outcome:
        node.status = RUNNING
        if node.original_body is None:
            node.original_body = node.body
        scope = {"args": js.to_js(node.in_), "return": js.to_js(node.ret)}
        try:
            raw = js.run(node.body, scope, self._fx(node), body=True, path=ref.path,
                         effectful=bool(node.effects))
            value = coerce(raw, node.type.returns, inner, yaml=False, path=ref.path)
        except js.JsError as e:
            return self._quiesce(node, ref, f"code error: {e}")
        except Reject as e:
            return self._quiesce(node, ref, f"rejected: {e}")
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
        cold = node.status == QUIESCED
        node.status, node.note = RUNNING, ""
        node.attempts += 1
        if node.original_body is None:
            node.original_body = node.body
        self.episodes_started += 1
        session = Session(self, node, ref.env, cold=cold)
        note = self.agent_factory(node).run(session)
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
                inst.in_["item"] = copy.deepcopy(item)
                if wants_index:
                    inst.in_["index"] = i
                node.slots.append(inst)
        node.status = RUNNING
        rt = ref.env.resolve(ref.type) if ref.type is not None else None
        elem = rt.elem if isinstance(rt, ListT) else node.type.b
        stuck = []
        for i, slot in enumerate(node.slots):
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

    # -- Fold
    def _run_fold(self, node: FoldNode, ref: Ref, inner: TypeEnv) -> Outcome:
        node.status = RUNNING
        if node.acc is MISSING:
            node.acc, node.at = copy.deepcopy(node.init), 0
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
            cref = Ref(type=node.type.s, env=inner, path=f"{ref.path}/current", holder=node, attr="current")
            out = self.trigger(cref, None)
            if out.kind != "done":
                return self._quiesce(node, ref, f"step {node.at} {out.kind}: {out.detail}")
            node.acc, node.current, node.at = node.current, None, node.at + 1
        return self._swap_out(node, ref, node.acc)

    # -- Iterate
    def _run_iterate(self, node: IterateNode, ref: Ref, inner: TypeEnv) -> Outcome:
        node.status = RUNNING
        if node.state is MISSING:
            node.state, node.iteration = copy.deepcopy(node.init), 0
            node.recent, node.seen_hashes = [copy.deepcopy(node.init)], [_hash(node.init)]
        while node.iteration < node.max:
            if node.current is None:
                inst = _instantiate(node.step)
                inst.in_["state"] = copy.deepcopy(node.state)
                node.current = inst
            cref = Ref(type=node.type.s, env=inner, path=f"{ref.path}/current", holder=node, attr="current")
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
            chk.in_["recent"], chk.in_["iteration"] = copy.deepcopy(node.recent), node.iteration
            box = _Box(chk)
            kref = Ref(type=chk.type.returns, env=inner, path=f"{ref.path}/check", holder=box, attr="value")
            out = self.trigger(kref, None)
            if out.kind != "done":
                return self._quiesce(node, ref, f"check {out.kind}: {out.detail}")
            verdict = box.value
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
            try:
                out = self.capabilities[name](args)
                entry["status"] = "ok"
                return out
            except Exception:
                entry["status"] = "error"
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

    # -- acting
    def act(self, text: str) -> Result:
        if self.completed:
            return Result("error", "the episode has ended")
        if self.actions >= MAX_ACTIONS:
            return Result("budget", "action budget exhausted")
        self.actions += 1
        self.lam.steps += 1
        try:
            action = parse_action(text)
            result = getattr(self, "_do_" + action.tool)(action)
        except Reject as e:
            result = Result("rejected", "rejected\n" + "\n".join(map(str, e.diags)), e.diags)
        except Refuse as e:
            result = Result("refused", "refused\n" + "\n".join(map(str, e.diags)), e.diags)
        except js.JsError as e:
            result = Result("error", f"error: {e}")
        self.rt.trace.append({"lambda": id(self.lam), "n": self.actions, "action": text,
                              "kind": result.kind, "result": result.text})
        return result

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
        p, ref = self.resolve(a.path, create=True)
        self._writable(ref)
        if p.rng or p.meta:
            raise reject(a.path, "not-writable")
        try:
            ref.env.check_names(stated)
        except TypeSyntaxError as e:
            raise reject(a.path, "type-mismatch", "declared type names", str(e))
        self._check_fit(stated, ref)
        rs = ref.env.resolve(stated)
        body = a.body[:-1] if a.body.endswith("\n") else a.body
        if isinstance(rs, PENDING_TYPES):
            raw = _load_yaml(body, ref.path)
            value = build_pending(_WRAPPER_FOR[type(rs)], raw or {}, ref.env, yaml=True, path=ref.path,
                                  header_type=rs)
        elif rs == TEXT:
            value = body
        else:
            value = coerce(_load_yaml(body, ref.path), stated, ref.env, yaml=True, path=ref.path)
        self._check_effects(value, ref.path)
        if ref.holder is self.lam and ref.attr == "body":
            return self._write_text(ref, value + ("\n" if value and not value.endswith("\n") else ""))
        ref.set(value)
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
    def _do_copy(self, a: Action) -> Result:
        sp, src = self.resolve(a.path)
        value = src.get()
        if value is MISSING:
            raise reject(a.path, "no-such-path")
        stype = src.type
        if sp.rng:
            value, stype = _slice(value, sp, src)
        if is_pending(value):
            if value.status == RUNNING:
                raise reject(a.path, "frozen", "a node that is not running")
            value = _instantiate(value) if isinstance(value, Lambda) else copy.deepcopy(value)
            value.status = UNREDUCED
            stype = value.type
        else:
            value = copy.deepcopy(value)
        dp, dst = self.resolve(a.dst, create=True)
        self._writable(dst)
        if dp.rng or dp.meta:
            raise reject(a.dst, "not-writable")
        if stype is not None:
            self._check_fit(stype, dst)
        self._check_effects(value, dst.path)
        if dst.holder is self.lam and dst.attr == "body":
            return self._write_text(dst, value)
        dst.set(value)
        self._collapse()
        return Result("ok", "ok   " + self._summary())

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
        if isinstance(origin, Lambda) and not origin.is_crisp:
            lam = copy.deepcopy(origin)
            lam.body = (text or origin.original_body or "").rstrip("\n") + "\n"
        else:
            if not text:
                raise reject(a.path, "type-mismatch", "instructions for a value with no lambda origin")
            from .types import Record
            lam = Lambda(type=LambdaT(Record(()), ref.type), kind="instructions", body=text + "\n")
        lam.ret, lam.status, lam.note = copy.deepcopy(value), UNREDUCED, ""
        ref.set(lam)
        return Result("ok", f"ok   {ref.path}: {pending_line(lam)}, draft return prefilled")

    # -- eval
    def _do_eval(self, a: Action) -> Result:
        scope = {"instructions": self.lam.body, "args": js.to_js(self.lam.in_), "return": js.to_js(self.lam.ret)}
        out = js.run(a.body, scope, self.rt._fx(self.lam), body=False, path="eval",
                     effectful=bool(self.lam.effects))
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
