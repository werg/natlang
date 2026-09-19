"""Decoding grammars derived from the tree and its types (SPEC 5.10, TYPES 6.1).

Output is portable GBNF: no token-id terminals, no `{m,n}` repetition, no
special tokens (those belong in the prompt). The same text is accepted by
llama.cpp natively, by XGrammar in vLLM and SGLang, and by llguidance.

Decoding an action is two-phase:
  1. `header_grammar(session)`  -> one header line, constrained by the tree
  2. `body_grammar(session, action)` -> the body, constrained by the header's type
"""
from __future__ import annotations

from typing import Optional

from .nodes import MISSING, QUIESCED, RUNNING, UNREDUCED, FoldNode, IterateNode, Lambda, MapNode, is_pending
from .paths import META, parse_path
from .refs import Ref, resolve
from .render import INLINE, PREVIEW_ITEMS
from .types import (BOOL, NULL, NUM, TEXT, DictT, FoldT, IterateT, LambdaT, ListT, Lit, MapT, Prim,
                    Record, TypeEnv, UnionT, format_type, parse_type, PENDING_TYPES)

MAX_ENUM_PATHS = 8      # lists longer than this get an index pattern instead of literals
MAX_DEPTH = 6           # how deep below the acting lambda paths are enumerated
INDENT = "  "


def lit(s: str) -> str:
    out = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{out}"'


def alt(items) -> str:
    items = [i for i in dict.fromkeys(items) if i]
    return "( " + " | ".join(items) + " )" if len(items) != 1 else items[0]


# --------------------------------------------------------------------------- static rules

_STATIC = r'''
ident ::= [A-Za-z_] [A-Za-z0-9_]*
digits ::= [0-9]+
num ::= "-"? digits ( "." digits )?
line ::= [^\n]*
lines ::= ( [^\n] | "\n" )*
qstr ::= "\"" ( [^"\\\n] | "\\" ["\\nt] )* "\""
tstr ::= "\"" [^"\n]* "\""
tatom ::= TNAMES | tstr | num | trecord | "Dict<" type ">" | "(" type ")" | tpending
tpost ::= tatom ( "[]" )*
type ::= tpost ( " | " tpost )*
tfield ::= ident "?"? ": " type
trecord ::= "{}" | "{ " tfield ( ", " tfield )* " }"
tpending ::= "Lambda<" trecord ", " type ">" | "Map<" type ", " type ">" | "Fold<" type ", " type ">" | "Iterate<" type ">"
gscalar ::= qstr | num | "true" | "false" | "null" | [A-Za-z_] [A-Za-z0-9_ .,;!?'()/-]*
gflow ::= gscalar | "[]" | "[" gflow ( ", " gflow )* "]" | "{}" | "{ " ident ": " gflow ( ", " ident ": " gflow )* " }"
'''


class Builder:
    def __init__(self, names):
        self.rules: dict = {}
        self.names = list(names)
        self._n = 0
        self.cache: dict = {}

    def add(self, name: str, body: str) -> str:
        self.rules[name] = body
        return name

    def fresh(self, stem: str, body: str) -> str:
        self._n += 1
        return self.add(f"{stem}-{self._n}", body)

    def text(self, root_body: str) -> str:
        tnames = alt([lit(n) for n in ["Text", "Num", "Bool", "Null", "Blob", "LoopVerdict"] + self.names])
        static = _STATIC.replace("TNAMES", tnames).strip()
        dyn = "\n".join(f"{k} ::= {v}" for k, v in self.rules.items())
        return f"root ::= {root_body}\n{dyn}\n{static}\n"


# --------------------------------------------------------------------------- path enumeration

class Slot:
    __slots__ = ("path", "ref", "value")

    def __init__(self, path, ref):
        self.path, self.ref, self.value = path, ref, ref.get()


def enumerate_slots(lam: Lambda, outer_env: TypeEnv):
    """Every addressable slot under the acting lambda, as (path text, Ref).

    Long lists contribute a pattern path ending in `/#`, meaning "any index".
    """
    out = []

    def visit(path: str, depth: int):
        try:
            ref = resolve(lam, outer_env, parse_path(path), create=True)
        except Exception:
            return
        out.append(Slot(path, ref))
        if depth >= MAX_DEPTH:
            return
        v = ref.get()
        if is_pending(v):
            for part in _parts(v):
                visit(f"{path}/{part}", depth + 1)
            return
        rt = ref.env.resolve(ref.type) if ref.type is not None else None
        if isinstance(rt, UnionT):
            rt = next((ref.env.resolve(m) for m in rt.members
                       if isinstance(ref.env.resolve(m), (Record, ListT, DictT))), rt)
        if isinstance(rt, Record):
            for n in rt.names:
                visit(f"{path}/{n}", depth + 1)
        elif isinstance(rt, DictT) and isinstance(v, dict):
            for k in list(v)[:MAX_ENUM_PATHS]:
                visit(f"{path}/{k}", depth + 1)
        elif isinstance(rt, ListT):
            n = len(v) if isinstance(v, list) else 0
            for i in range(min(n, MAX_ENUM_PATHS)):
                visit(f"{path}/{i}", depth + 1)

    for part in _parts(lam):
        visit(part, 0)
    return out


def _parts(node):
    if isinstance(node, Lambda):
        return [node.kind, "args", "return"]
    if isinstance(node, MapNode):
        slots = [str(i) for i in range(min(len(node.slots or []), MAX_ENUM_PATHS))]
        return ["over", "fn"] + slots
    if isinstance(node, FoldNode):
        return ["over", "init", "step", "acc"] + (["current"] if node.current is not None else [])
    if isinstance(node, IterateNode):
        return ["init", "step", "check", "max", "state"] + (["current"] if node.current is not None else [])
    return []


def _resolved(slot: Slot):
    return slot.ref.env.resolve(slot.ref.type) if slot.ref.type is not None else None


# --------------------------------------------------------------------------- header grammar

def header_grammar(session) -> str:
    lam, env = session.lam, session.env
    b = Builder(_names_in_scope(env))
    slots = enumerate_slots(lam, session.outer_env)
    existing = [s for s in slots if s.value is not MISSING]
    writable = [s for s in slots if not s.ref.deny]

    def with_index_patterns(items, pred=lambda s: True):
        """Literal paths, plus `prefix/<index>` for long lists."""
        outs = [lit(s.path) for s in items if pred(s)]
        for s in items:
            if isinstance(s.value, list) and len(s.value) > MAX_ENUM_PATHS:
                outs.append(f'{lit(s.path + "/")} digits')
        return outs

    def span(n: int, first: int) -> str:
        """`[a..b]` over positions that exist: first..first+n-1."""
        if n <= 0:
            return ""
        last = first + n - 1
        if n <= 12:
            pairs = [lit(f"[{a}..{z}]") for a in range(first, last + 1) for z in range(a, last + 1)]
            return alt(pairs)
        if n <= 300:
            pos = b.fresh("pos", " | ".join(lit(str(i)) for i in range(first, last + 1)))
            return f'"[" {pos} ".." {pos} "]"'
        return '"[" digits ".." digits "]"'

    def ranged(s, optional=True) -> str:
        v = s.value
        r = span(len(v.splitlines()), 1) if isinstance(v, str) else span(len(v), 0) if isinstance(v, list) else ""
        if not r:
            return lit(s.path)
        return f'{lit(s.path)} ( {r} ){"?" if optional else ""}'

    own_body = lambda s: s.ref.holder is lam and s.ref.attr == "body"

    def hidden(s) -> bool:
        """Does the opening observation show less than the whole value?"""
        v = s.value
        if own_body(s):
            return False
        if is_pending(v):
            return True
        if isinstance(v, str):
            return len(v.rstrip("\n")) > INLINE or "\n" in v.rstrip("\n")
        if isinstance(v, list):
            return len(v) > PREVIEW_ITEMS
        return s.path.count("/") >= 3 and isinstance(v, (dict, list))

    reads = [ranged(s) for s in existing if hidden(s)]
    for s in existing:
        if isinstance(s.value, list) and len(s.value) > MAX_ENUM_PATHS:
            reads.append(f'{lit(s.path + "/")} digits')
    metas = [lit("return@problems")] if lam.ret is not MISSING else []   # nothing to report on an empty draft
    metas += [lit(s.path + "@note") for s in existing if is_pending(s.value) and s.value.status == QUIESCED]
    metas += [lit(s.path + "@origin") for s in existing if not is_pending(s.value) and s.ref.holder is None
              and session.rt is not None and s.ref.slot_key() in session.rt.origins]
    if lam.journal:
        metas.append(lit("return@effects"))
    actions = [f'"read " {alt(reads + metas)}'] if reads + metas else []

    b.add("spath", alt([ranged(s) for s in existing] +
                       [f'{lit(s.path + "/")} digits' for s in existing
                        if isinstance(s.value, list) and len(s.value) > MAX_ENUM_PATHS]))

    texts = [s for s in existing if isinstance(s.value, str) and _resolved(s) == TEXT and not s.ref.deny]
    if texts:
        b.add("tpath", alt([ranged(s) for s in texts]))
        actions.append('"edit " tpath')

    set_alts = []
    for s in writable:
        if s.ref.type is None or (s.ref.holder is not None and s.ref.attr == "in_"):
            continue
        set_alts.append(f'{lit(s.path)} " : " {_types_for_slot(b, s.ref)}')
        rt = _resolved(s)
        if isinstance(rt, ListT):
            elem_ref = Ref(type=rt.elem, env=s.ref.env, path=s.path + "/+")
            set_alts.append(f'{lit(s.path + "/+")} " : " {_types_for_slot(b, elem_ref)}')
    if set_alts:
        b.add("setarg", alt(set_alts))
        actions.append('"set " setarg')

    unsettable = [s for s in writable if s.value is not MISSING and s.ref.holder is None]
    if unsettable:
        b.add("upath", alt([lit(s.path) for s in unsettable]))
        actions.append('"unset " upath')

    dst = [s for s in writable if s.ref.type is not None and not (s.ref.holder is not None and s.ref.attr == "in_")]
    if existing and dst:
        dsts = [lit(s.path) for s in dst]
        dsts += [lit(s.path + "/+") for s in dst if isinstance(_resolved(s), ListT)]
        b.add("dpath", alt(dsts))
        actions.append('"copy " spath " to " dpath')

    reducible = [s for s in existing if is_pending(s.value) and s.value.status in (UNREDUCED, QUIESCED)
                 and not s.ref.deny]
    if reducible:
        b.add("ppath", alt([lit(s.path) for s in reducible]))
        actions.append('"reduce " ppath ( " " ppath )*')

    def produced_by_child(s) -> bool:
        if session.rt is None or is_pending(s.value) or s.ref.deny:
            return False
        origin = session.rt.origins.get(s.ref.slot_key())
        return isinstance(origin, Lambda) and not origin.is_crisp

    reopenable = [s for s in existing if produced_by_child(s)]
    if reopenable:
        b.add("opath", alt([lit(s.path) for s in reopenable]))
        actions.append('"reopen " opath')

    actions += ['"eval"', '"stuck"']
    return b.text(alt(actions) + ' "\\n"')


def _names_in_scope(env: TypeEnv):
    names, e = [], env
    while e is not None:
        names += list(e.names)
        e = e.parent
    return list(dict.fromkeys(names))


def _types_for_slot(b: Builder, ref: Ref) -> str:
    """GBNF expression for the types that may be stated when writing this slot."""
    rt = ref.env.resolve(ref.type)
    if isinstance(rt, LambdaT):  # higher-order slot: a body lambda, extra params allowed
        return f'"Lambda<" trecord ", " {_result_types(ref, rt.returns)} ">"'
    res = _result_types(ref, ref.type)
    alts = [res, f'"Lambda<" trecord ", " {res} ">"', f'"Fold<" type ", " {res} ">"', f'"Iterate<" {res} ">"']
    if isinstance(rt, ListT):
        alts.append(f'"Map<" type ", " {_result_types(ref, rt.elem)} ">"')
    return alt(alts)


def _result_types(ref: Ref, t) -> str:
    """The declared type as written, or a narrowed enum of it."""
    outs = [lit(format_type(t))]
    rt = ref.env.resolve(t)
    members = rt.members if isinstance(rt, UnionT) else ((rt,) if isinstance(rt, Lit) else ())
    if members and all(isinstance(m, Lit) for m in members):
        m = alt([lit(format_type(x)) for x in members])
        outs.append(f'{m} ( " | " {m} )*')
    return alt(outs)


# --------------------------------------------------------------------------- body grammar

def body_grammar(session, action) -> Optional[str]:
    """Grammar for the body of `action` (already parsed from its header). None = no body."""
    tool = action.tool
    if tool in ("read", "unset", "copy", "reduce"):
        return None
    b = Builder(_names_in_scope(session.env))
    if tool in ("eval", "edit", "reopen", "stuck"):
        return b.text("lines")
    stated = parse_type(action.type_text)
    p = parse_path(action.path)
    ref = resolve(session.lam, session.outer_env, p, create=True)
    rs = ref.env.resolve(stated)
    if rs == TEXT:
        return b.text("lines")
    if isinstance(rs, PENDING_TYPES):
        return b.text(_pending_body(b, rs, ref.env, "", header_typed=True))
    return b.text(_value_top(b, stated, ref.env))


def _scalar(b: Builder, t, env) -> Optional[str]:
    rt = env.resolve(t)
    if isinstance(rt, UnionT):
        parts = [_scalar(b, m, env) for m in rt.members]
        return alt(parts) if all(parts) else None
    if isinstance(rt, Lit):
        return lit(str(rt.value)) if isinstance(rt.value, str) else lit(repr(rt.value))
    if isinstance(rt, Prim):
        return {"Text": "qstr", "Blob": "qstr", "Num": "num", "Bool": '( "true" | "false" )',
                "Null": '"null"'}[rt.name]
    return None


def _flow(b: Builder, t, env, depth=0) -> str:
    """JSON-style flow form, used for list items and anything nested below a list."""
    s = _scalar(b, t, env)
    if s:
        return s
    rt = env.resolve(t)
    if depth > 4:
        return "gflow"
    if isinstance(rt, ListT):
        item = _flow(b, rt.elem, env, depth + 1)
        return b.fresh("flist", f'"[]" | "[" {item} ( ", " {item} )* "]"')
    if isinstance(rt, DictT):
        item = _flow(b, rt.elem, env, depth + 1)
        return b.fresh("fdict", f'"{{}}" | "{{ " ident ": " {item} ( ", " ident ": " {item} )* " }}"')
    if isinstance(rt, Record):
        fields = [f'{lit(n + ": ")} {_flow(b, ft, env, depth + 1)}' for n, ft, _ in rt.fields]
        if not fields:
            return '"{}"'
        chain = [None] * len(fields)
        for i in reversed(range(len(fields))):  # any in-order subset, comma separated
            rest = alt([chain[j] for j in range(i + 1, len(fields))]) if i + 1 < len(fields) else ""
            chain[i] = b.fresh("ff", fields[i] + (f' ( ", " {rest} )?' if rest else ""))
        return b.fresh("frec", f'"{{}}" | "{{ " {alt(chain)} " }}"')
    return "gflow"


def _value_top(b: Builder, t, env) -> str:
    """A value as the whole body of a `set`."""
    s = _scalar(b, t, env)
    rt = env.resolve(t)
    if s and not isinstance(rt, Prim) or (isinstance(rt, Prim) and rt.name != "Text"):
        return f'{s} "\\n"?'
    return _block(b, t, env, "", nested=0)


def _block(b: Builder, t, env, ind: str, nested: int) -> str:
    """Block-style YAML for a container at indentation `ind`."""
    rt = env.resolve(t)
    if isinstance(rt, UnionT):
        rt = next((env.resolve(m) for m in rt.members if isinstance(env.resolve(m), (Record, ListT, DictT))), rt)
    if isinstance(rt, Record):
        fields = [_field(b, n, ft, env, ind, nested) for n, ft, _ in rt.fields]
        if not fields:
            return '""'
        chain = [None] * len(fields)          # any in-order, non-empty subset of the fields
        for i in reversed(range(len(fields))):
            rest = alt([chain[j] for j in range(i + 1, len(fields))]) if i + 1 < len(fields) else ""
            chain[i] = b.fresh("bf", f"{fields[i]}" + (f" ( {rest} )?" if rest else ""))
        return b.fresh("brec", alt(chain))
    if isinstance(rt, ListT):
        item = _flow(b, rt.elem, env)
        return b.fresh("blist", f'"[]\\n" | ( {lit(ind + "- ")} {item} "\\n" )+')
    if isinstance(rt, DictT):
        item = _flow(b, rt.elem, env)
        return b.fresh("bdict", f'( {lit(ind)} ident ": " {item} "\\n" )*')
    return f'{_flow(b, t, env)} "\\n"'


def _field(b: Builder, name: str, t, env, ind: str, nested: int) -> str:
    key = lit(f"{ind}{name}:")
    forms = []
    s = _scalar(b, t, env)
    rt = env.resolve(t)
    if s:
        forms.append(f'" " {s} "\\n"')
        if rt == TEXT:
            forms.append(_block_text(ind + INDENT))
    elif isinstance(rt, (Record, ListT, DictT)):
        forms.append(f'" " {_flow(b, t, env)} "\\n"')
        forms.append(f'"\\n" {_block(b, t, env, ind + INDENT, nested)}')
    if nested < 2 and not isinstance(rt, LambdaT):
        forms.append(f'"\\n" {_wrapper(b, env, ind + INDENT, nested + 1)}')
    if isinstance(rt, LambdaT):
        forms = [f'"\\n" {lit(ind + INDENT + "$lambda:\n")} '
                 f'{_pending_body(b, None, env, ind + INDENT * 2, header_typed=False, only="Lambda", nested=nested + 1)}']
    return f"{key} {alt(forms)}"


def _block_text(ind: str) -> str:
    return f'" |\\n" ( {lit(ind)} line "\\n" )+'


def _wrapper(b: Builder, env, ind: str, nested: int) -> str:
    """A nested pending literal: `$lambda:` / `$map:` / ... with its own `type:` line."""
    key = ("wrap", ind, nested)
    if key in b.cache:
        return b.cache[key]
    b.cache[key] = name = f"wrap-{len(ind)}-{nested}"
    outs = []
    for w, kind in (("$lambda", "Lambda"), ("$map", "Map"), ("$fold", "Fold"), ("$iterate", "Iterate")):
        body = _pending_body(b, None, env, ind + INDENT, header_typed=False, only=kind, nested=nested)
        outs.append(f'{lit(ind + w + ":\n")} {body}')
    return b.add(name, " | ".join(outs))


def _pending_body(b: Builder, rs, env, ind: str, *, header_typed: bool, only: str = "", nested: int = 0) -> str:
    """The mapping that describes a pending node. With header_typed, its type is `rs`."""
    kind = only or {LambdaT: "Lambda", MapT: "Map", FoldT: "Fold", IterateT: "Iterate"}[type(rs)]
    key = ("pend", kind, ind, nested)
    if not header_typed:
        if key in b.cache:
            return b.cache[key]
        b.cache[key] = f"pend-{kind.lower()}-{len(ind)}-{nested}"
    parts = []
    if not header_typed:
        parts.append(f'{lit(ind + "type: " + chr(39))} type {lit(chr(39) + chr(10))}')
    parts.append(f'( {lit(ind + "types:\n")} ( {lit(ind + INDENT)} ident ": " {lit(chr(39))} type '
                 f'{lit(chr(39) + chr(10))} )+ )?')

    def generic_args(i):
        entry = f'{lit(i)} ident ":" ( " " gflow "\\n"' + \
                (f' | "\\n" {_wrapper(b, env, i + INDENT, nested + 1)}' if nested < 2 else "") + " )"
        return f"( {entry} )*"

    def part(name, t):
        if t is not None:
            return f"( {_field(b, name, t, env, ind, nested)} )?"
        forms = ['" " gflow "\\n"']
        if nested < 2:
            forms.append(f'"\\n" {_wrapper(b, env, ind + INDENT, nested + 1)}')
        return f'( {lit(ind + name + ":")} {alt(forms)} )?'

    if kind == "Lambda":
        parts.append(f'( {lit(ind + "effects: [")} ident "." ident ( ", " ident "." ident )* "]\\n" )?')
        body_i = f'{lit(ind + "instructions:")} ( " " line "\\n" | {_block_text(ind + INDENT)} )'
        body_c = f'{lit(ind + "code:")} {_block_text(ind + INDENT)}'
        parts.append(f"( {body_i} | {body_c} )")
        if header_typed and rs.params.fields:
            args = _block(b, rs.params, env, ind + INDENT, nested)
            parts.append(f'( {lit(ind + "args:\n")} {args} )?')
        else:
            parts.append(f'( {lit(ind + "args:\n")} {generic_args(ind + INDENT)} )?')
    elif kind == "Map":
        parts += [part("over", ListT(rs.a) if header_typed else None),
                  part("fn", LambdaT(Record((("item", rs.a, False),)), rs.b) if header_typed else None)]
        if not header_typed:
            parts[-1] = f'( {lit(ind + "fn:\n" + ind + INDENT + "$lambda:\n")} ' \
                        f'{_pending_body(b, None, env, ind + INDENT * 2, header_typed=False, only="Lambda", nested=nested + 1)} )?'
    elif kind == "Fold":
        lam_part = lambda n: f'( {lit(ind + n + ":\n" + ind + INDENT + "$lambda:\n")} ' \
                             f'{_pending_body(b, None, env, ind + INDENT * 2, header_typed=False, only="Lambda", nested=nested + 1)} )?'
        parts += [part("over", ListT(rs.a) if header_typed else None),
                  part("init", rs.s if header_typed else None), lam_part("step")]
    else:
        lam_part = lambda n: f'( {lit(ind + n + ":\n" + ind + INDENT + "$lambda:\n")} ' \
                             f'{_pending_body(b, None, env, ind + INDENT * 2, header_typed=False, only="Lambda", nested=nested + 1)} )?'
        parts += [f'( {lit(ind + "max: ")} digits "\\n" )?', part("init", rs.s if header_typed else None),
                  lam_part("step"), lam_part("check")]
    if not header_typed:
        return b.add(b.cache[key], " ".join(parts))
    return b.fresh("pend", " ".join(parts))
