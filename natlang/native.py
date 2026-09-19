"""Constrained decoding of tool calls in a model's native call text.

llama.cpp does not enforce tool argument schemas for LFM2.5's Pythonic call format, so the harness
does it: the prompt is rendered by the model's own chat template, and the completion runs under a
grammar built from the turn's tool schemas. The grammar covers the whole assistant turn:

    root ::= "<|tool_call_start|>[" call (", " call)* "]"  |  reply

so the model still chooses freely between calling tools and replying, but a call cannot name a path
that does not exist, put a value of the wrong type into a slot, invent a field, or omit a required
one. A tool may carry "x-natlang-alternatives": argument sets that belong together (a path and the
value type of that path), which JSON Schema alone cannot express at the top level.
"""
from __future__ import annotations

import ast
import copy
import json
from typing import Optional

from .decoder import ChatTurn, LlamaServerDecoder
from .grammar import lit

CALL_OPEN, CALL_CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"

_STATIC = r'''
pystr ::= "\"" ( [^"\\\n] | "\\" [^\n] )* "\"" | "'" ( [^'\\\n] | "\\" [^\n] )* "'"
pystr1 ::= "\"" ( [^"\\\n] | "\\" [^\n] )+ "\"" | "'" ( [^'\\\n] | "\\" [^\n] )+ "'"
pylocal ::= "\"let/" [a-z_] [a-z0-9_]* "\"" | "'let/" [a-z_] [a-z0-9_]* "'"
pyint ::= "-"? [0-9]+
pynum ::= "-"? [0-9]+ ( "." [0-9]+ )?
pybool ::= "True" | "False" | "true" | "false"
pyany ::= pystr | pynum | pybool | "None" | "null" | "[" ( pyany ( ", " pyany )* )? "]" | "{" ( pystr ": " pyany ( ", " pystr ": " pyany )* )? "}"
reply ::= [^<\[] ( [^\x00] )*
'''


def _alt(items):
    items = list(dict.fromkeys(i for i in items if i))
    return items[0] if len(items) == 1 else "( " + " | ".join(items) + " )"


def _pylit(v) -> str:
    """GBNF for one Python literal value."""
    if isinstance(v, str):
        esc = v.replace("\\", "\\\\")
        return _alt([lit('"' + esc.replace('"', '\\"') + '"'), lit("'" + esc.replace("'", "\\'") + "'")])
    if v is True or v is False:
        return _alt([lit(repr(v)), lit(repr(v).lower())])
    if v is None:
        return _alt([lit("None"), lit("null")])
    return lit(repr(v))


class PyGrammar:
    def __init__(self):
        self.rules, self._n, self._cache = {}, 0, {}

    def fresh(self, stem, body):
        self._n += 1
        name = f"{stem}-{self._n}"
        self.rules[name] = body
        return name

    def value(self, schema: dict, depth: int = 0) -> str:
        key = json.dumps(schema, sort_keys=True, default=str)
        if key in self._cache:
            return self._cache[key]
        self._cache[key] = out = self._value(schema, depth)
        return out

    def _value(self, s: dict, depth: int) -> str:
        if not s or depth > 8:
            return "pyany"
        if "const" in s:
            return _pylit(s["const"])
        if "enum" in s:
            return self.fresh("enum", " | ".join(_pylit(v) for v in s["enum"]))
        if "anyOf" in s or "oneOf" in s:
            return self.fresh("any", " | ".join(self.value(x, depth + 1) for x in s.get("anyOf") or s["oneOf"]))
        t = s.get("type")
        if s.get("x-natlang") == "new-local":          # the name of a local that does not exist yet
            return "pylocal"
        if t == "string":
            return "pystr"
        if t == "integer":
            return "pyint"
        if t == "number":
            return "pynum"
        if t == "boolean":
            return "pybool"
        if t == "null":
            return _alt([lit("None"), lit("null")])
        if t == "array":
            item = self.value(s.get("items") or {}, depth + 1)
            least = ' ' if s.get("minItems") else '?'
            body = f'{item} ( ", " {item} )*'
            return self.fresh("arr", f'"[" ( {body} ){least.strip()} "]"' if least == '?' else f'"[" {body} "]"')
        if t == "object":
            props = s.get("properties") or {}
            if not props:
                extra = s.get("additionalProperties")
                item = self.value(extra, depth + 1) if isinstance(extra, dict) else "pyany"
                return self.fresh("dict", f'"{{" ( pystr ": " {item} ( ", " pystr ": " {item} )* )? "}}"')
            return self.fresh("obj", '"{" ' + self._fields(props, s.get("required") or [], depth,
                                                           key=lambda k: _alt([lit(f"'{k}': "), lit(f'"{k}": ')])) + ' "}"')
        return "pyany"

    def _fields(self, props: dict, required: list, depth: int, key) -> str:
        """Required fields in declared order, then any of the optional ones, in order."""
        req = [k for k in props if k in required]
        opt = [k for k in props if k not in required]
        def field(k):
            # an optional field that is present must say something: `phone: ""` is an invented value
            # and `None` is how a model says "does not apply"; the harness reads it as absent.
            if k in opt and props[k].get("type") == "string" and "enum" not in props[k]:
                return f'{key(k)} ( pystr1 | "None" )'
            if k in opt:
                return f'{key(k)} ( {self.value(props[k], depth + 1)} | "None" )'
            return f"{key(k)} {self.value(props[k], depth + 1)}"

        if req:
            out = ' ", " '.join(field(k) for k in req)
            for k in opt:
                out += f' ( ", " {field(k)} )?'
            return out
        if not opt:
            return '""'
        chain = [None] * len(opt)                 # any non-empty in-order subset
        for i in reversed(range(len(opt))):
            rest = _alt([chain[j] for j in range(i + 1, len(opt))]) if i + 1 < len(opt) else ""
            chain[i] = self.fresh("opt", field(opt[i]) + (f' ( ", " {rest} )?' if rest else ""))
        return f"( {_alt(chain)} )?"

    def call(self, tool: dict) -> str:
        fn = tool["function"]
        params = fn.get("parameters") or {}
        alts = params.get("x-natlang-alternatives")
        kw = lambda k: lit(f"{k}=")
        if alts:
            bodies = []
            for a in alts:                             # "x-optional": keys of this alternative that may be left out
                opt = a.get("x-optional") or []
                props = {k: v for k, v in a.items() if k != "x-optional"}
                bodies.append(self._fields(props, [k for k in props if k not in opt], 0, key=kw))
            return self.fresh("call", " | ".join(f'{lit(fn["name"] + "(")} {b} ")"' for b in bodies))
        body = self._fields(params.get("properties") or {}, params.get("required") or [], 0, key=kw)
        return self.fresh("call", f'{lit(fn["name"] + "(")} {body} ")"')

    def text(self, tools: list, allow_reply: bool = True) -> str:
        calls = self.fresh("anycall", " | ".join(self.call(t) for t in tools))
        turn = f'{lit(CALL_OPEN + "[")} {calls} ( ", " {calls} )* "]"'
        root = f"{turn} | reply" if allow_reply else turn
        dyn = "\n".join(f"{k} ::= {v}" for k, v in self.rules.items())
        return f"root ::= {root}\n{dyn}\n{_STATIC.strip()}\n"


def call_grammar(tools: list, allow_reply: bool = True) -> str:
    return PyGrammar().text(tools, allow_reply)



def write_grammar_tools(tools, mode="typed"):
    """Ablation: keep call syntax/paths, but defer literal write typing to runtime.

    Rendering uses the original tools in both modes. No runtime validator is
    weakened, and copied-function/source alternatives keep their constraints.
    """
    if mode not in ("typed", "runtime"):
        raise ValueError("write_constraints must be typed or runtime")
    if mode == "typed":
        return tools
    result = copy.deepcopy(tools)
    for tool in result:
        fn = tool["function"]
        if fn["name"] != "write":
            continue
        params = fn["parameters"]
        for alt in params.get("x-natlang-alternatives", []):
            if "value" in alt:
                alt["type"] = {"type": "string"}
                alt["value"] = {}
    return result


def parse_calls(text: str) -> list:
    """`[write(path="return", value=True), run(paths=['return'])]` -> [(name, args), ...]"""
    text = text.strip()
    for tok in (CALL_OPEN, CALL_CLOSE):
        text = text.replace(tok, "")
    tree = ast.parse(text.strip(), mode="eval").body
    if not isinstance(tree, ast.List):
        raise ValueError("expected a list of calls")
    out = []
    for node in tree.elts:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.args:
            raise ValueError("expected name(keyword=value, ...)")
        out.append((node.func.id, {k.arg: _literal(k.value) for k in node.keywords}))
    return out


_JSON_NAMES = {"true": True, "false": False, "null": None}


def _literal(node):
    """A Python literal in which JSON's true / false / null may also appear: the model's own chat template
    renders nested arguments as JSON, so that is what the model sees in its history and may write."""
    class Fix(ast.NodeTransformer):
        def visit_Name(self, n):
            if n.id in _JSON_NAMES:
                return ast.copy_location(ast.Constant(_JSON_NAMES[n.id]), n)
            return n
    return ast.literal_eval(ast.fix_missing_locations(Fix().visit(node)))


class NativeCallDecoder(LlamaServerDecoder):
    """Tool calling by raw completion under our own grammar, in the model's native call text."""

    def __init__(self, *a, allow_reply: bool = True, write_constraints: str = "typed", probability_log: Optional[list] = None, **kw):
        if write_constraints not in ("typed", "runtime"):
            raise ValueError("write_constraints must be typed or runtime")
        self.write_constraints = write_constraints
        self.probability_log = probability_log
        super().__init__(*a, **kw)
        self.allow_reply = allow_reply
        self.stats = {"turns": 0, "calls": 0, "replies": 0, "p_call_first": []}

    def render(self, messages: list, tools: list) -> str:
        body = json.dumps({"messages": messages, "tools": _strip_private(tools)}).encode()
        import urllib.request
        req = urllib.request.Request(self.base_url + "/apply-template", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.request_timeout()) as resp:
            return json.loads(resp.read())["prompt"]

    def chat(self, messages, tools, *, temperature, seed=None, max_tokens=700, allow_reply: Optional[bool] = None):
        allow = self.allow_reply if allow_reply is None else allow_reply
        prompt = self.render(messages, tools)
        gen = self.generate(prompt, grammar=call_grammar(write_grammar_tools(tools, self.write_constraints), allow), max_tokens=max_tokens,
                            temperature=temperature, seed=seed, stop=[], n_probs=6)
        self.stats["turns"] += 1
        if self.probability_log is not None:
            self.probability_log.append({"text": gen.text, "tokens": gen.token_details})
        if gen.probs:
            # Special tokens may all have empty display text. Empty text is not
            # evidence for CALL_OPEN (it may be EOS); unavailable is not zero.
            self.stats["p_call_first"].append(next((p for tok, p in gen.probs[0] if tok == CALL_OPEN), None))
        text = gen.text.strip()
        if text.startswith("["):
            try:
                calls = parse_calls(text)
            except (ValueError, SyntaxError):
                # It is a call, not a reply: report it as one the harness rejects, so that it is
                # resampled. Filing it as prose would put a non-native call into the history, which
                # the model then imitates. (Python keywords as argument names cause this.)
                calls = [("unparseable_call", {"text": text[:200]})]
            if calls:
                self.stats["calls"] += len(calls)
                raw = [{"id": f"c{self.stats['turns']}_{i}", "type": "function",
                        "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
                return ChatTurn(calls, "", raw, gen.completion_tokens)
        self.stats["replies"] += 1
        return ChatTurn([], text, [], gen.completion_tokens)


def _strip_private(tools):
    out = []
    for t in tools:
        fn = dict(t["function"])
        fn["parameters"] = {k: v for k, v in (fn.get("parameters") or {}).items() if not k.startswith("x-")}
        out.append({**t, "function": fn})
    return out
