"""Constrained decoding of tool calls in a model's native call text.

llama.cpp does not enforce argument schemas for LFM2.5's Pythonic calls, so we
constrain native tool syntax and available paths ourselves. Literal write types
and values are proposals by default: the runtime validates before applying them.
The optional typed mode also constrains those proposals during decoding.

The grammar covers the whole assistant turn, with a reply alternative for normal
execution. A careful-mode review requires exactly one guided review-tool call.
"""

from __future__ import annotations

import ast
import copy
import math
import json
from typing import Optional

from .decoder import ChatTurn, LlamaServerDecoder, _cache_stable_tools

def lit(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return f'"{escaped}"'

CALL_OPEN, CALL_CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"

_STATIC = r'''
pystr ::= "\"" ( [^"\\\n] | "\\" [^\n] )* "\"" | "'" ( [^'\\\n] | "\\" [^\n] )* "'"
pystr1 ::= "\"" ( [^"\\\n] | "\\" [^\n] )+ "\"" | "'" ( [^'\\\n] | "\\" [^\n] )+ "'"
pylocal ::= "\"let/" [a-z_] [a-z0-9_]* "\"" | "'let/" [a-z_] [a-z0-9_]* "'"
pyint ::= "-"? [0-9]+
pynum ::= "-"? [0-9]+ ( "." [0-9]+ )?
pybool ::= "True" | "False" | "true" | "false"
pyany ::= pystr | pynum | pybool | "None" | "null" | "[" ( pyany ( ", " pyany )* )? "]" | "{" ( pystr ": " pyany ( ", " pystr ": " pyany )* )? "}"
reply ::= ( [^<\[] ( [^\x00] )* )?
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
            if "prefixItems" in s:
                prefix = s.get("prefixItems") or []
                lo, hi = int(s.get("minItems", len(prefix))), int(s.get("maxItems", len(prefix)))
                choices = []
                for length in range(lo, hi + 1):
                    fields = ' ", " '.join(self.value(item, depth + 1) for item in prefix[:length])
                    choices.append(f'"[" {fields} "]"' if fields else '"[]"')
                return self.fresh("tuple", " | ".join(choices))
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
                props = {k: v for k, v in a.items() if not k.startswith("x-")}
                bodies.append(self._fields(props, [k for k in props if k not in opt], 0, key=kw))
            return self.fresh("call", " | ".join(f'{lit(fn["name"] + "(")} {b} ")"' for b in bodies))
        body = self._fields(params.get("properties") or {}, params.get("required") or [], 0, key=kw)
        return self.fresh("call", f'{lit(fn["name"] + "(")} {body} ")"')

    def text(self, tools: list, allow_reply: bool = True, single_call: bool = False,
             include_open: bool = True) -> str:
        calls = self.fresh("anycall", " | ".join(self.call(t) for t in tools))
        tail = '"]"' if single_call else f'( ", " {calls} )* "]"'
        turn = f'{lit(CALL_OPEN + "[") if include_open else lit("[")} {calls} {tail}'
        root = f"{turn} | reply" if allow_reply else turn
        dyn = "\n".join(f"{k} ::= {v}" for k, v in self.rules.items())
        return f"root ::= {root}\n{dyn}\n{_STATIC.strip()}\n"


def call_grammar(tools: list, allow_reply: bool = True, *, single_call: bool = False) -> str:
    if not tools:
        if not allow_reply:
            raise ValueError("cannot require a tool call when no tools are available")
        return "root ::= reply\n" + _STATIC.strip() + "\n"
    return PyGrammar().text(tools, allow_reply, single_call)


def call_body_grammar(tools: list, *, single_call: bool = False) -> str:
    """Grammar after the native tool-call marker has already been committed.

    Generating a special marker and ordinary syntax under one grammar lets some
    tokenizers cross that boundary in a token whose visible bytes the grammar
    did not validate. Commit the marker in a separate decoding phase, then
    constrain the complete visible call body here.
    """
    if not tools:
        raise ValueError("cannot generate a tool-call body when no tools are available")
    grammar = PyGrammar().text(tools, allow_reply=False, single_call=single_call,
                               include_open=False)
    return grammar.replace("root ::= ", 'root ::= ws ', 1) + '\nws ::= [ \\t\\r\\n]*\n'



def write_grammar_tools(tools, mode="typed"):
    """Ablation: keep call syntax/paths, but defer literal write typing to runtime.

    Rendering uses the original tools in both modes. No runtime validator is
    weakened, and copied-function/source alternatives keep their constraints.
    In tools-v4 the destination and stated type remain guided while the literal
    itself may still be a fallible proposal.
    """
    if mode not in ("typed", "runtime"):
        raise ValueError("write_constraints must be typed or runtime")
    if mode == "typed":
        return tools
    result = copy.deepcopy(tools)
    for tool in result:
        fn = tool["function"]
        if fn["name"] not in ("write", "write_value"):
            continue
        params = fn["parameters"]
        for alt in params.get("x-natlang-alternatives", []):
            if "value" in alt:
                if fn["name"] == "write":
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



def written_value_confidence(text, details):
    """Align token bytes to AST literal spans; missing/ambiguous data stays unknown.

    This is raw next-token likelihood for write.value or edit.new, not a
    calibrated probability of semantic correctness. Include overlapping tokens
    whole: a token can span the value boundary as well as its first/last bytes.
    """
    raw = text.encode("utf-8")
    reconstructed = b"".join(bytes(t.get("bytes", [])) for t in details)
    if not details or reconstructed != raw:
        return []
    try:
        tree = ast.parse(text, mode="eval").body
    except (SyntaxError, ValueError):
        return []
    if not isinstance(tree, ast.List):
        return []
    lines = raw.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total)
        total += len(line)
    spans, pos = [], 0
    for t in details:
        end = pos + len(t.get("bytes", []))
        if end > pos:
            spans.append((pos, end, t))
        pos = end
    scores = []
    for call in tree.elts:
        score = None
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            key = {"write": "value", "edit": "new"}.get(call.func.id)
            node = next((k.value for k in call.keywords if k.arg == key), None) if key else None
            if node is not None:
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.end_lineno - 1] + node.end_col_offset
                selected = [t for a, b, t in spans if a < end and b > start]
                logps = [t.get("logprob") for t in selected]
                if logps and all(isinstance(p, (int, float)) and math.isfinite(p) and p <= 0 for p in logps):
                    score = {"geometric_mean": math.exp(sum(logps) / len(logps)),
                             "minimum": math.exp(min(logps)), "tokens": len(logps),
                             "field": key, "byte_span": [start, end]}
        scores.append(score)
    return scores


class NativeCallDecoder(LlamaServerDecoder):
    """Tool calling by raw completion under our own grammar, in the model's native call text."""

    def __init__(self, *a, allow_reply: bool = True, write_constraints: str = "runtime", probability_log: Optional[list] = None, **kw):
        if write_constraints not in ("typed", "runtime"):
            raise ValueError("write_constraints must be typed or runtime")
        self.write_constraints = write_constraints
        self.probability_log = probability_log
        super().__init__(*a, **kw)
        self.allow_reply = allow_reply
        self.stats = {"turns": 0, "calls": 0, "replies": 0, "p_call_first": []}

    def render(self, messages: list, tools: list) -> str:
        shown = _strip_private(tools)
        if self.cache_stable_tools:
            shown = _cache_stable_tools(shown)
        body = json.dumps({"messages": messages, "tools": shown}).encode()
        import urllib.request
        req = urllib.request.Request(self.base_url + "/apply-template", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.request_timeout()) as resp:
            return json.loads(resp.read())["prompt"]

    def review(self, messages, tools, **kwargs):
        """Exactly one review-tool call; no free-text affirmative interpretation."""
        return self.chat(messages, tools, allow_reply=False, single_call=True, **kwargs)

    def chat(self, messages, tools, *, temperature, seed=None, max_tokens=None, allow_reply: Optional[bool] = None, single_call: bool = False):
        allow = self.allow_reply if allow_reply is None else allow_reply
        prompt = self.render(messages, tools)
        gen = self.generate(prompt, grammar=call_grammar(write_grammar_tools(tools, self.write_constraints), allow, single_call=single_call), max_tokens=max_tokens,
                            temperature=temperature, seed=seed, stop=[], n_probs=6)
        return self._decode_turn(gen)

    def _decode_turn(self, gen, *, reasoning: Optional[str] = None):
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
                response = None
                if reasoning is not None:
                    response = {"choices": [{"message": {"content": "", "reasoning_content": reasoning,
                                                            "tool_calls": raw}}]}
                return ChatTurn(calls, "", raw, gen.completion_tokens,
                                value_confidence=written_value_confidence(gen.text, gen.token_details),
                                raw_response=response)
        self.stats["replies"] += 1
        response = None
        if reasoning is not None:
            response = {"choices": [{"message": {"content": text, "reasoning_content": reasoning,
                                                    "tool_calls": []}}]}
        return ChatTurn([], text, [], gen.completion_tokens, raw_response=response)


class ReasoningNativeCallDecoder(NativeCallDecoder):
    """Let a reasoning model deliberate freely, then constrain only its action.

    This avoids forcing a reasoning-tuned model to choose a call on its first
    token.  The thinking guard is local to this experimental decoder: ordinary
    inference remains uncapped.  Reaching it is reported as a failed turn rather
    than silently truncating a thought and manufacturing a call.
    """

    def __init__(self, *args, reasoning_tokens: int = 2048, **kwargs):
        if reasoning_tokens < 1:
            raise ValueError("reasoning_tokens must be positive")
        super().__init__(*args, **kwargs)
        self.reasoning_tokens = reasoning_tokens
        self.stats.update({"reasoning_tokens": 0, "reasoning_overflows": 0})

    def chat(self, messages, tools, *, temperature, seed=None, max_tokens=None,
             allow_reply: Optional[bool] = None, single_call: bool = False):
        allow = self.allow_reply if allow_reply is None else allow_reply
        prompt = self.render(messages, tools)
        thought = self.generate(prompt + "<think>", grammar=None, max_tokens=self.reasoning_tokens,
                                temperature=temperature, seed=seed, stop=["</think>"], n_probs=0)
        self.stats["reasoning_tokens"] += thought.completion_tokens or 0
        if thought.stopped != "</think>":
            self.stats["reasoning_overflows"] += 1
            raise ValueError("reasoning did not reach </think> before its diagnostic guard")
        used = thought.completion_tokens or 0
        remaining = None if max_tokens is None else max(1, max_tokens - used)
        prefix = prompt + "<think>" + thought.text + "</think>"
        if allow:
            decision = self.generate(prefix, grammar=None, max_tokens=remaining,
                                     temperature=temperature, seed=seed, stop=[CALL_OPEN], n_probs=6)
            used += decision.completion_tokens or 0
            if decision.stopped != CALL_OPEN:
                decision.completion_tokens = used
                return self._decode_turn(decision, reasoning=thought.text)
            prefix += decision.text + CALL_OPEN
            remaining = None if max_tokens is None else max(1, max_tokens - used)
        else:
            decision = None
            prefix += CALL_OPEN
        action = self.generate(prefix,
                               grammar=call_body_grammar(write_grammar_tools(tools, self.write_constraints),
                                                         single_call=single_call),
                               max_tokens=remaining, temperature=temperature, seed=seed, stop=[], n_probs=6)
        action.completion_tokens = used + (action.completion_tokens or 0)
        if decision is not None and decision.probs:
            action.probs = decision.probs
        return self._decode_turn(action, reasoning=thought.text)


def _strip_private(tools):
    out = []
    for t in tools:
        fn = dict(t["function"])
        fn["parameters"] = {k: v for k, v in (fn.get("parameters") or {}).items() if not k.startswith("x-")}
        out.append({**t, "function": fn})
    return out
