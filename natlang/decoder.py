"""Generation backends behind one interface (see PLAN: generation control).

The harness needs: a different grammar on every call, raw prompts with special
tokens, continuation of a sequence under a new grammar (two-phase decoding),
top-token probabilities, cheap resampling, and prefix-cache reuse. An
OpenAI-style chat endpoint offers none of that cleanly; llama.cpp's native
`/completion` endpoint offers all of it over HTTP.
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
import urllib.error
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class Generation:
    text: str
    probs: list = field(default_factory=list)  # per token: [(token_text, prob), ...] before the grammar
    token_details: list = field(default_factory=list, kw_only=True)  # selected token ids/logprobs and top alternatives
    stopped: str = ""
    completion_tokens: Optional[int] = None


@dataclass
class ChatTurn:
    value_confidence: list = field(default_factory=list, kw_only=True)  # one optional score per proposed call
    raw_response: Optional[dict] = field(default=None, kw_only=True)  # complete server reply, including any reasoning field
    calls: list = field(default_factory=list)       # [(tool name, arguments dict)]
    text: str = ""
    raw_calls: list = field(default_factory=list)   # the API's tool_calls, for the history
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None


class Decoder(Protocol):
    def format(self, messages: list) -> str: ...

    def generate(self, prompt: str, *, grammar: Optional[str], max_tokens: int, temperature: float,
                 seed: Optional[int], stop: list, n_probs: int = 0) -> Generation: ...


class TurnDriver(Protocol):
    """The small backend contract consumed by ToolAgent."""

    def chat(self, messages: list, tools: list, *, temperature: float,
             seed: Optional[int], max_tokens: int) -> ChatTurn: ...


def _compact_write_alternatives(alternatives: list) -> list:
    """Share one typed chat tool across destinations with the same value type.

    Source-copy alternatives group only when they have the same fitting source
    set, so every offered destination/source pair remains valid. Literal value
    schemas remain exact. This keeps the chat menu smaller as the workspace
    gains fields and locals without weakening type guidance.
    """
    grouped = {}
    for alternative in alternatives:
        alt = deepcopy(alternative)
        path = alt.get("path") or {}
        stated = alt.get("type") or {}
        if "const" not in path or "const" not in stated or not ("value" in alt or "source" in alt):
            key = ("unique", len(grouped))
            grouped[key] = alt
            continue
        kind = "value" if "value" in alt else "source"
        if kind == "source" and "enum" not in alt["source"]:
            grouped[("unique", len(grouped))] = alt
            continue
        key = (kind, stated["const"],
               json.dumps(alt.get("value") if kind == "value" else alt.get("source"), sort_keys=True),
               json.dumps(alt.get("done"), sort_keys=True))
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = alt
            continue
        paths = previous["path"]["enum"] if "enum" in previous["path"] else [previous["path"]["const"]]
        previous["path"] = {"enum": list(dict.fromkeys([*paths, path["const"]]))}
    return list(grouped.values())


def _cache_stable_tools(tools: list) -> list:
    """Remove state-dependent schema hints while preserving the tool contract.

    Bonsai's chat template renders tools before every other prompt token.  Path
    enums and line-number enums change after almost every action, which makes a
    growing conversation look like an unrelated prompt to the server cache.
    The runtime already validates every submitted action, so these fields can
    be presented as their underlying JSON types.  Function signatures remain
    visible in the user request and workspace state.
    """
    tools = deepcopy(tools)
    by_name = {tool["function"]["name"]: tool for tool in tools}

    def prop(name: str, field: str, schema: dict) -> None:
        tool = by_name.get(name)
        if tool:
            tool["function"]["parameters"].get("properties", {})[field] = schema

    prop("read", "path", {"type": "string", "description": "workspace path to read"})
    prop("edit", "path", {"type": "string", "description": "workspace text path to edit"})
    prop("write", "done", {"anyOf": [{"type": "integer"}, {"type": "array",
         "items": {"type": "integer"}, "minItems": 1, "maxItems": 2}]})
    write = by_name.get("write")
    if write:
        params = write["function"]["parameters"]
        if "anyOf" in params:
            params["anyOf"] = [{"required": ["value"]}, {"required": ["source"]},
                               {"required": ["type"]}]
    call = by_name.get("call") or by_name.get("call_function")
    if call:
        params = call["function"]["parameters"]
        properties = params.get("properties", {})
        properties["function"] = {"type": "string"}
        properties["inputs"] = {"type": "object", "additionalProperties": {"type": "string"}}
        properties["values"] = {"type": "object", "additionalProperties": True}
        if "done" in properties:
            properties["done"] = {"anyOf": [{"type": "integer"}, {"type": "array",
                "items": {"type": "integer"}, "minItems": 1, "maxItems": 2}]}
    return tools


class LlamaServerDecoder:
    """llama.cpp `llama-server`, native /completion endpoint.

    To verify against the running server version: that `n_probs` with
    post_sampling_probs=false reports probabilities before the grammar mask,
    and that prefix reuse works for this hybrid architecture.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", slot: Optional[int] = None,
                 timeout: Optional[float] = None,
                 chat_extra: Optional[dict] = None, tool_aliases: Optional[dict] = None,
                 json_text_values: bool = False, typed_alternatives: bool = False,
                 cache_stable_tools: bool = False):
        if json_text_values and typed_alternatives:
            raise ValueError("typed alternatives and JSON-text values are different transports")
        self.base_url, self.slot, self.timeout = base_url.rstrip("/"), slot, timeout
        self.deadline = None
        self._request_deadline = ContextVar(f"decoder-deadline-{id(self)}", default=None)
        self.json_text_values = json_text_values
        self.typed_alternatives = typed_alternatives
        self.cache_stable_tools = cache_stable_tools
        # per-model opt-in: harness tool name -> the name this model's server is shown. (Bonsai's server
        # cannot emit a tool literally named `call`: its tool-call format uses that word itself.)
        self.tool_aliases = tool_aliases or {}
        # extra fields for /v1/chat/completions, e.g. {"thinking_budget_tokens": 512, "top_p": 0.95, "top_k": 20}
        self.chat_extra = chat_extra or {}
        self.usage = {"turns": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def request_timeout(self):
        deadline = self._request_deadline.get()
        if deadline is None:
            deadline = self.deadline
        if deadline is None:
            return self.timeout
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("episode wall-clock budget exhausted")
        return left if self.timeout is None else min(self.timeout, left)

    @contextmanager
    def request_scope(self, *, deadline):
        token = self._request_deadline.set(deadline)
        try:
            yield
        finally:
            self._request_deadline.reset(token)

    def format(self, messages: list) -> str:
        """Render messages with the loaded model's own chat template, ending at the
        start of an assistant turn. The server adds the beginning-of-text token itself."""
        req = urllib.request.Request(self.base_url + "/apply-template",
                                     data=json.dumps({"messages": messages}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.request_timeout()) as resp:
            return json.loads(resp.read())["prompt"]

    def _chat_tools(self, tools: list) -> tuple[list, dict]:
        """Compile the stable natlang actions into this server's tool schemas."""
        # `x-...` keys are for our own grammar (natlang/native.py); a server only reads plain JSON Schema,
        # and the alternatives are most of the schema's size
        variant_names = {}
        if self.typed_alternatives:
            expanded = []
            for tool in tools:
                fn = tool["function"]
                alts = (fn.get("parameters") or {}).get("x-natlang-alternatives")
                if fn["name"] not in ("write", "call") or not alts:
                    expanded.append(tool)
                    continue
                if fn["name"] == "write":
                    alts = _compact_write_alternatives(alts)
                for index, alt in enumerate(alts):
                    name = f"{fn['name']}_alt_{index}"
                    variant_names[name] = fn["name"]
                    optional = alt.get("x-optional") or []
                    fields = {key: value for key, value in alt.items() if key != "x-optional"}
                    summary = ", ".join(f"{key}={value['const']}" for key, value in fields.items()
                                        if isinstance(value, dict) and "const" in value)
                    verb = "Write or copy" if fn["name"] == "write" else "Call"
                    expanded.append({**tool, "function": {**fn, "name": name,
                        "description": f"{verb}. {summary}.",
                        "parameters": {"type": "object", "properties": fields,
                                       "required": [key for key in fields if key not in optional],
                                       "additionalProperties": False}}})
            tools = expanded
        tools = [{**t, "function": {**t["function"], "parameters": {
            k: v for k, v in (t["function"].get("parameters") or {}).items() if not k.startswith("x-")}}} for t in tools]
        if self.json_text_values:
            # XML tool parsers need an unambiguous argument type. The runtime
            # already parses JSON text for non-Text slots; no wrapper repair.
            for tool in tools:
                if tool["function"]["name"] == "write":
                    params = tool["function"]["parameters"]
                    params["properties"] = {**params["properties"], "value": {
                        "type": "string", "description":
                        'For Text, plain text. For all other types, JSON text of the value itself, '
                        'for example 7, true, [1,2], or {"size":7}. Put that JSON directly in the parameter; '
                        'use normal JSON escaping within its string values. The runtime parses this JSON text.'}}
        if self.tool_aliases:
            out_name = lambda n: self.tool_aliases.get(n, n)
            tools = [{**t, "function": {**t["function"], "name": out_name(t["function"]["name"])}} for t in tools]
        if self.cache_stable_tools:
            tools = _cache_stable_tools(tools)
        return tools, variant_names

    def presented_tools(self, tools: list) -> list:
        """The exact schemas offered to the model, for trajectory capture."""
        return self._chat_tools(tools)[0]

    def presented_messages(self, messages: list) -> list:
        if not self.tool_aliases:
            return messages
        out_name = lambda n: self.tool_aliases.get(n, n)
        return [{**m, "tool_calls": [{**c, "function": {**c["function"],
                 "name": out_name(c["function"]["name"])}}
                 for c in m["tool_calls"]]} if m.get("tool_calls") else m for m in messages]

    def chat(self, messages: list, tools: list, *, temperature: float, seed: Optional[int] = None,
             max_tokens: Optional[int] = None) -> ChatTurn:
        """One assistant turn with native tool calling through the server's chat template."""
        tools, variant_names = self._chat_tools(tools)
        messages = self.presented_messages(messages)
        payload = {"messages": messages, "tools": tools, "tool_choice": "auto", "temperature": temperature,
                   "cache_prompt": True,
                   "parallel_tool_calls": True}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        payload.update(self.chat_extra)
        if max_tokens is not None:
            payload["max_tokens"] = min(payload.get("max_tokens", max_tokens), max_tokens)
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        import time
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout()) as resp:
                out = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            raise ValueError(f"chat server HTTP {exc.code}: {detail}") from exc
        raw_response = deepcopy(out)
        msg = out["choices"][0]["message"]
        self.usage["turns"] += 1
        self.usage["seconds"] += time.time() - t0
        usage = out.get("usage") or {}
        tokens = usage.get("completion_tokens", max_tokens or 0)
        self.usage["prompt_tokens"] += usage.get("prompt_tokens") or 0
        self.usage["completion_tokens"] += tokens
        calls = []
        back = {v: k for k, v in self.tool_aliases.items()}
        back.update(variant_names)
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            name = back.get(fn.get("name"), fn.get("name", ""))
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__unparsed__": fn.get("arguments")}
            calls.append((name, args if isinstance(args, dict) else {"value": args}))
        return ChatTurn(calls, (msg.get("content") or "").strip(), msg.get("tool_calls") or [], tokens,
                        prompt_tokens=usage.get("prompt_tokens"),
                        raw_response=raw_response)

    def generate(self, prompt, *, grammar, max_tokens, temperature, seed, stop, n_probs=0) -> Generation:
        payload = {"prompt": prompt, "n_predict": -1 if max_tokens is None else max_tokens,
                   "temperature": temperature, "stop": stop,
                   "cache_prompt": True, "n_probs": n_probs, "post_sampling_probs": False}
        if grammar:
            payload["grammar"] = grammar
        if seed is not None:
            payload["seed"] = seed
        if self.slot is not None:
            payload["id_slot"] = self.slot
        req = urllib.request.Request(self.base_url + "/completion", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.request_timeout()) as resp:
            out = json.loads(resp.read())
        tokens = out.get("tokens_predicted", max_tokens)
        self.usage["turns"] += 1
        self.usage["prompt_tokens"] += out.get("tokens_evaluated") or 0
        self.usage["completion_tokens"] += tokens
        self.usage["seconds"] += time.monotonic() - started
        probs = []
        for t in out.get("completion_probabilities") or []:
            cands = t.get("top_logprobs") or t.get("top_probs") or []
            probs.append([(c.get("token", ""), c["prob"] if "prob" in c else math.exp(c.get("logprob", -99.0)))
                          for c in cands])
        return Generation(out.get("content", ""), probs, out.get("stopping_word", ""), tokens,
                          token_details=out.get("completion_probabilities") or [])


@dataclass(frozen=True)
class Wrapper:
    """How an action sits inside the assistant turn, and how results come back.

    The protocol itself is template-agnostic: the prompt is rendered by the
    model's own chat template (`Decoder.format`), an action is plain text in
    the assistant turn, and results return as user turns, so roles strictly
    alternate. A wrapper is an optional per-model adaptation, intended for the
    models we fine-tune.
    """

    action_open: str = ""           # text placed at the start of the assistant turn
    action_close: str = ""          # text that closes a call when it is shown in history
    pythonic: bool = False          # render past calls with Python literals (True/None) rather than JSON
    result_role: str = "user"       # role used to feed an action's result back
    result_label: str = "RESULT\n"
    think_close: Optional[str] = None   # reasoning models: let the model think up to this marker first
    max_think_tokens: int = 2048


GENERIC = Wrapper()
LFM_TOOLCALL = Wrapper(action_open="<|tool_call_start|>", action_close="<|tool_call_end|>", pythonic=True,
                       result_role="tool", result_label="")
REASONING = Wrapper(think_close="</think>")
WRAPPERS = {"generic": GENERIC, "lfm": LFM_TOOLCALL, "reasoning": REASONING}
