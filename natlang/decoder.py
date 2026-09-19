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
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class Generation:
    text: str
    probs: list = field(default_factory=list)  # per token: [(token_text, prob), ...] before the grammar
    stopped: str = ""


@dataclass
class ChatTurn:
    calls: list = field(default_factory=list)       # [(tool name, arguments dict)]
    text: str = ""
    raw_calls: list = field(default_factory=list)   # the API's tool_calls, for the history


class Decoder(Protocol):
    def format(self, messages: list) -> str: ...

    def generate(self, prompt: str, *, grammar: Optional[str], max_tokens: int, temperature: float,
                 seed: Optional[int], stop: list, n_probs: int = 0) -> Generation: ...


class LlamaServerDecoder:
    """llama.cpp `llama-server`, native /completion endpoint.

    To verify against the running server version: that `n_probs` with
    post_sampling_probs=false reports probabilities before the grammar mask,
    and that prefix reuse works for this hybrid architecture.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", slot: Optional[int] = None, timeout: float = 120,
                 chat_extra: Optional[dict] = None):
        self.base_url, self.slot, self.timeout = base_url.rstrip("/"), slot, timeout
        # extra fields for /v1/chat/completions, e.g. {"thinking_budget_tokens": 512, "top_p": 0.95, "top_k": 20}
        self.chat_extra = chat_extra or {}
        self.usage = {"turns": 0, "completion_tokens": 0, "seconds": 0.0}

    def format(self, messages: list) -> str:
        """Render messages with the loaded model's own chat template, ending at the
        start of an assistant turn. The server adds the beginning-of-text token itself."""
        req = urllib.request.Request(self.base_url + "/apply-template",
                                     data=json.dumps({"messages": messages}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["prompt"]

    def chat(self, messages: list, tools: list, *, temperature: float, seed: Optional[int] = None,
             max_tokens: int = 700) -> ChatTurn:
        """One assistant turn with native tool calling. The server renders the model's own chat
        template, constrains arguments to each tool's JSON schema, and parses the model's native
        tool-call format, so this is the same call for every model."""
        payload = {"messages": messages, "tools": tools, "tool_choice": "auto", "temperature": temperature,
                   "max_tokens": max_tokens, "parallel_tool_calls": True}
        if seed is not None:
            payload["seed"] = seed
        payload.update(self.chat_extra)
        req = urllib.request.Request(self.base_url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        import time
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        msg = out["choices"][0]["message"]
        self.usage["turns"] += 1
        self.usage["seconds"] += time.time() - t0
        self.usage["completion_tokens"] += (out.get("usage") or {}).get("completion_tokens", 0)
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"__unparsed__": fn.get("arguments")}
            calls.append((fn.get("name", ""), args if isinstance(args, dict) else {"value": args}))
        return ChatTurn(calls, (msg.get("content") or "").strip(), msg.get("tool_calls") or [])

    def generate(self, prompt, *, grammar, max_tokens, temperature, seed, stop, n_probs=0) -> Generation:
        payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature, "stop": stop,
                   "cache_prompt": True, "n_probs": n_probs, "post_sampling_probs": False}
        if grammar:
            payload["grammar"] = grammar
        if seed is not None:
            payload["seed"] = seed
        if self.slot is not None:
            payload["id_slot"] = self.slot
        req = urllib.request.Request(self.base_url + "/completion", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        probs = []
        for t in out.get("completion_probabilities") or []:
            cands = t.get("top_logprobs") or t.get("top_probs") or []
            probs.append([(c.get("token", ""), c["prob"] if "prob" in c else math.exp(c.get("logprob", -99.0)))
                          for c in cands])
        return Generation(out.get("content", ""), probs, out.get("stopping_word", ""))


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
