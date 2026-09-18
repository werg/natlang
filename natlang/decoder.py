"""Generation backends behind one interface (see PLAN: generation control).

The harness needs: a different grammar on every call, raw prompts with special
tokens, continuation of a sequence under a new grammar (two-phase decoding),
top-token probabilities, cheap resampling, and prefix-cache reuse. An
OpenAI-style chat endpoint offers none of that cleanly; llama.cpp's native
`/completion` endpoint offers all of it over HTTP.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class Generation:
    text: str
    probs: list = field(default_factory=list)  # per token: [(token_text, prob), ...] before the grammar
    stopped: str = ""


class Decoder(Protocol):
    def generate(self, prompt: str, *, grammar: Optional[str], max_tokens: int, temperature: float,
                 seed: Optional[int], stop: list, n_probs: int = 0) -> Generation: ...


class LlamaServerDecoder:
    """llama.cpp `llama-server`, native /completion endpoint.

    To verify against the running server version: that `n_probs` with
    post_sampling_probs=false reports probabilities before the grammar mask,
    and that prefix reuse works for this hybrid architecture.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", slot: Optional[int] = None, timeout: float = 120):
        self.base_url, self.slot, self.timeout = base_url.rstrip("/"), slot, timeout

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
        probs = [[(c.get("token") or c.get("tok_str", ""), c.get("prob", 0.0))
                  for c in (t.get("top_probs") or t.get("probs") or [])]
                 for t in out.get("completion_probabilities") or []]
        return Generation(out.get("content", ""), probs, out.get("stopping_word", ""))


@dataclass
class ChatTemplate:
    """LFM2-style ChatML framing. Verify against the model's own chat template."""

    bos: str = "<|startoftext|>"
    start: str = "<|im_start|>"
    end: str = "<|im_end|>\n"
    action_open: str = "<|tool_call_start|>\n"
    action_close: str = "<|tool_call_end|>"

    def turn(self, role: str, text: str) -> str:
        return f"{self.start}{role}\n{text}{self.end}"

    def assistant_prefix(self) -> str:
        return f"{self.start}assistant\n{self.action_open}"
