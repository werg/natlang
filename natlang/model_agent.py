"""An agent backed by a language model, decoding each action in two constrained phases."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .actions import parse_action
from .decoder import ChatTemplate, Decoder
from .grammar import body_grammar, header_grammar

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "interpreter.md").read_text()
RESAMPLES = 3
MAX_BODY_TOKENS, MAX_HEADER_TOKENS = 600, 96


class ModelAgent:
    def __init__(self, decoder: Decoder, *, template: Optional[ChatTemplate] = None, temperature: float = 0.2,
                 system_prompt: str = SYSTEM_PROMPT, log: Optional[list] = None):
        self.dec, self.t = decoder, template or ChatTemplate()
        self.temperature, self.system, self.log = temperature, system_prompt, log if log is not None else []

    def run(self, session) -> Optional[str]:
        t = self.t
        context = t.bos + t.turn("system", self.system) + t.turn("user", session.observation())
        while True:
            chosen = None
            for attempt in range(RESAMPLES + 1):
                text = self._decode_action(session, context, seed=attempt)
                if text.startswith("close"):
                    return text.partition("\n")[2].strip() or "closed without a note"
                result = session.act(text)
                self.log.append({"action": text, "kind": result.kind, "attempt": attempt})
                if result.kind not in ("rejected", "error") or attempt == RESAMPLES:
                    chosen = (text, result)   # rejected samples are discarded, not shown
                    break
                session.actions -= 1          # a discarded sample does not spend the budget
            text, result = chosen
            if session.completed:
                return None
            if result.kind == "budget":
                return "budget exhausted"
            context += (f"{t.start}assistant\n{t.action_open}{text}{t.action_close}{t.end}"
                        + t.turn("tool", result.text))

    def _decode_action(self, session, context: str, seed: int) -> str:
        prefix = context + self.t.assistant_prefix()
        temp = self.temperature if seed == 0 else max(self.temperature, 0.7)
        head = self.dec.generate(prefix, grammar=header_grammar(session), max_tokens=MAX_HEADER_TOKENS,
                                 temperature=temp, seed=seed, stop=["\n"], n_probs=5)
        header = head.text.strip("\n")
        if header == "close":
            note = self.dec.generate(prefix + "close\n", grammar=None, max_tokens=120, temperature=temp,
                                     seed=seed, stop=[self.t.action_close, self.t.end.strip()])
            return "close\n" + note.text.strip()
        try:
            grammar = body_grammar(session, parse_action(header))
        except Exception:
            return header                     # let validation produce the diagnostic
        if grammar is None:
            return header
        body = self.dec.generate(prefix + header + "\n", grammar=grammar, max_tokens=MAX_BODY_TOKENS,
                                 temperature=temp, seed=seed, stop=[self.t.action_close, self.t.end.strip()],
                                 n_probs=5)
        return header + "\n" + body.text.rstrip("\n")
