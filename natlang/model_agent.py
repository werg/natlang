"""An agent backed by a language model, decoding each action in two constrained phases."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .actions import parse_action
from .decoder import GENERIC, Decoder, Wrapper
from .grammar import body_grammar, header_grammar

PROMPTS = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS / "interpreter.md").read_text()          # for a teacher model
SMALL_PROMPT = (PROMPTS / "interpreter_small.md").read_text()     # for an untuned small model: short, and it
                                                                  # never names `stuck`, which such a model parrots
RESAMPLES = 3
MAX_BODY_TOKENS, MAX_HEADER_TOKENS = 600, 96


class ModelAgent:
    def __init__(self, decoder: Decoder, *, wrapper: Wrapper = GENERIC, temperature: float = 0.2,
                 system_prompt: str = SYSTEM_PROMPT, log: Optional[list] = None):
        self.dec, self.w = decoder, wrapper
        self.temperature, self.system, self.log = temperature, system_prompt, log if log is not None else []

    def run(self, session) -> Optional[str]:
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": session.observation()}]
        while True:
            chosen = None
            for attempt in range(RESAMPLES + 1):
                text = self._decode_action(session, messages, seed=attempt)
                if text.startswith("stuck"):
                    return text.partition("\n")[2].strip() or "stuck, no note given"
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
            messages += [{"role": "assistant", "content": self.w.action_open + text},
                         {"role": self.w.result_role, "content": self.w.result_label + result.text}]

    def _decode_action(self, session, messages: list, seed: int) -> str:
        prefix = self.dec.format(messages)
        temp = self.temperature if seed == 0 else max(self.temperature, 0.7)
        if self.w.think_close:                # a reasoning model thinks freely first; the action is constrained
            thought = self.dec.generate(prefix, grammar=None, max_tokens=self.w.max_think_tokens,
                                        temperature=temp, seed=seed, stop=[self.w.think_close])
            prefix += thought.text + self.w.think_close + "\n"
        prefix += self.w.action_open
        head = self.dec.generate(prefix, grammar=header_grammar(session), max_tokens=MAX_HEADER_TOKENS,
                                 temperature=temp, seed=seed, stop=["\n"], n_probs=5)
        header = head.text.strip("\n")
        if header == "stuck":
            note = self.dec.generate(prefix + "stuck\n", grammar=None, max_tokens=120, temperature=temp,
                                     seed=seed, stop=[])
            return "stuck\n" + note.text.strip()
        try:
            grammar = body_grammar(session, parse_action(header))
        except Exception:
            return header                     # let validation produce the diagnostic
        if grammar is None:
            return header
        body = self.dec.generate(prefix + header + "\n", grammar=grammar, max_tokens=MAX_BODY_TOKENS,
                                 temperature=temp, seed=seed, stop=[], n_probs=5)
        return header + "\n" + body.text.rstrip("\n")
