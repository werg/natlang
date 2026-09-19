"""An agent backed by a language model, decoding each action in two constrained phases."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .actions import parse_action
from .decoder import GENERIC, Decoder, Wrapper
from .grammar import body_grammar, header_grammar

PROMPTS = Path(__file__).parent / "prompts"
SYSTEM_PROMPT = (PROMPTS / "interpreter.md").read_text()          # for a teacher model
SMALL_PROMPT = (PROMPTS / "interpreter_small.md").read_text()     # for an untuned small model: short, and it
                                                                  # never names `stuck`, which such a model parrots
MAX_BODY_TOKENS, MAX_HEADER_TOKENS = 600, 96


class ModelAgent:
    def __init__(self, decoder: Decoder, *, wrapper: Wrapper = GENERIC, temperature: float = 0.2,
                 system_prompt: str = SYSTEM_PROMPT, log: Optional[list] = None):
        self.dec, self.w = decoder, wrapper
        self.temperature, self.system, self.log = temperature, system_prompt, log if log is not None else []

    def run(self, session) -> Optional[str]:
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": session.observation()}]
        self.tokens = self.turns = 0
        previous_deadline = getattr(self.dec, "deadline", None)
        deadline = time.monotonic() + 900
        previous_runtime_deadline = session.rt.deadline
        if previous_runtime_deadline is not None:
            deadline = min(deadline, previous_runtime_deadline)
        self.dec.deadline = min(deadline, previous_deadline) if previous_deadline is not None else deadline
        session.rt.deadline = self.dec.deadline
        try:
            while True:
                text = self._decode_action(session, messages, seed=0)
                if text.startswith("stuck"):
                    return text.partition("\n")[2].strip() or "stuck, no note given"
                result = session.act(text)
                self.log.append({"action": text, "kind": result.kind, "attempt": 0})
                if session.completed:
                    return None
                if result.kind == "budget":
                    return "budget exhausted"
                messages += [{"role": "assistant", "content": self.w.action_open + text},
                             {"role": self.w.result_role, "content": self.w.result_label + result.text}]
        except TimeoutError:
            return "episode turn, token, or wall-clock budget exhausted"
        finally:
            self.dec.deadline = previous_deadline
            session.rt.deadline = previous_runtime_deadline

    def _generate(self, *args, max_tokens, **kwargs):
        if self.turns >= 64 or self.tokens >= 4000 or time.monotonic() >= self.dec.deadline:
            raise TimeoutError("episode budget exhausted")
        allowance = min(max_tokens, 4000 - self.tokens)
        result = self.dec.generate(*args, max_tokens=allowance, **kwargs)
        self.turns += 1
        used = getattr(result, "completion_tokens", None)
        self.tokens += allowance if used is None else max(1, used)
        if self.tokens > 4000 or time.monotonic() >= self.dec.deadline:
            raise TimeoutError("episode budget exhausted")
        return result

    def _decode_action(self, session, messages: list, seed: int) -> str:
        prefix = self.dec.format(messages)
        temp = self.temperature if seed == 0 else max(self.temperature, 0.7)
        if self.w.think_close:                # a reasoning model thinks freely first; the action is constrained
            thought = self._generate(prefix, grammar=None, max_tokens=self.w.max_think_tokens,
                                        temperature=temp, seed=seed, stop=[self.w.think_close])
            prefix += thought.text + self.w.think_close + "\n"
        prefix += self.w.action_open
        head = self._generate(prefix, grammar=header_grammar(session), max_tokens=MAX_HEADER_TOKENS,
                                 temperature=temp, seed=seed, stop=["\n"], n_probs=5)
        header = head.text.strip("\n")
        if header == "stuck":
            note = self._generate(prefix + "stuck\n", grammar=None, max_tokens=120, temperature=temp,
                                     seed=seed, stop=[])
            return "stuck\n" + note.text.strip()
        try:
            grammar = body_grammar(session, parse_action(header))
        except Exception:
            return header                     # let validation produce the diagnostic
        if grammar is None:
            return header
        body = self._generate(prefix + header + "\n", grammar=grammar, max_tokens=MAX_BODY_TOKENS,
                                 temperature=temp, seed=seed, stop=[], n_probs=5)
        return header + "\n" + body.text.rstrip("\n")
