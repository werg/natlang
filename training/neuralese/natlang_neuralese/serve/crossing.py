"""The adapter crossing test (LEARNING_CONTINUUM.md §6.1, "Boundary check").

A block written inside an adapted call and read outside it must still mean the same thing. `crossing` writes a block
under the adapters (a forced write after `write_messages`), then scores a decision that reads it (`read_messages`
with the placeholder part `{"type": "neuralese", "id": "$block"}`) twice: under the adapters and without them. The
crossing divergence is KL(read under adapters ‖ read without). For scale, the same is reported for a block written
without adapters (`reference`): how much the adapters change the reading of an ordinary block.

The result is a measurement, judged by rubric next to the adapter's own gain; where it is large the adapter is trained
with the crossing as a consistency term.
"""

from __future__ import annotations

import math

import torch

from .engine import GenerationRequest
from .grad import decide


def _with_block(messages: list, block_id: str) -> list:
    def swap(value):
        if isinstance(value, dict):
            if value.get("type") == "neuralese" and value.get("id") == "$block":
                return {**value, "id": block_id}
            return {k: swap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [swap(v) for v in value]
        return value
    return swap(messages)


def _kl(p: list[float], q: list[float]) -> float:
    return float(sum(a * (math.log(max(a, 1e-12)) - math.log(max(b, 1e-12))) for a, b in zip(p, q) if a > 0))


def _distribution(engine, messages, options, adapters) -> list[float]:
    scores = decide(engine, {"messages": messages, "options": options, **({"adapters": adapters} if adapters else {})})
    return torch.softmax(torch.tensor(scores["log_probs"]), 0).tolist()


def crossing(engine, adapters: list, write_messages: list, read_messages: list, options: list[str],
             forced_prefix: str = "Note: ") -> dict:
    """Write under `adapters`, read with and without them. Returns the distributions and divergences."""
    def write(bound):
        request = GenerationRequest(messages=write_messages, forced=[forced_prefix, {"neuralese": "write"}],
                                    max_tokens=engine.max_block + 8, adapters=bound or None)
        response = engine.generate(request)
        return response["neuralese"]["blocks"][0]["id"]

    adapted_block, plain_block = write(adapters), write(None)
    inside = _distribution(engine, _with_block(read_messages, adapted_block), options, adapters)
    outside = _distribution(engine, _with_block(read_messages, adapted_block), options, None)
    reference_inside = _distribution(engine, _with_block(read_messages, plain_block), options, adapters)
    reference_outside = _distribution(engine, _with_block(read_messages, plain_block), options, None)
    return {"blocks": {"adapted": adapted_block, "plain": plain_block},
            "inside": inside, "outside": outside, "crossing_kl": _kl(inside, outside),
            "reference_kl": _kl(reference_inside, reference_outside)}
