"""Generation gate for a ternary conversion (plans/mellum-port.md, conversion v3): greedy generation on fixed probe
prompts, checked for chat structure, not teacher-forced CE. Conversion v2 reached a better held CE than the BF16
original and still generated loops without <think> or <|im_end|>; a teacher-forced metric cannot see that.

Checks per probe (those that apply): ``think`` (thinking on: the reply opens and closes <think>; off: no new <think>),
``end`` (<|im_end|> is emitted), ``tool`` (a <tool_call> whose JSON parses, names an offered tool and has its required
arguments), ``no_loop`` (no 8-token n-gram four or more times), ``exact`` (the expected number appears in the final
answer). A probe passes when all its checks pass.
"""
from __future__ import annotations

import contextlib
import json
import re

import torch
from torch.nn.utils import parametrize

from .ternary import QUANT_MIX


@contextlib.contextmanager
def deployed_cache():
    """Compute every ternary (ramped) weight once for the whole block: attention through torch's parametrization
    cache, experts through DenseExperts' unbound cache. Costs one deployed copy of the weights while active."""
    QUANT_MIX["cache"] = {}
    try:
        with parametrize.cached():
            yield
    finally:
        QUANT_MIX["cache"] = None


@torch.no_grad()
def greedy(model, ids: list[int], max_new: int, eos: int) -> list[int]:
    """Greedy continuation through the port's KV cache (MaplePortBackbone over the bare MapleModel)."""
    from .maple_port import MaplePortBackbone

    runner = MaplePortBackbone.runner(model.model)
    checkpointing, runner.checkpoint_layers = runner.checkpoint_layers, False
    device = next(model.parameters()).device
    embed = model.model.embed_tokens  # marker-free: the runner's forward_ids would splice control rows
    out = runner.forward_embeds(embed(torch.tensor([ids], device=device)), logits=False)
    generated = []
    for _ in range(max_new):
        h = model.model.norm(out["h_final"][:, -1:])
        token = int(model.lm_head(h)[0, -1].float().argmax())
        generated.append(token)
        if token == eos:
            break
        out = runner.forward_embeds(embed(torch.tensor([[token]], device=device)), cache=out["cache"], logits=False)
    runner.checkpoint_layers = checkpointing
    return generated


def _loops(tokens: list[int], n: int = 8, repeats: int = 4) -> bool:
    counts: dict[tuple, int] = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i:i + n])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] >= repeats:
            return True
    return False


def check(probe: dict, text: str, tokens: list[int], eos: int) -> dict:
    checks = {"end": bool(tokens) and tokens[-1] == eos, "no_loop": not _loops(tokens)}
    if probe.get("enable_thinking"):
        open_at, close_at = text.find("<think>"), text.find("</think>")
        checks["think"] = open_at >= 0 and close_at > open_at
    else:
        checks["think"] = "<think>" not in text
    final = text.split("</think>")[-1]
    if probe.get("tools"):
        names = {t["function"]["name"]: t["function"]["parameters"].get("required", []) for t in probe["tools"]}
        calls = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", final, flags=re.S)
        ok = False
        for call in calls:
            try:
                parsed = json.loads(call)
            except json.JSONDecodeError:
                continue
            name, arguments = parsed.get("name"), parsed.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = None
            expected = probe.get("expected_tool")
            if name in names and isinstance(arguments, dict) and all(k in arguments for k in names[name]) and \
                    (expected is None or name == expected):
                ok = True
        if probe.get("kind") in ("tool", "natlang"):
            checks["tool"] = ok
    if probe.get("answer") is not None:
        checks["exact"] = probe["answer"] in re.findall(r"-?\d+", final.replace(",", ""))
    return checks


def run_gate(model, tokenizer, probes: list[dict], render, eos: int, max_new: int = 256, max_new_thinking: int = 512,
             keep_text: bool = False) -> dict:
    """Greedy-generate every probe and check it. Returns the pass rate, per-check rates and per-kind pass rates."""
    results = []
    with deployed_cache():
        for probe in probes:
            limit = max_new_thinking if probe.get("enable_thinking") else max_new
            tokens = greedy(model, render(tokenizer, probe), limit, eos)
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            checks = check(probe, text, tokens, eos)
            results.append({"id": probe["id"], "kind": probe.get("kind"), "checks": checks, "pass": all(checks.values()),
                            "tokens": len(tokens), **({"text": text[:1500]} if keep_text else {})})
    names = sorted({k for r in results for k in r["checks"]})
    report = {"gate_pass": sum(r["pass"] for r in results) / len(results),
              "gate_checks": {k: round(sum(r["checks"][k] for r in results if k in r["checks"]) /
                                       max(1, sum(k in r["checks"] for r in results)), 3) for k in names},
              "gate_kinds": {}}
    for kind in sorted({r["kind"] for r in results}):
        rows = [r for r in results if r["kind"] == kind]
        report["gate_kinds"][kind] = round(sum(r["pass"] for r in rows) / len(rows), 3)
    report["gate_results"] = results
    return report
