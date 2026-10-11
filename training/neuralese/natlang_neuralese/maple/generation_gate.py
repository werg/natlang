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
import torch
from torch.nn.utils import parametrize

from ..eval.generation_checks import check, summarize
from .ternary import QUANT_MIX


@contextlib.contextmanager
def deployed_cache():
    """Compute every ternary (ramped) weight once for the whole block: attention through torch's parametrization
    cache, experts through DenseExperts' unbound cache. Costs one deployed copy of the weights while active. Reentrant:
    inside an active block (a whole precision column, train/text_warmup.py) it keeps the outer cache. Only for
    no-grad forwards at one fixed precision point: the cached values are not refreshed when the point changes."""
    if QUANT_MIX.get("cache") is not None:
        yield
        return
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
    report = summarize(results)
    report["gate_results"] = results
    return report


def behaviour_report(model, tokenizer, probes: list[dict], render, eos: int, held: dict | None = None,
                     held_rows: int = 48, prompt_weight: float = 0.25) -> dict:
    """The conversion's behaviour gate as one report: the generation gate on ``probes`` and, with ``held`` (teacher
    shards from ``qat_convert._load_record_shards``), the held KL to the BF16 teacher's top-k (``weighted_topk_kl``,
    prompt positions weighted ``prompt_weight``) and greedy agreement. Shared by conversion v3's verify_export, the
    quantization component's per-precision gate columns and its init gate (train/quantization.py): one
    implementation at whatever precision point is active."""
    gate = run_gate(model, tokenizer, probes, render, eos, keep_text=True)
    report = {"gate_pass": gate["gate_pass"], "gate_checks": gate["gate_checks"], "gate_kinds": gate["gate_kinds"],
              "gate_results": gate["gate_results"]}
    if held is not None:
        report.update(held_kl(model, held, held_rows, prompt_weight))
    return report


def held_kl(model, held: dict, rows: int = 48, prompt_weight: float = 0.25) -> dict:
    """Teacher-forced KL to the BF16 teacher's top-k on the first ``rows`` held records (``held_kl``, its prompt and
    response parts, response CE and greedy agreement): the behaviour gate's drift measure, without generation. The
    text warm-up also logs it at every cadence evaluation (``behaviour_drift``, with its guard)."""
    from .qat_convert import weighted_topk_kl

    device = next(model.parameters()).device
    sums: dict = {}
    count = min(rows, len(held["ids"]))
    with torch.no_grad():
        for i in range(count):
            scores = weighted_topk_kl(model, held["ids"][i][None].long().to(device), held["top_ids"][i],
                                      held["top_logp"][i], held["prompt_len"][i], prompt_weight)
            for key, value in scores.items():
                sums[key] = sums.get(key, 0.0) + float(value)
    return {"held_" + k.replace("loss_", ""): v / max(count, 1) for k, v in sums.items()}
