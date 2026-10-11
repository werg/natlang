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
import math
import torch
from torch.nn.utils import parametrize

from ..eval.generation_checks import check, summarize, output_budget
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
            limit = output_budget(probe, ordinary=max_new, thinking=max_new_thinking)
            tokens = greedy_any(model, render(tokenizer, probe), limit, eos)
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            checks = check(probe, text, tokens, eos)
            results.append({"id": probe["id"], "kind": probe.get("kind"), "checks": checks, "pass": all(checks.values()),
                            "tokens": len(tokens), "max_new_tokens": limit,
                            "budget_exhausted": len(tokens) == limit and (not tokens or tokens[-1] != eos),
                            **({"signal": probe["signal"]} if "signal" in probe else {}),
                            **({"text": text} if keep_text else {})})
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
    report = dict(gate)
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
    metric_rows: dict = {}
    count = min(rows, len(held["ids"]))
    if count < 1:
        raise ValueError("held drift monitor needs at least one row")
    response_rows = sum(len(held['ids'][i]) > int(held['prompt_len'][i]) for i in range(count))
    with torch.no_grad():
        for i in range(count):
            scores = weighted_topk_kl(model, held["ids"][i][None].long().to(device), held["top_ids"][i],
                                      held["top_logp"][i], held["prompt_len"][i], prompt_weight)
            for key, value in scores.items():
                value = float(value)
                if not math.isfinite(value):
                    if key.startswith('response_') and len(held['ids'][i]) <= int(held['prompt_len'][i]):
                        continue  # A historical truncated row has no response observation, not zero drift.
                    raise ValueError(f"nonfinite held drift metric {key} at row {i}")
                sums[key] = sums.get(key, 0.0) + value
                metric_rows[key] = metric_rows.get(key, 0) + 1
    report = {"held_" + k.replace("loss_", ""): v / metric_rows[k] for k, v in sums.items()}
    report['held_coverage'] = {'rows': count, 'response_rows': response_rows,
                               'response_missing_rows': count - response_rows,
                               'response_complete': response_rows == count,
                               'metric_rows': metric_rows}
    return report


@torch.no_grad()
def capture_reference(model, tokenizer, prompts: list[dict], render, eos: int, *, max_new: int = 256, k: int = 64,
                      max_tokens: int = 4096, generated_responses: list[list[int]] | None = None) -> dict:
    """A drift-monitor reference captured from ``model`` itself (the base model, before any update), in the teacher
    shards' format (``ids``, ``top_ids``, ``top_logp``, ``prompt_len``): each prompt rendered, answered greedily, and
    the model's own top-``k`` log-probabilities at every position. ``held_kl`` against it measures how far a later
    model has moved from this one: the monitor path of a lineage without a registered teacher artifact (no
    preservation loss reads it)."""
    if generated_responses is not None and len(generated_responses) != len(prompts):
        raise ValueError('saved reference responses must align one-to-one with prompts')
    device = next(model.parameters()).device
    reference = {"ids": [], "top_ids": [], "top_logp": [], "prompt_len": [], "capture_coverage": []}
    for index, prompt in enumerate(prompts):
        prefix = render(tokenizer, prompt)
        budget = output_budget(prompt, ordinary=max_new, thinking=max_new)
        generated = greedy_any(model, prefix, budget, eos) if generated_responses is None else generated_responses[index]
        if not isinstance(generated, list) or not generated or any(type(t) is not int or t < 0 for t in generated):
            raise ValueError('reference responses must be nonempty lists of token IDs')
        # Never chop the prompt to fit the teacher shard budget: that used to remove
        # every response position from long native prompts and produce NaN drift.
        kept = generated[:max(1, max_tokens - len(prefix))]
        if not kept:
            raise ValueError("reference capture produced no response tokens")
        ids = prefix + kept
        reference['capture_coverage'].append({'id': prompt.get('id'), 'prompt_tokens': len(prefix),
                                              'generated_tokens': len(generated), 'response_tokens': len(kept),
                                              'response_truncated': len(kept) < len(generated),
                                              'max_new_tokens': budget, 'effective_tokens': len(ids)})
        sequence = torch.tensor([ids], device=device)
        hidden = model.model(input_ids=sequence).last_hidden_state[0, :-1]
        top_ids, top_logp = [], []
        for start in range(0, hidden.shape[0], 512):
            logp = torch.log_softmax(model.lm_head(hidden[start:start + 512]).float(), -1)
            values, indices = logp.topk(k, -1)
            top_ids.append(indices.to(torch.int32).cpu())
            top_logp.append(values.to(torch.float16).cpu())
        reference["ids"].append(torch.tensor(ids, dtype=torch.int32))
        reference["top_ids"].append(torch.cat(top_ids))
        reference["top_logp"].append(torch.cat(top_logp))
        reference["prompt_len"].append(len(prefix))
    return reference


@torch.no_grad()
def greedy_any(model, ids: list[int], max_new: int, eos: int) -> list[int]:
    """Greedy continuation on any backbone: the Maple port's KV-cache path (``greedy``) for a Maple-family model, the
    Hugging Face ``generate`` otherwise (LFM, Qwen)."""
    from .model import MapleForCausalLM
    if isinstance(model, MapleForCausalLM):
        return greedy(model, ids, max_new, eos)
    device = next(model.parameters()).device
    out = model.generate(torch.tensor([ids], device=device), max_new_tokens=max_new, do_sample=False,
                         eos_token_id=eos, pad_token_id=eos)
    return out[0, len(ids):].tolist()
