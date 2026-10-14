#!/usr/bin/env python3
"""Compare a restored direct-stage student against its native HF forward path.

Use the frozen runtime through PYTHONPATH. Inputs are exact captured collector
requests; no generation or training data is published by this diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from functools import partial
from pathlib import Path

import torch

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.eval.task_cadence import restore_engine


def compare(reference: torch.Tensor, candidate: torch.Tensor, *, excluded_ids=()) -> dict:
    a, b = reference.float().reshape(-1), candidate.float().reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"logit shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
    delta = a - b
    log_a, log_b = a.log_softmax(-1), b.log_softmax(-1)
    sampled_a, sampled_b = (a / 0.6).softmax(-1), (b / 0.6).softmax(-1)
    ordered = delta.abs().topk(4)
    keep = torch.ones_like(a, dtype=torch.bool)
    if excluded_ids:
        keep[list(excluded_ids)] = False
    aa, bb = a[keep], b[keep]
    log_aa, log_bb = aa.log_softmax(-1), bb.log_softmax(-1)
    return {
        "max_logit_abs": float(delta.abs().max()),
        "rms_logit": float(delta.square().mean().sqrt()),
        "kl_hf_to_port": float(torch.sum(log_a.exp() * (log_a - log_b))),
        "top_diff_token_ids": [int(i) for i in ordered.indices],
        "top_diff_values": [float(v) for v in ordered.values],
        "non_control_max_logit_abs": float((aa - bb).abs().max()),
        "non_control_kl_hf_to_port": float(torch.sum(log_aa.exp() * (log_aa - log_bb))),
        "hf_argmax": int(a.argmax()), "port_argmax": int(b.argmax()),
        "hf_control_probability_at_t06": {str(i): float(sampled_a[i]) for i in excluded_ids},
        "port_control_probability_at_t06": {str(i): float(sampled_b[i]) for i in excluded_ids},
        "hf_control_total_at_t06": float(sampled_a[list(excluded_ids)].sum()),
        "port_control_total_at_t06": float(sampled_b[list(excluded_ids)].sum()),
        "hf_margin": float(a.topk(2).values.diff().abs().item()),
        "port_margin": float(b.topk(2).values.diff().abs().item()),
    }


def hf_forward(model, ids: torch.Tensor, cache=None):
    kwargs = {"input_ids": ids, "past_key_values": cache, "use_cache": True, "logits_to_keep": 1}
    return model(**kwargs)


def port_forward(engine, ids: torch.Tensor, cache=None):
    out = engine.backbone.forward_ids(ids, cache=cache, cutoff=engine.heads.cutoff, logits=False)
    logits, _ = engine.advance_token_readout(None, out["h_final"][:, -1:], last_only=True)
    return logits[:, -1], out["cache"]


def selected_requests(path: Path, engine):
    for item in json.loads(path.read_text()):
        row_path = Path(item["row"])
        lines = [line for line in row_path.read_bytes().split(b"\n") if line]
        if len(lines) != 1:
            raise ValueError(f"expected one exact collector row: {row_path}")
        row = json.loads(lines[0])
        turn = row["trajectory"][item["turn"]]
        if turn.get("phase") != "action":
            raise ValueError("selected turn must be an action request")
        engine.prompt_rendering = row["provenance"].get("prompt_rendering", "full")
        plan = engine._prompt_plan(turn["context"], turn["tools_offered"])
        if any(kind != "tokens" for kind, _ in plan):
            raise ValueError(f"captured request contains a neuralese block: {item['label']}")
        ids = [token for _, segment in plan for token in segment]
        if not ids:
            raise ValueError("empty captured prompt")
        yield {"label": item["label"], "row": str(row_path), "row_sha256": sha256_file_hex(row_path),
               "turn": item["turn"], "request_sha256": turn["request_sha256"],
               "prompt_rendering": row["provenance"].get("prompt_rendering"),
               "token_ids": ids, "token_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}


@torch.inference_mode()
def run_one(engine, item, steps: int, fresh_every: int):
    model = engine.backbone.hf
    controls = (engine.backbone.controls.open_id, engine.backbone.controls.close_id)
    score = partial(compare, excluded_ids=controls)
    device = engine.backbone.embedding_weight.device
    ids = torch.tensor([item["token_ids"]], device=device)
    hf = hf_forward(model, ids)
    hf_logits = hf.logits[:, -1]
    port_logits, port_cache = port_forward(engine, ids)
    result = {k: v for k, v in item.items() if k != "token_ids"}
    result["prefix_tokens"] = ids.shape[1]
    native_embeds = model.get_input_embeddings()(ids)
    port_embeds = engine.backbone.embed(ids)
    result["input_embedding_max_abs"] = float((native_embeds.float() - port_embeds.float()).abs().max())
    result["prefill"] = score(hf_logits, port_logits)
    result["steps"] = []
    forced = []
    hf_cache = hf.past_key_values
    for step in range(steps):
        token = int(hf_logits.float().argmax())
        forced.append(token)
        one = torch.tensor([[token]], device=device)
        hf = hf_forward(model, one, hf_cache)
        hf_cache = hf.past_key_values
        hf_logits = hf.logits[:, -1]
        port_logits, port_cache = port_forward(engine, one, port_cache)
        observation = {
            "position": int(ids.shape[1] + step), "forced_token": token,
            "cached_hf_vs_port": score(hf_logits, port_logits),
            "hf_cache_length": hf_cache.get_seq_length() if hasattr(hf_cache, "get_seq_length") else None,
            "port_cache_lengths": list(port_cache.lengths),
        }
        if step == 0 or step + 1 == steps or (step + 1) % fresh_every == 0:
            full_ids = torch.cat([ids, torch.tensor([forced], device=device)], dim=1)
            fresh_hf = hf_forward(model, full_ids).logits[:, -1]
            fresh_port, _ = port_forward(engine, full_ids)
            observation.update(hf_cached_vs_fresh=score(fresh_hf, hf_logits),
                               port_cached_vs_fresh=score(fresh_port, port_logits),
                               fresh_hf_vs_port=score(fresh_hf, fresh_port))
        result["steps"].append(observation)
    result["forced_token_ids"] = forced
    result["forced_text"] = engine.tokenizer.decode(forced, skip_special_tokens=False)
    return result


@torch.inference_mode()
def run_interleaved(engine, pair, steps: int, fresh_every: int):
    """Keep two independent HF and port caches live while advancing A, B, A, B."""
    model = engine.backbone.hf
    controls = (engine.backbone.controls.open_id, engine.backbone.controls.close_id)
    score = partial(compare, excluded_ids=controls)
    device = engine.backbone.embedding_weight.device
    states = []
    for item in pair:
        ids = torch.tensor([item["token_ids"]], device=device)
        hf = hf_forward(model, ids)
        port_logits, port_cache = port_forward(engine, ids)
        states.append({"label": item["label"], "ids": ids, "hf_cache": hf.past_key_values,
                       "hf_logits": hf.logits[:, -1], "port_cache": port_cache,
                       "port_logits": port_logits, "forced": [], "observations": []})
    for step in range(steps):
        for state in states:
            token = int(state["hf_logits"].float().argmax())
            state["forced"].append(token)
            one = torch.tensor([[token]], device=device)
            hf = hf_forward(model, one, state["hf_cache"])
            state["hf_cache"], state["hf_logits"] = hf.past_key_values, hf.logits[:, -1]
            state["port_logits"], state["port_cache"] = port_forward(engine, one, state["port_cache"])
            observation = {
                "forced_token": token,
                "cached_hf_vs_port": score(state["hf_logits"], state["port_logits"]),
            }
            if step == 0 or step + 1 == steps or (step + 1) % fresh_every == 0:
                full = torch.cat([state["ids"], torch.tensor([state["forced"]], device=device)], 1)
                fresh_hf = hf_forward(model, full).logits[:, -1]
                fresh_port, _ = port_forward(engine, full)
                observation.update(hf_cached_vs_fresh=score(fresh_hf, state["hf_logits"]),
                                   port_cached_vs_fresh=score(fresh_port, state["port_logits"]))
            state["observations"].append(observation)
    return [{"label": state["label"], "forced_token_ids": state["forced"],
             "forced_text": engine.tokenizer.decode(state["forced"], skip_special_tokens=False),
             "observations": state["observations"]} for state in states]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fresh-every", type=int, default=16)
    args = parser.parse_args()
    if args.steps < 1 or args.fresh_every < 1:
        parser.error("steps and fresh-every must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as temporary:
        request_dir = Path(temporary)
        (request_dir / "request.json").write_text(json.dumps({"weights": str(args.checkpoint.resolve())}))
        engine = restore_engine({"dir": str(request_dir), "heads": str(args.heads.resolve()),
                                 "device": "cuda", "prompt_rendering": "full"})
        report = {"schema": "natlang.qad-hf-port-parity/1", "checkpoint": str(args.checkpoint.resolve()),
                  "checkpoint_sha256": sha256_file_hex(args.checkpoint), "heads": str(args.heads.resolve()),
                  "heads_sha256": sha256_file_hex(args.heads), "requests_sha256": sha256_file_hex(args.requests),
                  "runtime_module": str(Path(restore_engine.__code__.co_filename).resolve()),
                  "runtime_module_sha256": sha256_file_hex(Path(restore_engine.__code__.co_filename)),
                  "torch": torch.__version__, "dtype": str(engine.backbone.embedding_weight.dtype),
                  "control_token_ids": [engine.backbone.controls.open_id, engine.backbone.controls.close_id],
                  "device": str(engine.backbone.embedding_weight.device), "cases": []}
        items = list(selected_requests(args.requests, engine))
        for item in items:
            case = run_one(engine, item, args.steps, args.fresh_every)
            report["cases"].append(case)
            (args.out / "report.partial.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"case": case["label"], "prefix_tokens": case["prefix_tokens"],
                              "decode_steps": len(case["steps"]),
                              "non_control_max": max(s["cached_hf_vs_port"]["non_control_max_logit_abs"]
                                                     for s in case["steps"]),
                              "argmax_mismatches": sum(s["cached_hf_vs_port"]["hf_argmax"] !=
                                                       s["cached_hf_vs_port"]["port_argmax"]
                                                       for s in case["steps"])}), flush=True)
            torch.cuda.empty_cache()
        if len(items) >= 2:
            pair = (items[0], items[2] if len(items) >= 3 else items[1])
            report["interleaved_two_requests"] = run_interleaved(engine, pair, args.steps, args.fresh_every)
        (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.out / "report.partial.json").unlink()


if __name__ == "__main__":
    main()
