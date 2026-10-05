#!/usr/bin/env python3
"""vLLM rollouts with Neuralese against the reference engine (natlang_neuralese/serve/vllm_rollout.py): greedy plain
and block-reading prompts give the same message, a forced write gives the same block (port in float32), and the
throughput of 64 sampled sequences on each. Runs inside the DGX vLLM image: scripts/neuralese_vllm_parity.sh HEADS.
Prints `REPORT {json}`.
"""
import json, re, sys, time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "neuralese"))
import torch
from natlang_neuralese.serve.vllm_rollout import VllmNeuralese
from natlang_neuralese.serve.engine import GenerationRequest
from natlang_neuralese.serve.grad import embed_text

def main():
    heads = sys.argv[1]
    # `lora` mode: vLLM's LoRA kernels run in bf16 only, so vLLM serves bf16 (the reference stays float32) and only the
    # adapter cases (and a plain baseline) run.
    lora = len(sys.argv) > 2 and sys.argv[2] == "lora"
    rollout = VllmNeuralese.load(heads=heads, gpu_memory_utilization=0.06, dtype="bfloat16" if lora else "float32",
                                 max_model_len=4096, port_dtype=torch.float32, max_lora_rank=16 if lora else 0)
    engine = rollout.engine
    engine.start()
    report = {}
    hint = embed_text(engine, "the capital of France", type="Neuralese<string>")
    # Two xs adapters bound together (served by vLLM as one LoRA of rank 8 + 4).
    from natlang_neuralese.serve.store import make_block
    bank = engine.adapter_bank
    adapters = []
    for rank, seed in ((8, 1), (4, 2)):
        spec = bank.spec(kind="xs", rank=rank, cutoff=engine.heads.cutoff)
        generator = torch.Generator().manual_seed(seed)
        coefficients = 1.5 * torch.randn(len(bank.matrices(spec)), spec.width, generator=generator)
        adapters.append({"id": engine.store.put(make_block(coefficients, spec.dialect(), type="Adapter")).id, "scale": 0.8})
    cases = {
        "plain": {"messages": [{"role": "user", "content": "Say hi in five words."}], "max_tokens": 16},
        "read": {"messages": [{"role": "user", "content": [{"type": "text", "text": "Hint: "}, {"type": "neuralese", "id": hint.id},
                                                         {"type": "text", "text": "\nWhich city? One word."}]}], "max_tokens": 8},
        "write": {"messages": [{"role": "user", "content": "Write a note about Paris."}], "max_tokens": 24,
                  "forced": ["Note: ", {"neuralese": "write"}]},
        "adapters": {"messages": [{"role": "user", "content": "Name three cities in France."}], "max_tokens": 16,
                     "adapters": adapters},
        "adapters-write": {"messages": [{"role": "user", "content": "Write a note about Lyon."}], "max_tokens": 24,
                           "forced": ["Note: ", {"neuralese": "write"}], "adapters": adapters},
    }
    cases = {k: v for k, v in cases.items() if (k == "plain" or "adapters" in v) == lora or k == "plain"}
    for name, case in cases.items():
        ours = rollout.generate([dict(case, seed=7)])[0]
        ref = engine.generate(GenerationRequest(messages=case["messages"], max_tokens=case["max_tokens"], seed=7,
                                                forced=case.get("forced"), adapters=case.get("adapters")))
        if case.get("adapters"):
            base = engine.generate(GenerationRequest(messages=case["messages"], max_tokens=case["max_tokens"], seed=7,
                                                     forced=case.get("forced")))
            report.setdefault("adapter_changes_reply", {})[name] = base["choices"][0]["message"] != ref["choices"][0]["message"]
        a, b = ours["choices"][0]["message"], ref["choices"][0]["message"]
        # Block IDs hash float payloads: compare messages with blocks numbered (payloads are compared below).
        same = lambda m: re.sub(r"nz1_[a-z2-7]+", "<block>", json.dumps(m))
        entry = {"vllm": json.dumps(a)[:300], "reference": json.dumps(b)[:300], "same_message": same(a) == same(b)}
        if ours["neuralese"]["blocks"]:
            x = engine.lookup(ours["neuralese"]["blocks"][0]["id"]).payload.float()
            y = engine.lookup(ref["neuralese"]["blocks"][0]["id"]).payload.float()
            entry["block_lengths"] = [x.shape[0], y.shape[0]]
            entry["payload_rms"] = float(y.pow(2).mean().sqrt())
            if x.shape == y.shape:
                entry["payload_max_diff"] = float((x - y).abs().max())
        report[name] = entry
    if lora:
        print("REPORT " + json.dumps(report))
        engine.stop()
        return
    # Throughput: 64 plain prompts, 64 tokens each, sampled.
    prompts = [{"messages": [{"role": "user", "content": f"Tell me fact number {i} about rivers."}], "max_tokens": 64,
                "temperature": 0.7, "seed": i} for i in range(64)]
    started = time.time(); out = rollout.generate(prompts); seconds = time.time() - started
    tokens = sum(o["usage"]["completion_tokens"] for o in out)
    report["throughput_vllm"] = {"sequences": 64, "tokens": tokens, "seconds": round(seconds, 2), "tokens_per_second": round(tokens / seconds, 1)}
    started = time.time()
    futures = [engine.submit(GenerationRequest(messages=p["messages"], max_tokens=64, temperature=0.7, seed=p["seed"])) for p in prompts]
    ref_out = [f.result() for f in futures]; seconds = time.time() - started
    tokens = sum(o["usage"]["completion_tokens"] for o in ref_out)
    report["throughput_reference"] = {"sequences": 64, "tokens": tokens, "seconds": round(seconds, 2), "tokens_per_second": round(tokens / seconds, 1)}
    print("REPORT " + json.dumps(report))
    engine.stop()


if __name__ == "__main__":
    main()
