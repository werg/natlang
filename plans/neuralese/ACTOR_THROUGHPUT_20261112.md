# RL actor throughput (2026-11-12, review §1.2 actor side)

Goal: rollout collection at >= 3,000 generated tok/s aggregate (DGX GB10 + Pop RTX 4060), measured end to end with the
real collector and harness on training tasks. Exactness of the actor is secondary: the learner recomputes logprobs and
truncated IS covers the gap.

## Where the time goes (measured)

Per episode (exp0 train-task eval, 134 episodes, base policy, 2 actors x 38 slots on the DGX):
- model calls are the whole episode: 9.8 calls, 664 generated tokens per call on average (p90 1,658), harness and tool
  work between calls is milliseconds (trace `model_request` start/end vs `action` events); the episode's first call
  starts about 7 s after the episode (process + harness start).
- prompts average 11.4k tokens (p10 5.9k, p90 17.8k); w5 round 4 captures: 11.5k average, 3,675 tokens shared by every
  prompt (system prompt + tool schemas), 92% of a call's prompt is the previous call's prompt + reply, 784 new tokens per
  call on average (median 151). Radix prefix hits work (`#cached-token` 5k-17k per prefill line).
- each sequence decodes at 9-14 tok/s. Prefill is not the bottleneck: decode is.

Per decode step (torch profile of the fork server, bs 96, 6k context, shared GPU): flashinfer paged attention of the six
full-attention layers, the projections (~0.6 ms a layer at bs 96, bandwidth bound) and the lm head; the sampler's
capture (`log_softmax` FP32 over 128k, top-k, FP64 tail-mass sum when a request asks for k > 0) adds a few ms. The
scheduler's CPU work is ~7 ms a step (overlap is off for correctness, see serve_native_checkpoint.py).

The GB10 is memory-bandwidth bound in decode: a step reads the weights once (~3.5 GB) plus every running sequence's KV
of its full-attention layers (6 layers x 4 KV heads x 128 x 2 x 2 bytes = 12 KB per context token in BF16, 6 KB in
FP8). At 11.5k tokens a sequence costs 140 MB per step in BF16, 70 MB in FP8: at ~210 GB/s the BF16 ceiling is about
1,500 tok/s however many sequences run, FP8 about 3,000.

What made the old setup slow:
1. Concurrency: 16-38 collector slots on an actor with `--sglang-max-running 32-48`: a step's fixed weight read is
   shared by few sequences.
2. KV pool 131k-262k tokens (mem fraction 0.2-0.3): at 30 running 11k-token sequences the pool sat at 93% usage with
   2-5 requests queued and evicted radix histories re-prefilled (`#queue-req`, prefill lines with 8k new tokens).
3. Two SGLang engines time-sliced on one GPU (exp0 ran two actors): every kernel of one waits for the other's slice;
   one engine with twice the slots is strictly better.
4. Pop ran the native engine (40-80 tok/s).

## Changes

- `serve_native_checkpoint.py` defaults (what `held_eval.serve`, `grpo.py` and every actor launch get):
  `--sglang-max-running 128`; `--sglang-kv-gb 12` fixes the KV pool in GB (tokens = GB / KV bytes per token from the
  export config: 532k BF16, 1.07M FP8) and derives SGLang's `mem-fraction-static` from the memory free at launch
  (SGLang keeps (1 - fraction) of the free memory as headroom: a fixed low fraction shrank the pool when other jobs
  held memory, a high one let the first actor take everything behind the ledger). Ledger budget = kv GB + 14.
  `--sglang-kv-cache-dtype fp8_e4m3` is available (recorded in the actor format `...-projection-kv-fp8_e4m3`).
- `held_eval.ACTOR_ARGS` no longer overrides those defaults with the slow ones; held_eval/grpo actor budgets 26 GB.
- JIT toolkit: without a system CUDA toolkit (Pop's `/usr/bin/nvcc` is 11.5 and rejects `-std=c++20`) the actor uses
  the fork venv's `nvidia/cu13` wheel toolkit. Pop's fork copy is now a git checkout at the fork commit (provenance).
- Harness variant `reply-text-with-calls` (off until measured): keeps a reply's prose before tool calls.

## Measurements

(filled in below as measured)
