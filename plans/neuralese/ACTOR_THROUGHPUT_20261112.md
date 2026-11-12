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
  held memory, a high one let the first actor take everything behind the ledger). Ledger budget = kv GB + 24 (measured: SGLang 23 GB at 16 GB KV, front 14 GB with the native model reloaded for write_text).
  `--sglang-kv-cache-dtype fp8_e4m3` is available (recorded in the actor format `...-projection-kv-fp8_e4m3`).
- `held_eval.ACTOR_ARGS` no longer overrides those defaults with the slow ones; held_eval/grpo actor budgets 26 GB.
- JIT toolkit: without a system CUDA toolkit (Pop's `/usr/bin/nvcc` is 11.5 and rejects `-std=c++20`) the actor uses
  the fork venv's `nvidia/cu13` wheel toolkit. Pop's fork copy is now a git checkout at the fork commit (provenance).
- Harness variant `reply-text-with-calls` (off until measured): keeps a reply's prose before tool calls.

## Measurements

End to end, real collectors and harness, base policy 5ff07638, training tasks (generated tokens per wall second from
the actors' counters; the GPU is shared with the GRPO learner in both new rows):

| setup | DGX actor | Pop actor | aggregate | per-episode stream |
|---|---|---|---|---|
| w5 round 4 (16 DGX + 4 Pop slots, BF16, 32 running, 131k KV) | | native | 269-311 tok/s | 22.9 tok/s, 776 s/episode |
| exp0 eval (2 BF16 actors x 38 slots, time-sliced) | 2 x ~165-230 | - | ~465 tok/s | 8.5-9.5 tok/s, 867-1223 s |
| exp1 + held-v2 A/A, BF16, 96 running, 393k KV (11:00, 9 min) | 915 | 285 (SGLang, 8 slots) | ~1,200 tok/s | 35.6 tok/s |
| same, FP8 KV, 16 GB = 1.42M tokens, 128 running (11:43, 13 min) | 1,086 | 191 | ~1,280 tok/s | 38.1 tok/s (early finishers) |

Scheduler view: BF16 at 90 running sat at 90% KV usage with 38-42 requests queued and 43% of prompt tokens prefilled again
after radix evictions (422k new vs 560k cached in 3 min); FP8 at 110 running: 66% usage, no queue, 12% new (371k vs
2.82M cached), decode 1,440 tok/s. With the pool no longer binding, the limits are the number of slots the collectors can
afford (host memory, below) and the GRPO learner sharing the GPU.

## Collector host memory (2026-11-12)

Per slot: a `run_bonsai_queue.py` supervisor (~39 MB) plus a node harness process (`teacher/cli.js`): ~135-160 MB at
start (node, the bundle, an 18 MB TypeScript library source string and its code), 180-300 MB during an episode (median
224 MB, p90 357, max 617 over 112 live processes); the episode's own live heap is tens of MB (heap snapshot: 57 MB live
of which 18 MB is the TypeScript source). V8 flags (`--max-semi-space-size=1`, a 384 MB old space) did not move the
peak. 112 slots held 28 GB in node + 4.3 GB in supervisors.
`NATLANG_COLLECTOR_SHARED=N` (run_bonsai_queue): invocations run in shared node processes (`cli.js --serve SOCKET
--capacity N`, started on demand per supervising unit/runtime/model-facing variants, exit when idle); closing the
connection aborts that invocation only. Live test: 4 episodes in one process ~300-360 MB vs ~800 MB in four.

