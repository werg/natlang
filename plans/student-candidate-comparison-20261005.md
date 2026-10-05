# Student candidate comparison — 2026-10-05

## Completed comparison

Same 23 eligible protected tasks, frozen runtime and gold/source policy, greedy
sampling, 16K context, 1024 output tokens, eight turns and 16 requests per case.
One additional case was held by policy. Both candidates are untuned.

| Candidate | Passed | Semantic failures | Contract failures | Incomplete | Total case time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Maple 20B-A1B, TQ2_0/F16 head | 12/23 (52.2%) | 8 | 0 | 3 | 14m 33s |
| Ling 7.9B-A1.3B, BF16, native thinking | 11/23 (47.8%) | 2 | 1 | 9 | 19m 05s |

Seven tasks passed both; five passed only Maple; four passed only Ling; seven failed
both. The one-case difference does not establish a superior student. Ling's
completed-only accuracy must not hide its incomplete tasks. Our strongest evaluated
350M SFT checkpoint also passed 11/23, but it has been trained, unlike these two.

Maple used our llama.cpp fork on DGX CPU (eight decode/sixteen prefill threads).
Ling used native vLLM on DGX GPU, eager execution, explicit 2 GiB KV cache. Task
latency includes tool work and generated reasoning, not just raw decode speed.

Receipts:

- `runs/maple-preview-evaluation-20261005/execution-v5/report.json`
- `runs/ling-maple-comparison-20261005/execution-v2/report.json`
- `runs/ling-maple-comparison-20261005/comparison-v1.json`

## What the failures suggest

Ling's 114 streamed completion responses included 37 ending with `length`. Seven
incomplete cases reached the eight-turn budget; two ended with malformed tool JSON.
Some invoice/trace cases spent tens of thousands of reasoning characters without
finishing. The unchanged older protected runtime predates our truncated-tool-call
handling improvement; the current transport already treats a cut-off action as
truncated rather than retrying it as malformed. Keep the baseline evidence intact.

Both models failed the two TAT-QA file/evidence/answer contracts. Ling also edited a
CommitPack replacement by retyping it and losing the replacement's final newline.
The supplied edit request was explicit: parsing its JSON fields and applying them
verbatim avoids this. Preserve exact file contracts; do not relax them to inflate
scores. These are candidates for general contract/structured-input skills.

Maple's remaining failures include mistaken semantic judgments, result key/format
mistakes and an invoice case confusing supplied evidence with the host transcript
API. Full Maple v5 decisions were not saved by the original evaluator, so detailed
causal claims need separate diagnostic captures. The evaluator now saves full rows.

## Formatting and infrastructure fixes

- Native Ling history needs modern vLLM's `reasoning` field. Our runtime used
  `reasoning_content`; carry both exact aliases. Current transport is fixed. The
  frozen comparison uses a pinned boundary bridge retaining exact wires.
- Ling template proof passes all 24 first requests against independent publisher
  template/tokenizer rendering. All 32 BF16 shards matched their HF blob hashes.
- Spark unified-memory reclaim can make free GPU memory increase during vLLM's
  startup profiler. An explicit 2 GiB KV budget avoids the failing profiler.
- Qwen restoration is automatic after experiments, but `docker start` does not mean
  ready for inference. Repair now checks deployment readiness independently of task
  execution limits. The initial short queued wait expired; restarted repair is active.

## Performance interpretation

Publisher-build Maple CPU benchmark measured DGX 94.56 decode tok/s at eight
threads. An actual-fork benchmark measured 58.42 while Ling inference was active.
A later sequential stack/configuration check measured publisher30.71, fork-auto31.10,
fork-resident94.61 and fork-no-repack86.93. Background workloads changed during the
sequence; this does **not** isolate a lazy-loading effect or a fork regression.
It establishes that our fork can reach about95tok/s in one measured configuration.
Treat shared CPU/GPU memory bandwidth and concurrent work as possible confounders.
Do not replace a serving configuration based on these ordered measurements alone.

## Next experiment and decision

Native Ling thinking-disabled comparison is queued after the current repair batch:
`natlang-ling-no-thinking-evaluate-v3-20261005`, same tasks and resource limits,
`runs/ling-maple-comparison-20261005/evaluation-plan-v3.json`. This tests whether a
practical operating mode completes more tasks with less compute. It restores Qwen.

Before choosing a target student, inspect that result and broader semantic/skill-use
coverage, then weigh training memory/throughput and the already integrated Maple
QAT/neuralese work. No student switch has been made from this small evaluation.
