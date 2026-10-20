# Reuse readout CE after exact native prefix packing

Actual captured LF18754 training already shares prefixes: 10 ordinary graphs,
61,844 original tokens, 8,096 physical tokens, 53,748 reused tokens. The update
took 3.326s, corresponding to 18.6k original tokens/s and 2.43k physical tokens/s.
This is one sampled batch, not a run-wide average. A zero-reuse communication
batch does not show that prefix reuse is disabled.

The remaining repeated work was the full-vocabulary loss head: original
examples still projected the same physical prefix states separately. The shared
trainer now coalesces identical `(physical predecessor, actual target ID)` CE
terms after each original graph's normalization. The original graph count,
including duplicate trajectories, remains the averaging denominator. Different
branch target IDs stay separate. Existing zero masks and the source-bound CE
mass continuation remain in force. A single shared checkpointed CE kernel serves
both the unchanged path and this optimization.

Activation needs no new flag: a future reviewed runtime using the existing
`share_causal_prefixes=true` automatically takes this path during ordinary CE
updates with FP32 hidden/readout. Per-graph accuracy/held evaluation and all explicit policy/preference
objectives retain their old path. Imported/producing graphs remain intact and
retain their current packing. No cloud runtime was altered during development.
All model/optimizer/RNG/source/exposure continuation mechanisms stay unchanged.

The first64 retained admitted TRAIN rows per source yield these shape counts
under ordinary source-order 65,536-token packing:

| Source | Original CE rows | Coalesced CE rows |
| --- | ---: | ---: |
| LFM original Qwen V5 | 424,380 | 77,984 |
| LFM additional KEEP view | 426,733 | 80,268 |
| Mellum original Qwen V5 | 428,463 | 81,244 |

These are about81% fewer readout rows; 32,768-token packing saved about69%.
They are bounded actual source-shape counts, not production exposure, population
statistics or measured CUDA speedup. Source hashes and per-batch IDs are in
`/home/werg/data/dgxops/native-prefix-readout-shapes-audit-20261020.json`.
Exact metadata captures are retained in
`/home/werg/data/dgxops/native-training-execution-audit-20261020.json`.

CPU qualification covers both canonical tiny ports, every trainable gradient,
different branch targets, duplicate/internal-terminal examples, zero masks,
unequal per-graph weights, chunks of1/3/128 and the existing two actual MuonSR
updates with the strict momentum-cast oracle. Runtime progress now distinguishes
`readout_original_rows`, `readout_forward_rows`, and `reused_readout_rows`.
Actual tiny CUDA checks passed on both ports in production `bf16x6` arithmetic,
FP32 sparse rows/norm gains and the shared cotangent scope. The FP32 folded path
meets the existing loss/gradient tolerances. BF16 folding initially matched loss
but changed actual derivatives (Mellum maximum difference0.009765625) by moving
the per-use BF16 adjoint cast. That is a concrete arithmetic difference, not a
reason to widen tolerances. The final shared dtype guard retains original BF16
readouts and qualifies scalar/every derivative bitwise against the prior path.
Coefficient assembly uses fixed-order FP32 segmented reduction, avoiding CUDA
atomic addition. A separate declared FP32 readout-adjoint profile would be needed
before offering this reuse on BF16 hidden states; it is not silently enabled.
No new full-size model was allocated, and no production speedup is claimed.

Mellum's captured previous SFT24831 was one genuine communication graph with
38,777 physical tokens at1.84k tokens/s; it could not use ordinary prefix reuse.
Its activation scratch-file traffic is a separate opportunity requiring actual
largest-case headroom, not an excuse to disable offload. Current Mellum DPO is
a distinct objective and is not claimed to benefit from this CE optimization.
