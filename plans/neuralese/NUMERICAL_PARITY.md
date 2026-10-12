# Neuralese numerical parity: current findings

Updated 2026-10-11. Pop owns these LFM2.5-350M experiments; DGX owns
Mellum execution. This tracks active correctness and performance questions.

## RoPE buffer corruption in diagnostics: cause and fix

Whole-module `model.to(dtype=torch.bfloat16)` casts floating buffers, including
RoPE frequencies. Casting back to FP32 does not recover their original values.
A diagnostic reused that mutated model across conditions and produced a false
FP32 discrepancy. Identical parameter hashes alone failed to detect it.

The production HF BF16 loader preserves FP32 RoPE buffers. Fresh production
loads confirmed that behavior; DGX added shared loader checks in `e2095698`.
Fresh FP32 comparisons agree closely (payload relative L2 approximately
2e-6 to 7e-6 on the measured lengths). This is not evidence of a production
HF-loading bug in our current paths.

`eval/cache_diagnostics.py` now casts parameters only, preserving loaded
buffers, and records buffer dtypes in each result. Its FP32 condition explicitly
promotes the same rounded BF16 parameter values to isolate arithmetic; it is
not a comparison with independently loaded full-precision weights. New
qualification comparisons load each condition afresh and fingerprint both
parameters and buffers. Superseded contaminated outputs are not certificates.

## Remaining BF16 execution difference: unresolved

Fresh production-equivalent BF16 loads, with identical FP32 RoPE buffers,
still differ between incremental and full-sequence execution. Measured payload
relative L2 is 0.033–0.038; consumer-logit relative L2 is 0.055–0.099 for fixed
lengths 8, 32 and 102. This is numerical evidence, not a demonstrated typed-task
failure or an accepted qualification result.

The measured convolution full/step operation is identical on the same inputs.
A single projection FP32-accumulation intervention worsened downstream
logit/gradient agreement and latency, so it is not an accepted fix. The remaining
hypothesis is amplified BF16 batch-shape arithmetic/rounding across layers.
Canonical fresh-load evidence is under the current V8 artifact's
`bf16-forced-sequence-dragnet-v23-native-loader/run-v1/`; its execution receipt
binds the exact artifact, script, image and report.

FP32 computation with BF16 transported payloads retains close execution parity:
payload transport rounding changes consumer logits by about 0.002 relative L2,
and the measured CE difference is about 0.002 nats. The corresponding control
is `fp32-bf16-transport-v24/run-v1/`. This separates payload storage precision
from backbone computation precision; it does not qualify the whole runtime.

## Mellum batch-row dragnet: shape-dependent expert GEMMs and routing flips (DGX, 2026-10-12)

Trigger: in the direct trainer smoke, two identical unpadded rows of one packed Mellum batch gave final hidden states
about 2.4% apart (BF16, 92.5% top-1 agreement). Probe: the first 4 Mellum layers (dense BF16 experts), real text at
512 and 1500 tokens, hooks on every op; row 0 against row 1 of one batch, and against a batch of one
(`runs/neuralese-direct-mellum-20261011/batch-row-dragnet/`).

- Input norm, q/k/v projections, attention output, o_proj and post-attention norm: bitwise equal in every
  comparison. The first difference is always layer 0's MoE: row 0 against row 1 about 1e-6, batch of two against
  batch of one 1e-4.
- The amplifier is top-k routing: after layer 0 the next layers' routers pick a different expert set for some tokens
  (layer 3: up to 9 of 512 tokens row against row, 41 of 1500 batch against batch), and the MoE output differs by
  3e-3 to 1.7e-2 relative from layer 1 on. That is consistent with 2.4% at the end of 28 layers.
- The source: a BF16 cuBLAS GEMM gives a row a different result depending on how many rows share the call (the
  kernel is chosen by shape). One Mellum expert weight, the same rows alone and inside twice as many rows: unequal at
  1, 37 and 64 rows (5.7e-5 to 7.5e-5 relative), equal at 7 and 129. A token's expert output therefore depends on how
  many tokens were routed to the same expert, which differs between a batch of one and a packed batch, and between
  a decode step and a sequence pass.
- Ruled out: `allow_bf16_reduced_precision_reduction=False` (no change); FP32 expert products (smaller, still 1e-4
  batch against batch: the NGC image sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, so FP32 products run in TF32 and are
  shape-dependent too, and the router's FP32 logits carry 3e-4 relative TF32 error); TF32 off through the torch flags
  alone (routing flips remained, from the expert GEMMs); torch's grouped GEMM (`moe_kernel` "grouped", equally
  shape-dependent); fixed-size cuBLAS blocks through `bmm` (row-invariant on the real model at 256-row blocks, but
  unit tests found shapes and batch counts where it is not, so it is not invariant by construction).
- Full FP32 with TF32 off (`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`): rows bitwise equal, batch against batch 1e-6, no
  routing flips. BF16 alone is not the problem; shape-dependent rounding plus discrete routing is.

Fix: `moe_kernel` "invariant" (`maple/fused_moe.invariant_dense_experts`, the new default of the MoE and of the three
trainers' `--moe-kernel`). Every routed row is computed by one Triton tile with a fixed configuration (64-row expert
blocks, 256 x 64 tiles, the k loop in one order), so a row's result depends only on its own input and its expert's
weights. The router's FP32 logits use the same kernel in exact FP32 (IEEE `tl.dot`), independent of the TF32 override.
The weight gradient is one grouped GEMM (it need not be invariant). On the same 4-layer probe at 512 and 1500
tokens, every op is bitwise equal row against row and batch against batch, with no routing flips. The tests compare
it with the loop (values and gradients, with and without precision points) and check identical outputs for one row
alone, beside a copy, beside 3 other rows, and in decode-sized calls.

Cost (one Mellum MoE layer, top-8 of 64, GB10): decode-sized calls are faster (1 token: 2.4 ms forward against 5.9 ms);
16k tokens forward and backward 630 ms against 467 ms for "grouped" and 1391 ms for "loop" (3k tokens: 156 against
129 and 302). Recipes that declare `moe_kernel: grouped` (Mellum direct v3) keep it; that choice now trades the
invariance for about 1.35x MoE speed.

Not covered: dense projections and attention at decode against sequence shapes. They were bitwise equal at 512 and
1024 rows here, but a single-row GEMM differs (above). The forced-sequence against incremental writer gap on Mellum is
to be re-measured with the invariant kernel (`eval/direct_forced.py`). LFM2.5 has no MoE, so Pop's LFM gap is not
explained by routing.

## Course of action and acceptance

The owner prefers lower precision and removing costly checkpointing where it
fits. BF16 remains the preferred training candidate; FP32 is the diagnostic
control. Measure actual shared optimizer updates, peak memory and throughput
with checkpointing on/off and packed update sizing. Do not infer training speed
from short inference timings, or adopt a blanket FP32 fallback.

Qualification must measure token/task behavior and gradient replay against the
exact selected weights and actual production execution path. Do not silently
relax gates because a numerical comparison failed. Distinguish expected
low-precision noise from state, position, cache and transport defects before
choosing any accuracy/performance trade-off. No new BF16 runtime certificate has
been issued by these diagnostics.

## First shared optimizer-update measurements

These are single-update fit measurements, not sustained throughput. The matched
16,384-token-budget updates have three windows, 13,440 forward tokens and 311
target-bearing tokens, using the shared MuonSR trainer.

| Compute | Activation checkpointing | Update time | Peak allocated VRAM |
| --- | --- | --- | --- |
| BF16 | on | 2.61 s | 2.87 GiB |
| FP32 | on | 5.01 s | 4.95 GiB |
| BF16 | off | OOM before update | exceeds available 8 GB GPU memory |

An uncheckpointed 4,096-budget update fits at 5.42 GiB / 1.51 s, but contains
only one window and six targets; it is not a comparable speed result. The next
optimization is per-window gradient accumulation with the original packed-batch
normalization and one optimizer update, if the windows are independent. Partial
checkpointing is another option. Neither change may alter the supervision or
silently turn every window into a separate optimizer step.

## BF16 window accumulation: speed improvement under qualification

A second matched update uses the corrected V2 program-header corpus: nine
windows, 411 target tokens and 1,394 supervised positions, with a 16,384-token
packing budget. Each condition starts from the same weights and seed and makes
one optimizer update with the same global loss denominators.

| Execution | Activation checkpointing | Update time | Peak allocated VRAM |
| --- | --- | --- | --- |
| Packed batch | on | 3.015 s | 3.073 GiB |
| Layer-staged windows | on | 2.967 s | 2.502 GiB |
| Sequential window backward, accumulated gradients | off | 2.244 s | 3.602 GiB |

Sequential backward retains the gradient buffers, releases each window's
activation graph before the next window, and uses the existing optimizer once
after accumulation. It is about 26% faster in this one-update measurement. The
layer-staged alternative retains layer graphs until its coordinated backward;
its lower measured peak does not imply that all activations are released between
windows.

The sequential and layer-staged conditions have the same reported forward loss
(4.235538), CE (4.720232) and value error (0.554313). The packed condition has a
slightly different loss (4.234075), since padding/batch shape changes low-precision
operations. Updated parameter comparisons also include stochastic rounding and
backward scheduling effects; they are not exact-gradient equality proofs.

Evidence is in the active V8 artifact's `window-gradacc-v31/` recipes, logs and
checkpoints. Sustained speed, gradient/update correctness and larger-window
memory policy remain under qualification. This does not resolve or qualify the
separate incremental/full-sequence runtime transport difference.

## Sustained BF16 continuation and memory-policy correction

The full-source continuation completed 627 updates through the shared declared V14 recipe
with BF16 backbone parameters, FP32 floating buffers and activation checkpointing.
At update 100, held CE improved from 3.57669 to 3.45925; response KL was 0.02170,
response agreement 0.99688 and the behavior guard remained clear. Value relative
MSE rose from 0.21104 to 0.22447, so the value channel needs separate monitoring.
These interim results do not qualify runtime transport, stopping or gradient replay.
Evaluation and resumable checkpoints occur every 100 updates for this continuation.

The automatic uncheckpointed attempt failed before its first update. Its memory
geometry omitted FP32 activations retained by LFM2 RMSNorm. The shared estimator
now counts those tensors, and logs the selected plan before a forward pass. A
bounded CPU saved-tensor inspection established one unique full-shape FP32 input
per norm; repeated hook entries were storage aliases. The actual model has 16
layers, width 1024, six attention layers and FFN width 4608. The earlier offline
4101-token first-window reconstruction was not verified against execution and
is superseded: V14's first recorded plan had 6807 context tokens.

V14 stopped cleanly at 627 updates. Held CE at 600 was 3.29795, while value
relative MSE was about 0.241; improved text loss does not qualify the value channel.
The corrected V3 header corpus preserves every supervised target and source group.
Its next phase ran 143 updates before a forward OOM at 5375 context tokens.
Uncheckpointed short windows took about 0.14 seconds/update; the automatic policy
checkpointed a larger window. The failing plan credited fragmented reserved memory
as reusable, leaving only about 58 MB of estimated headroom. Shared memory accounting
now subtracts measured native inactive split bytes from that cache credit and logs
allocator statistics. Continue in BF16 with selective checkpointing; these results
do not justify either full-backbone FP32 or blanket checkpointing. No checkpoint
preserved those 143 updates, so restart from the retained step-627 parent.

Continuation now declares hash-bound predecessor identity and monitor-reference
sidecars explicitly. Moving checkpoint files or resolving symlinks must not
change qualification provenance or silently reset the original drift reference.

## Exact trained V16 artifact: transport versus execution parity

The terminal V16 checkpoint is `d02969e6e8f62a34a833a0de8db3cce82b298b9b08e9a377965168edbf96599f`.
Its shared direct-runtime diagnostic at lengths 8/32/102 passed finite payloads,
cache/readback, exact producer replay and producer/input gradient replay; consumer
gradients reach the actual trained projection. Report SHA:
`6cffe8734a68f6610685cb89a650cbf3cf9b1a7a8f901a0cbedd2b8e296fbe53`.
This mechanism result does not qualify consumer task quality or autonomous stopping.

The separate BF16 forced-sequence versus incremental diagnostic still disagrees:
payload relative L2 is 0.0255/0.0486/0.0378 and downstream logits relative L2 is
0.0870/0.1986/0.1081 at those lengths. Combined gradients also differ substantially.
That diagnostic uses a synthetic token-choice projection only for its downstream
consumer; its CE and speedup are mechanism measurements, not trained-agent scores.
Report SHA: `cb866582337300a97bc2f265f11c2edbc50164f51e2904839a9f8a87c19e345f`.
Fresh independent restores confirmed identical unchanged parameters and buffers.
The first difference is a one-BF16-ULP `layers.0.conv.in_proj` Linear result with
identical inputs; disabling reduced-precision GEMM reductions did not change it.
Equal-input reference convolution and cache-window updates matched bit-for-bit.
The unavailable fused convolution path has not been checked. Ordinary crisp text
on the same token history already showed about 2.19% hidden-state and 1.87% logit
relative L2 drift, with identical next-token CE on the seven diagnostic targets.
Control report SHA: `1c1aa37b3fd118ef5a5e8540222e119eac4adec09476c22ebfca4218210e8383`.
These findings do not establish task quality, but do not support attributing all
BF16 drift to Neuralese transport. Keep BF16; next compare actual paired crisp and
Neuralese consumer/stopping tasks rather than require bitwise agreement or adopt
full-backbone FP32. The accumulation flag remains unchanged because its control
provided no improvement.

## Practical paired consumer check and precision policy

The exact V16 artifact's V8 paired check used eight math rows and an independent
reader given only a generic instruction and the generated value block, without
the original arithmetic question or gold answer. Crisp task accuracy was 2/8;
Neuralese writer plus reader accuracy was also 2/8. Reading a crisp generated
text block likewise passed 2/8, while reading the gold value block passed 8/8.
All writers and readers stopped naturally, with no truncations. On the six wrong
rows the reader preserved the writer's wrong value. This small set shows no
observed task gap; it does not establish general task qualification.

With identical prefix/block/suffix prefill call boundaries, gold `read_back`
and direct gold embedding inputs produced exactly identical final logits on all
eight rows. Earlier V7 comparator differences came from different prefill chunk
boundaries. V8 report SHA:
`1b7e2da4ed68cb92cb8df9bd496937edf6cd6530fa3da91ae8877cfcb1f53378`;
rows SHA: `cd515d4259ea8c9610849c56645b0bd599248bd5c749e7af48029040e0e1792e`.

The owner reiterated the preference for lower precision and avoiding recompute.
Use BF16 with measured adaptive checkpointing for the next substantive run;
forced checkpointing was temporary recovery. Ordinary BF16 shape rounding does
not justify a blanket FP32 conversion. An official source-built causal-conv1d
kernel is integrated in the shared loader, with the invalid Hub revision removed.
It remains disabled pending same-input CUDA forward/backward/update fidelity and
actual throughput qualification on Pop. Further precision reductions require a
supported implementation and evidence of a real training throughput benefit.

### Actual update and FP8 checks

Five matched, uncheckpointed BF16 updates from the exact V16 checkpoint used
context/target sizes 1023/256, 1027/4, 416/161, 1791/256 and 1818/27.
After the first cold update, median time was 0.17937 seconds with Torch convolution
and 0.17380 seconds with the compiled kernel. This small difference does not
establish a sustained training throughput gain. The apparent large first-update
speedup was dominated by startup. Both modes had the same initial loss; held CE
after five updates was 3.93810 versus 3.93852, and both drift guards stayed clear.
Maximum incremental allocation was about 1.906 GiB in both modes; run-wide total
allocation peaks, including setup/evaluation, were 3.664 and 4.743 GiB. Do not
report incremental allocation as total VRAM. Comparison SHA:
`905bab0168d3d828a078fb4dcd41a75688a26f0d940a03907aca93e754a03dc8`.

The isolated feed-forward FP8 CurrentScaling check used the exact V16 first
layer's actual 1024→4608→1024 geometry, original BF16 parameter objects, unchanged
SiLU gate and shared LionSR update. Transformer Engine 2.19 was already installed
in the pinned image. Five warm measurements gave 5.585→4.825 ms at 1024 tokens and
27.453→19.671 ms at 6144 tokens (about 14% and 28% faster). Output relative L2 was
about 0.043; gradient relative L2 was 0.073–0.088 with cosine 0.996–0.997.
Memory was unchanged at 1024 and slightly higher at 6144. These are one-layer
results, not whole-model speed or quality qualification. A diagnostic shared
direct-stage wrapper was subsequently checked; serving remained BF16. The unused
probe outputs and wrapper code have since been removed.

The shared whole-model wrapper was then exercised on ten identical scheduled
windows from V16 step 173, with checkpointing off: contexts 1023, 1027, 416,
1791, 1818, 955, 973, 2164, 1073 and 3071. Native BF16 evaluation was identical
before training. Warm median update time (steps 2–10) was 0.141 seconds BF16
versus 0.214 FP8; total update time including startup was 7.944 versus 9.066.
Only the longest window showed a clear FP8 advantage: 0.367→0.307 seconds.
After ten updates held CE was 3.934481/3.934944 and value MSE
0.438815/0.441736; both response guards passed. Run-wide allocation peaks were
4.907/5.156 GiB, so FP8 did not save memory here. TE requires flattened token rows
padded to multiples of 16 for these backward GEMMs; trim synthetic rows after
the token-local FFN, leaving real token positions and targets unchanged.

The discarded run directories are removed. Their measured comparison is retained
in `causal-conv-qualification-v1/fp8-ffn-precision-decision-v1.json`, SHA:
`5cf0763334fa1e0a2b4b1011ffc9549a20677636013aeb364497730cebb9df81`.

A real 6645-token/246-target row, repeated three times with checkpointing,
gave warm BF16 times 1.019/1.029 seconds versus FP8 0.906/0.918 (about 12%
faster), but a mixed-policy follow-up did not reproduce it: BF16 warm long
rows 1.030/1.032 versus FP8 1.703/1.679; even the short native BF16 branch
slowed down. Clocks, power and CPU pressure were not captured, so the cause
remains undetermined. Held response guards passed, with slightly higher FP8
memory. This does not establish a stable workload gain. The owner decision is
to remove both unadopted FP8 production options, prune unused probe outputs,
and continue BF16 with measured adaptive checkpointing. Retain these concise
findings rather than introducing an unused alternate training path.

### RMSNorm memory tradeoff and exact resume (2026-10-12)

A norm-local backward replay preserved the HF forward expression, retaining
the original input and weight instead of its FP32 intermediate tape. All 16
CPU and 16 CUDA local output/input-gradient/weight-gradient cases matched
bitwise across BF16/FP32 input and weight, strided inputs and frozen weights.
This did not establish whole-model gradient equality.

From the exact step 838 checkpoint, matched authentic-document probes at
contexts 1024/3072 with 128 targets gave identical forward outputs/predictions.
Incremental allocation fell 1.406→1.185GB and 3.521→2.994GB (about 15%). Warm
median times changed 0.12131→0.12183s and 0.34292→0.35242s, respectively. Model
gradient relative L2 differences were 0.55%/0.60%. Backward nondeterminism and
BF16 accumulation order have not been separated with an old-vs-old control.
The report is `rmsnorm-recompute-comparison-v1/wholemodel-report.json` beneath
the V18 output; SHA-256
`c60a0f6b7b3e9f0c26745c2066aeaffc227e59b8eccc2cdee89dacee3c8b55fa`.

The normal BF16 path resumed from step 838. The unadopted implementation was
removed. These probes were already uncheckpointed: they do **not** answer
whether memory savings near the fit boundary can remove whole-layer replay
and improve throughput. The later adjacent-context observations below exercise
the estimator at two longer uncheckpointed boundaries; they do not test the
RMSNorm replay intervention.

The container launcher now supports exact in-place resume and serializes
launches per output (`a63f0acc`, `939c470e`, `1956a2c9`). A new-corpus
`continue_from` phase is different: it starts its own schedule/baseline.
In-place resume restored update 828 and optimizer/RNG, then saved update 838.
The promised separate update 828 fallback was not copied before the rotating
slot was overwritten; its bytes are unavailable. The extra step-838 comparison
checkpoint was deleted after the diagnostic; only the report remains. Before
experimental updates, copy/verify one bounded fallback first; after deciding,
prune unnecessary copies. Do not retain every periodic checkpoint.

### Adjacent-context calibration boundary observations (2026-10-12)

The shared estimator's adjacent-context fallback was exercised during the
BF16-auto continuation from the verified step-1610 full-state checkpoint. Rows
at steps 1633 and 1655 used context lengths 4313 and 4607, target length 256,
and selected sequential execution with layer checkpointing off. Both estimates
used 72 samples from same-mode/checkpoint context-bin 11 target bins 7 and 8,
at q95 plus the existing margin. Forecasts were 5,487,257,167 and
5,809,589,319 bytes; observed incremental peaks were 4,775,798,272 and
5,594,913,280 bytes. Both rows completed without OOM. This is evidence for
these two estimator decisions only; it is not a global throughput, quality, or
model qualification claim. The compact digest-bound receipt, including exact
JSONL line hashes, is
[`measurement-receipt.json`](../../runs/neuralese-direct-memory-estimator-boundary-v1/measurement-receipt.json)
(SHA-256 `59fb277cff8bd2ba3ac979cd2ac9d8c3b285901e5ac18adc8646fd585c6511b1`).
