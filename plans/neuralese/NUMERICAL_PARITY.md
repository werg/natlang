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
direct-stage wrapper was subsequently checked; serving remained BF16. Probe SHA:
`f24014803970fc30823919bab3ea0c1e1ceaf11c56ed6483e09b3b5035dd4aa3`.

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

BF16 report (`te-full-model-pair-v4/bf16/direct_sft/report.json`) SHA:
`a54a55b05bd2abec0b689193b7763c8401355f14a60dbbf6095b5611a13932cb`.
FP8 report (`te-full-model-pair-v5/fp8-current-scaling-hybrid/direct_sft/report.json`) SHA:
`1e101496791babf5bb1599af2d0c302e05ca932b056cc2d7c915290d96a62636`.

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
