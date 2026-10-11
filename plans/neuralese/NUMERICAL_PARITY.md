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
