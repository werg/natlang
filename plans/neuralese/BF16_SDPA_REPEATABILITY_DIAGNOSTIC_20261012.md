# BF16 attention and repeatability diagnostic

This is a declared follow-up diagnostic, not a qualification result. It uses the
shared `neuralese_direct` trainer and its same optimizer, window scheduler,
masking, projection loss, and full-state checkpoint format. The diagnostic does
not by itself qualify a production BF16 training run.

## 2026-10-14 Pop results and course correction

The native BF16 same-seed controls have now run in the pinned
`natlang-neuralese:26.09-pop` image. At 2,048 context tokens, both default and
math attention repeats started from identical parameter, optimizer, and RNG
hashes. The default repeats disagreed in 109/146 first-update named gradient
hashes and in their post-update parameters. The math repeats matched all
146 gradient hashes, post-update parameters, and the complete diagnostic JSON
byte for byte (SHA-256
`a5b8c2938983e97560ded41cc19523b8b58c86a1c8248dfd0418ae6018e1ad67`).
After two updates, default A/B differed in 106/146 parameter tensors, with
maximum absolute element difference `0.0009765625` and RMS across changed
elements `2.56346e-6`; math A/B remained exactly equal. Held CE was 4.47050
versus 4.47676 for default A/B, and 4.47522 for both math arms. This shows
a real repeated-run discrepancy, but its two-update size does not explain
the long QAD behaviour result on its own.

The 6,144-token math control OOMed on Pop's 8 GB card during a 2.25 GiB
attention-backward allocation, so a blanket switch to math is not viable
there. A standalone BF16 causal SDPA probe in the same image showed that
`torch.use_deterministic_algorithms(True)` makes the default Flash backward
bitwise repeatable at both 2,048 and 6,144 tokens. Its profiler still named
`aten::_scaled_dot_product_flash_attention`, so it did not silently switch to
math. A complete-trainer paired repeat with the shared direct trainer's
deterministic setting completed: all 146 first-step gradient hashes, post-step
parameter/optimizer/RNG hashes and the complete diagnostic JSON matched
exactly (SHA `8d04109f7fbb80bf1c22a47d0e4783e9b9e9ecca35287f743e986d2b654d9f66`).
The two full-state final checkpoints were byte identical (SHA
`dbe78f8d4f6de6cfb00bf3944a12c8969d56ccc06ccf391fe126145d4487fb87`),
with both held CE values `4.4751035372`. On the second, warm update the
deterministic Flash runs took 0.792/0.797 seconds versus 0.756/0.760 for the
old default repeats and 1.091/1.091 for math. These two updates suggest a
small steady cost, not a throughput qualification for a full training run.

The code now sets the cuBLAS workspace before CUDA work and enables PyTorch's
deterministic algorithms for the shared direct trainer. A complete-model
6,144-token two-update run also finished on Pop without OOM or unsupported
operation. Its second update took 2.288 seconds and peaked at 2.955 GiB
reserved by the CUDA allocator, versus 2.161–2.176 seconds and the same
2.955 GiB reserved for the old default repeats. This is roughly a 5% warm
update cost with no measured peak-reservation increase in this one geometry.
It still needs the full post-change gate before publication. The paired
recipe and receipts are under
`/home/werg/data/bf16-deterministic-direct-20261014-v1/`, and the long
geometry run is under `/home/werg/data/bf16-deterministic-long-20261014-v1/`.

Pop retained recipes, logs, reports, and first-step hashes under
`/home/werg/data/bf16-sdpa-short-20261014-v1/` and the longer default repeats
under `/srv/storage/natlang-artifacts/bf16-sdpa-repeat-20261014-v1/`.
The six completed comparison checkpoints and their six same-inode best links
were discarded after exact comparison, reclaiming 6,898,858,146 bytes.

## Existing TE repeat evidence, not native BF16 repeats

At the time this diagnostic was written, the only pinned same-seed repeat supporting the gradient-drift report
is under
`/srv/storage/natlang-artifacts/fp8-lfm-paired-diagnostics-20261012-v1/run-v6-te-repeatability-2step-v1/`.
Its repeat plan SHA-256 is
`5d7e1b3d304113e3e01a9a46249b558b2d6e80583330a16f11688b071d885221`.
The exact shared input subpack is
`/srv/storage/natlang-artifacts/fp8-lfm-paired-diagnostics-20261012-v1/input-32k-r2-subpack-v1/text-data.jsonl`
(SHA-256 `825fc187cf71320f4864087a3b910def36234797d548aa790ffc14147b5ce4d7`);
the clean-base heads are
`/srv/storage/natlang-artifacts/luna-neuralese-direct-clean-base-monitor-20261011-v2/artifact/clean-base-untrained-heads.pt`
(SHA-256 `3853846b8d69798382fb3b6341aed8b1ac6277ba7eea795ec460c19d1c208fe1`).
Their `train.jsonl` SHA-256 values are:

- A: `df057fdabb6e34afd946710e2c71881a6edfd2b97b93494ba979490175fe611e`
- B: `bb395db709b73b4f576fc3a49161f2067e0d7139200851b2ff90b3b52916124a`
- CPU checkpoint comparison: `1f8a854162d49b9b56de7d7d3ab7a460ac24421847e0100c24822d64da5cae30`

The checkpoints are A `e4a60cbf37f89bdd33a1c8676cfdf113d793cd4987b7e7b881c1c83f60c2b3fe`
and B `e567a377e227aae0ce73774fc868252a2cfdae1b68287bdc0db95169e9ed4ff4`.
These runs used BF16 master parameters **with TE FP8 DelayedScaling enabled**;
they are not a native BF16-only repeat. No native BF16-only same-seed repeat
was pinned then. At update 1, scalar losses matched
(`2.1158322096`) and the existing 158-entry gradient audit differed: median
relative L2-norm delta `1.539e-5`, p95 `1.13%`, maximum `6.55%` at
`backbone.model.layers.0.ffn_norm.weight`. After two updates, 106/146 student
parameters, 12/12 projection tensors, 218/278 optimizer entries and 92/92 TE
scale-state entries differed, while the recorded RNG state matched. This
establishes a TE-enabled repeated-run discrepancy but does not localize its
cause or prove native BF16 has the same drift. The
existing audit stores norms and maxima, not exact elementwise gradient hashes.

## Next controlled comparison

Prepare two native-BF16 arms from the same step-zero state and identical
declared training input, seed, window order, optimizer configuration, update
count, and runtime package. One declares `sdpa_backend: default`; the other
declares `sdpa_backend: math`. Use the same short authentic windows and the
same multi-window sequential update geometry in both arms. Both arms enable
`gradient_digest_steps: 1` and `state_hash_steps: 1`. The first completed
update writes `diagnostic-step-000001.json` containing SHA-256 per named
gradient tensor, per named trainable parameter before and after the update,
optimizer state before and after, CPU/CUDA/Python RNG digests, and TE scaling
state digests when TE is enabled. Training defaults and recipe identity retain
the chosen SDPA backend; the audit switches are operational diagnostics.

Compare gradient hashes first, then parameter and optimizer-state hashes.
Record scalar loss/CE, exact update windows, and backend in the same receipts.
The math arm changes the SDPA implementation only. It is not presumed to be
more deterministic or more accurate; it isolates whether the default SDPA
backend contributes to same-seed gradient drift.

After the native-BF16 comparison, a separate TE-only localization may compare
`fp8_recipe: delayed` with `fp8_recipe: current`, again recording per-tensor
gradients and scaling-state hashes. DelayedScaling's first-update amax history
calibration is a testable hypothesis for TE repeat drift, not an established
cause. Do not attribute the current discrepancy to kernel nondeterminism,
scaling calibration, or SDPA until those controls distinguish them.
