# Latent optimizer bakeoff (2026-10-11)

Status: code merged (74f06ed4, c957cccc, 0889bace, 266a9350); Mellum arms queued behind v2c and the E2E smoke
(`natlang-bakeoff-queue`, log `/home/werg/data/optimizer-bakeoff-20261011/queue.log`); LFM check requested from Pop.
Results below are filled as arms finish.

## Decision rule (owner, 2026-10-11)

- Pick on learning per wall-clock hour (held CE reached per hour) within acceptable drift, not on drift alone.
- Owner tie-break: "If results are at all comparable, I would favor a cheaper Muon for both, since it is quite battle
  tested." If the cheap MuonSR is anywhere near comparable, it becomes the shared default for both Mellum and LFM
  (one implementation; only hyperparameters may differ, as declared backbone-inherent settings). Lion wins only if it
  is clearly better.
- Lean bakeoff: few arms, short runs, s/update next to everything.

## Provenance facts

| model | pretraining optimizer | fine-tuning optimizer | source |
|---|---|---|---|
| Mellum2 (Mellum2.1 base lineage) | Muon (Moonlight style): hidden matrices on Muon, embeddings and output layer on Adam. Momentum 0.95 Nesterov, 5 Newton-Schulz steps, update scaled 0.2·sqrt(max(m, n)), wd 0.1, grad clip 1.0; Adam β (0.9, 0.95), ε 1e-8; FP8 hybrid precision, Warmup-Hold-Decay | SFT also Muon, peak lr 3e-5, cosine to 3e-6 | Mellum2 Technical Report, arXiv:2605.31268, §3.4.1, §5.1.2 (abstract: "optimized with Muon under FP8 hybrid precision") |
| LFM2 / LFM2.5 | not stated in the LFM2 report or the LFM2.5 model cards (unverified) | SFT with AdamW β (0.9, 0.95), wd 0.1, lr 3e-5 | LFM2 Technical Report, arXiv:2511.23404; LFM2.5 model cards |

Fine-tuning with the pretraining optimizer forgets less and learns better (Moonlight Table 6; arXiv:2605.06654;
Qu et al., arXiv:2605.10468). For Mellum that argues for Muon, not an Adam-family optimizer.

Our setup vs Mellum2's Muon: Nesterov 0.95, 5 NS steps and the 0.2·sqrt(max) scale were already in MuonSR; added
cubic5 NS, wd as a recipe option (`latent_weight_decay`), and `split_rows` (Mellum's fused expert `gate_up` is the gate
and up projections that its pretraining Muon saw as two matrices). Not reproduced: grad clip 1.0 (the latents step
inside the backward, before the global norm is known; Newton-Schulz normalizes the step anyway), Adam β2 0.95 on routers
and norms (ours: AdamW defaults, lr 7.5e-6). Embeddings and the output head are frozen in the direct stage, so the
"Adam on embeddings/head" split does not arise.

## What changed in code

- LionSR (all paths: unfused, fused chunk, row_scale, step_in_backward): idle-expert skipping, default on (defect fix).
  Lion moves every coordinate by the full lr even where the gradient is zero, because the sign of a decaying momentum
  stays ±1, so an expert that was routed no tokens kept drifting at full rate. Now a matrix of a stacked latent whose
  gradient slice is exactly zero gets no update and no weight decay, and its momentum is left unchanged (lazy
  semantics; MuonSR instead lets an idle expert's momentum decay). 2-D latents are unchanged (same math, same RNG draws).
- Options (recipe keys of `neuralese_direct`; defaults keep the old behaviour and stay out of the resume identity):
  `latent_skip_idle`, `latent_cautious` (Liang et al., arXiv:2411.16085, without the `d/(nnz+1)` renormalization),
  `latent_optimizer: adaminisr` (Adam-mini, arXiv:2406.16793: BF16 m, FP32 v per head for q/k and per row otherwise,
  bias-corrected, SR writes, in-backward), `latent_beta2_half_life_tokens` (β2 = 0.5^(tokens·batch / half-life),
  Marek et al., arXiv:2507.07101), `latent_ns` (`quintic` | `cubic5`, arXiv:2606.00371), `latent_weight_decay`.
  AdamMiniSR is kept in the code, but no lineage uses it (owner: LFM gets the same cheap Muon).
- MuonSR: cubic5, `split_rows`, no weight decay on idle experts; under `muonsr`, matrices with a side < 16 (LFM's
  short-conv kernels `[d, 1, k]`) go to AdamW with the routers and norms.
- `direct_sft`: operational `stop_at` (end at update N while schedules such as the QAT ramps follow `steps`) and
  `weight_drift_base` (report-only latent drift against a base snapshot, per expert, with idle-update counts).
- ECO error feedback (arXiv:2601.22101) was not added. Its zero-memory form folds the rounding error into a momentum
  that the next update consumes linearly (SGD momentum). Our updates are sign (Lion) or Newton-Schulz (Muon) of the
  momentum, so the folded error is not applied as written, and stochastic rounding is already unbiased. It is not
  cheap to do correctly, so it was skipped (owner: "only if it's cheap").

## Newton-Schulz cost (Mellum shapes, GB10, BF16, measured while v2c shared the GPU)

| per update (28 layers) | quintic | cubic5 |
|---|---|---|
| fused gate_up [64, 1792, 2304] | 29.9 s | 22.6 s |
| gate_up split into [64, 2, 896, 2304] | 20.2 s | 17.9 s |

The expert stacks dominate (gate_up 0.38 s and down 0.21 s per layer with cubic5 + split; attention 0.05 s). The
measured s/update of the arms (below) is the real number.

Orthogonalization quality on a gradient-like spectrum (singular values 1 down to 1e-4, 256 x 640): median output
singular value after quintic 3/4/5 steps is 0.11/0.36/0.70, after cubic 3/4/5 steps 0.06/0.14/0.31. cubic5 inflates
the small directions less than quintic5 does, about like quintic4. With batch-1 gradients those directions are mostly
noise, so this is not obviously worse. Fewer cubic steps were not tried.

## Mellum arms

Setup: `neuralese-direct-mellum-v2` from the base model, the same data, seed and schedule as run v2c (2000 declared
steps, so no QAT point is active before update 300), stopped at update 300 (`stop_at`); drift monitor every 100;
final eval (held CE on 8 held documents) and behaviour gate at 300. Recipes:
`training/neuralese/recipes/optimizer-bakeoff-mellum-*-20261011.json`. Outputs:
`runs/optimizer-bakeoff-20261011/<arm>/`.

Update size: Mellum's mean ternary row scale is about 0.04 (q/k/v/o and expert matrices, measured on layer 10). The
per-element step is therefore 1.2e-7 for LionSR at 3e-6 (lr x row scale) and 2e-6 RMS for MuonSR at 1e-5
(0.2 x lr). "lion-matched" uses LionSR at 5e-5 (5e-5 x 0.04 = 2e-6).

Reference, LionSR as before the fix (run v2c, same first 300 updates): response KL 0.180 / 0.256 / 0.347 at
100 / 200 / 300, agreement 0.900 / 0.874 / 0.862, gate 0.83 at 100 and 0.75 at 200 (base 0.75), about 6.2 s/update.

RESULTS: pending.

## Pop LFM comparison prepared (17:30 UTC)

Declared diagnostic recipes `optimizer-bakeoff-lfm-lion-20261011` and
`optimizer-bakeoff-lfm-muon-20261011` extend the shared LFM direct-v2 base:
4962 union, 32k context, 256 supervised tokens/update, same seed and 8000-update
schedule, stopping at300. Lion uses current3e-7 row-scale units; Muon uses
cubic5/1e-5/weight-decay0.1 and shared narrow-matrix AdamW dispatch. These rates
are optimizer-specific units, not an assertion of equal update magnitudes.
The requested100-update KL measurements are diagnostic-only, not the cadence
for normal full runs. Full state is retained for resumability. No arm launched
while the current native epoch owns Pop GPU. Next window first checks native
crisp/raw-vs-TCP quality and serving restore, then executes the finite comparison.
No additional trainer implementation and no optimizer default change yet.

### LFM first arm (18:10 UTC)

Requested cubic5 Muon1e-5 stopped at100 on hard response-KL bound:0.53572
(bound0.5), responseagreement0.73070. Held subsetCE1.50535 vs base5.88885;
100 updates took133.9s (reported training loop wall, startup excluded), peak2.923GiB.
Checkpoint/negative results preserved under Pop optimizer artifact root; not
qualified and not a completed300-update comparison. Matched Lion3e-7 is running.
A declared gentle Muon5e-7 arm is prepared, based on earlier LFM V4 update20
KL0.01213 evidence before its owner-requested optimizer switch. Same data/seed/
schedule/guards; no quality bound relaxed. Requested Muon and Lion rates use
optimizer-specific units; failed high rate alone does not rank optimizer families.
