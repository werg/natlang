# Latent optimizer bakeoff (2026-10-11)

Status: **decided 2026-10-11: MuonSR is the shared latent optimizer for both lines** (neuralese-direct-v2;
Mellum latent_lr 1e-5, LFM 3e-6). Code: 74f06ed4, c957cccc, 0889bace, 266a9350, 6070232a. The remaining Lion arms
were cut (owner: training time over a precise comparison; coordinator: skip them once Muon passes the gate).

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

### Results

Mellum, same data order as v2c for updates 1-100 (train CE is on the same windows there; mean over each range):

| updates | LionSR 3e-6 (v2c, before the idle fix) train CE | MuonSR 1e-5 cubic5 wd 0.1 train CE |
|---|---|---|
| 1-25 | 2.229 | 2.005 |
| 26-50 | 1.910 | 1.437 |
| 51-75 | 1.715 | 1.098 |
| 76-100 | 1.571 | 0.847 |
| 201-300 (Lion only) | 0.99 | — |
| 301-400 (Lion only) | 0.81 | — |

| arm | s/update | wall to update 100 | held CE subset (4 windows) at 100 | response KL / agreement at 100 | gate at 100 (base 0.75) | CUDA peak |
|---|---|---|---|---|---|---|
| LionSR 3e-6 (v2c) | 6.2 | 10 min | not measured | 0.180 / 0.900 | 0.83 | ~93 GB ledger peak (whole run) |
| MuonSR 1e-5 | 18.8 | 31 min | 0.832 (base 2.285) | 0.304 / 0.867 | 0.75 | 55.6 GB allocated |

- Per wall-clock hour MuonSR 1e-5 is at or slightly ahead of LionSR 3e-6. Muon's train CE at 100 updates (31 min) is
  0.85; Lion's is 0.99 at updates 201-300 (21-31 min) and reaches 0.81 only at 301-400 (31-41 min). Muon costs 3x
  per update, and almost all of that is Newton-Schulz on the expert stacks.
- MuonSR 1e-5 passes the behaviour gate (0.75 = base) at update 100. The arm was stopped after the 100-update drift
  check to save GPU time.
- LFM (Pop, 300 updates): MuonSR 3e-6 took held CE from 6.19 to 1.05 against LionSR's 5.10 at the same budget, with
  behaviour 26/48 = base and response KL 1.16 on the final 48 rows. MuonSR 1e-5 took held CE subset from 5.89 to 1.51
  in 100 updates, then hit the old 0.5 KL stop (KL 0.54, agreement 0.73). 1.34 s/update.
- Weight drift per expert was not captured for muon-1e-5: the report crashed on MuonSR's missing idle counts, fixed in
  6070232a.

### Decision

Muon is comparable on Mellum and clearly ahead on LFM. Under the owner's tie-break (a cheap, battle-tested Muon for
both lines when comparable) MuonSR is the shared default in `neuralese-direct-v2`: cubic5, wd 0.1, idle experts
skipped, `split_rows` for Mellum's fused gate_up. latent_lr is declared per line as a backbone-inherent setting in
`line-parity.json`: Mellum 1e-5, LFM 3e-6 (the shared base value). In the same change the drift policy moved:
the behaviour gate (≥ base − 0.1) is the hard criterion, response KL warns at 0.5 and stops at 1.5
(LINE_PARITY.md, NEURALESE_DIRECT_V1.md, AGENTS.md).

Cost caveat: on Mellum MuonSR is about 18.8 s/update against LionSR's 6.2 at 8k context. At 32k context
(mellum-v3, about 10.4 s/update with Lion) expect about 22 s/update, so 2000 updates take about 12 h. Muon learns
more per update, so a shorter run could reach the same held CE. Further cheapening options: Newton-Schulz on the gram
(about 20% fewer FLOPs), or fewer steps (they lose orthogonality, see above).

LionSR keeps its idle-expert fix and stays available (`latent_optimizer: lionsr`). AdamMiniSR stays in the code
unused.

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

### Follow-ups (queued, not blocking mellum-v3)

- Move the E2E stage (`trajectories.py`, `latent_optimizer`/`qat_latent_lr`, line-parity e2e base config) to MuonSR.
- Reproduce Mellum2's grad clip 1.0 for in-backward latents: clip each latent's gradient inside the hook by a running
  estimate of the global gradient norm (the true norm is known only after the backward).

Run length (coordinator, 2026-10-11): the shared supervision budget is 768,000 positions (about 2 epochs): Mellum v3
750 updates at target 1024 (about 4.6 h at ~22 s/update), LFM 3000 at target 256.
