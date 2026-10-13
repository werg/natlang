# E2E collapse dragnet (2026-10-12 onward)

Owner of this file: the DGX e2e lead (coord identity `dgx-e2e`). Goal (owner): a stable joint recipe in which the
projections (NeuraleseProjection = writer, TokenChoiceProjection = token readout) and the backbone train together,
held crisp CE does not rise, written-value CE improves, and the V−T gap shrinks toward or below zero. Freezing a
projection is a diagnostic only, never the fix (owner rule).

Setup of every arm unless stated: LFM2.5-350M, Pop's direct export u10006 (`runs/pop-lfm-muon-main-v2-export-step10006`),
`train/trajectories.py --direct-artifact` through the recipe launcher, 60 updates at batch 1, held evaluation every
10 updates on 32 paired held readers (selection sha `6d7b5c4d…`), one arm at a time beside Mellum v4. Runs and the
summary script: `runs/e2e-collapse-dragnet-20261012/` (`python3 runs/e2e-collapse-dragnet-20261012/summarize.py`).

Measures (all on held data): **crisp**, the held records' CE in their crisp rendering (every piece and value as text);
**V**, a paired consumer's target CE reading the producer's written Neuralese block; **T**, the same consumer reading
the producer's gold crisp text; **shuffled** and **zeroed**, the controls.

## Findings

### Round 0 (Pop, 2026-10-12): 200-update E2E arm collapsed

Held written CE 0.47 → 3.71, crisp 0.36 → 2.94; seen train readers fit (CE → 0). Two contributing causes found:
- **Sampler (ruled in, fixed):** the record order was the file order, so all 200 updates drew one family
  (`curriculum_decision_extract_chain`). Fixed with `--record-order family-interleave` (7d2feabc), and for corpus v2
  with the streamed CohortDocumentSampler (1b97df22).
- **TCP rate (ruled in, fixed):** see round 1.

### Round 1 (10 updates): the token readout's rate breaks token ability

Held crisp CE from 0.599: TCP frozen 0.600; TCP on Muon at the head rate 3e-4 (the recipe then) 0.768; AdamW 3e-4
1.01; AdamW 3e-5 0.698. The TokenChoiceProjection is the token readout (lm_head(norm(TCP(h)))) and trained in the heads
group at 3e-4, its matrices on Muon (fixed-size orthogonalised steps). New knobs `token_choice_lr`,
`token_choice_optimizer` (7d2feabc).

### Round 2 (60 updates): TCP at the backbone's latent scale keeps crisp CE

| arm | crisp 0→50 | V 0→50 | T 0→50 | V−T 0→50 |
|---|---|---|---|---|
| TCP Muon 3e-4 | 0.599 → 0.876 | 0.472 → 0.827 | 0.362 → 0.653 | 0.11 → 0.17 |
| TCP Muon 3e-6 | 0.599 → 0.547 | 0.472 → 0.559 | 0.362 → 0.387 | 0.11 → 0.17 |
| TCP AdamW 3e-6 | 0.599 → 0.539 | 0.472 → 0.591 | 0.362 → 0.386 | 0.11 → 0.21 |

TCP at 3e-6, jointly trained, keeps and improves held crisp CE: the TCP-rate collapse is fixed. V still rises.

### Soft-piece confound (hypothesis, ruled out by round 4 arm A)

In the evaluation code (`evaluate_written`), V scores the consumer with `soft_messages` plus the trained soft parameters,
T with `crisp_messages`, and in rounds 0–2 instructions pieces were trained soft prompts (`--soft-parts trained`, at
`lr` 3e-4). The held "soft" CE at step 0 is 0.896 against crisp 0.599, and it rose with V during training, so the
V−T widening could have been the soft prompts rather than the values. Round 4 arm A tested this: round 2's TCP Muon
3e-6 arm with instructions pieces as their text (`soft_parts text`, 1b97df22). Result: V at step 0 is the same (0.471
against 0.472), so V's rendering of the paired readers never differed, and V degrades just as before (0.471 → 0.617 at
step 50, round 2: 0.559). **Ruled out.** The "soft" label is the whole held set with every read written, not a
soft-prompt measure. (Round 3, queued on the pre-1b97df22 worktree, was cancelled for this hypothesis before it ran;
round 4 covers its arms.)

### Round 4 arm A: which readers degrade

Per paired held reader, step 0 → 30 (arm A): the 21 readers that both channels already solve (V0 ≈ T0 ≈ 0.01, the
decision extraction family) each lose a nearly uniform +0.08 to +0.19 nats on V, while their T moves 0.01 → 0.02.
One reader (67afce33 decision 0005) adds +1.2, and its T rises too (2.97 → 3.73). A uniform loss on readers that
were already perfect looks like a systematic perturbation of every written value, not learned misbehaviour.

### Where the writer moves (round 2 checkpoint against the u10006 export)

Relative change of the NeuraleseProjection's parameters after 60 updates (TCP Muon 3e-6 arm; heads_lr 3e-4, Muon):
the residual MLP moves 18–26% (`up.weight` 0.21 / 0.18, `down.weight` 0.26 / 0.19, block 1 `down.bias` 0.48),
while taps and norms move 0.2–0.5%. These weights were learned over the direct stage's 10,006 updates, and 60
batch-1 updates of consumer gradient move them by about a fifth: Muon's fixed-size orthogonalised steps at 3e-4 on
small matrices, driven by one record per update. (Script: proj_change.py in the e2e lead's scratch; the TCP in the
same arm moved 0.1% of its taps.)

### The writer before e2e: the direct stage never beat the mixture (Pop's eval history, 2026-10-13)

LFM direct v2 (the u10006 lineage), held value relative MSE to E[gold next] against the mixture's, per evaluation:
step 0 0.4212 / 0.4254 (delta −0.0042); 1000 +0.0017; 2000 −0.0000; 3000 +0.0005; 5000 +0.0007; 7000 −0.0030;
9000 +0.0001; 10000 0.2919 / 0.2886 (delta +0.0033). Nearest-token accuracy 0.559 → 0.666, but the mixture
improved by the same amount: that is the backbone learning, not the writer. Mellum v4 reached −0.039 at step 2000
(−0.20 at step 800).
- The stack's parameters did move: `down` (zero at init) has norm 1.47 / 1.75, `up` shrank from ~9.2 (a fresh
  `nn.Linear(1024, 256)`; the untouched TCP copy reads 9.23) to 5.37 / 5.97, the norm weights from 32 to ~29.2, and
  the taps from the identity (norm 32) to ~29.1. So it trained, but its output never improved on the mixture.
- Optimiser: direct SFT puts the writer in `PortMuonAdamW` at `head_lr` 3e-4. Matrices go to `torch.optim.Muon`
  (`adjust_lr_fn match_rms_adamw`, weight decay 0), including the taps (4×1024, Muon-eligible), whose first row is the
  identity path. Vectors go to AdamW (weight decay 0). The objective is `neuralese_weight` (1.0) × relative MSE to the
  gold next-token embedding, on detached h and mixture.
- So at e2e start the LFM writer is the mixture plus a correction that does not help. Round 4's "the writer moves
  18–26%" measures the e2e consumer gradient moving the stack away from where the direct stage left it.

### Owner decisions, 2026-10-13

- No value or manifold auxiliary in e2e (an arm with the direct stage's value objective as an auxiliary was proposed
  and cancelled before it ran).
- The soft-token mixture is removed. `dgx-projection` builds the replacement in shared code: value = learned P(h),
  fitted to E[chosen token] (nearest-token CE plus cosine/norm) on u10006 and v4, so e2e starts with V ≈ T. When P is
  fitted on u10006, the e2e arm reruns from it as the main e2e baseline.
- Arm D tests the owner's exact-token hypothesis now (consumers read convex blends of embeddings poorly): value =
  E[chosen] + correction, correction zero at start, so V = T wherever the chosen token is the gold one.

### Resume bug in the e2e trainer (found 2026-10-13, fixed on main c2e88dcb)

A direct-artifact e2e run could not resume its own checkpoint: `resumed backbone parameter mismatch:
model.layers.0.conv.in_proj.weight`. Checkpoint names and shapes equal the base model's (146 trainables, 0
mismatches), but with the quantisation component installed the live weight is
`...in_proj.parametrizations.weight.original`, and `trajectories.py` looked names up in plain `named_parameters()`.
Fix: the resume uses the shared `backbone_policy.restore_backbone_trainables` (plain names through
`plain_named_tensors`), as the other trainers do. Confirmed in a real run: arm B, stopped by a preemption after its 60
updates, resumed on the new code from its step-60 checkpoint and wrote its final report.

### Streamed e2e could not start from the HDD corpus (fixed on main 27de5139)

The recipe launcher stages HDD-resident data inputs to the NVMe cache, but only the bound file. `--record-source
trajectories` then refused the staged `records.jsonl` for lack of `chains.jsonl` beside it, so no streamed e2e run of
neuralese-e2e-v1 (both lines) could start from corpus v2. `hdd_guard` now stages declared sidecars with the input.

### Arm D, step 0: the exact-token hypothesis does not explain V−T (2026-10-13)

Owner hypothesis: the model reads convex blends of embeddings poorly (the embedding space is not a VAE space), so the
mixture writer's values cost consumers. Arm D writes value = E[chosen token] + correction, with the correction reset
to zero and the chosen token the argmax of the TCP readout (the token the writer emits). Step 0, before any update,
the same 32 paired held readers:
- chosen = gold at 92.6% of the 1,875 held written positions (96.3% on train readers);
- held V 0.4675, T 0.3622, V−T 0.105; the mixture writer at step 0: V 0.471, V−T 0.109; train readers V−T 0.023;
- per reader: 23 of 32 have V ≈ T within ±0.006, which is the identity check where tokens match. The gap sits in 9
  readers: 4ca467ab decision 0028 +1.45 (alone 0.045 of the 0.105 mean), f365cb4a 0003 +0.54, 1f0d9235 0005 +0.44,
  1f0d9235 0006 +0.29, 39cdded0 0029 +0.29, 39cdded0 0028 +0.16, 67afce33 0004 / 0005 +0.09, f365cb4a 0005 +0.06.

Reading: at step 0, convex blends cost almost nothing beyond the chosen tokens (0.109 against 0.105). V−T at step 0 is
the writer's own token errors. T reads the gold text, an oracle; V carries what the writer would emit, and a few
readers hinge on the tokens it gets wrong. So gold-T is an upper reference that the writer's argmax cannot reach, and
V ≤ gold-T needs values that carry more than the argmax token, which the consumer gradient in e2e must teach. The
writer's own greedy text (greedy-T) is the fair runtime comparison, and V(chosen) ≈ greedy-T is expected at step 0.

### Round 4 (running): instructions as text

Code: 1b97df22 (worktree `/home/werg/data/wt-e2e-lead`); recipes `diag-e2e-lfm-u10006-r4-*`; v1 union cohort, the
same order, seed and held readers as round 2.

| arm | crisp 0→50 | V 0→50 | T 0→50 |
|---|---|---|---|
| A `r4-text-heads-3e-4` (writer 3e-4) | 0.599 → 0.544 | 0.471 → 0.617 | 0.362 → 0.387 |
| B `r4-text-heads-3e-6` (writer 3e-6) | 0.599 → 0.565 | 0.471 → 0.529 | 0.362 → 0.422 |
| D `d-chosen-heads-3e-4` (E[chosen] + correction, writer 3e-4) | 0.599 → 0.544 | 0.468 → 0.531 | 0.362 → 0.390 |
| PH `r4-ph-heads-3e-6` (program-header v1.1, writer 3e-6) | 0.443 → 0.469 | not measurable (below) | not measurable |

Arm D, writer 3e-4: V−T 0.106 → 0.141, the same picture as arm A (0.109 → 0.230). The writer rate, not the value
base, drives the degradation. Arm D at writer 3e-6 is skipped: arm B covers the stable rate.

PH (program-header v1.1, writer 3e-6): held crisp CE starts 0.156 nats lower than the full rendering (0.443 against
0.599). The parent export was trained in program-header, while e2e ran in full. Under the same updates it drifts up
slightly (0.443 → 0.469), where full rendering improves (0.599 → 0.565). V and T are not measurable in PH:
`select_paired_held` proves a reader pair through the producer's function-body soft IDs and its tool-argument schema,
and the program-header rendering drops both (pieces resolved to text, tools `[]`), so 0 of 453 reader rows are
eligible. Fix to make: pair on the full-rendering twin (v1.1 has v1's record ids and re-authenticated equal edges)
and evaluate in PH.

Arm B: with the writer at 3e-6, V−T stays flat (0.109 → 0.107 at step 50, 0.108 at step 60: V 0.510, T 0.402;
arm A: 0.109 → 0.230). The writer rate
is what widened V−T. **Ruled in.** Two things remain:
- the writer at 3e-6 hardly learns, so V does not improve either;
- T rises on the paired readers (0.362 → 0.422) while held crisp CE falls (0.599 → 0.565). Per reader, the 18
  readers both channels already solve (T0 < 0.05) lose a median +0.06 on T and +0.08 on V, and one reader (67afce33
  decision 0005) adds +1.2 to both. The backbone and TCP updates, trained on 60 decision records, cost the paired
  readers on both channels. With 32 readers, this needs a longer, broader arm (corpus v2) to judge.

### Round 5: the writer on AdamW at 3e-5 (hypothesis 2)

`r5-writer-adamw-3e-5` (`--neuralese-projection-optimizer adamw`, writer 3e-5, TCP Muon 3e-6, the same 32 paired
held readers): V 0.471 → 0.595 at step 10, 0.557 at 20, 0.568 at 30; T 0.362 → 0.382; crisp 0.599 → 0.569. V−T
0.109 → 0.186 at step 30. AdamW does not rescue a faster writer: the near-identity writer degrades consumers on
either optimiser once its rate is above about 3e-6. **Hypothesis 2 ruled out** at this rate. These arms used the
soft-token mixture writer, which main has replaced by value = E[chosen] + C(h) with C = 0 at entry (fc32044e,
dgx-projection); the next arms run on it from the aligned u10006 P.

### Corpus v2 arm could not start: Neuralese<HandoverNote> writers (fixed on main 47617d77)

The streamed corpus v2 arm stopped at startup in `write_value_type`: 46 `compact_history` handover writes in v2 are
typed `Neuralese<HandoverNote>` (the compiler's `HANDOVER_TYPE`, a crisp string note; corpus v2 declares 46,983
`Neuralese<string>`, 2,221 `Neuralese<unknown>` and these 46). A named Neuralese type is now a string write when it is
the handover type or its argument schema is a string; any other named type still fails, with the type in the error.

## Checked and ruled out (with how)

- **Context CE over the written block:** `GradSession._context_logp` never targets block payloads; the token after a
  block is predicted from the block's close (serve/grad.py). Consumer reply CE and context CE are each token means,
  weighted 1:1 (`context_weight` 1.0, feedback 0.25).
- **No gradient into the backbone from consumers:** the shared writer (`serve/direct_write.py`, `decide` and
  `forced_sequence`) computes the soft-token mixture with gradients; only direct SFT's `soft_token_mixture` is
  `no_grad`, and it is not used by the e2e writer. Consumer CE reaches the producer's backbone through the mixture and
  the correction.
- **Train loss near 0 is memorisation:** no. Each update draws a new record (60 distinct decision records in round
  2), and the low per-record losses (0.004–0.1) are short, easy targets of `decision_extract_chain`; held V moves
  little before step 20.

## Open hypotheses (in order)

1. Writer rate: ruled in by arm B (V−T flat at writer 3e-6). Evidence: the NeuraleseProjection moves 0.024–0.040 per update on Muon at 3e-4, about 100× the TCP's step,
   and its zero-initialised `down` matrices get fixed-norm orthogonalised steps regardless of gradient size (Muon on a
   near-identity residual). Round 4 arm B.
2. Per-group optimiser scale for the near-identity projections (taps and residual MLP): AdamW at a small rate, or
   Muon with a scale relative to the parameter norm.
3. The gradient through the mixture: consumer CE moves the producer's base readout distribution, which is also the
   token readout while TCP is near the identity.
