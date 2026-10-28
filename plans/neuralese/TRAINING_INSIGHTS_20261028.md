# Training insights from owner-supplied sources (2026-10-28) and how we apply them

Sources: Vuk Rosić's posts (tiny-model Muon experiments, pasted by the owner; X itself is not fetchable here) and
three papers: Dang, Wen, Malladi, *The Best Optimizer Depends on Batch Size* (arXiv 2610.08975); Marek, Ryabinin,
*Score Centering Stabilizes Off-policy Reinforcement Learning* (arXiv 2609.20807); Zhang et al., *Cortex:
Consolidating Context into Parametric Memory* (arXiv 2610.20399, ByteDance Seed). Tiny-model findings are starting
points to measure at 1.7B, not established facts for our scale.

## 1. Muon learning rate: what moves it and what does not

- Batch: lr x sqrt(batch ratio) for 2-4x changes (tiny models, widths 32-96; consistent with 2610.08975 in that range).
  For 8-16x, keep lr and raise weight decay instead. 2610.08975 also reports that no batch rule for Muon works
  consistently across settings: a rule gives the starting point, the development packet decides.
- Momentum: lr x ((1 - m_new) / (1 - m_old))^0.6 (exponent from tiny models).
- Decay length: longer decay tolerates a larger lr.
- Reported not to move the best lr: width, depth, sequence length, warmup, weight decay, data.
**Apply:** tune lr once at the local LFM3 geometry; DiLoCo learners keep the same per-learner batch, so the inner lr
transfers unchanged; any batch change uses the rules above as the initial value, then development measurement.

## 2. No pre-committed end: constant lr + EMA, anneal before release

LR decay to zero fixes the end step in advance; restarting after it loses part of the gain. Constant lr with weight
averaging nearly matches decay (Hägele et al. 2024; checked for Muon), and an EMA beat the plain average of the last
20% at every size/context tried. Practitioner caveat: constant lr + EMA does not give all of the annealing benefit;
anneal + EMA does.
**Apply:** production stages run constant lr (after warmup) and maintain an EMA of the weights (BF16 with
stochastic-rounding updates, per the no-full-precision decision; part of the checkpointed state). Evaluation and
selection read the EMA. A release/qualification checkpoint gets a short anneal branch from the current state with
EMA, without stopping the main constant-lr lineage. Under DiLoCo the coordinator keeps the EMA of global versions.

## 3. Muon grows every matrix; only Q/K growth hurts, QK-norm fixes it

Constant-size Muon steps grow all matrices 5-10x; Q·K growth sharpens attention (scores ~1 -> ~73) and cost sorting
accuracy, more at larger width. QK-norm removes the harm; V/O/MLP growth was harmless.
**Apply:** LFM3 already normalizes q and k (q_norm/k_norm) in both full and sliding attention, so the harmful path
is closed. Keep decoupled weight decay, and add telemetry: per-matrix norm growth (q,k,v,o,gate,w1-3,conv proj)
and max attention logit, to catch growth anywhere QK-norm does not cover (e.g. Neuralese-imported keys at consumer
positions, which pass through the importer's normalization path).

## 4. Per-head Muon: not for LFM3's main training

Per-head orthogonalization takes greedier steps (faster early loss), normal Muon catches up over long runs; it pays
mainly when the heads together are wider than the input (MLA). LFM3 attention is standard GQA (16 x 128 = 2048 =
hidden size). **Apply:** normal per-matrix Muon for q/k/v/o. Per-head on Q/K only is a candidate for short
program-tuning fits, measured there.

## 5. RL: score centering (2609.20807)

Instability under training/inference mismatch is mostly drift: a persistent trainer/sampler bias that accumulates.
Score centering is an additive per-token correction:

    L = -R ( sg[w_y] log p_y  -  sum_{v in H} sg[ q_v w_v - alpha p_v ] log p_v ),  alpha = rho f(1/rho)
    rho = (1 - sum_H q_v) / (1 - sum_H p_v),  H = the sampler's logged top-k tokens (k = 128; k = 32 matched full)

with q the sampler's and p the trainer's next-token probabilities, R the group-centered advantage, w = f(p/q) an
optional importance weight (f = 1 plain; TIS f = min(r, 2); MIS r in [0.5, 5] else 0). It needs only the sampler's
top-k log-probs per sampled position, costs <1% wall time, matched/beat importance sampling under quantized
samplers (int8, FP4 KV) and, combined with TIS/MIS, under staleness (up to 64 steps), 0.6B-30B.
**Apply:** adopt it as the shared whole-graph RL policy loss (every sampled position of every producer/consumer
turn), composed with TIS when rollouts are stale. Actors log top-k log-probs. This makes quantized rollouts usable:
actors can serve the deployed NVFP4/FP8 policy (on-policy for what we ship) while the learner trains BF16 latents,
and it makes the asynchronous multi-machine RL pattern viable (see LFM3_SCALEOUT_20261028.md phase 3).

## 6. Cortex: consolidating context into parametric memory (2610.20399)

Teacher = frozen model with the context; student = same model without it, training only new FFN parameters
(synaptic: low-rank increments on W_G/W_U/W_D; structural: k added SwiGLU channels; both fold into the FFN).
Loss: beta T^2 KL over the teacher's top-K vocab + alpha/(L d_h) sum_l ||h_S - h_T||^2 of post-FFN-residual
states, on aligned answer/action positions. Prospective rehearsal: the model generates queries anticipating uses
of the context; agent tasks use environment queries filtered to successful trajectories. Sequential updates freeze
earlier writes and replay earlier queries with regenerated responses; generic queries with the model's own answers
protect retention. FFN-only beat FFN+attention on consolidation and retention; structural expansion forgot less.
Metrics: consolidation rate CR = (Perf(M', Q+) - Perf(M, Q+)) / context gap; retention RR = Perf(M', Q-) / Perf(M, Q-).
**Apply (memetic learning / program tuning lane, after the LFM main-training hold lifts):** a program's
instructions and skill documents are the context; curriculum invocations of the program (training split) are the
prospective queries; teacher = served model with the full program context, student = without; FFN-only adapters
(synaptic and structural variants) and the hidden-state MSE term next to our teacher-SFT; CR/RR measured on
development and confirmation invocations. Note the interaction with NVFP4: LFM3's FFN is the quantized part, so
folded FFN writes must be re-quantized and measured at NVFP4.
