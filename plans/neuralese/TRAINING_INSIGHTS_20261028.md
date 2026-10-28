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

## 7. JOLT: one policy as privileged teacher and as student (Perplexity, "A good teacher meets the student where they are")

One shared policy in two roles: the teacher sees privileged information (verified answer, reference solution, execution
feedback, summaries of earlier attempts, private verifier info), the student only the task. Student update (OPSD):
student samples its own trajectories, the same model with the privileged context in place scores every visited token,
and the student minimizes reverse KL to that teacher (dense, on-policy). Teacher update: centered return of the
teacher's own attempts on log-likelihood under the privileged context, minus beta_T KL to the student's policy, so the
teacher's guidance stays within reach of the student ("reward-consistent" teacher, target pi_T* proportional to
pi_student exp(A/beta_T)). JOLT+ adds the ordinary centered outcome reward on student attempts. Reported: JOLT+ beats
tuned GRPO on GSM8K (+1.5), MATH (+4.2), LiveCodeBench v6 (+3.0), AppWorld (+9.9), Terminal-Bench 2.1 (+2.7, 0.6B-30B
Qwen/DeepSeek models); on MATH 6.5x fewer processed and ~13x fewer completion tokens than GRPO to the same accuracy.
Without the outcome term it trails GRPO on the agentic benchmarks. No beta values, no ablations, code not released.

**Fit with our work.** Strong:
- Our curriculum already carries exactly the privileged information it needs, for training worlds: authored reference
  solutions and child reference values, oracle checks, verified teacher trajectories of the same world, and the
  runtime's error/repair feedback from earlier attempts.
- It densifies reward where GRPO gives nothing: all-fail and all-success groups currently fall back to RFT seeding;
  the self-teacher still scores every token. It also reduces dependence on the external teacher (one Luna worker,
  Qwen3.6 sandbox) because the student's own model teaches with privileged context.
- It is the RL-time counterpart of Cortex (teacher with context, student without) and of our producer/consumer design:
  a teacher trained so that its guidance is usable by the student is what we want from Neuralese producers and
  stored guidance.
- Composes with score centering: the outcome terms (teacher attempts, JOLT+ student term) use the score-centered
  estimator; the OPSD term is computed by the learner's own forward with the privileged context, so it has no
  sampler mismatch.
**Requirements/risks:** privileged context only from training worlds (never held/dev); the teacher scoring forward must
run on the whole actual graph (privileged block added per invocation, consumer import positions handled by the shared
positional splice, gradients through producers/consumers) rather than on detached turns; leakage of reference text
into student behaviour is limited because the student only learns preferences over its own sampled tokens; beta_S,
beta_T untested at our scale.
**Plan:** implement as an option of the shared whole-graph RL learner (JOLT+ with score-centered outcome terms), pilot
on the development curriculum against score-centered GRPO at equal tokens, then use it in phase 3.

## 8. Softmax/cross-entropy numerics near saturation and near agreement (Barron et al., jonbarron.info/softmaxfix; owner-supplied summary figure by @brandondamos)

Two failure modes of standard log-softmax/CE implementations:
- **Softmax collapse (hard targets):** when the target logit leads by margin m, its gradient rounds to zero while
  non-target gradients stay positive, so the gradient no longer sums to zero. Cliff at m ~ 5.5 (bf16), 16.6 (fp32),
  36.7 (fp64). Fix: shift so the max logit is exactly 0, m = -LSE(non-target shifted logits), L = softplus(-m),
  g_j = e^{z_j - z_t}/(1 + e^{-m}) (j != t), g_t = -e^{-m}/(1 + e^{-m}); exact zero-sum down to exp underflow.
  Reported: removes grokking "slingshot" spikes (100.0% vs 94.3% test accuracy at the fp32 cliff; 85% vs 23% without
  weight decay); with AdamW + weight decay the margin settles near log(1/eps).
- **Softmax cancellation (soft targets, near agreement):** subtracting nearly equal probabilities loses precision, and
  forward softmax and backward log-softmax use different operation sequences, so equal logits still give a nonzero
  gradient (~1e-3 in bf16, ~1e-8 in fp32). Fixed gradients, with delta = z_S - z_T, p = softmax(z_T),
  q = softmax(z_S):
  - forward KL / soft CE: d = delta - E_p[delta]; r = d - log1p(E_p[expm1(d)]); u = p expm1(r); g = u - q sum(u)
  - reverse KL: g = q (delta - E_q[delta])  (centering under q cancels the log-normalizer before rounding)
  - generalized JSD (0 < beta < 1): h = -log1p(beta expm1(-r)); g = (1 - beta) q (h - E_q[h])
  Reported: a student initialized at the teacher drifts under standard soft CE in bf16 and has a ~1e-6 KL floor in
  fp32; the fixed version keeps KL = 0 in all precisions, without upcasting.

**Why it matters for us now:** the LFM3 line is BF16 everywhere, where the hard-target cliff is a 5.5-logit margin
(confident next tokens exceed that routinely unless logits are upcast), and our new objectives are near-agreement
by construction: JOLT's OPSD reverse KL (teacher = the same model with extra context), the JOLT teacher's KL to the
student, score centering's q_v - rho p_v coefficients, QAT multi-precision distillation and gates against the BF16
teacher, Cortex-style consolidation KL, the learned optimizer's distillation KL, and held-KL/drift telemetry.
**Apply:** one shared numerics module (hard CE, soft CE/forward KL, reverse KL, generalized JSD, score-centering
coefficients) with these gradients, verified against fp64 references, used by default in every trainer, RL loss and
distillation path, including chunked/fused CE paths.
