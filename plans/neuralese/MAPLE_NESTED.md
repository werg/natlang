# Nested Maple: a 1–1.5 GB Maple inside the full one

Written 2026-10-05 after the owner chose a pruned Maple as the small partner (MAPLE_QWEN_JOINT §6 J7), sized to
about 1 GB (at most 1.5 GB) after ternary export, MoE kept, and asked how much of the connection to the full model
can be kept. Joint training, distillation and the shared Neuralese space follow
[MAPLE_QWEN_JOINT.md](MAPLE_QWEN_JOINT.md); this document replaces its "Qwen3-0.6B" with a nested Maple.

## 1. Size

Maple's file is dominated by its experts: 24 layers × 256 experts × 3.15 M weights = 19.3 B of the 20 B weights. Per
layer, attention is 10.5 M weights. The embedding and output head are 311 M weights each (151,936 × 2048).

Ternary `TQ2_0` costs 2.06 bits per weight. Embedding and head formats: `Q4_K` 4.5 bits (lookup table, tolerant),
`Q6_K` 6.56 bits, `Q8_0` 8.5 bits, F16 16 bits. Router rows F32.

| Experts kept per layer | Ternary weights | Ternary part | + embedding Q4_K, head Q6_K | + embedding/head F16 |
| --- | --- | --- | --- | --- |
| 24 | 2.06 B | 0.53 GB | **0.97 GB** | 1.78 GB |
| 32 | 2.67 B | 0.69 GB | **1.12 GB** | 1.94 GB |
| 40 | 3.27 B | 0.84 GB | **1.28 GB** | 2.10 GB |
| 48 | 3.88 B | 1.00 GB | **1.44 GB** | 2.25 GB |
| 64 | 5.08 B | 1.31 GB | 1.75 GB | 2.57 GB |
| 256 (Maple) | 19.6 B | 5.05 GB | 5.5 GB | 6.35 GB (official file) |

So the target is **32–48 experts per layer (of 256) with a 4-bit embedding and 6-bit head**. Two consequences:

- The embedding and head must leave F16: at F16 they alone are 1.24 GB. Their quantization is a measured step
  (N1); the head is the sensitive one (`Q8_0` instead of `Q6_K` adds 0.07 GB).
- Keeping 12–19% of the experts is not gentle by count. It can still be gentle by function: routing on a narrow
  domain concentrates on few experts, so the question is how much of the routing mass the kept experts carry on our
  corpus. N0 measures it before anything is trained.

**Speed.** Pruning experts does not reduce the work per token (still 8 experts in 24 layers), so the nested model
runs at about Maple's speed (95 tok/s generation on the DGX CPU, 57 on the laptop) with a fifth of the memory and a
better cache footprint. If more speed is wanted, the number of active experts (top-k) is a second nested dimension:
k = 4 halves the expert work. It is measured, not assumed.

Dropping layers would also shrink the file (12 layers × 96 experts ≈ 1.4 GB) and double the speed, but it changes
the residual stream after the first dropped layer, which ends exact nesting (§2). Not in the first version.

## 2. Keeping the connection: options

| Option | What is shared | Connection after training | Cost |
| --- | --- | --- | --- |
| A. Prune, then train separately | Initial weights only | Only through distillation and the learned shared space; the two drift apart | Simplest; a second 20 GB of frozen weights is not needed either (the small one is small), but nothing ties them |
| **B. Nested (Matryoshka)** | The small model is literally a sub-network of the large one: same attention, same embedding/head, its experts are a subset of the large one's | Permanent: one set of weights; training improves both | Every step trains both sizes (two forwards); the shared weights serve two masters |
| C. Nested + private adapters | As B, plus a small zero-initialised adapter used only by the small model (and per-size private router rows) | Mostly permanent; the private part is a measured dial | As B, plus a few MB |

**Recommendation: C** (B with a dial). It keeps the connection wherever it costs nothing and lets the small model
deviate only where it must, with the size of that deviation visible in the private adapter's norm.

### What nesting buys

- **One family of artifacts.** The small file is a slice of the large one. A browser downloads the small model
  (1–1.5 GB); a desktop later adds the remaining experts to get full Maple (progressive download).
- **Elastic sizes.** Order each layer's experts by importance; then every prefix is a model: 32 ⊂ 48 ⊂ 64 ⊂ 128 ⊂
  256. Training samples prefix sizes (as MatFormer samples widths), so intermediate sizes work without being trained
  as separate models.
- **One Neuralese space for free.** Both sizes share embedding, attention and residual stream, so at matching cutoff
  layers their Neuralese spaces coincide up to drift. The shared space of MAPLE_QWEN_JOINT §4 becomes C = all 2048
  dimensions with no private part, and the bijections start at the identity (a learned near-identity correction
  stays available). The port heads can be shared by both sizes, and cross-reading (small writes, large reads, or
  the reverse) is trained directly.
- **Speculative decoding.** Same tokenizer and a nested distribution: the small model is a natural draft model for
  the large one, so full Maple on a fast machine runs faster too.

### Why nesting is exact for Maple

- **Router.** Maple takes a softmax over all 256 experts, keeps the top 8 and renormalises them. After
  renormalisation the softmax denominator cancels, so routing over a subset is exactly: router logits restricted to
  the subset, top 8 among them, softmax over the 8. No approximation in the restriction.
- **Ternary scales.** Maple's scale is per output row of each matrix, so dropping whole experts leaves every kept
  matrix's ternary codes and scales unchanged: the small file is a byte-exact slice of the large one (attention,
  kept experts) plus the re-quantized embedding/head.
- **Experts are independent.** Nothing else in a layer depends on which experts exist.

## 2a. A family of sizes, and a sub-0.5 GB core (owner, 2026-10-05)

The owner chose option C with continual distillation and asked for a whole family of sizes, ideally down to a core
"sketch" model under 0.5 GB fully quantized. Numbers below are from the configuration; the real cut points come
from N0/N1 measurements.

**What sets the floor.** The vocabulary tables, not the experts: embedding + head are 2 × 311 M weights, 0.35 GB
even at 4 bits each (0.43 GB with a 6-bit head). Below 0.5 GB almost nothing is left for experts in a 24-layer
model:

| Layers | Experts/layer | Ternary part | Total, embedding/head Q4_K/Q4_K | Q4_K/Q6_K |
| --- | --- | --- | --- | --- |
| 24 | 4 (top-4) | 0.14 GB | 0.49 GB | 0.57 GB |
| 24 | 8 (all active) | 0.22 GB | 0.57 GB | 0.65 GB |
| 12 | 8 | 0.11 GB | 0.46 GB | 0.54 GB |
| 8 | 16 | 0.13 GB | 0.48 GB | 0.56 GB |
| 8 | 24 | 0.18 GB | 0.53 GB | 0.61 GB |

Three routes under 0.5 GB, all measured before choosing:

1. **Fewer experts per token** (24 layers, 4 experts, top-4): keeps depth, nested in experts *and* in k.
2. **Depth-nested sketch** (the first 8 layers, 16 experts): Neuralese sketching already runs only the layers below
   the port cutoff (cutoff 8 is a Maple candidate), so the "sketch model" is the sketch stage of the family: the first
   8 layers with a small expert prefix plus the shared tables. It is nested in depth as well as width (a prefix of
   layers and of experts), so its residual stream is literally the large model's at layer 8. It cannot finish text
   on its own without a head readout from layer 8 (the port's feedback readout already does this); as a text model
   it would need an early-exit head trained by distillation.
3. **Smaller vocabulary tables**: 3-bit embedding (lookup only) or pruning head rows the domain never emits. Keeps
   the tokenizer (IDs unchanged, rows dropped are never predicted); saves up to ~0.1–0.2 GB; risky for open text.

Proposed family (prefixes; each trained by sampled-prefix distillation, §4):

| Member | Shape | Size (Q4_K/Q6_K tables) | Role |
| --- | --- | --- | --- |
| Core / sketch | 8 layers × 16 experts (or 24 × 4, top-4) | ~0.5 GB | Neuralese sketching, drafting, smallest devices |
| Small | 24 × 32 | ~1.1 GB | Browser |
| Medium | 24 × 64 | ~1.75 GB | Laptops |
| Full | 24 × 256 | ~5.5 GB | Desktop / server |

## 2b. Coupling the full model's routing to the nested core (owner question)

Should the full model's router be shaped so the core's experts are chosen more often by the full model, with the
remaining experts acting as specialisations around them? Options, from least to most invasive:

| Option | Mechanism | Inference change | Risk |
| --- | --- | --- | --- |
| R0 Order only | Core = the experts the full model already uses most (N0) | none | none; coupling only as strong as the natural concentration |
| R1 Soft core-mass regulariser | Add a loss on the full model: −log of the router probability mass that falls on the core prefix (weight β, ramped); router weights trainable (0.5 M per layer) | none (still plain top-8 softmax: llama.cpp unchanged) | the full model's routing departs from pretraining; watch its CE |
| R2 Overlap regulariser | Reward overlap between the full model's top-8 and the nested model's top-8 for the same token | none | as R1, more targeted |
| R3 Hierarchical routing | Fixed split: e.g. 4 experts chosen from the core, 4 from the specialists (DeepSeek-style shared + routed) | yes: custom routing in the fork, browser and trainer | changes the architecture; largest quality risk; best nesting |
| R4 Specialist grouping | Order non-core experts by co-activation with core experts, so each prefix tier adds the specialists of the core experts most in use (greedy marginal coverage instead of raw mass) | none | none; only changes which prefix contains what |

What coupling buys: a smaller KL between nested and full (better small models), higher speculative-decoding
acceptance, closer Neuralese spaces. What it costs: the full model's routing was learned for quality; pulling mass
onto a few experts reduces its effective capacity on the margin.

**Recommendation: R0 + R4 now, R1 behind a measurement.** N0 tells how concentrated routing already is. If the top
32 already carry most of the mass, R0/R4 give most of the benefit for free. If not, R1 with a small β, the router
unfrozen, and the full model's held-out CE as the guard. R3 only if nested quality stays inadequate after training,
because it forks the architecture away from upstream llama.cpp.

## 3. Expert ordering (N0)

Run full Maple (BF16 latent, ternarized forward: the deployed model) over the training corpus and record per layer,
per expert: routing mass (sum of renormalised gate weights), selection count, and output contribution
(gate weight × ‖expert output‖). Order experts per layer by routing mass (contribution as a tie-break check).
Report per layer the fraction of routing mass covered by the first 24/32/48/64 experts. The permutation is applied
to the expert lists and router rows, so "the first n" is the nested model.

Decision point after N0: the kept count (24–48), and whether coverage is so low in some layers that those layers
deserve more experts (non-uniform budgets per layer are allowed: the slice is still a prefix per layer).

## 3b. Deployed slices, untrained (N1, `/home/werg/data/maple-slices/results.jsonl`)

Cut from the official GGUF (byte-exact experts and router rows in N0 order), embedding re-quantized to Q4_K and head
to Q6_K, served by our llama.cpp fork on the DGX CPU (16 threads); perplexity over 12 chunks of 1,024 tokens of
natlang text (prompts included):

| Model | File | Perplexity | Generation | Prompt |
| --- | --- | --- | --- | --- |
| Full (F16 tables, official) | 6.35 GB | 12.2 | 78 tok/s | 466 tok/s |
| 128 experts | 3.02 GB | 17.5 | 112 tok/s | 489 tok/s |
| 64 experts | 1.76 GB | 59.1 | 104 tok/s | 486 tok/s |
| 48 experts | 1.45 GB | 115.5 | 110 tok/s | 489 tok/s |
| 32 experts | 1.13 GB | 147.6 | 95 tok/s | 489 tok/s |

Sizes match §1. Speed gains are modest (same work per token, smaller footprint). Untrained, 128 experts loses
little; 64 and below need the bootstrap/joint training to be usable.

## 3a. N0 result (2026-10-05, `runs/maple-nested-20261005/n0-v1/report.json`)

Routing statistics of full Maple over 327 post-training rows (~1.3 M tokens), held-out cost on 47 rows (3,390
completion tokens):

| Experts kept | 16 | 24 | 32 | 40 | 48 | 64 | 96 | 128 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Held-out NLL | 6.32 | 3.33 | 2.78 | 2.44 | 2.20 | 1.69 | 1.25 | 1.06 | 0.95 |
| KL from full | 6.07 | 2.96 | 2.41 | 2.06 | 1.81 | 1.22 | 0.66 | 0.38 | 0 |
| Top-1 agreement | 0.16 | 0.47 | 0.54 | 0.58 | 0.61 | 0.70 | 0.81 | 0.87 | 1 |

Routing mass covered by the first n experts, per layer: n = 32 covers 34–53%, n = 64 covers 53–73%, n = 128
covers 78–92%. Layers 0–2 and 23 are the most diffuse (top-32 ≈ 34%); middle layers the most concentrated. Almost
every expert is used (0–4 unused per layer).

Reading: routing on our domain is **not** concentrated, so there is no natural small core and an untrained 1 GB
member (32–48 experts) is badly damaged (NLL 2.2–2.8 against 0.95). Consequences:

- Healing by training carries the small members; the bootstrap/joint phases are the experiment, and the 64- and
  128-expert members are the safer early targets.
- Coupling the full model's routing toward the core (§2b R1) becomes more attractive than R0/R4 alone.
- Per-layer budgets (more experts in layers 0–2 and 23) would help; llama.cpp holds one expert count per model, so
  this needs either padding or a small fork change. Option, measured later.
- The ordering used the narrow post-training rows; it is recomputed on the full v13 corpus when that is the
  training data.

## 3c. First bootstrap smoke (2026-10-05, `runs/maple-nested-20261005/n2a-smoke-v1`)

20 steps of the private-parts bootstrap (shared weights frozen; private router rows, norm gains, rank-4 attention
deltas; KL + hidden matching + 0.1 CE), members 24x32, 24x64, 8x16, 374-row post-training data, 2×1,024 tokens per
step, 8.7 s/step, 41 GB peak. Held-out (6 rows) CE / KL to full, before → after:

| Member | CE | KL to full |
| --- | --- | --- |
| full (frozen) | 1.40 | — |
| 24x64 | 2.37 → 1.95 | 1.32 → 1.24 |
| 24x32 | 3.29 → 2.41 | 2.40 → 1.72 |
| 8x16 (early exit) | 11.97 → 5.45 | 11.40 → 4.85 |

The machinery works end to end on real Maple; the full bootstrap runs on the 111,232-row v13 render
(`n2a-v1`: 800 steps × 4 sequences, evaluation every 100 steps).

## 3d. Bootstrap ceiling of the private parts (2026-10-05, `runs/maple-nested-20261005/n2a-v1`)

200 steps × 4 sequences (≤1,024 tokens) of the v13 Maple render, shared weights frozen, all three members every
step (18 s/step, 41 GB). Held-out 16 rows (full Maple CE 2.25 on these), CE / KL to full:

| Step | 24x64 | 24x32 | 8x16 |
| --- | --- | --- | --- |
| 0 | 4.47 / 2.58 | 5.70 / 3.91 | 12.42 / 10.98 |
| 50 | 3.50 / 1.85 | 3.94 / 2.34 | 6.78 / 5.34 |
| 100 | 3.33 / 1.66 | 3.73 / 2.10 | 6.34 / 4.94 |
| 150 | 3.22 / 1.56 | 3.50 / 1.88 | 6.13 / 4.82 |
| 200 | 3.09 / 1.48 | 3.39 / 1.84 | 5.98 / 4.66 |

Reading: most of what private parts can do happens in the first 50 steps; afterwards KL falls by ~0.05–0.2 per 50
steps and flattens. Private parts alone leave large gaps (24x64: CE 3.09 vs 2.25; 24x32: 3.39; the 8x16 early exit:
5.98). So the members need the joint phase (shared weights and expert scales trained with every member as an
objective) and/or more private capacity; the depth-8 early exit is far from a usable text model and is better
treated as the Neuralese sketch stage than as a standalone member, or replaced by a deeper core.

## 4. Training

Per step, on one token batch (the natlang SFT corpus in Maple's chat format):

```text
full  = Maple(all experts)          → CE_full                          (QAT LoRA, shared)
small = Maple(first n experts)      → CE_small + λ·KL(full ‖ small)     (same LoRA + private adapter)
loss  = CE_full + w_small · (CE_small + λ·KL)
```

- Prefix size n sampled per step from {n_target, a larger size, 256} with most weight on the target, so the
  intermediate sizes stay usable.
- Teacher side detached in the KL (the full model is not pulled toward the small one).
- Shared trainable parameters: attention QAT LoRA (all layers), expert QAT LoRA on the kept experts (they are the
  experts both sizes use most). Private: zero-initialised low-rank adapter on attention outputs for the small size,
  private router rows per size (a trainable copy of the member's n router rows per layer; the member's GGUF carries them as its own `ffn_gate_inp`, so they export exactly; a bias would need llama.cpp's selection-only `exp_probs_b`, which gets no gradient).
- Memory: the frozen weights are shared, so no second copy. Extra: the small forward's activations (~the same as
  the full one's, since active work is the same). Fits the budget in MAPLE_QWEN_JOINT §3 with room to spare
  (no 11 GB student).
- Watch: CE_full must not degrade against the full-only QAT run (M1); w_small and the private adapter are the
  levers.

## 4a. State-of-the-art QAT and genuine member objectives (owner, 2026-10-05)

The owner asked for state-of-the-art QAT throughout, a thorough distillation of every smaller member's private parts
at bootstrap, and every family member treated as a genuine joint optimisation target throughout (not only a
by-product distilled from the full model).

### QAT recipe

Constraint first: the published weights are already ternary (maple-qat §2.0), so there is no full-precision latent
and no full-precision teacher. What current ternary work offers, and what we take:

| Technique | Source | Use here |
| --- | --- | --- |
| **Learned scales** (LSQ-style), zero-inclusive symmetric ternary grid | ParetoQ (2025): learned-scale grids are the best QAT choice at ternary | **Yes, and finer than Maple's rule.** TQ2_0 stores one FP16 scale per 256 weights, and llama.cpp does not care how it was chosen. Train a scale per 256-weight block (per row today). Experts get **scale-only QAT**: 75 M continuous parameters over all 19.3 B expert weights, ~0.9 GB with Adam, no dead zone, exact export. |
| **Dead-zone trapping fix** | Tequila (ICLR 2026): weights stuck at the zero/nonzero threshold get only noisy STE gradients | Measured first (flip rate, logged). Tequila's fix repurposes dead-zone weights as a bias, which needs a bias add in the llama.cpp Maple graph (a fork change). Adopt only if flip rate shows trapping. |
| Distillation-driven QAT | BitDistill (2025): logits KD + MiniLM-style attention-relation distillation; LLM-QAT | **Only for the members.** BitDistill recovers quality lost when quantizing a full-precision model; full Maple is already ternary and our aim is to *change and improve* it, so for the full model this is plain fine-tuning under the ternary constraint (STE + learned scales) on the task objective. The members are genuinely compressed versions of the full model, so they take logits KL plus hidden-state matching from it (relation distillation as an option). |
| Forgetting guard | KL to the reference model, as in RLHF; mixed-domain data (two-stage reasoning QAT, ICLR 2026) | **On broad rows only, as a dial.** KL to the frozen original Maple (adapters off, no extra memory) applies only to broad chat/text rows, never to task rows, so it guards general ability without resisting the task change. Weight 0.5 by default, 0 turns it off; the full model's held-out broad CE shows whether it is needed. |
| Stronger token-level teacher for the full model | standard KD | **Option.** Qwen3.6 (our campaign teacher) uses a new 248k vocabulary, so it teaches only through generated text (the SFT data). The 2025 Qwen3 models share Maple's tokenizer exactly (e.g. Qwen3-30B-A3B-Instruct-2507, Qwen3-235B-A22B-2507): exact token-level KD into full Maple, online (FP8 ~31 GB for 30B-A3B) or from precomputed top-k logits. Worth a measured trial if SFT alone plateaus. |
| Full-weight latents | standard QAT | Attention only (252 M weights: FP32 latent + Adam ≈ 3 GB). For experts, scales plus LoRA latents on the experts the member family uses most. |

### Data: breadth through natlang domains, including chat and writing (owner, 2026-10-05)

Forgetting general ability is acceptable; breadth comes from expanding the natlang training data to more and more
domains, not from mixing in broad chat/web text or anchoring to the original model. So the trainers default to
anchor weight 0 and no broad rows (both remain options). The first corpus is the full v13 natlang SFT corpus
(111,301 rows) rendered in Maple's chat format (`/home/werg/data/maple-sft/maple-v13.sft.jsonl`), not the
374-row post-training subset used for smoke tests.

Chat and writing become natlang domains themselves: natural-language functions whose bodies the model interprets,
called from programs like any other task family. Proposed families, each with an admission oracle in the natlang
style:

| Family | Example function | Admission |
| --- | --- | --- |
| Constrained writing | `write(brief, constraints): string` (length, format, required/forbidden content, structure, tone markers) | Exact: the constraints are checked by code (IFEval-style verifiable instructions) |
| Editing and rewriting | `rewrite(text, instruction): string`, `summarize(text, budget)`, `translate_register(text, audience)` | Exact parts (budget, preserved facts/entities, format) plus a judge for fidelity |
| Chat turn | `reply(conversation, persona?): string` over real chat prompts (smoltalk and similar as prompt sources) | Teacher-judge rubric with preference labels (the existing Clef/Decider labelling path); exact checks where the prompt has a checkable answer |
| Writing inside workflows | existing workflow families extended with a drafting step (emails, reports, explanations of computed results) | The workflow's oracle plus constraint checks on the draft |

These are generated by the teacher campaign through the natlang runtime like the existing families, so they share
tool use, the runtime contract and the protected evaluation split policy. A plan of its own belongs with the data
programme; noted here because it is the intended replacement for broad-data mixing in Maple training.

### Bootstrap: private parts first (phase N2a)

Before any shared weight moves, each member's **private parts** are distilled from full Maple with the shared
weights frozen:

- per-size private router rows (exported as the slice's own `ffn_gate_inp`), per-size RMSNorm gain corrections (cheap; exact in the slice's own norm tensors),
- per-size private attention LoRA inside the quantizer (`Q(W + ΔW_shared + ΔW_size)`; the member's slice then
  carries its own ternary attention tensors, +65 MB; the experts stay byte-shared),
- for the depth-nested sketch member, its early-exit readout (final norm + head) at the cutoff layer.

Losses: logits KL to full Maple, hidden-state matching at the member's depths, small CE weight. Run to convergence
(held-out KL flat), on mixed-domain data. Report each member's held-out NLL/KL before and after bootstrap.

### Joint phase (N2b): every member is an objective

Per step, the full model and **several members** (all members in rotation; at least two per step) each take:

```text
L_m = CE_m (task data)                     — the member's own objective
    + λ_KD · KL(full ‖ m)  + λ_H · hidden-state match     — stays close to the family head
L_full = CE_full + λ_anchor · KL(original Maple ‖ full)   (broad rows only; λ_anchor may be 0)
loss = L_full + Σ_m w_m · L_m      (w_m normalised per member; shared weights get every member's gradient)
```

Evaluation treats members as products, not by-products: every evaluation runs every member (held-out NLL, KL to
full, and the protected execution evaluation through its own llama.cpp slice). Checkpoint selection needs no
member to regress against its bootstrap result; a member that falls behind gets its weight raised. Gradient
conflict between members (measured as cosine between member gradients on the shared parameters) is logged; if it
binds, per-member gradient projection is the next step.

## 5. Neuralese

One port design for both sizes: same cutoffs (4, 8 or 12), same markers (151,669/151,670), port heads shared with a
per-size flag available if they need to differ. Training adds the cross-read losses of MAPLE_QWEN_JOINT §4.5 between
sizes; the shared-space maps start at the identity.

## 6. Export

- Full: as maple-qat §2 (ternary `TQ2_0`, embedding/head per the chosen format).
- Small: the first n experts per layer and router rows of the same export; embedding `Q4_K`, head `Q6_K` (or
  `Q8_0`). Verified by slicing equality (the kept tensors equal the full file's bytes) and by llama.cpp next-token
  agreement with our backbone.
- The fork's Maple support handles fewer experts through the GGUF's `expert_count` key: verified in N1.

## 7. Phases

| Phase | Work | Review |
| --- | --- | --- |
| N0 Ordering | Routing statistics of full Maple on the corpus; coverage table per layer. | Kept count per layer decided. |
| N1 Slice, untrained | Export slices at 24/32/48/64 experts with embedding/head formats; protected evaluation and NLL vs full Maple; llama.cpp loads them. | How much pruning costs before any training. |
| N2 Nested QAT | §4 joint training. | Small: protected evaluation at target size; full: no regression vs M1. |
| N3 Neuralese | §5 on both sizes. | S3 reviews per size; cross-read. |
| N4 Deployment | Slices served by the fork, browser build of the small one, speculative decoding test. | Conformance; browser loads it within 4 GB. |

## 8. Decisions (owner)

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| N-a | Connection | A separate / B nested / C nested + private adapters | C |
| N-b | Target size | 24 (0.97 GB) / 32 (1.12 GB) / 48 (1.44 GB) experts | Decide after N0's coverage; 32 as the working target |
| N-c | Embedding/head formats | Q4_K/Q6_K / Q4_K/Q8_0 | Q4_K/Q6_K unless N1 shows a head-quantization loss |
| N-d | Speed lever | top-8 (Maple's speed) / nested top-k (k=4: faster, measured) | top-8 first |
| N-e | Elastic sizes | target only / sampled prefixes | sampled prefixes (cheap, gives the 32 ⊂ … ⊂ 256 family) |

## 9. Before launching the Maple Neuralese run (2026-10-05)

Data (unified corpus; registry `training/neuralese_corpora.json`, owner protocol `plans/MACHINE_COORDINATION.md`):

1. Register the DGX data not yet in any snapshot (inbox note 2026-10-05): student post-training turns (Oct 4–5),
   latest source-episode pools, translation sources (NLLB 241k, static 25k).
2. Generate teacher data for the writing families (and a chat-turn family), then register it.
3. Admission for Maple: one manifest-addressed training snapshot. Its parts:
   - neuralese v8 conversion, plus the new recurrence/combinator generation;
   - S1 final;
   - admitted S2 and skill episodes, after paired gates and neuralese conversion;
   - post-training turns;
   - writing.

   Protected data is excluded and split closure is checked. The records are text, so they need no re-tokenization:
   they are rendered with Maple's chat template at load time, with the empty-think rule for unreasoned turns.

Code:

4. S3 pilot loads only the LFM backbone (`train/pilot.py` → `load_backbone`): add a Maple path through
   `MaplePortBackbone`. Markers are 151,669/151,670; cutoffs are 4/8/12. Pick the cutoff with a short sweep.
5. Nested family inside S3: run the S3 step with member keys, with the member losses of §4a. The bootstrap state is
   `runs/maple-nested-20261005/n2a-v1`; run the planned 200-step joint phase first to confirm that members improve
   under joint training.
6. Fork serving of Maple Neuralese writes and reads: llama.cpp parity (M0.3, `scripts/maple_llamacpp_parity.py`, not
   yet run), then the port heads in the fork.
7. Protected execution evaluation per member (not just perplexity), as a periodic evaluation in the run.
8. Memory: full Maple with QAT LoRA, member deltas and port heads, without gradient checkpointing. Measure peak memory
   in a 20-step smoke through the ledger, then set the budget.

### 9a. Port initialization comes first (owner 2026-10-05)

The Maple port must start inside the model's own input and output manifold, not from freshly initialized heads (the
LFM lineage found that skipping this left a weak channel; see TRAINING_RECIPE.md). The Maple lineage therefore runs
the shared foundation recipe before any port training:

- `recipes/foundation-maple-v1.json`: token identity, then causal output-state → raw next-token-embedding distillation
  at cutoff 12, same gates as LFM (held agreement ≥ 0.9 and KL ≤ 0.25, per source and context stratum). Heads:
  `maple/foundation_heads.py` (untrained heads at the cutoff; `serve.load_engine` rebuilds the Maple student from the
  heads checkpoint, its nested state pinned by sha256).
- Run `runs/maple-foundation-20261005/c12` (student: n2b-v1 state; data: local-recurrence-inputs-20261005, 220 train /
  9 held admitted sources + 128 context windows). Token identity passed exactly (raw transport, readback logits and the
  full-depth reference all 0 difference). Distillation step 0: held agreement 0.019, KL 8.85 (layer 12 read through the
  frozen head); 8,192 steps.
- `train/pilot.py --backbone maple` refuses to train without the foundation (only `--unqualified-smoke` for memory
  smokes): it still builds the legacy marker/RMS feedback heads. The S3 Maple run consumes the qualified projection
  through the raw token-aligned runtime (TRAINING_RECIPE stage 3, Pop's integration in progress); the legacy S3 smoke
  was withdrawn. If cutoff 12 fails its gate, sweep deeper cutoffs before shallower ones.

### 9b. Member execution evaluation (protected 24-case plan, CPU llama.cpp, 2026-10-05)

| Member (state) | Complete success | Incomplete task | Semantic failure | Contract failure | Policy held |
| --- | --- | --- | --- | --- | --- |
| Official full Maple | 12 | | | | |
| 8x16 (n2a) | 2 | 4 | 17 | 0 | 1 |
| 8x16 (n2b) | 1 | 17 | 5 | 0 | 1 |
| 24x32 (n2b) | 4 | 12 | 5 | 2 | 1 |
| 24x64 (n2b) | 3 | 12 | 6 | 2 | 1 |

Members do not yet execute: half their cases end incomplete (the call never finishes). The joint phase alone does not
close the gap; the members need the S3 data and the foundation-initialized port. Driver:
`scripts/maple_member_execution_eval.sh STATE OUT "MEMBERS" PORT`.
