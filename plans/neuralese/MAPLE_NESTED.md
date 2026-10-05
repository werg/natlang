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
| C. Nested + private adapters | As B, plus a small zero-initialised adapter used only by the small model (and per-size router biases) | Mostly permanent; the private part is a measured dial | As B, plus a few MB |

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

## 3. Expert ordering (N0)

Run full Maple (BF16 latent, ternarized forward: the deployed model) over the training corpus and record per layer,
per expert: routing mass (sum of renormalised gate weights), selection count, and output contribution
(gate weight × ‖expert output‖). Order experts per layer by routing mass (contribution as a tie-break check).
Report per layer the fraction of routing mass covered by the first 24/32/48/64 experts. The permutation is applied
to the expert lists and router rows, so "the first n" is the nested model.

Decision point after N0: the kept count (24–48), and whether coverage is so low in some layers that those layers
deserve more experts (non-uniform budgets per layer are allowed: the slice is still a prefix per layer).

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
  router bias per size (256 numbers per layer per size).
- Memory: the frozen weights are shared, so no second copy. Extra: the small forward's activations (~the same as
  the full one's, since active work is the same). Fits the budget in MAPLE_QWEN_JOINT §3 with room to spare
  (no 11 GB student).
- Watch: CE_full must not degrade against the full-only QAT run (M1); w_small and the private adapter are the
  levers.

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
