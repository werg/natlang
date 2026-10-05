# Maple + Qwen3-0.6B: joint training, distillation, and a shared Neuralese space

Written 2026-10-05. Companion to [../maple-qat.md](../maple-qat.md) (Maple QAT and Neuralese on Maple).

## 1. The partner model

The owner asked about Maple's 0.5B sibling. deepgrove's only small model is **Bonsai** (0.5B, March 2025), and it
does **not** share Maple's tokenizer: Llama architecture, Mistral 32k vocabulary, 2,048-token context, base model
only, trained on under 5B tokens. Distilling into it would need cross-tokenizer alignment (approximate), and it
starts from a much weaker base.

Maple's tokenizer is byte-for-byte Qwen3's (vocabulary, merges, pre-tokenizer, normalizer, all 26 added tokens with
the same IDs). So the natural small partner is **Qwen3-0.6B** (Apache-2.0): 28 layers, width 1024, 16 query / 8 KV
heads, tied embeddings, 40k context, an instruct model with Qwen3's thinking and tool-call format. Identical token
streams give three things nothing else gives:

1. **Exact distillation**: Maple's next-token distribution is over the same 151,936 tokens, position for position.
2. **Paired activations**: both models read the same tokens at the same positions, so their hidden states pair up
   without any alignment step; that is what initialises the shared space (§4.3).
3. **Shared markers**: `<|neuralese|>`/`<|/neuralese|>` get the same IDs (151,669/151,670) in both.

Alternatives, for the record: Qwen3-1.7B (same tokenizer, 3× larger, still browser-feasible quantized); our
LFM2.5-350M student (different tokenizer: only approximate, sequence-level distillation; it could still join the
shared space through character-offset pairing, §4.7).

## 2. One process, both models

Everything runs in one training process on the DGX GPU, on one token batch per step:

```text
batch (Maple/Qwen3 chat format, identical token IDs)
  ├─ Maple (frozen ternary weights + QAT LoRA)  → CE_M,  final hidden states H_M (kept, detached)
  └─ Qwen3-0.6B (full fine-tune, BF16)           → CE_Q + λ · KL(p_M ‖ p_Q)
```

- **No logit storage.** Keeping teacher logits for 8k tokens is 2.5 GB; instead keep Maple's final-norm hidden states
  (2048 per token) and, inside the chunked loss, compute Maple's and Qwen's logits for a chunk of positions, take CE
  and the full-vocabulary KL there, and discard. Exact (no top-k truncation) and memory-flat.
- **One teacher forward serves both**: Maple's forward is needed anyway for its own CE.
- **Teacher drift.** Maple is itself training. The KL uses Maple's current weights, detached; λ ramps from 0 over the
  first ~1k steps, so Qwen first follows the data and then the improving teacher. Alternative (§6 J2): a frozen
  Maple snapshot refreshed every N steps.
- **KL direction and temperature.** Forward KL `KL(p_M ‖ p_Q)` at T = 1 (mode-covering; standard for distillation into
  a much smaller model), on supervised positions only; CE on the same positions.
- **Data.** The same natlang SFT corpus as Maple's QAT (v13 render), rendered once. The two chat templates share
  ChatML, the tool prompt and `<tool_call>` JSON, and `<think>` blocks; they differ in one rule: Qwen3 drops the
  reasoning of assistant turns before the last user query, Maple keeps any reasoning it is given. With empty think
  blocks (the default, maple-qat §9 D5) they coincide; the renderer emits Maple's form and a check compares it with
  both templates on a sample.

## 3. Memory (no gradient checkpointing)

| Item | GB |
| --- | --- |
| Maple frozen weights, BF16 ternary (int8 codes + scale: 20) | 40 |
| Maple QAT LoRA (attention, rank 8) + Adam | < 0.1 |
| Maple activations, 8k tokens/step (MoE: 1B active) | 10–15 |
| Qwen3-0.6B full fine-tune: BF16 weights, FP32 master, Adam, grads | ~11 |
| Qwen3-0.6B activations, 8k tokens | 6–8 |
| Ports (both models), shared-space maps, chunked-loss workspace | 2–4 |
| **Total** | **~70–80** (50–60 with int8 Maple) |

Beside the teacher campaign (~60 GB) that does not fit; with the campaign paused it fits with room for a larger
batch. It shares the pause with the S3 full run (which needs ~44 GB): run them one after the other, or together if
the int8 variant brings this one to ~55 GB.

## 4. The shared Neuralese space

### 4.1 What is being shared

A Neuralese block is a sequence of vectors written at a model's port cutoff layer (its residual stream at layer k)
and read by splicing into the same layer (S3). Each model's native Neuralese space is therefore its residual space at
its cutoff: R^2048 for Maple, R^1024 for Qwen3-0.6B.

### 4.2 Shape: a common part plus a Maple-private part

An exact bijection needs equal dimensions. So:

```text
Z = C ⊕ P            C = R^1024 (common),  P = R^1024 (Maple-private)

f_M : R^2048 → C ⊕ P       bijective (Maple residual ↔ whole Z)
f_Q : R^1024 → C           bijective (Qwen residual ↔ common part)
g   : C → P                learned conditional prior (mean, and a log-variance for training)
```

- Maple → shared → Maple: `f_M⁻¹(f_M(h)) = h` exactly. Qwen likewise. **Each model's own Neuralese is untouched by
  the shared space**, whatever it learns; the space can be added after each port is trained, and changing it later
  never breaks self-use.
- Maple → Qwen: `f_Q⁻¹(c)` where `(c, p) = f_M(h)`; P is dropped (information Qwen cannot represent).
- Qwen → Maple: `f_M⁻¹(c, g(c))`: the missing private part is filled with the prior's mean.

### 4.3 The maps

Each map is an invertible linear layer plus optional affine coupling layers (RealNVP-style), all exactly invertible:

- **Linear**: `W = P · L · (U + diag(exp s))` (LU form with a fixed permutation; invertible by construction, cheap
  log-determinant and inverse via triangular solves), plus a bias.
- **Coupling** (0–4 layers): split the coordinates, `y₂ = x₂ ⊙ exp(tanh(a(x₁))) + b(x₁)`, alternating halves.
  Start with zero layers (linear only); add them if cross-read quality plateaus (§6 J4).
- **Scale**: residual-stream norms differ greatly between models (and within a model by position: the first-token
  outliers). The maps include the RMS normalisation statistics of each side, and pairing excludes outlier positions.

### 4.4 Initialisation from paired activations

Run both models on the same text (the training corpus; no Neuralese yet) and collect paired residuals at the two
cutoffs, ~1M positions. Then:

1. **Choose the pairing of cutoffs.** For Maple k_M ∈ {4, 8, 12} and Qwen k_Q over its 28 layers, score linear
   predictability (CCA / linear CKA) and pick, for each k_M, the most aligned k_Q. Qwen's cutoff also passes its
   own port sweep (S3 §7).
2. **CCA.** Canonical directions of Maple (2048) and Qwen (1024) residuals: the top-1024 Maple canonical directions
   map to C, matched with Qwen's 1024 canonical directions (whitened), and the remaining 1024 Maple directions (the
   orthogonal complement, ordered by variance) map to P.
3. The resulting matrices are full rank, so they directly initialise the linear layers (LU-decomposed); couplings
   start at identity; `g` starts as the least-squares regression of P on C.

### 4.5 Training the shared space (functional alignment)

The CCA start aligns what both models *represent*; the following make Neuralese *mean* the same thing to both.
All run in the joint process on the same batches, with both ports trained (§5):

| Loss | What it does |
| --- | --- |
| **Cross-read** | Maple writes a block; Qwen reads it through `f_Q⁻¹(C)` and must produce the trajectory's continuation (CE on the answer tokens). And the reverse: Qwen writes, Maple reads `f_M⁻¹(c, g(c))`. The S3 reader losses, applied across models. |
| **Write agreement** | For the same trajectory, Qwen's written block in C should match Maple's (MSE in the whitened C coordinates, Maple side detached: the large model teaches the small one's writer). |
| **Length / stop distillation** | Qwen's stop head matches Maple's stop decisions (KL on the stop probability), so blocks have comparable lengths. |
| **Prior** | Gaussian negative log-likelihood of Maple's P given C, trains `g`. |
| **Content consistency** | Each model's content head decodes the other's block (through the maps) to the same value text. |

Gradients flow into the maps and the readers' ports; the backbones are frozen in this phase (or LoRA at a low rate),
so the self-use invariance of §4.2 holds and only cross-use improves.

### 4.6 What "shared" buys

- The small model can consume a big model's thinking (Maple writes, Qwen reads in the browser), and a big model can
  check or continue a small model's block.
- A single Neuralese format on the wire: the runtime stores `z` vectors with a model tag and translates on read.
- A route for further models: each new one needs only its bijection onto C (or onto C ⊕ its own private part).

### 4.7 Extension: models with other tokenizers

Our LFM2.5-350M student could join with its own `f_L : R^1024 → C`. Pairing positions then goes through character
offsets (token boundaries that coincide in both tokenizations); distillation stays sequence-level. Not in the first
version.

## 5. Phases

| Phase | Work | Review |
| --- | --- | --- |
| J0 Checks | Tokenizer identity asserted at load (hash of `tokenizer.json` vocab+merges+added tokens). Both chat templates render the corpus identically. Paired-activation dump; cutoff pairing (§4.4.1). | Identity holds; cutoff pairing chosen. |
| J1 Joint SFT + distillation | Maple QAT LoRA (attention, then the M2 scope) + Qwen full fine-tune with CE + λ·KL. | Both exported models on the protected execution evaluation: Qwen against Qwen trained without distillation (ablation, same steps). |
| J2 Ports | S3 curriculum A–F on both models in the same process and batches; Qwen's stop head distilled from Maple's. | S3 G1 review per model. |
| J3 Shared space | CCA initialisation, then §4.5 losses with frozen backbones. | Cross-read quality vs self-read; Qwen→Maple vs a zero prior. |
| J4 Joint refinement | All of the above together at low rates; optional coupling layers. | Self-use unchanged (exact by construction, but checked); cross-use improved. |
| J5 Serving | Fork: Qwen3 and Maple layer-range graphs; shared-space maps in the heads file; browser runs Qwen3. | Conformance. |

## 6. Decisions (owner)

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| J1 | Student training | Full fine-tune (11 GB, most capacity) / LoRA (cheap, weaker) | Full fine-tune |
| J2 | Teacher for distillation | Current Maple (co-training, λ ramp) / snapshot refreshed every N steps / Maple frozen at its J1 result, Qwen trained after | Current Maple with ramp; snapshot if Qwen's loss oscillates |
| J3 | Common width | C = 1024 (all of Qwen) / smaller C (e.g. 512) with a Qwen-private part too | 1024: Qwen's whole space is shared, and every Qwen block is readable by Maple |
| J4 | Maps | Linear only / linear + coupling | Linear first |
| J5 | Order | Everything at once / J1 then J2 then J3 in one process | Sequential phases in one process (each phase reviewable, memory the same) |
| J6 | Campaign pause | Joint run needs ~70–80 GB | Run after the S3 full run, in the same pause |

## 7. Implementation status

See `natlang_neuralese/maple/` (ternary rule and QAT), `natlang_neuralese/shared_space.py` (bijections, CCA
initialisation, prior), `natlang_neuralese/train/joint_kd.py` (joint distillation step), and the Qwen3/Maple port
backbone `natlang_neuralese/model/hf_port.py`. Each lands with tests; `plans/HANDOVER.md` records progress.
