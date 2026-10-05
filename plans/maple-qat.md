# Maple: quantization-aware fine-tuning, Neuralese, and deployment

Revised 2026-10-05 (second version). The first version (commit `204b4f9437`) stopped mid-sentence in §6.1 and
referred to appendices A and B that it did not contain; it also targeted Apple's MLX runtime, while our stack
serves through the llama.cpp fork. This version keeps its sound parts (the exact ternary rule, the
merged-weight QAT forward, the trainable scopes, the loss and precision cautions), fills the gaps, retargets
deployment to llama.cpp, adds Neuralese, and links the joint-training plan
([neuralese/MAPLE_QWEN_JOINT.md](neuralese/MAPLE_QWEN_JOINT.md)).

## 1. Why Maple

Maple-Preview (deepgrove, MIT): a 20B-total, ~1B-active reasoning MoE trained with ternary weights. 24 layers, width
2048, 256 experts per layer with 8 active, attention in a repeating pattern of three 512-token sliding-window layers
and one global layer (`SSSG` ×6: global at layers 3, 7, 11, 15, 19, 23), partial rotary (half of each head) on the
sliding layers and none on the global ones, Q/K norm, 131k context, untied embedding and output head, vocabulary
151,936 (the Qwen3 tokenizer, verified identical: §8).

Measured on our hardware by the evaluation session (official `TQ2_0`-with-F16-head GGUF, 6.35 GB, upstream llama.cpp
CPU, `runs/maple-preview-evaluation-20261005/`):

| Machine | Prompt (512 tokens) | Generation (128 tokens) |
| --- | --- | --- |
| DGX GB10 ARM CPU, 16 threads | 467 tok/s | 86–95 tok/s |
| Local Ryzen 5 8645HS, 6 threads | 320 tok/s | 57 tok/s |

That is faster than our 350M student on the same CPUs while carrying a 20B model's knowledge. Untuned, on the
protected natlang execution evaluation (`execution-v5/report.json`): 12 of 20 completed cases semantically correct
(3 infrastructure/incomplete, 1 held by policy). Our best tuned LFM student scores 11 of 23. So Maple starts about
where our fine-tuned student ends.

## 2. Deployment contract (our target: the llama.cpp fork)

| Component | Official GGUF representation | Training treatment |
| --- | --- | --- |
| Attention Q/K/V/O | `TQ2_0` (ternary codes, one F16 scale per 256-weight block) | QAT on the trainable scope; frozen ones ternarized once |
| Expert gate/up/down | `TQ2_0`, per expert | same |
| Router (`ffn_gate_inp`) | F32 | frozen; routing computed in FP32 outside autocast |
| Norms (incl. Q/K norm) | F32 | frozen |
| Token embedding, output head | **F16** | frozen; F16 rounding is negligible next to BF16 training |
| Activations, KV cache | native floating point | unchanged |

Upstream llama.cpp already supports Maple (PR #27000, merged as `3d10bcd19`, present in our fork); the official GGUF
carries the required `swiglu_clamp_exp = 7.0` metadata (verified by the evaluation session). The MLX build
(`maple-preview-2bit-mlx`, 4-bit embedding/head) stays an optional secondary target: if it is ever needed, simulate
its affine 4-bit embedding/head as the first version described, against the pinned MLX exporter.

### 2.1 The ternary rule, and why export must apply it before `TQ2_0`

Maple's rule, per output row of the BF16 weight (deepgrove `mlx_lm/ternary.py`): in FP32,
`threshold = 0.7 · mean|w|`, `mask = |w| > threshold` (strict), `alpha = mean(|w| over mask)` (0 for an all-zero
row), `alpha = BF16(alpha)`, `Q(w) = sign(w) · mask · alpha`.

llama.cpp's `TQ2_0` quantizer is a different rule: per 256-weight block, `d = max|x|` and `q = round(x / d)`. Fed raw
BF16 weights, it would zero nearly everything below half the block maximum and rescale the rest, which is not what
Maple was trained for. Fed **pre-ternarized** weights (`Q(w)` above), every nonzero in a block equals `±alpha_row`, so
`d = alpha_row` and the codes are exact; the only loss is `alpha` going BF16 → F16 (exact for normal values; check
underflow below 6e-5). So export is: merge → BF16 → Maple rule → F32/BF16 tensor of `{−α, 0, +α}` → `TQ2_0`.
Gate M0.2 verifies that the official GGUF equals exactly this applied to the official BF16 weights.

## 3. QAT: forward contract and parameterization

Unchanged from the first version, now implemented in `natlang_neuralese/maple/ternary.py` (tests in
`tests/neuralese/test_maple_ternary.py`):

```text
W_effective = W_base + (alpha_lora / r) · B @ A        (A, B in FP32; B zero-initialised)
W_forward   = Q(BF16(W_effective))                      (Maple's rule on the BF16-rounded weight)
y           = x @ W_forwardᵀ
```

Never `linear(x, Q(W_base)) + adapter(x)`: `Q(W + ΔW) ≠ Q(W) + ΔW`, and only the first is what ships. Backward:
identity straight-through to `W_effective` (the scale computation included), to be validated against a pilot.
The parameterization intercepts `.weight` (`torch.nn.utils.parametrize`), so it works with our own port backbone,
which reads weights through `F.linear`.

Trainable scopes (rank 8, alpha 16; counts from the configuration):

| Scope | Matrices | Adapter parameters |
| --- | --- | --- |
| Attention Q/K/V/O, all layers (pilot) | 96 | 2.56 M |
| + all experts of layers 20–23 | + 3,072 | 65.5 M |
| every expert | 18,432 | ~380 M |

Frozen matrices are ternarized once. **Memory decision (§9 D3):** keep frozen ternary weights as BF16 `{−α,0,α}`
(40 GB, simplest, any kernel) or as FP8 codes plus a per-row scale (20 GB, exact, FP8 matmuls: §3.1).

### 3.1 Low-precision compute (FP8)

The BF16 checkpoint is only the *latent* weight. What the forward actually uses is `{−α, 0, +α}` per row, which is
**exact in FP8** (codes ±1/0 in e4m3, the row's `α` as the scale). So:

- **Frozen matrices** (everything outside the trainable scope: all experts in the pilot) need no BF16 latent at all:
  store FP8 codes plus a per-row scale (20 GB instead of 40) and feed them to FP8 matrix multiplies with no weight
  error. Only the trainable scope keeps a BF16 latent, because `Q(W + ΔW)` re-thresholds the latent values
  (attention, all layers: ~0.5 GB; experts of layers 20–23: ~6.4 GB more).
- **Activations in FP8** are an approximation Maple was not trained with. The deployed runtime itself quantizes
  activations (llama.cpp's `TQ2_0` dot product runs against 8-bit `Q8_K` activation blocks), so 8-bit activations
  are not foreign to deployment; still, it is checked (M0.3: NLL and agreement with FP8 activations vs BF16).
- **Measured on the GB10** (torch 2.11, while the campaign runs): dense BF16 92 TFLOP/s; dense FP8 175 (per-tensor
  scale) / 166 (per-row) — ~1.8×. Grouped BF16 (all experts in one call) 74. **Grouped FP8 is not available** on
  this GPU in PyTorch (`_scaled_grouped_mm` supports compute capability 9.0/10.0 only; GB10 is 12.1), so FP8
  experts need either a per-expert loop of FP8 matmuls or a Triton grouped kernel. Plan: BF16 grouped first
  (correctness), then a Triton FP8 grouped kernel if expert matmuls dominate the step profile.
- **NVFP4 (4-bit) storage is exact too.** Ternary codes ±1/0 are representable in FP4 (e2m1). NVFP4's block scale
  is FP8 (e4m3), which cannot hold an arbitrary `α` exactly, so set every block scale to 1 and apply the row's `α`
  after the matmul (one multiply per output column). That gives ~0.56 bytes per weight: **~11 GB** for Maple instead
  of 40 (BF16) or 20 (FP8). Measured: dense NVFP4 matmul 256 TFLOP/s on the GB10 (2.8× BF16); MXFP4 is not offered
  by PyTorch here. The catch: NVFP4 matmuls need *both* operands in FP4, and 4-bit activations are far coarser than
  the deployed runtime's 8-bit ones. So the practical form is FP4 *storage* with BF16 or FP8 activations
  (unpack in a Triton kernel), not FP4 compute; FP4 compute would be a measured experiment.
- **Gradient checkpointing is not needed in any variant.** Memory here is dominated by weights, not activations:
  saved activations for backward are ~1.4 MB per token over 24 layers (routed inputs, expert intermediates,
  attention), ~12 GB at 8k tokens per step. FP8/FP4 storage frees memory for larger batches (or to fit beside the
  teacher campaign), not to avoid checkpointing.
- **Backward** with frozen experts only needs gradients with respect to activations (`dX = dY · Wᵀ`), which uses the
  same exact FP8 weights; gradients themselves stay BF16.
- **Qwen3-0.6B** (joint plan) trains in BF16: at 0.6B parameters FP8 saves little and adds risk to the student.
- The vocabulary projection (151,936 × 2048) and the distillation logits stay BF16/FP32: the KL is sensitive to
  logit precision.

## 4. Training on our data

- **Corpus.** The natlang SFT corpus (v13 render) re-rendered with Maple's own chat template
  (`chat_template.jinja`: ChatML, `<think>` blocks, Qwen-style JSON tool calls). The runtime already speaks this
  format to the Qwen3.6 teacher; the renderer must emit it for Maple without the LFM pythonic tool-call form.
- **Thinking.** Maple is a reasoning model. Natlang turns are short tool calls; train with an empty think block by
  default, keep the reasoning mode available, and measure both (§9 D5).
- **Loss.** Maple's model code calls a non-standard loss convention; we never use it: our backbone returns hidden
  states and the trainer computes chunked cross-entropy over the vocabulary (151,936 columns: never materialise
  all token×vocab logits for backward), normalised over supervised tokens across accumulation.
- **Precision.** Adapter masters and optimiser state FP32; quantizer statistics FP32 after BF16 rounding; router in
  FP32 outside autocast; norms as the runtime.
- **Selection.** Checkpoints are chosen on the exported model (ternarized, served by llama.cpp) on the protected
  execution evaluation, never on latent-BF16 loss alone.

## 5. Neuralese on Maple

The port design carries over (one write procedure, read by splicing, learned stopping; S3), with these Maple
specifics:

- **Backbone.** Our own layer-range backbone (`natlang_neuralese/model/hf_port.py`), not Maple's remote code
  (written for Transformers 4.57; ours is 5.18; it imports FlashAttention, unavailable on the GB10). It implements
  sliding-window and global attention with per-layer caches (a sliding layer keeps at most 512 keys), partial
  rotary, NoPE on global layers, top-8 routing in FP32 with normalised weights and the SwiGLU clamp. Parity: against
  llama.cpp's Maple graph on the official GGUF (logits on fixed tokens), the same check the port has on LFM2.
- **Cutoff.** "Just after an attention layer" means after a global layer: k ∈ {4, 8, 12}. Sweep as S3 §7 did.
- **Markers.** The vocabulary has 267 unused rows (151,669–151,935). `<|neuralese|>` and `<|/neuralese|>` take IDs
  151,669 and 151,670 for Maple and Qwen3 alike (same tokenizer, so the same marker IDs: §8). The head is untied, so
  each marker has a trainable input row and a trainable output row.
- **Quantization.** The port trains against the deployed backbone: the frozen backbone is the ternarized model,
  phase F's LoRA is the QAT parameterization above. Port heads (feedback, content, stop) stay floating point; they
  ship as the existing `neuralese.gguf` heads file.
- **Serving.** The fork's Neuralese service builds LFM2 layer-range graphs today; it needs the same for Maple (and
  Qwen3). Until then, Maple Neuralese runs on the reference server.
- **Browser.** 6.35 GB exceeds wasm32's 4 GB. Maple is the native and laptop model; the browser keeps a small model
  (Qwen3-0.6B, which shares the Neuralese space: [MAPLE_QWEN_JOINT.md](neuralese/MAPLE_QWEN_JOINT.md)).

## 6. Stages and gates

Gates are reviews (no fixed thresholds), as everywhere in this programme.

| Stage | Work | Review |
| --- | --- | --- |
| M0 Baseline | M0.1 pin BF16 (`ac1ddd79…`) and GGUF (`f5466f91…`) revisions. M0.2 verify the official GGUF = Maple rule on the BF16 weights (codes and scales). M0.3 our backbone on the ternarized weights vs llama.cpp on the official GGUF: next-token agreement and NLL on fixed sequences. M0.4 protected execution evaluation of the official model through our runtime. | Representations agree; baseline quality known. |
| M1 QAT pilot | Attention-only rank-8 QAT on the natlang SFT corpus; export; evaluate through llama.cpp. | The gain survives export; ordinary capability holds. |
| M2 Candidate | Wider scope if M1 warrants (layers 20–23 experts); selection on the exported model. | Beats the official model and our LFM student on the protected evaluation at their speeds. |
| M3 Neuralese | S3 curriculum on Maple (A–F), cutoff sweep, with the LFM full run's settings as the start. | S3's G1 review on Maple. |
| M4 Joint | Qwen3-0.6B trained beside Maple with distillation, shared Neuralese space ([MAPLE_QWEN_JOINT.md](neuralese/MAPLE_QWEN_JOINT.md)). | Its own reviews. |
| M5 Serving | Fork: Maple and Qwen3 layer-range graphs for Neuralese; conformance against the reference. | Conformance passes. |

## 7. Compute

The DGX GB10 has 128 GB of unified memory; the teacher campaign holds ~60 GB while it runs.

| Job | Resident | Fits beside the campaign? |
| --- | --- | --- |
| M0 checks (CPU or GPU, no training) | ~45 GB BF16, or stream shards | yes, one shard at a time |
| M1 attention-only QAT, frozen BF16 | ~40 GB weights + ~10–20 GB activations | no |
| M1 with FP8 frozen experts | ~20 GB + ~10–20 GB | marginal (ledger headroom ~38 GB) |
| M4 joint (Maple + Qwen3-0.6B full FT + ports) | ~75–90 GB | no: needs the campaign paused |

No gradient checkpointing (owner policy); MoE activations are small (1B active), so sequence length and batch set the
activation budget. Expert execution: grouped by expert (sort tokens by routed expert, one matmul per used expert);
the reference loop over 256 experts per layer is too slow.

## 8. Tokenizer facts

Maple's `tokenizer.json` equals Qwen3's (vocabulary, merges, pre-tokenizer, normalizer, and all 26 added tokens with
the same IDs; checked against `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B`). Qwen2.5 has the same vocabulary and
merges; it only lacks four added tokens (`<tool_response>`, `</tool_response>`, `<think>`, `</think>`, IDs
151,665–151,668), which are unused rows of its embedding. Qwen3.5 uses a new 248k vocabulary: not compatible.
deepgrove's own small model, Bonsai (0.5B, March 2025), does not share it: Llama architecture, Mistral 32k tokenizer,
2,048-token context, base model only, trained on under 5B tokens.

## 9. Decisions (owner)

| # | Decision | Options | Recommendation |
| --- | --- | --- | --- |
| D1 | Deployment target | llama.cpp TQ2_0 (our stack, CPU-fast, browser-capable runtime) / MLX 2-bit (Apple only) | llama.cpp; MLX optional |
| D2 | Small partner | Qwen3-0.6B (identical tokenizer) / Bonsai (the "sibling": different tokenizer, base only, 2k context) / our LFM2.5-350M (different tokenizer) | Qwen3-0.6B |
| D3 | Frozen-weight storage and compute | BF16 ternary (40 GB) / FP8 codes + row scale (20 GB, exact, ~1.8× dense matmuls) / NVFP4 codes + row scale (11 GB, exact, needs a Triton unpack kernel) | FP8 storage first (simplest exact kernels); NVFP4 storage (11 GB) if the joint run must fit beside the campaign; FP8 activations after a precision check |
| D4 | When to run M1+ | needs ~60–90 GB: campaign pause (as the S3 full run) | schedule with the S3 full run's pause, after it |
| D5 | Thinking in natlang turns | empty think / short reasoning | empty by default, measured both ways |
| D6 | QAT for the small partner | keep Qwen3-0.6B in BF16 / ternary QAT too (~150 MB weights, a browser-sized Maple-like student) | BF16 first; ternary as an experiment |

## 10. References

- BF16 model and code: `deepgrove/maple-preview` at `ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07`
  (local: `/home/werg/data/models/maple-preview-bf16`).
- Official GGUF: `deepgrove/maple-preview-GGUF` at `f5466f918e0c50cdb9d4d47a6f35813509a42a30`, file
  `maple-preview-TQ2_0-head-F16.gguf`, SHA-256 `2fad7b49…039`.
- Ternary rule: deepgrove `mlx-lm-deepgrove/mlx_lm/ternary.py`; llama.cpp `quantize_row_tq2_0_ref`
  (`ggml/src/ggml-quants.c`).
- Upstream support: llama.cpp PR #27000 (`3d10bcd19`); `conversion/maple.py`.
- Evaluation session: `plans/HANDOVER.md` 2026-10-05 Maple entries; `runs/maple-preview-evaluation-20261005/`.
