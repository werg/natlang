# Maple-Preview: quantization-aware fine-tuning and native-speed deployment

**Audience:** An engineer integrating Maple into an existing fine-tuning pipeline.  
**Source review date:** October 5, 2026.  
**Training assumption:** An existing PyTorch/CUDA pipeline, with its own data loader, distributed training, checkpointing, and evaluation infrastructure.  
**Deployment target:** DeepGrove's Apple Silicon MLX runtime and Maple-specific packed checkpoint—not generic CUDA, GGUF, GPTQ, AWQ, or bitsandbytes inference.  
**Validation status:** Source-reviewed implementation plan. The reference QAT core in Appendix A passed local CPU tests with PyTorch 2.10.0. No full Maple fine-tuning run, CUDA integration test, or MLX conversion/performance test was executed for this document.

## 1. Objective and recommended approach

Replace the pipeline's current model with **Maple-Preview**, fine-tune against its deployment quantization, and produce an **adapter-free checkpoint that retains the original architecture and native inference format**.

The recommended implementation is **quantization-aware low-rank fine-tuning**: keep the pretrained BF16 base frozen, learn FP32 low-rank updates, and apply Maple's hard quantizer to the *merged effective weight on every training forward pass*. After training, materialize that same merged weight as BF16, then run DeepGrove's official ternary exporter.

For a projection, train:

```text
W_effective = W_base + (lora_alpha / rank) × B @ A
W_export    = BF16(W_effective)
W_forward   = MapleTernaryQuantizer(W_export)
y           = linear(x, W_forward)
```

Do **not** substitute the ordinary adapter computation:

```text
y = linear(x, Q(W_base)) + adapter(x)
```

These are different objectives: generally `Q(W_base + ΔW) != Q(W_base) + ΔW`. Training the first form makes the model encounter the weight representation used after export. The low-rank matrices are a training parameterization; they do not remain as inference branches.

This is a proposed engineering recipe, **not a claim to reproduce DeepGrove's unreleased optimizer or original backward estimator**. Full-weight QAT can use the same forward contract; Section 10 describes that extension.

### Deliverables and completion gates

| Stage | Required output | Gate before proceeding |
|---|---|---|
| Establish baseline | Pinned BF16 source, official packed reference, local no-training conversion, baseline metrics | Conversion/version differences are understood. |
| Integrate QAT | Weight-level QAT, deployment-equivalent frozen weights, correct loss and data handling | Quantizer, gradient, precision, and checkpoint tests pass. |
| Run pilot | Small QAT fine-tune and its exported checkpoint | Improvement survives export; general capability remains acceptable. |
| Train candidate | Selected training checkpoint and complete run manifest | Candidate selected using quantized validation, not latent-BF16 scores alone. |
| Export | Clean canonical BF16 checkpoint and compact MLX checkpoint | Structural and numerical export tests pass. |
| Release | Quality, speed, memory, and fast-path report | Agreed budgets pass on the actual target hardware. |

**“Same speed” is an acceptance test, not an automatic consequence of producing a 2-bit file.** Preserve the representation and runtime, then demonstrate comparable throughput on the same hardware and workload.

## 2. Model introduction and required sources

Maple-Preview is DeepGrove's MIT-licensed **20B-total, approximately 1B-active reasoning MoE model**. Its configuration has 24 layers, 256 experts per layer, and eight selected experts per token. The attention pattern repeats three 512-token sliding-window layers followed by one full-attention layer. The advertised context limit is 131,072 tokens. [R1, R2]

The BF16 source repository is approximately **40.4 GB**. Importantly, DeepGrove's converter explicitly identifies its BF16 inputs as QAT-trained weights intended for recovery into the ternary representation. Use this BF16 release as the training source; there is no need to reconstruct floating-point inputs from the packed checkpoint. This does not establish that every original training-state artifact is available. [R3, R4]

| Artifact | Source | Purpose |
|---|---|---|
| BF16 model and custom Transformers implementation | [`deepgrove/maple-preview`](https://huggingface.co/deepgrove/maple-preview) | Training source, tokenizer, configuration, canonical tensor names. |
| Packed reference model | [`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx) | Published deployment reference. |
| Native runtime and exporter | [`deepgrove-ai/mlx-lm-deepgrove`](https://github.com/deepgrove-ai/mlx-lm-deepgrove) | Conversion, packing, inference, FlashHead, kernel tests. |
| Quantizer source | [`mlx_lm/ternary.py`](https://github.com/deepgrove-ai/mlx-lm-deepgrove/blob/main/mlx_lm/ternary.py) | Authoritative forward/export quantization rules. |
| Deployment model source | [`mlx_lm/models/maple.py`](https://github.com/deepgrove-ai/mlx-lm-deepgrove/blob/main/mlx_lm/models/maple.py) | Authoritative deployment semantics and optimized-path checks. |

The inspected BF16 revision is:

```text
ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07
```

Runtime references in this document point to `main` as reviewed on the date above. Resolve and record an immutable Git commit before implementation; archive the reviewed source files. Later revisions must pass the same compatibility tests rather than being silently substituted. [R15]

### Configuration that must remain unchanged

```text
hidden_size             = 2048
moe_intermediate_size   = 512
num_hidden_layers       = 24
num_experts             = 256
num_experts_per_tok     = 8
num_attention_heads     = 16
num_key_value_heads     = 4
head_dim                = 128
vocab_size              = 151936
sliding_window          = 512
partial_rotary_factor   = 0.5
max_position_embeddings = 131072
tie_word_embeddings     = false
```

Preserve the explicit `layer_types`, Q/K normalization, rotary settings, and global-attention NoPE behavior. Do not reconstruct the model as a generic MoE with superficially similar dimensions. [R2]

## 3. Deployment contract: what the training forward must represent

| Component | Deployment representation | Initial training treatment |
|---|---|---|
| Attention Q/K/V/O projection weights | Ternary, one scale per output row, packed into two bits | Apply QAT to selected projections; quantize frozen projections once for the working model. |
| Expert gate/up/down projection weights | Same ternary rule, independently per expert and row | Same policy as attention. |
| MoE routing weights | Not ternarized; routing arithmetic uses FP32 | Freeze initially and enforce FP32 routing computation. |
| Normalization weights | Floating-point, not ternarized | Freeze initially; preserve deployment dtype and normalization arithmetic. |
| Token embedding and output head | Affine 4-bit, group size 64 | Freeze initially; train through their dequantized deployment values. |
| Activations and KV cache | Keep the native runtime's existing floating-point behavior | Do not add activation or KV quantization in this project. |

The compact exporter stores ternary packed weights plus `row_alpha`; the runtime reconstructs repeated group scales and offsets at load. Each `uint32` contains 16 two-bit codes. Codes `0,1,2` correspond to `-alpha,0,+alpha`; code `3` is not used for ternary weights. The logical affine packing group size is 128, but **the scale is per row, not independently learned per group**. [R4, R5]

Keep the final deployment free of adapters, QAT wrappers, training-only tensors, resized embeddings, changed expert counts, and changed attention dimensions. Avoid the compatibility-only `--group-scales` option when targeting the compact layout. Compare checkpoint size against a baseline produced by the **same exporter revision and options**, not against a historical download-size headline.

## 4. Phase 1 — Pin environments and establish a no-training baseline

### 4.1 Training environment

Keep the existing CUDA training environment where possible. Add a **reviewed local copy** of Maple's custom model code and its associated configuration/attention files. Load the pinned checkpoint, not unreviewed moving remote code.

The checkpoint records Transformers `4.57.1`. Treat that as a starting compatibility reference, not proof that the training integration is complete. The model's attention wrapper imports FlashAttention interfaces; installing an arbitrary new Transformers version or specifying an SDPA flag does not automatically replace that code path. Lock the actual PyTorch, CUDA, Triton/FlashAttention, Transformers, and optional Liger versions that pass the tests. [R2, R6]

A starting download command is:

```bash
hf download deepgrove/maple-preview \
  --revision ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07 \
  --local-dir ./maple-base-bf16
```

Load with BF16 base weights and the pipeline's **training-aware** placement/sharding mechanism. Remove the previous model's bitsandbytes/NF4 loading configuration and ordinary LoRA injection for this integration. Do not use an inference-oriented `device_map="auto"` as a substitute for distributed training configuration.

Create trainable parameters before constructing the optimizer or distributed wrapper. Preserve FP32 adapter masters under mixed precision; do not subsequently call `model.bfloat16()` on the complete parameterized model.

### 4.2 Export and inference environment

Use an Apple Silicon Mac. The reviewed runtime README requires macOS 26.2 or later and supplies `setup.sh`. Run conversion and deployment tests in this environment, separately from CUDA training. [R7, R8]

```bash
git clone https://github.com/deepgrove-ai/mlx-lm-deepgrove.git
cd mlx-lm-deepgrove

git rev-parse HEAD > ../maple-runtime-commit.txt
# Archive this commit and do not pull updates during the experiment.
./setup.sh
source .venv/bin/activate

# Resolve the published packed model once, then download that exact revision.
PACKED_REV="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("deepgrove/maple-preview-2bit-mlx").sha)')"
printf '%s\n' "$PACKED_REV" > ../maple-packed-revision.txt
hf download deepgrove/maple-preview-2bit-mlx \
  --revision "$PACKED_REV" --local-dir ../maple-official-2bit

# The BF16 source directory must be copied/downloaded onto this host first.
# Use a new, empty destination.
python -m mlx_lm.ternary ../maple-base-bf16 \
  -o ../maple-base-roundtrip --threshold-scale 0.7

uv pip install pytest
python -m pytest tests/test_maple_kernels.py -v
```

The runtime and model revision locks, environment lockfiles, hardware information, tokenizer hashes, and conversion command belong in the run manifest. The `hf` invocation follows the official Hub download interface. [R16]

### 4.3 Define three baselines

Maintain these as distinct artifacts:

| Baseline | Definition | What it establishes |
|---|---|---|
| B0 | Published packed model on the pinned runtime | Public reference behavior and speed. |
| B1 | Your conversion of the unchanged BF16 source | Your exporter round trip and compact-format reference. |
| B2 | PyTorch working model with deployment-equivalent weight quantization | Training/inference numerical alignment before any optimization. |

Compare B0 with B1, then B1 with B2 on identical token IDs and teacher-forced sequences, with FlashHead disabled. Compare canonicalized tensor representations as well as model outputs: shard boundaries, tensor ordering, or scale-storage layout can differ without changing logical weights.

Record held-out negative log-likelihood, next-token agreement, task results, and intermediate diagnostics. Do not require arbitrary cross-backend bitwise equality of entire logits; first characterize differences from floating-point reductions and kernels. However, substantial unexplained degradation or differing model semantics is a stop condition.

**Raw, unquantized BF16 inference is not the deployment-fidelity baseline.** The reference for this project is the native quantized model.

## 5. Phase 2 — Implement the QAT forward and parameterization

### 5.1 Exact ternary rule

For a row of the **BF16-export-rounded** effective weight, calculate in FP32:

```text
threshold_i = 0.7 × mean_j(abs(w_ij))
mask_ij     = abs(w_ij) > threshold_i
alpha_i     = sum_j(abs(w_ij) × mask_ij) / max(sum_j(mask_ij), 1)
alpha_i     = BF16(alpha_i)
Q(w)_ij     = sign(w_ij) × mask_ij × alpha_i
```

The strict comparison, FP32 statistics, BF16 scale rounding, and all-zero-row behavior are significant. Do not replace this with nearest-neighbor ternarization or a standard uniform two-bit observer. [R4]

For FP32 updates, **simulate BF16 serialization before computing the quantizer statistics**. Otherwise the model trains against `Q(W_fp32)` but ships `Q(BF16(W_fp32))`, introducing another avoidable mismatch.

Use an identity straight-through estimator initially: the forward returns hard deployment weights; the backward passes gradients to the effective floating-point weight. In this reference design, the entire hard quantizer—including its scale computation—uses a surrogate gradient, rather than differentiating the threshold or separately optimizing free scales. Appendix A implements this contract. The backward choice is a proposal to validate experimentally.

### 5.2 Integrate at the weight level

**Observed integration hazards:** Maple's attention and expert code reads `.weight` directly and calls functional linear operations. The configuration flag `quantize=true` does not itself insert the ternary training forward. Its inference-only MoE branch is decorated with `no_grad`. The labels path also uses a loss-call convention that differs from stock Transformers 4.57.1. [R9, R10]

Use `torch.nn.utils.parametrize.register_parametrization` or an equivalent explicit weight transformation at every relevant functional-linear call site. A parameterization intercepts `.weight` access; a conventional module-forward LoRA wrapper may not. PyTorch documents both the property-based behavior and the location of the original parameter. [R11]

Integration sketch, using Appendix A:

```python
import torch
from torch.nn.utils import parametrize
from maple_qat_reference import QATTernaryLoRA, hard_ternary

# Load/move the BF16 base first. Do this BEFORE optimizer/distributed wrapping.
model.requires_grad_(False)

# Enumerate canonical projection modules BEFORE changing the module tree.
# `projection_modules` must cover every attention and expert projection.
for name, module in projection_modules:
    if name in selected_trainable_projection_names:
        p = QATTernaryLoRA(module.weight, rank=8, lora_alpha=16.0)
        parametrize.register_parametrization(module, "weight", p)
    else:
        with torch.no_grad():
            module.weight.copy_(hard_ternary(module.weight))

# Frozen head/embedding working copies are replaced separately: Section 5.4.
# Construct the optimizer only after parameter registration.
trainable = [p for p in model.parameters() if p.requires_grad]
assert trainable
```

The two enumeration variables above are integration points in the existing pipeline, not upstream APIs. Match these canonical module-name families:

```text
model.layers.{layer}.self_attn.{q_proj,k_proj,v_proj,o_proj}
model.layers.{layer}.mlp.experts.{expert}.{gate_proj,up_proj,down_proj}
```

With this configuration, expect **96 attention matrices and 18,432 expert matrices**. These counts are calculated from the configuration. Exclude `model.layers.{layer}.mlp.gate.weight`: that is the **router**, not an expert's `gate_proj`.

Frozen working weights may be quantized once to avoid recomputing unchanged tensors. Retain the immutable original BF16 checkpoint on disk, because the working model is **not** the canonical export source for frozen weights.

### 5.3 Initial trainable scope

Start with an **attention-only integration pilot**: Q/K/V/O in all 24 layers, rank 8, `lora_alpha=16`, zero-initialized B, randomly initialized A, no adapter dropout. This contains approximately **2.56 million trainable adapter parameters**, calculated from the published dimensions.

For the first more expressive candidate, add all gate/up/down projections of all experts in the final four blocks, layers **20–23**. At the same rank this brings the total to approximately **65.47 million parameters**. These are proposed experiment scopes, not validated Maple hyperparameters. If attention-only adaptation already meets the task and retention criteria, retain the smaller scope.

Expanding to every expert in every layer reaches approximately **380 million rank-8 adapter parameters**. Do not assume that “LoRA” or “1B active” makes an all-expert training configuration small.

### 5.4 Include frozen 4-bit components in the simulation

Freezing the embedding and head is not sufficient if they remain raw BF16 during training but become four-bit at deployment.

For the initial implementation, use B1 to produce fixed dequantized BF16 copies of its quantized embedding and output head. Load these **only into the PyTorch working model**, leave them frozen, and allow gradients to propagate through their computations into trainable upstream layers. Appendix B provides an MLX extraction script.

During final export, copy the original BF16 embedding/head tensors from the immutable source checkpoint. The pinned exporter will reproduce the same four-bit tensors. **Do not feed the dequantized working copies back into a second round of four-bit quantization.**

If these components later become trainable, implement and test the exact MLX four-bit quantize/dequantize forward against the exporter. Generic NF4 or an assumed min/max formula is not an interchangeable replacement.

## 6. Phase 3 — Adapt the existing training pipeline correctly

### 6.1 Loss, gradients, and precision

Implement a pipeline-owned loss adapter rather than assuming `model(..., labels=...)` is directly compatible. The inspected Maple call supplies hidden states, head weights, and labels to `self.loss_function`, whereas the stock 4.57.1 causal-LM loss expects logits, labels, and vocabulary size and returns a scalar. [R9, R10]

For a **small smoke test**, the reference computation is:

```python
import torch.nn.functional as F

# Run model.train() before training, including for frozen MoE blocks.
out = model.model(
    input_ids=batch["input_ids"],
    attention_mask=batch.get("attention_mask"),
    position_ids=batch.get("position_ids"),
    use_cache=False,
    return_dict=True,
)
h = out.last_hidden_state
logits = F.linear(h[:, :-1, :], model.lm_head.weight)
targets = batch["labels"][:, 1:]
loss = F.cross_entropy(
    logits.float().reshape(-1, logits.shape[-1]),
    targets.reshape(-1),
    ignore_index=-100,
)
```

Use this to verify shifting and masking, not as the production long-context loss implementation. The vocabulary is large. Integrate the pipeline's validated fused or checkpointed/chunked linear-cross-entropy path, and confirm that it really avoids retaining all token-by-vocabulary logits for backward. Simply accumulating separate cross-entropy chunks can still retain their activations.

Normalize consistently over **supervised token counts**, including gradient accumulation and distributed reduction. Do not shift twice, average equally over differently sized token batches unintentionally, or accept batches with no supervised tokens.

Enforce these precision requirements:

| Area | Required integration behavior |
|---|---|
| Adapter A/B and optimizer state | FP32 masters. Compute `B @ A` and latent merging outside autocast. |
| Quantizer statistics | FP32, after BF16 export rounding. |
| Router | Disable autocast around FP32 linear, softmax, top-k scoring, and renormalization. Casting inputs to FP32 alone can be insufficient inside autocast. |
| Normalization | Match the native runti