# Mellum 2.1 as a Maple replacement — evaluation and port plan (2026-10-08)

Owner request: evaluate JetBrains Mellum2.1-12B-A2.5B-Thinking (Apache-2.0). Switch from Maple if it trains faster
and does acceptably at raw agentic coding in our harness out of the box. If adopted: relatively aggressive QAT.

## The model

- 28 layers, width 2304, GQA 32/4, head_dim 128, q/k RMSNorm.
- 64 experts, top-8, renormalized; expert FFN 896; no shared expert; untied head; vocab 98,304.
- 3 sliding (window 1024) : 1 global, the same pattern as Maple.
  - RoPE θ=500k on all dimensions; YaRN (factor 16, original 8192) on global layers only.
- Released in BF16. 12B total, 2.5B active per token. Maple: 20B total, about 1B active, published ternary.
- Self-reported scores: SWE-bench Verified 47.0, Terminal-Bench 2.1 17.4, BFCL v4 62.3, LiveCodeBench v6 82.0.
- Mellum 2 (June 2026) also ships Base/Instruct/Thinking, intermediate pre-training checkpoints and a technical report
  (arXiv 2605.31268). No training code.

## Existing support (reusable as is)

| Stack | Status |
|---|---|
| vLLM (`vllm-node` image) | `MellumForCausalLM`, per-layer sliding window and per-layer RoPE; `--reasoning-parser qwen3 --tool-call-parser hermes` |
| transformers 5.19 (training container) | native `models/mellum`; exact reference; expert loop is Python unless `experts_implementation=grouped_mm` |
| Our llama.cpp fork | `LLM_ARCH_MELLUM` (MoE, SWA pattern 4, plain RoPE on sliding / YaRN on global), `mellum2` pre-tokenizer, HF→GGUF converter `conversion/mellum.py` |
| JetBrains GGUFs | BF16, Q8_0, Q6_K, Q4_K_M, MXFP4_MOE |

## Done

- 9d55780b — the Maple family implementation loads Mellum exactly:
  - `MapleConfig` gains `model_type`, `swiglu_clamp`, `rope_parameters` and `dense_experts`.
  - Per-layer-type RoPE uses transformers' own `MellumRotaryEmbedding`.
  - `DenseExperts` keeps exact BF16 experts; the SwiGLU clamp is optional.
  - Exact logit parity with transformers on a tiny random Mellum (`tests/neuralese/test_mellum_port.py`).
  - So the Maple port runner, trainer, fused-kernel path and QAT code apply to Mellum.
- `training/neuralese/scripts/bench_hf_training_step.py`: native-transformers forward / forward+backward timing on a
  real token window.
- Evaluation harness prepared: `runs/mellum-evaluation-20261008/` (vLLM serve script, runner copy).
  - Same 23-task protected packet, limits and greedy decoding as the Maple/Ling comparison
    (`plans/student-candidate-comparison-20261005.md`).

## Next, in order

1. Real-checkpoint parity on GPU: our loader vs transformers on the downloaded weights (first logits, held-text CE).
2. Harness evaluation: vLLM BF16, thinking on (as Ling), then thinking off.
   - Baselines: Maple 12/23, Ling 11/23 / 10/23.
3. Training speed:
   - Our trainer's warm-up step on the same windows as Maple v10. Maple v10: ~11 s per two-pass update over ~5.5k
     tokens with QAT.
   - Native-transformers bench as a cross-check.
4. Neuralese markers: Mellum has no unused vocabulary rows; use the spare added tokens `<|extra_token_7|>` and
   `<|extra_token_8|>` (IDs from the tokenizer). `serve.load_engine` and `foundation_heads` currently hard-code
   Qwen/Maple marker IDs (151669/151670) and need a per-tokenizer choice.
5. GGUF: convert with the fork's converter, then check perplexity/logit parity vs our port (Maple's M0.3 equivalent).

## QAT design (if adopted)

- Maple's experts were published ternary, so its expert QAT is scale-only (fixed codes, learned 256-column block
  scales).
- Mellum is BF16, so aggressive QAT needs latent weights whose ternary codes are re-derived every step (STE).
  - Experts: ternary codes from latent BF16 weights with absmean scales per 128-column block.
    - The fused kernel's 256 blocks do not divide 896; 128 does (896 = 7×128, 2304 = 18×128).
    - The kernel takes the block size as a parameter.
  - Attention: the Maple attention QAT path (ternarize plus LoRA) applies unchanged.
  - Memory: about 24 GB BF16 latents + 24 GB gradients + Muon momentum 24 GB, plus activations. This fits the GB10
    with checkpointing.
    - Alternatives if tight: QAT layer groups in turn, or train LoRA on frozen ternary codes after a short full-latent
      phase.
- Distill from the BF16 original (KL on logits) during QAT. Same recipe, data and Neuralese stages as Maple; only
  backbone-inherent differences are declared.
