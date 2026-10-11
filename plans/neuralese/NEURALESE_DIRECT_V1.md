# Neuralese direct v1: plain target SFT with an identity-initialised readout projection

Owner decision, 2026-10-11. It replaces the warm-up audit fixes (4), (5) and (7) and the variant A/B plan. It applies
to both lines (Mellum, LFM) with the same shared code, and only declared, backbone-inherent differences.

Evidence:
- plans/mellum-port.md "Behaviour gate bug hunt";
- plans/neuralese/WARMUP_AUDIT_20261011.md.

## What the stage does

1. **Plain target SFT.** Whole-trajectory CE on the native records' supervised spans, with the existing
   `text_supervision.balanced_position_weights`:
   - assistant targets at weight 1;
   - inputs and prompts at the context weight;
   - tool output at the feedback weight (0.25);
   - the leading system prompt is context only.

   There is no self-distillation or preserve stream.
2. **One pass on ordinary token embeddings.** There is no input map, no mapped history and no second pass. Every
   input position, including Neuralese positions, reads the ordinary embedding of its token (gold during training).
3. **Readout projection P.** It maps the backbone's last hidden state h into the hidden-state space that the frozen
   final norm and lm_head read. Token logits are lm_head(norm(P(h))).
   - **Architecture:** `blocks` (2) residual blocks. Each block is RMS-norm → depthwise causal conv over positions
     (kernel 4) → pointwise MLP (width → `hidden` → width, SiLU) → zero-initialised output, added to the residual.
     P is therefore exactly the identity at step 0, and the model is exactly the base model.
   - **Default size:** `hidden` = width/4. Mellum (2304): 2 × (4·2304 + 2·2304·576 + 2304) ≈ 5.3M parameters.
     LFM-350M (1024): ≈ 1.1M.
   - **Neuralese positions:** the stage's supervised span positions, the same positions the old warm-up treated as
     Neuralese. Context positions are never read out. P runs causally over the whole window, so each read
     position's conv sees its real predecessors.
4. **The Neuralese value is P(h).** P(h) lives in the space the frozen readout reads, so every Neuralese value has an
   exact token reading, lm_head(norm(P(h))). That gives a crisp verifier of what a stage produces.
   - At step 0, P(h) = h, the base model's own state.
   - Feedback into the model (later stages) can start from the soft-token mixture softmax(logits)·E of that same
     reading.
   - h itself stays the backbone's private state; P absorbs the readout adaptation.
5. **Start from the base models.** BF16 Mellum and base LFM. There is no warm-up stage and no projection-first
   phase: the backbone and P train from update 1.

## Carried over

- **Windows include the system prompt** (audit 1): `--tokens 8192 --target-tokens 1024`, or the length of the
  longest leading system prompt plus the target, whichever is larger.
- **Optimizers:**
  - Backbone attention and expert matrices step on the latent optimizer partition (LionSR or MuonSR; the SFT
    sweep decides the optimizer and lr).
  - Routers are not on any sign or matrix optimizer (audit 6). They use AdamW at the backbone lr, like the norms.
  - P uses the shared Muon/AdamW (PortMuonAdamW) at the head lr.
- **Monitors:** behaviour_drift on response KL and agreement, with its guard, at every cadence eval. The behaviour
  gate (texts kept) runs at the end. The teacher's held rows are data for measurement only.
- **QAT stays woven in.** The quantization component's precision points are sampled per update exactly as before,
  and are trained on the **same SFT targets**.
  - **FLAG for the owner:** with self-distillation out, there is no BF16 KL teacher for the quantized points.
  - A component that declares a preserve stream with weight > 0 is refused.

## Where it lives (one implementation per concept)

| part | decision |
|---|---|
| Stage | New recipe stage kind `neuralese_direct`, trainer module `train/direct_sft.py`. The old trainer's objective, schedule and gates are all mapped-history specific, so the new objective is a small module rather than a mode flag across ~3k lines. It reuses every shared concept: windows and weights (`text_warmup.prepare_text_windows`, `load_text_rows`, `text_supervision`), readout CE (`text_warmup.chunked_readout`), optimizer partition (`optim.PortMuonAdamW` + LionSR/MuonSR), quantization (`train/quantization.py`), drift and gate (`generation_gate`, `text_warmup.behaviour_drift_level`), loop/stop/checkpoint (`loop.py`, `checkpoint_policy.py`) and backbone policy (`backbone_policy`). |
| P | Own module `model/neuralese_readout.py` (`NeuraleseReadout`), saved in the stage artifact next to the backbone latents. |
| Window length, P size, optimizers, lrs | Recipe parameters. |
| Recipes | `recipes/neuralese-direct-v1.json` (shared), with `neuralese-direct-mellum-v1` and `neuralese-direct-lfm-v1` via extends/overrides. Backbone-inherent differences: model, MoE kernel, the latent partition for Mellum's dense experts, P width. |
| Old core_text_warmup / input_map / two-pass code | Kept unchanged for the existing lineages (run-v9, Pop's V8) and their artifacts. No new recipe uses it. It is removed once those lineages are retired (owner decision). |

## Qualification (v1)

The stage report lists:
- held CE against base on the same windows, plus gold accuracy and agreement with base;
- drift (response KL and agreement) at every eval;
- the final behaviour gate.

The stage passes when held CE is below base, the drift guard never stopped, and the behaviour gate passes against its
recipe thresholds.
