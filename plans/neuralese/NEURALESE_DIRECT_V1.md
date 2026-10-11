# Neuralese direct v1: plain target SFT with a NeuraleseProjection (TokenChoiceProjection declared, off)

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
3. **Two separate projections, kept visibly separate.** Each has its own module, name, recipe key, checkpoint key
   and tests. Both are built from one shared block, `CausalConvStack`, so the convolution is implemented once.
   - **CausalConvStack:** `blocks` (2) residual blocks. Each block is RMS-norm → depthwise causal conv over positions
     (kernel 4) → pointwise MLP (width → `hidden` → width, SiLU) → zero-initialised output, added to the residual.
     The default `hidden` is width/4: Mellum (2304) has about 5.3M parameters per stack, LFM-350M (1024) about
     1.1M.
   - **NeuraleseProjection** (`neuralese_projection`; checkpoint key `neuralese_projection`) produces the
     Neuralese output: the value passed on and fed back. It is the successor of the old content projection. See
     4 for its space, initialisation and target.
   - **TokenChoiceProjection** (`token_choice_projection`; checkpoint key `token_choice_projection`) maps the last
     hidden state into the hidden space that the frozen final norm and lm_head read, for token prediction.
     - It is identity-initialised (zero-initialised residual), so at step 0 the logits equal base.
     - It is a recipe-declared stage component and is **off in this stage**. Token prediction here is plain
       lm_head(norm(h)).
     - It switches on in the stage that trains the backbone end-to-end on Neuralese, where h is meant to drift and
       token choice needs its own adapter.
4. **The Neuralese value: space, initialisation and target (proposal).**
   - **Space: the input token-embedding space.** Consumers read ordinary token embeddings (2), so a value in that
     space can be fed back or passed on with no read adapter. It is also comparable, position for position, with
     what the consumer would otherwise read.
   - **Initialisation: the model's own soft-token mixture.** value(h) = softmax(lm_head(norm(h)))·E + stack(h),
     with the stack's output at zero. At step 0 the value is exactly the expected next-token embedding under the
     base model's own prediction. That is the owner's "soft-token mixture is the natural initialisation", and
     unlike the old zero-initialised content head it starts as a meaningful, consumer-readable value. Audit
     finding 4 showed that head never beat its zero init.
   - **Target in this stage:** the gold next token's embedding E[y], by relative MSE, at Neuralese (supervised)
     positions.
     - The mixture is the squared-error-optimal estimate when the model is calibrated, so the stack learns what
       the mixture misses, e.g. sharpening where the model is confident and right.
     - The projection reads h **detached**. Its objective cannot bend the backbone (audit 4); the backbone learns
       from the SFT CE alone.
     - It is reported against the mixture-only baseline (relative MSE delta, which must fall below 0) and as
       nearest-embedding accuracy.
   - **Later:** when values are fed back (the recurrence stage), the target becomes the consumer's need: the
     consumer's CE when it reads the value. That is when the backbone trains end-to-end on Neuralese and the
     TokenChoiceProjection switches on.
5. **Start from the base models.** BF16 Mellum and base LFM. There is no warm-up stage and no projection-first
   phase: the backbone and P train from update 1.

## Data and labels

- **Inputs:** the admitted native records and pieces (`gold_text_rows`), split by source group, as before.
- **Failed assistant context is never a positive target** (Pop, 2026-10-11).
  - Some assistant turns are answered by a runtime failure. Three forms count, as ts-host/src/native/runtime.ts
    words them (`text_supervision.FAILED_TOOL_FEEDBACK`):
    - a rejected action (`rejected\ncode: …`);
    - a diagnostic that discarded the eval ("Nothing else from this eval was kept");
    - a return value refused as the result (", so it is not the result").
  - Such turns are masked (`text_supervision.failed_assistant_mask`, applied through
    `balanced_position_weights(exclude=...)`). They stay in the context, but carry zero weight in both halves of the
    supervision.
  - A window made only of failed turns contributes no update.
  - In the assembled v3 records the forms occur 153, 349 and 859 times among 10,916 tool responses (some responses
    match more than one form).
  - Turns that were later corrected without a runtime failure cannot be told apart from the data. A producer
    therefore declares them: a text row's `context_only_spans` (token coordinates `[begin, end)` in its
    `token_ids`) are masked the same way. Example: excluded bad assistant targets kept in retained histories.
    Pop's V73 derivative needs this for 82 histories.

## Monitors (both backbones, no preservation loss)

- **Drift:** `generation_gate.held_kl` on `drift_rows` held rows at every cadence eval, reported as response KL,
  prompt KL and agreement. The reference comes from one of two sources:
  - the quantization component's `preserve.teacher` artifact, used as held data only (Mellum); or
  - **`monitor.data`**: chat prompts whose base-model answers and top-k are captured in-run before the first
    update (`generation_gate.capture_reference`, kept as `monitor-reference.pt`). This is the shared capture path
    for a lineage without a teacher artifact (LFM). It works on any backbone (`greedy_any`).
- **Guard (owner, 2026-10-11):**
  - Response KL above `drift_warn` (0.05), or agreement below 0.97, is a warning. At that eval the behaviour gate
    (24 probes, texts kept) runs, and the stage stops when the gate falls below its reference minus
    `max_gate_drop`.
  - Response KL above `drift_stop` (0.5) stops at once.
  - The gate runs only while drift is past warn, which keeps the eval share in budget.
- **Behaviour gate:** `quantization.gate`, or `monitor.gate` without a quantization component. Its greedy decode is
  the Maple port's, so on other backbones the gate is reported unavailable and drift still runs.

## Carried over

- **Windows include the system prompt** (audit 1): `--tokens 8192 --target-tokens 1024`, or the length of the
  longest leading system prompt plus the target, whichever is larger.
- **Optimizers:**
  - Backbone attention and expert matrices step on the latent optimizer partition (LionSR or MuonSR; the SFT
    sweep decides the optimizer and lr).
  - Routers are not on any sign or matrix optimizer (audit 6). They use AdamW at the backbone lr, like the norms.
  - The projections use the shared Muon/AdamW (PortMuonAdamW) at the head lr.
- **Monitors:** see "Monitors" above.
- **QAT stays woven in.** The quantization component's precision points are sampled per update exactly as before,
  and are trained on the **same SFT targets**.
  - **FLAG for the owner:** with self-distillation out, there is no BF16 KL teacher for the quantized points.
  - A component that declares a preserve stream with weight > 0 is refused.

## Where it lives (one implementation per concept)

| part | decision |
|---|---|
| Stage | New recipe stage kind `neuralese_direct`, trainer module `train/direct_sft.py`. The old trainer's objective, schedule and gates are all mapped-history specific, so the new objective is a small module rather than a mode flag across ~3k lines. It reuses every shared concept: windows and weights (`text_warmup.prepare_text_windows`, `load_text_rows`, `text_supervision`), readout CE (`text_warmup.chunked_readout`), optimizer partition (`optim.PortMuonAdamW` + LionSR/MuonSR), quantization (`train/quantization.py`), drift and gate (`generation_gate`, `text_warmup.behaviour_drift_level`), loop/stop/checkpoint (`loop.py`, `checkpoint_policy.py`) and backbone policy (`backbone_policy`). |
| Projections | Own module `model/neuralese_projection.py`, holding `CausalConvStack`, `NeuraleseProjection` and `TokenChoiceProjection`. Each is saved under its own key in the stage artifact, next to the backbone latents. |
| Window length, projection sizes, `token_choice_projection` on/off, `neuralese_weight`, optimizers, lrs | Recipe parameters. |
| Recipes | `recipes/neuralese-direct-v1.json` (shared), with `neuralese-direct-mellum-v1` and `neuralese-direct-lfm-v1` via extends/overrides. Backbone-inherent differences: model, MoE kernel, the latent partition for Mellum's dense experts, projection width. |
| Old core_text_warmup / input_map / two-pass code | Kept unchanged for the existing lineages (run-v9, Pop's V8) and their artifacts. No new recipe uses it. It is removed once those lineages are retired (owner decision). |

## Qualification (v1)

The stage report lists:
- held CE against base on the same windows, plus gold accuracy and agreement with base;
- the Neuralese value's relative MSE to the gold embedding against its mixture baseline, and nearest-embedding
  accuracy;
- drift (response KL and agreement) at every eval;
- the final behaviour gate.

The stage passes when held CE is below base, the drift guard never stopped, and the behaviour gate passes against its
recipe thresholds.

## Exact semantics (answers to Pop, pop-f7c5 / pop-b359)

1. **Positional shift and Neuralese span.**
   - A window is `ids = prefix ++ span`. The backbone runs once over `prefix ++ span[:-1]` on ordinary token
     embeddings.
   - With `start = len(prefix) - 1`, the hidden state at absolute position `start + j` predicts `span[j]`. That is
     the standard next-token shift: h_t predicts token t+1.
   - **Neuralese positions** are exactly those read positions `start .. start + len(span) - 1`, the window's
     supervised targets. The prefix, including the whole leading system prompt, is context and never a Neuralese
     position.
   - Positions inside the span with weight 0 (e.g. a leading system message reaching into a span) are masked out of
     every loss.
   - The Neuralese value at position t is a value *about token t+1* (target E[y_{t+1}]).
2. **Re-entry at call and return boundaries.**
   - In this stage nothing is re-entered. Every input, inside and outside Neuralese spans, is the ordinary
     embedding of the gold token, and there is no reader mapping or read adapter.
   - Once values are fed back (the recurrence stage), a value v_t, which lives in input-embedding space, enters as
     the input at position t+1 in place of E[token_{t+1}], within the producing call's Neuralese span.
   - At a call's return, the returned values enter the caller's context as input embeddings at the return
     positions.
   - A read adapter is not part of the design. If the recurrence stage measures that one is needed, it becomes its
     own declared component, separate from both projections.
3. **The causal conv.**
   - Depthwise, left-padded by kernel−1, so the output at t reads h_{t−k+1..t}. There is no look-ahead.
   - It takes a `segments` tensor (one id per position). A tap contributes only when its position has the same
     segment id as t, so the conv state resets at every segment boundary.
   - In training, a segment is a maximal run of one chat role inside the supervised span, i.e. one turn. Context
     positions have no segment. No state crosses a document (windows are single documents), turn or call
     boundary.
   - At serving time each Neuralese call span is its own segment.
4. **Stop and readout.**
   - Token prediction, including the stop token `<|im_end|>`, is the unchanged base readout: frozen final norm +
     lm_head on h.
   - The TokenChoiceProjection is absent in this stage (recipe `token_choice_projection: null`). When declared
     (the end-to-end Neuralese stage), the readout becomes lm_head(norm(TokenChoiceProjection(h))), identity at its
     first step.
   - The NeuraleseProjection reads h and the base readout's distribution at the same position (for its mixture).
     It never feeds the token readout.
5. **Gradient semantics.**
   - **Backbone:** trained by the weighted whole-trajectory CE only. Attention and expert matrices use the latent
     optimizer; routers and norms use AdamW at the backbone lr.
   - **Embedding, final norm and lm_head:** frozen.
   - **NeuraleseProjection:** trained by relative MSE to E[y_{t+1}] at Neuralese positions, at `neuralese_weight`.
     It reads h and the mixture detached, so its gradient never reaches the backbone.
   - **Quantization:** precision points are sampled per update as before and trained on the same CE. There is no
     BF16 KL teacher, which is flagged for the owner.
6. **Qualification.**
   - **Step 0:** two tests — TokenChoiceProjection(h) = h exactly with logits equal to base, and
     NeuraleseProjection(h) = softmax(lm_head(norm(h)))·E exactly. The trainer also checks the latter on its first
     held batch before training.
   - **Runtime qualification:** the stage report's evals and behaviour gate run on the exact trained weights, and
     the artifact records their hashes.
   - No warm-up, foundation or channel certificate of an old lineage carries over. A direct lineage starts from the
     base model and qualifies itself.

**LFM** (Pop): base LiquidAI/LFM2.5-350M at revision 9e6c6ccf, Transformers 5.19, on an 8 GB GPU.
- The context budget is set in `neuralese-direct-lfm-v1` overrides: `tokens`, `target_tokens`, `batch`.
- Example: `tokens` covering the longest system prompt plus 256 targets, at batch 1.
