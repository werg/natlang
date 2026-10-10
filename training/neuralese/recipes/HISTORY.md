# Recipe history

Experiment history moved out of `description` for recipes that use `extends`/`overrides` (plans/ARCHITECTURE_IMPROVEMENT.md C3). Each entry is the original description, verbatim. Base: `foundation-maple-v1`.

## token-preserving-foundation-maple-c12-ctx512

foundation-maple-v1 at cutoff 12 with 512 context windows (v1 at cutoff 12 with 128 overfit: train KL 0.22, held KL 1.24 / agreement 0.54). 1,024 windows peak ~51 GB on unified memory, more than fits beside the teacher.

## token-preserving-foundation-maple-c12-ctx1k

foundation-maple-v1 at cutoff 12 with 1,024 context windows (v1 at cutoff 12 with 128 overfit: train KL 0.22, held KL 1.24 / agreement 0.54).

## token-preserving-foundation-maple-c18

foundation-maple-v1 at cutoff 18 (128 context windows; fits beside the teacher). Cutoff 12 plateaued at held agreement 0.54.

## token-preserving-foundation-maple-c18-ctx512

foundation-maple-v1 at cutoff 18 with 512 context windows (v1 at cutoff 12 with 128 overfit: train KL 0.22, held KL 1.24 / agreement 0.54). 1,024 windows peak ~51 GB on unified memory, more than fits beside the teacher.

## token-preserving-foundation-maple-c18-ctx1k

foundation-maple-v1 at cutoff 18 with 1,024 context windows (v1 at cutoff 12 with 128 overfit: train KL 0.22, held KL 1.24 / agreement 0.54).

## token-preserving-foundation-maple-c21-ctx512

foundation-maple at cutoff 21 with 512 context windows (cutoff 18: held agreement ~0.64 with 512 windows vs 0.61 with 128; depth helps more than data).

## token-preserving-foundation-maple-c23-ctx512

foundation-maple at cutoff 23 with 512 context windows (held best agreement/KL: c12 0.54/1.24, c18 0.646/0.748, c21 0.789/0.283 still improving at step 8064; depth helps most).

## token-preserving-foundation-maple-c23-ctx1k

foundation-maple at cutoff 23 with 1,024 context windows (c23 with 512 plateaued at held agreement 0.854 / KL 0.126 from ~4k steps: KL passes, agreement does not; more data next).

## token-preserving-foundation-maple-c23-calibrated-v1

foundation-maple-c23-ctx512 with the agreement bound calibrated for Maple (owner 2026-10-06: no bug -> move on). Diagnostic of c23-ctx512 (runs/maple-foundation-20261005/diagnostics/gate-diagnostic-c23.json): the readout is exact at initialization; agreement is 99.6-100% where Maple puts >=0.9 on its top token and 63-66% below 0.3; the projection's token is in Maple's top 5 at 98.6-98.8%; ~3% of positions are exact bf16 ties. Top-1 agreement on Maple is bounded by its own next-token entropy (source positions: 43% below 0.3), so the bound is 0.75 per stratum; KL keeps 0.25.

## raw-recurrence-v2

raw-recurrence-v1 plus the owner's 2026-10-09 decisions, which apply to every line (not backbone-specific): the reader-side
Neuralese read adapter (`read_adapter`, heads.NeuraleseReadAdapter: drift in the Neuralese blocks is learned on the reader
side, not held to the token-embedding geometry) and the projection anchor to raw next-token embeddings phased out over
256 updates after the warm-up (`projection_anchor_decay_steps`; owner: anchors must not cap learning). The recurrence
handler whitelist gained both options with this recipe.

## raw-recurrence-mellum-v1

raw-recurrence-v2 for Mellum2.1-12B-A2.5B after the full-latent QAT conversion (maple/qat_convert.py) exported in the
deployed ternary form (maple/qat_export.py; heads from maple/foundation_heads.py --model EXPORT, Mellum markers
<|extra_token_7|>/<|extra_token_8|> = ids 33/34 via model/hf_port.family_controls). Backbone-inherent differences only:
- cutoff 27 = layers - 1 with 512 context windows (Maple: depth helped most; c23 of 24 layers, recipes above). The
  agreement gate stays the shared 0.9; Maple's 0.75 calibration came from a Maple-specific gate diagnostic and is not
  inherited without a Mellum one.
- Ternary QAT (`backbone_training: qat`, Maple's policy: attention dense latents and learned block scales, expert block
  scales, routers, norm gains) with latent learning rates in units of each latent's ternary scale (`qat_latent_lr`
  0.003; Maple's single Muon rate left codes practically unflipped, DECISIONS.md 2026-10-09).
- Nested-family member objectives (Maple-family members; LFM is not nested): member_weight 0.1 on 2048-token windows,
  4 pinned eval windows, members' system prompts masked in the recurrence member term (v3 debugging: ~89% of member
  windows were memorisable system/tool 16-grams) with the full model's crisp CE as anchor (member_full_weight 1.0).
Corpus note (plans/mellum-port.md): ~86% of the gold text corpus tokens are the shared system prompt, which chat models
do not predict (base Mellum CE ~4.5 there vs 1.0-1.9 on prose/code); report role-stratified text metrics.

## token-preserving-foundation-mellum-v1

foundation-maple at cutoff 27 (Mellum: 28 layers) with 512 context windows, the Maple lesson (c12 0.54 → c23 0.854 held
agreement: depth helps most) applied from the start.

Update (2026-10-09 evening): the text warm-up gained `read_adapter` and `member_mask_system` (same semantics as the
recurrence trainer; maple/family.leading_system_tokens). raw-recurrence-v2 declares `read_adapter` for
core_text_warmup too, so the adapter trains from the first stage that reads Neuralese and carries into the recurrence;
raw-recurrence-mellum-v1 declares both for its text warm-up stages. foundation-mellum-v1 has only the token-foundation
stages (no text warm-up handler), so the options live in raw-recurrence-mellum-v1.

## raw-recurrence-v3 and raw-recurrence-mellum-v2 (2026-10-10)

raw-recurrence-v2 and raw-recurrence-mellum-v1 (unchanged; the Mellum line's planned recipe keeps its identity) plus one
declaration shared by both lines: the `harness_bench` cohort (plans/neuralese/HARNESS_BENCH.md §6), **held**
(`admitted: false`) until the owner and both recipe owners admit it. raw-recurrence-mellum-v2 repeats mellum-v1's
backbone-inherent overrides on top of v3; a test proves both resolve to their predecessors plus the cohort.
- Text stages (core_text_warmup, autoregressive_text_fixup): the tokenizer twins
  `harness-bench-swe-rebench-openhands-pi-text-lfm25-350m-20261010-v1` and `...-text-mellum21-12b-20261010-v1` of
  `harness-bench-swe-rebench-openhands-pi-records-20261010-v3` (gold_text_rows, natlang.native_gold_chat/3, views as
  previews), mixed into the line's native text cohort at weight 0.25 of training windows, never replacing it.
- Whole-trajectory supervision: the text warm-up's all-positions/suffix halves; context_weight 1.0 and
  feedback_weight 0.25 declared. text_warmup has no per-role feedback weight yet; that is a precondition for admission.
- Recurrence (recurrence_warmup), after each student's runtime qualification for its exact weights and the view operator
  gate: the v3 records with `--view written --distill 1.0 --view-window 4096 --context-weight 1.0 --feedback-weight
  0.25`. The raw_recurrence_training handler whitelist does not carry these options yet; declared for its owner.
- History reasoning (owner 2026-10-10, recorded in place: a declaration, no change to any rendered row): each twin
  carries its backbone's declared, backbone-inherent `history_reasoning` (`serve/chat.py` BACKBONE_HISTORY_REASONING):
  LFM2.5 `last_turn_only` (its post-training template default; `preserve_thinking` is never set), Mellum2.1 `keep`.
  Target-turn reasoning is trained for both. The renderer asserts the template behaves as declared and records the
  policy in each new receipt; the reference server binds the same policy. Both v1 twins already render as declared.

Update (2026-10-10, owner decision "harness-bench cohort admitted to training"): the cohort is admitted, effective per
twin and stage as the trainer supports it. Admitted now: the LFM2.5 twin in the text stages (shared
cohort/document sampler and tool-feedback weighting, train/text_supervision.py; assembled text input,
scripts/assemble_neuralese_cohorts.py; registry training_admission true). Pending: the Mellum twin (its template renders
tool results as user turns, so text_warmup's 0.25 tool weight does not apply yet) and the recurrence (runtime and view
gates; written views do not yet run on harness records, see cohorts.harness_bench.recurrence.status). The recurrence
declaration gains context_coverage records, cohort_weights {native 0.75, harness_bench 0.25} and qualification_cohort
native; the raw_recurrence_training handler carries and validates them. The text warm-up's context_weight,
feedback_weight and qualification_cohort defaults (the text_warmup CLI defaults) are declared in v3 rather than v1, so
raw-recurrence-v1/v2 and raw-recurrence-mellum-v1 keep their identity.

## raw-recurrence-v4 and raw-recurrence-mellum-v3 (2026-10-10)

raw-recurrence-v3 and raw-recurrence-mellum-v2 (unchanged) plus the declared, **held** view operator stage
(DECISIONS.md 2026-10-09 "one summarizer family"; TRAINING_RECIPE.md "The view operator stage"):
- `view_operator` (raw_recurrence_training, after `adapted_runtime`): the view-stage records of `view-ask-20261010-v2`
  (`view-stage/records.jsonl`, pinned by SHA-256), `--view written`, faithful parts for reconstruction, distill 1.0,
  purpose_contrast 0.25 / margin 0.1, view_tokens_per_vector 4 within max_write_vectors 512, stop_pg 0.01 (combining
  writes), backward_policy joint, cohort `view`.
- `view_gate` (view_operator_gate, eval/view_gate.py) on that stage's checkpoint, with the harness-bench v3 records for
  pi's next-action loss; proposed thresholds as parameters.
- `recurrence_warmup` additionally requires `view_gate`; the harness cohort's recurrence declaration gains
  `view_tokens_per_vector 4`. Both stages carry `admitted: false` (licence review and runtime qualification).
- New shared mechanism: `overrides.stages_added` (insert a stage after a named one). raw-recurrence-mellum-v3 repeats
  mellum-v2's backbone-inherent overrides and applies its recurrence overrides to `view_operator` too; a test proves
  both resolve to their predecessors plus these additions.

