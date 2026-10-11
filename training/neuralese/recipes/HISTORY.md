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
- cutoff 27 = layers - 1 with 512 context windows (WITHDRAWN 2026-10-10, raw-recurrence-mellum-v6 below: not
  backbone-inherent; Maple: depth helped most; c23 of 24 layers, recipes above). The
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
- `view_operator` (raw_recurrence_training, after `adapted_runtime`): the view-stage records of `view-ask-20261010-v2` (v3 since the repin below)
  (`view-stage/records.jsonl`, pinned by SHA-256), `--view written`, faithful parts for reconstruction, distill 1.0,
  purpose_contrast 0.25 / margin 0.1, view_tokens_per_vector 4 within max_write_vectors 512, stop_pg 0.01 (combining
  writes), backward_policy joint, cohort `view`.
- `view_gate` (view_operator_gate, eval/view_gate.py) on that stage's checkpoint, with the harness-bench v3 records for
  pi's next-action loss; proposed thresholds as parameters.
- `recurrence_warmup` additionally requires `view_gate`; the harness cohort's recurrence declaration gains
  `view_tokens_per_vector 4`. Both stages carry `admitted: false` (runtime qualification and the stage's own
  qualification).
- Repinned 2026-10-10 (stages held, never run): the view stage's records are those of `view-ask-20261010-v3`
  (`view-stage/records.jsonl` SHA-256 85f693b7…, converter view_records@2); the licence hold is removed from the
  admission texts (owner rule 2026-10-10: licences are provenance facts only).
- Repinned again 2026-10-10 (stages still held, never run): `view-ask-20261010-v4` (`view-stage/records.jsonl`
  SHA-256 a708dc97…, 160,118 records): v3's data with the SWE-rebench tool outputs split by the harness bench's
  S1-aligned repository placements and closed against harness-bench v4 (VIEW_CORPUS.md §5.4). raw-recurrence-v5 and
  raw-recurrence-mellum-v3/-v4 inherit the pin.
- New shared mechanism: `overrides.stages_added` (insert a stage after a named one). raw-recurrence-mellum-v3 repeats
  mellum-v2's backbone-inherent overrides and applies its recurrence overrides to `view_operator` too; a test proves
  both resolve to their predecessors plus these additions.


## raw-recurrence-v5 and raw-recurrence-mellum-v4 (2026-10-10)

raw-recurrence-v4 and raw-recurrence-mellum-v3 (unchanged; v3's harness cohort is admitted and already used, so its
pins stay) with the harness_bench cohort repinned. Harness-bench v3 split by `split_of(repo, 5)` while S1 holds all
321 of its repositories, 53 in the other split (the view corpus v2 closure dropped 2,826 tool-output records for it).
`harness-bench-swe-rebench-openhands-pi-records-20261010-v4` is the same 3,843 records with S1's split for every
repository S1 holds (records.py `placements` over cross_corpus `place`; 2,969 train / 874 test), and its twins
`...-text-lfm25-350m-20261010-v2` / `...-text-mellum21-12b-20261010-v2` render it (same tokens as v1, other splits).
Pins: input bindings `harness-bench-records-v4` / `-pieces-v4` (view_gate), cohort source and recurrence inputs, both
twins. Admission: the Mellum twin is admitted for the text stages too, since Pop's 89982d41 labels Mellum's
tool-response user wrappers as tool (checked on the v2 twin: 87,337 tool turns); the recurrence stays held.
raw-recurrence-mellum-v4 repeats mellum-v3's overrides on v5; a test proves both differ from their predecessors only
in these pins.


## raw-recurrence-v6, raw-recurrence-mellum-v5 and token-preserving-foundation-mellum-v2 (2026-10-10)

QAT inside the stages (owner 2026-10-10; DECISIONS.md "QAT inside every training stage"; TRAINING_RECIPE.md "QAT inside
the stages"). raw-recurrence-v6 = v5 plus the top-level `quantization` component (points `q4` required 0.2–0.35,
`ternary-experts` required 0.35–0.7, `ternary` reported 0.6–0.95; stage progress warm-up 0–0.35, AR fixup
0.35–0.45, recurrence and view stage 0.45–1; sampled objective, BF16 weight 1) and the backbone rate 7.5e-6 for the
warm-up default and both recurrence stages (Pop's full-backbone drift at 3e-5). Shared by every line; the LFM line's
`auto` policy resolves to `full`, which the component quantizes the same way. raw-recurrence-mellum-v5 extends v6
with the backbone-inherent overrides of mellum-v4 except the ternary ones: cutoff 27, 512 contexts, the `latent`
policy in place of `qat`/`qat_latent_lr`, and no member terms (a BF16 student has no nested-family state).
token-preserving-foundation-mellum-v2 is foundation-mellum-v1 started from BF16 Mellum heads
(`foundation_heads.py --precision bf16`); its stages are unchanged.
Same day, before any run: the ternary points use Maple's per-row rule as conversion v3 and the TQ2_0 export do (no
nesting), ramps re-sized from v3's trend (q4 0.15–0.3, ternary-experts 0.3–0.7, ternary 0.55–0.95), and
raw-recurrence-mellum-v5 reuses conversion v3 (backbone-inherent: Mellum-tokenized): `init` from its gated best
latents with a ramp floor on `ternary`, `quantization.preserve` (teacher-v3 top-64 KL stream, weight 0.5) and
`quantization.gate` (v3's generation gate and held KL per precision, BF16 reference 0.75).
raw-recurrence-v6 (still before any run) declares full-state checkpoints every 3 h of wall clock in the text warm-ups
and both recurrence stages (`checkpoint_minutes` 180, `checkpoint_every` 100000; owner "checkpoint every few hours";
stops write the full state within the memory ledger's grace). The recurrence trainer gains `--checkpoint-minutes`.
Later the same day (owner: evaluations and checkpoints on declared STEP points, reproducible across runs, resumes and
machines; no added evaluations) raw-recurrence-v6 replaces the wall clock with step points sized from measured step
times (Mellum: embedding distillation 0.45 s/step measured on this lineage, the rest from conversion v3's ~5 s/step):

| stage | steps | eval_every (points) | checkpoint_every (writes incl. end) |
|---|---|---|---|
| embedding_distillation | 8192 | 512 (16) | 8192 (end only; ~1 h stage) |
| core_text_warmup | 4096 | 256 (16) | 2048 (2, ~2.8 h apart) |
| autoregressive_text_fixup | 1024 | 64 (16) | 128 (8; 16k-token steps, est. ~80 s) |
| view_operator | 4096 | 256 (16) | 2048 (2) |
| recurrence_warmup | 2048 | 128 (16) | 2048 (end only, ~2.8 h) |

`checkpoint_minutes`/`eval_minutes` are retired (a recipe declaring them is refused with the fix). The text warm-up's
curriculum advances per evaluation (plateau, ramps, gate streak need about 13 points), so 16 points per stage is
close to its floor; re-size from the measured step times once the stages run.

## raw-recurrence-mellum-v6, token-preserving-foundation-mellum-v3 (2026-10-10): cutoff 27 withdrawn

The Mellum line's `cutoff 27` (and its 512 context windows) is removed from the backbone-inherent list. It was not
backbone-inherent: it carried the Maple c12–c23 shallow-cutoff lesson from lineages the 2026-10-06 owner decision
("out port at the top", DECISIONS.md) had already declared to measure the wrong thing, and it broke the unify rule.
raw-recurrence-mellum-v5's foundation failed the shared embedding-distillation gate at cutoff 27 (source agreement
0.857, flat over the last ~1,000 of 8,192 steps; context 0.915), while the same heads and data at full depth qualify
exactly at initialization (KL 0, agreement 1.0, 0 updates; plans/mellum-port.md "Foundation embedding_distillation
gate failure"). The 512 contexts only enlarged the held sample and had no declared reason. raw-recurrence-mellum-v6
(= v5 minus both overrides) and token-preserving-foundation-mellum-v3 (file foundation-mellum-v3.json; = v2 minus the ones inherited from
token-preserving-foundation-mellum-v1) inherit the shared foundation unchanged; heads come from
`maple/foundation_heads.py --precision bf16` at full depth. Both lines now share a full-depth foundation. A shallow
cutoff stays possible only as a separate, optional efficiency variant with its own gate.

## raw-recurrence-v7, raw-recurrence-mellum-v7 (2026-10-10): fixup full-state writes 128 -> 512

Coordinator review of the v6 step points: `autoregressive_text_fixup` (1024 steps of 16k tokens) wrote its full state
every 128 steps. Its step time is unmeasured on Mellum (estimate 10-80 s), so that could be eight ~47 GB writes in a
few hours. v7 writes at step 512 and the end (>= ~1.4 h apart at 10 s/step; at 80 s/step ~11 h, which bounds crash
loss to half the stage). eval_every stays 64 (16 points; the warm-up curriculum needs about 13). Every other v6 stage
was re-checked and is unchanged: embedding_distillation end only (~1 h at 0.45 s/step; at full depth it qualifies at
initialization), core_text_warmup every 2048 of 4096 (~2.8 h at ~5 s/step), view_operator every 2048 of 4096 and
recurrence_warmup end only (2048 steps, ~2.8 h at ~5 s/step), each with 16 evaluation points. raw-recurrence-mellum-v7
= mellum-v6 plus the same override (the inheritance chain is linear). Re-size from measured step times.

## raw-recurrence-v8, raw-recurrence-mellum-v8 (2026-10-10): evaluations sized for the 3% budget

Owner: all evaluations together ~3% of a stage's wall time, 5% the hard ceiling. Measured on Mellum (eval-only,
step 0): a text-warm-up evaluation took 255 s with 16 held documents per cohort, 213 s with 4, unchanged by
ar_control_steps 256 -> 32; a py-spy trace showed ~3.5 min in `weights_digest` (a host SHA-256 of 23 GB), now an
on-device fingerprint (shared code). v8 also halves the held sample (8 documents per cohort, first and last window
each), evaluates the precision gate columns on the first 8 held documents; ar_control_steps (not a recipe parameter) stays 256, it cost nothing measurable. Projection
phase 0.5 s/step; the backbone phase is to be measured; re-size from the trainers' logged eval_share.

## raw-recurrence-mellum-v9 (2026-10-10): LionSR for the latent policy's text stages

run-v7 (mellum-v8) stalled at its first backbone update (step 2561): the preflight forecast a 115 GB increment
(gradients 23 GB + an FP32-assumed Muon momentum 46 GB, scaled by the projection phase's calibration ratio) against
34 GB usable. Two fixes in shared code (text_warmup): the memory forecast's calibration is per phase (projection vs
backbone) and the optimizer floor uses each optimizer's real state layout; a forecast above the whole device fails fast
(MemoryError) instead of waiting. mellum-v9 steps the backbone weight matrices with LionSR as conversion v3 did
(train/optim.py; PortMuonAdamW latent partition): one BF16 momentum and stochastic-rounding writes, so updates below
BF16 resolution are not lost; lr 3e-4 per row ternary scale (v3). Expected increment ~ gradients 23 GB + momentum 23 GB +
activations, peak ~73 GB. Updates still happen after the full backward (step-in-backward, which would also drop the
gradient buffer, needs one backward per update; the warm-up runs one per pass, precision and preserve stream).


## raw-recurrence-mellum-v10 (2026-10-10): LionSR latents step inside one layer-lockstep backward per update

v9's backbone phase still held a full BF16 gradient of the latents (~23 GB) because each update ran several backwards
(one per sequence pass, the sampled precision point and the preserve stream) before one optimizer step. Summing the
terms and backpropagating once is not enough: autograd runs the later-built graph completely before the earlier one, so
each layer's partial gradient waits in the engine's buffers for the other graph (measured on 1-3 real Mellum layers:
+0.79 GB per layer with two passes, i.e. the whole gradient again). Shared code:
`LionSR.step_in_backward(gated=True)` + `in_backward()` (train/optim.py) step each latent in its post-accumulate-grad
hook only inside one update's backward, look up the live parameter group (staged learning rates and reloaded state
apply), refuse a second gradient for the same latent in one update, and record the stepped gradients' FP64 norm.
`LayerStaging` (model/layer_staging.py; the ports' `run_layers` are decorated) runs each layer on detached leaves while
active and then backpropagates all graphs' layer i in one engine call, deepest first: each layer's weights get their
whole gradient at once, step and free it (measured: +0.03 GB per layer with two passes and checkpointing). A stream that
reads a deeper layer of another stream at its input is refused. text_warmup `--latent-step-in-backward` (needs
`--latent-optimizer lionsr`, one precision point per update, no family term) uses both inside the precision context;
the parametrized weights are built once and shared by the passes. Exact apart from clipping: the latents take the
unclipped gradient (Lion's sign update is invariant to a constant scale; only the step-to-step variation of the clip
coefficient differs), while heads and norms are clipped by the same global norm as before (the stepped latents count in
it). A failure after the first latent stepped writes nothing; the last written checkpoint is the resume point, as for a
failed optimizer step. Gradients are freed after each step (also in the old path), so evaluations no longer sit on
them. The preserve stream's long record is now really checkpointed on Mellum (its model had no `checkpoint_layers`
attribute, so the old guard left it uncheckpointed). The memory preflight models the change: no latent gradient buffer
(four copies of the largest latent in flight instead), passes summed rather than maxed in the geometry, and a separate
`:single` calibration kind. The recurrence trainer gained the same options (`latent_optimizer`,
`latent_step_in_backward`; joint backward policy, batch 1) through the same code; no recurrence stage declares them.
Tiny-model equivalence (tests/neuralese/test_latent_step_in_backward.py): the text warm-up end to end, a real port's two
passes under LayerStaging and a parametrized, checkpointed multi-pass model write identical FP32 latents to separate
backwards plus one step; in BF16 with a fixed rounding offset <=1% of elements differ by one sign step (gradient
contributions are summed in another order). Expected backbone-phase peak with v10's settings (batch 2, 1k tokens, layer
checkpointing): weights 25.5 GB + LionSR momentum 23.4 GB + ~2.5 GB update transients + ~4 GB activations, preserve
stream and heads, about 56 GB allocated (v9: ~75 GB).


## raw-recurrence-mellum-v11 (2026-10-10): dense experts as grouped GEMMs; fused precision ramp

The warm-up profile (plans/mellum-port.md "Warm-up step profile") found a q4 update at 34 s against 9.7 s for bf16:
the eager precision ramp of the experts (~15 full-size FP32 passes per weight) ran in all 4 forwards of an update.
Code-level and bit-identical, so every recipe gets it: `precision_value` (maple/ternary.py) keeps the rule's
reductions eager and shared with export (`q4_scales`, `ternary_row_stats`, also behind `q4_0` and `ternary_codes`), and
runs the elementwise rest as one Triton kernel per weight (maple/precision_kernels.py; IEEE division, no FMA
contraction). Zero differing elements against the eager rule on 2 real Mellum layers (attention and experts), q4 group
32/64 and ternary, mixes 1.0/0.47/0.03 (tests/neuralese/test_precision_kernels.py). The conversion's fused ternary
ramp uses the same kernel (its compiled version differed at ~4e-5 of threshold ties). The LionSR in-backward norm
reduces each latent in FP32 (FP64 was ~1 s per update on the GB10) and still totals in FP64.
v11 itself sets `moe_kernel: grouped` for the text warm-up and the recurrence stages: SparseMoE's dense experts run as
two `torch._grouped_mm` calls over the expert-sorted routed pairs (maple/fused_moe.dense_grouped_experts) instead of
the 64-way loop; same pairs, order and combine, BF16 accumulation order only (relative error < 1e-2 in output and
gradients, tests/neuralese/test_dense_grouped_moe.py). Backbone-inherent: dense trainable MoE experts.

## raw-recurrence-mellum-v12 (2026-10-11): core_text_warmup extended to 6,144 updates; cheaper report columns

Lineage run-v9's core_text_warmup ended at 4,096 unqualified (report preserved as
`run-v9/core_text_warmup/report-step4096-unqualified.json`): BF16 failed `pass-1-length-long-tail-last256` ce_delta
0.1255 > 0.1, and BF16 and q4 both failed the behaviour gate (gate_pass 0.33 / 0.25 against 0.75 − 0.1, held KL
0.92 / 0.92 against 0.25). v12 extends the stage to 6,144 updates. The quantization component keeps the 4,096-update
range (`quantization_steps`), so q4 stays at λ 1 and ternary-experts at 0.125. The pass-1 last256 ce_delta fell
0.042 per 1,000 updates over 2,560–4,096. The fit gives 0.115 at 4,096 with residual sd 0.036 on 8 held documents,
so the mean reaches 0.05 near update 5,600; two consecutive passes need about that much margin. Report-only precision
columns (gated points this stage does not require) run only at the decision that ends the stage, on a quarter of the
column's held windows, gate probes and held rows (`report_column_fraction`, operational).

## direct recurrence stage runtime evidence (2026-10-11)

The shared `direct_shared_stage` validator now requires a direct
`raw_recurrence_training` stage to carry exactly one SHA-pinned `adapted_runtime`
report in `lineage.evidence`. Before creating outputs, it applies the normal
`text_warmup_runtime` gate and binds `parent_sha256` to the exact `heads` input;
the report must attest `weights_unmodified` and `retained_trained_heads`.
This closes the gap where a direct recurrence invocation could record arbitrary
lineage evidence without proving the exact adapted runtime handoff. It adds no
foundation certificate, source admission, task-quality or autonomous-stopping
claim. Existing full-recipe stage dependencies are unchanged.

## neuralese-direct-v1, neuralese-direct-mellum-v1, neuralese-direct-lfm-v1 (2026-10-11): owner decision after the behaviour-gate bug hunt

These recipes replace the warm-up line, whose shared core_text_warmup drifted the backbone 11–14% on both lines and
broke plain behaviour. They are plain whole-trajectory target SFT from the base models, with no warm-up stage, input
map, second pass or self-distillation. The NeuraleseProjection produces the Neuralese value: the soft-token mixture plus
a zero-initialised causal conv, trained on detached states toward the gold next-token embedding. The
TokenChoiceProjection is declared but off.

Windows hold the whole preceding document, system prompt included. Routers and norms train on AdamW, attention and
expert matrices on the latent optimizer. The monitors are behaviour_drift (response KL and agreement) with its guard,
plus the final behaviour gate.

On Mellum, QAT points train on the same targets, with no BF16 KL teacher (flagged for the owner).

Design: plans/neuralese/NEURALESE_DIRECT_V1.md. Evidence: plans/mellum-port.md "Behaviour gate bug hunt" and
plans/neuralese/WARMUP_AUDIT_20261011.md.

## neuralese-direct-v1 qualification and cadence (2026-10-11, before the first full run)

- Qualification now also requires a useful Neuralese value (Pop review, accepted by the coordinator). The final held
  `value_relative_mse_delta` (NeuraleseProjection minus the soft-token mixture, relative MSE to the gold next-token
  embedding) must be below `-value_margin`, with `value_margin` 0.05. Smokes measured −0.15 to −0.27 within 60
  steps. The report records this as `value_gate`.
- Eval and checkpoint cadence is 1000, so they happen at the stage end only, plus the step-0 base eval. At the
  measured 5.5 s/update, a 3 h cadence is about 1960 updates, more than the 1000-step stage. A gated eval costs about
  190 s, so one intermediate eval would put evals above 3% of wall time. The final eval is the gate. If the gate
  already ran at that step on a warn, the final gate reuses it instead of running again.
- Best (Pop 20f57d6d): an eval that improves held CE writes the full state with that metric, and CheckpointPolicy
  hard-links it as `best-checkpoint.pt`.
- The behaviour gate is no longer restricted to Maple. `run_gate` decodes through the shared `greedy_any` (Pop
  178a6160), so LFM gets the same gate.

## neuralese-direct-v1 drift monitor cadence (2026-10-11, after run-v1's launch)

`drift_every` is an operational stage parameter, set to 100 in the shared recipe. Between evals, the drift monitor
runs every 100 updates and logs to `drift.jsonl`. It computes held KL on the 16 drift rows, applies its guard, and runs
the behaviour gate at a warn. A guard stop stops the stage with its full state, and qualification counts stops from
both logs. Before this, the guard ran only inside evals; at the stage-end eval cadence (d5dc7342) it never saw
run-v1. Cost: one held-KL pass over 16 rows per check, plus about 3.5 min when a warn runs the gate.

## neuralese-direct-v1 qualification: value report-only (owner decision 2026-10-11)

The Neuralese value requirement moves to the end-to-end stage, measured as consumers reading values against consumers
reading tokens, on task success and loss. Direct SFT qualification is now the step-0 identity check, held CE below
base, no drift stop and the behaviour gate. `value_margin` stays as a report-only threshold for `value_qualified`.
Pop's LFM recipes that declare it are unaffected.

## neuralese-direct-mellum-v2 (2026-10-11): the forward Mellum direct lineage

The coordinator's decisions under the owner's "keep moving forward" rule. Changes relative to v1:
- The clean anchor is raw-token-v1 heads at the full stack, with final stop source and no stop position. They are
  built by `maple.foundation_heads --profile raw-token-v1`, and the direct serving path (DirectWriter, Pop's Maple
  export adapter) opens them. v1's legacy-rms-v1 anchor cannot be served directly.
- Attention latents train, through the a8972488 fix.
- `drift_every` is 100.
- Behaviour gate `held_kl_bound: response`: `max_held_kl` 0.25 bounds the response-position KL, as the guard does.
  Prompt KL stays report-only. The gate field is new; it defaults to combined for older lineages.
- 2000 steps. Eval at the end only, full state at 1000 and 2000.

run-v1 stays as evidence only: attention untrained, legacy anchor.

## neuralese-direct-v1 gate cadence while warning (2026-10-11)

`drift_gate_every` (300) and `drift_gate_now` (0.35) are operational parameters, in the shared defaults. KL is still
checked every `drift_every` (100). While the guard warns, the behaviour gate runs at most every 300 updates, and at
once when response KL reaches 0.35. The hard stop stays at 0.5, and the stage end always runs the gate. Reason: in v2,
response KL was above the warn level from step 100, so every 100-update check ran the full gate (about 3.3 min per
10 min of training, a third of wall time). The owner's eval budget is 3–5%. A deferred check logs
`gate_at_warn: {deferred, last_gate}`. `last_gate` is kept in the full state.

## neuralese-direct-v1 gate bound and gate trigger (coordinator decisions 2026-10-11)

- `gate_response_kl` 0.5 is a shared declared stage parameter for both backbones. It overrides the declared gate's
  KL bound for the direct stage, on response positions, at the hard stop. Response KL measures change from base, not
  damage. The damage measures are the behaviour gate's pass rate (≥ base − 0.1, texts kept) and held CE.
  Mellum v2 plateaued at response KL 0.36–0.38 with the gate at 0.75–0.83 (base 0.75), which the 0.25 bound would
  have failed.
- `drift_gate_rise` 0.05 replaces `drift_gate_now`. The gate runs while warning every `drift_gate_every` (300)
  updates, and at once when response KL has risen 0.05 since the last gate. In v2 a level trigger at 0.35 fired at
  every check once KL plateaued above it, which brought back ~30% gate overhead.
- `drift_ce_windows` 4: every drift check logs held CE on the first 4 held windows, with its step-0 base, so the
  held-CE minimum is visible between evals (v2's train CE 0.07 suggested memorisation).

## neuralese-direct-v2 budget in epochs, packed updates, two phases (2026-10-11)

The data audit (plans/neuralese/DATA_INVENTORY_20261011.md, section E) found that `steps x target_tokens` (750 x 1024
= 768,000 "about 2 epochs") covered 0.11 epoch on Mellum: one window per update, sampled with replacement over all
6,801 train windows (only 3,258 hold a decision target), each supervising ~508 positions while MuonSR costs ~17.4 s
fixed per update. Changes (shared base, both lines):
- Budget: `epochs` per stage. direct_sft builds an epoch-ordered document shuffle without replacement over the
  target-bearing train windows (the shared `CohortDocumentSampler.epoch`, `text_supervision.epoch_schedule`),
  context-only windows skipped, and derives the update count; `steps` is gone from the direct recipes (legacy
  `--steps` still runs through the same schedule). Each update logs windows, supervised/target/forward tokens,
  epochs covered and supervised tokens per second; the report gives the budget and epochs covered.
- Packing: `batch_tokens` (fit-only; Mellum 32768, LFM 16384): length-bucketed right-padded updates (runs of 256,
  count x longest within the budget). Right padding is exact for causal attention and causal convolutions; padding
  carries no weight; the loss is the token mean over real positions (a batch of one is the old computation).
- Phases as recipe stages: `direct_warmup` (0.5 epoch of the full-prompt union twin) then `direct_sft` (2 epochs of
  the program-header twin), continuing from the warm-up's full state (`--continue-from`, runner-supplied) with the
  base model's drift reference and the inherited identity check. `stages_added` accepts `before`.
- Quantization: a stage budgeted in epochs passes `end: "derived"`; direct_sft binds it to the derived update count.
- Line parity: `derived.supervision` (epochs) replaces `derived.supervised_positions`; multi-stage recipes compare per
  stage. The Mellum program-header twin is registered as the tokenizer twin of the LFM program-header set.
