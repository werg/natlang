# Recipe history

2026-10-12: deleted 104 discarded recipes (families foundation, gold-text, latent-sketch-consumer, local-stage-sketch-credit, sketch-training-defaults, learned-residual-curriculum, semantic-exposure-data-contract, native-text-recurrence, luna-ar, raw-recurrence, fp8-lfm-direct, luna-direct-lfm-native-e2e except the token-writer endpoint-epoch v3 arm, luna-neuralese-direct-lfm-base except its v8 parent, optimizer-bakeoff) per owner decision (REVIEW_20261012 rec. 9); history in git before 03339381 (`scripts/prune_stale_recipes.py`).

Experiment history moved out of `description` for recipes that use `extends`/`overrides` (plans/ARCHITECTURE_IMPROVEMENT.md C3).

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

## 2026-10-12: E2E stage memory routing, crisp term and state writes (trajectories, shared)

Mellum E2E smoke (2026-10-11) ran out of memory at update 1 in the separate crisp-SFT backward, which had no fallback,
after the router predicted 39.8 GB for an update that peaked at 72 GB.
- Crisp SFT is one more term of the update's routed backward (the joint attempt, the staged path after a joint miss,
  or the in-backward update), so a joint miss falls back with the whole objective; its reader-sized tape is in the
  routed estimate. Before, it was a second backward taken after the reader's gradients already existed.
- Router: the fixed part of a joint attempt (resident memory plus this update's gradient buffers; latents stepped in
  backward excluded) is counted exactly. Only the graph tape is predicted and calibrated (`joint_tape_ratios`, the
  older whole-peak `joint_ratios` are not reused), and a bin not seen yet borrows every bin's observations. A joint
  result outside 0.67-1.25x of its estimate prints `memory_estimate_miss`.
- `--save-state/--no-save-state` (as in the direct stage): smokes write logs and the report only, with no checkpoint,
  best link, heads.pt or adapter.pt. No full-state write before the first update in any run: a fresh start has
  nothing to protect, and a resume reads a state already on disk.
- `--latent-step-in-backward` is refused with `--direct-artifact`: under the layer-staged backward the direct writer's
  child context received no gradient (relative change 0.0 against 0.065 with the separate backward, CPU test). The
  writer alone is exact under staging (a new test compares the context gradient with and without it, forced and
  step-by-step). The loss is in the order: the caller reads the child's value, made from the writer's top layer,
  at the caller's layer-0 input. The lockstep walk reaches that input edge last, after the writer's layers were
  passed, so the gradient arriving at the writer's top leaf is never propagated, and the staging check misses it
  (the leaf was never processed). Open work: a correct order for such cross-stream edges, with each layer's
  latents still stepping once.

## 2026-10-12: layer staging resolves cross-stream edges (direct writes under --latent-step-in-backward)

The refusal of `--latent-step-in-backward` with `--direct-artifact` (same day, above) is lifted. `LayerStaging`
(model/layer_staging.py) finds edges whose source reads exit leaves of the same or deeper layers (the caller's input
made from the child writer's top layer). Before the layer walk, an activation-only pre-pass (`torch.autograd.grad` to
the layer leaves: no weight gradient, no latent steps) walks the same layers and sends each such edge's gradient to the
exit leaves it reads. It repeats while it reaches new cross edges (nested writes). The main walk then starts with those
gradients in place, so each layer's latents still step once with their whole gradient. When the walk reaches such an
edge, it checks that the edge sends the pre-pass's gradient and discards the duplicate. Any other gradient arriving at
an exit leaf after its layer was passed is now refused; before, it was dropped silently.
Cost: one extra activation-only backward (with checkpointed layers, one more recompute) when cross edges exist; none
otherwise. Tests: a two-stream tiny-LFM update equals separate backwards plus one step (latents bitwise, context
gradient); the direct E2E stage moves the child instructions like the separate backward (within 5%).

## neuralese-direct-v3, neuralese-direct-mellum-v4, neuralese-direct-lfm-v3 (2026-10-12)

Shared base moved to the admitted target union r2 (natlang-target-union-20261011-r2) with a third phase, direct_trajectory (whole-trajectory rendering, 1 epoch, continuing from direct_sft), following Pop's main LFM line (pop-2356, agreed pop-86dc). Mellum's v3 lineage keeps its finished 4962-document warm-up: neuralese-direct-mellum-v4-from-v3-warmup binds it as direct_sft's declared continuation inputs (historical one-off, line-parity.json historical_differences). Mellum's QAT curriculum continues over the new phase (direct_trajectory [0.35, 0.5]).

## neuralese-direct-mellum-v4-from-v3-warmup: direct_trajectory on union r3 (2026-10-12)

The running Mellum v4 lineage's second phase, direct_trajectory (1 epoch), binds union r3's whole-trajectory Mellum
rendering (natlang-target-union-20261011-r3; 165,822 train / 1,549 test documents, 50.2M supervised tokens) in place of
r2's. direct_sft is unchanged. It is taken by a `--handoff` while direct_sft runs: `lineage_handoff` now allows new input
content for a stage that has not started (`unstarted_stage_input_changes`). It is a historical one-off in
line-parity.json, and the parity recipes stay on r2. Details: plans/neuralese/NEURALESE_DIRECT_V1.md "Mellum v4".

## neuralese-direct-mellum-v4-from-v3-warmup: direct_sft data continuation onto union r3 (2026-10-12)

Owner decision: v4's active direct_sft continues on r3's program-header rendering from its current state, not from
scratch. The recipe binds `text_data` to r3 and declares `continued_text_data` as the r2 file it trained on. This is a
shared `neuralese_direct` handler role and the one declaration under which a `--handoff` may change an active stage's
inputs. direct_sft keeps its counter, weights, optimizer and QAT state, and trains r3 minus the seen r2 documents for
the rest of its epoch. The QAT progress continues from its value at the continuation update over the new total
(31,314 updates at update 400). The base is re-measured on r3's held set. Details: plans/neuralese/NEURALESE_DIRECT_V1.md.

## Shared QAT curriculum and nested family; objective continuation of Mellum v4 (2026-10-12)

`neuralese-direct-v3` (both lines) declares the curriculum direct_sft [0, 0.75], direct_trajectory [0.75, 1.0]. It
also declares the q4 point (the LFM line's light QAT) and the nested-family component: LFM depth members 12xall and
8xall, ramp 0.05-0.10. `neuralese-direct-mellum-v4` keeps its ternary points and declares the members 28x32, 28x16 and
21x32 over the N0 order. `neuralese-direct-mellum-v4-from-v3-warmup` declares `objective_continuation` of direct_sft,
pinning the quantization component it trained with (ebebbaf8...). Its next `--handoff` rebases the curriculum from
p_k (about 0.03) to 0.75 without a jump. Details: plans/neuralese/DECISIONS.md, 2026-10-12.
## neuralese-e2e-v1, neuralese-e2e-mellum-v1, neuralese-e2e-lfm-v1 (2026-10-12)

Owner decision 2026-10-12 (plans/REVIEW_20261012.md item 2, clarified): the end-to-end stage is the successor of
text-only direct SFT on both lines, started as soon as it is ready. `line-parity.json` stages.e2e now names a shared
base recipe instead of an inline config, and both lines declare a recipe (no pending line).
- Lineage: `direct_runtime` (new handler `direct_runtime_qualification`, eval/direct_runtime.py on the exact export)
  then `e2e` (train/trajectories.py --direct-artifact, the c2eea5b2 stage; the parameters of the former inline base
  config, plus `latent_step_in_backward` and `producer_batch_size` 1 moved into the base now that direct writes train
  under the layer-staged backward, 4f3bb938). identity_qualification `direct_e2e_runtime_qualified_export`; no --heads.
- The direct export is a lineage binding (`input_bindings.direct_artifact`) of a launch recipe extending a line recipe.
- Acceptance `e2e_acceptance` (train/e2e_acceptance.py, e2e-certificate.json): V CE at most T CE (gold crisp text),
  V beats shuffled and zeroed by 0.05 nats, at least 32 paired readers, exact-oracle task success within 0.02 of T
  (pending until the task-success evaluator exists), no drift stop. The behaviour gate, token-readout regression and
  greedy-T arm are reported. trajectories' paired-reader evaluation now also reads T and a zeroed block on the same rows.
- Handoff moment `parent_readiness`: held CE plateau (3 evaluations, at most 2% relative), no drift stop, task success
  reported until measured. Mellum: QAT component held at the parent's progress (0.5 after the full direct lineage).
- `neuralese-e2e-lfm-v1-qual-muon-main-v2-u9000`: mechanics qualification arm on Pop's LFM main at update 9000
  (200 updates).
