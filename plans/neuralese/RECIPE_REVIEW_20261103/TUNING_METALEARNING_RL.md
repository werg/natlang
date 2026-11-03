# Recipe review: program tuning, metalearner and the RL stage (2026-11-03)

Read-only audit. Owner question: "Can we also make sure that we firmly include tuning and meta-learning in the
recipe?" plus "are the RL tasks formulaic?". Nothing here was run on a GPU; nothing was modified except this file.
Scripts used for the task census ran under the memory ledger (`audit-tune-rltasks`, `audit-tune-rlnl`,
`audit-tune-show`, 4 GiB, low priority); their outputs are in
`/tmp/claude-1000/-home-werg-natlang/4c9bd489-afe4-4847-b74c-378caa27432c/scratchpad/audit-tune/`.

## 0. Summary

- **The production recipe does not contain tuning, the metalearner or RL.** The actual recipe
  (`ops/earned179-joint-r38-a4-window64-1103/recipe-runtime.json`) has one stage, `native_joint_sft_v4`
  (`kind: native_state_sft`). RL lives in a separate controller config (`rl/production.py`) that wraps that SFT stage
  as a "mandatory SFT" obligation. Tuning and the metalearner are a shell-script lane
  (`training/neuralese/learned-optimizer/run-lfm3-lane.sh`) whose README says it is "a lane beside the production
  recipe, not a stage inside its JSON".
- **The recipe's stage gates do not look at task behaviour.** `train/recipe.py:require_gate` accepts a stage when its
  report says `completed` and `numerics_ok`. The SFT stage declares `task_eval` with `min_successes: 0` and
  `task_eval_every: 0`, so it runs no in-stage evaluation. Promotion is decided by hand from separate Pop evaluations.
- **Tuning on LFM3: never run. Metalearner: never run anywhere.** All measured tuning results are from the discarded
  LFM2.5-350M lineage (step 39,495, before the 2026-10-29 LFM3 restart). The best of those: ticket routing went from
  0/20 (base) to 12/20 (teacher SFT) to 17/20 (SFT+RL) on 20 development roots, with no confirmation run. The
  learned-optimizer operation directory `/mnt/external/natlang-learned-optimizer-20261027/` holds specs and a 92 KB
  empty store. `JOINT_LEARNED_OPTIMIZER_STATUS.md` says: "No training run, measurement or capability claim exists yet."
- **The RL tasks are largely formulaic.** The live RL source has 61,399 train tasks in 269 nominal families:
  - 97.6% come from one generator (`natlang.broad_external_rl_tasks/1`), which wraps an external dataset item in a
    single typed root function with 2-3 paraphrased instructions per family.
  - 99.1% are `single_call`, 69% are single-file programs, `iterateOn` appears in 0.13% and inline delegation is
    required in 0.9%.
  - The selector is configured with 4 rows per round and 4 samples each, and it starts every unexplored family at its
    easiest level. That structurally produces the observed 20/20 zero-spread rounds.
- **The program supply for tuning and the metalearner is also formulaic.** It has 51 programs in 18 families, and every
  one has the same shape: a crisp helper plus an NL ordered rule list that returns a category, with an untrusted note.
  That is too narrow to train a learned optimizer that should transfer across held families.

## 1. Tuning and metalearner machinery: what exists and its state

### 1.1 Program tuning (`natlang tune`)

| Piece | Where | Tested | Real runs |
|---|---|---|---|
| Scope tree, per-call adapter stacks, release/activation/rollback | `ts-host/src/adaptation/weights/`, `ts-host/src/tune/` | TS unit tests (`ts-host/test/tune-*.test.mjs`, 18 files) | LFM2.5 only |
| Joint trainer: nested shared bases or free LoRA, soft instruction/skill states, one whole-graph request builder for SFT/RL/RFT | `training/neuralese/natlang_neuralese/tune/` (`trainer.py`, `rl.py`, `rows.py`, `soft.py`, `bases.py` …) | `tests/neuralese/test_tune_*.py` (trainer, rl, rows, soft states, export, families, incremental, supply …) | LFM2.5 only |
| SFT/repair/RL/RFT rounds, development selection, execution budgets | `ts-host/src/tune/{rounds,rollouts,repairs,schedule}.ts` | unit tests | LFM2.5: ticket 12 RL + 12 SFT updates |
| Teacher-free supply (`tune.teacher: "self"`, privileged references for training roots only) | `ts-host/src/tune/selfteach.ts`, `tune/supply.py privileged` | `tune-selfteach.test.mjs`, `test_tune_supply.py` | never |
| Families, placement, sweeps, distillation export | `tune/families.py`, `--placement`, `--sweep`, `tune/distill.py` | stand-in or fixture only | never on a real model (distill used once on LFM2.5: 1,075 rows per model, registered `luna-program-training-worlds-native-graphs-20261027-v1`) |
| Serving: native reference server (exact), llama.cpp fork per-sequence LoRA and soft states, wllama | fork `werg/llama.cpp-neuralese`, `tune/soft_export.py` (lfm2 and lfm3) | parity on LFM2.5-350M CPU | — |

Real results (all on the **discarded LFM2.5-350M** parent, `plans/PROGRAM_TUNING_RUNS_20261026.md`):

| Program | Train/dev/conf roots | Outcome |
|---|---|---|
| ticket-routing | 94/20/14 (later 301 roots with leaf invocations) | base 0/20, teacher-SFT 12/20, SFT+RL (12 RL + 12 SFT updates) **17/20** on development. Release `b0a4faf8…` activated. Confirmation never run. |
| invoice-review | 186/60/10 | SFT fits only (24, then +33 updates); paused 2026-10-28 for an LFM arithmetic measurement (`invoice-paused-for-lfm-arithmetic-20261028.json`) and never resumed. Base 0/41 of 60 dev roots. |
| inventory-routing | 84/34/10 | 12 SFT updates; baseline never measured. |

Lessons from those runs that carry over:

- Prompt-dominated loss: about 14-16% of weighted supervision was on the reply, which was fixed with reply-only SFT.
- Failed intermediate actions had been eligible as targets, fixed by action-outcome policy v2.
- Native provenance was missing in evaluation, which invalidated a 0/20 cohort.
- 683 of 868 profiled seconds went to ancestry verification.
- RL needs informative groups. It got there by sealing tranches of nonconstant groups, not by waiting for a full
  cohort.

### 1.2 Learned optimizer, LoRA bank/basis, component-token mixtures (the "metalearner")

Code: `training/neuralese/natlang_neuralese/tune/learned/` (24 modules, about 3.6k lines). It covers:

- `bank.py`: shared basis and component bank.
- `mixture.py`: component-token distribution with null row, masked softmax, top-p and gains. This is owner decision
  13 and replaced the grid and coefficient head.
- `writer.py`, `trainer.py`: outer trainer with writer adapter, head, keys and optional `train_bank`/`train_basis`.
- `meta_rl.py`: expected-score meta-RL.
- `fit.py`, `growth.py`, `ladder.py`, `controls.py`, `meta.py`: meta-examples for the optimizer's own stages.

The natlang program is `applications/learned-optimizer/` (`adapt.nl`, `train.nl`). The lane is `run-lfm3-lane.sh`
with the specs `lfm3-{bank-init,corpus,train,meta}-spec.json`. `train_basis`, `train_bank` and `train_keys` are
already `true` in `lfm3-train-spec.json`, as AGENTS.md "Learned LoRA mixtures" requires.

State:

- Tested on tiny fixtures (`tests/neuralese/test_learned_optimizer_*.py`, 9 files) and a lane `--dry-run`.
- Never run on a real model. No bank, corpus, outer step, meta-RL proposal or meta-example exists.
- The LFM2.5-era operation dir has only specs and a hold script (`run-after-lfm-main.sh`).
- The LFM3 lane root (`/mnt/external/natlang-learned-optimizer-lfm3`) does not exist.

Open items from `JOINT_LEARNED_OPTIMIZER_STATUS.md`:

- dense JOLT+ term in the tune trainer;
- batched rollouts and evaluations;
- label-only refits;
- registry admission of target manifests;
- scheduled planted controls;
- live measurement of the model-facing writer template and NL stages.

Its own cost estimate is 450-900 GPU-hours per full lane, two thirds of which shards per program.

Recipe-system hooks that exist:

- `native_coefficient_rl` is a declared stage kind (coefficient-policy RL with its own gate schema
  `natlang.native-coefficient-training-report/1`). It is the only recipe-native adaptation stage, and no recipe uses
  it.
- Coefficient policies (`model/native_coefficient_policy.py`) are typed artifacts bound to an exact base.

### 1.3 Program supply for tuning and the metalearner

- `/home/werg/data/program-supply-20261027/programs`: 51 programs (18 families × base, v1 and v2 variants). Each has
  128 roots over 64 worlds, persisted splits and exact oracle values.
- Example `refund-decision`: `main.nl` calls `remaining(order)` (crisp `.ts`), then `chooseRefund`, whose body is
  "Apply in order: delivered=false -> manual; damaged=true -> refund; …". Every family (access-request, alert-paging,
  claim-triage, loan-prescreen, …) has this same "ordered business rules → enum" shape.
- These are useful smoke programs. They are not the diverse codebase/operation supply that the owner asked for and
  that a transferable learned optimizer needs: held families would differ only in nouns.

## 2. Can the recipe express tuning and metalearner stages?

Today, no. The concrete gaps:

1. **No stage kinds.** `HANDLERS` in `train/recipe.py` has `native_state_sft`, `native_state_rl`,
   `native_offline_preference` and `native_coefficient_rl`. There is no kind for a per-program tuning campaign or for
   the learned-optimizer lane. The lane is a bash script with its own operation store, specs and ledger units, and the
   recipe and its lineage identity never see it.
2. **Gates are mechanical, not behavioural.** `require_gate` checks completion and numerics. There is no declared
   promotion contract (packet, matched runtime, comparison parent, minimum delta, confirmation). The stage's
   `task_eval` is pure telemetry (`min_successes: 0`, `task_eval_every: 0`).
3. **RL is outside the recipe.** The controller config (`ops/quality-parent-onpolicy-runtime19-1102/config.json`)
   binds `production_recipe`/`production_stage` for its SFT obligation. Rows per round, group size, curriculum policy,
   policy loss and task source are controller settings with no recipe-level entry or exit criteria. `native_state_rl`
   only consumes rollouts the controller already collected.
4. **The backbone-to-adaptation handoff is undeclared.** Nothing in the recipe records which earned backbone
   checkpoint the lane binds or which development threshold makes it eligible. Nothing feeds back distilled
   training-world tuned executions as a source continuation. The lane's `operation.json` binds a checkpoint SHA, but
   no recipe stage requires it.
5. **No recipe-level progress record.** Lineage progress counts SFT updates and RL `trained_rounds`. It does not
   count programs tuned with measured lift, bank publications, outer steps or meta-RL rounds. "Comprehensive training
   through the entire recipe" therefore cannot be checked from the recipe.

What is needed to make tuning and the metalearner first-class:

- **New handler kinds** delegating to the existing mechanisms; no new trainer:
  - `program_adaptation` (module: a thin Python wrapper running `natlang tune P --rounds` plus paired
    `--evaluate development` per program, sharded);
  - `learned_optimizer` (module: lane stages `bank-init`, `corpus`, outer cycles, `outer-rl`, `meta-examples`,
    controller, held arms).
  - Required inputs: `continue_from` (exact backbone full state), `programs` (supply manifest with persisted splits),
    `protection` (protected-worlds bundle). Optional: `previous_lane` (adapter, bank and optimizer histories to
    continue).
  - Result: a lane report plus artifact manifest.
- **A behavioural gate adapter**:
  - a `promotion` declaration per stage, `{packet, cases, runtime, compare_to: parent|base|baseline_arm,
    min_delta, min_programs, confirmation}`;
  - `require_gate` refuses advancement until the stage's report carries a matching
    `natlang.behavior-promotion/1` receipt computed from paired arms at identical invocation limits;
  - completion and numerics checks stay as they are.
- **An RL stage declaration** binding the controller's curriculum, rows and group, and policy loss into the recipe
  identity, with its own promotion contract. The controller remains the executor.

## 3. RL stage: wiring and curriculum

### 3.1 Wiring

- The controller (`rl/production.py run`) collects with the native Python actor (`actor_engine: native-python`),
  scores, and admits whole graphs into `native_state_rl` with JOLT+ (`beta_s 1.0, beta_t 0.5`, score-centered k=128,
  TIS).
- Interleaved SFT runs through `mandatory_sft` / `native_supervised.execute`.
- Current config: `rows_per_round: 4`, `group: 4`, `workers: 4`, `max_model_requests: 64`, `task_curriculum
  {breadth_fraction .25, promotion_success_rate .6}` with the defaults `family_exploration_fraction .5,
  frontier_scope semantic-family`.
- Production status (handoff 2026-11-03): no RL is running. The only recent RL product, diagnostic jolt72 (five
  updates), is rejected (11/16 vs parent 12/16 on the protected development cases).
- The last round, `quality-parent-onpolicy-runtime19-1102/round-000000`, had 4 tasks × 4 samples, all `task_success`,
  `spread: false` on every group. The tasks were:
  - MATH counting level-2, curriculum level 0;
  - `child_opt_out`, level 0;
  - bAbI task 1, level 1;
  - 2Wiki bridge.
- Cloud RL tooling exists (`scripts/coreweave/rl_cloud.py`, `lfm3_rl_lineage.sh`), but the handoff lists concrete
  defects: the lineage script treats SFT-only progress as failure, the export skip uses RL counters, and the old run
  file binds obsolete fresh-init parents.

### 3.2 Task source census

Source: `/home/werg/data/broad-rl-tasks-20261031/full-core-effects-20261102-20261102T011031Z-3015925/cases.jsonl`.
I read programs and metadata only, never `expected`. The protected development packet was not opened.

| Measure | Value |
|---|---|
| Train tasks | 61,399 (all `split: train`) |
| Nominal families / shapes | 269 / 1,079 |
| Generator | `broad_external_rl_tasks/1` 59,945 (97.6%); procedural `inline_curriculum_generator/1` 1,454 |
| Mode | `single_call` 60,837 (99.1%); `followup` 562 |
| Inline delegation | optional 60,469; required 568 (0.9%); avoid 362 |
| Program files | 1 file: 42,307 (69%); 2 files: 13,126; ≥3 files: about 5,966 (mostly codebase families, where the files are the codebase) |
| Neuralese-helper framing | 15,213 tasks (24.8%) have a `Neuralese<string>` helper (per-document notes etc.) |
| `iterateOn` in program text | 80 tasks (0.13%) |
| Services | 17,630 tasks (SQL db, repo, tool receipts) |
| Curriculum levels | L0 2,240; L1 26,691; L2 21,693; L3 10,775 (levels derive from upstream difficulty labels or hop/step counts) |
| Oracles | answer-match kinds: normalized/exact/span-F1/numeric tolerance/result-set/answer-list. Effect or state-checking oracles are rare (codebase mutation 664, design-brief gates 180) |

Distinct root-program texts per family, with digits normalized:

- 207 families (26,770 tasks) have ≤3 distinct instruction texts. Example: gsm8k has 1,500 tasks and 3 texts, all
  "Solve problem … compute in eval … return a number".
- 27 families (20,455 tasks) have 4-12.
- 35 families (14,174 tasks) have more than 12, but only because a symbol or field name is substituted into a fixed
  question type. Example: "Find every function … whose own body calls something named kill" across 4,688 tasks.

Nominal family counts overstate breadth:

- 60 families are `ext_design_brief_NNN` with 3 tasks each (180 tasks in all).
- 59 are reasoning-gym puzzle types with 60 tasks each, all at level 1.
- 20 are bAbI tasks with 70 each.

**Verdict: formulaic.** The source is broad in content (23 external datasets plus 8 Python/TS/natlang codebase
question types), and the oracles are well qualified: negative controls and alias probes are documented in
`BROAD_RL_TASKS_20261031.md`. It is narrow in program structure, though. Almost every task is:

> one typed root function wrapping a dataset item, single call, answer-match oracle

Only half of the tasks offer a Neuralese helper, and almost none need `iterateOn`, multi-function call graphs,
stateful services with effect verification, or several operations against one reusable application. That reusable
application with several operations is the unit the owner asked for ("use diverse datasets to build substantial
Natlang codebases, then generate several different calls/tasks against each codebase"; "a generic dataset-answer
wrapper is insufficient").

The procedural inline curriculum (1,454 tasks, 49 families × about 30) is the part that teaches natlang idioms. It
is 2.4% of the source, and many of its families have one template (`idempotent_retry`, `stateful_dates`,
`recursion_rewrite`, …).

### 3.3 Why rounds have zero outcome spread

This is structural, not bad luck. In `rl/curriculum.py:select`, with `rows=4`:

- one row is rotating breadth;
- of the 3 practice slots, `ceil(3 × 0.5) = 2` go to family exploration;
- "new families start at their easiest admitted task".

With 269 families and about 2 exploration slots per round, the selector spends about 130 rounds mostly sampling
level-0/1 representatives of unvisited families. Those are the tasks a 12/16-dev model solves 4/4 times.

The remaining practice slot prefers mixed outcomes, but there is no observed history yet. With group 4, a task solved
at rate p gives a zero-spread group with probability p⁴ + (1−p)⁴, which is 0.66 at p = 0.9. The JOLT+ dense KL term
still fires, so updates are dominated by privileged KL and efficiency terms, which is what the jolt72 review found.

## 4. Proposal: recipe stage sequence

Principles:

- Promotion is by paired task behaviour at identical invocation limits, never by step counts.
- Stages overlap as owner policy requires: RL interleaves with SFT, and adaptation runs as soon as its entry
  criterion holds while backbone training continues.
- Optimizer, RNG and exposure are preserved at every boundary.

| Stage (kind) | Runs | Entry | Exit / promotion | Placement |
|---|---|---|---|---|
| **S1 joint SFT** (`native_state_sft`, existing) | natlang + native Neuralese SFT over the full source | earned parent | Not a terminal stage. Its checkpoints are *eligible parents* when the paired dev arm (dev v2, about 16-32 cases, same runtime) is ≥ the current selected parent and no protected-family regression is unexplained. Turn on `task_eval_every` with a cheap 4-case cadence as telemetry. | DGX learner |
| **S2 whole-graph JOLT+ RL, interleaved** (new declared `rl_controller` stage wrapping `production.py`) | rollouts from the actual checkpoint plus `native_state_rl`; mandatory SFT obligation retained | parent eligible from S1 | Per window: informative-group fraction ≥ 0.3 (else raise difficulty automatically, §5.2). Promote a window when paired dev ≥ parent and training-world frontier levels advance in ≥ 3 families. Two consecutive dev regressions trigger diagnosis and reselection, as today. | Actors: 2-3 CoreWeave 1-GPU sandboxes (native actor, 8 CPU, measured RAM). Learner: DGX, or a cloud learner once its launch defects are fixed |
| **S3 program adaptation** (new `program_adaptation`) | per program: self-teach round 0, SFT/RL/RFT rounds, paired dev (base/untuned vs tuned), activation; distill training-world successes back as an S1 source continuation | a backbone parent selected in S1/S2 whose untuned dev success on ≥ 3 supply programs is >0 and <1 (headroom both ways). Bounded pilot allowed earlier (§5.1) | A program is *done* at a dev plateau (patience 2) with tuned-vs-untuned lift and world-cluster interval. The stage is complete when ≥ 12 programs across ≥ 6 structurally different families have measured releases, and a confirmation arm has run once per selected release. Retune when the backbone parent changes and paired dev moves by ≥ 2 cases. | CoreWeave 1-GPU sandboxes, one program per sandbox (4 in parallel) |
| **S4 learned optimizer** (new `learned_optimizer`) | bank-init → corpus → outer cycles (`train_bank`/`train_basis`/`train_keys` on) → outer meta-RL with real task outcomes → meta-examples → controller tune → held arms | S3 complete on the corpus side; held side = 25% of families never used by S4 | Held-family adaptation: learned release ≥ crisp-baseline and ≥ in-bank gradient fit at matched compute on development, then confirmation once. Report recovered-gain ladder. Planted-control recovery passes. | bank-init, corpus and outer training sequential on one GPU (DGX between backbone windows, or one sandbox). Meta-RL scoring and meta-examples sharded on sandboxes |
| **S5 continuous** | backbone continues S1/S2 with distilled S3 data; S3 retunes on new parents; S4 adopts new banks with label refits | — | per-stage contracts above | — |

### Use of the four CoreWeave accounts, now

- **cw1-cw2: RL actors** for the DGX learner. Rollout collection is the RL bottleneck: 4 rows took a full round of a
  32k-context native actor.
- **cw3-cw4: the S3 pilot** (§5.1) on the current selected parent. It measures per-root time, self-teach yield and
  first lift, and it shakes out lane bugs. It is a machinery qualification bounded at about 24 GPU-hours per program,
  not a production adapter, and it respects the owner rule not to fit early checkpoints for hours.
- When S3 entry holds, move all four to program tunes and keep DGX on the backbone.

## 5. Changes, ranked

| # | Change | Why | Effort |
|---|---|---|---|
| 1 | **Fix RL selection signal.** Raise `rows_per_round` to 16-32 and `group` to 6-8 for tasks at the frontier. New families start at the family's *median* level, or at a level predicted from sibling families, not the easiest. Drop level-0 tasks from practice once a family is 4/4 at level ≥ 1. Add an informative-group-fraction target: when the last window is below 0.3, shift practice one level up in every saturated family. Record `spread_yield` per window. | Directly removes the structural zero-spread cause (§3.3). Config plus `rl/curriculum.py`; keep breadth and custody. | S: 0.5-1 day plus tests |
| 2 | **Behavioural promotion gates in the recipe**: `promotion` declaration, `natlang.behavior-promotion/1` receipt from paired arms, `require_gate` extension; enable a cheap `task_eval_every`. | Ties every stage to task behaviour, as the owner asked. Reuses the existing task_eval and Pop evaluation receipts. | M: 2-3 days |
| 3 | **`program_adaptation` stage kind** wrapping `natlang tune` (rounds, paired evaluate, activation, distill export) with sharded execution and an S3 report schema. Inputs and outputs as in §2. | Makes tuning a firm recipe stage with lineage-bound inputs and progress. | M: 3-4 days |
| 4 | **`learned_optimizer` stage kind** wrapping the lane stages (not the bash script) with the operation store as stage state; report includes bank publications, outer steps, meta-RL rounds and held arms. | Firm metalearner stage. The mechanism exists; this is orchestration plus a gate adapter. | M: 3-4 days |
| 5 | **Diverse program and task supply**: build natlang codebases from the existing 98 design-brief codebases and the codebase sources, each a reusable application with internal APIs and state and 4-8 distinct operations (investigate, repair, implement, plan, changing-state control) with exact effect/state oracles. Feed them to RL and as tuning programs. Grow the program supply beyond "ordered rules → enum". | Fixes formulaic RL and gives the metalearner families that actually differ. Oracle-built, no teacher generation. | L: 1-2 weeks, parallelizable; start with 10 codebases × 5 operations |
| 6 | **`rl_controller` stage declaration**: bind task source, curriculum, rows and group, and policy loss into recipe identity, with the promotion contract. Fix the cloud lineage defects (SFT-only advancement, export criterion, unconditional SGLang install). | RL becomes a recipe stage, and cloud actors become usable. | M: 2-3 days |
| 7 | **Lane prerequisites**: dense JOLT+ term in the tune trainer (reuse `rl/privileged_scoring.teacher_scores`), batched concurrent rollouts and evaluations per program, label-only refits after bank publication. | Cuts S3/S4 cost by an estimated 2-4× and aligns tune RL with backbone RL. | M: 3-5 days |
| 8 | **S3 pilot on current parent**: port ticket, invoice and inventory to LFM3 plus 2 supply programs; self-teach, 2 RL rounds, paired dev. | Real cost and yield numbers, and lane bugs found before the full lane. | S to run: about 2 days wall on 2 sandboxes |
| 9 | Registry admission of distilled training-world tuned rows and joint-target manifests (`training/corpus-manifests/`) | Closes the S3 → S1 loop through normal source admission. | S: 1 day |

Order to execute:

1. Item 1 immediately (next RL window).
2. Item 8 in parallel on idle cloud GPUs.
3. Items 2, 3 and 4, which make the stages firm.
4. Item 5 continuously.
5. Items 6 and 7 before S3/S4 run at scale.

## 6. Evidence pointers

- Recipe: `/home/werg/data/lfm3-rl-dgx-20261029/ops/earned179-joint-r38-a4-window64-1103/recipe-runtime.json`.
  One stage; `task_eval.min_successes 0`, `task_eval_every 0`.
- Stage kinds and gates: `training/neuralese/natlang_neuralese/train/recipe.py` (`HANDLERS`, `require_gate`).
- RL config: `/home/werg/data/lfm3-rl-dgx-20261029/ops/quality-parent-onpolicy-runtime19-1102/config.json`.
  Last round: `/home/werg/data/lfm3-rl-dgx-20261029/quality-parent-onpolicy-runtime19-1102/state.json`
  (`group_stats` all `spread: false`).
- Selector: `training/neuralese/natlang_neuralese/rl/curriculum.py`.
- Lane: `training/neuralese/learned-optimizer/README.md`, `run-lfm3-lane.sh`;
  `plans/neuralese/JOINT_LEARNED_OPTIMIZER_STATUS.md`.
- Past tuning runs: `plans/PROGRAM_TUNING_RUNS_20261026.md`; `/home/werg/data/program-tuning-20261026/`.
- Program supply: `/home/werg/data/program-supply-20261027/programs/programs-prepared.json`.
- Census outputs: `/tmp/claude-1000/-home-werg-natlang/4c9bd489-afe4-4847-b74c-378caa27432c/scratchpad/audit-tune/rltasks.json`
  (and `.samples.json`).
