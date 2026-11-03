# Continuous meta-learning of the shared LoRA bank (design and first run, 2026-11-03)

Owner decision 2026-11-03 (AGENTS.md "Meta-learning main line"): the LFM3 meta-learner is a shared LoRA
bank/basis meta-trained so that adapting to a new program is a cheap fit of mixture coefficients. The natlang
learned updater of [JOINT_LEARNED_OPTIMIZER.md](JOINT_LEARNED_OPTIMIZER.md) becomes a later refinement.
Mechanism: [`tune/learned/meta_bank.py`](../../training/neuralese/natlang_neuralese/tune/learned/meta_bank.py),
stage [`tune/learned/bank_stage.py`](../../training/neuralese/natlang_neuralese/tune/learned/bank_stage.py),
recipe kind `bank_metalearning` ([`train/adaptation_stages.py`](../../training/neuralese/natlang_neuralese/train/adaptation_stages.py)).

## Why this replaces the learned-optimizer main line

- The learned-optimizer lane never ran on a real model. Its outer loop needs complete inner fits that take hours
  each. A live tuning pilot collects about one root per 20 minutes, and the plan estimated 450-900 GPU-hours.
- The tuning program supply is one structural shape (`PROGRAM_ADAPTATION_PILOT_20261103.md`), so a learned
  optimizer has nothing transferable to learn from it.
- The part with solid support is a shared low-rank basis with small per-task mixtures. LoRAHub composes fixed LoRAs
  by gradient-free coefficient search. Polytropon/MHR route tasks over a shared inventory of LoRA modules trained
  jointly. CAVIA meta-learns a network so that a small context vector is the only per-task parameter. ANIL and
  BOIL show that inner loops can be restricted to a small parameter subset. Reptile and first-order MAML show that
  the outer update needs no second derivatives. Text-to-LoRA and HyperLoRA amortize the per-task fit with a
  hypernetwork. The bank is CAVIA/MHR with the bank as the meta-learned body and the mixture as the context.

## Algorithm

Revised 2026-11-03 (supervisor decision from an owner request; supersedes the first-run law below it).

**Initialization from data.** Plain rank-8 LoRAs are fitted on every meta-training family's support worlds, with
the baseline's parameterization and pass count (`family_lora_fits`, saved per family and per parent). The basis
`U_m, V_m` (rank R = 32) is the top singular directions of each matrix's stacked deltas `[D_1 ... D_F]`
(`delta_basis`, computed from the factors). The components are the uncentered principal directions of the fits'
basis coordinates, jointly over all matrices with weights `sqrt(q_m)`. Directions are kept up to `variance`
(0.9) of the energy, capped at `components` (64) and factored to rank `component_rank` (16) (`pca_components`).
The learned initial coefficients start at the families' mean projection. The per-coefficient step sizes start at
two spreads over the inner steps. A random bank on a weight-SVD basis (`init: random`) is the first-run
construction; it recovered about 9% of LoRA's gain at step 0.

**Fast weights.** Each program fits free, signed coefficients `a[k, g]` per component and coefficient group, where
the groups are depth slices x matrix family (`layer_groups` = 8). The executed cores are
`C_m = sum_k a[k, g(m)] P_k Q_k^T`. The inner step is Meta-SGD: `inner_steps` (12) Adam-preconditioned steps from
the learned initial coefficients `a0`, each scaled by learned per-coefficient step sizes `exp(log_step)`.
The coefficients are signed and unbounded: the mixture-token law (`c_kg = g_g p_k`, owner decision 13) remains
the learned optimizer's writer law. A meta-learned bank publishes its own law (`bank-manifest.json`
`inner_fit`), and `natlang tune --learned` deploys by it (`apply.Context.meta_fit`, root scope).

**Residual.** Optionally, a per-program residual core `X_m Y_m^T` of rank `residual_rank` (2) in basis
coordinates is fitted alongside, by Adam. It is evaluated as its own arm (`meta_residual`). Meta-training's inner
step is coefficient-only.

**Outer step**, first order:

1. Sample `meta_batch` meta-training families, each with disjoint support (3) and query (2) worlds.
2. Run the inner step on the support rows with the bank held fixed.
3. Backpropagate the query loss at the fitted coefficients into the live factors and basis. The coefficients'
   gradient `g` also gives the meta-parameters their first-order gradients: `dL/da0 = g` and
   `dL/dlog_step = g * (-travel) * step`, where `travel` is the sum of the inner step directions.
4. Add a **weight-space regression** over `aux_families` (8) meta-training families. For each one, take the share of
   its LoRA delta's relative energy that the live bank leaves unexplained at its least-squares per-group
   coefficients. This needs no model forward and gives a dense gradient into the components and basis.
5. Take one Adam step (bank 1e-2, basis 1e-4, meta-parameters 1e-2).
6. **Invariants** (`restore_invariants`): QR re-orthonormalize the basis, re-expressing the factors exactly, and
   renormalize each component to unit relative energy. Coefficients and step sizes are rescaled by the inverse
   factor, so every composed update is unchanged (`_rescale_meta`).

Second-order MAML is not used.

### Losses

Support and query losses are the native graph loss (`data.native_graph.graph_loss`) of verified trajectories, with
assistant actions and typed answers as targets and the prompt, tool output and feedback at zero weight (AGENTS.md
"Supervise the model's own outputs"). Episodes on whole graphs keep their producer/consumer edges, because
`SupportSet` runs the graph forward. The current pools hold single-node graphs.

On-policy reward in the inner step is a later extension. The mixture law already supports the exact expected
score over component tokens (`mixture.expected_score_loss`). Once rollouts of adapted programs are cheap enough,
an RL inner step plugs in there.

### Schedule and continuity

- The stage runs on the latest promoted (behaviour-selected) parent and never advances backbone exposure.
- **Following the backbone.** Rerunning the stage with a newer `--continue-from` continues the same live bank,
  optimizer and episode RNG through a recorded parent transition (`state.parents`). Basis and factors are
  coordinates of weight updates, valid on any same-architecture parent. The relative energies `q_m` are recomputed
  on the new weights and the components renormalized. The un-meta-trained comparison bank is rebuilt by the same
  seeded construction on the new parent.
- **Staleness bound.** The bank is meta-trained against parent P while the backbone continues. A bank trained on a
  parent more than `max_parent_lag_steps` (default 200) backbone optimizer steps behind the current promoted parent
  must take the parent transition before its next evaluation or publication. Promotion evidence is valid only for
  the parent it was measured on (`report.checkpoint_sha256`). The supervisor applies the transition at each parent
  promotion. The stage refuses nothing silently: the report names the parent.
- **Cadence.** Outer steps run continuously. A held development evaluation runs every `eval_every` steps, and
  confirmation runs once, only when development promotion holds.

### Promotion gate: held-family task outcomes (owner, 2026-11-03, AGENTS 12f799442)

A bank snapshot **promotes when it beats the current published bank on held-family task outcomes**: exact-oracle
task success of the adapted model on development tasks. The published bank is initially the un-meta-trained bank of
this lineage (the PCA bank) and afterwards the last promoted snapshot (`state.published`). Confirmation tasks are
scored once, only after development holds (`bank_outcomes.py`, `outcome_promotion`; `require_gate` demands
`held.basis == "task-outcome"`).

- **Held task families** are RL task families from the stage's `task_source` (train rows only, protected-related
  rows dropped, never a meta-training family; `task_families`). Each family's tasks are cut by a seeded hash into
  support (4), development (4) and confirmation (4).
- **Adaptation evidence** is the parent's own oracle-verified successful rollouts on the support tasks. They are
  collected once per parent and shared by every arm. A family where the parent solves no support task cannot be
  adapted, and every arm there is the parent. Families with evidence are counted (`min_task_families`, 4).
- **Execution.** Each arm's fitted adapter becomes the default adapter of an in-process native server on the
  stage's own engine. The family's tasks run through the production collector (`rl.production.collect`, student
  root mode, full prompt contract, `task_samples` (4) attempts per task). They are scored by the shared task scorer.
  The actor receipt binds the arm's adapter (`rl_policy_binding.bank_adaptation`). Collections are
  resumable and keyed by the exact adapter.
- **References, not gates:** the parent, and plain per-program LoRA at 1, 4 and 24 passes. Deployment tuning picks
  whichever adapts better at its own budget.
- **Steering:** teacher-forced held dev loss (`evaluate`) runs every `eval_every` outer steps. It reports means over
  all held families and over the families without a near-duplicate scored world (`similarity` < 0.8, the highest
  target-token `difflib` ratio to a support world), for the parent, meta, meta+residual, initial and LoRA arms (LoRA
  at the matched and reference budgets). It is recorded as `steering` in the report, never as promotion.

## Episode supply

Verified trajectories are required, and the stage takes them from an explicit pool built by `bank_stage pool`.
Every row passes the protected-worlds union when the pool is built and again when the stage starts.

- **Parent rollouts (target supply).** `rollout_rows` lowers successful, oracle-checked RL rollouts
  (`outcome == task_success`, quality 1.0, train split, optionally restricted to the parent's checkpoint SHA). Each
  model turn becomes one node with its sampled reply as the target. Family is the task family and world is the task.
  Episodes that exchanged Neuralese state go through the whole-graph path and are counted, not lowered.
- **Verified teacher programs (first run).** `graph_rows` reads admitted native graph sources whose graph id names
  a family and a world. The admitted joint-SFT source has 27,908 teacher programs; 20,458 of them have exactly five
  decision invocations, and some have up to 24. A program is a family and a decision invocation is a world: a
  program with several invocations, which is the unit of program tuning.

Supply available at the first run (measured 2026-11-03): only **20** rollouts of full67 exist
(`quality-parent-onpolicy-runtime19-1102`). There are 448 successful rollouts across all LFM3 checkpoints, mostly
1-4 tasks per family, so too few worlds to form support/query splits. The collection-only selector pilot on the
57-family pool had produced no completed results after 17 minutes on the contended DGX. The first run therefore
uses verified teacher-program trajectories. Actor provenance is recorded per row (`actor.kind: verified-source`).
Parent rollouts enter the same pool format as collection produces them; the RL controller's rounds and the
selector pilots are the source. No paid or new teacher generation is involved.

## Compute per outer step

Per family in the meta batch there are `inner_steps` (12) support passes over 3 rows, plus one query
forward/backward over 2 rows: 2 x (12 x 3 + 2) = 76 row forward/backward-equivalents per outer step. The
weight-space regression adds no model forward. The PCA initialization costs one LoRA fit per meta-training
family: 12 passes over 3 rows, measured at 138 s on the contended GB10 (64 families, about 2.5 h once per parent).
A task-outcome evaluation costs, per held family:
- one parent support collection (4 tasks x 4 attempts);
- for each of the parent, published, candidate and 3 LoRA-reference arms, a fit plus 4 tasks x 4 attempts. Fixed
  arms are reused while the parent is unchanged.

## Deployment and program adaptation

- **Deployment tuning is the inner step.** `natlang tune <program> --learned spec.json` accepts a meta-learned bank
  without optimizer `state` (policy `crisp-baseline`). The release is the bank's published inner-fit law
  (`apply.Context.meta_fit`: Meta-SGD coefficients on the root scope from the learned initial values; older banks
  without a law refine their mixtures), kept only when it lowers the exact support loss. The release, validation,
  activation and rollback are the ordinary pipeline. `bank-manifest.json` is the bank file the spec names; its
  store is the stage's `store/`.
- **`program_adaptation`** reruns per promoted parent, warm-started from the previous releases where the tuner can
  continue them. Its fits are the per-program baselines this stage is compared against (bound as `adaptation`) and
  remain distillation sources for ordinary source admission. Tuner fits on the bank basis (`Recipe.basis`) are exact
  bank coordinates (`ComponentBank.adapter_cores`), so they can also seed bank growth (`growth.py`).

## Amortized predictor (second phase, not implemented yet)

A coefficient predictor maps program text plus a few examples to the mixture (logits and gains). It is the
existing writer and mixture head (`writer.py`, `mixture.MixtureHead`): the decision hidden state after the
program's text and the available components' keys. It is trained first by KL/gain regression on fitted coefficients
(`mixture.mixture_kl`, `gain_loss`, labels from the meta-learner's inner fits on the meta-trained bank), then refined
by the exact expected-score RL (`meta_rl.py`). This reuses the learned-optimizer outer trainer with
`train_bank: false` against a frozen bank snapshot. It starts once the meta-trained bank passes its gate, so that
labels are not refit after every bank update. Prediction replaces the inner fit only where it matches the fit's
held loss at lower compute.

## What changes in the learned-optimizer lane (S6 and JOINT_LEARNED_OPTIMIZER)

- `bank_metalearning` is the production meta-learner stage. `learned_optimizer` stays declared as a later, optional
  refinement stage, a natlang updater over the trained bank. It is not part of the successor recipe and no longer
  requires being run for the main line. Its components (bank, mixture law, in-bank fit, writer/head, meta-RL) are
  reused here, not forked.
- S6 section 4 (direct tuning as the reference method) and section 8 (matched-compute comparison on held families)
  hold unchanged. The plain LoRA arm is the direct-tuning reference. S6 section 5 (learned updaters) is deferred
  behind this line. Its first-order default (section 5.5) matches the choice here.
- The bank initialization is seeded random components on the weight-SVD basis, not gradient-subspace or source
  components (`growth.py`). This makes the un-meta-trained comparison clean. Growth from tuner fits remains
  available once the bank is in production.

## Recipe: gated on parent promotion, concurrent with SFT

Successor recipe (new file; the existing `adaptation-stages-successor-1103` is unchanged):
`/home/werg/data/lfm3-rl-dgx-20261029/ops/bank-metalearning-successor-1103/recipe-runtime.json`, SHA-256
`d3befd55ad67973e35b0a8f85ccea90ea73a7db3a6c850395d705c78b9c1f881`. Stages:

- `native_joint_sft_v4` is unchanged.
- `program_adaptation_s3` binds `continue_from: promoted_parent` (full67).
- `bank_metalearning_s4` binds `episodes: bank_episode_pool`, `protection` and `continue_from: promoted_parent`.
  Its parameters: 2000 outer steps, held evaluation every 100 steps, 8 held families minimum.
- `learned_optimizer_s4` is removed from the recipe and remains a declared stage kind for the later refinement.

Both adaptation stages declare `requires: [native_joint_sft_v4]`. The recipe runner lets an adaptation stage start
before its headless SFT predecessor completes once `continue_from` and `parent_selection` are both bound. So they run
concurrently with joint SFT, gated on promotion, not on SFT completion; SFT completion is about 200 days out at
current throughput.

They stay `admitted: false` for one concrete missing input. The selected full67 parent has no
`natlang.behavior-promotion-selection` record (`train.promotion select-parent`). It has only the supervisor's
behavioural reselection notes, and this stage does not fabricate one. Admission is a recorded handoff that binds that
record as `promoted_parent_selection` and sets `admitted: true`. Each later promotion rebinds `promoted_parent` and
`promoted_parent_selection`. `bank_metalearning` continues its live bank on the new parent; `program_adaptation`
reruns in a fresh stage directory.

Warm-started program adaptation is the intended semantics, but `tune.stage` has no warm-start input yet. Each rerun
tunes from the new parent. This is an open item for the tuning lane.

## Production placement and cadence

- **Where.** The DGX GB10 has had no admissible headroom for a 1.7B-parameter fit with backward all day
  (2026-11-03: the RL pilot, weight-space jobs and SFT preparation hold it; ledger headroom near or below zero). The
  meta-learner needs one GPU with about 20-30 GB, plus an episode pool of tens of MB. Production runs on one dedicated
  cloud GPU sandbox through the stage-shard lineage (`rl_cloud.py`, lineage `tune.module =
  natlang_neuralese.tune.learned.bank_stage`), exported every 30 minutes and at the end to the DGX hub. When the DGX
  frees, the same stage command runs there through the ledger at medium priority.
- **Cadence.** Run continuously in blocks of `outer_steps` with a held development evaluation every `eval_every`
  steps. At each backbone promotion, stop at the next completed outer step (the state is saved after every step),
  rebind the parent and resume (recorded parent transition). Confirmation is spent only when development promotion
  holds. Grow the episode pool as RL collection produces verified successful parent rollouts (`bank_stage pool
  --rollouts`). A grown pool is a new stage directory (its `operation.json` binds the pool). Seeding it with the previous `learner/` state continues the same bank: the learner binds its recipe and initial bank, not the pool.
- **Operational note (owner rule, AGENTS 43c88f234).** Never stop a qualifying GPU sandbox while GPU capacity is
  still needed. A changed configuration restarts inside the held sandbox (push the new seed/stage dir, move the old
  run's state aside, start the new driver in place), as the cloud agent did for production on 8120d5a5. On
  2026-11-03, stopping the obsolete random-bank sandbox 6618cead and requesting a new one lost the cw1 GPU to a
  StartupTimeout.
- **Operational note.** A stage-shard relaunch from a final (failed) export is shut down by the exporter immediately.
  Move those records under `.cloud/superseded/` first, as the tuning pilot does.

## First run (2026-11-03)

Setup: parent `bf95-quality-source-zero-sgd-v2-1102` (sha 71c3eb45…), pool `pool-teacher-v1` (64 meta-train / 16 held
teacher-program families, assistant-only targets, at most 4096 tokens), defaults from `MetaRecipe` (rank 8, 16
components of rank 4, 6+2 inner steps = 24 support passes per fit), targets restricted to the token-mixing projections
(`self_attn|conv.in_proj|conv.out_proj`, 81 matrices). DGX ledger unit `bankmeta-run-dgx-5` (medium, resumable),
out `/mnt/external/natlang-bank-metalearning-20261103/run-full67-dgx-v2`.

**Step-0 held development (teacher-forced loss per target token, mean over 16 held families):**

| arm | mean dev loss | gain over parent |
|---|---|---|
| parent (no adaptation) | 2.310 | 0 |
| un-meta-trained (random) bank, coefficient fit | 2.171 | 0.139 |
| plain per-program LoRA (rank 8, 24 passes) | 0.785 | 1.525 |

The meta arm equals the random bank at step 0. The coefficient fit on a random bank recovers only about 9% of
what a free LoRA recovers at matched passes. That gap is what meta-training must close. The LoRA advantage is
inflated by about 4–5 families whose dev worlds nearly repeat their support worlds (LoRA dev loss 0.002–0.09 on
2757f652, e26c62b2, f48bf161, a4a1a48a, a7af7b61). Exact repeats are removed by the world dedupe in `build_pool`
(pool v3); near-repeats are not.

**Compute:** one outer step (meta-batch 2: 2 x (24 support passes + 2 query rows with backward)) took 187 s on the
contended GB10 (first step: grad norm bank 0.157, basis 1.55). A full four-arm 16-family held evaluation took 54 min.
Later evaluations reuse the parent/initial/LoRA arms (measured once per parent) and refit only the meta arm.

**Findings during the run:**
- All 153 projection targets with dense FP32 deltas exceed one H100 (77.9 GB). Token-mixing projections only (81)
  fit in about 18–40 GB. FFN targets need sparse/factored delta application before they can join.
- cuSOLVER SVD can return a numerically non-orthonormal basis; the basis is re-orthonormalized by QR.
- The cloud runs were capacity-limited (ResourceExhausted) and then OOM. The DGX run is preemptible and resumes per family.

The random-bank run was cut over after one outer step (supervisor decision). The replacement run uses the
data-driven PCA initialization, more components, per-layer-group Meta-SGD coefficients, a residual arm and the
weight-space regression (see "Algorithm"). It runs on pool v3: exact-duplicate worlds removed, 64 meta-train and
16 held families, 729 rows. The run is DGX ledger unit `bankmeta-pca-5`, out `run-pca-v4`.

**PCA initialization (measured):**
- 64 meta-train LoRA fits took 30-140 s each on the GB10.
- The rank-32 LoRA-delta basis captures 53% of the fits' relative delta energy.
- 28 components explain 90% of the captured energy; the first two carry 34% and 22%.
- 70 targets remain: the 16-row attention gates are below rank 32 and stay with the parent.
- There are 16 coefficient groups.

**Step-0 held development, PCA bank** (teacher-forced loss per target token, 12-pass fits; the 12 families without
a near-duplicate dev world in parentheses):

| arm | mean dev loss |
|---|---|
| parent | 2.325 (2.461) |
| PCA bank, learned initial coefficients, no fit (`initial_unfitted`, 10 families) | 1.166 |
| PCA bank coefficient fit (`initial` = `meta` at step 0) | **0.675 (0.818)** |
| PCA bank fit + rank-2 residual | 0.752 (0.930) |
| plain LoRA rank 8, 1 pass | 1.659 (1.803) |
| plain LoRA, 4 passes | 0.908 (1.091) |
| plain LoRA, 12 passes (matched) | 1.345 (1.694) |
| plain LoRA, 24 passes | 2.138 (2.707) |

At step 0 the PCA bank's coefficient fit beats the matched-pass LoRA on 14 of 16 held families and every LoRA
budget on average. The random bank was at 2.171 against LoRA 0.785 on pool v1.

- About 70% of the gain over the parent is a shift shared by all teacher programs: the learned mean coefficients
  alone reach 1.166. The per-program fit adds about 0.49 nats more.
- Plain LoRA at lr 1e-3 overfits three support rows beyond 4 passes. The LoRA references are untuned baselines.
- The residual arm (lr 1e-2) hurts.
- These are teacher-forced losses on held teacher-program families. They steer meta-training; they are not the
  promotion gate, which is task outcome.

**Outer steps:** 70 s each on the GB10 (meta-batch 2, 12 inner steps). At step 1 the weight-space regression left
51% unexplained, consistent with the basis capturing 53%.
