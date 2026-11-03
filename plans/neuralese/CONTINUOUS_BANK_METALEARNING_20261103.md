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

Fast weights are one program's component-token mixture over the bank: free logits over the null token and K
components, plus one bounded gain per target group (depth quarter x matrix family). This is the executed law
`c_kg = g_g * p_k` of `mixture.py`, owner decision 13, the same parameterization the writer head, `natlang tune
--learned` and the release format use. No new mixture form is introduced. Slow weights are the bank:
component factors `P_k, Q_k` (rank `r_c` inside the basis) and the shared orthonormal basis `U_m, V_m`
(rank R per target matrix). Every attention, convolution and FFN matrix of the native target table is included.
`train_bank` and `train_basis` are explicit and required (AGENTS.md "Learned LoRA mixtures and bases").

One outer step:

1. Sample `meta_batch` meta-training families. For each one, draw disjoint support (3) and query (2) worlds from its
   train worlds, one verified trajectory per world.
2. **Inner step** (`fit.fit_in_bank`, unchanged except that it now accepts live bank values): Adam on the free
   logits and gain logits for `inner_steps`, then on the fixed top-p law for `refine_steps`. The loss is the support
   rows' teacher-forced loss, a mean over rows of per-graph CE on assistant targets only. The bank is held fixed. The
   result is the executed coefficients `c*`.
3. **Outer step**, first order: the query rows' teacher-forced loss at `c*`, held constant, backpropagates into the
   live component factors and basis through the exact composition `U (sum_k c*_k P_k Q_k^T) V^T`. The gradient of
   the sampled families is averaged and one Adam step is taken (bank lr 1e-2, basis lr 1e-4). This is FO-MAML on
   the slow weights. The adapted parameters (coefficients) and the meta-trained ones (bank) are disjoint, so Reptile's
   weight-difference update does not apply.
4. **Invariants** (`restore_invariants`): the basis is re-orthonormalized by QR, with the factors re-expressed
   exactly (`P <- R_u P`, `Q <- R_v Q`, so the composed update is unchanged). Every component is then rescaled to
   unit relative energy `sum_m q_m ||C_km||^2 = 1`. Coefficients therefore stay comparable, and the bounded gains
   bound every adapted update. The renormalization rescales a component, so Adam moments are approximate for one
   step. This is recorded, not hidden.

Second-order MAML is not used. With coefficient-only inner loops the dominant first-order term is the query
gradient at the adapted point. The second-order term would need a Hessian-vector product through 8 inner steps of a
1.7B model per episode.

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

### Promotion evidence (`bank_metalearning` gate)

Held families are the held-out unit: whole families never enter meta-training. Within a held family, worlds split
into train (support for the evaluation fit), dev (scored) and confirmation (scored once). Four arms run at matched
adaptation compute, meaning the same number of support forward/backward passes (`inner_steps + refine_steps`):

| arm | what is fitted |
| --- | --- |
| parent | nothing (zero coefficients, exactly the parent) |
| meta | coefficients over the meta-trained bank |
| initial | coefficients over the identical bank before meta-training (seeded random components, weight-SVD basis) |
| lora | a plain per-program free LoRA (`free-lora`, rank 8, B zero), the tuner's own parameterization |

Promotion requires the meta arm's mean dev loss to be strictly below the initial, LoRA and parent arms on at
least `min_held_families` families, then the same on confirmation worlds once. The gate (`require_gate`) also
requires actual outer updates of both bank and basis. Losses are teacher-forced losses on verified trajectories.
Task-outcome lift needs rollouts of adapted programs (the program-adaptation development arms) and is not claimed
by this gate.

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

Per family in the meta batch: `inner_steps + refine_steps` support passes, each a forward and backward over 3 rows,
plus one fitted support forward and one query forward/backward over 2 rows. The default of 6+2 steps and
meta_batch 2 gives 2 x (8 x 3 + 3 + 2) = 58 row forward/backward-equivalents per outer step. Rows are capped at
4096 tokens. Measured per-step seconds are in "First run" below. A held evaluation costs, per held family,
3 fitting arms x (8 x 3 support passes + 2 dev rows) plus 2 parent rows.

## Deployment and program adaptation

- **Deployment tuning is the inner step.** `natlang tune <program> --learned spec.json` accepts a meta-learned bank
  without optimizer `state` (policy `crisp-baseline`). The release is the in-bank fit (`apply.inner_fit`: `refine`
  from the incoming bundle, `fit_steps`), kept only when it lowers the exact support loss. The release, validation,
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
- **Operational note.** A stage-shard relaunch from a final (failed) export is shut down by the exporter immediately.
  Move those records under `.cloud/superseded/` first, as the tuning pilot does.

## First run (2026-11-03)

Results are filled in below as measured.
