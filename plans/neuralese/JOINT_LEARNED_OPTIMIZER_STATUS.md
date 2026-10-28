# Joint learned optimizer: implementation status

Implementation of [JOINT_LEARNED_OPTIMIZER.md](JOINT_LEARNED_OPTIMIZER.md), 2026-10-27. Mechanism:
[`training/neuralese/natlang_neuralese/tune/learned/`](../../training/neuralese/natlang_neuralese/tune/learned/DECOMPOSITION.md).
Natlang program: [`applications/learned-optimizer/`](../../applications/learned-optimizer/DECOMPOSITION.md).

**No training run, measurement or capability claim exists yet.** Everything below is implemented and tested on the
tiny native fixture models (LFM2 and the shared paths; LFM3 target tables and tokenizer on the pinned config and
tokenizer, the LFM3 lane end to end only as a dry run on the tiny fixture); nothing has been trained on real programs, and no task
outcome, ladder gap or lift has been measured. Saved artifacts and passing tests are not capability.

## Implemented and tested

| Plan area | Where | Evidence |
| --- | --- | --- |
| Number grid (signed single digits, forced template, verified tokenizer spelling; real LFM2.5 tokenizer checked) | `numbers.py` | `test_learned_optimizer_bank.py` |
| `token-expectation-v1` (exact with the other-sign branch; declared realized-magnitude approximation otherwise) | `numbers.py`, `writer.py` | exact enumeration test; branch logits equal the teacher-forced flipped sign |
| `coefficient-head-v1` (zero init, bounded refinement, centered group modulation, sign/mean/zero invariants, head-off) | `head.py` | invariants and finite-difference gradients |
| Shared basis (orthonormal, learned method `bank-shared-basis-v1`), component bank (low-rank cores, normalization in relative weight energy), snapshots, coordinate transforms, projections, residuals | `bank.py` | composed adapter executes its exact dense update through the real adapter path; planted mixture recovered by projection; snapshot round trip |
| `resolve_adapters` accepts bank-basis slices (`bank-basis-slice-v1`) with live trainable basis leaves | `model/native_adapters.py` | basis gradients exactly where cores are nonzero |
| Gradient subspaces (current-loss-weighted, equal-weight pooling with weight-SVD) and the gradient probe | `subspace.py` | probe equals autograd through the composition (1e-3) and finite differences |
| Bank views (probe/key/full selection, recorded dropout masks, array contract, view recall) | `views.py` | used throughout; contract failures raise |
| Bundle manifest, exact composition check, proposals, edits | `bundle.py` | `verify_bundle` on proposed, refined and retrieved bundles |
| Exact evidence facts, whitelisted proposer view, compact rows, crisp key-evidence/focus baselines | `evidence.py` | apply/service tests |
| Writer: one differentiable incremental native computation (contextual re-encoding with extra slots, cued arrays, decision hidden states, greedy/sampled/teacher-forced decoding) | `writer.py` | equals one exact graph forward; cotangent replay equals the connected computation (1e-4) |
| Objectives: distillation KL, gold-action CE, field-normalized state loss with exact alignment, balanced mixture CE, decoded weight loss from factors, head regression, gradient shares | `objectives.py` | distillation direction test; trainer tests |
| In-bank fits in the learner's parameterization; leave-one-binding-out focus labels | `fit.py` | lowers support loss; labels execute exactly through the head |
| Outer trainer: one snapshot, three-phase cotangent replay, writer adapter (immutable lineage), head (AdamW FP32), keys (`optimize_state`), optional bank/basis training with publication cadence, per-route cotangent telemetry, resume | `trainer.py` | writer/head/keys update; resume restores head and lineage; behavior reaches sign/digit logits only through the expectation estimator |
| Meta-RL: paired reward with separate shaping, leave-one-out advantages, SNR, filtered BC selection, `joint-ratio-v1` with regenerated bundle, returns-to-go | `meta_rl.py`, `trainer.policy_step`, `episode.py` | proposal replays at ratio 1 at the proposing snapshot; policy update moves the writer |
| Proposal collection with shaping rewards and contrast accounting | `__main__.collect` | CLI test |
| Target records (three tiers), tuner boundary export after each committed update, episode variants, exposure ledger | `targets.py`, `tune/build.py` (`emitBoundaries`) | boundary test through the real `tune.build` entry |
| Bank growth: source components, free-tune components, residual basis growth, merge/retire, coverage, saturation | `growth.py` | `bank-init` via CLI test (source components) |
| Ceiling ladder, gaps, aggregate recovered gain, warm-start effect, evaluation power, family split | `ladder.py` | CLI ladder test |
| Planted controls (mixtures, rows from planted adapters, recovery score, aligned vs descriptions-only permutation, shuffled evidence) | `controls.py` | helpers only (see below) |
| Resumable entrypoint `bank-init/targets/train/collect/propose/service/ladder` over a durable operation store | `__main__.py` | whole operation end to end incl. resume |
| Application with the `tune.build` contract; shared adaptation services (evidence, propose, evaluate, refine, retrieve, release) | `apply.py` | apply and service tests |
| `natlang tune <program> --learned spec.json` (validation/activation/rollback by the ordinary pipeline) | `ts-host/src/tune/learned.ts`, `cli/tune.ts`, `pipeline.ts` | typecheck; Python side tested |
| Natlang program: `adapt.nl` (`iterateOn` episode) and `train.nl` (scheduled training), services over the Python services | `applications/learned-optimizer/` | builds with all applications |
| Improvement records: `joint` references, `joint-bundle` kind, `proposerView` without `after`/target, `jointStep` | `ts-host/src/improvement/step-record.ts` | typecheck |

## One representation from tuner to optimizer (2026-10-27, second pass)

- **Basis.** `tune.trainer.Recipe.basis` names a bank basis artifact: every native-subspace node of a joint fit is then
  a leading slice of the learned optimizer's shared basis (`bank-basis-slice-v1`, created by the ordinary
  `create_adapter` nested path). Such a fit's adapters are exact bank coordinates (`ComponentBank.adapter_cores`,
  capture 1 in `targets.project_adapter`). Fits without `basis` are unchanged (their input identity is unchanged too).
- **Incoming bundles.** Mixtures revise an incoming adapter exactly (`composed_cores(offset=...)`; the mixture encoding
  records `incoming`, and `verify_bundle` composes with it). `natlang tune --learned` with `incoming` revises a tuned
  program and releases it unchanged when nothing improves.
- **Program input.** `program.ProgramContext` reads the tuner's own `natlang.tune-plan/v1` and build (turns lowered
  as the tuner lowers them, its `soft-init.json`, scope tree and update history). `apply`, `targets`, `collect`,
  `controls` and `corpus` all use it.
- **One target schema.** `assemble.training_target` writes `natlang.joint-target/1` (`njt1:`), with tier, program,
  incoming states and adapters, target bundle, `target_factors` (adapter IDs resolved through the tuner's exact
  `adapter_factors`) and lineage. The duplicate `targets.target_record` schema is gone. Free-tune targets come from
  builds: the final residual from the common incoming bundle, plus each recorded update as a revision of the partly
  tuned bundle (fits on the bank basis). Older fits on the weight-SVD basis still give final-residual targets
  (behavior and state targets exact; the decoded weight loss uses their exact effective factors).
- **Export.** Learned releases and tuner nodes share `tune.trainer.artifact_member` and `build.export_release_nodes`.
- **Corpus.** `python -m natlang_neuralese.tune.learned corpus` reruns each program's existing tune plan on the bank
  basis with update boundaries and labels targets (resumable). `controls` measures planted-mixture recovery of the
  optimizer against the in-bank fit reference.
- **Program supply.** `python -m natlang_neuralese.tune.supply`: 18 typed decision-program families, 51 programs with
  variants (64 worlds, two invocations per world, exact oracle values, label-balanced; every program builds with the
  ts-host compiler). Prepared at `/home/werg/data/program-supply-20261027/programs`. They need the ordinary split,
  teacher collection and import before they supply tune rows; no teacher action exists yet.

## Tuning the meta-learner itself (2026-10-27)

The optimizer's policy is the Natlang program `applications/learned-optimizer`, tuned like any program. Its focus
decision is now its own stage, `adapt/chooseFocus` (used by `interpretEvidence`), so three stages have exact expected
values: `chooseFocus` (leave-one-binding-out hindsight over in-bank fits; `fit.fit_in_bank(fixed_states=...)`),
`chooseOperator` (every operator run from the same state and scored; the best one when it beats the incumbent, else
`keep`) and `shouldStop` (hindsight over a full-budget episode). `python -m natlang_neuralese.tune.learned
meta-examples` writes them in the tuner's `tune.examples.jsonl` format from counterfactual-best episodes, grouped by
program (`tune/learned/meta.py`). `natlang tune <copy of the program> --examples ...` then tunes the stages (teacher
SFT, RL). With teacher rows, the optimizer program's plan joins the corpus, so the optimizer also adapts itself.
Not covered by exact labels yet: `chooseRevision` (its operations need per-operation counterfactual services),
`interpretEvidence` digests (native spans; trained through the writer's whole-graph gradients) and
`scheduleTraining`. The writer adapter, head and keys are the trained part of the meta-learner already (outer
training).

Lane declaration and runbook: `training/neuralese/learned-optimizer/README.md` (the LFM3 lane, below).

## Tuning is whole-graph, graded and includes RL everywhere (2026-10-27)

- **One training path.** Program tuning's supervised rows, RL episodes and RFT episodes all run through one request
  builder (`tune.trainer.JointTrainer._graph_request`) and the production native gradient service: one forward over
  the whole graph, each invocation under its own binding's stack, gradients through every producer/consumer
  dependency (Neuralese included) into every adapter node, gate and soft state that ran. The earlier hand-written
  per-turn SFT loop is gone; a supervised row may hold several invocations (`bindings`), and a consumer's loss trains
  its producer's path (`test_supervised_row_loss_reaches_the_producer_through_its_neuralese_edge`). Production SFT
  already trained whole graphs this way; program tuning's separate loop was the exception.
- **RFT.** A third objective, `rft`: global supervision of whole successful episodes through all their dependencies,
  also from groups where every sample succeeded. RL rounds now alternate `sft`, `rl` and `rft` by default.
- **Graded scores.** One root score (`ts-host/src/tune/reward.ts`) for collection, evaluation and rollouts: an exact
  oracle, or a program verifier `tune.reward.mjs` in [0, 1]. The production replay and the tuner accept graded
  qualities; GRPO uses their spread, RFT a threshold.
- **Meta-learner.** `applications/learned-optimizer/tune.reward.mjs` grades its stages by the global target --
  support-loss improvement: an operator's share of the best measured improvement, the share of an episode's gain an
  early stop loses, focus F1 -- from the `reference` measurements that `meta-examples` records.
- Recipe identities are the full recipe: earlier fits continue through `parent` (weights, soft states, optimizer
  histories), not through identity exemptions.

## Whole-episode RL, effective revision plans and real task outcomes (2026-10-27)

- **Revision plans take effect.** `chooseRevision` chooses per binding among exactly the operations the services
  implement (`apply.Context.OPERATIONS`: `reencode` its instruction state, `replace-skill` its skill states, `weights`
  the LoRA mixtures on its path, `keep`); `rewrite`/`reuse-skill` are gone. `Context.selection` turns the plan into
  the soft keys and scopes an operator may change; proposal, refinement and retrieval all change exactly that
  selection and carry every other entry over from the parent (`Context._carry`), and a plan that selects nothing
  fails with "choose the operator keep". `interpretEvidence`'s `keyEvidence` reaches the writer's request
  (`ProgramContext.request(key_evidence=)`, validated against the view).
- **Meta examples for it.** `meta-examples` labels `chooseRevision` counterfactually (each operation alone per
  focused binding, scored on the support evidence), runs the episode on that plan, and also emits the whole
  episode as an `adapt.nl` root (`reference: {plan, out, incomingLoss, bestLoss}`). `tune.reward.mjs` grades a
  revision plan per binding (share of the best single-operation gain) and a whole episode by the services' own
  measurement of the bundle it returns (`<out>/bundles/<id>.json`, written by every service call), never by the loss
  the episode reports.
- **Tune rollouts reach the services.** A program may provide `tune.services.mjs` (default export
  `(example) => services`, optional `declarations`); `ts-host/src/tune/services.ts` resolves a root's services once
  for collection, evaluation, rollouts and repairs (a host's services keep the program's declarations). The learned
  optimizer's builds its `optimizer` from the root's `reference` context, so `natlang tune` runs SFT, RL and RFT over
  whole `adapt` episodes.
- **One persistent Python service.** `python -m natlang_neuralese.tune.learned serve`: JSON lines, every package
  command plus the adaptation services, contexts kept loaded per (plan, out) while plan, bank and optimizer state
  are unchanged. `applications/learned-optimizer/python.mjs` (`PythonService`, `AdaptationServices`) is the one
  client, used by the program, its training services and its tune services.
- **Real task outcomes for the outer meta-RL.** `proposals` samples as `collect` does and writes each proposal and
  each target's incoming bundle as a release fragment plus one request per program root;
  `natlang tune ROOT --score-proposals REQUEST` seals each fragment (`pipeline.releaseFromBuild`, the export stage
  shared with fits), runs it on the target's query roots (training roots outside its support, else development)
  and writes `{bundle: {before, after}}`; `collect` reads the proposals back with those outcomes. The runbook runs
  this round (`outer-rl-round.sh`) between supervised outer training and meta-examples; `TrainingServices` does the
  same when its operation names `outcomes`.

## LFM3-1.7B lane, teacher-free supply (2026-10-28)

Owner 2026-10-28: the metalearner is the LFM3 line's goal after joint SFT + RL; no additional teacher generation; no
cross-model dependence. Prepared on the worktree branch, nothing run on real models:

- **LFM3 targets.** `tune/targets.py` `lfm3`: attention q/k/v/o and the per-head output gate `self_attn.gate_proj`
  (attention class), conv `in_proj`/`out_proj`, feed-forward `w1`/`w2`/`w3` (11 attention + 13 conv layers on the
  pinned config; 12 bank groups). `DEPLOY_QUANTIZED`: the required NVFP4 point quantizes w1/w2/w3; `bank-init`
  records those targets, and folded writes there must be measured at NVFP4. Partial inventories that LFM2 and LFM3 both
  cover resolve when both tables map them identically.
- **Number grid.** The real LFM3 tokenizer spells `[ 3, 0, -2]` exactly like LFM2.5 (sign-bearing space token, one
  digit); `GridTokens.from_encoder` verifies it (`test_learned_optimizer_lfm3.py`).
- **Teacher-free supply.** `supply privileged` writes exact per-stage oracle references for training-partition roots;
  `natlang tune` defaults to `tune.teacher: "self"` (`ts-host/src/tune/selfteach.ts`): earlier verified data is reused
  first, then the student runs each remaining training root with the reference as privileged context; rows record
  the plain task, need the root oracle and admissible actions, and roots that echo the reference are excluded.
  Development roots are never collected by the self teacher; a reference naming a non-training root is an error. RL
  rounds then train on oracle rewards. This is sequence-level privileged self-teaching; the dense per-token JOLT+
  term (reverse KL to the privileged-context scoring of student samples) is not implemented (item 2 below).
- **Bank adoption.** `adopt-bank` (`trainer.adopt_published_bank`): after a publication and a label refit the trainer
  continues on the published snapshot (keys follow their components; bank moments continue exactly without basis
  training, else restart with a recorded transition, never reinterpreted through the QR change of coordinates). Before
  this, a publication left the operation stale with no continuation (`refitLabels` relabeled against the old bank file).
- **Lane.** `training/neuralese/learned-optimizer/run-lfm3-lane.sh <checkpoint> [op]` and `--dry-run`, replacing
  `run-after-lfm-main.sh` and the LFM2.5 specs (README beside it).

### Compute estimate (LFM3-1.7B; assumptions to replace with the first program's measured numbers)

Assumptions: native-server decode 60-150 tok/s single stream (llama.cpp measured ~250 tok/s on an RTX PRO 6000; the
native Python server is unmeasured on LFM3), a supply root of ~5 model turns with ~1.5k generated tokens (reasoning
on) and ~12-30 s wall; adapter/soft-state fits at ~40 TFLOP/s effective with ~1e10 FLOP per trained token (frozen
backbone, forward + backward through activations). Supply: 51 programs, 4,416 train / 1,416 development roots.

| Part | Per unit | Total | Placement |
| --- | --- | --- | --- |
| Self-teaching collection (round 0) | ~87 train roots x 12-30 s = 17-45 min/program | 15-37 GPU-h | parallel per program (sandboxes) |
| Round-0 SFT fit (3 epochs, ~300 rows x 3k tokens) | ~10-15 min/program | ~10 GPU-h | with the program |
| RL rounds (3 x 4 samples x 87 roots + fits) | 3.5-8.5 h/program (rollouts dominate, sequential per root) | 180-430 GPU-h | parallel per program |
| Development evaluations (4 rounds + paired final, 28 roots) | 0.5-1.4 h/program | 25-70 GPU-h | with the program |
| **Program tunes** | **~5-12 h/program** | **~230-550 GPU-h** | 8 one-GPU sandboxes: ~1.5-3 days wall |
| bank-init (~38 corpus-side programs) | gradients + 20-step components | ~4-6 GPU-h | one GPU, sequential |
| corpus (refit + labels: 3 subsets x 28 fit steps x 24 rows) | ~0.5 h/program | ~18 GPU-h per corpus run | one GPU |
| outer cycles (400 steps, 4 publications, 4 corpus refits) | minutes/step (writer + cotangent replay over support rows) | ~30 GPU-h + 4 x 18 GPU-h | one GPU, sequential |
| outer meta-RL round (4 proposals/target, scored on ~20 query roots) | ~1,500 proposals | ~60-120 GPU-h | scoring parallel per program |
| meta-examples (counterfactual episodes) | ~1-2 h/program | ~40-80 GPU-h | parallel per program |

Order of magnitude: **~450-900 GPU-hours** for one full lane, two thirds of it per-program work that shards across
one-GPU CoreWeave sandboxes (tunes, meta-RL scoring, meta-examples); bank-init, corpus cycles and outer training
(~120 GPU-h) stay sequential on one GPU. The largest levers: batched concurrent rollouts per program (rollouts and
evaluations run one root at a time today), fewer RL rounds where development success saturates (patience 2), and
label-only refits after publications instead of full corpus refits (item 4).

## Not yet done (next work, in rough priority)

1. **Run the LFM3 lane** on the trained LFM3 checkpoint once the LFM3 backbone port is on main (native loader and
   server for `lfm3`): first one program to measure per-root time, row yield of self-teaching and the RL/development
   outcomes, then the full lane (`run-lfm3-lane.sh`), sharded across sandboxes for the tunes. The first real
   bank-init validates BF16 numerics of the lane engine (the shared paths were tested in FP32 fixtures).
2. **Dense JOLT+ term**: per-token reverse KL of student samples to the same policy scored with the privileged
   reference, as an objective of the shared whole-graph learner (trainer `objectives`), plus the teacher-role update.
   The supply and custody it needs (privileged references for training roots only) exist.
3. **Planted controls run continuously** as a scheduled worker (`controls` command exists; no scheduled job).
4. **Label-only refits** after bank publications (`targets` against the published bank from the cycle-0 builds) to
   avoid full corpus refits; needs a check that meta-examples and apply bind the adopted basis correctly.
5. **Batched rollouts/evaluations** per program (concurrent roots on the native server) for tune throughput.
6. **Reachability measurement** of directly tuned state targets (per-target writer context fit) is not implemented.
7. **Joint memetic search**: `rl/memetic_program.py` is not generalized to bundle candidates; the `adapt` program's
   operators (learned proposal, refinement, retrieval) cover the episode.
8. **Registry/recipe admission** of target manifests under `training/corpus-manifests/`.
9. **Live measurement** of the model-facing request, writer template, NL stages and the privileged block
   (`plans/MODEL_FACING_CHANGES.md`).
10. **llama.cpp soft-state export for LFM3** (`tune/soft_export.py` writes LFM2 spans only); the lane serves through
    the native backend meanwhile.
11. Coefficient head readout on the native HTTP server (the Python engine path has it; serving the executor never
    needs it).

## Notes from implementation

- Backward must run inside `run_native` (its precision carriers depend on it): a reference that backpropagated
  outside gave ~1e-3 relative adapter-gradient errors on an FP32 fixture.
- Unconstrained continuous fits can give a component opposite signs across groups, which grid + bounded modulation
  cannot execute; fits therefore optimize in the learner's own parameterization from the start.
- Component units: relative weight energy (`q_m = 1/(numel ms(W_m))`); a grid step of 0.005 saturated where useful
  coefficients were ~0.6 on the fixture. The step must be sized from fits.
