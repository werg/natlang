# Joint learned optimizer: implementation status

Implementation of [JOINT_LEARNED_OPTIMIZER.md](JOINT_LEARNED_OPTIMIZER.md), 2026-10-27. Mechanism:
[`training/neuralese/natlang_neuralese/tune/learned/`](../../training/neuralese/natlang_neuralese/tune/learned/DECOMPOSITION.md).
Natlang program: [`applications/learned-optimizer/`](../../applications/learned-optimizer/DECOMPOSITION.md).

**No training run, measurement or capability claim exists yet.** Everything below is implemented and tested on the
tiny native fixture models (LFM2 and the shared paths); nothing has been trained on real programs, and no task
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

## Not yet done (next work, in rough priority)

1. **Run it — on hold until LFM main training is done (owner, 2026-10-27).** Prepared, not running: specs and
   `run-after-lfm-main.sh <final checkpoint>` under `/mnt/external/natlang-learned-optimizer-20261027` (bank-init
   rank 16 from the ticket, invoice and inventory plans, then `corpus` with `rebase` onto the final checkpoint). A first
   bank-init on the step-39,495 parent was stopped unfinished; nothing was produced. The 51 supplied programs have
   persisted grouped splits (`/home/werg/data/program-supply-20261027/splits-persisted.json`: 4,416 train, 1,416
   development, 696 confirmation roots); case export binds the student checkpoint, so export, teacher collection and
   import also wait for the final LFM model. Afterwards: outer training, `natlang tune --learned` on development
   roots, the first ladder, and the grid step from fitted coefficients (`numbers.grid_step_for`; 0.05 is a start).
2. **Oracle outcomes in collection.** `collect` uses the shaping reward; per-proposal oracle task outcomes are accepted
   (`outcomes`) but not yet produced by a TS rollout worker that executes proposals on query worlds. The release path
   already executes development roots through `evaluateRelease`.
3. **Planted controls run continuously** as a scheduled worker (`controls` command exists; no scheduled job).
4. **Teacher collection for the supplied programs** (case export against the final student, reviewed Luna campaign,
   import), then tune rounds and `corpus` over them; held with item 1, owned by the program-tuning lane.
5. **Mellum**: the code is shared and generic over expert axes, but shared hidden-side expert bases are not tied
   (expert bases are per expert); no Mellum run.
6. **Reachability measurement** of directly tuned state targets (per-target writer context fit) is not implemented.
7. **Joint memetic search**: `rl/memetic_program.py` is not generalized to bundle candidates; the `adapt` program's
   operators (learned proposal, refinement, retrieval) cover the episode.
8. **Registry/recipe admission** of target manifests under `training/corpus-manifests/` and production-recipe
   scheduling of the optimizer objectives.
9. **Live measurement** of the model-facing request, writer template and NL stages (`plans/MODEL_FACING_CHANGES.md`).
10. Coefficient head readout on the native HTTP server (the Python engine path has it; serving the executor never
    needs it).

## Notes from implementation

- Backward must run inside `run_native` (its precision carriers depend on it): a reference that backpropagated
  outside gave ~1e-3 relative adapter-gradient errors on an FP32 fixture.
- Unconstrained continuous fits can give a component opposite signs across groups, which grid + bounded modulation
  cannot execute; fits therefore optimize in the learner's own parameterization from the start.
- Component units: relative weight energy (`q_m = 1/(numel ms(W_m))`); a grid step of 0.005 saturated where useful
  coefficients were ~0.6 on the fixture. The step must be sized from fits.
