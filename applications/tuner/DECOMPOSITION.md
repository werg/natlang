# Tuner: decomposition, part by part

Status: decomposition for [plans/HIERARCHICAL_LORA_TUNING.md](../../plans/HIERARCHICAL_LORA_TUNING.md), written
before implementation (owner rule: DECOMPOSITION.md first, review after the fact in `plans/OWNER_REVIEW.md`).
The tuner turns an ordinary natlang program into a tuned release that the normal `natlang run`/`call`/SDK path
uses without any author-visible adapter handling.

Decisions: **fn** (its own natural-language function), **inline** (one instruction of another function),
**implicit**, **crisp** (exact helper), **service** (outside world), **host** (CLI/plumbing), **pluggable**
(one interface, crisp and natural-language implementations, setting selects; `ts-host/src/runtime/pluggable.ts`).

## Policy and mechanism

- **Natural language decides.** Which root inputs to synthesize, which authorized teacher to route a call to,
  how to read a teacher disagreement, which families/subtree roots/removals to propose, which rank change to try,
  and how to explain a build or activation failure to the author.
- **Crisp measures, verifies and commits.** Splits, replay, typed/oracle verification, autodiff and optimizers,
  tensor algebra and export, ablation execution and cost arithmetic, dependency hashes, lexical binding resolution,
  cache identities and the atomic release pointer. A model cannot be asked for these.
- **Existing mechanisms are extended, not duplicated.** Program inventory: `ProgramDescriptor`
  (`ts-host/src/adaptation/inventory.ts`). Per-task pinning: `ProgramView` on `NatlangTask`. Weight actors:
  `nnt1` coefficient artifacts (`training/neuralese/natlang_neuralese/model/native_adapters.py`), whose
  `resolve_adapters` already composes an ordered stack with scales. Explicit experiment scopes:
  `withNativeAdapters` (`ts-host/src/neuralese/learning.ts`). Model dispatch: `activeNativeAdapters()` in
  `ts-host/src/native/model-recording.ts`. Hashing: `ts-host/src/adaptation/identity.ts` and
  `natlang_neuralese/common/hashing.py`. Splits: `rl/memetic_batch.py`.
- **State model.** A build is decide-then-commit: snapshot the program, data and settings; policies decide;
  crisp stages verify; `publishRelease` commits one immutable release and swaps the active pointer atomically.
  Derived values form a DAG with acyclic keys: source/data/config → trained artifacts → stack exports → release.

## Parts

### Discovery and binding (runtime, TypeScript)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Program inventory (named lambdas, inline sites, guidance) | crisp | existing `describeProgram` | Already the compiler's exact component list with stable keys and contract hashes. |
| Lexical scope tree over bound definitions | crisp | `compileScopeTree` in `ts-host/src/adaptation/weights/scopes.ts` | Deterministic structure: program root, folder per directory/`natlang.d/`, lambda per component; author tree roots and pins are inputs. Exact, not a judgment. |
| Binding manifest `natlang.adapters.json` | crisp | `ts-host/src/adaptation/weights/manifest.ts` | Versioned data contract: bindings, scope ancestry, nodes, families/gates, fallback, receipts. Validation is exact. |
| Per-call stack resolution | crisp | `WeightBinding.resolve` in `ts-host/src/adaptation/weights/binding.ts` | Resolved from the callee's definition/component, never the caller chain; replaces the lexical stack and appends explicit `withNativeAdapters` scopes. Must be exact and identical in Node and browser, so it rides on `Frame`, not AsyncLocalStorage. |
| Requested/effective identity, skipped artifacts, fallback | crisp | same | Dependency-consistent fallback is a rule over the manifest's declared fallback chain, not a judgment. |
| Release record, activation, rollback | crisp | `ts-host/src/adaptation/weights/release.ts` | Exact durability: immutable release directories, atomic pointer rename, previous coherent release retained. |
| Automatic discovery in `run`/`call`/SDK | host | `ts-host/src/cli/main.ts`, `createNatlangRuntime` option `weights` | Plumbing: the loader finds the active release beside the program; no flags. |
| Status `untuned/tuning/ready/stale/degraded` and author-facing reason | crisp state + fn explanation | `explainTuneStatus.nl` (pluggable, crisp default) | The state is computed exactly; the sentence for the author about what to do next is natural language. |

### Data (`natlang tune` collection stage)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Split roots by source/world component before expansion | crisp | extend `rl/memetic_batch.py` selector | Leakage prevention must be exact and global. |
| Propose synthetic root inputs from training-only signatures/examples | fn | `proposeRootInputs.nl` | Writing plausible inputs is a judgment about meaning. Inner arguments are never invented. |
| Whole-program execution with a teacher, capture of inner calls | crisp + service | existing runtime trace/graph capture; teacher via configured services | Exact capture of what ran; the teacher is an outside service. |
| Replayable worlds | crisp | existing journaled services (`frame.services`) | Side effects must not repeat. |
| Route a call to an authorized teacher | pluggable | `routeTeacher` (crisp default: measured quality/cost frontier with minimum quality; NL alternative `routeTeacher.nl`) | Hot policy; the crisp default is the measured frontier rule, the NL implementation can weigh call descriptions. Only owner-authorized actors are eligible (crisp filter). |
| Interpret disagreement among samples | fn | `diagnoseDisagreement.nl` | Reading why answers differ is a judgment; the evidence fields stay separate and crisp. |
| Evidence fields (oracle/enclosing/type/agreement/judge) | crisp | `evidence` record in data schema | Separate exact facts; no collapsed "verified" flag. |
| On-policy repair of a student prefix | service + crisp | teacher continuation + reverification | The teacher writes; crisp reverifies the continuation and downstream result. |

### Training (`natlang_neuralese/tune/`)

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Basis, parameterizations, stack algebra, export | crisp | `tune/bases.py`, `parameterization.py`, `stack.py`, `export.py` | Exact mathematics. |
| Joint objective and per-binding path routing | crisp | `tune/objective.py` + shared native trainer | Autodiff and loss normalization. |
| Family gates and amplitude convention | crisp | `tune/gates.py` | Exact projection and norms. |
| Propose families from effective-delta similarity | fn over crisp evidence | `proposeFamilies.nl` reading `tune/similarity.py` output | Similarity is exact; naming and grouping candidates by meaning is a judgment. Membership is published crisply as a version. |
| Propose placement removals, subtree roots, rank changes | fn | `proposePlacement.nl` | Choosing what to try next from ablation evidence is policy. Execution, refit and paired measurement are crisp. |
| Accept a placement candidate | crisp | quality-envelope rule in the manifest | The rule (lowest measured cost within the declared success envelope) is exact once measured. |
| Rank/penalty sweep scheduling | host | shared recipe scheduler | Operational. |
| Distillation into each base | crisp + existing recipe | shared production SFT recipe admission | Uses the existing admission/exposure mechanism. |

### Serving

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Native reference per-call actor binding | crisp | existing `run_native` with ordered stack | Already exact. |
| vLLM / llama.cpp fork / wllama loading, routing and batching | service + crisp | backend adapters under `ts-host/src/model/` and `vendor/` | Outside-world integration; parity is verified crisply. |
| Capability matrix | crisp | generated from parity tests | Facts, not claims. |

### Lifecycle

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Build keys and reverse dependencies | crisp | `ts-host/src/adaptation/weights/keys.ts` | Exact hashing; semantic digests exclude generated adapters to avoid cycles. |
| Invalidation from an edit | crisp | invalidation matrix in `keys.ts` | Rules over the dependency graph. |
| Resumable jobs and watch coalescing | host | `tune` command job directory | Plumbing with compare-before-publish. |
| Report | crisp facts + fn summary | `tune --report` + `summarizeTuneReport.nl` | Numbers are exact; the author-facing summary is natural language. |

## Open questions

1. Whether `routeTeacher` needs the NL implementation before measured calibration data exists. Until then the
   crisp frontier rule runs and the NL variant is measured live (about 48 samples) before adoption.
2. Inline site IDs today are `site:<label>` or `exact:<definitionId>`; the plan requires structural matching on
   edits with migration maps. Until that exists, an unlabeled inline site whose definition ID changes gets a new
   binding (never another site's adapter).
