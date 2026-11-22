# WS8: Neuralese, memetic learning and self-improving programs inside semantic FP

**Parent:** [Campaign index](README.md). **New in v2** (2026-11-23). v1 gave this one row in WS5.

AGENTS.md calls Neuralese compaction "the intended runtime design", memetic learning and soft skills "a core user
capability", and the backbone running a natlang improvement program "the long-term meta-learning direction". This
workstream makes them part of semantic FP: a program written with `@natlang/std` combinators gets compressed state,
learned operators and self-improvement through the types and combinators it already uses. The author adds no separate
subsystem.

## 1. What exists (verified 2026-11-23)

| Capability | Code | State |
|---|---|---|
| `Neuralese<T, D>` soft values: typed, opaque, content-addressed native state; `read`, `view`, `convert`; `natlang:neuralese` combinators (`map`, `zip`, `ap`, `combine`, `empty`, `split`, `read`, `ask`) with laws | `spec/ext/neuralese.md` (0.5-draft), `spec/neuralese.d.ts`, `ts-host/src/neuralese/{combinators,view,text-init}.ts`, `compiler/neuralese.ts`, `compiler/rewrites.ts` | Built and specified. Used by 0 `.nl` files and 3 TS files in the repo (audit 00 §2). The law rewrites are gated off (`DEFAULT_REWRITE_GATE`). |
| Generic soft/hard results `R: T \| Neuralese<T>` | `runtime/loader.ts` `generic:`, `compiler/neuralese.ts` `checkGenericResults` | The only generic form the language has. WS1 1B.3 generalises it. |
| `natlang:learning`: `grad`, `valueAndGrad`, `stopGradient`, objectives, immutable optimisers, `save`, second-order (`{ order: 2 }`) | `ts-host/src/neuralese/learning.ts`, `spec/ext/learning.md` | Built against the Neuralese server's gradient replay. Conformance cases `grad-context-item`, `grad-stop-gradient`, `grad-nested-first-order`, `grad-log-likelihood`, `graph-replay-fixed-choices` are `status: pending` (`conformance/neuralese/cases/termination-learning-graph.yaml`). |
| Runtime compaction hooks (`compaction` pluggable, `serverCompactor`, `POST /v1/neuralese/operator`) | `ts-host/src/model/native-operator.ts`, `native/operator-frames.ts`, `native/agent.ts` | Built; setting default `crisp` (text cut). FALLOW C7. |
| Learned compaction operators (compact-value / compact-params as `view`) | `operator_rows.py`, `operator_ablation.py`, `native.capture`, `grammarCapture`, variant `learned-compaction` | Rows train as replay corpus `operators` (weight 0.05) in GRPO; FALLOW C8 next action is the held/ablation gate before flipping C7. |
| Memetic improver: backbone studies a program's support invocations and commits guidance/changes, scored on held query worlds | `applications/memetic-improver/`, `rl/memetic_workspace.py`, `grpo.py --memetic-tasks --memetic-share` | Running in GRPO at share 0.02 (FALLOW P5; the gap is telemetry, owner 2026-11-18). Not in expert-iteration collection. |
| Program improver (diagnose / hypothesize / edit / experiment over `Folder` branches, Pareto selection) | `applications/program-improver/`; driven by `natlang improve CASE.json --out DIR` (`ts-host/src/improvement/`: `program.ts`, `adoption.ts` adopt/rollback, `transformations.ts`, pluggable search policies `findOpportunity`/`chooseParent`/`selectIncumbent`/`shouldStop`, default crisp) | Product CLI exists, but a user must author a `CASE.json`; it does not read a program's own call store or `examples.json`, and no front-door surface mentions it. |
| Program tuning and LoRA bank meta-learning | `scripts/neuralese_data/program_tune.py`, `T/tune/learned/meta_bank.py`, `applications/tuner/` | FALLOW T1/T2: one fit done, eight queued, bank never run on LFM3. |
| Soft prompt states for operators and per-lambda states (text-initialised in the real call) | MODEL_FACING_CHANGES 2026-11-20 | In use for operators. |

So the parts are built. Nothing a consumer writes reaches them: no `@natlang/std` combinator takes or returns
`Neuralese<T>`, no front-door surface teaches them, and `natlang improve` needs a hand-authored `CASE.json` instead of the program's own invocations.

## 2. Design: one family of primed soft values inside the combinators

Owner 2026-11-20: Neuralese spans, soft instructions and skills, memetic guidance and operators are one family of
text-primed KV states and one mechanism. Semantic FP adds the typing and the flow:

1. **Combinators are polymorphic over soft state.** Every `@natlang/std` combinator whose state or intermediate value is
   carried between model calls declares `S | Neuralese<S>` (WS1 1B.3 generics generalise today's `R: T | Neuralese<T>`).
   `semanticReduce(items, init, step, { carry: 'neuralese' })` carries `Neuralese<S>` between steps and `read`s once at
   the end. `semanticMap` over a large document runs `view` once per item and passes the view. The author chooses with
   one option, and the crisp text form stays the default whenever no Neuralese server is bound
   (`neuralese-unsupported-backend` otherwise, as the spec says).
2. **Learned operators are semantic functions.** A compaction operator is a `.nl` function whose result type is
   `Neuralese<T>` and whose body is its natural-language instruction (text-initialised in the real call, AGENTS.md).
   `compactValue.nl` and `compactParams.nl` live in `@natlang/std/neuralese/` next to `view`. The runtime's
   `compaction` pluggable calls them. The training side already trains their rows (C8). This removes the special case:
   an operator is checked, traced, specialised and tuned like every other function.
3. **Guidance learned from a program's own invocations is a typed artifact of that program.** `natlang improve DIR`
   (a new form of the existing command; the policy stays `applications/program-improver`, extended with
   `applications/memetic-improver`'s guidance changes) reads
   the program's call store and `examples.json`, splits invocations into support / development / confirmation, runs the
   improver on support, and proposes a change set (program guidance state, function guidance, instruction edits,
   crisp cases from `natlang specialize`) as a `Folder` proposal. Exact checks (`natlang check`, `natlang test` on the
   development split, confirmation split pass rate not lower) decide acceptance. This is the same propose/verify/accept
   transaction WS1 1B.1 uses for `edit_code`: one implementation.
4. **`grad` through the combinators.** With a Neuralese server bound, `valueAndGrad(loss, soft)` over a pipeline built
   from `@natlang/std` combinators reaches the soft state carried through them (the replay already handles producer and
   consumer turns). The pending conformance cases above are the acceptance test: they go to `implemented` before any
   front-door surface mentions `grad`.
5. **The backbone runs the improvement program.** The long-term direction (AGENTS.md) becomes a concrete training
   family: improvement episodes on held-separated programs, scored by the confirmation-split gain, enter expert
   iteration as tasks (WS5 §3, family `improve-program`). The memetic GRPO share keeps running. Expert-iteration
   collection gains the same tasks, so the improver's own successes become SFT targets.

## 3. Work items

Sizes are agent-days (one agent, one working session of ~6-8 h = 1). Every item ships with an opt-out and targeted
tests, and lands on main.

| ID | Item | Depends | Size | Done when |
|---|---|---|---|---|
| N1 | Close the five pending `natlang:learning` conformance cases (or fix the code until they pass) on the Python native engine; record per-engine status in `docs/ENGINE_PARITY.md`. Only the Python native engine serves `/v1/neuralese/grad` (`serve/http.py:352`), and the browser throws `ModuleUnavailableError` for `natlang:learning` (`ts-host/src/browser/index.ts:42`). The row names the current blocker per engine: SGLang has no gradient replay endpoint, and llama-server and wllama have no backward pass. | none | 2 | the five cases are `implemented`, run in `check-main.sh`'s conformance step |
| N2 | `@natlang/std/neuralese`: `compactValue.nl`, `compactParams.nl` as `.nl` functions with `Neuralese<T>` results; runtime `compaction` pluggable's `nl` side calls them; crisp side unchanged and default | WS3 S1 (std package exists) | 2 | `compaction: 'nl'` runs them on the Python native engine and SGLang; the operator rows' prompts render byte-identical to the new functions' calls (one implementation) |
| N3 | Soft-state carry option on `semanticReduce`, `semanticMap` (view per item), `fixedPoint` | WS1 1B.3, WS3 S1 | 2 | mock-driver tests, and a Python-native smoke where `read` of the carried state passes the task's exact check |
| N4 | `natlang improve DIR`: the existing command (`ts-host/src/cli/improvement.ts`, `improvement/program.ts`) gains a program-directory form that builds the `ImprovementCase` set from `examples.json` plus the call store (support/dev/confirmation split by seed), adds memetic guidance changes as a transformation, and adopts through `improvement/adoption.ts` (already adopt/rollback); the same proposal/verify/accept transaction as WS1 1B.1 | WS1 1B.1 | 3 | on `examples/triage` and one flagship, an improve run proposes a change set, the exact acceptance passes or rejects with a named reason, rollback leaves the program byte-identical |
| N5 | `improve-program` task family for expert iteration: improvement episodes over held-separated programs (migrated corpus programs from WS7 are the supply), oracle = confirmation-split gain over the unimproved program | WS5 T1, WS7 M3 | 3 | `data_registry.py promote ... --role grpo-pool`; `held_eval verify` clean; the first round's collection contains the family |
| N6 | Front door: one playground example and one skill section (`references/neuralese.md` in natlang-authoring) showing `semanticReduce(..., { carry: 'neuralese' })` and `natlang improve` | N3, N4 | 1 | the example runs in the page (wllama consumes native states, ENGINE_PARITY "Native state") |

## 4. What flips C7 (compaction on by default)

FALLOW C7's gate stands: a trained operator must beat shuffled / zeroed / text-cut on held worlds. held-v4 (WS5 H1)
carries a Neuralese stratum so that gate is measured on the series we train against rather than on a side set. When it
passes, `compaction: 'nl'` becomes the default with `crisp` as the opt-out.
