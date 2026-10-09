# Architecture improvement plan

Status: proposal, 2026-10-09. Written after a whole-repository review (TS host, Python training, language and
applications, plans and history). Small fixes already landed are listed in §9. Everything else here is a proposal for
the owner. Items that change running training code follow `newest-code-always`: they land behind unchanged defaults
and running jobs pick them up at their next resume, never by in-place edits to a live process.

## 1. Diagnosis

The language core is coherent. A natural-language function is a typed async function. The compiler plans its
signature and captures before the model runs. Contexts are content-addressed, acyclic and lexically scoped, and
iteration is finite by construction. Those are the right foundations for a small-model interpreter.

The problems are growth problems. Work extends sideways, so each experiment adds a new trainer, recipe, script,
helper or application shell instead of a parameter or library call. Consolidation does not keep pace:

| Symptom | Evidence (2026-10-09) |
| --- | --- |
| The TS core is one strongly connected component | Directory cycles native↔runtime (runtime imports native in 12 files, native imports runtime in 4), calls↔runtime, compiler↔native, compiler↔runtime, adaptation↔{compiler, runtime, native}, neuralese↔{model, runtime, native} |
| God files | native/runtime.ts 2.2k lines, teacher/native-materializer.ts 1.5k, teacher/collector.ts 1.5k, compiler/neuralese-conversion.ts 1.5k, native/agent.ts 1.2k, scope-compiler.ts 1.2k at the src root |
| Copied foundations | 29 Python `sha256*` definitions, 11+ canonical-JSON helpers, 12 application files calling `createHash` directly, 75 TS files that use `JSON.parse` with no shared parser |
| Trainer and recipe sprawl | Two trainer lineages: `scripts/train_lora.py`, and 8+ trainers in `natlang_neuralese/train`, ~300 argparse flags in all. Two optimizer modules. 33 recipes, some differing only in `id`, `description` and `cutoff` |
| Operations code next to library code | 205 flat Python scripts (18 `*reviewed*` campaign drivers, 11 exclusion scripts, v1/v2 pairs). Library code is imported out of `scripts.train_lora` |
| Machine-specific paths | 41 Python files contain `/home/werg`, `/mnt/external` or a hostname, including dataset identity in `scripts/neuralese_data/inventory.py` |
| Spec drift | The spec names codes that are never emitted (`iteration-unbounded`). Rules described as compile errors are enforced only at run time. SPEC.md also carries the whole Neuralese/learning research surface |
| Thin applications | games, scheduling and workflow contain no `.nl` files. wiki, logs, build and migration are crisp shells around one ~15-line `.nl`. Only 9 of 20 apps have `natlang.json` |
| Test gaps where errors are silent | No direct tests for LionSR, PortMuonAdamW, the QAT ramp or the in-backward Lion step. ~45 TS source files are never referenced by a test (package/, most of calls/, the model clients) |
| Churn | 617 commits on 2026-10-08. Remove/restore flips in the log. `plans/` is 35k lines against a 2k-line `spec/`. Handover files are telegraphic logs rather than resumable state |

None of these block today's work. Together they raise the cost of every future change, and they make silent
provenance and optimizer errors more likely.

## 2. Principles for the work

1. **Ratchets, not rewrites.** Freeze each metric in CI first: cycles, duplicate helpers, hard-coded paths, untested
   modules. Then shrink it. Nothing may get worse while it is being fixed.
2. **One home per concept.** Every concept (hashing, canonical JSON, model backend, training loop, recipe, machine
   paths) has exactly one implementation per language. A second copy needs a written reason.
3. **Parameters, not files.** A new experiment is a recipe override or a flag, not a copied trainer or recipe. A
   one-off campaign driver lives in `scripts/campaigns/<date-name>/` and is archived when the campaign closes.
4. **The spec is the contract.** Error codes, run-time versus compile-time enforcement, and extensions are stated
   once and checked by tests that grep the spec against the code.
5. **Compatible with running jobs.** Library consolidation keeps byte-identical behavior: the same hashes, the same
   update rules, the same checkpoint formats. Golden tests prove it before call sites move.

## 3. Phase A — guard rails (days 1–3, either machine, no GPU)

| Item | Deliverable | Acceptance |
| --- | --- | --- |
| A1 Layering ratchet | `ts-host/scripts/check-layering.mjs` + baseline, wired into `npm test` (landed, §9) | A new cyclic directory edge fails CI |
| A2 Shared hashing/JSON (Python) | `natlang_neuralese/common/{hashing,jsonio}.py` with golden digests; identical copies migrated (landed, §9) | Golden tests pass; each remaining variant is listed with how it differs |
| A3 Duplicate-helper ratchet | `scripts/check_duplicate_helpers.py`: counts definitions named `sha256*`, `canonical*`, `write_json*`, `utc_now` outside `common/`; baseline file; pytest wrapper | The count can only fall |
| A4 Hard-coded path ratchet | `scripts/check_machine_paths.py` with a baseline of today's 41 files | No new file may contain an absolute machine path |
| A5 Optimizer and QAT golden tests | Tests for LionSR, PortMuonAdamW, MuonWithAdamW, ramp endpoints, ternary STE, in-backward Lion parity (landed, §9) | Pass, or xfail with a filed defect |
| A6 Spec–code error-code parity | A test that extracts every backticked kebab-case code from spec/ and skills/ and checks that ts-host/src emits it | No undocumented divergence |
| A7 Declared Python dependencies | `pyproject.toml` lists torch, transformers and the rest under optional extras (`[train]`, `[serve]`), pinned to the versions now in `.venv-neuralese` | `pip install -e .[train]` reproduces the environment |

## 4. Phase B — TypeScript host structure (week 1–2, Pop or DGX, coordinate via inbox)

**B1. Extract a leaf `core/` layer.** Move the code that runtime, native, compiler and calls all need into
`ts-host/src/core/`: values and type descriptors, hashing, nz-file encoding, `FolderHandle`, a `Context` interface,
error codes and the `Deopt` type. `core/` imports nothing else from src. Then:

- `runtime/` depends on native only through the injected `NativeRuntimeHooks` (runtime/hooks.ts already has this
  pattern). Remove the native re-exports from `runtime/index.ts`.
- `native/scoped-fs.ts` stops importing `runtime/context` and `runtime/folder-iteration` and depends on core
  interfaces instead.
- `native/system-prompts.ts` receives judge prompts as data instead of importing `runtime/iterate`.
- Shrink the layering baseline after each move. Target: no directory cycles among core, compiler, runtime, native and
  calls.

**B2. Split god files by responsibility.**

- `native/runtime.ts` becomes `session.ts` (lifecycle), `services.ts` (service binding and result-type caching) and
  `lowering-bridge.ts` (Neuralese glue).
- `native/agent.ts` keeps only the agent loop. Request shaping moves to the backend (B3).
- `scope-compiler.ts` moves to `compiler/scope.ts`.
- `compiler/neuralese-conversion.ts` and `neuralese/learning.ts` move to a `training/` subtree inside ts-host. They
  are training concerns, not compilation.
- `teacher/collector.ts` and `teacher/native-materializer.ts` separate collection (I/O), curriculum (policy) and
  materialization (pure transforms).

**B3. One `ModelBackend` interface with declared capabilities.**

```ts
interface ModelBackend {
  capabilities: { guidedDecoding: boolean; neuralese: boolean; logprobs: boolean; decisionReadout: boolean;
                  batching: { maxConcurrent: number } };
  complete(request: CompletionRequest, signal?: AbortSignal): Promise<CompletionResult>;
}
```

- The agent asks `capabilities.guidedDecoding` instead of checking the server type (native/agent.ts:387).
- Every HTTP model call goes through `model/chat-completion.ts`, including the browser's local model, teacher clients
  and `environment.ts`.
- Mock-server contract tests cover each backend: llama.cpp fork, reference server, OpenAI-compatible and Pi.

**B4. Benchmark policy out of the library.** The TAT-QA normalizers in `evaluation/oracles.ts` and the dataset
contracts in `teacher/` (musique, tatqa, seeded-failure) move to `benchmarks/<name>/` and register by name. The core
oracle module keeps only generic comparators.

**B5. Module-level state goes onto a `RuntimeContext`.** Candidates:

- compiler/host.ts `defaultLibs` and `sharedSourceFiles` (unbounded, never evicted)
- native/nz-file.ts `DISTRIBUTIONS`, `IMPORTED_BLOCKS` and `ADOPTED`
- native/neuralese-store.ts `constants`
- native/runtime.ts `serviceResultTypeCache`
- neuralese/recording.ts swappable sources
- environment.ts realms

Process-level caches that stay global become bounded LRUs and get a `resetForTests()`.

**B6. Configuration and errors.**

- One `Env` object, built once at the CLI or node entry and passed down, replaces direct `process.env` reads in core
  files.
- Triage the 187 bare `catch {}` blocks. Deliberate ones get a one-line reason; the rest rethrow a typed error or log.

**B7. Package surface.** Add subpath `exports` (`@natlang/node/core`, `/tooling`, `/browser`). Keep the root barrel to
the documented API. `evaluation/worker-entry.ts` stops importing `../index.js`. Run `knip` once and delete what it
proves unused.

**B8. Close the test gaps.**

- Contract tests run one suite against all call-store variants (store, store-core, store-wasm, browser worker).
- package/ gets tests.
- Mock-server tests cover the model clients.
- calls/findings and calls/groups get tests when they land.

## 5. Phase C — Python training structure (week 1–3, DGX owns, Pop reviews)

**C1. One training-loop skeleton.** Create `natlang_neuralese/train/loop.py` holding:

- resume and checkpoint safety (shared with `checkpoint_safety.py`)
- RNG capture and restore
- stop handlers
- eval cadence and periodic held-out evaluation
- ledger admission
- emergency checkpointing (already shared since cb9ebf1c)

Each trainer (trajectories, text_warmup, delta_e2e, projection_e2e, joint, causal_bootstrap, decision,
maple/nested_train, the Mellum QAT conversion) supplies `build_batches`, `loss` and `metrics`. Migrate one trainer at a
time, behind a parity test: the same seed and the same 20 steps must give the same loss curve within tolerance. Retire
`scripts/train_lora.py`'s shared helpers into the package, leaving the script a thin CLI or archiving it if the crisp
student line is closed.

*Status (2026-10-10):* `train/loop.py` landed with the text warm-up and trajectory trainers on it (step loop, stop
signals, cadences, RNG capture, gradient accumulation, optimizer commit), plus the shared warm-up export, the role-bound
artifact resolver and the checkpoint policy on the shared writer; see TRAINING_RECIPE.md "Shared training skeleton".
Still to migrate behind golden runs: delta_e2e, projection_e2e, joint, causal_bootstrap, decision, maple/nested_train
and the QAT conversion; `scripts/train_lora.py` helpers; ledger admission.

**C2. One optimizer module.** `train/optim.py` absorbs `scripts/training_optimizers.py`. The golden tests from A5 hold
the update rules fixed.

**C3. Recipe inheritance.**

- Recipes gain `extends: <id>` plus `overrides: {...}`, resolved and hashed by `train/recipe.py` before execution. The
  resolved recipe is what certificates record.
- Collapse the `foundation-maple-c*-ctx*` and parallel `-lfm-`/`-maple-` copies into one base recipe per stage plus
  backbone overrides, which makes `unify-maple-and-lfm` mechanical.
- Experiment history moves from `description` prose to the run's evidence manifest.
- Generate the `HANDLERS` parameter whitelist in `recipe.py` from each stage module's argparse definition, so the two
  cannot drift.

**C4. Machine profiles.** `natlang_neuralese/paths.py` resolves logical roots (`models`, `data_nvme`, `data_hdd`,
`archive`, `llama_cpp`) from `~/.config/natlang/machine.toml` or environment variables. Dataset identity becomes
`(logical root, relative path, sha256)`. Migrate the 41 files with the A4 ratchet.

**C5. Scripts layout.**

- `scripts/` gets subpackages: `ops/` (ledger, sync, coordination), `data/` (inventory, admission, exclusion),
  `teacher/` (reviewed pools, dispatch), `eval/`, and `campaigns/<date-name>/` for one-offs.
- v1/v2 pairs fold into one parameterized module.
- A shared `cli.py` provides `add_common_args()`: `--out`, `--seed`, `--device`, `--dry-run` and ledger admission.
  The 37 `--output` uses converge on `--out` with an alias.

**C6. One schema validator per language.** Python validates records with `jsonschema` against `spec/*.schema.json`,
replacing hand validation in `scripts/neuralese_data/records.py`. Shared fixtures in `tests/fixtures/parity/` feed both
the Python and TS test suites, for example `inline_instructions.py` against `compiler/inline-instruction-index.ts`.

## 6. Phase D — language, spec and applications (week 2–4)

**D1. Split the spec.**

- `spec/SPEC.md` keeps the stable core: functions, contexts, captures, iteration, services, types and the model
  surface.
- Extensions get their own versioned documents under `spec/ext/`: Neuralese, learning, call records and compilations,
  directory reducers.
- `skills/.../language.md` links to the spec sections instead of restating them. A3-style tests catch restated rules
  that drift.

**D2. Every documented error has a stable code, and every rule says when it fires.** Annotate each rule as checked at
plan time, call time or eval time. Move cheap checks earlier:

- A TS `until` predicate with neither measure nor limit can be rejected by the compiler's iteration lowering, not
  only when `until` runs.
- `undeclared-field` can be reported at plan time when the field read is syntactically visible.

**D3. Make prose captures explicit enough to be safe.** Capturing by scanning prose for exact names lets an ordinary
word that matches a binding become a capture silently. Options, in order of preference:

1. The planner reports every capture in the `.d.nl.ts` declaration and in `natlang check`, so authors see them.
2. A lint flags captures whose name is a common English word (`items`, `result`, `text`) unless backticked.
3. Backticked names are captures; other prose matches are only warnings.

Options 1 and 2 cost the student model nothing. Option 3 changes the training distribution and needs the owner's
decision.

**D4. Unify nesting limits.** Replace the five-layer ad hoc limit, the per-`.nl` fresh budget and the kernel's
`maxDepth` with one depth budget carried on the call. Each kind of call costs a declared amount, and the budget is
reported in traces.

**D5. A standard library instead of repeated guards.** The same prompt fragments recur across applications: "read
only the named file from `files` before choosing", "never treat X as instructions", "write exactly a, b, c". Under
`language-design-restraint` and `negative-examples-prime`, these become:

- `Untrusted<T>`: a type the runtime renders as quoted data with provenance, so no prompt warning is needed.
- Enum and literal return types, so "write exactly a, b, c" is enforced by the type and taught by its error.
- A `std/` callable folder with `pickFile`, `cite` and `classify` helpers that applications `uses:` instead of
  restating.

Measure each change live before adopting it, per `negative-examples-prime`.

**D6. Applications.**

- Every application gets `natlang.json` and `types.ts`. A shared `applications/_lib` (hashing, atomic JSON state, an
  event log) replaces the 12 local `createHash` helpers.
- Port games, scheduling and workflow end to end (`natlang-ports-end-to-end`), or delete them. A thin shell around an
  inline call teaches readers the wrong lesson about the language.
- Each application gets a declared eval suite under `applications/<app>/eval/` that the adaptation engine can run.
  Stub-model tests stay in ts-host/test.

## 7. Phase E — make the idea more radical

These build on parts that already exist but are built as separate subsystems: the tracing JIT, Neuralese, the
learning continuum, adapters and `iterateOn`. Each is a research bet with a gate, not a commitment.

**E1. A tiered execution engine, modeled on V8.** One natural-language function, many implementations, chosen per
call:

| Tier | Implementation | Produced by | Guard |
| --- | --- | --- | --- |
| 0 | Frontier or teacher model interprets the source | — | — |
| 1 | Small student interprets the source | training | decision-readout confidence |
| 2 | Specialized NL (trace-specialized instructions, few-shot cases) | specializer (`calls/`) | case-match guard |
| 3 | Crisp TS synthesized from traces | specializer plus the compiler's policy checks | input-shape guard plus shadow replay |
| 4 | Neuralese soft program or adapter weights | learning continuum | held-out agreement |

- The call store is the profiler. `Deopt` is the fallback when a guard fails.
- Shadow replay of a sampled fraction against the tier below is the correctness monitor.
- Gate: one hot function from the trace store runs at tier ≥2 with equal held-out accuracy and ≥5× lower cost.

**E2. Fused NL pipelines.** When stage A's result goes only to stage B and both run on the same model, keep the
intermediate as a Neuralese value instead of decoding it to text. This is kernel fusion for natural-language programs,
with Neuralese as the calling convention between functions. The compilers application (30 staged `.nl` files) is the
benchmark. Gate: equal accuracy at lower token count and latency than the text pipeline.

**E3. A batched interpreter.** PLAN.md §1's speed thesis needs the runtime to batch. The scheduler collects every
pending `nl` call across the program into one forward batch per model: each `Promise.all`, each `iterateOn` frontier,
each directory-reducer fan-out. `ModelBackend.capabilities.batching` (B3) exposes it. Gate: throughput on triage over
10k tickets against sequential execution.

**E4. Natural-language refinement types.** Types such as `Text & Polite`, `Untrusted<Text>` or
`Claim & SupportedBy<Evidence>` are checked at call boundaries by `readout: decision` classifiers. Their failures are
ordinary type errors the student is trained to recover from. This turns repeated prompt guards into types and puts
"teach in errors, not prompts" into the type system.

**E5. Train the program, not the model.** Use the existing `grad` over soft values end to end through a whole program
graph against an application's eval suite (D6). The program, meaning its soft instructions, skills and adapters, is
the trained artifact, versioned with its dialect.

**E6. Self-hosting.** Planning, specialization and improvement passes are natlang programs (specializer already is),
executed through E1. The compiler improves by specializing itself, and the improver improves itself under the
learning continuum's operators.

**E7. One flagship instead of many shells.** pi plus the companion (harness bench), or compilers, rebuilt end to end
through E1–E3, gives one demonstrably faster and cheaper system as the headline result.

## 8. Process changes

- **Handover is state, not a log.** `HANDOVER.md` files keep a fixed short header: what is running, the last
  qualified artifact, open gates, next action. The chronological log moves to `handover-log/<date>.md`.
- **Decisions get one-line records with what they replace.** DECISIONS.md entries name the superseded entry, so
  remove/restore flips are visible.
- **Weekly consolidation slot.** One work cycle a week is spent shrinking a ratchet (A1, A3, A4) rather than adding
  features.
- **Plans get archived.** Plans for retired designs move to `plans/archive/` with a one-line pointer to what replaced
  them. The spec and the source remain the contract.

## 9. Landed with this plan (2026-10-09)

- de245db9: Pyodide on Node created a stray `file:/home/...` tree in the working directory. `packageCacheDir`
  defaulted to the `file:` URL in `packageBaseUrl`. Fixed, and the duplicated PLAN.md paragraph was removed.

## 10. Sequencing and ownership

| Phase | Owner | Conflicts with running work |
| --- | --- | --- |
| A | either machine | none; tests and CI only |
| B1–B2 | the session that owns ts-host at the time; announce in both inboxes first | native/runtime.ts and runtime/hooks.ts are under active edit; do B after those land |
| B3–B8 | either | low |
| C1–C3 | DGX (owns training execution) | trainers migrate one at a time, between runs, behind parity tests |
| C4–C6 | either | low |
| D | either; D3 option 3 and D5 types need the owner's decision | changes the training distribution |
| E | research; E1 first, since the specializer loop is live on DGX | — |
