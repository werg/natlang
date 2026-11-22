# WS3: `@natlang/std`, and the campaign's own tools as natlang programs

**Parent:** [Campaign index](README.md). **v2** (2026-11-23).

v1 specified the idiom linter as a crisp TypeScript pass with keyword heuristics. It treated the migrator, task
authoring, qualification and escalation as tooling outside the paradigm. All of them decide things: whether a
paragraph is a prose algorithm, whether a task is genuinely semantic, whether to escalate. AGENTS.md's main rule says
a component that decides is a natlang program (policy natural language, mechanism crisp, `pluggable` when hot). So in
v2 each tool is a natlang program that runs on LFM3 and uses the standard library it ships with. Each is also a
flagship: if the paradigm cannot build its own tools, the plan's claims are wrong, and the trace will show where.

The pattern already exists and works. `applications/specializer` (promotion policy, `runtime/promotion.ts:28`
selects it with `pluggable`), `applications/migration`, `applications/program-improver` (search policies in
`optimization/policies.ts`, default crisp), `fusion/planner.ts`, `applications/ops-placement` and
`applications/ops-retention` are natlang programs that make product decisions. WS3 follows them.

## 1. Mechanism inventory

| Mechanism | Components | In scope |
|---|---|---|
| Combinator library | new package `ts-host/std/` (`@natlang/std`, re-exported as `@natlang/node/std` and `@natlang/browser/std`); `codebases/std/` (2 helpers: `any_true.ts`, `select_by_flags.ts`), folded in and removed | all |
| Selection between crisp and NL | `runtime/pluggable.ts` (`pluggable({crisp, nl}, setting, {default, name, serve, same})`, `pluggable_shadow` trace) | reused, unchanged |
| Diagnostics | `compiler/inline.ts:12-26` `NatlangDiagnostic` (`severity: 'error' \| 'warning'`), `compiler/project.ts:596,688,730`, `cli/commands.ts:978` (`hint`), `natlang check --json` | adds the crisp idiom floor and `--idiom` |
| Task authoring and qualification | `scripts/rl_tasks/{authored.py, authored_checks.py, authored_novelty.py, qualify_authored.py, blind_solve.py, authoring_rotation.py, measure_repetition.py, simulate_selection.py}`; `plans/neuralese/RL_TASK_AUTHORING_KIT.md` | T3/T4 put a natlang program in front of the Python mechanism. The Python stays as the crisp verifier and registry plumbing. |
| Engines | combinators issue ordinary calls and decisions; no new serving feature | none beyond WS2 R3 |

## 2. `@natlang/std` (S1-S3)

Each combinator is TypeScript mechanism (fan-out, bounded concurrency, exact grouping, ranking arithmetic, folder
transactions) around named `.nl` judgments that ship inside the package. A user can `read_code` them, `.in(context)`
them, tune them and specialise them like their own functions. Where a combinator carries state between model calls,
its signature is polymorphic over `S | Neuralese<S>` (WS1 L5, WS8 N3).

| Family | Combinators | Semantic parts (`.nl`, shipped) | Depends |
|---|---|---|---|
| Collections (S1) | `semanticFilter`, `semanticPartition`, `semanticMap` (bounded concurrency), `semanticReduce` (`linear` / `tree`; `carry: 'neuralese'`), `semanticMatch` (exact blocking + pair decisions), `semanticRank` (pairwise decisions + exact Copeland) | `meets.nl` (`boolean`, `readout: decision`), `classifyInto.nl` (generic finite), `relates.nl`, `beats.nl` | flat versions now; generic signatures after L5 |
| Convergence (S2) | `semanticRepair` (exact `check` feeds `fix` through `iterateOn`; returns `{ ok, value, attempts, history }`), `semanticMerge` (intent extraction, compatibility decision, explicit `unresolved[]`), `fixedPoint` | `fix` is the caller's; `intentOf.nl`, `compatible.nl` | none |
| Policy (S3) | `escalate` (WS2 R4), `compileRule` (prose rule → typed checked lambda, validated on examples before it is returned), `withContext` (sugar over `Context.ofCallable(fn).with(...)` and `.in`) | `shouldEscalate.nl`, `compileRule.nl` (returns `(x: I) => Promise<O>`) | `compileRule` needs L4 |
| Workspace (S2) | `proposeAndVerify`, `branchAndSelect` over `Folder` (`propose`/`accept`/`snapshot`/`branch`/`select` exist, `native/scoped-fs.ts:239-650`) | none (the verifier is the caller's) | none |

Discipline: Node and browser both work; every combinator nests its calls so `natlang traces`, `natlang profile` and
`natlang compilations` see per-item records; mock-driver unit tests plus one live LFM3 smoke per family
(`ts-host/test/std-*.test.mjs`).

## 3. The idiom reviewer (I1-I2), replacing v1's crisp linter

### 3.1 Split: crisp floor in `natlang check`, natural-language review in `natlang check --idiom`
`natlang check` promises "no model", and that promise stays. So:

**I1, crisp floor** (`ts-host/src/compiler/idiom.ts`, warnings in `natlang check`, 2 days). Only facts that are exact:
| Code | Exact condition |
|---|---|
| `idiom-decision-readout` | a leaf `.nl` (no companion folder, no `uses:`, no `Folder` parameter) whose `returns` is finite (`finiteValues` ≥ 2) and that declares no `readout:` |
| `idiom-fixed-cursor-iterate` | an `iterateOn` whose step's only state change is `index + 1` or `slice(1)` over an array that the step never otherwise reads or extends (AST pattern) |
| `idiom-code-in-prose` | a `.nl` body containing executable-code tokens (`JSON.parse(`, `.slice(`, `Math.`, `eval({`, `=>`) outside fenced examples |
| `idiom-interpolated-untrusted` | as L2 (`untrusted-instruction`); listed here so the floor is complete |

Each message is one sentence that teaches the fix (e.g. "`route.nl` returns one of 3 values: add `readout: decision`
to score them in one pass, or keep the tool loop if it must call helpers first."). There is no suppression syntax:
a warning that is wrong for a site is a defect in the check, and the check gets fixed.

**I2, natural-language reviewer** (`applications/idiom-reviewer/`, DECOMPOSITION.md first, 3 days, after I1 and
S1). `natlang check --idiom` runs it on the configured model (local LFM3 by default):
| Part | Form | Output |
|---|---|---|
| For each `.nl` body: which sentence groups are judgment, mechanism, a guard a type would express, or a redundant re-ask | `classifySentences.nl` (`readout: decision` per group) mapped with `semanticMap` | typed `SentenceClass[]` |
| For a function tree: is this a prose interpreter of plumbing (most mechanism, no `.ts` helpers)? | `isProseInterpreter.nl` over the class counts and the tree listing (`readout: decision`) | `boolean` + probability |
| Does a helper match literal cue words instead of judging meaning? | `matchesCueWords.nl` (`readout: decision`) | `boolean` |
| Which external text is typed `string` instead of `Untrusted<string>`? | `externalTextParams.nl` over the signature and how callers populate it | parameter names |
| One-sentence fix per finding | `fixHint.nl` | `Is<string, "one sentence, names the file and the construct to use">` |
| Severity | crisp: a finding with p ≥ 0.8 is an `error` under `--idiom=strict`, otherwise a `warning` | |

`classifySentences.nl` is the same function the migrator uses (WS7 §3): one implementation. Its calls in the
migration and in check runs are training data for it.

- Acceptance: on `applications/cave-mission`, `applications/pi/generation/answer.nl` and the old
  `playground/examples/natural.mjs` texts it reports prose-algorithm findings. On `examples/triage`, `examples/review`,
  `applications/notebook`, `applications/helpdesk` and every migrated program it reports none at `strict`.
  Disagreements between LFM3 and a teacher run of the reviewer are recorded (`fn.on({ model })` shadow) and feed its
  tuning (`natlang tune` on the reviewer).

## 4. Task authoring and qualification as natlang programs (T3-T4)

The cue-word trap (00 §4.3) came from a crisp pipeline: Haiku authors, a literal Sonnet blind-solve, and exact match.
In v2 the decisions in that pipeline become a natlang program. The exact parts stay crisp.

**T3, `applications/task-author/`** (DECOMPOSITION.md first; 3 days; after S1). Stages:
1. `pickSeed.nl`: choose a seed (dataset row, migrated program, flagship fixture) for an archetype the pool lacks.
   It reads `measure_repetition.py` output and construct shares as crisp facts.
2. `designWorld.nl`: write the world, the program skeleton and the oracle kind (`verifier`, `metamorphic`,
   `proper_score`, `exact` where the answer is canonical; WS5 §2).
3. `writeVerifier.nl` + crisp run: the verifier is TypeScript, checked by executing a reference solution (must pass)
   and ≥ 2 negative solutions (must fail). This is exact mechanism, `qualify_authored.py`'s replay extended.
4. `isGenuinelySemantic.nl` (`readout: decision`): would a keyword rule pass this task? Its crisp companion actually
   runs a keyword baseline (strongest cue-word solution the author can write). If the baseline passes the verifier,
   the task is rejected.
5. `paraphraseTwin.nl` for metamorphic tasks; crisp lexical-overlap check (content-lemma Jaccard < 0.25).

It runs on LFM3 where it can, and on Luna/Haiku via `fn.on({ model })` where LFM3 fails. Its admitted outputs go
through the unchanged registry path (`data_registry.py` register, `promote ID --role grpo-pool`,
`held_eval verify`). Its own successful calls are training data.

**T4, qualification as a natlang program** (2 days; after T3). For non-exact oracles, `qualify_authored.py` adds a
natlang stage that reviews whether the oracle can be gamed: `oracleGameable.nl` plus crisp red-team replays. It
replaces the literal Sonnet blind-solve for those oracles. Blind-solve stays for `exact` tasks, where it is the right
check.

## 5. Acceptance for WS3
- `@natlang/std` S1-S3 pass their suites on Node and browser. Each family's live LFM3 smoke passes its exact check.
- `natlang check` emits the I1 floor with one-sentence hints. `natlang check --idiom` runs the reviewer on LFM3.
- T3 has produced one admitted batch whose keyword baselines all fail their verifiers.
