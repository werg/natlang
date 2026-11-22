# WS7: Migrate the legacy prose corpora, or delete them

**Parent:** [Campaign index](README.md). **New in v2** (2026-11-23). It replaces v1's "quarantine" (old WS4 §2.6).

v1 proposed labelling `codebases/design_briefs/` and the 11 Astra apps as archive material and moving them aside. That
leaves 1,711 `.nl` files (83% of the repo's `.nl`) that teach the wrong style to anyone who greps, and it wastes their
one real asset: each comes with exact behavioural checks. v2 migrates every program worth keeping into idiomatic
semantic FP, checks each migration against the program's own cases, deletes the rest (AGENTS.md: lean codebase; git
history is the archive), and turns every verified before/after pair into training data.

## 1. What there is to migrate (verified 2026-11-23)

| Corpus | Programs | `.nl` | Behavioural checks that exist | Training references |
|---|---:|---:|---|---|
| `codebases/design_briefs/` | 100 | 1,390 | `training/semantic_applications/design_briefs/<NNN>/cases.json`: 800 exact-oracle cases (8 per brief; `root`, `inputs`, `expected`, `oracle: "exact"`, `split_group: design-brief-app-NNN`). Tests inside 3 projects (039, 095, 098; `ts-host/test/design-brief-applications.test.mjs`), plus `codebase-check.test.mjs` (017) and `design-brief-effects-and-money.test.mjs`. Host: `applications/design-briefs-host/` | `scripts/rl_tasks/sources/natcode.py` (static Q&A over the tree), `design_brief_gates.py` (off-repo packet), `assemble_selection_pool.py:58`. SFT: `luna-design-briefs-six-20261026-v8-tranche000004-lfm3` (teacher trajectories, admitted), manifest `design-briefs-100-native-20261025-v1`, `sft-teacher-qwen-october-20261120` (role `grpo-sd-teacher`). Not in any `grpo-pool` entry. |
| Astra / Sol apps in `applications/` (cave-mission, dialogue-perspective-ledger, field-gloss-workbench, polyphonic-edition-workshop, protocol-meaning-debugger, saga-forensics, seed-lineage-laboratory, semantic-procedure-compiler, treaty-obligation-ledger, turn-window-adjudicator, witness-rewrite-studio) | 11 | 321 | `training/semantic_applications/{astra,sol}/<app>/project.json` (`natlang.semantic-application-source/1`: fixture world, request, oracle, reference, external verifier `verify-result.mjs`, typed call graph); control scripts (e.g. cave `survey-v5-controls/check-controls.mjs`); reviews `training/semantic_applications/reviews/ASTRA_*_REVIEW_SOL.json`; `ts-host/test/trusted-exact-verifier.test.mjs` (cave, saga, protocol, witness) | `sft-teacher-qwen-october-20261120` manifest pins all 11 (role `grpo-sd-teacher`). Not in `grpo-pool`. Untouched since 2026-10-14. |

Not in scope here: `applications/` programs that are already idiomatic or are product tools (notebook, helpdesk, nldb,
program-improver, memetic-improver, specializer, migration, workflow, wiki, games, pi). Those with known
prose-algorithm leaves (`applications/pi/generation/answer.nl`, `applications/logs/investigate.nl`,
`applications/compilers/`, `codebases/{spell_arena,dependency_plan,semantic_merge}`) are migrated through the same
pipeline in a second tranche (M6).

## 2. What "idiomatic" means here (the target style)

The rule is AGENTS.md's: **policy is natural language, mechanism is crisp**. The migration does not make programs
"mostly TypeScript". It moves each part to where it belongs:

- **Exact work becomes TypeScript in the callable folder.** Set unions, ID membership, arithmetic, record assembly,
  projections, sorting, counting and fixed sequencing become `.ts` helpers. Example:
  `cave-mission/.../advanceFrameUnit.nl`'s 779 words (11 numbered paragraphs, most of them exact set and copy
  operations).
- **Judgment stays natural language, and gets smaller and typed.** A leaf `.nl` with a finite result declares
  `readout: decision`. A judgment over each item of a collection becomes an inline `nl<T>` (or a named `.nl`) mapped
  with `Promise.all`. A property of the result becomes `Is<T, P>`. Outside text becomes `Untrusted<T>`.
- **A decision about what to do next stays policy.** Choosing which stage runs, when a loop has converged, or whether
  to repair or escalate is a natural-language function or an `iterateOn(...).until(nl\`...\`)` predicate (or a
  `pluggable` with both sides). It is not frozen into TypeScript `if` chains. Numbered steps remain correct for a
  multi-step *judgment* (AGENTS.md "spell algorithms out as numbered steps over named data for small models"). What
  goes is numbered steps that are *mechanism*.
- **Rubric scar tissue goes.** Example: the Astra rubric's "visible TypeScript contains declarations only" produced
  27 redundant `nl.with` re-asks. Redundant re-asks, `iterateOn` used as a `for` loop over a fixed array, and prose
  guards that a type expresses are removed.

## 3. The migrator is a natlang program

Deciding how to decompose a prose program is semantic judgment, so the migrator is a natlang program (AGENTS.md main
rule; owner point 3). It extends `applications/migration/`, the existing repository-migration program. That program
already has the right shape: it decides on an immutable `RepoSnapshot`, gives effects as data (`{path, old, new}`
patches), applies through one service call, and runs a measured `iterateOn` repair loop with a natural-language
`settled` predicate. A new entry `migrateProgram` adds:

| Part | Form | Why |
|---|---|---|
| Classify each `.nl` body's sentences as judgment / mechanism / scar tissue | `.nl`, `readout: decision` per sentence group (`"judgment" \| "mechanism" \| "guard-for-type" \| "redundant"`) | Semantic judgment over prose. |
| Propose the decomposition: which helpers become `.ts`, which leaves keep `.nl`, which get `readout: decision`, `Is<T,P>`, `Untrusted<T>`, inline `nl` | `.nl` returning a typed `MigrationPlan` | Design judgment. |
| Write each `.ts` helper and each new `.nl` | `.nl` per file, checked by `natlang check` and `tsc` inside `semanticRepair` / `iterateOn` | Generation with an exact check. |
| Behavioural equivalence | crisp service: run the original and the migrated program on the corpus's own cases (design briefs: `cases.json` through `design-briefs-host`; Astra/Sol: `project.json` fixture + `verify-result.mjs`), same model, same seeds, k samples | Exact. |
| Accept / iterate / give up | `.nl` `settled` predicate over the equivalence report, bounded by `withMeasure(remaining attempts)` | Policy. |
| Keep or delete the program | `.nl` `worthKeeping` (`readout: decision`) over the brief, its cases, and the equivalence result; crisp floor: a program with no passing original case is deleted, not migrated | Policy with a crisp floor. |

The migrator runs on **LFM3** wherever it can (it dogfoods the paradigm on the model it is meant for) and on a teacher
model (Luna, Haiku) where LFM3 fails. `fn.on({ model })` selects the model per call and the trace records it. The
teacher-run migrations are the first supply. Every LFM3-run success is also an own-success trajectory for expert
iteration.

## 4. Acceptance per program

A migrated program replaces the original on main when all of these hold:

1. `natlang check --json` is clean. Once WS3's idiom reviewer exists, it raises no `error`-level finding. Until then
   the crisp floor of WS3 §2.3 applies.
2. **Behaviour:** on the program's own cases, the migrated program's pass rate at k=4 samples on LFM3 (DGX
   collection actor or Colab) is ≥ the original's minus one case, and it passes every case the original passes at
   4/4. For design briefs that is the 8 `cases.json` rows. For Astra/Sol it is the fixture world's verifier plus the
   control scripts.
3. **Cost:** model calls per case and prompt+completion tokens per case are recorded for both versions. This is
   telemetry, not a gate. The expected direction is down, because fewer calls interpret plumbing.
4. Every reference to the old paths is updated in the same commit: `natcode.py`, `assemble_selection_pool.py`,
   `design-briefs-host`, tests, `training/semantic_applications/*/project.json` `source_root`/`program_files`,
   `ts-host/scripts/design-briefs/lib.mjs`, `scripts/check-main.sh:25`, `website/index.html:707`. Registry entries
   that pin old file hashes (`sft-teacher-qwen-october-20261120`, `design-briefs-100-native-20261025-v1`) keep their
   immutable manifests. Training data that was produced from old sources stays valid as data, and its manifests keep
   the hashes. The registry is not rewritten. New entries reference the new sources.

A program deleted by `worthKeeping` is removed with its `training/semantic_applications/` world only if no admitted
training dataset or held manifest reads that world (`data_registry.py` consumers). Otherwise the world stays and only
the source program goes.

## 5. The pairs are training data

Each verified migration yields:

- **A refactor task** (`natlang.program/2`, family `refactor-to-semantic-fp`): the original program plus its cases as
  the world. The goal is "make this program pass its cases with exact work in TypeScript". The oracle is the
  equivalence service (§3) plus the crisp idiom floor. This is an execution oracle, not an exact-string one, so it
  sidesteps the cue-word trap (00 §4.3).
- **An idiomatic program** for the RL pool: the migrated program's cases become ordinary tasks whose root is the
  migrated program. That is supply for the 0%-coverage constructs (inline `nl`, `readout: decision`, `Is<T,P>`, `.ts`
  helpers in callable folders, library discovery through `read_code`).
- **Teacher demonstrations**: the teacher-run migrator's trajectories and the migrated programs' teacher-run cases,
  with student-filled reasoning (the `filled=...,replaces=teacher` tier).

Held separation: every design brief is its own `split_group` (`design-brief-app-NNN`). Before admission, a fixed
fraction of briefs (20 of 100, chosen by seed `held-v4-design-briefs`) and 2 of the 11 Astra/Sol apps are reserved
for held-v4 (WS5 H1). Their pairs never enter training. `held_eval verify` runs against them.

## 6. Work items

| ID | Item | Depends | Size (agent-days) | Done when |
|---|---|---|---|---|
| M1 | Equivalence service: run any design brief or Astra/Sol program on its own cases against an endpoint at k samples; report per-case pass, calls, tokens. One CLI `scripts/semantic_fp/equivalence.py` (crisp; reuses `design-briefs-host` and `verify-result.mjs`) | none | 2 | the 100 briefs' and 11 apps' *original* baselines are measured on LFM3 and recorded (`ops/semantic-fp/migration/baseline.jsonl`), so `worthKeeping` has data |
| M2 | `applications/migration` `migrateProgram` entry (DECOMPOSITION.md first, then the program), with the §3 parts | M1 | 3 | migrates 3 design briefs and 1 Astra app end-to-end on a teacher model, passing §4 |
| M3 | Run M2 over all 100 briefs + 11 apps (teacher first, LFM3 retry per program); commit accepted migrations in batches of ≤ 10 programs; delete what `worthKeeping` rejects | M2 | 3 (wall time is collection-bound; parallel across endpoints) | every one of the 111 programs is migrated or deleted; `git grep` of the old style's markers (WS3 §2.3 floor) in `codebases/design_briefs` and the 11 apps returns 0 |
| M4 | Pair export: refactor tasks + idiomatic-program tasks + teacher demonstrations, registered and promoted (`data_registry.py promote ID --role grpo-pool`), held-v4 reserve excluded | M3 (streams as batches land), WS5 H1 (reserve chosen first) | 1 | registry entries exist; `held_eval verify` against held-v3 and held-v4 is clean; `measure_repetition.py` run on the batch |
| M5 | Update every consumer of the old paths (§4.4) | streams with M3 | 1 | `check-main.sh --quick` green after each batch |
| M6 | Second tranche: prose-algorithm leaves in otherwise-idiomatic apps (`pi/generation/answer.nl`, `logs/investigate.nl`, `compilers/`, `codebases/{spell_arena,dependency_plan,semantic_merge}`) through the same pipeline; their existing tests are the equivalence check | M2 | 2 | as M3 |

Expected outcome: most of the 1,711 files go, because most of their sentences are mechanism. A migrated brief keeps
3-8 short `.nl` judgments plus `.ts` helpers. The repo's `.nl` census (00 §2) is re-run after M3 and recorded here.
