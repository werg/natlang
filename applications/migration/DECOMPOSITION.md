# Migration: decomposition, part by part

A repository migration turns a request ("rename sum to add") into a reviewable candidate: understand the change, find
every place it touches, decide what to do at each, write exact edits, check the result by running the repository's
checks, and repair from the failures. Every part has a decision:

- **fn**: its own natural-language function with a typed contract.
- **inline**: one instruction inside its caller.
- **crisp**: a TypeScript helper in a callable folder, only for trivially deterministic plumbing.
- **pluggable**: one interface with a crisp and a natural-language implementation, chosen by a setting
  (`repository.implementation(point)`).
- **service**: the outside world, in `RepositoryMigration` (index.ts).

The executors are small models, so every function spells its algorithm out over named data in types.ts, and each
function is one task a small model can finish.

## State model

- **Decide on a snapshot, then apply a pure, bounded commit.** Every stage reads an immutable `RepoSnapshot` (a
  revision) and returns data: an `Intent`, `Site`s, `Usage`s, a `Plan`, `Patch`es, `Finding`s. Applying patches is one
  service call, `repository.apply(base, patches)`: it either yields a new revision or rejects with a message, and
  nothing else is written. The original checkout is never touched.
- **Effects are data.** A patch is `{path, old, new}`. A round's effects are the patch list, then `apply`, then
  `validate`. A rejected apply becomes `RepairState.rejected`, text the next round reads.
- **Derived values form a DAG.** Intent from the request and manifest. Sites from the intent and a revision. Usages
  from sites. The plan from usages. Edits from the plan. Findings from a validation. Repair patches from findings.
  Nothing is stored that a function of the state would give.
- **No while loops.** The repair loop is `round.iterateOn(state)` with `withMeasure(s => s.remaining)` and a stopping
  judgment that is natural language (`settled`), or crisp when the setting says so. The measure falls every round, so
  a judgment that never says "stop" still ends the loop.

## Policy

- **Natural language: every decision a migration makes.**
  - what the request means, and what to search for;
  - where a hit really is (the enclosing lines);
  - what kind of usage a site is, and whether to edit it;
  - the order of edits and the risks;
  - the edit itself and whether it is exact and in scope;
  - what a failing check says went wrong, and the repair;
  - when to stop.
- **Crisp: only plumbing.**
  - the crisp side of each pluggable point;
  - nothing else.
- **Service: the outside world.**
  - the in-memory revisions, search, exact line ranges and occurrence counts;
  - applying patches with exact old/new matching (an old snippet must occur once);
  - temp-directory materialization and running the checks, with bounded output and timeouts;
  - the report of changed files with digests.
- **Teach in errors, not prompts.** Instructions state the wanted result. `apply` rejects with the path and the
  reason (`patch context missing or ambiguous: caller.mjs`), `count` shows how many times a snippet occurs, and the
  exactness check returns the problem. These messages reach the next attempt as `problem` or `rejected`.

## Layout

`migrate.nl` is the driver; `migrate/` holds its own stages (understand, survey, plan, round, settled, summarize).
Units that two callers need live in folders without a driver of their own and are listed in `uses:`: `site/` (locate,
classify, edit with patch and exact), used by the driver, survey and repair; `loop/` (triage, repair), used by round.
Putting them under `migrate/` fails the load, because a function that `uses` a child of its own ancestor reaches itself
(`migrate/repair.nl: uses migrate/locate, which uses this function in turn`).

## Parts

| Part | Decision | Unit | Why |
|---|---|---|---|
| Restate the request: old and new thing, queries, invariants (from the request, manifest and check ids) | fn | `migrate/understand` | Planning from the manifest. The search queries come from the meaning, not from the caller. |
| Seed queries from the caller | inline | `migrate/understand` | Added to the queries. |
| Search a revision for a query | service | `repository.search` | Exact substring search. |
| Merge hits into sites: union of queries, one site per place | fn | `migrate/survey` | Spelled grouping by file and line. The evidence is the hits. |
| Locate one site: widen a hit to the enclosing lines that hold a whole statement, with exact text | fn, per site, in parallel | `site/locate` | Reading code around a hit is judgment. The exact text comes from `repository.lines`, never retyped. |
| Classify a site's usage (declaration, call, import, export, type position, property access, comment or doc, string literal, test, unrelated) and edit-or-leave | fn, per site, in parallel | `site/classify` | The pattern decides the patch. Separate from writing it. |
| Plan: which sites to edit, in what order, what to leave, which risks to watch | fn | `migrate/plan` | Coordinates the usages: declarations before callers, tests with what they test. |
| Write the patch for one site from its usage | fn, per site, in parallel | `site/edit/patch` | The edit, as exact old/new snippets copied from the site's text. |
| Check a patch is exact: old occurs once, it changes something, only what the usage calls for | pluggable | `site/edit/exact` | A hot path (every patch). Crisp: occurrence count and difference. Natural language: `exact/judge`, which also judges scope. |
| Retry a patch once with the problem | inline | `site/edit` | `problem?: string` goes to `patch`; a second failure is a failed site with the note. |
| Apply all patches to the base | service | `repository.apply` | Exact. Rejects ambiguous or missing context. |
| Run the checks on the candidate | service | `repository.validate` | Temp directory, no shell state, bounded output. |
| Triage failures: each failing check or rejected patch to findings with kind, path, line, evidence, repairable | fn | `loop/triage` | Reading check output is judgment. The kinds say what to do next. |
| Repair: findings to patches on top of the candidate (locate, classify, patch, exact for each finding) | fn | `loop/repair` | The same units as the first proposal, driven by evidence. |
| Repair round: triage, repair, apply (a rejection is data), validate, new state | fn | `migrate/round` | One step of the loop. |
| Stop: checks pass, budget spent, nothing repairable, or a candidate repeats | pluggable | `migrate/settled` | A hot path of every round. Natural language (default): `settled`, the loop's stopping judgment. Crisp: `settledCrisp`. |
| The loop | inline | `migrate.nl` | `round.iterateOn(state).withMeasure(...).until(stop)`; an exhausted measure keeps the last state. |
| Report: changed files with digests, check results | service | `repository.report` | Exact digests of the candidate against the base. |
| Closing words: what changed, what is left, what to review | fn | `migrate/summarize` | Prose for the reviewer. |
| Command line | crisp | `main.ts` | The outside world. |

## Pluggable hot paths

| Point | Setting | Crisp | Natural language | Default |
|---|---|---|---|---|
| `exact` | `repository.implementation('exact')` | `site/edit/exact.ts`: counts and differences | `site/edit/exact/judge.nl` | natural-language |
| `settled` | `repository.implementation('settled')` | `settledCrisp.ts` | `settled.nl` | natural-language |

## Refinements

Refinement types (`Is<T, "predicate">`, plans/REFINEMENT_TYPES.md). Decisions: (a) adopted with a crisp checker in
`refinements.ts` (no model call), (b) a natural-language judge (proposed only: awaiting live evaluation, not wired),
(c) left to the check that already enforces it exactly (the repository service, the exact patch check), (d) not
adopted: the value alone does not show the property. The refined result types are `Checked*` aliases in `types.ts`,
named in the `returns` of the stage that produces the value; crisp code keeps the plain types.

| Slot | Proposed type | Decision |
|---|---|---|
| `Patch.old` | `Is<string, "text that occurs exactly once in the patch's file at the base revision, copied from the file">` | (c) The exact check (`site/edit/exact`) counts occurrences in the file, which the value does not show. "Non-empty" is adopted with `Patch.new` below. |
| `Patch.path` | `Is<string, "a path of the file manifest">` | (c) `repository.apply` refuses unknown paths; the manifest is not in the value. |
| `Patch.new` | `Is<string, "differs from old, and keeps the surrounding code valid">` | (a) `CheckedPatches`, for `patch` and `repair`: each patch names a path, replaces non-empty old text and differs from it. That the code stays valid is the checks' (d). |
| `Intent.queries` | `Is<string[], "exact strings that occur in the code, most specific first, none empty">` | (a) Weakened (`CheckedIntent`): non-empty strings, each once. That they occur in the code needs the repository (d). |
| `Intent.invariants` | `Is<string[], "behaviors the checks observe, each stated as what stays true">` | (b) Proposed, awaiting live evaluation (not wired): one judge call per invariant, once per migration. |
| `Site.text` | `Is<string, "exactly lines from to of path at the revision">` | (d) The file is not in the value; `repository.lines` supplies the text. |
| `Site.from`, `Site.to` | `Is<number, "a line of the file, from at most to">` | (a) Weakened (`CheckedSite`): line numbers from 1, from at most to, hits a whole number. Being a line of the file needs the file (d). |
| `Usage.pattern` | `Is<string, "one of the listed patterns">` | (a) `CheckedUsage`, with a non-empty reason. |
| `Plan.edits` | `Is<string[], "ids of sites whose usage action is edit, without duplicates">` | (a) Weakened (`CheckedPlan`): each id once and none also left. That they are the sites whose action is edit needs the classified sites (d). |
| `Plan.leave` | `Is<{site: string, reason: string}[], "ids of sites whose usage action is leave, each with a reason">` | (a) Weakened (`CheckedPlan`): each entry gives a site id and a reason. |
| `Finding.kind` | `Is<string, "one of missed-site, wrong-edit, test-expectation, environment, unrelated">` | (a) `CheckedFinding`. |
| `Finding.repairable` | `Is<boolean, "true exactly for missed-site, wrong-edit and test-expectation">` | (a) `CheckedFinding`. |
| `Finding.evidence` | `Is<string, "copied from the check output or the rejection message">` | (d) The check output is not in the value. |
| `repair` return | `Is<Patch[], "each patch applies to the candidate it was written for and addresses a repairable finding">` | (c) `repository.apply` refuses a patch that does not apply; the candidate and findings are not in the value. The shape of the patches is `CheckedPatches` (a). |
| `Exactness.problem` | `Is<string, "empty when exact; otherwise names the path and says what to change">` | (a) Weakened (`CheckedExactness`): empty exactly when exact, a sentence otherwise. Naming the path needs the patches (d). |
| `Summary.summary` | `Is<string, "states whether the checks pass and how many sites changed, in at most three sentences">` | (b) Proposed, awaiting live evaluation (not wired): one judge call per migration. |
| `RepairState.remaining` | `Is<number, "a non-negative integer that falls every round">` | (a) Weakened (`CheckedRepairState`): a non-negative whole number. Falling every round needs the previous state, and the measure of `iterateOn` already enforces it. |

Counts: (a) 10, (b) 2, (c) 3, (d) 2.

The model-facing text added to signatures is the `Is<...>` predicate of `CheckedPatches`, `CheckedIntent`, `CheckedSite`,
`CheckedUsage`, `CheckedPlan`, `CheckedFinding`, `CheckedExactness` and `CheckedRepairState`. No instruction sentence
was changed.
