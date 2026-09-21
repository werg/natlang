# S01 — Programs that invent and retain abstractions

Status: proposed. Part of [the semantic software plan](../SEMANTIC_SOFTWARE.md).

## Product and semantic ambition

An application should notice that a sequence of reasoning and computation is a
reusable method, turn it into source code, exercise it and make it available in
later work. The saved artifact is an executable natlang function with crisp
helpers where useful, rather than a prose memory saying how to do the task.

First example: Fieldnotes is asked to compare two measurements fairly. It
discovers a need for matching cohorts, develops a comparison function, and
later adapts that function to a differently shaped dataset. No `match_cohorts`
action is added to the studio's fixed JavaScript catalogue.

## User experience

- Normal work produces a “Methods developed here” collection alongside cells.
- Each method shows its purpose, signature, assumptions, examples, source,
  observed failures and which results use it.
- Users can inspect, edit, rename, fork, retire or invoke it like authored code.
- A later task can reuse the method automatically when its requirements hold.
  If they do not, natlang specializes it or develops a new method and explains
  the relevant difference.
- Local reversible candidate development need not require approval of every
  function. Publishing to another workspace or changing authority follows the
  user's chosen application policy, not a new language-level permission system.

## Program decomposition

Proposed ordinary functions:

1. `recognize_method.nl(work, examples) -> CandidatePurpose`: decide whether
   useful structure exists and what varies between examples.
2. `design_method.nl(purpose, available_sources) -> SourceProposal`: choose
   signature, decomposition, crisp engines and assumptions.
3. `exercise_method.nl(candidate, cases) -> Assessment`: run individual cases,
   inspect failures, develop additional counterexamples and revise source.
4. `find_method.nl(task, candidates) -> ReuseDecision`: judge applicability from
   source, contracts and evidence. Crisp text/SQL search retrieves candidates.
5. `adapt_method.nl(method, mismatch) -> SourceProposal`: preserve a versioned
   relationship to the original without overwriting it.
6. `integrate_method.nl(assessment, workspace) -> WorkspaceChange`: connect the
   method to cells and generated interactions; record remaining limitations.

These functions need not form one fixed lifecycle for every task. Natlang may
interleave design, execution and revision. Long iterative work remains allowed.

## Records and host facilities

`MethodRevision` contains identity, parent, source manifest, root function,
input/output type source, description, preconditions, dependencies, examples,
assessment references and originating run. Store statuses such as candidate,
usable and retired as ordinary application data with reasons.

Proposed host helpers, callable through the selected evaluator:

- `workspace.read/list/search` for exact source and metadata retrieval.
- `program.check(manifest, root)` using existing loader/type validation.
- `program.run(manifest, root, inputs, environment)` using existing child runs.
- `workspace.commit(candidate, expectedRevision)` to persist a version.

Pin an immutable dependency closure for a run. Generated imports must resolve
against this closure, not whichever version happens to be installed later.
Do not start with a new package solver; use explicit local references and the
existing package machinery when distribution becomes a real requirement.

## Required architectural work

1. Replace the studio's fixed six-file loader with manifest loading for this
   workspace. Preserve the authored six-file programs as ordinary manifests.
2. Expose check/run and diagnostics to natlang through eval. The existing IDE
   child runner is useful, but verify its supported source and binding closure
   explicitly rather than assuming that arbitrary generated programs work.
3. Persist generated source and examples independently of conversation history.
4. Bind host environments deliberately. Pure declared-input trials can use
   fresh environments; methods working with native tables can share selected
   host objects. Sharing is supported and records its effect/replay limitations.
5. Link actual child traces and results to method assessments. A model-written
   `passed` field is not evidence of execution.

No self-modifying active frame is needed. Invoke newly generated code as a new
checked child program; promote it into later program manifests after evaluation.

## Delivery sequence

1. Generate and execute one small natlang function from notebook context; show
   its source, signature and real result in the UI.
2. Persist it, reload, and reuse it on an unseen input through an ordinary call.
3. Add counterexample-driven revision and version comparisons, preserving
   failed variants and the input that exposed their error.
4. Support a library with multiple methods, retrieval and semantic applicability
   checking. Add a negative-transfer case where a superficially similar method
   must be rejected.
5. Add extraction from an extended successful investigation, rather than only
   synthesis from a direct “write a function” request.

## Tests and teacher tasks

Exact checks: source closure resolution, malformed source diagnostics, typed
input/output rejection, missing engine, worker cancellation, durable methods,
dependency version pinning and recovery after a completed trial.

Semantic cases: cohort comparison, date normalization with ambiguous locales,
reusable log grouping, a semantic merge helper and a data-quality predicate.
Include tasks where abstraction is premature or one example is misleading.
Hidden tests must be maintained separately from generated examples so the same
model does not define and satisfy its own entire oracle.

Acceptance: an originally absent method is inspectable source, executes after
reload, generalizes to held-out input and can be revised on concrete evidence.
Report both useful reuse and harmful reuse. A wrapper around an existing action
does not demonstrate invention; a new domain algorithm does.

## Risks and follow-on consumers

Watch for cosmetic abstractions, overgeneralization, hidden host dependencies,
large source closures and recursive attempts to repair the wrong specification.
Expose these as observable failures; keep long productive trials available.

Reuse in terminal recipes, build repairs, type-analysis passes, media workflows,
NPC behaviors and semantic merge strategies. Share source/run/storage machinery;
keep each domain's judgments in its natlang codebase.
