# P06 — Natlang IDE and training workbench

Status: headless natlang IDE workbench in `codebases/ide/` and
`applications/ide_workbench.mjs`, plus an interactive browser playground in
`ts-host/playground/`. [Shared capabilities](README.md). The browser shell now
provides multi-file editing, live source diagnostics, revision-pinned native
runs, trace inspection, source forks, reviewed cases, IR admission, local
pipeline jobs, and run/evaluation comparison. See the
[playground guide](../../ts-host/playground/README.md) for operation and limits.

Natlang interprets text edits into exact revisioned patches and composes the
editor view. The host retains editable source revisions, checks source graph
shape and basic crisp syntax, invokes a child programme pinned to a revision,
stores full traces, and exposes read-only trace cursor events. A scenario
store freezes inputs/expected values with source identity and re-evaluates
them exactly. Integration tests exercise natlang edit/run/view, escaped HTML,
scenario evaluation, stale edits and invalid syntax.

The older natlang IDE workbench remains a headless backend with generated HTML;
the new browser editor uses the native TypeScript runtime and deterministic UI
state. The browser trace cursor does not claim execution replay or live pause.
Dataset admission currently covers natural-language leaf traces; graph and
host-effect case adapters require additional semantic lowering. The browser
shell has not yet moved event handling into `ide/step.nl`.

## Natlang prerequisites

C1 exposes source/rendering/native application objects through eval. C3 provides load/check/run for source supplied as data. C4 is essential for a useful execution inspector; full replay/fork requires additional capture coverage. C5 drives the UI/application Fold. C2 configures comparisons and training runs. No global debugger, search or delegate tools are needed; meta-operations are crisp environment APIs.

## Programme and typed boundary

`step.nl(state, event) -> EditorState` delegates to editor, navigation, diagnostics, runner, debugger and dataset functions. Each module owns a small lexical menu. Extend the current semantic highlighter and P05 type tools. Natlang constructs views and chooses application behavior; exact rendering, text edit application and source validation are helpers.

State records hold document/revision IDs, selections, diagnostics, run IDs and view descriptions. Native buffers, DOM nodes, model sessions and training jobs remain in the host environment. Shared objects can mutate there, but source snapshots used by active runs remain fixed. Typed view records are an application convention, not a new natlang type primitive.

## Crisp environment

Illustrative bindings: `editor.read(revision)`, `editor.apply(patch, revision)`, `runtime.check(source)`, `runtime.start(program, inputs, config)`, `trace.read(run, range)` and `ui.render(view)`. Implement checked meta-calls with child-run identity and explicit execution settings. A child programme gets its own scope; editor authority is not automatically inherited.

A JS/browser host can deliberately share live editor/DOM objects with the `ts` environment. A remote runtime is a separate implementation with explicit serialization. The common trace viewer reads the portable event format and uses optional native previews; opening a trace never executes its recorded code.

## Reduction and stream shape

Fold over edits, selections, run commands, diagnostic completions and training progress. Keep steps bounded; expensive analysis and execution return later events. Pin source revision and request identity; ignore superseded results when updating the current view. Short exact browser interactions can run locally while semantic decisions are pending.

The debugger initially moves a cursor through recorded reductions. Pausing a live run should first occur at existing call/action boundaries through the host API. Replaying recorded decisions, re-executing source and forking state are distinct commands. Host mutations that were not captured prevent full replay and must remain visible limitations.

## Delivery and checks

1. Browse/edit/highlight one in-memory codebase. Gate: invalid source remains editable but cannot run as valid code.
2. Run a fixture and inspect call tree, values and diagnostics via C4. Gate: source/run identity and final reconstructed state agree.
3. Add recorded-decision replay and fixture forks. Gate: no live effect during inspection/replay; shared-host gaps are explicit.
4. Add scenario creation, teacher collection, admission review, dataset manifests, training-job launch, checkpoint evaluation and selection. Gate: every selected checkpoint traces to frozen data/config and paired evaluation.

Test stale edits/results, invalid frontmatter, huge traces, child budgets, interrupted training and incompatible checkpoints. Measure typing/inspection latency rather than interpreting every keystroke synchronously.

## Trace, teacher and completion boundary

Collect real editor workflow episodes, not host scripts that make all decisions. Exact checks validate edits, views and run state; semantic checks validate suggestions. The first IDE ends at edit/run/inspect. Full training management follows functioning CLI pipelines, preserving failed/rejected examples and programme-family splits.
