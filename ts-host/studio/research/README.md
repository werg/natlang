# Inquiry Lab

Inquiry Lab is a native natlang research application at
`http://127.0.0.1:8766/ts-host/studio/research/`. Start the existing Studio
companion with `node ts-host/scripts/serve-studio.mjs` after building `ts-host`.
The Studio sidebar links to it. It shares the browser client, local model
catalog, worker child runner and IndexedDB store with the other applications.
Loading a model is explicit; CPU is initially selected. The teacher server is
not involved.

The bundled example asks whether a deployment improved reliability. Its four
rows deliberately exhibit a population-mix reversal: both desktop and mobile
failure rates improve, while the aggregate rate worsens. The methods note
also cautions that a profile label may not identify a physical device. These
are evidence to examine, not a prewritten answer in application code.

## Natlang owns the work

The running controller is [`programs/reduce.nl`](programs/reduce.nl). It can
inspect the workspace, call [`learn.nl`](programs/learn.nl),
[`revise_schema.nl`](programs/revise_schema.nl),
[`invent_interaction.nl`](programs/invent_interaction.nl),
[`preserve_intent.nl`](programs/preserve_intent.nl) and
[`investigate_beliefs.nl`](programs/investigate_beliefs.nl), then run actual
methods, revise source, form conclusions, generate views and continue.
These are ordinary functions with typed inputs and outputs. Their proposals
do not commit themselves; the root decides what to do with them.

The selected TypeScript evaluator exposes exact helpers under `host.research`:
list/search/read artifacts, read/search full host-owned evidence, run checked
programs, inspect receipts, compare manifests, query belief links, save
candidates, commit and activate. The host does not decide whether a method is
useful, a source supports a claim, a migration preserves meaning, or two
intentions should merge. Newly authored `.nl` and `.ts` files execute from an
immutable manifest; a controller update takes effect on the next event, so a
running lexical frame is not changed underneath itself.
Candidate review includes the full base, proposed and active artifact versions
and identifies paths edited on both branches. It does not choose a merge.

The original Studio apps require final state to equal host-produced operation
state. Inquiry Lab instead lets natlang construct semantic State directly.
The host checks the active manifest, revision progression, view binding and
actual execution receipts at commit. An event may change semantic State
without a host operation.

## Artifacts and effects

- Content-addressed artifacts and manifests live in `research_workspaces` in
  IndexedDB. A manifest has one parent and a map from path to immutable content.
  Candidate manifests are durable without moving the active pointer. An active
  manifest changes through compare-and-swap; stale candidates need rebase.
- Sources, domain schemas, evidence, claims, assessments, method descriptions,
  migrations, intentions and views are ordinary artifact kinds. Generated
  source and structural types are checked by the existing natlang loader when
  run. Other artifact meaning remains an application judgment.
- A generated view is a versioned tree of DOM data plus bindings from stable
  control IDs to actual natlang handler roots. The renderer supports text,
  lists, tables, details, inputs, selects and meters. Controls emit revisioned
  events; the handler executes as natlang and its result becomes an observation
  for the parent reducer. Draft values survive repaint and reload.
- A child call receives a stable event-derived ID and a pinned source manifest.
  Its running receipt is saved before execution. A recovered running receipt
  becomes `unknown` and is never silently replayed. Actual values and trace
  references are saved. Every child receipt also records its model, root seed
  and selected evaluator, including failed calls. A cancelled call with
  uncertain effects is `unknown`.
- Large imported text is kept in `native_values`; its artifact holds a reference
  and preview. Natlang can search it or read explicit windows. Users can
  download the complete original. Export includes full native values, source,
  candidate history, receipts, app history and referenced child traces.
- Authored TypeScript methods can use `host.research.readNative(id)` inside a
  child run to process complete host-owned text. The child sees only native IDs
  referenced by its pinned manifest. This is shared eval authority, so methods
  with external effects need the same receipt discipline as other child runs.
- Imported receipts are marked as historical records in the new host. They
  document prior observations; external objects or services are not recreated
  by import.

The TypeScript eval environment can run trusted local code with host authority.
The generated view tree does not accept raw HTML, JavaScript or style code.
Page-authored collaborative content would need a different environment policy.

## Why the five capabilities share this application

| Direction | Executable path through Inquiry Lab |
|---|---|
| Learned tools | Create source artifact, run it with exact inputs, inspect receipt, revise and retain a method artifact. |
| Revisable schemas | Save candidate type and migration source, run old/new analyses and activate a coherent manifest. Raw evidence stays addressable. |
| Generated interactions | Save view tree and handler source, check bindings, render browser controls and feed their handler results back to natlang. |
| Intent-preserving changes | Save two candidates from a common base, inspect exact diffs and stated intents, produce a reconciled candidate or unresolved alternatives. |
| Beliefs and investigation | Save evidence-linked claims and assessments; query exact backlinks and search for missed semantic relationships before reconsidering. |

These are working execution paths, not measured claims that the current small
model is reliable at all five tasks. The semantic outcome depends on the
loaded model and on teacher/student training.

## Evaluation

[`scenarios.mjs`](scenarios.mjs) defines eight tasks covering population mix,
method transfer, schema revision, UI generation, compatible and incompatible
intentions, contrary evidence and irrelevant updates. Export a workspace and
run the structural audit:

```bash
node ts-host/scripts/research-audit.mjs research-export.json cohort_reliability
```

The audit validates the bundle and checks for structural evidence. Its report
leaves semantic review explicitly unreviewed and never admits a training
sample on its own. A reviewer must inspect source, real receipts, claims,
counterexamples and held-out behavior. Record model/tokenizer, decoder,
seed, source manifest, input order and evaluator bindings for repeatability
trials; matching seed and model alone is insufficient.

Wiring and browser checks:

```bash
node --test ts-host/test/research-*.test.mjs
NATLANG_CHROMIUM=/path/to/chrome node ts-host/scripts/research-smoke.mjs
```

The browser smoke keeps hardware GPU disabled and uses no live model. It
checks import, full large evidence through a generated method, generated view controls, drafts, reload,
export/import and mobile layout. Tests with scripted interpreter turns verify
source loading and state ownership, while a real child run verifies crisp
generated source. Live semantic quality and long teacher trajectories remain
unmeasured. The teacher server was not restarted for these checks.

## Current limits that should guide the next iteration

- The default generated UI path is a typed DOM tree. Truly novel canvas or
  spatial interactions may need a versioned custom renderer/module with its
  own lifecycle and environment policy. The present tree already supports
  interactive comparison, tables and inputs.
- Artifact manifests currently occupy one IndexedDB value per workspace;
  large evidence is referenced separately. Very large source/artifact
  collections will need per-artifact storage and garbage collection.
- The research app has one local writer and uses Web Locks plus compare-and-swap.
  It has candidate branches and semantic merge instructions, but no network
  replication transport or conventional CRDT convergence guarantee.
- Native evidence search is exact substring search. Natlang decides what to
  search for; richer ranking can remain a crisp library if cases demand it.
- Structural audit cannot decide whether a belief is warranted, a migration
  preserves meaning or a generated interaction helps a person. Those are
  reviewed scenario gates and training targets.
