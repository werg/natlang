# Algorithm patterns

## Long algorithmic work for small interpreters

Make the next meaningful action apparent from the source and typed state. A long program can be simple to execute if its steps, loop state, and helper contracts are clear. Do not equate a short trajectory with a good program.

For a graph traversal, store the frontier, visited IDs, accumulated results, and unresolved dependencies. State whether ordering is meaningful and how to handle cycles or unknown nodes. Let natlang choose priorities from semantic criteria; a crisp graph query can supply exact adjacency and readiness. For a notebook, natlang can traverse dependencies and choose which cells to run; the host executes a selected cell and returns a receipt. Avoid hiding traversal inside a host `doEverything()` call.

Extract a helper when it gives a meaningful contract or reduces repeated context. Avoid splitting every elementary operation into another model episode. Keep exact projections and reusable arithmetic in crisp functions. Measure source, tool-schema, state-preview, completion, and repeated-read tokens separately when optimizing.

A conversation checkpoint is not a new algorithm invocation. Preserve accumulator paths, completed iterations, source revisions, and effect observations. Working notes should carry unresolved choices, not duplicate all state. If a resumed model keeps rereading or restarts a loop, inspect the actual checkpoint and restored state before changing the algorithm. A truncated note or missing continuation data is a harness issue worth fixing.

## Natlang applications and state reducers

A useful shape is `reduce(state: State, event: Event) -> State`. The natlang reducer interprets the event, calls host operations, inspects observations, revises its plan, and produces the new state. A single event can require many operations. The host need not impose a single proposed action or a plan/execute split.

Keep exact persistence and transport contracts in host code: immutable source revisions, unique event IDs, storage commits, effect receipt identity, process polling. Include sufficient state to explain uncertainty and recover without the old conversation. Native objects stay accessible through crisp code; they need not be represented by a new universal language protocol.

Use event streams/folds for incoming data. A frontend can queue semantic events and render after each reduction. Fine-grained cursor movement and animation may remain crisp; intent, grouping, interaction selection, and application state can be natlang-driven. A generated UI should return events to versioned natlang handlers. UI generation without working event bindings is incomplete.

## Semantic merging, notionally CRDT-like

Merging can itself be the natlang algorithm. Do not replace semantic merge with crisp convergence rules unless requested. Choose the representation for the use case: operation history, base plus competing states, structured document sections, entity graphs, schedules, inventory intent, or narrative facts.

Define what the merger sees: a common base when available, both intentions, ordered/canonicalized inputs where required, and provenance. Preserve incompatible alternatives and explicit uncertainty rather than silently losing information. Test reordered delivery, duplicate events, delayed branches, contradictory updates, and replay of the same merge context. Evaluate semantic invariants for the actual domain.

A shared model and random seed are necessary parts of a reproducible profile, not proof of convergence. Also pin tokenizer/template, inference configuration, source, inputs, event order, and relevant host observations. Hardware/backend differences can still change results. Report measured agreement; do not claim classical CRDT guarantees for a semantic merger.

## Learned methods and generated source

Natlang can notice a repeated need, author a candidate method, ask the host to load/check it, execute test inputs, inspect failures, revise, and retain it. Store the method's source revision, declared types, evidence, test inputs, and actual receipts. Keep candidate and active revisions distinct when the application needs review or concurrent edits.

Schema changes need executable migrations and evidence preservation. Test missing records, duplicates, unexpected IDs, and changed field meanings. Mechanical preservation audits identify differences; natlang judges whether meaning was preserved. A new schema or a syntactically valid method alone is not a completed migration.

## Extension decisions

Before adding a runtime feature, try existing typed values, named functions,
ordinary TypeScript control flow, the crisp evaluator, and host-owned objects.
Search, SQL, shell commands, binary assets, stronger-model calls, and native
jobs can often be host libraries callable from crisp code. This does not mean
inventing a special workaround around a broken runtime contract.

When a general capability is missing, identify its semantic contract and implement it in the shared TypeScript runtime and browser path. Check all application callers; a broadly useful correction should not be activated only in the motivating demo. Keep portability and small-model cognitive cost explicit.

Use `Record<string, T>` when a keyed collection is an ordinary typed input.
Keep indexed searches and native database queries in crisp helpers that return
small typed results. Directory reducers are the only functions with model-side
filesystem access; their relative paths resolve within the reducer's input
folder. A direct call `await reducer(folder, ...args)` discards file changes,
while `await folder.apply(reducer, ...args)` retains the reducer's selected
changes.
