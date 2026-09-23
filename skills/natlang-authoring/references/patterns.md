# Algorithm patterns

## Orchestrate in code, judge in natural language

Most applications are ordinary TypeScript with a few natural-language decisions at the points where meaning matters. Write the control flow, validation, and bookkeeping in code, and call `nl` (or a named `.nl` function) for the judgment:

```ts
const labels = await Promise.all(tickets.map(ticket => classify(ticket, rubric)));   // parallel siblings
const real = tickets.filter((_, i) => labels[i] !== 'spam');                          // exact work stays exact
```

Move orchestration into a natural-language function only when the *order of operations itself* needs judgment: a notebook that decides which ready cell to run next, a research controller that chooses what to investigate. Then keep the operations exact and let natlang sequence them, inspecting each result.

## Long algorithmic work for small interpreters

Make the next meaningful action apparent from the instructions and typed state. A long run can be simple to execute if its steps, loop state, and helper contracts are clear. For a graph traversal keep the frontier, visited IDs, results, and unresolved dependencies in typed state; let natlang choose priorities, and let an exact helper supply adjacency and readiness. Extract a helper when it gives a meaningful contract or reduces repeated context, not for every elementary operation.

## Monitored iteration

Use `iterateOn` for open-ended refinement: repair until checks pass, shorten until short enough, advance until done. It records each step, reviews progress on fresh sites, and stops on a `divergent` verdict or a limit. Bound it with `withLimit` where the domain has a natural bound, and handle `IterationLimitError` deliberately (for example, keep the best honest state). Use plain finite loops for bounded work.

## Reducers and applications

A reducer `reduce(state, event) => state` is ordinary code that may call natlang. `EventLoop` applies events serially, suppresses duplicate IDs, commits before publishing, and retries a failed view without replaying the event. Keep persistence and transport exact in host code: event IDs, storage commits, effect receipts. A single event may require many operations; natlang can inspect each observation and continue.

For interfaces, let natlang choose grouping, explanation, and next steps (a view *plan*), and let exact code build the safe view tree. Generated controls must emit events that real handlers reduce.

## Semantic merging, notionally CRDT-like

Merging can be the natlang algorithm. Present the common base, each intention in a canonical order, and provenance; preserve incompatible alternatives as explicit unresolved conflicts rather than losing them. Check exact invariants (every update accounted for once, IDs preserved) in code. Test reordered delivery, duplicates, contradictions, and replay. A shared model and seed are part of a reproducibility profile, not a convergence proof; report measured agreement.

## Learned methods and generated source

Natlang can author a candidate method, run it, inspect failures, revise, and retain it. Store the source revision, declared types, test inputs, and actual receipts; keep candidate and active revisions distinct when review or concurrent edits matter. Schema changes need executable migrations and preserved evidence.

## Extension decisions

Before adding a runtime feature, try what exists: typed values, named functions, ordinary control flow, services, and `iterateOn`. Search, SQL, processes, binary assets, and stronger-model calls are services or callable-folder helpers. When a general capability is missing, implement it in the shared runtime for Node and browser alike rather than as an application-local workaround.
