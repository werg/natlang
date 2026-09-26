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

## Instructions a small interpreter can follow

What a large model infers, a small interpreter needs said. These held across thousands of teacher runs:

- **State the step a loop repeats.** An instruction that names `iterateOn` but not the step gets stepped by hand, one eval per move, until the turns run out, even with a hint. Say what the state is, what one step does, and when to stop: "Repeat one control step as often as it takes: heater on below 20.5 °C, vent on above 21.5 °C, then take a reading." The loop an instruction describes is the loop the model writes.
- **Say where the data is and how to reach it.** Name the helper and its paging (`facts.page(n)`, `facts.pages()`), and give helpers doc comments (page size, units, what a null means): the opening shows them. A model that does not see how to reach the data answers without it.
- **Keep the outcome words for outcomes.** "If the store lacks a fact the reasoning needs" reads as the `blocked` status, and models then report blocked instead of the answer. When "not found" or "unsupported" is a legitimate result, say so: "when no selection supports it, that is the answer: return supported false".
- **Say what may be batched.** For an environment driven by commands, say that one eval can carry out several in a row, stopping at the first unexpected observation; otherwise a model spends a turn per command.
- **Give each criterion one reading.** A distractor is good only if the instruction settles it ("archive a ticket once the customer confirmed the fix" does not cover a declined request the customer accepted); if careful readers could disagree, the program is ambiguous, not the model wrong.
- **Make the named helper the natural route.** Prefer a helper the model can already call over asking it to invent one, and when a specialist should do the work, give the specialist the access (a scoped service) rather than instructions to please delegate.

## Monitored iteration

Use `iterateOn` for open-ended refinement: repair until checks pass, shorten until short enough, advance until done. It records each step, reviews progress on fresh sites, and stops on a `divergent` verdict or a limit. Bound it with `withLimit` where the domain has a natural bound, and handle `IterationLimitError` deliberately (for example, keep the best honest state). Use plain finite loops for bounded work.

## Reducers and applications

A reducer `reduce(state, event) => state` is ordinary code that may call natlang. `EventLoop` applies events serially, suppresses duplicate IDs, commits before publishing, and retries a failed view without replaying the event. Keep persistence and transport exact in host code: event IDs, storage commits, effect receipts. A single event may require many operations; natlang can inspect each observation and continue.

For interfaces, let natlang choose grouping, explanation, and next steps (a view *plan*), and let exact code build the safe view tree. Generated controls must emit events that real handlers reduce.

## Semantic merging, notionally CRDT-like

Merging can be the natlang algorithm. Present the common base, each intention in a canonical order, and provenance; preserve incompatible alternatives as explicit unresolved conflicts rather than losing them. Check exact invariants (every update accounted for once, IDs preserved) in code. Test reordered delivery, duplicates, contradictions, and replay. A shared model and seed are part of a reproducibility profile, not a convergence proof; report measured agreement.

## Learned methods and generated source

Natlang can author a candidate method, run it, inspect failures, revise, and retain it. Store the source revision, declared types, test inputs, and actual receipts; keep candidate and active revisions distinct when review or concurrent edits matter. Schema changes need executable migrations and preserved evidence.

## Delegation by access

When a caller should get answers from specialists rather than work the evidence itself, the evidence belongs to the specialists: scope each store to its specialist's function (`serviceScopes`). The caller reads the specialists' instructions and the stores' declarations, sees that only the specialists can use the stores, and asks; each specialist queries its own store inside its call. Everything stays readable: this limits what a call can do, not what it can see.

## Extension decisions

Before adding a runtime feature, try what exists: typed values, named functions, ordinary control flow, services, and `iterateOn`. Search, SQL, processes, binary assets, and stronger-model calls are services or callable-folder helpers. When a general capability is missing, implement it in the shared runtime for Node and browser alike rather than as an application-local workaround.
