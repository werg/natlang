# Algorithm patterns

## Orchestrate in code, judge in natural language

Most applications are ordinary TypeScript with a few natural-language decisions at the points where meaning matters. Write the control flow, validation, and bookkeeping in code, and call `nl` (or a named `.nl` function) for the judgment:

```ts
const labels = await Promise.all(tickets.map(ticket => classify(ticket, rubric)));   // parallel siblings
const real = tickets.filter((_, i) => labels[i] !== 'spam');                          // exact work stays exact
```

Move orchestration into a natural-language function only when the *order of operations itself* needs judgment: a notebook that decides which ready cell to run next, a research controller that chooses what to investigate. Then keep the operations exact and let natlang sequence them, inspecting each result.

## Long algorithmic work for small interpreters

Make the next meaningful action apparent from the instructions and typed state. A long run can be simple to execute if its steps, loop state, and helper contracts are clear. A tree walk recurses on each node's children, passing depth or path as separate arguments. For a graph traversal keep the frontier, visited IDs, results, and unresolved dependencies in typed state; let natlang choose priorities, and let an exact helper supply adjacency and readiness. Extract a helper when it gives a meaningful contract or reduces repeated context, not for every elementary operation.

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

## Checked stages

When a stage's output can be checked mechanically (a verifier, a compiler, a test run, a schema), build the pipeline around the check. Give the stage the checker as a service so it can check its own answer before returning; check again in the host; send a rejected answer back once with the checker's message through an optional argument (`problem?: string`: "why an earlier answer was rejected"); if it still fails, keep the previous version and record the rejection. Check behavior, not only validity: run the program on its inputs before and after the stage. A check sees only what the inputs exercise: once a function is no longer called, a wrong rewrite of it changes nothing observable. Run every stage through the same check-and-retry path, the first one included: a stage called outside it ends the whole pipeline when its call fails.

A stage can write code from contracts another stage declared: the compilers' front ends declare each runtime function they need with its contract in a comment, and one shared runtime stage writes whatever a header declares, for any language.

## Judgments in synchronous code

A synchronous hook (an SQLite function, a sort comparator, a parser callback) cannot await a natural-language call. Decide first: collect the distinct values the hook will see, judge them in parallel (a `readout: decision` function is one scoring pass each), store the verdicts keyed by value, and let the hook look them up. The cache also makes repeated questions free.

## Decisions with floors

A finite judgment that gates an action (run a command, accept an answer, hand a step to a cheaper model) is a `readout: decision` function, and the host acts on its probabilities: `const d = await runtime.decide(risk, command, task)` gives every value's probability. Set a floor per action and escalate below it: refuse at p(destructive) ≥ 0.7, ask the user at p(review or worse) ≥ 0.5, run otherwise; give a turn to the small model only at p(routine) ≥ 0.8. A judgment about a stronger model's work is advice: add it to the tool result in brackets, say it comes from a quick check, and send a final answer back at most once. Without a scoring driver `decide` reports the sampled value with probability 1 and `scored: false`. (`applications/pi`)

## A model that writes natlang

A stronger model can author natlang at run time. Give it a tool that takes a script, and answer the tool by running the script in a natural-language function whose instructions say to run it: wrap the small model's driver so that the call whose opening carries those instructions gets `eval` of the script and then `return_result` with what it observed, and every other request, including the script's own `nl` calls, goes to the small model. The script reaches the world through services, gated like any other tool, and the stronger model reads only what the script reports. (`applications/pi`, codemode)

## One set of stages, two drivers

To run the same stages from a checked host driver and from a natural-language driver, put them in the natural-language driver's callable folder (`compiler.nl` beside `compiler/`) and call them from host code through the import's typed children (`compiler.opt.mem2reg(fn, context)`); items inside a callable folder cannot be imported on their own. Run the natural-language driver with `codeEdits: 'deny'` when its stages must not be rewritten during a run, limit model requests in flight by wrapping the `ModelDriver` (the driver schedules its own concurrent calls, which a host-level limiter never sees), and check its final result in the host. (`applications/compilers`)

## Extension decisions

Before adding a runtime feature, try what exists: typed values, named functions, ordinary control flow, services, and `iterateOn`. Search, SQL, processes, binary assets, and stronger-model calls are services or callable-folder helpers. When a general capability is missing, implement it in the shared runtime for Node and browser alike rather than as an application-local workaround.
