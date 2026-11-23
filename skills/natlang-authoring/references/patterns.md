# Patterns from working programs

Each pattern is how one of the curated programs in the natlang repository does it; read the program beside the
pattern. The five to start from: [`examples/triage`](https://github.com/werg/natlang/tree/main/examples/triage) and
[`examples/refunds`](https://github.com/werg/natlang/tree/main/examples/refunds) (typed judgments mapped over data,
calibrated decisions, escalation, a pluggable rule, a folder transaction; walked through in
[composition](composition.md)), [`examples/review`](https://github.com/werg/natlang/tree/main/examples/review) (a
named function whose callable folder holds a decision leaf and an exact aggregate),
[`applications/notebook`](https://github.com/werg/natlang/tree/main/applications/notebook) (a dependency graph run
one ready cell per step, with a measure and checked citations),
[`applications/helpdesk`](https://github.com/werg/natlang/tree/main/applications/helpdesk) (per-ticket event loops
whose policies are natural-language functions behind settings) and
[`applications/nldb`](https://github.com/werg/natlang/tree/main/applications/nldb) (a relational database whose
engine is natural-language stages over a documented folder format). Each application's `DECOMPOSITION.md` records,
part by part, what is a function, an instruction, a crisp helper, a service or host mechanism, and why.

## Types and exact steps in code, every judgment a typed function

Control flow, validation, data and bookkeeping are TypeScript; each judgment is a typed function called where its
answer is needed:

<!-- from: examples/triage/triage.ts -->
```ts
  // Only the tickets the local model is unsure of go to the stronger model.
  const classifyTicket = escalate(classify, { to: 'teacher', floor: 0.8 });
  const labels = (await Promise.all(tickets.map(ticket => classifyTicket(ticket, rubric)))).map(result => result.value);
  const real = tickets.filter((_, i) => labels[i] !== 'spam');
```

When the order of operations itself needs judgment, the orchestration is a natural-language function and the
operations stay exact: nldb's `database.nl` classifies a request and hands it to the query processor or to schema
definition; the notebook's `choose_goal.nl` picks the cell that answers a request and code runs the cells.

## A named function and its folder

`examples/review` is `review.nl` with a callable folder: `assess.nl` judges one observation and calls its own
`verdict.nl` (a decision, tested exactly in `examples.json`), and `summarize.ts` counts the verdicts. The parent gets
its children's results, not their raw inputs, and the count is code. Items of one folder do not see each other, so a
helper lives in the folder of the item that calls it.

## Decide on a snapshot, apply a pure commit

`applications/helpdesk` runs one `KeyedEventLoop` per ticket. `reduce` is crisp: it applies events, sets timers as
state (`due`, `wakeAt`) and schedules model work after the commit. Each model call reads a snapshot of the
conversation and returns a value; the reducer applies it only while the snapshot is current (a triage of more
messages replaces one of fewer, never the reverse). Exactly-once delivery by event ID, ordering and durability stay
mechanism. Its policies are natural-language functions behind settings, each with a crisp counterpart:

<!-- from: applications/helpdesk/index.ts -->
```ts
    this.allowance = pluggable<[Triage], number>({
      crisp: triage => this.times[triage.urgency],
      nl: async triage => (await deadlineFor(triage.urgency, triage.topic, deadlinePolicy)) * 60_000,
    }, options.deadlineMode, { ...servesCrisp, name: 'helpdesk.deadline' });
```

`deadlineFor.nl` reads the desk's written service policy in numbered steps; the crisp side is the table. A
natural-language side whose answer must have an exact shape is checked in code before it is served: the queue
ranking must list every offered ticket exactly once. For interfaces, natlang chooses grouping, explanation and next
steps (a view plan) and exact code builds the view; generated controls emit events that real handlers reduce.

## Decisions with floors

A finite judgment that gates an action (run a command, accept an answer, hand a step to a cheaper model) is a
`readout: decision` function, and its caller acts on the probabilities: `const d = await decide(risk, command, task)`
in host code or eval, or instructions that state the floors. nldb's server asks back when no class of a request has
probability at least 0.5; the refund desk escalates a verdict below 0.8. Set a floor per action and escalate below
it: refuse at p(destructive) >= 0.7, ask the user at p(review or worse) >= 0.5, run otherwise. A judgment about a
stronger model's work is advice: add it to the tool result in brackets, say it comes from a quick check, and send a
final answer back at most once. Without a scoring driver `decide` reports the sampled value with probability 1 and
`scored: false`.

## Iteration with a measure

The notebook runs the goal's dependency closure one ready cell per step: the state is the run, the step runs one cell
the previous steps made ready, and the measure is the number of required cells not yet run, which every continuing
step lowers.

<!-- from: applications/notebook/index.ts -->
```ts
export function remainingCells(run: NotebookRun): number {
  if (run.status !== 'running') return 0;
  const done = new Set(run.order);
  return [...requiredIds(run)].filter(id => !done.has(id)).length;
}
```

Use `iterateOn` for open-ended refinement (repair until checks pass, shorten until short enough, revise until a
natural-language check holds) and for steps that depend on the previous state; map independent items with
`Promise.all`. A TypeScript stopping check needs `withMeasure` or `withLimit`; when a limit is reached, handle
`IterationLimitError` deliberately (triage keeps the shortest honest summary). A natural-language check needs
neither: a progress judge reviews long runs.

## Instructions a small interpreter can follow

What a large model infers, a small interpreter needs said. These held across thousands of recorded runs:

- **State the step a loop repeats.** An instruction that names `iterateOn` but not the step gets stepped by hand, one eval per move, until the turns run out, even with a hint. Say what the state is, what one step does, and when to stop: "Repeat one control step as often as it takes: heater on below 20.5 °C, vent on above 21.5 °C, then take a reading." The loop an instruction describes is the loop the model writes.
- **Say where the data is and how to reach it.** Name the helper and its paging (`facts.page(n)`, `facts.pages()`), and give helpers doc comments (page size, units, what a null means): the opening shows them. A model that does not see how to reach the data answers without it.
- **Keep the outcome words for outcomes.** "If the store lacks a fact the reasoning needs" reads as the `blocked` status, and models then report blocked instead of the answer. When "not found" or "unsupported" is a legitimate result, say so: "when no selection supports it, that is the answer: return supported false".
- **Say what may be batched.** For an environment driven by commands, say that one eval can carry out several in a row, stopping at the first unexpected observation; otherwise a model spends a turn per command.
- **Give each criterion one reading.** A distractor is good only if the instruction settles it ("archive a ticket once the customer confirmed the fix" does not cover a declined request the customer accepted); if careful readers could disagree, the program is ambiguous, not the model wrong.
- **A function that acts for its caller says so in its result.** When a callee may itself perform the caller's last
  step (commit the next state, send the reply), return a value that names what it did, `{ committed: "running in
  phase summarize" }`, not `null` or a bare sentinel. A small interpreter reads `null` as "nothing happened" and does
  the step again over the callee's work (pi's compaction did, until its select and summarize returned `{ committed }`).
- **Make the named helper the natural route.** Prefer a helper the model can already call over asking it to invent one, and when a specialist should do the work, give the specialist the access (a scoped service) rather than instructions to please delegate.

## Checked stages

When a stage's output can be checked mechanically (a verifier, a compiler, a test run, a schema), build the pipeline around the check. Give the stage the checker as a service so it can check its own answer before returning; check again in the host; send a rejected answer back once with the checker's message through an optional argument (`problem?: string`: "why an earlier answer was rejected"); if it still fails, keep the previous version and record the rejection. Check behavior, not only validity: run the program on its inputs before and after the stage. A check sees only what the inputs exercise: once a function is no longer called, a wrong rewrite of it changes nothing observable. Run every stage through the same check-and-retry path, the first one included: a stage called outside it ends the whole pipeline when its call fails.

A stage can write code from contracts another stage declared: the compilers' front ends declare each runtime function they need with its contract in a comment, and one shared runtime stage writes whatever a header declares, for any language.

When intermediate results cannot be run (code over virtual registers before register allocation), check the chain of stages once its output runs, and redo the whole chain with the problem. Record each step of the chain on its own, so that a failure still shows where the time went.

The notebook checks its answer the same way: every `id@revision` the explanation cites must name a cell
that ran, and a failing answer is asked again once with the problem. Services that can reject a call should say why in one sentence with the valid form (the expected positional
arguments, the valid page range); the model recovers from that sentence, so the instructions need not list invalid
forms.

## A system rewritten in natural language

To port a real system (a compiler, a database, an agent harness), keep its architecture and write its logic as the stages' instructions:

- **State each stage's algorithm, not the tool it imitates.** "Do what InstCombine does" leaves the model to recall a whole system. Numbered steps with their data structures (a lattice with executable edges, a worklist, phis at iterated dominance frontiers, a linear scan by interval start) give it something to execute, often exactly, in eval.
- **Type the representation at every stage boundary.** Declare it in `types.ts`, with doc comments on the format: a syntax tree `Node`, a `Plan` of relational operators, `Flow` facts, machine code as a documented string alias. Pass the representation, and not what the previous stage started from. When IR generation gets the source text as well as the checked tree, it can skip the tree.
- **Compute analyses in the caller and pass them on.** Several stages need the same analysis (dominators, live intervals). Make it a stage of its own, run it where the stages are sequenced, and pass the result as an argument. Recompute it only after a stage has changed its input, as an analysis manager does. When stages in different folders must each call the same function themselves (a cut selection, a boundary rule), list it in their `uses:` instead.
- **Make a fixed sequence a function, not a model-run step.** `folder.apply`, `folder.propose` and `folder.iterateOn` take a TypeScript function `(folder, ...args)` as well as a `kind: directory-reducer` `.nl`: it runs on a private transaction copy of the folder and its changes are installed when it returns and discarded when it throws. When one iteration is a fixed sequence of exact checks and calls to natural-language stages (measure, call `diagnose`, call the editor, check, compare), write it as that function and let `iterateOn` apply it; a `.nl` step that tells the model to run the sequence in eval costs a request and a chance to get it wrong. Keep the `.nl` for the judgments inside the sequence. (`applications/program-improver/improveStep/lifecycle.ts`)
- **Nest transactions with `folder.apply`.** Inside a directory reducer, run each writing stage as `await folder.apply(stage, ...)`: its changes join the caller's only if it succeeds. Call a reading stage directly, so that anything it writes is dropped. A server function can return failures as results ({ kind, error }). The host then commits only the outcomes the server reports as done, and the only crisp part left is the durable commit.

(`applications/compilers`, `applications/nldb`, `applications/pi`)

## Delegation by access

When a caller should get answers from specialists rather than work the evidence itself, the evidence belongs to the specialists: scope each store to its specialist's function (`serviceScopes`). The caller reads the specialists' instructions and the stores' declarations, sees that only the specialists can use the stores, and asks; each specialist queries its own store inside its call. Everything stays readable: this limits what a call can do, not what it can see.

## Judgments in synchronous code

A synchronous hook (an SQLite function, a sort comparator, a parser callback) cannot await a natural-language call. Decide first: collect the distinct values the hook will see, judge them in parallel with `readout: decision` (or `readout: instant` on latency-critical paths), store the verdicts keyed by value, and let the hook look them up. The cache also makes repeated questions free.

## Semantic merging, notionally CRDT-like

Merging can be the natlang algorithm. Present the common base, each intention in a canonical order, and provenance; preserve incompatible alternatives as explicit unresolved conflicts rather than losing them. Check exact invariants (every update accounted for once, IDs preserved) in code. Test reordered delivery, duplicates, contradictions, and replay. A shared model and seed are part of a reproducibility profile, not a convergence proof; report measured agreement.

## Learned methods and generated source

Natlang can author a candidate method, run it, inspect failures, revise, and retain it. Store the source revision, declared types, test inputs, and actual receipts; keep candidate and active revisions distinct when review or concurrent edits matter. Schema changes need executable migrations and preserved evidence.

## Values a small interpreter must not retype

A function that returns what a service produced (a provider message, a derived view, model-written text) returns the variable from eval (`return message;` with `finish: true`), never a literal in `return_result`: small models rewrite long values from memory and drop or invent fields (signatures, usage, provider ids). The runtime makes this the easy path: a computed value is assignable as in TypeScript, so fields the declared type does not list are kept instead of refused. A refusal there used to be what sent executors back to retyping: pi's provider messages carry a `durationMs` its declared type lacked, and the executor answered the refusal with a hand-written copy with invented usage and a fake signature. Declare such types as what the function relies on, not as a full schema of the provider's value.

## One set of stages, two drivers

To run the same stages from a checked host driver and from a natural-language driver, put them in the natural-language driver's callable folder (`compiler.nl` beside `compiler/`) and call them from host code through the import's typed children (`compiler.opt.mem2reg(fn, context)`); items inside a callable folder cannot be imported on their own. Run the natural-language driver with `codeEdits: 'deny'` when its stages must not be rewritten during a run, limit model requests in flight by wrapping the `ModelDriver` (the driver schedules its own concurrent calls, which a host-level limiter never sees), and check its final result in the host. (`applications/compilers`)

## A model that writes natlang

A stronger model can author natlang at run time. Give it a tool that takes a script, and answer the tool by running the script in a natural-language function whose instructions say to run it: wrap the small model's driver so that the call whose opening carries those instructions gets `eval` of the script and then `return_result` with what it observed, and every other request, including the script's own `nl` calls, goes to the small model. The script reaches the world through services, gated like any other tool, and the stronger model reads only what the script reports. (`applications/pi`, codemode)

## An agent harness as a program

To port a durable agent harness (pi-durable), keep its host for what is mechanism — the commit line, storage, the scheduler's invocations and timers, the registry — and write its behaviour as functions: one entry per durable task kind (`generation.nl`, `tool.nl`, `compaction.nl`) whose phases are branches, each phase a stage of its own (prepare, request, classify, answer, the tool round, the tool-call policy, compaction's cut and summary), plus the scheduler's policy and admission. The agent is a provider model the functions call through a service (`ai.turn`); the functions are the harness around it, run by a small executor. Natural language never runs inside a commit: a function reads committed state, decides, and hands the host one list of write operations with guards on what it read (`durable.commit(ops, expect)`); a failed guard means the state changed, so it reads and decides again. Operations that judge a commit's own staged writes stay in the host. Shared rules several folders need (the boundary selection, the compaction cut, context derivation) are items listed in `uses:`; hot paths (context building, the system-entry plan and context estimate, the scheduler's policy, admission) have a crisp and a natural-language implementation behind one interface, chosen by a setting; a crisp side that needs host code reaches it through a service method, since callable-folder code imports only files of its own package and declared packages. Check the port with the original's own test suites, run against the natural-language task kinds with a real executor: they assert exact entries and documents. (`applications/pi`)

## Extension decisions

Before adding a runtime feature, try what exists: typed values, named functions, ordinary control flow, services, and `iterateOn`. Search, SQL, processes, binary assets, and stronger-model calls are services or callable-folder helpers. When a general capability is missing, implement it in the shared runtime for Node and browser alike rather than as an application-local workaround.

**Iteration measures.** `iterateOn(step, initial).withMeasure(state => n)` gives the loop a bound that holds without a step limit: `n` is a nonnegative integer counting the work still to do, and every step that continues the loop must lower it. The step that makes `until` true may leave it unchanged, so write the plain count (`items.length - done`), not a padded one. If the measure reaches 0 while `until` is false, the loop stops with `iteration-measure-exhausted`: let `until` be true where the measure is 0, or count one more unit for the work that remains.
