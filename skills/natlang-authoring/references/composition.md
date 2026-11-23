# Composing semantic functions

A natlang program is TypeScript in which every judgment is a typed function value. Those values compose like any
other functions: map them over data, pass them to other functions, gate on their probabilities, rebind them to another
context, put them behind a setting, iterate them until a check holds, and let them write files only through a
transaction the program checks. Each section below quotes a program in the natlang repository, so every form shown
builds and runs: [`examples/refunds`](https://github.com/werg/natlang/tree/main/examples/refunds) (a refund desk for
several shops), [`examples/triage`](https://github.com/werg/natlang/tree/main/examples/triage) and the curated
applications.

## Map judgments; keep the exact work in code

`settle` runs one independent pipeline per request, all at once. The data, the policy lookup, the floor and the
record assembly are TypeScript; eligibility, the verdict and the reply are judgments.

<!-- from: examples/refunds/desk.ts -->
```ts
export async function settle(requests: Request[], options: DeskOptions): Promise<Outcome[]> {
  const eligible = eligibility(options.eligibility);
  // One verdict function, one context per shop: the shop's policy is the `policy` its instructions name.
  const under = (shop: Shop) => verdict.in(Context.empty().with({ policy: options.policies[shop] }));
  ...
  return Promise.all(requests.map(async request => {
    const { value, confidence, escalated } = await judge(request);
    return { id: request.id, verdict: value, confidence, escalated, reply: await replyFor(request, value) };
  }));
}
```

Counting, sorting, parsing, arithmetic and record assembly stay in code even when a model is at hand: `triage` counts
its labels in a `for...of` loop over the model's answers.

## Inline functions and what they see

An inline `nl` is a function written where it is used. Its type argument (or the annotation it is assigned to) is its
signature; arguments at a call become parameters; a visible binding its instructions mention by exact name is
captured and read when the call runs.

<!-- from: examples/refunds/reply.ts -->
```ts
export async function replyFor(request: Request, verdict: Verdict): Promise<Reply> {
  const first: Reply = await nl`Write the reply to request that tells the customer the verdict.`();
  const revise = nl<(reply: Reply) => Reply>`Revise reply so that it also answers every question the customer asks in request. Keep the verdict.`;
  return iterateOn(revise, first).until(nl<(reply: Reply) => boolean>`reply answers every question the customer asks in request.`);
}
```

`first` takes no arguments: `request` and `verdict` are captured. `revise` names its parameter in its type, so its
instructions and its argument agree. To list the captures exactly instead of by mention, write
``nl.with({ rubric })`Grade essay against rubric.` ``: plain entries are snapshots taken there, and `live(n)` entries
are read at each call and written back. Outside text reaches an `nl` as an argument or a capture, never inside the
template literal.

## Decisions: probabilities, floors, escalation

A function with `readout: decision` and a finite result (a literal union or `boolean`) is scored in one pass: the
model rates every allowed value instead of writing one. `decide(fn, ...args)` (from `@natlang/node`, and the same
name in eval) calls it and returns `{ value, probabilities, confidence, scored }`. Act on the confidence in code.

<!-- from: examples/refunds/desk.ts -->
```ts
    const policy = under(request.order.shop);
    const { value, confidence } = await decide(policy, request);
    if (confidence >= (options.floor ?? 0.8)) return { value, confidence, escalated: false };
    // Unsure: the same function in the same context, on the stronger model.
    return { value: await policy.on({ model: options.stronger ?? 'teacher' })(request), confidence, escalated: true };
```

`fn.on({ model: NAME })` is the same function on another of the runtime's models: in host code an entry of the
runtime's `models`; under `natlang run`, the profile that the selected profile names as `escalate`. A name the runtime
does not have runs on its default model. The local model then settles most requests in one scoring pass each, and only
the unsure ones cost a stronger model's call. In a `.nl` body the same floors are instructions: nldb's `database.nl`
says "decide(classify, request, catalog) ... When its answer is unclear, or no answer has probability at least 0.5,
change nothing". A driver without a scorer still answers a decision function, with `scored: false` and probability 1.

## One function, another context: `.in`

Every natlang function is bound to a context: the items it can call and the data its instructions can name.
`fn.in(context)` is the same function bound to another one. The refund desk keeps one `verdict.nl` and binds it per
shop:

<!-- from: examples/refunds/verdict.nl -->
```yaml
---
description: Settle a refund request under the shop's refund policy.
args:
  request: Request
returns: Verdict
readout: decision
---
Settle request under policy, the shop's written refund policy: refund when policy grants the whole amount, partial
when it grants part of it, decline when it grants nothing.
```

<!-- from: examples/refunds/desk.ts -->
```ts
  const under = (shop: Shop) => verdict.in(Context.empty().with({ policy: options.policies[shop] }));
```

A data entry of the context (`policy`) is in scope for the call by its name. `Context.fromFolder(dir, nodeSourceFiles(root))` loads a
whole folder as a context: its `.nl` and `.ts` items, and its other files as data. The new context must offer every
item the function's instructions name, with the same signature; otherwise `.in` throws `context-interface-mismatch`
naming the item. `.on` keeps the context, so the escalated call above reads the same policy. A TypeScript helper in a
callable folder binds the same way: `import { Context } from '@natlang/node'` and `import verdict from './verdict.nl'`,
then a default-exported `under(shop)` returns `verdict.in(Context.empty().with({ policy }))` for its caller to call.

## A rule and a judgment behind one setting: `pluggable`

Where an exact rule and a judgment both answer one question, give them one interface and let a setting choose.

<!-- from: examples/refunds/rules.ts -->
```ts
export function eligibility(setting: PluggableSetting) {
  return pluggable<[Request], boolean>({ crisp: withinWindow, nl: request => eligibleNl(request) }, setting,
    { name: 'refunds.eligible', default: 'crisp', serve: 'crisp' });
}
```

`crisp` and `nl` run that side alone. `shadow` runs both on the same arguments, returns the `serve` side's answer and
records a `pluggable_shadow` trace event with both answers and `agree` (`natlang traces show CALL_ID`). Run in shadow,
read the disagreements, then move the setting. `applications/helpdesk` puts its response deadline, queue order and
escalation plan behind three such settings.

## Iterate until it holds

`iterateOn(step, initial).until(check)` runs a step on its own output until the check holds. The step returns the
next state. The check can be a natural-language function: `replyFor` above revises until
``nl<(reply: Reply) => boolean>`reply answers every question the customer asks in request.` `` holds. Such a loop
needs no step limit; a progress judge reviews long runs and stops one that no longer improves. A TypeScript check
needs a bound the loop can always reach: a measure that every continuing step lowers, as the notebook's run of its
cell graph has,

<!-- from: applications/notebook/index.ts -->
```ts
  const finished = await iterateOn((run: NotebookRun) => advance(notebook, run, options), initial)
    .withMeasure(remainingCells).until(run => run.status !== 'running');
```

or a step limit with a deliberate answer when it is reached:

<!-- from: examples/triage/triage.ts -->
```ts
    return await iterateOn(shorten, draft).withLimit({ maxSteps: 3 }).until(text => words(text) <= MAX_SUMMARY_WORDS);
  } catch (error) {
    // After three rewrites, keep the shortest honest summary rather than failing the report.
    if (error instanceof IterationLimitError) return error.lastState as Summary;
```

Use `iterateOn` when each step depends on the previous state. Independent items are a `Promise.all` map.

## Change files through a transaction

`folder.propose(reducer, ...args)` runs the reducer on a private copy of the folder and returns a proposal: the new
`folder` (a snapshot), the reducer's `value` and the `diff`. Nothing changes until `folder.accept(proposal)`, and
`accept` refuses a proposal whose base changed meanwhile (`FolderConflictError`). Check the proposal in code first.

<!-- from: examples/refunds/ledger.ts -->
```ts
  const proposal = await folder.propose<[], Ledger>(async (draft: FolderHandle) => {
    const notes = await Promise.all(outcomes.map(outcome => note(outcome)));
    const ledger = await draft.file('ledger.json').readJson<Ledger>();
    for (const [index, outcome] of outcomes.entries()) {
      await draft.file(`cases/${outcome.id}.md`).writeText(`# ${outcome.id}: ${outcome.verdict}\n\n${notes[index]}\n`);
      ledger[outcome.verdict] += 1;
    }
    await draft.file('ledger.json').writeJson(ledger);
    return ledger;
  });
  // Accept only what was meant: one case file per outcome and the ledger, nothing else.
  const expected = new Set(['ledger.json', ...outcomes.map(outcome => `cases/${outcome.id}.md`)]);
  if (proposal.diff.changes.some(change => !expected.has(change.path))) return undefined;
  await folder.accept(proposal);
```

A reducer is a TypeScript function `(folder, ...args)` or a `.nl` function with `kind: directory-reducer`.
`folder.apply(reducer, ...args)` proposes and accepts in one step when the reducer's success is the check: inside
nldb's `database.nl`, `folder.apply(execute, statement)` keeps a change only if it succeeds, and a question is run as
`execute(folder, statement)` so that nothing it writes is kept. `folder.snapshot()` captures a state,
`snapshot.branch()` gives a writable copy, and `folder.select(snapshot)` makes a snapshot current.
`folder.iterateOn(reducer, initial)` iterates a reducer over the folder; each successful step publishes the folder
and the state together. The check that
decides whether a change is good is code or a service the reducer cannot edit.

## Properties and outside text are types

`Is<T, "predicate">` is a `T` whose values satisfy the predicate; the runtime checks it where a value enters the type
and sends the producing model the fix when it fails. `Untrusted<T>` is outside text: the model sees it as a quoted
data block. Both go in `types.ts`, where every function that uses the type gets them.

<!-- from: examples/refunds/types.ts -->
```ts
/** A refund request as it arrives. The customer's text is outside data: the runtime fences it and never follows it. */
export type Request = { id: string; order: Order; text: Untrusted<string> };
/** How a request is settled. */
export type Verdict = 'refund' | 'partial' | 'decline';
/** What the customer is told. */
export type Reply = Is<string, 'two to four polite sentences that state the decision and promise nothing beyond it'>;
```

Text read from a file or a request becomes `Untrusted` where it enters the program:
`untrusted(request.text, 'customer request r1')` labels it with its source. More in [refinements](refinements.md).

## Neuralese between stages

A chain of typed stages can carry its intermediate values as Neuralese (compressed model state) instead of text,
without a change to the source: [fusion](fusion.md) says how to write chains the runtime can fuse. A function whose
result some callers read as text and others keep as Neuralese declares `generic: { R: string | Neuralese<string> }`
and `returns: R`; the builtin `view(value, instructions?)` is one ([language](language.md)).

## From judgment to code: the ladder

A function starts as instructions and moves toward code as evidence accumulates, with the function staying the
specification:

1. **A scored decision.** `readout: decision` on a finite result: one pass, calibrated through `decide`.
2. **Compiled cases.** Every call is recorded (`natlang traces`). `natlang specialize --definition NAME` compiles a
   function's recorded calls into guarded crisp TypeScript cases; the runtime loads them by itself, starts a new case
   in shadow, promotes it on evidence, and hands any input no case admits (or a case that fails) back to the model.
   `natlang compilations list`, `show` and `why CALL` explain what served a call
   ([`examples/specialization`](https://github.com/werg/natlang/tree/main/examples/specialization)).
3. **Escalation.** Below a floor, `fn.on({ model })` asks a stronger model. The escalated calls show which inputs
   the local model cannot settle yet: save them as cases with `natlang traces fix CALL_ID --expected JSON`.
4. **Tuning.** `natlang tune` trains the local model's weights on the program's `examples.json` and activates a
   release; `--rollback` returns to the previous one ([tuning](tuning.md)).
5. **Improvement.** `natlang improve init --export NAME` writes an improvement case for a program;
   `natlang improve CASE.json --out DIR --model MODEL` proposes source changes, scores them on the case's train,
   validation and test cases, and `natlang improve adopt DIR --into CHECKOUT` takes the winner
   ([adaptation](adaptation.md) covers `natlang eval` and `natlang optimize` for instruction search).
