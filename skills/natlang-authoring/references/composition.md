# Composing semantic functions

A natlang program is TypeScript in which every judgment is a typed function value. Those values compose like any
other functions: map them over data, pass them to other functions, gate on their probabilities, rebind them to another
context, put them behind a setting, iterate them until a check holds, and let them write files only through a
transaction the program checks. The repository examples below quote programs whose forms build and run: [`examples/refunds`](https://github.com/werg/natlang/tree/main/examples/refunds) (a refund desk for
several shops), [`examples/triage`](https://github.com/werg/natlang/tree/main/examples/triage) and the curated
applications, and [`examples/release`](https://github.com/werg/natlang/tree/main/examples/release) (a release
desk built on the standard library's combinators).

## Let a semantic function own a domain goal

Vibestudio's working tutorial-revision program, `reviseFromPublicResearch`, illustrates model-owned composition.
Its semantic parent owns the user's original request. Its callable folder provides `inspectRevisionContext` and
`identifyResearchQuestion`; declared `uses` also makes `investigatePublicQuestion` and `reviseWorkspaceDocument`
available. The research hierarchy owns public-source investigation and grounded evidence. The revision hierarchy
owns a proposal and reconciliation with the current document. Exact native reads and versioned writes stay behind
their typed operations.

An original recorded run shows the parent model invoking those named children, with further model-selected calls
inside the research and revision hierarchies. This is evidence of delegation in that run, not a mandatory route for
other goals. The parent carries the original goal and actual observations across calls, and can leave a change
unmade when the evidence does not justify it. A small pure decision function remains appropriate for one policy
question within that larger program. Keep authority in authored scopes; expose required operations by callable-folder
structure or explicit `uses`, without granting access to unrelated siblings.

## Map judgments; keep the exact work in code

`settle` runs one independent pipeline per request, all at once. The data, the policy lookup, the floor and the
record assembly are TypeScript; eligibility, the verdict and the reply are judgments.

<!-- from: examples/refunds/desk.ts -->
```ts
export async function settle(requests: Request[], options: DeskOptions): Promise<Outcome[]> {
  const eligible = eligibility(options.eligibility);
  // One verdict function, one context per shop: the shop's policy is the `policy` its instructions name.
  const under = (shop: Shop) => escalate(verdict.in(Context.empty().with({ policy: options.policies[shop] })), {
    to: options.stronger ?? 'teacher',
    floor: options.floor ?? 0.8,
  });
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

A function with `readout: decision` and a finite result (a literal union, `boolean`, a record whose fields are all
finite, or a union of such records) reasons first and then scores every allowed value (use `readout: instant` on hot
paths to skip reasoning). `decide(fn, ...args)` (from `@natlang/node`, and the same name in eval) calls it and returns
`{ value, probabilities, confidence, scored }`, plus `fields` for a record (each field's values with their
probabilities). A record type of up to 32 values is scored in one pass; a larger one field by field, and its
`confidence` is the probability of the record chosen. Act on the confidence in code (or wrap `fn` in `escalate` from
`@natlang/node/std`). A field typed `string` or marked optional makes the result open: split the open part into its own
function, or keep the tool loop. Literal unions may mix strings and numbers (`-2 | -1 | 1 | 2 | 'reset'`), and each
value names one meaning: when the same thing could be called two ways on some input (a section that occurs once is its
first and its last), give it its own value (`'only'`) or say in the type's doc comment which one it is.

**Judgments map words to referents; code finds the item.** A judgment reads what the words say and returns it as a
finite description; TypeScript resolves the description against the data. "The second right after the church" is
`{ side: 'right', ordinal: 2, after: 'church' }` from the judgment, and the map walk finds the road. A choice among items
known only at run time goes the same way: code numbers or describes the candidates and the judgment returns the
description, or `decide` runs one boolean judgment per candidate and code takes the most probable:

```ts
const scored = await Promise.all(candidates.map(async road => ({ road, p: (await decide(isMeant, sentence, road)).probabilities.find(o => o.value === true)?.probability ?? 0 })));
const meant = scored.reduce((best, next) => next.p > best.p ? next : best);
```

<!-- from: examples/refunds/desk.ts -->
```ts
  const under = (shop: Shop) => escalate(verdict.in(Context.empty().with({ policy: options.policies[shop] })), {
    to: options.stronger ?? 'teacher',
    floor: options.floor ?? 0.8,
  });
  const judge = async (request: Request): Promise<{ value: Verdict; confidence: number; escalated: boolean }> => {
    if (!await eligible(request)) return { value: 'decline', confidence: 1, escalated: false };
    const { value, confidence, escalated } = await under(request.order.shop)(request);
    return { value, confidence, escalated };
  };
```

`escalate(fn, { to, floor?, risk?, policy? })` (from `@natlang/node/std`) scores locally via `decide` and sends only
unsure or high-risk calls to `fn.on({ model: to })`, keeping `fn`'s context and recording a `decision_escalate` event
plus an `escalation` note on the local call. When the runtime has no model named `to`, `escalate` keeps the local
answer with `escalated: false` and a `reason` starting `unavailable:` rather than rerunning on the default model.
`policy: 'nl'` asks the shipped `shouldEscalate.nl`, which reads the function's recorded escalation and disagreement
history and measured tier cost from the call store. For a direct one-off retry without gating, `fn.on({ model: NAME })`
runs the same function on another runtime model (under `natlang run`, the profile that the selected profile names as
`escalate`; a name the runtime lacks runs on its default model). In a `.nl` body the same floors are instructions:
nldb's `database.nl` says "decide(classify, request, catalog) ... When its answer is unclear, or no answer has
probability at least 0.5, change nothing". A driver without a scorer still answers a decision function, with
`scored: false` and probability 1.

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
  const under = (shop: Shop) => escalate(verdict.in(Context.empty().with({ policy: options.policies[shop] })), {
    to: options.stronger ?? 'teacher',
    floor: options.floor ?? 0.8,
  });
```

A data entry of the context (`policy`) is in scope for the call by its name. `Context.fromFolder(dir, nodeSourceFiles(root))` loads a
whole folder as a context: its `.nl` and `.ts` items, and its other files as data. The new context must offer every
item the function's instructions name, with the same signature; otherwise `.in` throws `context-interface-mismatch`
naming the item. `.on` keeps the context, so the escalated call above reads the same policy. A TypeScript helper in a
callable folder binds the same way: `import { Context } from '@natlang/node'` and `import verdict from './verdict.nl'`,
then a default-exported `under(shop)` returns `verdict.in(Context.empty().with({ policy }))` for its caller to call.
Eval code has the same value as the built-in `NatlangContext` (a program type may be named `Context`), so instructions
can ask a call to bind a function itself: `const strict = verdict.in(NatlangContext.empty().with({ policy: strictPolicy }))`.
A record bound this way is a snapshot; later changes to it do not reach the bound function.

## Functions that take and return judgments

A judgment is a value of a function type, so programs build lists of them, bind their parameters, and get them back
from other functions. The screening example applies one check per requirement plus a fit judgment written for the
role:

<!-- from: examples/screening/screen.ts -->
```ts
  const checks: Check[] = [...requirements.map(requirement => meets.partial(requirement)), await fitFor(role)];
  const results = await Promise.all(applications.map(application => Promise.all(checks.map(check => check(application)))));
```

`meets.partial(requirement)` is `meets` with its first parameter bound, as `bind` binds it: a natlang function of the
remaining parameters, here `(application) => boolean`, with `on`, `in` and `iterateOn` kept. The bound values are
checked against their parameter types at once. `fitFor` is a `.nl` function whose result is a function:

<!-- from: examples/screening/fit_for.nl -->
```yaml
---
description: The judgment of fit for a role, written for that role.
args:
  role: string
returns: "(application: Application) => boolean"
---
```

Its call writes the judgment in eval and finishes with it (`return nl`...``); the runtime tells the call how. A
function type in a signature is awaited: `(application: Application) => boolean` and `(application: Application) =>
Promise<boolean>` are the same type, and TypeScript sees the promise. The same holds for parameters: a `.nl` function
with `args: { accept: "(t: Ticket) => boolean" }` takes any function of that type, and an inline
`nl<(judge: (t: Ticket) => Promise<boolean>, all: Ticket[]) => Ticket[]>` does too.

Every natlang function carries its signature, and a slot of function type checks it: parameters by position, each
able to take what the slot passes, and a result that fits the slot's. A natlang function that does not fit is refused
with `function-signature-mismatch`, naming both types. A TypeScript function passed there is checked at each call
instead: its arguments and its awaited result must have the slot's types. An `nl` that returns the function itself is
fine wherever a function is expected (a factory's `return nl`...``, a field typed as a function, an argument whose
parameter is a function); a factory without a declared result takes its signature from how the functions it makes are
called.

## A rule and a judgment behind one setting: `pluggable`

Where an exact rule and a judgment both answer one question, give them one interface and let a setting choose.

<!-- from: examples/refunds/rules.ts -->
```ts
export function eligibility(setting?: PluggableSetting) {
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
  const repaired = await semanticRepair(draft, {
    check: text => words(text) <= MAX_SUMMARY_WORDS ? [] : [`Summary has ${words(text)} words; shorten to at most ${MAX_SUMMARY_WORDS} words.`],
    fix: shorten,
    attempts: 3,
  });
```

Use `iterateOn` (or `semanticRepair` / `fixedPoint`) when each step depends on the previous state. Independent items are
a `semanticMap` or `Promise.all` map.

## Collections, policy, convergence & workspaces: `@natlang/node/std`

The standard library (`@natlang/node/std`, in the browser `@natlang/browser/std`) provides typed TypeScript combinators
around named `.nl` judgments shipped with the package (`meets`, `relates`, `beats`, `intentOf`, `compatible`, `combine`,
`settled`, `compileRuleFn`, `shouldEscalate`, `stepKind`). `instructionsOf(source)` reads a `.nl` source as its front matter, declaration
and steps, and `stepKind(step, signature, file)` says what one step asks for (`judgment`, `mechanism`,
`guard-for-type` or `redundant`): tools that review or rewrite programs read them this way. Every shipped judgment is an ordinary natlang function you can call, `decide`, `.in`, `.on`,
trace and specialise. Every combinator nests its child calls under a `std:<name>` span and passes values by reference so
`Neuralese<T>` states flow untouched.

- **Collections**: `semanticMap(items, fn, { concurrency? })`, `semanticFilter(items, criterionOrFn, { floor? })`,
  `semanticPartition(items, classifier, { floor? })` (`{ groups, uncertain }`),
  `semanticReduce(items, init, step, { mode?: 'linear' | 'tree', combine? })`,
  `semanticMatch(left, right, { relation?, match?, block?, floor? })`, and
  `semanticRank(items, goal, { debiased?: true })` (concurrent symmetric pairwise `beats` + exact Copeland scoring).
- **Policy & higher-order**: `escalate(fn, { to, floor?, risk?, policy? })` (confidence floor + asymmetric per-label risk
  floors, recording `decision_escalate`; on a runtime without the model `to` the local answer stands with `escalated: false`
  and a `reason` starting `unavailable:`; `policy: 'nl'` asks the shipped `shouldEscalate.nl`, which reads the function's
  own escalation history (runs, escalation rate, how often the fallback disagreed) and measured tier cost from the call
  store; `history`/`cost` override them, `false` leaves them out; each run leaves an `escalation` note on its local call), `withContext(fn, entries)` (`fn.in(Context.ofCallable(fn).with(entries))`, keeping
  companion helpers intact), and `compileRule(rule, { examples?, check?, attempts? })` (synthesizes and example-verifies
  a reusable judgment function).

`examples/pr-review` composes `withContext`, `semanticMap`, `semanticRank`, and asymmetric `escalate` over untrusted PR
diffs with `ensures:` verbatim quote checks:

<!-- from: examples/pr-review/review.ts -->
```ts
export async function reviewPullRequest(hunks: Hunk[], rules: Rule[]): Promise<ReviewReport> {
  // Bind check_hunk once per rule while keeping its protected `quotesMatch` verifier in context.
  const checks = rules.map(rule => ({
    rule,
    run: withContext(checkHunk, { rule: `${rule.name}: ${rule.guidance}` }),
  }));
  const pairs = hunks.flatMap(hunk => checks.map(check => ({ hunk, check })));
  const reviews = await semanticMap(pairs, ({ hunk, check }) => check.run(hunk), { concurrency: 6 });
  const rawFindings: Finding[] = reviews.flatMap(review => review.findings);
  const findings = rawFindings.length > 1
    ? (await semanticRank(rawFindings, 'most critical engineering risk to fix before merging')).ranked
    : rawFindings;
  // Asymmetric safety gate: escalate when unsure (< 0.85) OR when P(block) >= 0.25.
  const decideMerge = escalate(mergeVerdict, { to: 'teacher', floor: 0.85, risk: { block: 0.25 } });
  const decision = await decideMerge(findings);
```

`semanticRepair(value, { check, fix, attempts })` runs an exact check (a compiler, a schema, tests) and hands what it
finds to a fix until the check passes. The release desk drafts its notes, then repairs them against rules written in
TypeScript:

<!-- from: examples/release/notes.ts -->
```ts
export function notesProblems(notes: Notes, changes: Change[]): string[] {
  const problems: string[] = [];
  for (const change of changes)
    if (!notes.some(line => idOf(line) === change.id)) problems.push(`No line starts with ${change.id} (${change.title}).`);
  ...
  return problems;
}
```

<!-- from: examples/release/notes.ts -->
```ts
  const first: Notes = await nl`Write one line of release notes per change in changes: the change's id, then what it means for a user of the project.`();
  const fix = nl<(notes: Notes, problems: string[]) => Notes>`Rewrite notes so that none of problems applies any more. Keep every line that no problem names.`;
  const repaired = await semanticRepair(first, { check: notes => notesProblems(notes, changes), fix, attempts: 3 });
  return { notes: repaired.value, problems: repaired.problems };
```

The result is `{ ok, value, problems, attempts, history }`. A fix that returns a value seen before ends the repair, and
after `attempts` fixes `ok` is false and `value` is the value with the fewest problems, so the program decides what to
do with it.

`fixedPoint(step, initial)` applies a step to its own output until a revision no longer changes what the value says.
The shipped `settled` judgment (`readout: decision`) makes that call after each step, so the loop needs no step bound;
`{ settled: (previous, current) => boolean, maxSteps }` replaces it with an exact test.

<!-- from: examples/release/notes.ts -->
```ts
  const first: Summary = await nl`Write the paragraph that tells a user of the project what changed, from notes.`();
  const revise = nl<(summary: Summary) => Summary>`Revise summary so that it states every change in notes that a user notices, and nothing that notes do not say.`;
  const { value } = await fixedPoint(revise, first);
```

`semanticMerge(base, ours, theirs, options)` merges two edits of one value. Records merge field by field: a field only
one side changed takes that edit. Where both changed a field, `intentOf` states what each edit is for, `compatible`
scores whether one value can keep both, `combine` writes it, and the result must be the same kind of value and pass
your `check`. Everything else is listed in `unresolved` with both values, both intents and the reason, and the merged
value keeps `base` there (`keep: 'ours' | 'theirs'` to change that).

<!-- from: examples/release/merge.ts -->
```ts
export function mergeRelease(base: Release, ours: Release, theirs: Release): Promise<Merge<Release>> {
  return semanticMerge(base, ours, theirs, {
    check: (value, path) => path[0] === 'version' && !/^\d+\.\d+\.\d+$/.test(String(value)) ? [`${String(value)} is not a version number such as 2.4.0.`] : [],
  });
}
```

## Change files through a transaction

`folder.propose(reducer, ...args)` runs the reducer on a private copy of the folder and returns a proposal: the new
`folder` (a snapshot), the reducer's `value` and the `diff`. Nothing changes until `folder.accept(proposal)`, and
`accept` refuses a proposal whose base changed meanwhile (`FolderConflictError`). Check the proposal in code first.

<!-- from: examples/refunds/ledger.ts -->
```ts
  const proposal = await folder.propose(async (draft: FolderHandle) => {
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

## Verified and selected changes

`proposeAndVerify(folder, reducer, { verify, attempts, args })` is a transaction with a verifier and a repair loop:
the reducer writes a proposal on a private copy, `verify` (an exact check the reducer cannot edit) lists its
problems, and a failed proposal is proposed again from its own draft with the problems as the reducer's last
argument. Only a proposal with no problems is accepted.

<!-- from: examples/release/publish.ts -->
```ts
export async function publish(folder: Folder, version: string, notes: Notes, breaking: Change[] = []) {
  const priorChangelog = await folder.file('CHANGELOG.md').exists() ? await folder.file('CHANGELOG.md').readText() : '';
  return proposeAndVerify<string>(folder, writeReleaseDocs, {
    args: [version, notes, breaking],
    attempts: 2,
    verify: proposal => releaseDocProblems(proposal, version, priorChangelog, breaking),
  });
}
```

<!-- from: examples/release/publish.ts -->
```ts
  const migration = folder.file('MIGRATION.md');
  const previous = await migration.exists() ? await migration.readText() : '';
```

`branchAndSelect(folder, branches, { verify, goal })` runs several reducers, each on its own copy, drops the ones
`verify` rejects, and accepts one of the rest: the one your `select(candidates)` returns, or with a `goal` the winner
of pairwise `beats` decisions over each branch's result and changed files, or else the first.

<!-- from: examples/release/publish.ts -->
```ts
  return branchAndSelect<string>(folder, [
    { reducer: inStyle('plain sentences, most important change first'), args: [notes] },
    { reducer: inStyle('a bulleted list, one bullet per note'), args: [notes] },
  ], {
    verify: async proposal => (await proposal.folder.readText('ANNOUNCEMENT.md')).trim() ? [] : ['The announcement is empty.'],
    goal: 'a user who is about to upgrade learns first what will break and what to do about it',
  });
```

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

## Recurse on the data; hold every result to its contract

A nested input (a document's sections, a thread's replies, a spec's sub-specs, a directory tree) is handled one
level per call: the function calls itself on each part and assembles the results. `recursive: structural` lets it,
and its `ensures:` states what every result must satisfy in terms of the argument, so each level is checked before its
caller sees it:

<!-- from: examples/outline/digest.nl -->
```yaml
---
description: The bottom-up digest of a section integrated with its subsections.
args:
  section: Section
returns: Digest
recursive: structural
ensures:
  - check: "covers(result, section)"
    message: the digest keeps the section's title and has one part per subsection, in order
  - predicate: a digest whose gist states the main point of the section's own text and integrates any key findings from its subsections in one sentence
    against: [section]
---
```

The call sees `digest` by name and runs `Promise.all(section.sections.map(child => digest(child)))`: each
subsection is part of its argument, so each self-call is admitted, and a self-call on anything not smaller is refused
when it happens. The `check` runs first, in code: `covers` is a helper in `digest/`, and naming it in a check protects
it from the call's own edits. The `predicate` is judged only after the check holds, with `section` shown to the judge
beside the result (its `Untrusted` text fenced). A result that fails either goes back to the call as `Revise the result
so that the digest keeps the section's title ...`, and the call repairs it. When what shrinks is a count rather than a
part (a position that advances through a fixed list), declare `measure: "items.length - done"` instead.

## Neuralese between stages

A chain of typed stages can carry its intermediate values as Neuralese (compressed model state) instead of text,
without a change to the source: [fusion](fusion.md) says how to write chains the runtime can fuse. A function whose
result some callers read as text and others keep as Neuralese declares `generic: { R: string | Neuralese<string> }`
and `returns: R`; the builtin `view(value, instructions?)` is one ([language](language.md)).

## From judgment to code: the ladder

A function starts as instructions and moves toward code as evidence accumulates, with the function staying the
specification:

1. **A scored decision.** `readout: decision` on a finite result: reasons, then scores every choice with calibrated
   probabilities through `decide` (`readout: instant` skips reasoning on latency-critical hot paths).
2. **Compiled cases.** Every call is recorded (`natlang traces`). `natlang specialize --definition NAME` compiles a
   function's recorded calls into guarded crisp TypeScript cases; the runtime loads them by itself, starts a new case
   in shadow, promotes it on evidence, and hands any input no case admits (or a case that fails) back to the model.
   `natlang compilations list`, `show` and `why CALL` explain what served a call
   ([`examples/specialization`](https://github.com/werg/natlang/tree/main/examples/specialization)).
3. **Escalation.** Below a confidence floor or risk floor, `escalate` (from `@natlang/node/std`) asks a stronger model
   and records each disagreement on the local call. Save hard cases with `natlang traces fix CALL_ID --expected JSON`
   and inspect call-store metrics with `natlang profile`.
4. **Tuning.** `natlang tune` trains the local model's weights on the program's `examples.json` and activates a
   release; `--rollback` returns to the previous one ([tuning](tuning.md)).
5. **Improvement.** `natlang improve DIR --model MODEL` diagnoses failures, escalations, disagreements, and fixed
   traces from the call store, rewrites only instructions (front matter byte-identical), checks the change on
   deterministic `support`/`dev`/`confirmation` splits, and installs accepted changes with a byte-identical rollback
   record (`--proposal FILE` checks a patch you wrote; `natlang improve adopt-call CALL_ID` checks and installs a single
   call's `edit_code` fix; see [neuralese](neuralese.md) and [adaptation](adaptation.md)).
