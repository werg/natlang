# Language and source contracts

Natlang source is TypeScript plus natural-language functions. The rules live in the [specification](../../../spec/SPEC.md); this page says how to write against them, with one-line reminders and links. Resolve details against the installed declarations, `natlang check`, and the runtime's tests. Error codes, with when each fires, are in the [error index](../../../spec/ERRORS.md).

## Inline natural language

```ts
import { nl } from '@natlang/node';          // '@natlang/browser' in browser code

export async function triage(ticket: Ticket, style: string): Promise<Report> {
  // Immediate call: the arguments become parameters; the annotation is the return type.
  const priority: 'urgent' | 'normal' = await nl`Decide whether ticket is urgent.`(ticket);

  // A named callable: the declared function type is the signature.
  const summarize: (text: string) => Promise<string> = nl`Summarize text in one sentence, in the given style.`;

  // Captures: `style` is mentioned, so it is read live when the call runs.
  return { priority, note: await summarize(ticket.text) };
}
```

- Signatures, eval one-shots and interpolation: [Natural-language functions](../../../spec/SPEC.md#natural-language-functions). In eval code, ``await nl(`Is ${x} large?`)`` is a one-shot call; the instructions must be a literal the compiler can see.
- The compiler plans every `nl` expression before the model runs; conflicts are compile errors (`nl-ambiguous-signature`, `nl-unknown-parameter`, `nl-sync-callback`). Use `nl<T>` or an annotation to pin a type: the call is then asked for exactly that.
- Parameters come from a function-type annotation, the callback slot, an immediate call, or calls later in the same compilation unit. A saved `nl` with none of these (`const judge = nl<boolean>`...`` called only later, typical of a REPL eval) accepts whatever each call passes, as `input`, `input2`, …: name the parameters with a signature, `nl<(application: Application) => boolean>`, so its instructions and its arguments agree.
- Captures are exact-name mentions of visible bindings, read live at each call ([Captures](../../../spec/SPEC.md#captures)). A mentioned `let` can be reassigned by the model; the write-back is version-checked.
- An inline `nl` sees the nearest `natlang.d/` folder (application code) or its own callable folder; with no context it has only its inputs and captures ([Contexts](../../../spec/SPEC.md#contexts)).
- Build with `natlang build` (or `buildProject`) so `nl` is lowered; calling an uncompiled `nl` fails with an actionable error.

## Named functions and callable folders

```text
review/
  types.ts
  review.nl
  review/
    assess.nl
    summarize.ts
```

```yaml
---
description: Assess each observation and count the verdicts.
args:
  observations: string[]
  criterion: string
returns: Report
---
Assess every observation against the criterion with assess, then summarize the assessments.
```

- Frontmatter keys (`description`, `args`, `returns`, `generic`, `types`, `kind`, `readout`, `model`, `uses`): [Natural-language functions](../../../spec/SPEC.md#natural-language-functions). Quote YAML type strings that contain record syntax or YAML punctuation (the specification says types are read verbatim, so quoting is not required; see its Known discrepancies).
- A function whose result some callers read as text and others as Neuralese declares it generic: `generic: { R: string | Neuralese<string> }` with `returns: R`. Each call runs the form its result is used as (`const block: Neuralese<string> = await gist(x)` writes a block; `const text = await gist(x)` gets text), from one body ([Generic results](../../../spec/ext/neuralese.md#generic-results)). Hosts pick the form with `invokeAt(fn, args, { kind: "neuralese" })`. The runtime's own summarizer is such a function, the builtin `view(value, instructions?)`: faithful without instructions, what their purpose needs with them; a call's opening listing shows the Neuralese view of an argument it would cut off ([View](../../../spec/ext/neuralese.md#generic-results)). Do not write a separate digest or summary function for this; to ask a question of a block, `ask(block, question)` from `natlang:neuralese` is `read(map(block, question))`.
- `model: NAME` picks the runtime's model of that name for every call of the function, and the default model when there is none. Use a small model for quick judgments and a large one for the work that needs it; programs stay runnable on one model.
- `readout: decision` ([Model surface](../../../spec/SPEC.md#model-surface)) fits a function with a finite result (a literal union, `boolean`): one scoring pass instead of a tool loop. Use it for fast typed decisions (classify, gate, route, judge). Host code gets the distribution with `runtime.decide(fn, ...args)` and eval code with `decide(fn, ...args)`, so instructions can act on a floor ("refuse when destructive has probability at least 0.7").
- A named function calls exactly the items of its companion folder (`review/` beside `review.nl`), including their children as properties (`summarize`, `helpers.normalize`); other functions are out of reach ([Contexts](../../../spec/SPEC.md#contexts)). Items of one folder do not see each other: a helper that two items need lives in the folder of the item that calls it (`bash/risk.nl` for `bash.nl`).
- `uses: [harness/cut, harness/context]` lists package items besides the companion folder, by path from the package root without extension; each is called by its base name (`cut(view, 20000)`). Use it when several functions in different folders need the same item; keep a helper that one function needs in that function's folder. A sibling or other item of a folder that holds the function (`merge/block/refine.nl` with `uses: [merge/block/prose]`) is listed the same way and shares that item's record. A name that collides with a folder item, a missing item, or a function that would reach itself through `uses` is a load error; a cycle error prints the path (`a -> b -> a`), so remove one `uses` entry on it.
- Every call is recorded and may be answered from compiled crisp cases; the function stays the specification ([Call records](../../../spec/ext/call-records.md)). Keep values that must not be stored out of records with `"recording": { "exclude": ["auth/*.nl", "login.password"] }` in `natlang.json`. `natlang traces show CALL` and `natlang compilations why CALL` explain what happened to a call.
- `types.ts` supplies aliases to the functions in its folder and below; a named function also sees each enclosing directory's `types.ts` up to the package root, nearer winning. Doc comments on fields stay with the alias and reach the model in the call's opening: describe a data format there (where files live, units, what null means) and pass the typed value as an argument, instead of asking the model to read a format document first. Two applications share one definition with `export type { Passage } from "../evidence/types.js"` (or `export * from "./more.js"`) in `types.ts`: the loader follows relative re-exports of type aliases, so declare a shared type once and re-export it.
- Host TypeScript imports a named function directly: `import review from './review.nl'`. `natlang build` generates `review.d.nl.ts` so the import is typed, with its children as typed attributes. `natlang.d/` follows the same rules for inline `nl` in application code.
- Child names must be identifiers and must not collide with function properties (`call`, `apply`, `bind`, `name`, `length`, `prototype`, `then`, `iterateOn`, and similar; [Contexts](../../../spec/SPEC.md#contexts)). A file named `read-limits` becomes `readLimits`; a name such as `apply` takes a suffix (`applyStep`); each load error names the rename. In eval and instructions a child function keeps its name, so keep its result under another one (`const planResult = await plan(input)`); reusing the name is an `invalid-binding` error that says so, because `const plan = plan(...)` would read the variable before it exists in JavaScript too.

## The program's code, and what it only calls

A callable folder holds code the program owns: helpers it was given and code it may fix. The interpreter can read all of it with `read_code` and edit it with `edit_code`, and it does. Something that stands for the world outside the program (a store, a ticket board, a simulated environment, a verifier, an API) is a service instead ([Services and effects](../../../spec/SPEC.md#services-and-effects)): the model sees and reads its TypeScript declaration and calls it, but its implementation runs in the host and cannot be changed. Put such a system in a callable folder and a stuck run will edit it: in teacher data every run whose board refused a move rewrote the board module to accept it.

- Give each service a declaration (`serviceDeclarations`, see the integration skill's host reference): its members' types and doc comments are what the model learns the service from. Without one it sees method names only.
- Scope a service to the functions that should use it (`serviceScopes`): a specialist's data, reachable from its calls and the calls they make. The caller can still read every declaration and instruction, and is told who can use it; it asks. This is how to make delegation the way to the evidence. Hiding code is not: never design a program whose correctness depends on the interpreter not reading source, because it reads everything it can reach.
- Importable packages are external too: `read_code("pkg")` lists a package's exports from its type declarations, `read_code("pkg.name")` shows one.
- The built-ins of eval (`nl`, `iterateOn`, `transcript`) are documented, not defined, in the program: `read_code("nl")` shows how to use one.

## Callable-folder TypeScript

```ts
import expand from './expand.nl';                 // a sibling natural-language function
import { wiki } from 'natlang:services';          // a host service

export default async function prepare(goal: string): Promise<string[]> {
  const steps = await expand(goal);
  return steps.filter(step => !wiki.has(step));
}
```

- Module shape, allowed imports and the finite-iteration policy: [Callable-folder TypeScript](../../../spec/SPEC.md#callable-folder-typescript) and [Iteration and termination](../../../spec/SPEC.md#iteration-and-termination). A function may call itself only on a smaller argument; `eval` and `Function` are rejected; application TypeScript outside callable folders has none of these restrictions.
- Local files outside the folder are rejected. Node modules such as `node:crypto` are not importable; standard globals are, so hash with WebCrypto (`await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))`), which makes the function async.

## Types

Use ordinary TypeScript types in signatures; the supported set (records, arrays, `Record<string, T>`, intersections of object types, indexed access with a literal key, literal unions, optional fields, aliases, `Folder`, `Live<...>`) and the checking rules are in [Types](../../../spec/SPEC.md#types). In eval, reading a field that a declared type does not have is rejected before the eval runs (`undeclared-field`), with the type's fields named. A precise service declaration therefore catches the executor's wrong guesses about result shapes. A well-typed result can still be semantically wrong. A partial record can be built incrementally, but completion requires the declared type.

## What the interpreter does

The model carries out the instructions for one call: it answers directly when the work takes only reading and judgment, and otherwise uses a persistent TypeScript eval scope where parameters, captures, callable items and services are bindings, then finishes with `return_result` (`success`, `blocked`, or `failed`). The tools, cut-offs, transcript search and conversation compaction are specified in [Model surface](../../../spec/SPEC.md#model-surface), [Eval](../../../spec/SPEC.md#eval), [Cut-offs](../../../spec/SPEC.md#cut-offs) and [Conversation length](../../../spec/SPEC.md#conversation-length). Directory reducers additionally get file tools ([directory reducers](../../../spec/ext/directory-reducers.md)). Write instructions so the next meaningful action is apparent from them and the typed scope.

## Iteration

```ts
import { iterateOn } from '@natlang/node';
const final = await iterateOn(shorten, draft).withLimit({ maxSteps: 5 }).until(text => words(text) <= 60);
const plan = await step.iterateOn(initial).until(done);    // every natural-language function has .iterateOn
const result = await folder.iterateOn(crispStep, state, policy).withMeasure(s => limit - s.round).until(s => s.done);  // a directory reducer may be a TypeScript function (folder, state, policy)
```

`until` resolves to the first state that satisfies the predicate; `streamUntil`, `onStep`, `withSiteId`, `checkProgress`, the measure/limit requirement and the three failure errors (`IterationDivergedError`, `IterationStepError`, `IterationLimitError`, each carrying `lastState` and the trajectory) are in [Iteration and termination](../../../spec/SPEC.md#iteration-and-termination).

## Repository anchors

`ts-host/src/compiler/` (planning, policy, lowering), `ts-host/src/runtime/` (kernel, callables, loader, iterate), `ts-host/src/native/` (interpreter and eval), `ts-host/test/{compiler,runtime,project,interpreter}.test.mjs`, `applications/`, `examples/triage/`, `codebases/`, and `spec/SPEC.md` with its [extensions](../../../spec/ext/neuralese.md).
