# Language and source contracts

Natlang source is TypeScript plus natural-language functions. Resolve details against the installed declarations, `natlang check`, and the runtime's tests.

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

- In eval code, ``await nl(`Is ${x} large?`)`` is a one-shot call, the same as ``nl`Is ${x} large?`()``; the instructions must be a literal the compiler can see.
- The compiler plans every `nl` expression before the model runs: parameters, return type, and captures. Conflicting cases are compile errors (`nl-ambiguous-signature`, `nl-unknown-parameter`, `nl-sync-callback`); a result nothing types runs open (any value, shaped by the fields the code reads). Use `nl<T>` or an annotation to pin a type: the call is then asked for exactly that.
- Parameters come from a function-type annotation, the callback slot, an immediate call, or calls later in the same compilation unit. A saved `nl` with none of these (`const judge = nl<boolean>`...`` called only later, typical of a REPL eval) accepts whatever each call passes, as `input`, `input2`, …: name the parameters with a signature, `nl<(application: Application) => boolean>`, so its instructions and its arguments agree.
- Captures are exact-name mentions of visible bindings. They are read live at each call, not frozen at the tag. A mentioned `let` can be reassigned by the model; the write-back is version-checked.
- Template interpolations `${expr}` are evaluated at call time and become part of the instructions.
- An inline `nl` in application code may call the items of the nearest `natlang.d/` folder above it. Inside a callable folder it sees that folder's items. With no context it has only its inputs and captures.
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

- Frontmatter keys: `description`, `args`, `returns`, `types`, `kind` (`function` or `directory-reducer`), `readout`, `model`.
- `model: NAME` runs every call of the function on the runtime's model of that name (the `models` runtime option) and on the default model when the runtime has none: a small model for quick judgments, a large one for the work that needs it. Programs stay runnable on one model. Quote YAML type strings that contain record syntax or YAML punctuation.
- `readout: decision` on a function whose result is finite (a literal union, `boolean`) makes each call one scoring pass instead of a tool loop: the model scores every allowed value and the most probable is returned; the distribution is in the trace. Host code gets it with `runtime.decide(fn, ...args)`, and eval code with `decide(fn, ...args)`: `{ value, probabilities: [{ value, probability }], confidence, scored }`, so instructions can act on a floor ("refuse when destructive has probability at least 0.7"). Use it for fast typed decisions (classify, gate, route, judge).
- A named function may call exactly the items of its companion folder (`review/` beside `review.nl`), including their children as properties (`summarize`, `helpers.normalize`). It cannot reach other natural-language functions elsewhere in the project; that scoping is enforced. Items of one folder do not see each other: a helper that two items need lives in the folder of the item that calls it (`bash/risk.nl` for `bash.nl`).
- `types.ts` in a folder supplies aliases to the functions there and below; a named function also sees the `types.ts` of each enclosing directory up to its package root (`natlang.json` or `package.json`), nearer winning. Doc comments on fields stay with the alias and reach the model in the call's opening: describe a data format there (where files live, units, what null means) and pass the typed value as an argument, instead of asking the model to read a format document first.
- Host TypeScript imports a named function directly: `import review from './review.nl'`. `natlang build` generates `review.d.nl.ts` so the import is typed, with its children as typed attributes.
- `natlang.d/` follows the same rules and is the callable context for inline `nl` in application code below it (nearest wins, no merging).
- Child names must be identifiers and must not collide with function properties (`call`, `apply`, `bind`, `name`, `length`, `prototype`, `then`, `iterateOn`, and similar).

## The program's code, and what it only calls

A callable folder holds code the program owns: helpers it was given and code it may fix. The interpreter can read all of it with `read_code` and edit it with `edit_code`, and it does. Something that stands for the world outside the program (a store, a ticket board, a simulated environment, a verifier, an API) is a service instead: the model sees and reads its TypeScript declaration and calls it, but its implementation runs in the host and cannot be changed. Put such a system in a callable folder and a stuck run will edit it: in teacher data every run whose board refused a move rewrote the board module to accept it.

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

- A default-exported function makes the module itself callable; named exports become callable attributes; exported values become typed values.
- Imports are limited to sibling items (`./x.nl`, `./x.js`, `./folder/x.js`), declared npm packages, `natlang:services`, and `@natlang/node` / `@natlang/browser`. Local files outside the folder are rejected.
- The finite-iteration policy applies, and a function may call itself only on a smaller argument. `eval` and `Function` are rejected.
- Ordinary application TypeScript outside callable folders has none of these restrictions.

## Types

Use ordinary TypeScript types in signatures: `string`, `number`, `boolean`, `null`, records, arrays, `Record<string, T>`, literal unions, optional fields and parameters (`?`), aliases, and `Folder`. Values are checked structurally at call boundaries and at completion; a partial record can be built incrementally but completion requires the declared type. A well-typed result can still be semantically wrong.

Live values (functions, class instances, DOM nodes, native handles) are passed by reference and shown to the model as live bindings; a host contract can name them with `Live<"T", "tag" | "class" | "shape" | "function" | "any", "detail">`.

## What the interpreter does

The model carries out the instructions for one call. Work that takes only reading and judgment it answers directly; for computation, data work, and calls it uses a persistent TypeScript eval scope, where parameters (`const`), captures, callable items (as `helper(...)` and `folder.child(...)`), and services are bindings; the call's opening names them and declares them with their doc comments. It finishes with `return_result` and a status: `success` with a value of the declared type, `blocked` with a reason when information the instructions point to is missing, `failed` with a reason when they require an invalid operation. It can also return a value from an eval (`return value;`, staged) and reply done, or, for a string result, reply with the text. It may read callable items, service declarations, and packages with `read_code`, and edit its own callable items with `edit_code` and `diff_code`. Output too long to show is cut off with a `<<cut off: …>>` note naming the variable that holds all of it; `read_page` shows the rest of a long text. In a long call the model compacts its conversation into a note (`compact_history`), and `transcript.search(…)` / `transcript.entry(n)` in eval recover any earlier call in full. Directory reducers additionally get file tools; the folder changes present when they finish are kept. Write instructions so the next meaningful action is apparent from them and the typed scope.

## Iteration

```ts
import { iterateOn } from '@natlang/node';
const final = await iterateOn(shorten, draft).withLimit({ maxSteps: 5 }).until(text => words(text) <= 60);
const plan = await step.iterateOn(initial).until(done);    // every natural-language function has .iterateOn
```

`until` resolves to the first state that satisfies the predicate. `streamUntil` yields each step; `onStep` observes; `withSiteId` keys per-site statistics; `checkProgress` replaces the progress judge, which answers `continue` or `divergent`. Failures reject with `IterationDivergedError`, `IterationStepError`, or `IterationLimitError`, each carrying `lastState` and the trajectory.

## Repository anchors

`ts-host/src/compiler/` (planning, policy, lowering), `ts-host/src/runtime/` (kernel, callables, loader, iterate), `ts-host/src/native/` (interpreter and eval), `ts-host/test/{compiler,runtime,project,interpreter}.test.mjs`, `applications/`, `examples/triage/`, `codebases/`, and `spec/SPEC.md`.
