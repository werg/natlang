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

- The compiler plans every `nl` expression before the model runs: parameters, return type, and captures. Unresolvable cases are compile errors (`nl-unknown-return`, `nl-ambiguous-signature`, `nl-unknown-parameter`, `nl-sync-callback`). Use `nl<T>` or an annotation to pin a type.
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

- Frontmatter keys: `description`, `args`, `returns`, `types`, `kind` (`function` or `directory-reducer`). Quote YAML type strings that contain record syntax or YAML punctuation.
- A named function may call exactly the items of its companion folder (`review/` beside `review.nl`), including their children as properties (`summarize`, `helpers.normalize`). It cannot reach other natural-language functions elsewhere in the project; that scoping is enforced.
- `types.ts` in a folder supplies aliases to the functions there and below.
- Host TypeScript imports a named function directly: `import review from './review.nl'`. `natlang build` generates `review.d.nl.ts` so the import is typed, with its children as typed attributes.
- `natlang.d/` follows the same rules and is the callable context for inline `nl` in application code below it (nearest wins, no merging).
- Child names must be identifiers and must not collide with function properties (`call`, `apply`, `bind`, `name`, `length`, `prototype`, `then`, `iterateOn`, and similar).

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
- The finite-iteration and no-recursion policy applies. `eval` and `Function` are rejected.
- Ordinary application TypeScript outside callable folders has none of these restrictions.

## Types

Use ordinary TypeScript types in signatures: `string`, `number`, `boolean`, `null`, records, arrays, `Record<string, T>`, literal unions, optional fields and parameters (`?`), aliases, and `Folder`. Values are checked structurally at call boundaries and at completion; a partial record can be built incrementally but completion requires the declared type. A well-typed result can still be semantically wrong.

Live values (functions, class instances, DOM nodes, native handles) are passed by reference and shown to the model as live bindings; a host contract can name them with `Live<"T", "tag" | "class" | "shape" | "function" | "any", "detail">`.

## What the interpreter does

The model works in a persistent TypeScript eval scope. Parameters, captures, callable items (as `helper(...)` and `folder.child(...)`), and services are bindings; assigning `result` (or a final expression of the declared type) supplies the return value. It closes instruction lines with `mark_lines`, reports `report_blocker` for missing information and `report_error` for invalid work, and may inspect or edit callable items with `read_function`, `edit_function`, and `diff_functions`. Directory reducers additionally get file tools and `commit`. Write instructions so the next meaningful action is apparent from them and the typed scope.

## Iteration

```ts
import { iterateOn } from '@natlang/node';
const final = await iterateOn(shorten, draft).withLimit({ maxSteps: 5 }).until(text => words(text) <= 60);
const plan = await step.iterateOn(initial).until(done);    // every natural-language function has .iterateOn
```

`until` resolves to the first state that satisfies the predicate. `streamUntil` yields each step; `onStep` observes; `withSiteId` keys per-site statistics; `checkProgress` replaces the progress judge, which answers `continue` or `divergent`. Failures reject with `IterationDivergedError`, `IterationStepError`, or `IterationLimitError`, each carrying `lastState` and the trajectory.

## Repository anchors

`ts-host/src/compiler/` (planning, policy, lowering), `ts-host/src/runtime/` (kernel, callables, loader, iterate), `ts-host/src/native/` (interpreter and eval), `ts-host/test/{compiler,runtime,project,interpreter}.test.mjs`, `applications/`, `examples/triage/`, `codebases/`, and `spec/SPEC.md`.
