# natlang types and validation

The normative type rules and diagnostics are in [`spec/SPEC.md`](spec/SPEC.md).
This document summarizes the current clean-break TypeScript contract for
authors and training-data builders.

## Source types

Function signatures use standard TypeScript types:

```ts
type Ticket = {
  id: string;
  label: "urgent" | "normal" | "spam";
  confidence: number;
};

export default function classify(inbox: string[], rubric: string): Ticket[] {
  // crisp helper body
}
```

Supported data shapes include `string`, `number`, `boolean`, `null`, records,
arrays, `Record<string, T>`, named aliases, and literal unions. Ordinary crisp
helpers are default-exported TypeScript functions. Natural-language functions
declare typed parameters and a return type in their frontmatter; imported
natural-language and crisp functions use ordinary awaited positional calls.

Validation is structural. Function inputs, crisp returns, committed eval
transactions, and completed natural-language results must fit their declared
types. A valid shape does not establish semantic truth, provenance, or effect
completion. Applications still need domain-specific checks and real operation
receipts where those claims matter.

## Natural-language execution

Natural-language functions run in a persistent TypeScript scope. `eval(code)`
executes ordinary TypeScript declarations, assignments, branches, loops, and
function calls. `read_value(expression, start?, end?)` inspects scope values.
Every substantive instruction line is closed after it succeeds with
`mark_lines(start, end?, skipped?)`. `report_blocker` and `report_error` provide
explicit exits for missing information and invalid work.

The final eval expression that fits the declared return type supplies the
function result. The result and all substantive instruction lines must be
complete before the call finishes. Imported helpers are ordinary names in
scope; use `await helper(value, option)` and standard TypeScript collection
control flow such as `for...of`, array methods, and `Promise.all`.

Normal lambdas have function tools for inspecting and editing imported
functions. The fixed function set cannot be created, deleted, moved, or renamed
by those tools. Inspect and improve existing function source when it helps the
task.

## Files and reducers

Filesystem access is limited to directory reducers. A reducer operates on a
writable copy of its input folder, and every file path is relative to that
folder. It can create, edit, move, and remove files through its file tools and
folder filesystem API.

Calling `await reducer(folder, ...args)` returns the typed value and discards
file changes. Calling `await folder.apply(reducer, ...args)` retains the
reducer's selected changes in the folder. Ordinary functions do not receive
model-facing file tools or ambient workspace access.

## Host effects

Native APIs, databases, processes, binary objects, and other host-owned values
stay on the host side. Expose narrowly typed operations through crisp helpers
or declared effect callbacks. A failed acknowledgement does not prove an
external effect did not happen; preserve operation identity and observed
receipts before retrying.
