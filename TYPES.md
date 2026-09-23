# natlang types and validation

The normative rules are in [`spec/SPEC.md`](spec/SPEC.md). This page summarizes
them for authors and training-data builders.

## Source types

Signatures use TypeScript types: `string`, `number`, `boolean`, `null`,
records, arrays, `Record<string, T>`, literal unions, optional fields and
parameters, aliases, and `Folder`. Named functions declare them in frontmatter;
inline `nl` calls get them from the TypeScript checker.

```ts
type Ticket = { id: string, label: "urgent" | "normal" | "spam", confidence: number };

const label: Ticket["label"] = await nl`Label ticket by urgency.`(ticket);
```

```yaml
args:
  tickets: "Ticket[]"
  rubric: string
returns: "Ticket[]"
```

A folder's `types.ts` supplies aliases (`export type Name = ...;`) to the
functions in and below it. Host values that are not data (class instances,
functions, DOM nodes, native handles) cross by reference; a signature can name
them with `Live<"T", "tag" | "class" | "shape" | "function" | "any", "detail">`,
and the compiler maps TypeScript classes and built-ins to these contracts.

## Validation

Validation is structural. Arguments are checked when a function is called; the
model's result is checked after each eval and at completion. A partial record
can be built incrementally, but completion requires the declared type. Simple
scalar mistakes (a quoted number) may be coerced when the intended type is
unambiguous. A valid shape does not establish semantic truth, provenance, or
effect completion.

## Inline signatures

The compiler infers an inline call's signature from `nl<F>`, the contextual
type, an immediate call, or later uses of a local, and reports:

| Code | Meaning |
|---|---|
| `nl-unknown-return` | Nothing determines the result type; annotate or use `nl<T>` |
| `nl-ambiguous-signature` | Uses disagree about the signature |
| `nl-unknown-parameter` | A parameter's type cannot be determined |
| `nl-sync-callback` | The call is used where a synchronous function is required |
| `nl-unknown-name` | The instructions name something that is not visible |

## Effects and files

Services, live-object writes, and folder commits are the ways a call changes
the world; see the effect table in the
[integration skill](skills/natlang-integration/references/recovery.md).
Filesystem access is limited to directory reducers, whose paths are relative to
their input folder.
