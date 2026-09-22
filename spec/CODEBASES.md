# Source functions and imports

The normative source and execution contract is in [`SPEC.md`](SPEC.md).
Programs are graphs of named typed functions. Natural-language functions use
frontmatter signatures and line-by-line instructions. Crisp helpers are
ordinary TypeScript modules with default-exported functions:

```ts
export default function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}
```

Function parameters and return values use ordinary TypeScript data types,
including `string`, `number`, `boolean`, `null`, records, arrays,
`Record<string, T>`, aliases, and literal unions. Structural validation checks
values at call boundaries and completion. It does not prove domain semantics.

Imports give functions stable names. A function sees its own imports and
companion helpers rather than inheriting the caller's lexical dependencies.
Natural-language and crisp functions share ordinary awaited positional call
syntax, for example `await inspect(event, policy)`. Use standard TypeScript
loops and array methods for collection work.

Natural-language functions execute in persistent TypeScript scope through
`eval`. `read_value` inspects that scope, and `mark_lines` closes each completed
instruction line. Function tools inspect and edit imported function source.
Those tools cannot create, delete, move, or rename functions.

Directory reducers are the only functions with model-facing file tools. They
receive a writable copy of an input folder and use paths relative to that
folder. A direct `await reducer(folder, ...args)` returns the typed result and
discards file changes. `await folder.apply(reducer, ...args)` retains the
reducer's selected changes. See [`SPEC.md`](SPEC.md#directory-reducers) for
commit and folder details.
