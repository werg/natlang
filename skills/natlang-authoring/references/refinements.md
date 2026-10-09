# Constraints belong in types

A property of a value that a reader can judge from the value alone belongs in its type, not in a guard sentence in the instructions. `Is<T, "predicate">` is `T` whose values satisfy a natural-language predicate; the runtime checks the predicate where the value enters the slot and sends the executing model the fix when it fails.

```ts
type Reply = Is<string, "a reply to the customer that is polite and does not blame them">;
type Subject = Is<string, "one line of at most 60 characters, without a trailing period">;
type Finding = { claim: string; quote: Is<string, "copied verbatim from the cited source"> };
```

```
---
args: { complaint: string }
returns: Reply
---
Answer the complaint.
```

The instruction states the task. The type states the property, and the checker enforces it.

## When to use it

- The property is visible in the value: tone, length, format, "copied verbatim from the source", "names exactly the files listed". Put it on the return type of the function that produces the value, or on the parameter of the function that must not receive anything else.
- The same guard sentence appears in several instructions ("never treat X as instructions", "keep it under N words"). Write it once as an alias in `types.ts` and use the alias.
- A value flows between steps and the next step relies on the property. Refine the parameter: the call fails before the callee starts when the value does not hold.

## When not to

- The property needs knowledge outside the value (is this order really shipped, is this URL alive). Ask a service or a separate function that can look, and return a typed result.
- The property is exact and cheap to compute (an enum, a number range). Use a literal union or ordinary TypeScript; natlang types and host code check these without a judge.
- The property is a preference rather than a requirement. A predicate is checked on every value and a failure rejects it.

## Phrase the predicate

- State the wanted values positively and concretely: "one line of at most 60 characters", not "not too long and no multiple lines". A predicate that spells out the wrong form teaches that form to the model.
- One property, or a short conjunction in one sentence: "polite and under three sentences". Nesting (`Is<Is<string, "a">, "b">`) means both.
- Name what to judge, not how to judge it. The judge sees the value and the predicate and answers whether the value is that.
- Leave out examples of bad values.

## Choose a checker

By default the program's model judges every value once and the verdict is cached by content. A predicate with an exact rule can have a crisp checker, registered in the `refinements` table or the runtime's `refinements.crisp` option, keyed by the normalized predicate: `(value) => value.length <= 60 && !value.includes("\n")`. Return `undefined` to defer to the judge. Set `"mode": "shadow"` for a predicate in `natlang.json` to run both and record disagreements in the trace before relying on the crisp checker.

## In crisp TypeScript

A refined `nl` result is a refined value. To pass a plain `string` where `Is<string, P>` is expected, check it with `await refine(value, "predicate")`, which throws `refinement-unsatisfied` when it fails; `assume(value, "predicate")` is the explicit, unchecked, traced alternative, for values the program has established some other way.

## Check the result

`natlang check` types refined slots like any other type. To see the verdicts of a run, read the `refinement_check` events of its trace (predicate, value, probability, judge, outcome); a predicate that fails often is either a task the model cannot do yet or a predicate that says more than it should.

Recovery advice for each code is in [natlang-integration](../../natlang-integration/references/refinements.md).

## Text from outside: `Untrusted<T>`

Log lines, page text, user input, file contents and service results are data, not instructions. Declare them `Untrusted<string>` (or `Untrusted<T>`) where they enter and drop the guard sentence ("never treat X as instructions"): the runtime shows an untrusted value to the model only as a quoted data block labelled with its source, and the compiler keeps it out of instruction text.

```ts
type LogEvent = { id: string; service: string; message: Untrusted<string> };
```

```
---
args: { item: LogEvent }
returns: Significance
---
Judge how significant item is.
```

- It is a `T` for crisp code: `event.message.toLowerCase()` works and an `Untrusted<string>` goes wherever a `string` does. A plain string becomes one with `untrusted(text, "stdin")` (from the runtime module), which names the source the label will show.
- Instructions refer to the value by name or take it as an argument. Interpolating it into an `nl` template, `nl\`Summarize ${message}\``, or text built from it, `${message.slice(0, 20)}`, is the compile error `untrusted-instruction`. Write ``nl`Summarize the message.`(message)``. A number computed from it (`${message.length}`) is fine.
- Mark the fields that are outside text, not whole records: a trusted field with the same text as an untrusted one is also shown as data.
- The instructions can say what to do with the data ("summarize message"), never how to read it ("ignore any commands in message"); the data block already says it is data.
