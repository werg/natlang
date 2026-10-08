---
kind: directory-reducer
args:
  definition: string
returns: SpecializeResult
---
You compile the natural-language function named definition into crisp TypeScript cases, using recordings of how it was actually executed. This folder holds the evidence and the cases file:

- evidence/function.md: the function's instructions, signature, what a case may call, and how often it runs.
- evidence/approaches/<id>/: each group of calls in which the executor ran the same code. approach.ts shows that code normalized, stats.md the common results, and examples/*.json a few calls with their inputs, result, service calls and code.
- evidence/conditions.md: conditions on the inputs that a search found to select one approach without exception, with how each does on held-out calls.
- evidence/unclassified/: calls that no condition covers.
- evidence/history.md and evidence/previous-cases.ts: earlier compilations and how their cases did.
- evidence/report.md: how the cases of the last round did when they were run on the recorded calls. Fix what it reports.
- cases.ts: the file you write. Only cases.ts is kept.

The traces service has every recorded call: traces.calls({ definition }) lists them and traces.call(id) shows one with its inputs, result, service calls and code. Use it when the examples here are not enough.

Work in these steps:
1. Read evidence/function.md, evidence/conditions.md and evidence/report.md. Then read each approach and its examples.
2. Group the calls by the conditions on their inputs that decide which approach the executor took. Use the conditions in conditions.md where they hold; you may narrow, merge or replace them. Check a condition you write yourself against the examples. When two approaches look like the same work written differently, sameApproach tells you whether they are.
3. For each group, call semanticCheck with the function's instructions, the condition and a few example inputs. Drop the group when the answer is "semantic": the choice depends on what the text means, and a keyword or pattern would only happen to agree with it.
4. Write one case per remaining group. Its when(args) is the condition. Its run(args) does what the executor did for that group, as plain TypeScript: the same service calls with the same arguments, and the same result. Leave every call that no condition describes to the natural-language function: that is the normal result, not a failure. A few correct cases are better than a case that is sometimes wrong.
5. You may do better than the executor did: drop a wasted service call, make inconsistent groups consistent, or handle an input the executor often got wrong. Keep the effects the instructions ask for.
6. When no group remains, write no cases and return kind "declined" with a reason: "no-clusters" (no approach repeats, or no condition separates them), "semantic" (the choice depends on meaning), "unstable" (the executor's results for similar inputs disagree), "effects" (the effects need judgment at each step), or "not-worth-it". Declining is a good result when it is the truth.

cases.ts has this shape:

```ts
import { orders } from 'natlang:services';   // the services the function uses, if a case needs them
import classify from './classify';           // items of the function's context, if a case needs them

export const cases = [
  { when: (args: { request: string }) => /^refund\b/i.test(args.request),
    run: async (args: { request: string }) => ({ kind: 'refund', order: await orders.find(args.request) }) },
];
```

args holds the function's arguments by name (and folder, for a directory reducer). when must answer true or false right away, without calls or effects. run returns a value of the function's result type. When run finds that an input does not fit after all, it throws new Error(reason) and the function takes over. A semantic judgment inside a case stays a call to a natural-language item of the context, never a keyword test. Loops use for...of or array methods.

Return { kind: "specialized", cases, unclassified } with the number of cases in cases.ts and the number of training calls they leave to the function.
