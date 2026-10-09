---
kind: directory-reducer
args:
  definition: string
  group: string
returns: CaseResult
---
You write one crisp case for one group of recorded calls of the natural-language function named definition. A case is a condition on the function's inputs and TypeScript that does what the executor did for the calls it admits. Calls no case admits stay with the function; that is normal.

This folder holds:

- function.md: the function's instructions, signature, what a case may call, and how often it runs.
- group.md: what the calls of this group did, the code the executor ran for them, their results, and conditions to start from, each with how many of this group's calls and of other groups' calls it admits.
- examples/*.json: calls of this group with their inputs, result, service calls and code.
- others.md: calls of other groups that look like this group's. Your condition must not admit them.
- report.md: how your previous case for this group did when it was run on the recorded calls. Fix what it reports.
- case.ts: the file you write.

group.measure(condition) checks a condition exactly. condition is a JavaScript expression over args, for example `/^refund \d+$/i.test(args.request)`. It returns how many of this group's calls the condition admits, how many calls of other groups it admits, and a few of those.

Work in these steps:
1. Read group.md, then function.md. Open an example when you need its details.
2. Choose a condition that admits this group's calls and no call of another group. Start from group.md. Check the condition with group.measure. It may admit fewer of this group's calls; that is fine.
3. Call semanticCheck with the function's instructions, the condition and a few example inputs. When the answer is "semantic" for a word found anywhere in free text, try a narrower condition that fixes the whole form of the input (an anchored pattern such as /^refund (\d+)$/i), measure it and ask again.
4. When you have a condition that semanticCheck calls "structural" and that admits no other group's call, write case.ts. Its `when` is the condition. Its `run` does what the executor did for these calls: the same service calls with the same arguments, and the same result. You may do better than the executor did: drop a wasted service call, or make inconsistent results consistent. Keep the effects the instructions ask for. Then return { kind: "case", admits } with how many of this group's calls the condition admits.
5. Otherwise write no case.ts and return { kind: "skip", reason, why }. The reason is "semantic" (no condition decides it without understanding the text), "unstable" (the executor's results for these calls disagree in ways the instructions do not explain), "effects" (the effects need judgment at each step), or "no-condition" (no test on the inputs separates this group). Skipping is a good result when it is the truth.

case.ts has this shape:

```ts
import { orders } from 'natlang:services';   // the services the function uses, if run needs them
import classify from './classify';           // items of the function's context, if run needs them

export const when = (args: { request: string }) => /^refund (\d+)$/i.test(args.request);
export const run = async (args: { request: string }) => {
  const number = /^refund (\d+)$/i.exec(args.request)![1];
  await orders.refund(number);
  return `refunded ${number}`;
};
```

args holds the function's arguments by name (and folder, for a directory reducer). when must answer true or false right away, without calls or effects. run returns a value of the function's result type. When run finds that an input does not fit after all, it throws new Error(reason) and the function takes over. A semantic judgment inside run stays a call to a natural-language item of the context, never a keyword test.
