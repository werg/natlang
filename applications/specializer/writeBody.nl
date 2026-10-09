---
kind: directory-reducer
args:
  definition: string
  groupId: string
  condition: string
returns: BodyResult
---
You write case.ts for the group of recorded calls numbered groupId of the natural-language function named definition. The condition is chosen and measured; it admits this group's calls. The case does what the executor did for the calls the condition admits.

This folder holds function.md (the function's instructions, signature, what a case may call), group.md, examples/*.json (calls of this group with their inputs, result, service calls and code), and report.md (how your previous case for this group did on the recorded calls; fix what it reports). case.ts is the file you write.

Work in these steps:
1. Write `export const when = (args) => <condition>` with the chosen condition. It is a pure test over args that answers true or false right away.
2. Write `run`: make the same service calls with the same arguments as the executor made for these calls (see the examples), and return the same result.
3. When the instructions allow it, drop a service call that the results show to be wasted, and make inconsistent results consistent. Keep the effects the instructions ask for.
4. When an input does not fit after all, `run` throws new Error(reason) and the function takes over.
5. Return { kind: "case" }.
6. When the effects need judgment at each step, write no case.ts and return { kind: "skip", reason: "effects", why }. When the executor's results for these calls disagree in ways the instructions do not explain, return reason "unstable". The why is one sentence naming what prevents a case.

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

args holds the function's arguments by name (and folder, for a directory reducer). run returns a value of the function's result type. A semantic judgment inside run is a call to a natural-language item of the context.
