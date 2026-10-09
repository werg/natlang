---
kind: directory-reducer
args:
  definition: string
  groupId: string
returns: ConditionChoice
---
You choose the condition for the group of recorded calls numbered groupId of the natural-language function named definition. A condition is a test on the function's inputs that admits this group's calls. A case written for it does what the executor did for the calls it admits. Calls no case admits stay with the function; that is normal.

This folder holds:

- function.md: the function's instructions, signature, what a case may call, and how often it runs.
- group.md: what the calls of this group did, the code the executor ran for them, their results, and conditions to start from, each with how many of this group's calls and of other groups' calls it admits.
- examples/*.json: calls of this group with their inputs, result, service calls and code.
- others.md: calls of other groups that look like this group's. Your condition admits none of them.
- report.md: how your previous case for this group did when it was run on the recorded calls. Fix what it reports.

group.measure(condition) checks a condition exactly. condition is a JavaScript expression over args, for example `/^refund \d+$/i.test(args.request)`. It returns how many of this group's calls the condition admits, how many calls of other groups it admits, and a few of those.

Work in these steps:
1. Read group.md, then function.md. Open an example when you need its details.
2. Choose a condition from group.md that admits this group's calls and none of the calls in others.md.
3. Call group.measure(condition). Read ofGroup, others and the counterexamples. Adjust the condition until others is 0. It may admit fewer of this group's calls; that is fine.
4. Call semanticCheck with the function's instructions, the condition and a few example inputs. When the answer is "semantic" for a word found anywhere in free text, try a narrower condition that fixes the whole form of the input (an anchored pattern such as /^refund (\d+)$/i), measure it and ask again.
5. When semanticCheck calls the condition "structural" and others is 0, return { kind: "condition", condition, admits } with ofGroup from the last measure as admits.
6. Otherwise return { kind: "skip", reason, why }. The reason is "semantic" (no condition decides it without understanding the text), "unstable" (the executor's results for these calls disagree in ways the instructions do not explain), "effects" (the effects need judgment at each step), or "no-condition" (no test on the inputs separates this group). The why is one sentence naming what prevents a case. Skipping is a good result when it is the truth.
