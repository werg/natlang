---
args: { candidates: string, minSuccess: number }
returns: { teacher: string | null; reason: string }
---
You choose which teacher runs a program to produce training data for a smaller student model. candidates is JSON: a list of teachers that were each measured on the same development examples of this program. Every teacher in the list already meets the owner's minimum success rate, minSuccess. For each teacher:

- teacher: its exact identity. Answer with this string.
- roots: how many development examples it ran.
- success: how many it got right.
- failures: how many runs stopped with an error, such as an unreachable service.
- tokensPerRoot: input plus output tokens it used per example, retries included.
- msPerRoot: elapsed time per example, in milliseconds.

Work in these steps and write down the numbers you use:

1. For each teacher compute success divided by roots.
2. Find the highest success rate. Keep every teacher whose rate is within 0.05 of it. Training data from a weaker teacher teaches the student its mistakes, so a lower success rate is only acceptable when it is this close.
3. Among the teachers you kept, prefer the one with the fewest tokensPerRoot. When two are within 10% of each other on tokens, prefer the one with fewer msPerRoot.
4. A teacher with many failures compared with roots is unreliable. Skip it when another kept teacher has clearly fewer failures.

Answer with teacher set to the chosen identity, and a reason of one or two sentences that names the numbers you used. When the list is empty, answer with teacher null and say that no teacher meets the minimum.
