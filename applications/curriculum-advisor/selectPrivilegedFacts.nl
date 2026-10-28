---
description: Choose which verified training facts the self-teacher sees in one call, most useful first.
args:
  invocation: PrivilegedInvocation
  facts: PrivilegedFact[]
  budgetTokens: number
returns: string[]
---
1. The facts belong to one training world and are all verified. The teacher sees the chosen facts in this call's context; the student never sees them and learns only from the teacher's preferences over its own tokens.
2. Choose facts that tell this call what a correct result is and how to reach it from where the student is. For a child call, its own verified answer comes first. For the root call, put the verified result and required stops, files or effects first, then the reference solution.
3. Put feedback from earlier attempts right after the facts about the correct result, so the teacher can steer away from the mistake the student actually made.
4. Leave out a fact that repeats another chosen fact or does not bear on this call.
5. Order the chosen facts from most to least useful. Facts are kept whole in that order while they fit in budgetTokens, so put short decisive facts before long ones.
6. Return only the ids of chosen facts, each at most once.
