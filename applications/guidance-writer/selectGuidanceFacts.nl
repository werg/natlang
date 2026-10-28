---
description: Choose which verified facts the guidance for one call is written from, most useful first.
args:
  consumer: GuidanceConsumer
  facts: GuidanceFact[]
  budgetTokens: number
returns: string[]
---
1. Every fact in facts is verified and may be shown to consumer. Depending on what the guidance is for, they are reference facts of the task (expected results, stops, files and effects, reference solutions, verified trajectories), feedback that earlier attempts received (kind attempt_feedback), or what a fitted adaptation learned in hindsight (kind fit_hindsight).
2. Choose the facts that tell consumer what a correct result is and how to reach it from where it starts. For a child call, its own verified answer comes first. For any other call, put the verified result and required stops, files or effects first, then the reference solution.
3. Put attempt_feedback and fit_hindsight facts right after the facts about the correct result, so the guidance can steer away from the mistake that actually happened.
4. Leave out a fact that repeats another chosen fact or does not bear on this call.
5. Order the chosen facts from most to least useful. Facts are kept whole in that order while they fit in budgetTokens, so put short decisive facts before long ones.
6. Return only the ids of chosen facts, each at most once.
