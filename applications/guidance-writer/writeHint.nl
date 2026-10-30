---
description: Write a short hint for one call of a task from its verified successful execution.
args:
  consumer: GuidanceConsumer
  execution: TrajectoryCall
  overview: TrajectoryOverview
  childHints: Neuralese<string>[]
  facts: GuidanceFact[]
returns: Neuralese<string>
---
1. consumer is a call about to run; execution is a verified successful execution of the same call. consumer reads your hint as context before it starts; it does not see execution.
2. overview lists execution's own steps with their tools, statuses and sizes, and its child calls. childHints holds, in the same order as execution.children, the hints already written for those child calls. facts are verified facts about this call.
3. Use eval to read the parts of execution that decided its outcome: the steps that computed or found the result, the final result, any error that a later step repaired, and what each child call was asked and returned.
4. Write a hint for consumer: the key steps in a few numbered lines, the exact names, values and calls that matter, what to ask each child call, and the mistake to avoid when execution repaired one. Use childHints for what the child calls do rather than reading their executions again.
5. Keep it short. Name what consumer needs to reach the same correct result by its own steps; do not copy execution's code or outputs.
6. Return the hint text.
