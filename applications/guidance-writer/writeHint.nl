---
description: Write a short hint for one call of a task from its verified successful execution.
args:
  consumer: GuidanceConsumer
  execution: TrajectoryCall
  overview: TrajectoryOverview
  facts: GuidanceFact[]
returns: string
---
1. consumer is a call about to run; execution is a verified successful execution of the same call. consumer reads your hint as context before it starts; it does not see execution.
2. overview lists execution's own steps with their tools, statuses and sizes, and its child calls. facts are verified facts about this call.
3. Use eval to read the parts of execution that decided its outcome: the steps that computed or found the result, the final result, any error that a later step repaired, and what each child call was asked and returned.
4. Write a hint for consumer: the steps that decided execution's outcome as a short procedure, each with the evidence it rests on (which file, field, request or child answer to read, and which call to make). Describe only what execution did.
5. The hint has at most 64 tokens (about 250 characters). consumer computes every result itself, so the hint says how to find each value, not the value.
6. Return only the hint text.
