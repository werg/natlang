---
description: Write brief guidance for one call from the verified facts chosen for it.
args:
  consumer: GuidanceConsumer
  facts: GuidanceFact[]
returns: string
---
1. consumer is a call about to run, and facts are verified facts about its task. consumer reads your guidance as context before it starts; it computes its result itself.
2. Say where consumer's result comes from: which of its inputs, files, functions or child calls hold the evidence, and what decides the result.
3. Say in a few numbered steps how consumer gets there from its own inputs.
4. When a fact is feedback from an earlier attempt, say what to do instead. Describe only the correct way; do not repeat the mistaken result or action.
5. consumer computes every value of its result, so say how to find each value, not the value.
6. Keep it brief: only what helps consumer find its result. Everything you need is in consumer and facts; return the guidance text.
