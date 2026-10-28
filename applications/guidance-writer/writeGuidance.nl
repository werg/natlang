---
description: Write brief guidance for one call from the verified facts chosen for it.
args:
  consumer: GuidanceConsumer
  facts: GuidanceFact[]
returns: Neuralese<string>
readout: template
---
1. consumer is a call about to run, and facts are verified facts about its task. consumer reads your guidance as context before it starts; it is not shown as an answer.
2. Say what a correct result of consumer is, using the facts.
3. Say in a few numbered steps how consumer gets there from its own inputs.
4. When a fact is feedback from an earlier attempt, say what to do instead, so consumer reaches the correct result. Describe only the correct way; do not repeat the mistaken result or action.
5. Keep it brief: only what helps consumer reach a correct result.
