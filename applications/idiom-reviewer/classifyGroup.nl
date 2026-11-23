---
description: Classify one sentence group of a .nl instruction as judgment, mechanism, or prose-interpreter.
args:
  group: SentenceGroup
returns: GroupKind
readout: decision
---
Read group.text, one sentence group from a natlang `.nl` instruction body.
Classify what it expresses:
- `judgment`: semantic criteria, policy, tone, meaning, synthesis, or open-ended evaluation that genuinely needs a language model.
- `mechanism`: deterministic plumbing, formatting, counting, sorting, exact filtering, JSON assembly, or lookup that belongs in TypeScript.
- `prose-interpreter`: a step-by-step imperative algorithm or loop written in English telling the model to execute mechanical steps one by one.
