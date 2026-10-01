---
args:
  evidence: unknown[]
  goal: string
returns: '{ hypotheses: { statement: string; citations: string[]; predictedChange: string }[] }'
---
Return one concise, testable explanation of the observed failure relative to goal. Cite evidence identities. If no evidence supports a change, return {hypotheses: []}. Otherwise return {hypotheses: [{statement, citations, predictedChange}]}. These are hypotheses, not proven causes. Stop once you have one useful explanation.
