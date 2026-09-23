---
description: Interpret an exact experiment summary without hiding failures or
  pending semantic review.
args:
  question: string
  plan: Plan
  candidates: Candidate[]
  trials: Trial[]
returns: Report
---
function report(question, plan, candidates, trials) -> Report
  metrics = summarize(candidates, trials)
  analysis = interpret(question, plan, metrics, trials)
  return assemble(plan, metrics, trials, analysis)
